"""Görev 1: Kafe içinde kişi sayımı + cinsiyet sınıflandırma.

Pipeline:
    Video -> YOLO11 (person) -> ByteTrack (ID) -> LineCounter (kapı çizgisi)
          -> CLIP cinsiyet (track oylaması) -> overlay video + CSV/JSON + dashboard

Örnek:
    python -m src.task1_people.run --video kisi_sayimi.mp4 --line "0.30,0.55,0.70,0.55"
    python -m src.task1_people.run --video kisi_sayimi.mp4 --max-frames 300 --show
"""
from __future__ import annotations

import argparse
import os
import time
import warnings

warnings.filterwarnings("ignore")

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from src.common.export import rel_to_root, write_csv, write_dashboard_data, write_json
from src.common.line_counter import LineCounter, parse_line
from src.common.video_io import FFmpegWriter, FPSMeter, ThreadedVideoReader, fmt_ts
from src.common import visualize as viz
from src.task1_people.gender import ClipGenderClassifier, GenderVoter, crop_person

DISCLAIMER = "Cinsiyet: gorsel model tahminidir, kesin degildir ve hata payi icerir."


def parse_args():
    ap = argparse.ArgumentParser(description="Görev 1 - Kişi sayımı ve cinsiyet sınıflandırma")
    ap.add_argument("--video", default="kisi_sayimi.mp4")
    ap.add_argument("--line", default="0.25,0.60,0.75,0.60",
                    help="Sayım çizgisi 'x1,y1,x2,y2' (0-1 oran veya piksel)")
    ap.add_argument("--invert", action="store_true", help="Giriş/çıkış yönünü ters çevir")
    ap.add_argument("--anchor", choices=["bottom", "center"], default="bottom",
                    help="Çizgi testi için referans noktası (bottom = ayak)")
    ap.add_argument("--model", default="models/yolo11m.pt")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--tracker", default="configs/bytetrack_people.yaml")
    ap.add_argument("--clip-model", default="openai/clip-vit-base-patch16")
    ap.add_argument("--no-gender", action="store_true")
    ap.add_argument("--min-track-frames", type=int, default=8,
                    help="Benzersiz kişi sayılması için minimum görülme (kare)")
    ap.add_argument("--out-dir", default="outputs/task1")
    ap.add_argument("--out-width", type=int, default=1920)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--stride", type=int, default=1, help="Her N karede bir işle")
    ap.add_argument("--show", action="store_true", help="Canlı pencere göster (q: çıkış)")
    return ap.parse_args()


def main():
    a = parse_args()
    os.makedirs(os.path.join(a.out_dir, "crops"), exist_ok=True)
    device = 0 if torch.cuda.is_available() else "cpu"
    half = torch.cuda.is_available()
    print(f"[bilgi] cihaz: {'CUDA ' + torch.cuda.get_device_name(0) if half else 'CPU'}")

    model = YOLO(a.model)
    gender = None if a.no_gender else ClipGenderClassifier(a.clip_model)
    voter = GenderVoter()

    reader = ThreadedVideoReader(a.video, a.start, a.max_frames, a.stride)
    W, H, fps = reader.width, reader.height, reader.fps
    eff_fps = fps / a.stride
    out_w = min(a.out_width, W)
    s = out_w / W
    out_h = int(round(H * s / 2) * 2)
    out_w = int(round(out_w / 2) * 2)
    p1, p2 = parse_line(a.line, W, H)
    counter = LineCounter(p1, p2, margin=max(4.0, H * 0.006), invert=a.invert)
    writer = FFmpegWriter(os.path.join(a.out_dir, "annotated.mp4"), out_w, out_h, eff_fps)
    print(f"[bilgi] video {W}x{H}@{fps:.1f} -> çıktı {out_w}x{out_h}, çizgi {p1}->{p2}")

    meter = FPSMeter()
    tracks: dict[int, dict] = {}
    events: list[dict] = []
    timeline: list[dict] = []
    flash_until = -1
    ui = viz.ui_scale(np.zeros((out_h, out_w, 3), np.uint8))
    total = reader.total_to_process
    t0 = time.time()

    try:
        for n, (fidx, frame) in enumerate(reader):
            res = model.track(frame, persist=True, tracker=a.tracker, classes=[0], conf=a.conf,
                              imgsz=a.imgsz, device=device, verbose=False)[0]
            dets = []
            if res.boxes is not None and res.boxes.id is not None:
                xyxy = res.boxes.xyxy.cpu().numpy()
                ids = res.boxes.id.int().cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                dets = list(zip(ids.tolist(), xyxy, confs.tolist()))

            # --- tracking durumu + sayım
            crops, crop_meta = [], []
            for tid, box, cf in dets:
                t = tracks.setdefault(tid, {"first": fidx, "last": fidx, "n": 0, "conf_sum": 0.0,
                                            "best_crop": None, "best_area": 0})
                t["last"] = fidx
                t["n"] += 1
                t["conf_sum"] += cf
                x1, y1, x2, y2 = box
                pt = ((x1 + x2) / 2, y2) if a.anchor == "bottom" else ((x1 + x2) / 2, (y1 + y2) / 2)
                ev = counter.update(tid, pt, fidx)
                if ev is not None:
                    flash_until = n + int(eff_fps * 0.4)
                    events.append({"track_id": tid, "direction": "GIRIS" if ev.direction == "in" else "CIKIS",
                                   "frame": fidx, "time": fmt_ts(fidx / fps), "det_conf": round(cf, 3)})
                    c = crop_person(frame, box, min_h=10, min_w=5)
                    if c is not None:
                        cv2.imwrite(os.path.join(a.out_dir, "crops", f"person_{tid}.jpg"), c)
                if gender is not None and voter.need_sample(tid, fidx):
                    c = crop_person(frame, box)
                    if c is not None:
                        crops.append(c)
                        # ağırlık: tespit güveni x boyut (küçük/uzak kişiler daha az etkili)
                        crop_meta.append((tid, cf * min(1.0, (y2 - y1) / (0.35 * H))))
            if crops:
                probs = gender.predict(crops)
                for (tid, w), p in zip(crop_meta, probs):
                    voter.add(tid, p, w, fidx)

            # --- çizim
            vis = cv2.resize(frame, (out_w, out_h)) if s != 1 else frame.copy()
            viz.draw_count_line(vis, (p1[0] * s, p1[1] * s), (p2[0] * s, p2[1] * s), ui,
                                label_in="GIRIS", label_out="CIKIS",
                                in_normal_sign=(1 if a.invert else -1), flash=n <= flash_until)
            g_counts = {"Erkek": 0, "Kadin": 0, "?": 0}
            for tid, box, cf in dets:
                g, gc, _ = voter.get(tid)
                color = viz.C_MALE if g == "Erkek" else viz.C_FEMALE if g == "Kadin" else viz.C_UNKNOWN
                lab = f"ID {tid} | kisi {cf:.2f}"
                if gender is not None:
                    lab += f" | {g} {gc:.2f}" if g != "?" else " | ?"
                viz.draw_box(vis, box * s, color, lab, ui)
                x1, y1, x2, y2 = box * s
                cv2.circle(vis, (int((x1 + x2) / 2), int(y2 if a.anchor == "bottom" else (y1 + y2) / 2)),
                           int(4 * ui), color, -1, cv2.LINE_AA)
            # sayılan kişilerin cinsiyet dağılımı
            for e in events:
                g_counts[voter.get(e["track_id"])[0]] += 1
            unique = sum(1 for t in tracks.values() if t["n"] >= a.min_track_frames)
            
            # Anlik icerideki kisi sayisini hesapla (cizginin ic kisminda kalanlar)
            inside_now = 0
            for tid, box, cf in dets:
                x1, y1, x2, y2 = box
                pt = ((x1 + x2) / 2, y2 if a.anchor == "bottom" else (y1 + y2) / 2)
                cross = (p2[0] - p1[0]) * (pt[1] - p1[1]) - (p2[1] - p1[1]) * (pt[0] - p1[0])
                is_in = (cross > 0)
                if a.invert:
                    is_in = not is_in
                if is_in:
                    inside_now += 1

            cur_fps = meter.tick()
            rows = [
                ("Giris", str(counter.count_in), viz.C_IN),
                ("Cikis", str(counter.count_out), viz.C_OUT),
                ("Toplam gecis", str(counter.total), viz.C_WHITE),
                ("Icerideki (anlik)", str(inside_now), viz.C_WHITE),
                ("Benzersiz kisi (ID)", str(unique), viz.C_ACCENT),
                ("Kadrajdaki kisi", str(len(dets)), viz.C_WHITE),
            ]
            if gender is not None:
                rows += [("Gecen erkek", str(g_counts["Erkek"]), viz.C_MALE),
                         ("Gecen kadin", str(g_counts["Kadin"]), viz.C_FEMALE)]
            rows.append(("FPS", f"{cur_fps:.1f}", (0, 255, 255)))
            viz.draw_panel(vis, "KAFE KISI SAYIMI", rows, (int(15 * ui), int(15 * ui)), 330, ui)
            viz.draw_label(vis, DISCLAIMER, (int(15 * ui), out_h - int(10 * ui)), (40, 40, 40), 0.5 * ui)
            writer.write(vis)

            if n % max(1, int(eff_fps)) == 0:
                timeline.append({"t": round(fidx / fps, 2), "in": counter.count_in, "out": counter.count_out,
                                 "in_frame": len(dets)})
            counter.cleanup(fidx)
            if a.show:
                cv2.imshow("Gorev 1 - Kisi Sayimi", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if n % 50 == 0:
                print(f"\r[{n}/{total}] fps={cur_fps:5.1f} giris={counter.count_in} cikis={counter.count_out} "
                      f"benzersiz={unique}", end="", flush=True)
    finally:
        reader.release()
        writer.release()
        if a.show:
            cv2.destroyAllWindows()

    # --- sonuçlar (cinsiyet nihai oylama ile güncellenir)
    for e in events:
        g, gc, gn = voter.get(e["track_id"])
        e.update({"gender": g, "gender_conf": round(gc, 3), "gender_samples": gn,
                  "crop": f"crops/person_{e['track_id']}.jpg"})
    people = []
    for tid, t in sorted(tracks.items()):
        if t["n"] < a.min_track_frames:
            continue
        g, gc, gn = voter.get(tid)
        st = counter.states.get(tid)
        people.append({"track_id": tid, "first_time": fmt_ts(t["first"] / fps), "last_time": fmt_ts(t["last"] / fps),
                       "frames_seen": t["n"], "det_conf_mean": round(t["conf_sum"] / t["n"], 3),
                       "gender": g, "gender_conf": round(gc, 3), "gender_samples": gn,
                       "crossed": ",".join(sorted(st.counted)) if st else ""})
    gc_events = {k: sum(1 for e in events if e["gender"] == k) for k in ("Erkek", "Kadin", "?")}
    gc_people = {k: sum(1 for p in people if p["gender"] == k) for k in ("Erkek", "Kadin", "?")}
    elapsed = time.time() - t0
    summary = {
        "task": "Gorev 1 - Kafe kisi sayimi ve cinsiyet siniflandirma",
        "video": os.path.basename(a.video), "resolution": f"{W}x{H}", "video_fps": round(fps, 2),
        "frames_processed": meter.n, "processing_seconds": round(elapsed, 1),
        "avg_fps": round(meter.n / max(elapsed, 1e-6), 2),
        "count_in": counter.count_in, "count_out": counter.count_out, "total_crossings": counter.total,
        "unique_people": len(people), "gender_crossings": gc_events, "gender_all_tracks": gc_people,
        "line": [list(map(float, p1)), list(map(float, p2))],
        "models": {"detector": os.path.basename(a.model) + " (Ultralytics YOLO11, COCO person)",
                   "tracker": "ByteTrack (Ultralytics)",
                   "gender": None if gender is None else f"{a.clip_model} zero-shot + track oylamasi"},
        "disclaimer": DISCLAIMER,
        "video_out": rel_to_root(os.path.join(a.out_dir, "annotated.mp4")),
        "out_dir": rel_to_root(a.out_dir),
    }
    write_csv(events, os.path.join(a.out_dir, "crossings.csv"))
    write_csv(people, os.path.join(a.out_dir, "people.csv"))
    write_json({"summary": summary, "crossings": events, "people": people, "timeline": timeline},
               os.path.join(a.out_dir, "results.json"))
    write_dashboard_data("task1", {"summary": summary, "crossings": events, "people": people, "timeline": timeline})
    print(f"\n[tamam] giris={counter.count_in} cikis={counter.count_out} benzersiz={len(people)} "
          f"ort.fps={summary['avg_fps']} -> {a.out_dir}")


if __name__ == "__main__":
    main()
