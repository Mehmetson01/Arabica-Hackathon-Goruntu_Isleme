"""Görev 2: Şehre giren araç sayımı + plaka tespiti + OCR.

Pipeline:
    Video (4K) -> YOLO11 (car/motorcycle/bus/truck) -> ByteTrack -> LineCounter
               -> araç crop (tam çözünürlük) -> plaka YOLO -> plaka crop -> EasyOCR
               -> TR format düzeltme + track oylaması -> overlay video + CSV/JSON + crop'lar

Örnek:
    python -m src.task2_vehicles.run --video "4K Road traffic video....mp4" --line "0.0,0.6,1.0,0.6"
"""
from __future__ import annotations

import argparse
import glob
import os
import time
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from src.common import visualize as viz
from src.common.export import rel_to_root, write_csv, write_dashboard_data, write_json
from src.common.line_counter import LineCounter, parse_line
from src.common.video_io import FFmpegWriter, FPSMeter, ThreadedVideoReader, fmt_ts
from src.task2_vehicles.plate import PlateDetector, PlateOCR, PlateVoter
from src.task2_vehicles.plate_format import format_tr

VEHICLE_CLASSES = {2: "otomobil", 3: "motosiklet", 5: "otobus", 7: "kamyon"}
TYPE_COLORS = {"otomobil": (255, 191, 0), "motosiklet": (0, 215, 255), "otobus": (120, 220, 60),
               "kamyon": (60, 90, 255)}


def default_video():
    c = glob.glob("4K*.mp4")
    return c[0] if c else "traffic.mp4"


def parse_args():
    ap = argparse.ArgumentParser(description="Görev 2 - Araç sayımı ve plaka tanıma")
    ap.add_argument("--video", default=default_video())
    ap.add_argument("--line", default="0.0,0.62,1.0,0.62", help="Şehir giriş çizgisi 'x1,y1,x2,y2'")
    ap.add_argument("--invert", action="store_true", help="Giriş/çıkış yönünü ters çevir")
    ap.add_argument("--model", default="models/yolo11m.pt")
    ap.add_argument("--plate-model", default="models/plate_yolo11s.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--tracker", default="configs/bytetrack_vehicles.yaml")
    ap.add_argument("--ocr-min-h", type=float, default=0.06,
                    help="OCR denemesi için minimum araç kutusu yüksekliği (kare yüksekliğine oran)")
    ap.add_argument("--ocr-zone", default="0.0,1.0", help="OCR yapılacak dikey bant 'y1,y2' (oran)")
    ap.add_argument("--ocr-per-frame", type=int, default=6, help="Karede en fazla kaç araç için OCR")
    ap.add_argument("--no-ocr", action="store_true")
    ap.add_argument("--out-dir", default="outputs/task2")
    ap.add_argument("--out-width", type=int, default=1920)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--show", action="store_true")
    return ap.parse_args()


def crop(frame, box, pad=0.04):
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    x1 = int(max(0, x1 - bw * pad)); y1 = int(max(0, y1 - bh * pad))
    x2 = int(min(W, x2 + bw * pad)); y2 = int(min(H, y2 + bh * pad))
    return frame[y1:y2, x1:x2], (x1, y1)


def draw_pass_list(vis, records, voter, ui, max_rows=8):
    """Sağ üstte son geçişler listesi (zaman | ID | tür | plaka + plaka crop)."""
    H, W = vis.shape[:2]
    w = int(560 * ui)
    x = W - w - int(15 * ui)
    y = int(15 * ui)
    row_h = int(38 * ui)
    rows = records[-max_rows:][::-1]
    h = int(44 * ui) + row_h * max(1, len(rows)) + int(8 * ui)
    roi = vis[y:y + h, x:x + w]
    bg = np.empty_like(roi)
    bg[:] = viz.C_PANEL
    cv2.addWeighted(bg, 0.72, roi, 0.28, 0, roi)
    cv2.rectangle(vis, (x, y), (x + w, y + int(4 * ui)), viz.C_ACCENT, -1)
    cv2.putText(vis, "SON GECISLER  (zaman | ID | tur | plaka)", (x + int(12 * ui), y + int(30 * ui)),
                viz.FONT, 0.6 * ui, viz.C_ACCENT, max(1, int(1.4 * ui)), cv2.LINE_AA)
    cy = y + int(44 * ui)
    if not rows:
        cv2.putText(vis, "henuz gecis yok", (x + int(14 * ui), cy + int(26 * ui)), viz.FONT, 0.55 * ui,
                    (180, 180, 180), 1, cv2.LINE_AA)
    for r in rows:
        txt, oc, _, tr = voter.result(r["vehicle_id"]) if voter else ("", 0, 0, False)
        plate = (format_tr(txt) if tr else txt) or "----"
        color = viz.C_IN if r["direction"] == "GIRIS" else viz.C_OUT
        cv2.circle(vis, (x + int(16 * ui), cy + int(19 * ui)), int(5 * ui), color, -1, cv2.LINE_AA)
        line = f"{r['time'][3:11]}  ID{r['vehicle_id']:<4} {r['type'][:9]:<9}"
        cv2.putText(vis, line, (x + int(28 * ui), cy + int(25 * ui)), viz.FONT, 0.52 * ui, (230, 230, 230),
                    max(1, int(ui)), cv2.LINE_AA)
        pc = (0, 255, 120) if tr else (0, 215, 255) if txt else (150, 150, 150)
        cv2.putText(vis, f"{plate} {oc:.2f}" if txt else plate, (x + int(300 * ui), cy + int(25 * ui)), viz.FONT,
                    0.55 * ui, pc, max(1, int(1.3 * ui)), cv2.LINE_AA)
        thumb = voter.best_crop(r["vehicle_id"]) if voter else None
        if thumb is not None:
            viz.paste_thumb(vis, thumb, x + w - int(95 * ui), cy + int(5 * ui), int(28 * ui))
        cy += row_h


def main():
    a = parse_args()
    for d in ("vehicles", "plates"):
        os.makedirs(os.path.join(a.out_dir, d), exist_ok=True)
    half = torch.cuda.is_available()
    device = 0 if half else "cpu"
    print(f"[bilgi] cihaz: {'CUDA ' + torch.cuda.get_device_name(0) if half else 'CPU'}")

    model = YOLO(a.model)
    pdet = None if a.no_ocr else PlateDetector(a.plate_model)
    ocr = None if a.no_ocr else PlateOCR()
    voter = PlateVoter()

    reader = ThreadedVideoReader(a.video, a.start, a.max_frames, a.stride)
    W, H, fps = reader.width, reader.height, reader.fps
    eff_fps = fps / a.stride
    out_w = int(round(min(a.out_width, W) / 2) * 2)
    s = out_w / W
    out_h = int(round(H * s / 2) * 2)
    p1, p2 = parse_line(a.line, W, H)
    counter = LineCounter(p1, p2, margin=max(4.0, H * 0.004), invert=a.invert)
    zy1, zy2 = [float(v) * H for v in a.ocr_zone.split(",")]
    writer = FFmpegWriter(os.path.join(a.out_dir, "annotated.mp4"), out_w, out_h, eff_fps)
    print(f"[bilgi] video {W}x{H}@{fps:.1f} -> çıktı {out_w}x{out_h}, çizgi {p1}->{p2}")

    ui = viz.ui_scale(np.zeros((out_h, out_w, 3), np.uint8))
    meter = FPSMeter()
    cls_votes: dict[int, dict] = defaultdict(lambda: defaultdict(float))
    vconf: dict[int, list] = defaultdict(list)
    last_plate_box: dict[int, tuple] = {}
    records: list[dict] = []
    timeline: list[dict] = []
    flash_until = -1
    total = reader.total_to_process
    t0 = time.time()
    ocr_time = 0.0

    def vtype(tid):
        v = cls_votes.get(tid)
        return VEHICLE_CLASSES[max(v, key=v.get)] if v else "arac"

    try:
        for n, (fidx, frame) in enumerate(reader):
            res = model.track(frame, persist=True, tracker=a.tracker, classes=list(VEHICLE_CLASSES),
                              conf=a.conf, imgsz=a.imgsz, device=device, verbose=False)[0]
            dets = []
            if res.boxes is not None and res.boxes.id is not None:
                xyxy = res.boxes.xyxy.cpu().numpy()
                ids = res.boxes.id.int().cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                clss = res.boxes.cls.int().cpu().numpy()
                dets = list(zip(ids.tolist(), xyxy, confs.tolist(), clss.tolist()))

            ocr_cands = []
            for tid, box, cf, c in dets:
                cls_votes[tid][c] += cf
                vconf[tid].append(cf)
                x1, y1, x2, y2 = box
                ev = counter.update(tid, ((x1 + x2) / 2, (y1 + y2) / 2), fidx)
                if ev is not None:
                    flash_until = n + int(eff_fps * 0.3)
                    vc, _ = crop(frame, box, 0.02)
                    vpath = os.path.join(a.out_dir, "vehicles", f"vehicle_{tid}.jpg")
                    cv2.imwrite(vpath, vc)
                    records.append({"vehicle_id": tid, "direction": "GIRIS" if ev.direction == "in" else "CIKIS",
                                    "frame": fidx, "time": fmt_ts(fidx / fps),
                                    "wall_time": time.strftime("%Y-%m-%d %H:%M:%S"), "type": vtype(tid),
                                    "vehicle_crop": f"vehicles/vehicle_{tid}.jpg"})
                bh = y2 - y1
                cy = (y1 + y2) / 2
                edge = x1 < 2 or y1 < 2 or x2 > W - 2 or y2 > H - 2
                if (pdet is not None and bh >= a.ocr_min_h * H and zy1 <= cy <= zy2 and not edge
                        and voter.need(tid, fidx)):
                    ocr_cands.append((bh, tid, box))

            # --- plaka tespiti + OCR (en büyük/en yakın araçlar öncelikli)
            if ocr_cands:
                ts = time.perf_counter()
                ocr_cands.sort(key=lambda z: -z[0])
                ocr_cands = ocr_cands[:a.ocr_per_frame]
                vcrops, offs = [], []
                for _, tid, box in ocr_cands:
                    vc, off = crop(frame, box)
                    vcrops.append(vc)
                    offs.append(off)
                    voter.mark_attempt(tid, fidx)
                for (_, tid, _), vc, off, pb in zip(ocr_cands, vcrops, offs, pdet.detect(vcrops)):
                    if pb is None:
                        continue
                    px1, py1, px2, py2, pconf = pb
                    pw, ph = px2 - px1, py2 - py1
                    pc = vc[int(max(0, py1 - ph * 0.12)):int(min(vc.shape[0], py2 + ph * 0.12)),
                            int(max(0, px1 - pw * 0.05)):int(min(vc.shape[1], px2 + pw * 0.05))]
                    if pc.size == 0 or pc.shape[0] < 8:
                        continue
                    last_plate_box[tid] = (px1 + off[0], py1 + off[1], px2 + off[0], py2 + off[1], pconf, n)
                    raw, oc = ocr.read(pc)
                    if raw:
                        voter.add(tid, raw, oc, pconf, pc)
                ocr_time += time.perf_counter() - ts

            # --- çizim
            vis = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA) if s != 1 else frame.copy()
            viz.draw_count_line(vis, (p1[0] * s, p1[1] * s), (p2[0] * s, p2[1] * s), ui,
                                label_in="SEHRE GIRIS", label_out="CIKIS",
                                in_normal_sign=(1 if a.invert else -1), flash=n <= flash_until)
            for tid, box, cf, c in dets:
                t = vtype(tid)
                color = TYPE_COLORS.get(t, viz.C_ACCENT)
                viz.draw_box(vis, box * s, color, f"ID {tid} | {t} {cf:.2f}", ui)
                pb = last_plate_box.get(tid)
                if pb and n - pb[5] <= eff_fps * 0.5:
                    bx = np.array(pb[:4]) * s
                    cv2.rectangle(vis, (int(bx[0]), int(bx[1])), (int(bx[2]), int(bx[3])), (0, 255, 120),
                                  max(1, int(2 * ui)), cv2.LINE_AA)
                txt, oc, votes, tr = voter.result(tid)
                if txt:
                    lab = f"{format_tr(txt) if tr else txt}  ocr {oc:.2f}" + ("  TR" if tr else "")
                    viz.draw_label(vis, lab, (box[0] * s, box[3] * s + 24 * ui),
                                   (0, 200, 90) if tr else (0, 170, 230), 0.6 * ui, max(1, int(1.3 * ui)))
            n_in = sum(1 for r in records if r["direction"] == "GIRIS")
            types = defaultdict(int)
            for r in records:
                if r["direction"] == "GIRIS":
                    types[vtype(r["vehicle_id"])] += 1
            plates_read = sum(1 for r in records if voter.result(r["vehicle_id"])[0])
            cur_fps = meter.tick()
            rows = [("Sehre giren", str(n_in), viz.C_IN),
                    ("Cikan (ters yon)", str(counter.count_out), viz.C_OUT),
                    ("Toplam gecis", str(counter.total), viz.C_WHITE)]
            rows += [(f"  {k}", str(types.get(k, 0)), TYPE_COLORS[k]) for k in VEHICLE_CLASSES.values()]
            rows += [("Plaka okunan", f"{plates_read}/{len(records)}", (0, 255, 120)),
                     ("Kadrajdaki arac", str(len(dets)), viz.C_WHITE),
                     ("FPS", f"{cur_fps:.1f}", (0, 255, 255))]
            viz.draw_panel(vis, "SEHIR GIRISI ARAC SAYIMI", rows, (int(15 * ui), int(15 * ui)), 340, ui)
            draw_pass_list(vis, records, voter, ui)
            writer.write(vis)

            if n % max(1, int(eff_fps)) == 0:
                timeline.append({"t": round(fidx / fps, 2), "in": n_in, "out": counter.count_out,
                                 "in_frame": len(dets)})
            counter.cleanup(fidx)
            if a.show:
                cv2.imshow("Gorev 2 - Arac Sayimi", cv2.resize(vis, (1280, int(1280 * out_h / out_w))))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if n % 25 == 0:
                print(f"\r[{n}/{total}] fps={cur_fps:5.1f} giris={n_in} cikis={counter.count_out} "
                      f"plaka={plates_read}", end="", flush=True)
    finally:
        reader.release()
        writer.release()
        if a.show:
            cv2.destroyAllWindows()

    # --- sonuçların kesinleştirilmesi
    seen_plates: dict[str, list] = defaultdict(list)
    for r in records:
        tid = r["vehicle_id"]
        txt, oc, votes, tr = voter.result(tid)
        r["type"] = vtype(tid)
        r["vehicle_conf"] = round(float(np.mean(vconf[tid])), 3) if vconf[tid] else 0.0
        r["plate"] = txt
        r["plate_display"] = format_tr(txt) if tr else txt
        r["plate_ocr_conf"] = round(oc, 3)
        r["plate_det_conf"] = round(voter.plate_det_conf(tid), 3)
        r["plate_votes"] = votes
        r["tr_format_valid"] = bool(tr)
        r["plate_crop"] = ""
        bc = voter.best_crop(tid)
        if bc is not None:
            cv2.imwrite(os.path.join(a.out_dir, "plates", f"plate_{tid}.jpg"), bc)
            r["plate_crop"] = f"plates/plate_{tid}.jpg"
        r["repeat_pass"] = bool(txt) and any(o != tid for o in seen_plates[txt])
        if txt:
            seen_plates[txt].append(tid)

    elapsed = time.time() - t0
    types_all = defaultdict(int)
    for r in records:
        types_all[r["type"]] += 1
    summary = {
        "task": "Gorev 2 - Sehre giren arac sayimi ve plaka tespiti",
        "video": os.path.basename(a.video), "resolution": f"{W}x{H}", "video_fps": round(fps, 2),
        "frames_processed": meter.n, "processing_seconds": round(elapsed, 1),
        "avg_fps": round(meter.n / max(elapsed, 1e-6), 2), "ocr_seconds": round(ocr_time, 1),
        "count_in": sum(1 for r in records if r["direction"] == "GIRIS"),
        "count_out": sum(1 for r in records if r["direction"] == "CIKIS"),
        "total_crossings": len(records), "types": dict(types_all),
        "plates_read": sum(1 for r in records if r["plate"]),
        "plates_tr_valid": sum(1 for r in records if r["tr_format_valid"]),
        "repeat_passes": sum(1 for r in records if r["repeat_pass"]),
        "line": [list(map(float, p1)), list(map(float, p2))],
        "models": {"detector": os.path.basename(a.model) + " (Ultralytics YOLO11, COCO car/motorcycle/bus/truck)",
                   "tracker": "ByteTrack (Ultralytics)",
                   "plate_detector": "morsetechlab/yolov11-license-plate-detection (YOLO11s, Hugging Face)",
                   "ocr": "EasyOCR (en, GPU) + TR format duzeltme + track oylamasi"},
        "video_out": rel_to_root(os.path.join(a.out_dir, "annotated.mp4")),
        "out_dir": rel_to_root(a.out_dir),
    }
    write_csv(records, os.path.join(a.out_dir, "vehicles.csv"))
    write_json({"summary": summary, "records": records, "timeline": timeline},
               os.path.join(a.out_dir, "results.json"))
    with open(os.path.join(a.out_dir, "plates.txt"), "w", encoding="utf-8") as f:
        for r in records:
            f.write(f"{r['time']}\tID {r['vehicle_id']}\t{r['direction']}\t{r['type']}\t"
                    f"{r['plate_display'] or '-'}\t{r['plate_ocr_conf']}\n")
    write_dashboard_data("task2", {"summary": summary, "records": records, "timeline": timeline})
    print(f"\n[tamam] giris={summary['count_in']} cikis={summary['count_out']} plaka={summary['plates_read']} "
          f"ort.fps={summary['avg_fps']} -> {a.out_dir}")


if __name__ == "__main__":
    main()
