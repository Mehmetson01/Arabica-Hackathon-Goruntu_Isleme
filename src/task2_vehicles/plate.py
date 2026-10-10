"""Plaka bölgesi tespiti, OCR ve track bazlı plaka oylaması.

- PlateDetector: Hugging Face 'morsetechlab/yolov11-license-plate-detection'
  (YOLO11s, Roboflow plaka veri setiyle fine-tune) araç crop'ları üzerinde çalışır.
- PlateOCR: EasyOCR (GPU) + ön işleme (büyütme, CLAHE, keskinleştirme).
- PlateVoter: Bir aracın birden çok karedeki okumalarını birleştirir; TR formatına
  uyan okumalar ekstra ağırlık alır.
"""
from __future__ import annotations

from collections import defaultdict

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from src.task2_vehicles.plate_format import clean, correct_tr

ALNUM = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


class PlateDetector:
    def __init__(self, weights: str = "models/plate_yolo11s.pt", conf: float = 0.30, imgsz: int = 640):
        self.model = YOLO(weights)
        self.conf = conf
        self.imgsz = imgsz
        self.half = torch.cuda.is_available()
        self.device = 0 if self.half else "cpu"

    def detect(self, vehicle_crops: list[np.ndarray]):
        """Her araç crop'u için en güvenli plaka kutusu -> [(x1,y1,x2,y2,conf) | None]."""
        if not vehicle_crops:
            return []
        results = self.model.predict(vehicle_crops, imgsz=self.imgsz, conf=self.conf,
                                     device=self.device, verbose=False)
        out = []
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                out.append(None)
                continue
            i = int(r.boxes.conf.argmax())
            x1, y1, x2, y2 = r.boxes.xyxy[i].cpu().numpy().tolist()
            out.append((x1, y1, x2, y2, float(r.boxes.conf[i])))
        return out


def preprocess_plate(img: np.ndarray, target_h: int = 96) -> np.ndarray:
    h, w = img.shape[:2]
    r = target_h / max(h, 1)
    img = cv2.resize(img, (max(1, int(w * r)), target_h), interpolation=cv2.INTER_CUBIC)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 8)).apply(g)
    blur = cv2.GaussianBlur(g, (0, 0), 1.2)
    g = cv2.addWeighted(g, 1.6, blur, -0.6, 0)  # unsharp mask
    return g


class PlateOCR:
    def __init__(self, gpu: bool | None = None):
        import easyocr
        self.reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available() if gpu is None else gpu,
                                     verbose=False)

    def read(self, plate_img: np.ndarray):
        """-> (ham_metin, güven). Birden çok metin kutusu soldan sağa / yukarıdan aşağı birleştirilir."""
        if plate_img is None or plate_img.size == 0:
            return "", 0.0
        g = preprocess_plate(plate_img)
        res = self.reader.readtext(g, allowlist=ALNUM, detail=1, paragraph=False,
                                   decoder="beamsearch", beamWidth=5, text_threshold=0.5,
                                   low_text=0.3, width_ths=0.9, mag_ratio=1.0)
        if not res:
            return "", 0.0
        h = g.shape[0]
        # çok küçük parçaları (vida, logo, ülke kodu) ele
        res = [r for r in res if (max(p[1] for p in r[0]) - min(p[1] for p in r[0])) > 0.35 * h] or res
        res.sort(key=lambda r: (round(min(p[1] for p in r[0]) / (0.5 * h)), min(p[0] for p in r[0])))
        text = "".join(clean(r[1]) for r in res)
        if not text:
            return "", 0.0
        conf = sum(r[2] * len(clean(r[1])) for r in res) / max(1, sum(len(clean(r[1])) for r in res))
        return text, float(conf)


class PlateVoter:
    def __init__(self, max_attempts: int = 14, interval: int = 2, stop_conf: float = 0.85, stop_votes: int = 3,
                 tr_bonus: float = 1.5):
        self.max_attempts = max_attempts
        self.interval = interval
        self.stop_conf = stop_conf
        self.stop_votes = stop_votes
        self.tr_bonus = tr_bonus
        self.tracks: dict[int, dict] = {}

    def _t(self, tid):
        return self.tracks.setdefault(tid, {"attempts": 0, "last": -10**9, "reads": [], "best_crop": None,
                                            "best_score": -1.0, "det_conf": 0.0})

    def need(self, tid: int, frame_idx: int) -> bool:
        t = self._t(tid)
        if t["attempts"] >= self.max_attempts or frame_idx - t["last"] < self.interval:
            return False
        txt, conf, votes, _ = self.result(tid)
        return not (txt and conf >= self.stop_conf and votes >= self.stop_votes)

    def mark_attempt(self, tid: int, frame_idx: int):
        t = self._t(tid)
        t["attempts"] += 1
        t["last"] = frame_idx

    def add(self, tid: int, raw: str, ocr_conf: float, det_conf: float, plate_crop: np.ndarray):
        t = self._t(tid)
        text, tr_ok, _ = correct_tr(raw)
        if len(text) < 4 or len(text) > 10:
            return
        t["reads"].append((text, ocr_conf, det_conf, tr_ok))
        score = ocr_conf * det_conf * (self.tr_bonus if tr_ok else 1.0)
        if score > t["best_score"]:
            t["best_score"] = score
            t["best_crop"] = plate_crop.copy()
            t["det_conf"] = det_conf

    def result(self, tid: int):
        """-> (plaka, ocr_güveni, oy_sayısı, tr_geçerli)"""
        t = self.tracks.get(tid)
        if not t or not t["reads"]:
            return "", 0.0, 0, False
        agg = defaultdict(lambda: [0.0, 0, 0.0, False])
        for text, oc, dc, tr in t["reads"]:
            a = agg[text]
            a[0] += oc * (self.tr_bonus if tr else 1.0)
            a[1] += 1
            a[2] = max(a[2], oc)
            a[3] = tr
        text, (score, votes, maxc, tr) = max(agg.items(), key=lambda kv: kv[1][0])
        mean_c = np.mean([oc for tx, oc, _, _ in t["reads"] if tx == text])
        return text, float(mean_c), votes, tr

    def plate_det_conf(self, tid: int) -> float:
        t = self.tracks.get(tid)
        return float(t["det_conf"]) if t else 0.0

    def best_crop(self, tid: int):
        t = self.tracks.get(tid)
        return None if not t else t["best_crop"]
