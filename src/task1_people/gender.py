"""Cinsiyet tahmini: CLIP zero-shot + track bazlı oylama.

Kafe kameralarında yüzler çoğunlukla küçük/arkası dönük olduğu için yüz tabanlı
modeller yerine TÜM VÜCUT crop'u üzerinde CLIP (OpenAI, Hugging Face) zero-shot
sınıflandırma kullanılır. Her track için birden fazla karede tahmin alınır ve
ağırlıklı ortalaması hesaplanır -> etiket titremez, güven skoru stabilleşir.

UYARI: Bu çıktı yalnızca görsel bir model tahminidir; kesin kimlik veya kişisel
özellik tespiti değildir ve hata payı içerir.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch

LABELS = ("Erkek", "Kadin")

PROMPTS = {
    "Erkek": [
        "a photo of a man",
        "a photo of a male person",
        "a cctv photo of a man",
        "a photo of a man walking",
        "a photo of a boy",
    ],
    "Kadin": [
        "a photo of a woman",
        "a photo of a female person",
        "a cctv photo of a woman",
        "a photo of a woman walking",
        "a photo of a girl",
    ],
}


class ClipGenderClassifier:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch16", device: str | None = None,
                 half: bool = True):
        from transformers import CLIPModel, CLIPProcessor

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if (half and self.device.startswith("cuda")) else torch.float32
        self.model = CLIPModel.from_pretrained(model_name, torch_dtype=self.dtype).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model_name = model_name
        with torch.no_grad():
            feats = []
            for lab in LABELS:
                tok = self.processor(text=PROMPTS[lab], return_tensors="pt", padding=True).to(self.device)
                t_out = self.model.text_model(**tok)
                t = self.model.text_projection(t_out[1])
                t = t / t.norm(dim=-1, keepdim=True)
                t = t.mean(0)
                feats.append(t / t.norm())
            self.text_feats = torch.stack(feats)  # (2, D)
        self.logit_scale = float(self.model.logit_scale.exp().item())

    @torch.no_grad()
    def predict(self, crops_bgr: list[np.ndarray]) -> np.ndarray:
        """crops -> (N, 2) olasılık [Erkek, Kadin]."""
        if not crops_bgr:
            return np.zeros((0, 2), dtype=np.float32)
        imgs = [cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in crops_bgr]
        px = self.processor(images=imgs, return_tensors="pt")["pixel_values"].to(self.device, self.dtype)
        f_out = self.model.vision_model(pixel_values=px)
        f = self.model.visual_projection(f_out[1])
        f = f / f.norm(dim=-1, keepdim=True)
        logits = self.logit_scale * f @ self.text_feats.T
        return logits.float().softmax(-1).cpu().numpy()


class GenderVoter:
    """Track başına ağırlıklı olasılık ortalaması."""

    def __init__(self, max_samples: int = 25, min_samples: int = 3, interval: int = 3,
                 threshold: float = 0.58):
        self.max_samples = max_samples
        self.min_samples = min_samples
        self.interval = interval
        self.threshold = threshold
        self.data: dict[int, dict] = {}

    def need_sample(self, tid: int, frame_idx: int) -> bool:
        d = self.data.get(tid)
        if d is None:
            return True
        return d["n"] < self.max_samples and frame_idx - d["last"] >= self.interval

    def add(self, tid: int, probs: np.ndarray, weight: float, frame_idx: int):
        d = self.data.setdefault(tid, {"sum": np.zeros(2, np.float64), "w": 0.0, "n": 0, "last": -10**9})
        d["sum"] += probs * weight
        d["w"] += weight
        d["n"] += 1
        d["last"] = frame_idx

    def get(self, tid: int):
        """-> (etiket, güven, örnek_sayısı). Yetersiz veri/belirsizlikte etiket '?'."""
        d = self.data.get(tid)
        if d is None or d["w"] <= 0:
            return "?", 0.0, 0
        p = d["sum"] / d["w"]
        k = int(np.argmax(p))
        conf = float(p[k])
        if d["n"] < self.min_samples or conf < self.threshold:
            return "?", conf, d["n"]
        return LABELS[k], conf, d["n"]


def crop_person(frame: np.ndarray, box, pad: float = 0.06, min_h: int = 48, min_w: int = 20):
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    if bh < min_h or bw < min_w:
        return None
    x1 = int(max(0, x1 - bw * pad)); x2 = int(min(W, x2 + bw * pad))
    y1 = int(max(0, y1 - bh * pad)); y2 = int(min(H, y2 + bh * pad))
    c = frame[y1:y2, x1:x2]
    return c if c.size else None
