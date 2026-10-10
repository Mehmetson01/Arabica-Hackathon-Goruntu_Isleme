"""Görselleştirme yardımcıları (bbox, etiket, sayaç paneli, sayım çizgisi).

OpenCV Hershey fontları Türkçe karakter desteklemediği için video üstü metinler
ASCII'ye çevrilir (ç->c, ş->s ...).
"""
from __future__ import annotations

import colorsys

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_DUPLEX
_TR = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")

# Tema renkleri (BGR)
C_ACCENT = (255, 191, 0)      # turkuaz-mavi
C_IN = (120, 220, 60)         # yeşil
C_OUT = (60, 90, 255)         # kırmızı-turuncu
C_MALE = (255, 160, 40)       # mavi
C_FEMALE = (180, 80, 255)     # pembe
C_UNKNOWN = (180, 180, 180)
C_WHITE = (255, 255, 255)
C_PANEL = (25, 20, 18)


def ascii_tr(text: str) -> str:
    return str(text).translate(_TR)


def id_color(track_id: int):
    h = (track_id * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.65, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def ui_scale(frame: np.ndarray) -> float:
    return max(0.5, frame.shape[1] / 1920.0)


def draw_label(frame, text, org, color, scale=0.6, thickness=1, bg_alpha=0.75, pad=4):
    text = ascii_tr(text)
    (tw, th), base = cv2.getTextSize(text, FONT, scale, thickness)
    x, y = int(org[0]), int(org[1])
    x = max(0, min(x, frame.shape[1] - tw - 2 * pad))
    y = max(th + 2 * pad, y)
    x1, y1, x2, y2 = x, y - th - 2 * pad, x + tw + 2 * pad, y + base
    roi = frame[max(0, y1):y2, max(0, x1):x2]
    if roi.size:
        bg = np.empty_like(roi)
        bg[:] = color
        cv2.addWeighted(bg, bg_alpha, roi, 1 - bg_alpha, 0, roi)
    lum = 0.114 * color[0] + 0.587 * color[1] + 0.299 * color[2]
    fg = (20, 20, 20) if lum > 150 else C_WHITE
    cv2.putText(frame, text, (x + pad, y - pad), FONT, scale, fg, thickness, cv2.LINE_AA)
    return x2, y2


def draw_box(frame, box, color, label: str | None = None, scale: float = 1.0, thickness: int = 2):
    x1, y1, x2, y2 = [int(v) for v in box]
    t = max(1, int(thickness * scale))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, t, cv2.LINE_AA)
    # köşe vurguları
    L = max(8, int(min(x2 - x1, y2 - y1) * 0.18))
    for (cx, cy, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(frame, (cx, cy), (cx + dx * L, cy), color, t + 2, cv2.LINE_AA)
        cv2.line(frame, (cx, cy), (cx, cy + dy * L), color, t + 2, cv2.LINE_AA)
    if label:
        draw_label(frame, label, (x1, y1 - 2), color, scale=0.55 * scale, thickness=max(1, int(scale)))


def draw_count_line(frame, p1, p2, scale=1.0, label_in="GIRIS", label_out="CIKIS",
                    in_normal_sign: int = -1, flash: bool = False):
    """Sayım çizgisini ve yön oklarını çizer.

    in_normal_sign: "in" yönünün çizgi normaline göre işareti (LineCounter'da
    +1 -> -1 geçişi "in" olduğu için varsayılan -1).
    """
    p1 = np.array(p1, dtype=np.float32)
    p2 = np.array(p2, dtype=np.float32)
    color = (0, 255, 255) if flash else C_ACCENT
    t = max(2, int(3 * scale))
    overlay = frame.copy()
    cv2.line(overlay, tuple(p1.astype(int)), tuple(p2.astype(int)), color, t * 4, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
    cv2.line(frame, tuple(p1.astype(int)), tuple(p2.astype(int)), color, t, cv2.LINE_AA)
    for p in (p1, p2):
        cv2.circle(frame, tuple(p.astype(int)), t * 2, color, -1, cv2.LINE_AA)
    d = p2 - p1
    n = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-6)
    mid = (p1 + p2) / 2
    alen = 45 * scale
    a_in = mid + n * in_normal_sign * alen
    a_out = mid - n * in_normal_sign * alen
    off = d / (np.linalg.norm(d) + 1e-6) * 30 * scale
    cv2.arrowedLine(frame, tuple((mid - off).astype(int)), tuple((a_in - off).astype(int)), C_IN, t, cv2.LINE_AA, tipLength=0.35)
    cv2.arrowedLine(frame, tuple((mid + off).astype(int)), tuple((a_out + off).astype(int)), C_OUT, t, cv2.LINE_AA, tipLength=0.35)
    draw_label(frame, label_in, tuple((a_in - off * 2).astype(int)), C_IN, scale=0.5 * scale)
    draw_label(frame, label_out, tuple((a_out + off * 2).astype(int)), C_OUT, scale=0.5 * scale)


def draw_panel(frame, title: str, rows: list, origin=(20, 20), width: int = 360,
               scale: float = 1.0, alpha: float = 0.72):
    """Yarı saydam bilgi paneli.

    rows: [(metin, renk), ...] veya [(sol, sağ, renk), ...]
    """
    x, y = int(origin[0]), int(origin[1])
    w = int(width * scale)
    line_h = int(30 * scale)
    title_h = int(40 * scale)
    h = title_h + line_h * len(rows) + int(12 * scale)
    x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
    roi = frame[y:y2, x:x2]
    if roi.size == 0:
        return y
    bg = np.empty_like(roi)
    bg[:] = C_PANEL
    cv2.addWeighted(bg, alpha, roi, 1 - alpha, 0, roi)
    cv2.rectangle(frame, (x, y), (x2, y + int(4 * scale)), C_ACCENT, -1)
    cv2.putText(frame, ascii_tr(title), (x + int(12 * scale), y + int(30 * scale)), FONT,
                0.7 * scale, C_ACCENT, max(1, int(1.5 * scale)), cv2.LINE_AA)
    cy = y + title_h + int(20 * scale)
    for row in rows:
        if len(row) == 3:
            left, right, color = row
            cv2.putText(frame, ascii_tr(left), (x + int(14 * scale), cy), FONT, 0.58 * scale,
                        (220, 220, 220), max(1, int(scale)), cv2.LINE_AA)
            rt = ascii_tr(right)
            (tw, _), _ = cv2.getTextSize(rt, FONT, 0.66 * scale, max(1, int(1.5 * scale)))
            cv2.putText(frame, rt, (x2 - tw - int(14 * scale), cy), FONT, 0.66 * scale, color,
                        max(1, int(1.5 * scale)), cv2.LINE_AA)
        else:
            text, color = row
            cv2.putText(frame, ascii_tr(text), (x + int(14 * scale), cy), FONT, 0.55 * scale, color,
                        max(1, int(scale)), cv2.LINE_AA)
        cy += line_h
    return y2


def paste_thumb(frame, img, x, y, h):
    """Küçük görseli (ör. plaka crop) kareye yapıştırır. Bitiş x'ini döndürür."""
    if img is None or img.size == 0:
        return x
    r = h / img.shape[0]
    w = max(1, int(img.shape[1] * r))
    th = cv2.resize(img, (w, int(h)))
    x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + int(h))
    if x2 <= x or y2 <= y:
        return x
    frame[y:y2, x:x2] = th[: y2 - y, : x2 - x]
    cv2.rectangle(frame, (x, y), (x2, y2), C_ACCENT, 1)
    return x2
