"""Videodan önizleme kareleri çıkarır (çizgi konumunu belirlemek için).

Karelerin üzerine normalize (0-1) koordinat ızgarası çizilir; böylece
--line parametresi kolayca seçilebilir.

Kullanım:
    python tools/preview.py --video kisi_sayimi.mp4 --out outputs/preview/cafe --n 6
"""
import argparse
import os

import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--width", type=int, default=1280)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cap = cv2.VideoCapture(a.video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"{a.video}: {w}x{h} @ {fps:.2f} fps, {n} kare, {n / max(fps, 1):.1f} sn")
    for i in range(a.n):
        f = int(n * (i + 0.5) / a.n)
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            continue
        s = a.width / img.shape[1]
        img = cv2.resize(img, (a.width, int(img.shape[0] * s)))
        H, W = img.shape[:2]
        for k in range(1, 10):
            x, y = int(W * k / 10), int(H * k / 10)
            cv2.line(img, (x, 0), (x, H), (0, 255, 255), 1)
            cv2.line(img, (0, y), (W, y), (0, 255, 255), 1)
            cv2.putText(img, f"{k / 10:.1f}", (x + 2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
            cv2.putText(img, f"{k / 10:.1f}", (2, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        cv2.putText(img, f"frame {f}  t={f / max(fps, 1):.1f}s", (10, H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        p = os.path.join(a.out, f"frame_{i:02d}_{f}.jpg")
        cv2.imwrite(p, img)
        print("kaydedildi:", p)


if __name__ == "__main__":
    main()
