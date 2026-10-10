"""Videonun ilk karesinde fare ile 2 nokta tıklayarak sayım çizgisi seçme aracı.

Kullanım:
    python tools/pick_line.py --video kisi_sayimi.mp4 [--frame 100]
Sol tık: nokta ekle (2 nokta) | r: sıfırla | Enter/q: bitir
Çıktı: --line parametresine verilecek normalize "x1,y1,x2,y2" değeri.
"""
import argparse

import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    a = ap.parse_args()
    cap = cv2.VideoCapture(a.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, a.frame)
    ok, img = cap.read()
    if not ok:
        raise SystemExit("Kare okunamadı")
    s = a.width / img.shape[1]
    img = cv2.resize(img, (a.width, int(img.shape[0] * s)))
    H, W = img.shape[:2]
    pts = []

    def on_mouse(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN and len(pts) < 2:
            pts.append((x, y))

    cv2.namedWindow("pick_line")
    cv2.setMouseCallback("pick_line", on_mouse)
    while True:
        vis = img.copy()
        for p in pts:
            cv2.circle(vis, p, 6, (0, 255, 255), -1)
        if len(pts) == 2:
            cv2.line(vis, pts[0], pts[1], (0, 255, 255), 2)
        cv2.putText(vis, "Sol tik: 2 nokta | r: sifirla | Enter: bitir", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow("pick_line", vis)
        k = cv2.waitKey(20) & 0xFF
        if k == ord("r"):
            pts.clear()
        elif k in (13, ord("q")) and len(pts) == 2:
            break
    cv2.destroyAllWindows()
    (x1, y1), (x2, y2) = pts
    line_str = f"{x1 / W:.3f},{y1 / H:.3f},{x2 / W:.3f},{y2 / H:.3f}"
    with open("configs/task1_line.txt", "w", encoding="utf-8") as f:
        f.write(line_str)
    print(f"Yeni cizgi kaydedildi: {line_str}")


if __name__ == "__main__":
    main()
