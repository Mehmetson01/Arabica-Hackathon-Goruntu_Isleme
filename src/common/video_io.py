"""Video okuma/yazma ve FPS ölçümü.

- ThreadedVideoReader: decode işlemini ayrı thread'de yapar (GPU inference ile paralel).
- FFmpegWriter: imageio-ffmpeg ile gelen ffmpeg binary'sine raw kare pipe eder ->
  tarayıcıda oynatılabilen H.264 mp4 üretir. ffmpeg yoksa OpenCV mp4v'ye düşer.
"""
from __future__ import annotations

import queue
import subprocess
import threading
import time

import cv2
import numpy as np


class ThreadedVideoReader:
    def __init__(self, path: str, start_frame: int = 0, max_frames: int | None = None,
                 stride: int = 1, queue_size: int = 16):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Video açılamadı: {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if start_frame > 0:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        self.start_frame = start_frame
        self.stride = max(1, stride)
        self.max_frames = max_frames
        self.q: queue.Queue = queue.Queue(maxsize=queue_size)
        self._stop = False
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    @property
    def total_to_process(self) -> int:
        n = max(0, self.frame_count - self.start_frame)
        if self.max_frames:
            n = min(n, self.max_frames * self.stride)
        return n // self.stride

    def _run(self):
        idx = self.start_frame
        produced = 0
        while not self._stop:
            ok, frame = self.cap.read()
            if not ok:
                break
            if (idx - self.start_frame) % self.stride == 0:
                self.q.put((idx, frame))
                produced += 1
                if self.max_frames and produced >= self.max_frames:
                    break
            idx += 1
        self.q.put(None)

    def __iter__(self):
        while True:
            item = self.q.get()
            if item is None:
                return
            yield item

    def release(self):
        self._stop = True
        try:
            while True:
                self.q.get_nowait()
        except queue.Empty:
            pass
        self.cap.release()


class FFmpegWriter:
    def __init__(self, path: str, width: int, height: int, fps: float):
        self.path = path
        self.size = (width, height)
        self.proc = None
        self.cv_writer = None
        try:
            import imageio_ffmpeg
            exe = imageio_ffmpeg.get_ffmpeg_exe()
            cmd = [exe, "-y", "-loglevel", "error",
                   "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
                   "-r", f"{fps:.3f}", "-i", "-",
                   "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                   "-pix_fmt", "yuv420p", "-movflags", "+faststart", path]
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        except Exception as e:  # pragma: no cover
            print(f"[uyarı] ffmpeg kullanılamadı ({e}), OpenCV mp4v kullanılıyor")
            self.cv_writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    def write(self, frame: np.ndarray):
        if frame.shape[1] != self.size[0] or frame.shape[0] != self.size[1]:
            frame = cv2.resize(frame, self.size)
        if self.proc is not None:
            self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        else:
            self.cv_writer.write(frame)

    def release(self):
        if self.proc is not None:
            self.proc.stdin.close()
            self.proc.wait()
        if self.cv_writer is not None:
            self.cv_writer.release()


class FPSMeter:
    """Üstel hareketli ortalama ile anlık FPS."""

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.fps = 0.0
        self._last = None
        self.start = time.perf_counter()
        self.n = 0

    def tick(self) -> float:
        now = time.perf_counter()
        if self._last is not None:
            inst = 1.0 / max(now - self._last, 1e-6)
            self.fps = inst if self.fps == 0 else (1 - self.alpha) * self.fps + self.alpha * inst
        self._last = now
        self.n += 1
        return self.fps

    @property
    def average(self) -> float:
        return self.n / max(time.perf_counter() - self.start, 1e-6)


def fmt_ts(seconds: float) -> str:
    m, s = divmod(max(0.0, seconds), 60)
    h, m = divmod(int(m), 60)
    return f"{h:02d}:{m:02d}:{s:06.3f}"
