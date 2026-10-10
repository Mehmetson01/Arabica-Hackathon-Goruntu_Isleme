"""Sanal çizgi geçiş sayacı.

Her track için referans noktasının (ör. ayak noktası) çizginin hangi tarafında
olduğu takip edilir. Taraf değiştiğinde (histerezis payı aşılarak) geçiş sayılır.
Aynı ID her yön için en fazla BİR kez sayılır -> tekrar sayım engellenir.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CrossEvent:
    track_id: int
    direction: str  # "in" | "out"
    frame_idx: int
    point: tuple[float, float]


@dataclass
class _TrackState:
    side: int = 0                 # -1 / +1 (0 = henüz belirlenmedi)
    counted: set = field(default_factory=set)
    last_seen: int = 0


class LineCounter:
    """İki noktayla tanımlı çizgi için yönlü sayaç.

    Args:
        p1, p2: Çizginin uç noktaları (piksel).
        margin: Histerezis payı (piksel). Nokta çizgiye bu mesafeden yakınsa
            taraf güncellenmez; titreşimden kaynaklı sahte sayımları önler.
        segment_pad: Çizgi doğrultusunda uçların dışına ne kadar taşmaya izin
            verildiği (çizgi uzunluğunun oranı).
        invert: "in" ve "out" yönlerini ters çevirir.
    """

    def __init__(self, p1, p2, margin: float = 6.0, segment_pad: float = 0.05,
                 invert: bool = False, forget_after: int = 300):
        self.p1 = np.array(p1, dtype=np.float32)
        self.p2 = np.array(p2, dtype=np.float32)
        d = self.p2 - self.p1
        self.length = float(np.linalg.norm(d)) + 1e-6
        self.dir = d / self.length
        self.normal = np.array([-self.dir[1], self.dir[0]], dtype=np.float32)
        self.margin = margin
        self.segment_pad = segment_pad
        self.invert = invert
        self.forget_after = forget_after
        self.states: dict[int, _TrackState] = {}
        self.count_in = 0
        self.count_out = 0
        self.events: list[CrossEvent] = []

    # ------------------------------------------------------------------
    def signed_distance(self, pt) -> float:
        return float(np.dot(np.asarray(pt, dtype=np.float32) - self.p1, self.normal))

    def _within_segment(self, pt) -> bool:
        t = float(np.dot(np.asarray(pt, dtype=np.float32) - self.p1, self.dir)) / self.length
        return -self.segment_pad <= t <= 1.0 + self.segment_pad

    def update(self, track_id: int, pt, frame_idx: int) -> CrossEvent | None:
        st = self.states.setdefault(track_id, _TrackState())
        st.last_seen = frame_idx
        dist = self.signed_distance(pt)
        if abs(dist) < self.margin:
            return None
        side = 1 if dist > 0 else -1
        event = None
        if st.side != 0 and side != st.side and self._within_segment(pt):
            # +1 -> -1 geçişi "in" kabul edilir (invert ile ters çevrilir)
            direction = "in" if (st.side == 1) else "out"
            if self.invert:
                direction = "out" if direction == "in" else "in"
            if direction not in st.counted:
                st.counted.add(direction)
                if direction == "in":
                    self.count_in += 1
                else:
                    self.count_out += 1
                event = CrossEvent(track_id, direction, frame_idx, (float(pt[0]), float(pt[1])))
                self.events.append(event)
        st.side = side
        return event

    def cleanup(self, frame_idx: int):
        """Uzun süredir görülmeyen track durumlarını sil (bellek)."""
        dead = [k for k, s in self.states.items() if frame_idx - s.last_seen > self.forget_after]
        for k in dead:
            del self.states[k]

    def has_crossed(self, track_id: int) -> bool:
        st = self.states.get(track_id)
        return bool(st and st.counted)

    @property
    def total(self) -> int:
        return self.count_in + self.count_out


def parse_line(spec: str, width: int, height: int):
    """"x1,y1,x2,y2" biçimindeki çizgiyi piksele çevirir.

    Değerler <= 1.0 ise oran (normalize) kabul edilir.
    """
    vals = [float(v) for v in spec.split(",")]
    if len(vals) != 4:
        raise ValueError("Çizgi 'x1,y1,x2,y2' formatında olmalı")
    if all(v <= 1.0 for v in vals):
        vals = [vals[0] * width, vals[1] * height, vals[2] * width, vals[3] * height]
    return (vals[0], vals[1]), (vals[2], vals[3])
