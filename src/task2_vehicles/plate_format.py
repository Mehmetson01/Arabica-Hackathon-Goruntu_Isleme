"""Plaka metni temizleme ve Türkiye plaka formatı doğrulama/düzeltme.

TR formatı: [İl kodu 01-81][1-3 harf][2-4 rakam]
    1 harf  -> 4 rakam      (34 A 1234)
    2 harf  -> 3 veya 4     (34 AB 123 / 34 AB 1234)
    3 harf  -> 2 veya 3     (34 ABC 12 / 34 ABC 123)
Plakada Q, W, X ve Türkçe karakterler kullanılmaz.
OCR'ın sık karıştırdığı karakterler konuma göre düzeltilir (O<->0, I<->1, B<->8 ...).
"""
from __future__ import annotations

import re

TR_REGEX = re.compile(r"^(0[1-9]|[1-7][0-9]|8[01])([A-PR-VYZ]{1,3})(\d{2,4})$")
TR_LETTERS = set("ABCDEFGHIJKLMNOPRSTUVYZ")
TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "U": "0", "I": "1", "L": "1", "J": "1", "T": "7", "Z": "2",
            "S": "5", "B": "8", "G": "6", "A": "4"}
TO_LETTER = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G", "4": "A", "7": "T", "Q": "O",
             "W": "V", "X": "K"}
VALID_COMBOS = {(1, 4), (2, 3), (2, 4), (3, 2), (3, 3)}


def clean(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.upper().replace("İ", "I"))


def is_valid_tr(text: str) -> bool:
    m = TR_REGEX.match(text)
    if not m:
        return False
    return (len(m.group(2)), len(m.group(3))) in VALID_COMBOS


def _coerce(seg: str, table: dict, want_digit: bool):
    out, changes = [], 0
    for ch in seg:
        ok = ch.isdigit() if want_digit else (ch in TR_LETTERS)
        if ok:
            out.append(ch)
        elif ch in table:
            out.append(table[ch])
            changes += 1
        else:
            return None, 99
    return "".join(out), changes


def correct_tr(text: str):
    """(Guncellendi) Yabanci plakalar icin TR format zorlamasi kaldirildi."""
    t = clean(text)
    return t, True, 0


def format_tr(text: str) -> str:
    """(Guncellendi) Yabanci plakalar oldugu gibi gosterilecek."""
    return text
