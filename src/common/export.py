"""Sonuçların CSV / JSON ve dashboard veri dosyası olarak dışa aktarılması."""
from __future__ import annotations

import csv
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def write_csv(rows: list[dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        open(path, "w", encoding="utf-8-sig").close()
        return
    keys = list(rows[0].keys())
    for r in rows[1:]:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:  # utf-8-sig: Excel Türkçe uyumu
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def write_json(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_dashboard_data(key: str, obj):
    """dashboard/data/<key>.js -> file:// üzerinden de çalışan (fetch'siz) veri dosyası."""
    d = os.path.join(ROOT, "dashboard", "data")
    os.makedirs(d, exist_ok=True)
    js = ("window.HACKATHON_DATA = window.HACKATHON_DATA || {};\n"
          f"window.HACKATHON_DATA[{json.dumps(key)}] = {json.dumps(obj, ensure_ascii=False)};\n")
    with open(os.path.join(d, f"{key}.js"), "w", encoding="utf-8") as f:
        f.write(js)


def rel_to_root(path: str) -> str:
    return os.path.relpath(os.path.abspath(path), ROOT).replace("\\", "/")
