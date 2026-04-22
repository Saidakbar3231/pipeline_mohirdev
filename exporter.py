"""
exporter.py
─────────────────────────────────────────────────────────────────
Natijalarni JSONL va CSV formatida saqlaydi.
"""

import os
import json
import csv
from datetime import datetime
from config import (
    OUTPUT_DIR, OUTPUT_V1_JSONL, OUTPUT_V1_CSV,
    OUTPUT_V2_JSONL, OUTPUT_V2_CSV, REPORT_FILE
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Har bir pipeline uchun maydonlar tartibi
V1_FIELDS = [
    "file_name", "transcription",
    "source", "source_url", "duration",
    "change_ratio", "pipeline", "status"
]

V2_FIELDS = [
    "file_name", "transcription",
    "source", "source_url", "duration",
    "pipeline", "status"
]


def _clean(entry: dict, fields: list, extra_fields: list) -> dict:
    """Faqat kerakli maydonlarni oladi, bo'lmasa None."""
    all_fields = fields + [f for f in extra_fields if f not in fields]
    return {k: entry.get(k) for k in all_fields}


def append_jsonl(entry: dict, pipeline: str = "v1",
                 extra_fields: list = None, path: str = None):
    """Bitta yozuvni JSONL ga qo'shadi (append rejimi)."""
    extra_fields = extra_fields or []
    if path is None:
        path = OUTPUT_V1_JSONL if pipeline == "v1" else OUTPUT_V2_JSONL
    fields = V1_FIELDS if pipeline == "v1" else V2_FIELDS

    clean = _clean(entry, fields, extra_fields)
    clean = {k: v for k, v in clean.items() if v is not None}

    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(clean, ensure_ascii=False) + "\n")


def _write_audiodir_sidecar(metadata_path: str, audio_dir: str):
    """Metadata fayl yoniga .audiodir sidecar yozadi."""
    sidecar = os.path.splitext(metadata_path)[0] + ".audiodir"
    with open(sidecar, "w", encoding="utf-8") as f:
        f.write(os.path.abspath(audio_dir))


def read_audiodir_sidecar(metadata_path: str) -> str:
    """Metadata fayl yonidagi .audiodir sidecar ni o'qiydi. Topilmasa ''."""
    sidecar = os.path.splitext(metadata_path)[0] + ".audiodir"
    if os.path.exists(sidecar):
        return open(sidecar, encoding="utf-8").read().strip()
    return ""


def save_all(entries: list, pipeline: str = "v1",
             save_jsonl: bool = True, save_csv: bool = True,
             extra_fields: list = None,
             jsonl_path: str = None, csv_path: str = None,
             audio_dir: str = None):
    """
    Barcha yozuvlarni saqlaydi (to'liq qayta yozish).
    audio_dir berilsa yoniga .audiodir sidecar ham yoziladi —
    Tab 6 da HF Push uchun audio papkasini avtomatik to'ldiradi.
    """
    extra_fields = extra_fields or []
    fields = V1_FIELDS if pipeline == "v1" else V2_FIELDS

    jp = jsonl_path or (OUTPUT_V1_JSONL if pipeline == "v1" else OUTPUT_V2_JSONL)
    cp = csv_path   or (OUTPUT_V1_CSV   if pipeline == "v1" else OUTPUT_V2_CSV)

    all_fields = fields + [f for f in extra_fields if f not in fields]
    cleaned    = [_clean(e, fields, extra_fields) for e in entries]
    cleaned    = [{k: v for k, v in e.items() if v is not None} for e in cleaned]

    if save_jsonl:
        with open(jp, "w", encoding="utf-8") as f:
            for entry in cleaned:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    if save_csv:
        with open(cp, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_fields,
                                    extrasaction="ignore")
            writer.writeheader()
            writer.writerows(cleaned)

    # Audio papka yo'lini sidecar fayl sifatida saqlash
    _adir = audio_dir or os.path.abspath("segments")
    if save_jsonl:
        _write_audiodir_sidecar(jp, _adir)
    if save_csv:
        _write_audiodir_sidecar(cp, _adir)

    return jp, cp


def save_report(stats: dict):
    """Pipeline statistikasini JSON ga saqlaydi."""
    stats["generated_at"] = datetime.now().isoformat()
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


def load_existing_jsonl(pipeline: str = "v1") -> dict:
    """Allaqachon ishlangan fayllarni yuklaydi."""
    path = OUTPUT_V1_JSONL if pipeline == "v1" else OUTPUT_V2_JSONL
    done = {}
    if not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    e = json.loads(line)
                    done[e.get("file_name", "")] = e
                except Exception:
                    pass
    return done
