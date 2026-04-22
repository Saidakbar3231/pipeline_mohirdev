"""
json_processor.py
─────────────────────────────────────────────────────────────────
JSON fayl yuklash orqali dataset yaratishning 2 yo'li:

  Yo'l A — Python Only:
    JSON ichidagi mavjud diarization (start/end/text) dan foydalanib
    audio URL ni yuklab oladi, timestamplar bo'yicha qirqadi va
    dataset yaratadi. Gemini API talab qilinmaydi.

  Yo'l B — Gemini:
    JSON ichidagi audio URL ni yuklab oladi, har bir diarization
    chunkini Gemini API orqali QAYTA transkripsiya qiladi va
    yangi matn bilan dataset yaratadi.

INPUT JSON FORMATI:
  [
    {
      "id": 123456,
      "audio_url": "https://...",
      "diarization": [
        {"start": 0.0, "end": 3.5, "text": "Alo.", "speaker": "Operator"},
        ...
      ],
      "duration": 120.5
    },
    ...
  ]
"""

import os
import json
import re
import subprocess
import requests
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse

from config import DOWNLOAD_DIR, SAMPLE_RATE, OUTPUT_DIR

# ─── Trash filter ────────────────────────────────────────────────
TRASH_REGEX = re.compile(
    r"^\[(GUDOK|SHOVQIN|TINCH|MUSIQA|TUSHUNARSIZ)\]$",
    re.IGNORECASE
)

# ─── Non-Latin filter ─────────────────────────────────────────────
# Detect symbols from Cyrillic, Arabic, etc. blocks
NON_LATIN_REGEX = re.compile(
    r"[\u0400-\u04FF\u0500-\u052F\u2DE0-\u2DFF\uA640-\uA69F\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]"
)

def _is_trash(text: str) -> bool:
    return bool(TRASH_REGEX.match(text.strip()))


def _is_non_latin(text: str) -> bool:
    return bool(NON_LATIN_REGEX.search(text))


def _safe_unpack_gemini(result) -> tuple:
    """Gemini result ni xavfsiz (text, tags) tuple ga ochadi."""
    if isinstance(result, tuple) and len(result) >= 2:
        return result[0], result[1]
    if isinstance(result, tuple) and len(result) == 1:
        return result[0], []
    return result, []


# =====================================================================
# SHARED HELPER — JSON → Sliced Audio Items for main pipeline
# =====================================================================

def extract_chunks_from_json(json_path: str, output_dir: str, log_cb=print) -> list:
    """
    JSON fayldagi diarization ma'lumotlaridan foydalanib audio chunklar yaratadi.
    Har bir chunk standard audio_item dict formatida qaytariladi:
      {file, file_name, source, source_url, reference_text, start, end, duration, speaker}

    Bu chunklar keyinchalik Yo'l 1 yoki Yo'l 2 (Gemini) pipeline orqali
    ishlangan — xuddi YouTube yoki HuggingFace dan kelgan audio kabi.
    """
    os.makedirs(output_dir, exist_ok=True)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = data if isinstance(data, list) else [data]
    log_cb(f"📂 JSON: {len(records)} ta yozuv")

    audio_items = []

    for rec_idx, record in enumerate(records, 1):
        record_id   = record.get("id", f"rec_{rec_idx}")
        audio_url   = record.get("audio_url", "")
        diarization = record.get("diarization", [])

        if not audio_url:
            log_cb(f"  ⚠ [{rec_idx}] audio_url yo'q, o'tkazildi")
            continue

        log_cb(f"  [{rec_idx}/{len(records)}] Record {record_id} — {len(diarization)} segment")

        # Download full audio
        safe_id    = re.sub(r'[\\/*?:"<>|]', '_', str(record_id))
        full_fname = f"full_{safe_id}.wav"
        full_path  = os.path.join(output_dir, full_fname)
        if not _download_audio(audio_url, full_path, log_cb):
            continue

        # Slice by diarization timestamps
        for chunk_idx, chunk in enumerate(diarization):
            text    = chunk.get("text", "").strip()
            start   = chunk.get("start", 0)
            end     = chunk.get("end", 0)
            speaker = chunk.get("speaker", "")

            # Pre-filter obvious trash — don't even slice these
            if _is_trash(text):
                continue
            if end - start < 0.5:
                continue

            chunk_fname = f"{safe_id}_seg{chunk_idx:04d}.wav"
            chunk_path  = os.path.join(output_dir, chunk_fname)

            if not _slice_audio(full_path, chunk_path, start, end):
                continue

            audio_items.append({
                "file":           chunk_path,
                "file_name":      chunk_fname,
                "source":         "json",
                "source_url":     audio_url,
                "reference_text": text,       # original transcript (not used by Gemini path)
                "speaker":        speaker,
                "start":          start,
                "end":            end,
                "duration":       round(end - start, 3),
            })

    log_cb(f"  ✅ {len(audio_items)} ta chunk yaratildi")
    return audio_items


def _download_audio(audio_url: str, out_path: str, log_cb=print) -> bool:
    """URL dan audio yuklab, WAV ga o'tkazadi. True = muvaffaqiyatli."""
    if os.path.exists(out_path):
        return True
    try:
        log_cb(f"  ⬇ Yuklanmoqda: {audio_url[:70]}...")
        r = requests.get(audio_url, timeout=120, stream=True)
        r.raise_for_status()
        tmp = out_path + ".tmp"
        with open(tmp, "wb") as f:
            for chunk_data in r.iter_content(chunk_size=16384):
                f.write(chunk_data)
        # ffmpeg bilan WAV 16kHz mono
        cmd = [
            "ffmpeg", "-y", "-i", tmp,
            "-ar", str(SAMPLE_RATE), "-ac", "1",
            "-f", "wav", out_path
        ]
        result = subprocess.run(cmd, capture_output=True)
        os.remove(tmp)
        if result.returncode != 0:
            log_cb(f"  ❌ ffmpeg xato: {result.stderr.decode()[:200]}")
            return False
        return True
    except Exception as e:
        log_cb(f"  ❌ Yuklab olishda xato: {e}")
        return False


def _slice_audio(src: str, out_path: str, start: float, end: float) -> bool:
    """ffmpeg bilan audio ni start-end oralig'ida kesadi."""
    if os.path.exists(out_path):
        return True
    duration = round(end - start, 3)
    if duration <= 0:
        return False
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", src,
        "-t", str(duration),
        "-ar", str(SAMPLE_RATE), "-ac", "1",
        "-f", "wav", out_path
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        print(f"  ❌ slice_audio xato: {result.stderr.decode()[:200]}")
    return result.returncode == 0


# =====================================================================
# YO'L A — PYTHON ONLY
# =====================================================================

def process_json_python(
    json_path: str,
    output_dir: str,
    log_cb=print
) -> list:
    """
    JSON fayl ichidagi diarization (start/end/text) dan foydalanib
    audio ni qirqadi va dataset yaratadi.
    Gemini API ishlatilmaydi.

    Returns: list of output file paths (JSONL + audio chunks folder)
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    audio_out_dir = os.path.join(output_dir, f"audio_python_{ts}")
    os.makedirs(audio_out_dir, exist_ok=True)
    jsonl_path = os.path.join(output_dir, f"dataset_python_{ts}.jsonl")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = data if isinstance(data, list) else [data]
    log_cb(f"📂 Jami {len(records)} ta yozuv topildi.")

    total_chunks = 0
    skipped_trash = 0
    skipped_nonlatin = 0
    skipped_download = 0

    for rec_idx, record in enumerate(records, 1):
        record_id   = record.get("id", f"rec_{rec_idx}")
        audio_url   = record.get("audio_url", "")
        diarization = record.get("diarization", [])

        log_cb(f"\n[{rec_idx}/{len(records)}] Record ID: {record_id} — {len(diarization)} segment")

        if not audio_url:
            log_cb(f"  ⚠ audio_url yo'q, o'tkazildi.")
            continue

        # Audio yuklab olish
        safe_id = re.sub(r'[\\/*?:"<>|]', '_', str(record_id))
        audio_fname = f"record_{safe_id}.wav"
        audio_path  = os.path.join(audio_out_dir, audio_fname)

        ok = _download_audio(audio_url, audio_path, log_cb)
        if not ok:
            skipped_download += 1
            continue

        # Har bir chunk uchun qirqish
        with open(jsonl_path, "a", encoding="utf-8") as jf:
            for chunk_idx, chunk in enumerate(diarization):
                text    = chunk.get("text", "").strip()
                start   = chunk.get("start", 0)
                end     = chunk.get("end", 0)
                speaker = chunk.get("speaker", "")

                # 1. Trash filter
                if _is_trash(text):
                    skipped_trash += 1
                    continue

                # 2. Non-Latin filter
                if _is_non_latin(text):
                    skipped_nonlatin += 1
                    continue

                # 3. Duration check
                if end - start < 0.5:
                    continue

                # 4. Audio qirqish
                chunk_fname = f"{safe_id}_chunk{chunk_idx:04d}.wav"
                chunk_path  = os.path.join(audio_out_dir, chunk_fname)
                if not _slice_audio(audio_path, chunk_path, start, end):
                    continue

                # 5. Metadata yozish
                entry = {
                    "file_name":     chunk_fname,
                    "transcription": text,
                    "speaker":       speaker,
                    "start":         start,
                    "end":           end,
                    "duration":      round(end - start, 3),
                    "source":        "json",
                    "source_url":    audio_url,
                    "record_id":     str(record_id),
                    "pipeline":      "python_only",
                }
                jf.write(json.dumps(entry, ensure_ascii=False) + "\n")
                total_chunks += 1

    log_cb(f"\n✅ Tugadi!")
    log_cb(f"  Yaratilgan chunklar: {total_chunks}")
    log_cb(f"  Trash o'chirildi  : {skipped_trash}")
    log_cb(f"  Non-Latin o'chirildi: {skipped_nonlatin}")
    log_cb(f"  Yuklab bo'lmadi   : {skipped_download}")
    log_cb(f"  JSONL: {jsonl_path}")
    log_cb(f"  Audio: {audio_out_dir}/")

    return [jsonl_path, audio_out_dir]


# =====================================================================
# YO'L B — GEMINI QAYTA TRANSKRIPSIYA
# =====================================================================

def process_json_gemini(
    json_path: str,
    output_dir: str,
    log_cb=print
) -> list:
    """
    JSON ichidagi audio URL ni yuklab oladi, diarization timestamplar
    bo'yicha qirqadi, har bir chunkni Gemini orqali QAYTA transkripsiya
    qiladi va dataset yaratadi. Mavjud 'text' ishlatilmaydi.
    """
    from gemini_utils import transcribe_audio_gemini

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    audio_out_dir = os.path.join(output_dir, f"audio_gemini_{ts}")
    os.makedirs(audio_out_dir, exist_ok=True)
    jsonl_path = os.path.join(output_dir, f"dataset_gemini_{ts}.jsonl")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = data if isinstance(data, list) else [data]
    log_cb(f"📂 Jami {len(records)} ta yozuv topildi.")

    total_chunks = 0
    skipped_trash = 0

    for rec_idx, record in enumerate(records, 1):
        record_id   = record.get("id", f"rec_{rec_idx}")
        audio_url   = record.get("audio_url", "")
        diarization = record.get("diarization", [])

        log_cb(f"\n[{rec_idx}/{len(records)}] Record ID: {record_id} — {len(diarization)} segment")

        if not audio_url:
            log_cb(f"  ⚠ audio_url yo'q, o'tkazildi.")
            continue

        # Audio yuklab olish
        safe_id = re.sub(r'[\\/*?:"<>|]', '_', str(record_id))
        audio_fname = f"record_{safe_id}.wav"
        audio_path  = os.path.join(audio_out_dir, audio_fname)

        ok = _download_audio(audio_url, audio_path, log_cb)
        if not ok:
            continue

        # Har bir chunk — Gemini transkripsiya
        with open(jsonl_path, "a", encoding="utf-8") as jf:
            for chunk_idx, chunk in enumerate(diarization):
                original_text = chunk.get("text", "").strip()
                start   = chunk.get("start", 0)
                end     = chunk.get("end", 0)
                speaker = chunk.get("speaker", "")

                # Trash chunklar (original matndan) o'chiriladi
                if _is_trash(original_text):
                    skipped_trash += 1
                    continue

                if end - start < 0.5:
                    continue

                # Audio qirqish
                chunk_fname = f"{safe_id}_chunk{chunk_idx:04d}.wav"
                chunk_path  = os.path.join(audio_out_dir, chunk_fname)
                if not _slice_audio(audio_path, chunk_path, start, end):
                    continue

                log_cb(f"  🤖 Gemini transkripsiya: chunk {chunk_idx} [{start:.1f}s - {end:.1f}s]")

                # Gemini orqali qayta transkripsiya
                try:
                    result = transcribe_audio_gemini(chunk_path)
                    gemini_text, filter_tags = _safe_unpack_gemini(result)
                except Exception as e:
                    log_cb(f"  ❌ Gemini xato: {e}")
                    continue

                # [TUSHUNARSIZ], [MUSIQA_BOR], [KO'P_OVOZ] yoki bo'sh natija o'chiriladi
                if not gemini_text or _is_trash(gemini_text) or filter_tags:
                    continue

                entry = {
                    "file_name":        chunk_fname,
                    "transcription":    gemini_text,
                    "original_text":    original_text,
                    "speaker":          speaker,
                    "start":            start,
                    "end":              end,
                    "duration":         round(end - start, 3),
                    "source":           "json",
                    "source_url":       audio_url,
                    "record_id":        str(record_id),
                    "gemini_filters":   filter_tags,
                    "pipeline":         "gemini",
                }
                jf.write(json.dumps(entry, ensure_ascii=False) + "\n")
                total_chunks += 1

    log_cb(f"\n✅ Tugadi!")
    log_cb(f"  Gemini transkripsiya qildi: {total_chunks} chunk")
    log_cb(f"  Trash o'chirildi: {skipped_trash}")
    log_cb(f"  JSONL: {jsonl_path}")
    log_cb(f"  Audio: {audio_out_dir}/")

    return [jsonl_path, audio_out_dir]


# =====================================================================
# YO'L 3 — PYTHON FILTER (Trash + Non-Latin + Timestamp Style)
# =====================================================================

def _detect_style(diarization: list) -> str:
    if not diarization:
        return "short_clip"
    
    prev_end = None
    max_val  = 0
    
    for entry in diarization:
        start = entry.get("start", 0)
        end   = entry.get("end", 0)
        max_val = max(max_val, start, end)
        
        if prev_end is not None and start < prev_end - 5:
            return "has_resets"
            
        prev_end = end
        
    if max_val > 3600:
        return "possibly_ms"
    elif max_val > 60:
        return "normal"
    else:
        return "short_clip"

def process_json_python_filter(
    json_path: str,
    output_dir: str,
    log_cb=print
) -> list:
    """
    JSON fayl ichidagi diarization dan foydalanadi.
    Trash, Non-Latin filterlari qilinadi, Style aniqlanadi. Audio qirqiladi.
    Review panel uchun {file, transcription, speaker, duration, style} formatda list qaytaradi.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    audio_out_dir = os.path.join(output_dir, f"audio_v3_{ts}")
    os.makedirs(audio_out_dir, exist_ok=True)
    jsonl_path = os.path.join(output_dir, f"dataset_v3_{ts}.jsonl")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = data if isinstance(data, list) else [data]
    log_cb(f"📂 Jami {len(records)} ta yozuv topildi (Yo'l 3).")

    total_chunks = 0
    skipped_trash = 0
    skipped_nonlatin = 0
    skipped_download = 0
    review_results = []

    for rec_idx, record in enumerate(records, 1):
        record_id   = record.get("id", f"rec_{rec_idx}")
        audio_url   = record.get("audio_url", "")
        diarization = record.get("diarization", [])

        if not audio_url:
            log_cb(f"  ⚠ [{rec_idx}] audio_url yo'q, o'tkazildi.")
            continue

        # Oldin filterlar
        cleaned_diar = []
        for chunk in diarization:
            text = chunk.get("text", "").strip()
            if _is_trash(text):
                skipped_trash += 1
                continue
            if _is_non_latin(text):
                skipped_nonlatin += 1
                continue
            cleaned_diar.append(chunk)

        style = _detect_style(cleaned_diar)
        
        if not cleaned_diar:
             continue
        
        log_cb(f"\n[{rec_idx}/{len(records)}] Record ID: {record_id} — Style: {style} ({len(cleaned_diar)} ta clean chunk)")

        # Audio yuklab olish
        safe_id = re.sub(r'[\\/*?:"<>|]', '_', str(record_id))
        audio_fname = f"record_{safe_id}.wav"
        audio_path  = os.path.join(audio_out_dir, audio_fname)

        ok = _download_audio(audio_url, audio_path, log_cb)
        if not ok:
            skipped_download += 1
            continue

        with open(jsonl_path, "a", encoding="utf-8") as jf:
            for chunk_idx, chunk in enumerate(cleaned_diar):
                text    = chunk.get("text", "").strip()
                start   = chunk.get("start", 0)
                end     = chunk.get("end", 0)
                speaker = chunk.get("speaker", "")

                if end - start < 0.4:
                    print(f"  ⚠ [{record_id}] qisqa chunk o'tkazildi: {end-start:.2f}s")
                    continue

                chunk_fname = f"{safe_id}_chunk{chunk_idx:04d}.wav"
                chunk_path  = os.path.join(audio_out_dir, chunk_fname)
                if not _slice_audio(audio_path, chunk_path, start, end):
                    continue

                duration_s = round(end - start, 3)

                entry = {
                    "file_name":     chunk_fname,
                    "transcription": text,
                    "speaker":       speaker,
                    "start":         start,
                    "end":           end,
                    "duration":      duration_s,
                    "source":        "json",
                    "source_url":    audio_url,
                    "record_id":     str(record_id),
                    "style":         style,
                    "pipeline":      "v3_python_filter",
                    "file":          os.path.abspath(chunk_path)
                }
                jf.write(json.dumps(entry, ensure_ascii=False) + "\n")
                total_chunks += 1
                review_results.append(entry)

    log_cb(f"\n✅ Yo'l 3 yakunlandi!")
    log_cb(f"  Yaratilgan chunklar: {total_chunks}")
    log_cb(f"  Trash/NonLatin: {skipped_trash} / {skipped_nonlatin}")
    log_cb(f"  JSONL: {jsonl_path}")

    return review_results
