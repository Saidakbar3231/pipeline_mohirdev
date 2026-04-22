"""
stt_v1.py — AIsha API async pipeline
"""
 
import time
import requests
from pathlib import Path
 
from config import (
    STT_API_KEY, STT_LANGUAGE,
    V1_SNR_MIN, V1_DURATION_MIN, V1_DURATION_MAX,
    V1_SILENCE_MAX, V1_WORD_MIN, V1_REPEAT_MAX, V1_CHANGE_MAX,
)
from filter_audio import filter_audio, compute_change_ratio
from filter_text import filter_text_v1
from gemini_utils import polish_text
 
STT_POST_URL  = "https://back.aisha.group/api/v2/stt/post/"
STT_GET_URL   = "https://back.aisha.group/api/v2/stt/get/"
POLL_INTERVAL = 3
POLL_TIMEOUT  = 300
 
 
def submit_audio(file_path: str, api_key: str) -> str:
    headers = {"x-api-key": api_key}
    fname = Path(file_path).name
    with open(file_path, "rb") as f:
        files = {"audio": (fname, f, "audio/wav")}
        data  = {"title": fname, "has_diarization": "false", "language": STT_LANGUAGE}
        r = requests.post(STT_POST_URL, headers=headers,
                          files=files, data=data, timeout=60)
    r.raise_for_status()
    result = r.json()
    job_id = result.get("id") or result.get("job_id") or result.get("task_id") or ""
    if not job_id:
        raise ValueError(f"job_id topilmadi: {result}")
    return str(job_id)
 
 
def poll_result(job_id: str, api_key: str) -> str:
    headers = {"x-api-key": api_key}
    url = f"{STT_GET_URL}{job_id}/"
    elapsed = 0
 
    while elapsed < POLL_TIMEOUT:
        time.sleep(POLL_INTERVAL)
        elapsed += POLL_INTERVAL
 
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        data = r.json()
 
        status = str(data.get("status") or "").upper()
 
        if status == "SUCCESS":
            # AIsha API "transcript" field ishlatadi
            text = (data.get("transcript") or
                    data.get("text") or
                    data.get("transcription") or "")
            return str(text).strip()
 
        elif status in ("FAILED", "ERROR"):
            raise ValueError(f"AIsha API xato: {data}")
 
        # PENDING, PROCESSING — davom ettirish
 
    raise TimeoutError(f"job_id={job_id} {POLL_TIMEOUT}s da natija kelmadi")
 
 
def call_stt_api(file_path: str, api_key: str = "") -> str:
    key = api_key or STT_API_KEY
    if not key:
        raise ValueError("AIsha API key kiritilmagan!")
    job_id = submit_audio(file_path, key)
    return poll_result(job_id, key)
 
 
def process_segment_v1(segment: dict, filters: dict, polish_opts: dict,
                        api_key: str = "") -> dict:
    result = {**segment, "pipeline": "aisha_stt", "status": "pending"}
 
    # ── 1. Audio filtri ──────────────────────────────────────────
    ok, reason = filter_audio(
        segment,
        snr_min=filters.get("snr_min", V1_SNR_MIN),
        duration_min=filters.get("duration_min", V1_DURATION_MIN),
        duration_max=filters.get("duration_max", V1_DURATION_MAX),
        silence_max=filters.get("silence_max", V1_SILENCE_MAX),
        check_language=filters.get("check_language", False),
    )
    result["snr_score"]     = segment.get("snr_score")
    result["silence_ratio"] = segment.get("silence_ratio")
 
    if not ok:
        return {**result, "status": "filtered", "reason": f"audio: {reason}"}
 
    # ── 2. AIsha STT ─────────────────────────────────────────────
    try:
        raw_text = call_stt_api(segment["file"], api_key=api_key)
    except Exception as e:
        return {**result, "status": "filtered", "reason": f"STT xato: {e}"}
 
    if not raw_text:
        return {**result, "status": "filtered", "reason": "STT bo'sh natija"}
 
    result["original_text"] = raw_text
 
    # ── 3. Matn filtri ───────────────────────────────────────────
    ok, reason = filter_text_v1(
        raw_text,
        word_min=filters.get("word_min", V1_WORD_MIN),
        repeat_max=filters.get("repeat_max", V1_REPEAT_MAX),
        check_noise=filters.get("check_noise", True),
        check_mixed=filters.get("check_mixed", False),
    )
    if not ok:
        return {**result, "status": "filtered", "reason": f"matn: {reason}"}
 
    # ── 4. Gemini tozalash ───────────────────────────────────────
    if polish_opts.get("enabled", True):
        try:
            polished = polish_text(
                raw_text,
                normalize_numbers=polish_opts.get("normalize_numbers", True),
                fix_spelling=polish_opts.get("fix_spelling", True),
                fix_punctuation=polish_opts.get("fix_punctuation", True),
                transliterate_ru=polish_opts.get("transliterate_ru", False),
            )
            result["transcription"] = polished
            change_pct = compute_change_ratio(raw_text, polished)
            result["change_ratio"]  = change_pct
            change_max = filters.get("gemini_change_max", V1_CHANGE_MAX)
 
            if change_pct < 20:
                result["status"] = "approved"
            elif change_pct <= change_max:
                result["status"] = "pending"
            else:
                result["transcription"] = raw_text
                result["status"]        = "pending"
                result["reason"]        = f"Gemini ko'p o'zgartirdi ({change_pct:.0f}%)"
 
        except Exception as e:
            result["transcription"] = raw_text
            result["status"]        = "pending"
            result["reason"]        = f"Gemini xato: {e}"
    else:
        result["transcription"] = raw_text
        result["status"]        = "approved"
 
    return result