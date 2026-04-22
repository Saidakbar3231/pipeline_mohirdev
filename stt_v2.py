"""
stt_v2.py
─────────────────────────────────────────────────────────────────
Yo'l 2: Gemini STT Pipeline
  Audio → Audio filter → Gemini STT → Matn filter → Sifat baho
  → Status → metadata_v2.jsonl
"""

from filter_audio import filter_audio
from filter_text import filter_text_v2
from gemini_utils import transcribe_audio_gemini, score_transcription, determine_status_v2
from config import (
    V2_SNR_MIN, V2_DURATION_MIN, V2_DURATION_MAX,
    V2_SILENCE_MAX, V2_WORD_MIN, V2_REPEAT_MAX,
    V2_SCORE_AUTO_APPROVE, V2_SCORE_AUTO_REJECT,
)


def process_segment_v2(segment: dict, filters: dict) -> dict:
    result = {**segment, "pipeline": "gemini_stt", "status": "pending"}

    # ── 1. Audio filtri ───────────────────────────────────────────
    # filter_noisy: yoqilganda SNR tekshiruvi qattiqroq (15 dB chegara)
    snr_min = 15.0 if filters.get("filter_noisy", False) else V2_SNR_MIN
    # filter_long_silence: o'chirilganda sukunat tekshiruvi bekor
    silence_max = V2_SILENCE_MAX if filters.get("filter_long_silence", True) else 100.0

    ok, reason = filter_audio(
        segment,
        snr_min=snr_min,
        duration_min=filters.get("duration_min", V2_DURATION_MIN),
        duration_max=filters.get("duration_max", V2_DURATION_MAX),
        silence_max=silence_max,
        check_language=filters.get("check_language", False),
    )

    result["snr_score"]     = segment.get("snr_score")
    result["silence_ratio"] = segment.get("silence_ratio")

    if not ok:
        return {**result, "status": "filtered", "reason": f"audio: {reason}"}

    # ── 2. Gemini STT (filter flaglari prompt orqali uzatiladi) ───
    try:
        text, filter_tags = transcribe_audio_gemini(segment["file"], filters=filters)
    except Exception as e:
        return {**result, "status": "filtered", "reason": f"Gemini STT xato: {e}"}

    if not text or "TUSHUNARSIZ" in filter_tags:
        return {**result, "status": "filtered", "reason": "tushunarsiz audio"}

    # filter_background_music: Gemini MUSIQA_BOR tegi qaytarsa filtrla
    if filters.get("filter_background_music", True) and "MUSIQA_BOR" in filter_tags:
        return {**result, "status": "filtered", "reason": "fon musiqasi aniqlandi"}

    # filter_multiple_speakers: Gemini KO'P_OVOZ tegi qaytarsa filtrla
    if filters.get("filter_multiple_speakers", True) and "KO'P_OVOZ" in filter_tags:
        return {**result, "status": "filtered", "reason": "ko'p ovoz aniqlandi"}

    result["transcription"] = text
    result["gemini_filters"] = filter_tags

    # ── 3. Matn filtri ────────────────────────────────────────────
    ok, reason = filter_text_v2(
        text,
        word_min=filters.get("word_min", 5),
        repeat_max=filters.get("repeat_max", 50.0),
        check_mixed=filters.get("check_mixed", True),
    )
    if not ok:
        return {**result, "status": "filtered", "reason": f"matn: {reason}"}

    # ── 4. Gemini sifat bahosi ────────────────────────────────────
    score = score_transcription(text)
    result["gemini_score"] = score

    status = determine_status_v2(
        score,
        auto_approve_min=filters.get("score_approve_min", 4),
        auto_reject_max=filters.get("score_reject_max", 2),
    )
    result["status"] = status

    return result
