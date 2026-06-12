"""
stt_v2.py
─────────────────────────────────────────────────────────────────
Yo'l 2: Gemini STT Pipeline
  Audio → Audio filter → Gemini STT → Matn filter → Sifat baho
  → Status → metadata_v2.jsonl
"""

import os

from filter_audio import filter_audio
from filter_text import filter_text_v2
from gemini_utils import transcribe_audio_gemini, score_transcription, determine_status_v2
from config import (
    V2_SNR_MIN, V2_DURATION_MIN, V2_DURATION_MAX,
    V2_SILENCE_MAX, V2_WORD_MIN, V2_REPEAT_MAX,
    V2_SCORE_AUTO_APPROVE, V2_SCORE_AUTO_REJECT,
)


def process_segment_v2(segment: dict, filters: dict, log=None) -> dict:
    """Yo'l 2 segment pipeline.

    log: optional callable(str). When the caller passes its UI-log sink (e.g.
    app._log), every rejection point emits a structured, per-filter
    `[FILTER_REJECT] <file> → reason: ...` line into the UI log stream so a run's
    segment loss can be attributed to an exact filter. Segments that pass STT but
    only reach the manual-review queue emit a `[FILTER_PENDING]` line — those are
    NOT rejected, they simply did not auto-approve.
    """
    def _emit(msg):
        if log:
            log(msg)

    fname = segment.get("file_name") or os.path.basename(segment.get("file", "") or "segment")
    result = {**segment, "pipeline": "gemini_stt", "status": "pending"}

    # ── 0. Lokal audio kontent tahlili — teglashtirish, filtrlash emas ─
    # Musiqa yoki ko'p ovoz aniqlansa segment O'CHIRILMAYDI.
    # Natija metadataga flag sifatida qo'shiladi va transcription davom etadi.
    if filters.get("filter_background_music", True):
        from filter_audio import detect_music
        result["has_background_music"] = detect_music(segment["file"])

    if filters.get("filter_multiple_speakers", True):
        from filter_audio import detect_multiple_speakers
        result["has_multiple_speakers"] = detect_multiple_speakers(segment["file"])

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
        _emit(f"[FILTER_REJECT] {fname} → reason: audio/{reason}")
        return {**result, "status": "filtered", "reason": f"audio: {reason}"}

    # ── 2. Gemini STT (filter flaglari prompt orqali uzatiladi) ───
    try:
        text, filter_tags = transcribe_audio_gemini(segment["file"], filters=filters)
    except Exception as e:
        _emit(f"[FILTER_REJECT] {fname} → reason: Gemini STT error: {e}")
        return {**result, "status": "filtered", "reason": f"Gemini STT xato: {e}"}

    if not text or "TUSHUNARSIZ" in filter_tags:
        _emit(f"[FILTER_REJECT] {fname} → reason: unintelligible audio (Gemini→TUSHUNARSIZ)")
        return {**result, "status": "filtered", "reason": "tushunarsiz audio"}

    result["transcription"] = text

    # ── 3. Matn filtri ────────────────────────────────────────────
    ok, reason = filter_text_v2(
        text,
        word_min=filters.get("word_min", 5),
        repeat_max=filters.get("repeat_max", 50.0),
        check_mixed=filters.get("check_mixed", True),
    )
    if not ok:
        _emit(f"[FILTER_REJECT] {fname} → reason: text/{reason}")
        return {**result, "status": "filtered", "reason": f"matn: {reason}"}

    # ── 4. Gemini sifat bahosi ────────────────────────────────────
    # NOTE: scoring logic itself is unchanged. We only read the same
    # approve/reject thresholds to label the structured log line.
    score = score_transcription(text)
    result["gemini_score"] = score

    approve_min = filters.get("score_approve_min", V2_SCORE_AUTO_APPROVE)
    reject_max  = filters.get("score_reject_max", V2_SCORE_AUTO_REJECT)
    status = determine_status_v2(
        score,
        auto_approve_min=approve_min,
        auto_reject_max=reject_max,
    )
    result["status"] = status

    if status == "rejected":
        _emit(f"[FILTER_REJECT] {fname} → reason: Gemini score too low "
              f"({score} ≤ {reject_max} reject-max)")
    elif status == "pending":
        _emit(f"[FILTER_PENDING] {fname} → reason: Gemini score {score} "
              f"(< {approve_min} approve-min) — manual review, NOT auto-approved")

    return result
