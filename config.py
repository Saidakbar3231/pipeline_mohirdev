"""
config.py — Barcha sozlamalar
"""
import os

# ─── API Kalitlar ────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# ─── Fine-tuned Model API (Yo'l 1) ──────────────────────────────
STT_API_URL      = os.getenv("STT_API_URL", "https://back.aisha.group/api/v2/stt/post/")
STT_API_KEY      = os.getenv("STT_API_KEY", "")
STT_LANGUAGE     = "uz"
# API formati: "openai" yoki "custom"
STT_API_FORMAT   = "custom"
STT_FILE_FIELD   = "audio"       # multipart field nomi
STT_TEXT_FIELD   = "text"        # javobdan matn olinadigan field

# ─── Gemini ─────────────────────────────────────────────────────
GEMINI_MODEL_STT    = "gemini-2.5-flash"   # Yo'l 2: STT uchun
GEMINI_MODEL_POLISH = "gemini-2.5-flash"   # Yo'l 1: tozalash uchun

# ─── Audio sozlamalari ───────────────────────────────────────────
AUDIO_FORMAT     = "wav"
SAMPLE_RATE      = 16000
CHANNELS         = 1

# ─── VAD sozlamalari ─────────────────────────────────────────────
VAD_MIN_SILENCE  = 500     # ms
VAD_SILENCE_DB   = -40     # dBFS
VAD_KEEP_SILENCE = 200     # ms

# ─── Yo'l 1 default filtrlari ────────────────────────────────────
V1_SNR_MIN       = 0.0     # 0 = SNR filtri o'chirilgan, qiymat saqlanadi
V1_DURATION_MIN  = 3.0     # soniya
V1_DURATION_MAX  = 30.0    # soniya
V1_SILENCE_MAX   = 80      # %
V1_WORD_MIN      = 3       # so'z
V1_REPEAT_MAX    = 70      # %
V1_CHANGE_MAX    = 50      # % — Gemini o'zgarish chegarasi

# ─── Yo'l 2 default filtrlari ────────────────────────────────────
V2_SNR_MIN       = 0.0     # 0 = SNR filtri o'chirilgan, qiymat saqlanadi
V2_DURATION_MIN  = 5.0     # soniya
V2_DURATION_MAX  = 30.0    # soniya
V2_SILENCE_MAX   = 70      # %
V2_WORD_MIN      = 5       # so'z
V2_REPEAT_MAX    = 50      # %
V2_SCORE_AUTO_APPROVE = 3  # va undan yuqori (3 = "o'rtacha" ham auto-approve)
V2_SCORE_AUTO_REJECT  = 2  # va undan past

# ─── Papkalar ────────────────────────────────────────────────────
DOWNLOAD_DIR     = "downloads"    # yuklab olingan audio
SEGMENTS_DIR     = "segments"     # qirqilgan segmentlar
OUTPUT_DIR       = "outputs"      # natija fayllar

# ─── Natija ──────────────────────────────────────────────────────
OUTPUT_V1_JSONL  = "outputs/metadata_v1.jsonl"
OUTPUT_V1_CSV    = "outputs/metadata_v1.csv"
OUTPUT_V2_JSONL  = "outputs/metadata_v2.jsonl"
OUTPUT_V2_CSV    = "outputs/metadata_v2.csv"
REPORT_FILE      = "outputs/report.json"
