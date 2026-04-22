"""
filter_audio.py  +  filter_text.py
─────────────────────────────────────────────────────────────────
Audio va matn sifat filtrlari.
"""

import re
import numpy as np


# ════════════════════════════════════════════════════════════════
# AUDIO FILTRLARI
# ════════════════════════════════════════════════════════════════

def compute_snr(file_path: str) -> float:
    """
    Signal-to-Noise Ratio (dB) hisoblaydi.
    Yuqori = sifatli audio.
    """
    try:
        import librosa
        y, sr = librosa.load(file_path, sr=None, mono=True)

        # RMS signal
        signal_rms = np.sqrt(np.mean(y ** 2))
        if signal_rms == 0:
            return 0.0

        # Shovqin: eng past 10% dan hisoblash
        frame_size  = int(sr * 0.02)
        hop_size    = int(sr * 0.01)
        frames      = librosa.util.frame(y, frame_length=frame_size, hop_length=hop_size)
        frame_rms   = np.sqrt(np.mean(frames ** 2, axis=0))
        noise_rms   = np.percentile(frame_rms, 10)

        if noise_rms == 0:
            return 60.0

        snr = 20 * np.log10(signal_rms / noise_rms)
        return round(float(snr), 2)

    except Exception:
        return -1.0


def compute_silence_ratio(file_path: str, threshold_db: float = -40) -> float:
    """
    Fayldagi sukunat ulushini % da qaytaradi.
    """
    try:
        from pydub import AudioSegment
        audio    = AudioSegment.from_file(file_path)
        total_ms = len(audio)
        silent   = sum(1 for ms in range(0, total_ms, 10)
                       if audio[ms:ms+10].dBFS < threshold_db)
        return round(silent * 10 / total_ms * 100, 1)
    except Exception:
        return -1.0


def detect_language(file_path: str) -> str:
    """
    Audio tilini aniqlaydi (langdetect orqali).
    Qaytaradi: "uz", "ru", "en", "unknown"
    """
    try:
        import whisper
        model = whisper.load_model("tiny")
        audio = whisper.load_audio(file_path)
        audio = whisper.pad_or_trim(audio)
        mel   = whisper.log_mel_spectrogram(audio)
        _, probs = model.detect_language(mel)
        return max(probs, key=probs.get)
    except Exception:
        return "unknown"


def filter_audio(segment: dict,
                 snr_min: float = 15.0,
                 duration_min: float = 3.0,
                 duration_max: float = 30.0,
                 silence_max: float = 80.0,
                 check_language: bool = False,
                 target_lang: str = "uz") -> tuple[bool, str]:
    """
    Audio segmentni filtrlaydi.

    Returns:
        (True, "ok")          — filtrdan o'tdi
        (False, "sabab")      — tashlab yuboriladi
    """
    file_path = segment["file"]
    duration  = segment.get("duration", 0)

    # 1. Uzunlik
    if duration < duration_min:
        return False, f"juda qisqa ({duration:.1f}s < {duration_min}s)"
    if duration > duration_max:
        return False, f"juda uzun ({duration:.1f}s > {duration_max}s)"

    # 2. SNR
    snr = compute_snr(file_path)
    segment["snr_score"] = snr
    if snr_min > 0 and snr >= 0 and snr < snr_min:
        return False, f"SNR past ({snr:.1f}dB < {snr_min}dB)"

    # 3. Sukunat
    silence = compute_silence_ratio(file_path)
    segment["silence_ratio"] = silence
    if silence > 0 and silence > silence_max:
        return False, f"ko'p sukunat ({silence:.0f}% > {silence_max}%)"

    # 4. Til (ixtiyoriy, sekin)
    if check_language:
        lang = detect_language(file_path)
        segment["detected_lang"] = lang
        if lang != target_lang and lang != "unknown":
            return False, f"til mos emas ({lang} != {target_lang})"

    return True, "ok"


# ════════════════════════════════════════════════════════════════
# MATN FILTRLARI
# ════════════════════════════════════════════════════════════════

# Shovqin belgilari
NOISE_PATTERNS = [
    r"^(mm+|uh+|ah+|eh+|um+|hmm+)[\s.,!?]*$",   # faqat undovlar
    r"^\W+$",                                       # faqat tinish
    r"^\.{3,}$",                                    # faqat nuqtalar
]

def compute_repeat_ratio(text: str) -> float:
    """Takror so'zlar ulushini % da hisoblaydi."""
    words = text.lower().split()
    if not words:
        return 100.0
    unique = set(words)
    return round((1 - len(unique) / len(words)) * 100, 1)


def has_mixed_scripts(text: str) -> bool:
    """Kirill va Lotin aralash ekanligini tekshiradi."""
    has_cyrillic = bool(re.search(r'[а-яёА-ЯЁ]', text))
    has_latin    = bool(re.search(r'[a-zA-Z]', text))
    # Qisqa lotin so'zlar (allo, ok) — normal, lekin ko'p bo'lsa shubhali
    if has_cyrillic and has_latin:
        latin_words = len(re.findall(r'[a-zA-Z]+', text))
        total_words = len(text.split())
        return latin_words / max(total_words, 1) > 0.3
    return False


def filter_text_v1(text: str,
                   word_min: int = 3,
                   repeat_max: float = 70.0,
                   check_noise: bool = True,
                   check_mixed: bool = False) -> tuple[bool, str]:
    """
    Yo'l 1 matn filtri (o'rtacha qattiq).
    """
    if not text or not text.strip():
        return False, "bo'sh matn"

    words = text.strip().split()

    # 1. Min so'z
    if len(words) < word_min:
        return False, f"kam so'z ({len(words)} < {word_min})"

    # 2. Takror
    repeat = compute_repeat_ratio(text)
    if repeat > repeat_max:
        return False, f"ko'p takror ({repeat:.0f}% > {repeat_max}%)"

    # 3. Shovqin belgilar
    if check_noise:
        for pat in NOISE_PATTERNS:
            if re.match(pat, text.strip(), re.IGNORECASE):
                return False, "shovqin/undov matni"

    # 4. Aralash yozuv
    if check_mixed and has_mixed_scripts(text):
        return False, "kirill/lotin aralash"

    return True, "ok"


def filter_text_v2(text: str,
                   word_min: int = 5,
                   repeat_max: float = 50.0,
                   check_mixed: bool = True) -> tuple[bool, str]:
    """
    Yo'l 2 matn filtri (qattiqroq).
    """
    if not text or not text.strip():
        return False, "bo'sh matn"

    words = text.strip().split()

    if len(words) < word_min:
        return False, f"kam so'z ({len(words)} < {word_min})"

    repeat = compute_repeat_ratio(text)
    if repeat > repeat_max:
        return False, f"ko'p takror ({repeat:.0f}% > {repeat_max}%)"

    # Shovqin belgilar — Yo'l 2 da har doim tekshiriladi
    for pat in NOISE_PATTERNS:
        if re.match(pat, text.strip(), re.IGNORECASE):
            return False, "shovqin/undov matni"

    if check_mixed and has_mixed_scripts(text):
        return False, "til aralash"

    return True, "ok"


def compute_change_ratio(original: str, polished: str) -> float:
    """Gemini qancha o'zgartirganini % da hisoblaydi."""
    orig_words     = set(original.lower().split())
    polished_words = set(polished.lower().split())
    if not orig_words:
        return 100.0
    changed = orig_words.symmetric_difference(polished_words)
    return round(len(changed) / len(orig_words) * 100, 1)
