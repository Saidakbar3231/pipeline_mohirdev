"""
audio_utils.py
─────────────────────────────────────────────────────────────────
Audio fayllarni VAD orqali segmentlarga qirqadi.
"""

import os
import numpy as np
from pathlib import Path
from pydub import AudioSegment
from pydub.silence import split_on_silence

from config import (
    SEGMENTS_DIR, SAMPLE_RATE,
    VAD_MIN_SILENCE, VAD_SILENCE_DB, VAD_KEEP_SILENCE,
)

os.makedirs(SEGMENTS_DIR, exist_ok=True)


def to_mono_16k(audio: AudioSegment) -> AudioSegment:
    """Whisper uchun optimal: mono, 16kHz, 16-bit."""
    return audio.set_channels(1).set_frame_rate(SAMPLE_RATE).set_sample_width(2)


def reduce_noise(audio: AudioSegment, strength: float = 0.75) -> AudioSegment:
    """
    noisereduce orqali orqa fon shovqinini kamaytiradi.

    strength: 0.0 (ta'sirsiz) — 1.0 (maksimal tozalash)
              0.75 default — nutqni saqlab shovqinni kamaytiradi
    """
    try:
        import noisereduce as nr

        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        rate    = audio.frame_rate

        reduced = nr.reduce_noise(
            y=samples,
            sr=rate,
            prop_decrease=strength,   # shovqin kamayish kuchi
            stationary=False,         # musiqa/dinamik shovqin uchun
        )

        reduced = np.clip(reduced, -32768, 32767).astype(np.int16)
        clean   = audio._spawn(reduced.tobytes())
        return clean

    except ImportError:
        print("[audio_utils] noisereduce topilmadi: pip install noisereduce")
        return audio
    except Exception as e:
        print(f"[audio_utils] reduce_noise xato: {e}")
        return audio


def vad_chunk(audio_info: dict,
              min_silence: int = VAD_MIN_SILENCE,
              silence_db: float = VAD_SILENCE_DB,
              keep_silence: int = VAD_KEEP_SILENCE,
              min_sec: float = 3.0,
              max_sec: float = 30.0,
              noise_reduce: bool = True,
              noise_strength: float = 0.75,
              log=None) -> list[dict]:
    """
    Bitta audio faylni VAD orqali segmentlarga qirqadi.

    noise_reduce: True bo'lsa VAD dan oldin shovqin kamaytiriladi
    noise_strength: 0.0 — 1.0 (default 0.75)

    Returns:
        [{"file": "segments/xxx_seg_001.wav", "duration": 8.3, ...}, ...]
    """
    file_path = audio_info["file"]
    stem = Path(file_path).stem

    audio = AudioSegment.from_file(file_path)
    audio = to_mono_16k(audio)

    # Shovqin kamaytirish — VAD dan oldin
    if noise_reduce:
        audio = reduce_noise(audio, strength=noise_strength)

    duration_s = len(audio) / 1000

    # min_sec dan qisqa fayl — o'tkazib yuborish
    if duration_s < min_sec:
        if log:
            log(f"[VAD_DROP] {stem} → reason: butun fayl juda qisqa "
                f"({duration_s:.1f}s < {min_sec:.1f}s) — STT ga yetib bormaydi")
        return []

    # max_sec dan qisqa — qirqmasdan o'zi qaytaramiz
    if duration_s <= max_sec:
        out_path = os.path.join(SEGMENTS_DIR, f"{stem}_seg_001.wav")
        audio.export(out_path, format="wav")
        return [{**audio_info, "file": out_path,
                 "file_name": f"{stem}_seg_001.wav",
                 "duration": duration_s}]

    # VAD qirqish
    chunks = split_on_silence(
        audio,
        min_silence_len=min_silence,
        silence_thresh=silence_db,
        keep_silence=keep_silence
    )

    max_ms = int(max_sec * 1000)
    min_ms = int(min_sec * 1000)

    # Uzunlarni bo'lish
    split = []
    for c in chunks:
        if len(c) > max_ms:
            for start in range(0, len(c), max_ms):
                split.append(c[start:start + max_ms])
        else:
            split.append(c)

    # Qisqalarni tashlab ketish
    _before = len(split)
    split = [c for c in split if len(c) >= min_ms]
    _dropped = _before - len(split)
    if _dropped and log:
        # Bu yo'qotish STT/filtrlardan OLDIN sodir bo'ladi — process_segment_v2
        # logida ko'rinmaydi. Uzbek suhbat nutqida tabiiy pauzalar ko'p bo'lgani
        # uchun bu odatda eng katta segment yo'qotish manbai.
        log(f"[VAD_DROP] {stem} → reason: {_dropped} ta bo'lak {min_sec:.1f}s dan "
            f"qisqa — tashlandi (STT/filtrga yetib bormaydi)")

    if not split:
        return []

    results = []
    for i, chunk in enumerate(split, 1):
        fname = f"{stem}_seg_{i:03d}.wav"
        out_path = os.path.join(SEGMENTS_DIR, fname)
        chunk.export(out_path, format="wav")
        results.append({
            **audio_info,
            "file":      out_path,
            "file_name": fname,
            "duration":  len(chunk) / 1000,
        })

    return results


def get_duration(file_path: str) -> float:
    """Audio uzunligini soniyada qaytaradi."""
    audio = AudioSegment.from_file(file_path)
    return len(audio) / 1000
