import os
import sys
import argparse
from collections import Counter
import numpy as np
import librosa
from datasets import load_from_disk, concatenate_datasets
from audiomentations import (
    Compose, AddBackgroundNoise, AddGaussianNoise,
    TimeStretch, PitchShift, OneOf, Gain, LowPassFilter
)

MUSAN_PATH = os.environ.get("MUSAN_PATH", "./musan")
INPUT_DS   = os.environ.get("INPUT_DS", "birlashtirilgan_dataset")
OUTPUT_DS  = os.environ.get("OUTPUT_DS", "birlashtirilgan_dataset_augmented")
SR         = 16000
NUM_PROC   = int(os.environ.get("NUM_PROC", "16"))

AUG_PROB   = float(os.environ.get("AUG_PROB", "1.0"))


def _noise_snr_range(noise_level):
    """Background-noise loudness percent (0-100) → (min_snr_db, max_snr_db).

    Lower SNR = louder noise. Per spec formula:
        min_snr = 30 - (noise_level * 0.27);  max_snr = min_snr + 5
        100% → 3.0/8.0 dB (loud)
         50% → 16.5/21.5 dB (medium, default)
          0% → 30.0/35.0 dB (barely audible)
    Input is clamped to [0, 100] so a bad value can never widen the range.
    """
    noise_level = max(0.0, min(100.0, float(noise_level)))
    min_snr = 30.0 - (noise_level * 0.27)
    return min_snr, min_snr + 5.0


# ─────────────────────────────────────────
# 1. MUSAN filter
# ─────────────────────────────────────────

def filter_musan_files(musan_path, min_duration=1.0):
    """
    Identify MUSAN .wav files that are too short or unreadable and
    move them into <musan_path>/_quarantine/ (preserving relative
    structure). NEVER deletes user data — quarantine is reversible.
    Files inside _quarantine are outside the subdirs that
    AddBackgroundNoise scans, so they are effectively excluded.
    """
    quarantine_root = os.path.join(musan_path, "_quarantine")
    quarantined = 0
    skipped = 0

    for root, _, files in os.walk(musan_path):
        # Don't descend into the quarantine folder itself
        if os.path.commonpath([os.path.abspath(root),
                               os.path.abspath(quarantine_root)]) == \
           os.path.abspath(quarantine_root):
            continue

        for f in files:
            if not f.endswith(".wav"):
                continue
            path = os.path.join(root, f)

            reason = None
            try:
                duration = librosa.get_duration(path=path)
                if duration < min_duration:
                    reason = f"too short ({duration:.2f}s < {min_duration}s)"
            except Exception as e:
                reason = f"unreadable ({type(e).__name__}: {e})"

            if reason is None:
                continue

            # Move to quarantine, preserving relative path
            rel = os.path.relpath(path, musan_path)
            dest = os.path.join(quarantine_root, rel)
            try:
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                os.replace(path, dest)
                quarantined += 1
                print(f"[INFO] Quarantined {rel}: {reason}")
            except Exception as move_err:
                skipped += 1
                print(f"[WARN] Could not quarantine {rel} ({reason}): "
                      f"{move_err}. Leaving file in place.")

    print(f"[INFO] MUSAN filter: quarantined {quarantined}, "
          f"skipped {skipped} (left in place). "
          f"Quarantine dir: {quarantine_root}")


# ─────────────────────────────────────────
# 2. Universal audio normalizer
# ─────────────────────────────────────────

def to_mono_1d(arr) -> np.ndarray:
    """
    Har qanday audio array ni 1D float32 mono ga o'tkazadi.
    (samples,)        → o'zgarishsiz
    (1, samples)      → squeeze
    (2, samples)      → stereo mix down
    (samples, 1)      → squeeze
    (samples, 2)      → stereo mix down
    boshqa shape      → flatten
    """
    if arr is None:
        return np.zeros(SR, dtype=np.float32)

    wav = np.array(arr, dtype=np.float32)

    if wav.ndim == 1:
        return wav

    if wav.ndim == 2:
        # (1, samples) yoki (samples, 1)
        if wav.shape[0] == 1:
            return wav[0]
        if wav.shape[1] == 1:
            return wav[:, 0]
        # (2, samples)
        if wav.shape[0] == 2:
            return wav.mean(axis=0)
        # (samples, 2)
        if wav.shape[1] == 2:
            return wav.mean(axis=1)
        # boshqa 2D
        return wav.mean(axis=0) if wav.shape[0] < wav.shape[1] else wav.mean(axis=1)

    # 3D+ → flatten
    return wav.flatten()


# ─────────────────────────────────────────
# 3. Pipeline — multiprocessing safe
# ─────────────────────────────────────────

_PIPELINE = None


def _get_or_build_pipeline():
    """
    Process-local lazy cache.
    datasets.map(num_proc=N) forks/spawns N workers; each worker's
    _PIPELINE starts as None and is populated on first sample, then
    reused for every subsequent sample in that worker's shard.
    """
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = get_pipeline()
    return _PIPELINE


def get_pipeline():
    # Noise loudness is driven by NOISE_LEVEL (0-100%). It is read from the
    # environment — NOT a CLI arg directly — because dataset.map() spawns/forks
    # worker processes that re-import this module; env vars are inherited, argv
    # is not. main() sets NOISE_LEVEL from the --noise-level CLI arg before the
    # map() call, so both the in-process (num_proc=1) and worker paths agree.
    noise_level = float(os.environ.get("NOISE_LEVEL", "50.0"))
    min_snr, max_snr = _noise_snr_range(noise_level)
    return Compose([
        TimeStretch(min_rate=0.8, max_rate=1.2, p=0.8),
        PitchShift(min_semitones=-4, max_semitones=4, p=0.8),
        OneOf([
            AddBackgroundNoise(
                sounds_path=f"{MUSAN_PATH}/noise/free-sound",
                min_snr_db=min_snr, max_snr_db=max_snr, p=1.0,
            ),
            AddBackgroundNoise(
                sounds_path=f"{MUSAN_PATH}/noise/sound-bible",
                min_snr_db=min_snr, max_snr_db=max_snr, p=1.0,
            ),
            AddBackgroundNoise(
                sounds_path=f"{MUSAN_PATH}/speech/librivox",
                min_snr_db=min_snr, max_snr_db=max_snr, p=1.0,
            ),
            AddBackgroundNoise(
                sounds_path=f"{MUSAN_PATH}/music/jamendo",
                min_snr_db=min_snr, max_snr_db=max_snr, p=1.0,
            ),
        ], p=1.0),
        LowPassFilter(min_cutoff_freq=500, max_cutoff_freq=4000, p=0.5),
        Gain(min_gain_db=-2, max_gain_db=2, p=0.5),
    ])


# ─────────────────────────────────────────
# 4. Apply — 100% ehtimol (Har safar augment qilinadi)
# ─────────────────────────────────────────

def _load_wav(audio_data):
    """
    Load a 1D float32 mono waveform at SR from an HF Audio(decode=False) value.

    CRITICAL: after Dataset.save_to_disk(), HF Datasets *embeds* the audio as
    in-memory bytes and rewrites `audio["path"]` to a bare basename that no
    longer exists on disk. So the embedded `bytes` are the authoritative source;
    a filesystem path is only a fallback for datasets that still reference real
    files (e.g. a freshly built, not-yet-saved Dataset).

    Returns (wav | None, base_name). wav is None only when the audio is truly
    unreadable (no usable bytes AND no existing path).
    """
    import io

    # String form: audio is a plain filesystem path.
    if isinstance(audio_data, str):
        base_name = os.path.basename(audio_data) or "audio.wav"
        if audio_data and os.path.exists(audio_data):
            wav, _ = librosa.load(audio_data, sr=SR)
            return to_mono_1d(wav), base_name
        return None, base_name

    if isinstance(audio_data, dict):
        base_name = os.path.basename(audio_data.get("path") or "") or "audio.wav"
        raw = audio_data.get("bytes")
        # Embedded bytes first — this is what save_to_disk leaves behind.
        if raw:
            try:
                wav, _ = librosa.load(io.BytesIO(raw), sr=SR)
                return to_mono_1d(wav), base_name
            except Exception:
                pass  # fall through to a path-based read if one is available
        path = audio_data.get("path")
        if path and os.path.exists(path):
            wav, _ = librosa.load(path, sr=SR)
            return to_mono_1d(wav), base_name
        return None, base_name

    return None, "audio.wav"


def apply_augment(example):
    import io
    import soundfile as sf

    # Load the waveform from embedded bytes (authoritative after save_to_disk)
    # or, failing that, a real filesystem path. A None result means the sample
    # is genuinely unreadable — NOT merely that `path` points at a basename.
    wav, base_name = _load_wav(example["audio"])
    if wav is None:
        # Source truly unreadable — schema-stable pass-through.
        return {**example, "aug_status": "invalid_input"}

    # Ehtimollik tekshiruvi (masalan, AUG_PROB=0.3 bo'lsa, 70% hollarda original qoladi)
    # Tizim random soni AUG_PROB dan katta bo'lsa original qoladi degani:
    # 1.0 (100%) bo'lganda random.random() hech qachon 1.0 dan katta bo'lmaydi, hamma augmentga o'tadi
    if np.random.random() > AUG_PROB:
        buf = io.BytesIO()
        sf.write(buf, wav, SR, format='WAV')
        wav_bytes = buf.getvalue()

        # Random draw > AUG_PROB — sample intentionally NOT augmented (probability skip).
        return {
            "audio": {"path": "aug_" + base_name, "bytes": wav_bytes},
            "text": example["text"],
            "duration": round(len(wav) / SR, 4),
            "aug_status": "skip_prob",
        }

    # Bo'sh tekshiruv
    if len(wav) == 0:
        # File loaded but contains zero samples — nothing to augment.
        return {**example, "aug_status": "invalid_input"}

    duration = example.get("duration") or (len(wav) / SR)
    if duration > 24.5:
        buf = io.BytesIO()
        sf.write(buf, wav, SR, format='WAV')
        wav_bytes = buf.getvalue()
        # Audio exceeds the 24.5s upper bound — pass-through without augmentation.
        return {
            "audio": {"path": "aug_" + base_name, "bytes": wav_bytes},
            "text": example["text"],
            "duration": round(len(wav) / SR, 4),
            "aug_status": "skip_long",
        }

    try:
        pipeline = _get_or_build_pipeline()
        # Audiomentations call expects samples as first positional arg, sample_rate as second
        wav_aug = pipeline(wav, sample_rate=SR)

        # Augmentdan keyin ham normalize
        wav_aug = to_mono_1d(wav_aug)

        if not np.isfinite(wav_aug).all():
            buf = io.BytesIO()
            sf.write(buf, wav, SR, format='WAV')
            wav_bytes = buf.getvalue()
            # Pipeline produced NaN/Inf — numeric failure, fall back to original.
            return {
                "audio": {"path": "aug_" + base_name, "bytes": wav_bytes},
                "text": example["text"],
                "duration": round(len(wav) / SR, 4),
                "aug_status": "failed",
            }

        wav_aug = np.clip(wav_aug, -1.0, 1.0)

        buf = io.BytesIO()
        sf.write(buf, wav_aug, SR, format='WAV')
        wav_bytes = buf.getvalue()

        # Successful augmentation — the only branch that produces real augmented audio.
        return {
            "audio": {"path": "augmented_" + base_name, "bytes": wav_bytes},
            "text":  example["text"],
            "duration": round(len(wav_aug) / SR, 4),
            "aug_status": "augmented",
        }
    except Exception as e:
        import traceback
        with open("aug_worker_error.log", "a") as f:
            f.write(f"Error on {base_name}: {e}\n{traceback.format_exc()}\n")
        print(f"[WARNING] Augment failed: {e}, using original")
        buf = io.BytesIO()
        sf.write(buf, wav, SR, format='WAV')
        wav_bytes = buf.getvalue()
        # Augmentation raised an exception — caught error, fall back to original.
        return {
            "audio": {"path": "aug_" + base_name, "bytes": wav_bytes},
            "text": example["text"],
            "duration": round(len(wav) / SR, 4),
            "aug_status": "failed",
        }


# ─────────────────────────────────────────
# 5. Main
# ─────────────────────────────────────────

def main():
    # ── CLI args ──────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="MUSAN + audiomentations augmentation")
    parser.add_argument(
        "--noise-level", type=float, default=50.0,
        help="Background noise loudness, 0-100%% (default 50). "
             "Higher = louder noise (lower SNR).",
    )
    # parse_known_args: the orchestrator may add flags later; never hard-fail here.
    args, _ = parser.parse_known_args()
    noise_level = max(0.0, min(100.0, args.noise_level))
    # Propagate to dataset.map() workers (spawn/fork inherit the environment).
    os.environ["NOISE_LEVEL"] = str(noise_level)
    _min_snr, _max_snr = _noise_snr_range(noise_level)
    print(f"[INFO] Noise level: {noise_level:.0f}% -> SNR {_min_snr:.1f}..{_max_snr:.1f} dB "
          f"(lower = louder)")

    # MUSAN startup check — the OneOf(noise) stage has p=1.0, so MUSAN is
    # mandatory. Stop immediately with a clear message instead of letting every
    # sample silently fall back to its un-augmented original.
    if not os.path.isdir(MUSAN_PATH):
        print(f"[ERROR] MUSAN_PATH not found: {MUSAN_PATH}")
        print("        Set MUSAN_PATH to the MUSAN corpus root and re-run.")
        sys.exit(2)
    missing = [
        sub for sub in ("noise/free-sound", "noise/sound-bible",
                        "speech/librivox", "music/jamendo")
        if not os.path.isdir(os.path.join(MUSAN_PATH, sub))
    ]
    if missing:
        print(f"[ERROR] MUSAN_PATH={MUSAN_PATH} is missing required subdirs: {missing}")
        print("        Augmentation cannot add background noise without them. Aborting.")
        sys.exit(2)

    # MUSAN filter
    print("[INFO] Filtering MUSAN files...")
    filter_musan_files(MUSAN_PATH, min_duration=1.0)

    # Load
    print(f"\n[INFO] Loading: {INPUT_DS}")
    ds = load_from_disk(INPUT_DS)
    # Cast to avoid torchcodec decode on map
    from datasets import Audio
    ds = ds.cast_column("audio", Audio(decode=False))
    
    total_hours = sum(d for d in ds["duration"] if d is not None) / 3600
    print(f"[INFO] Original: {len(ds):,} samples | ~{total_hours:.0f} soat")

    # Augment
    print(f"\n[INFO] Augmenting (100% per sample)...")
    ds_aug = ds.map(
        apply_augment,
        num_proc=NUM_PROC,
        desc="augmenting",
        keep_in_memory=False,
    )

    # Post-map sanity guard. If NO sample was actually augmented, the run
    # produced a dataset of un-augmented originals — a silent no-op that used to
    # look identical to success. Surface it loudly and exit non-zero so the UI
    # reports failure instead of "complete". (AUG_PROB==0 deliberately augments
    # nothing, so it is exempt.)
    if "aug_status" in ds_aug.column_names:
        status_counts = dict(Counter(ds_aug["aug_status"]))
        print(f"[INFO] aug_status distribution: {status_counts}")
        augmented = status_counts.get("augmented", 0)
        if AUG_PROB > 0 and augmented == 0:
            print("[ERROR] 0 of {:,} samples were augmented — every row hit a "
                  "skip/invalid/failed branch.".format(len(ds_aug)))
            print("        Likely causes: audio could not be loaded (embedded "
                  "bytes unreadable) or MUSAN noise sources are empty.")
            print("        Refusing to save an un-augmented dataset. Aborting.")
            sys.exit(2)

    # Shuffle
    print("[INFO] Shuffling...")
    ds_full = ds_aug.shuffle(seed=42)

    # Stats
    total_final = sum(d for d in ds_full["duration"] if d is not None) / 3600
    print(f"\n{'='*50}")
    print(f"[INFO] Original  : {len(ds):,} samples | ~{total_hours:.0f} soat")
    print(f"[INFO] Final     : {len(ds_full):,} samples | ~{total_final:.0f} soat")
    print(f"[INFO] Taxminiy disk: ~{total_final * 0.25:.0f} GB")
    print(f"{'='*50}")

    # Save
    print(f"\n[INFO] Saving to {OUTPUT_DS}...")
    ds_full.save_to_disk(
        OUTPUT_DS,
        num_proc=NUM_PROC,
        max_shard_size="2GB",
    )
    print(f"\n[INFO] Done! -> {OUTPUT_DS}")


if __name__ == "__main__":
    main()