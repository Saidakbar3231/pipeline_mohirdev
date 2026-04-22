import os
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

# ─────────────────────────────────────────
# 1. MUSAN filter
# ─────────────────────────────────────────

def filter_musan_files(musan_path, min_duration=1.0):
    removed = 0
    for root, _, files in os.walk(musan_path):
        for f in files:
            if not f.endswith(".wav"):
                continue
            path = os.path.join(root, f)
            try:
                duration = librosa.get_duration(path=path)
                if duration < min_duration:
                    os.remove(path)
                    removed += 1
            except Exception:
                os.remove(path)
                removed += 1
    print(f"[INFO] Removed {removed} short/invalid MUSAN files")


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

def get_pipeline():
    MUSAN_PATH = "./musan"
    return Compose([
        TimeStretch(min_rate=0.8, max_rate=1.2, p=0.8),
        PitchShift(min_semitones=-4, max_semitones=4, p=0.8),
        OneOf([
            AddBackgroundNoise(
                sounds_path=f"{MUSAN_PATH}/noise/free-sound",
                min_snr_db=-5, max_snr_db=5, p=1.0,
            ),
            AddBackgroundNoise(
                sounds_path=f"{MUSAN_PATH}/noise/sound-bible",
                min_snr_db=-5, max_snr_db=5, p=1.0,
            ),
            AddBackgroundNoise(
                sounds_path=f"{MUSAN_PATH}/speech/librivox",
                min_snr_db=0, max_snr_db=10, p=1.0,
            ),
            AddBackgroundNoise(
                sounds_path=f"{MUSAN_PATH}/music/jamendo",
                min_snr_db=-5, max_snr_db=5, p=1.0,
            ),
        ], p=1.0),
        LowPassFilter(min_cutoff_freq=500, max_cutoff_freq=4000, p=0.5),
        Gain(min_gain_db=-2, max_gain_db=2, p=0.5),
    ])


# ─────────────────────────────────────────
# 4. Apply — 100% ehtimol (Har safar augment qilinadi)
# ─────────────────────────────────────────

def apply_augment(example):
    import io
    import soundfile as sf
    import librosa
    
    # Ehtimollik tekshiruvi (masalan, AUG_PROB=0.3 bo'lsa, 70% hollarda original qoladi)
    # Tizim random soni AUG_PROB dan katta bo'lsa original qoladi degani:
    # 1.0 (100%) bo'lganda random.random() hech qachon 1.0 dan katta bo'lmaydi, hamma augmentga o'tadi
    if np.random.random() > AUG_PROB:
        audio_data = example["audio"]
        audio_path = audio_data if isinstance(audio_data, str) else audio_data.get("path")
        if not audio_path or not os.path.exists(audio_path):
            return example
            
        wav, _ = librosa.load(audio_path, sr=SR)
        wav = to_mono_1d(wav)
        
        buf = io.BytesIO()
        sf.write(buf, wav, SR, format='WAV')
        wav_bytes = buf.getvalue()
        
        return {
            "audio": {"path": "aug_" + os.path.basename(audio_path), "bytes": wav_bytes},
            "text": example["text"],
            "duration": round(len(wav) / SR, 4),
        }

    audio_data = example["audio"]
    audio_path = audio_data if isinstance(audio_data, str) else audio_data.get("path")
    if not audio_path or not os.path.exists(audio_path):
        return example
        
    wav, _ = librosa.load(audio_path, sr=SR)
    wav = to_mono_1d(wav)

    # Bo'sh tekshiruv
    if len(wav) == 0:
        return example

    duration = example.get("duration") or (len(wav) / SR)
    if duration > 24.5:
        buf = io.BytesIO()
        sf.write(buf, wav, SR, format='WAV')
        wav_bytes = buf.getvalue()
        return {
            "audio": {"path": "aug_" + os.path.basename(audio_path), "bytes": wav_bytes},
            "text": example["text"],
            "duration": round(len(wav) / SR, 4),
        }

    try:
        pipeline = get_pipeline()
        # Audiomentations call expects samples as first positional arg, sample_rate as second
        wav_aug = pipeline(wav, sample_rate=SR)

        print("[DEBUG] Successfully ran pipeline augmentation for file:", audio_path)

        # Augmentdan keyin ham normalize
        wav_aug = to_mono_1d(wav_aug)

        if not np.isfinite(wav_aug).all():
            buf = io.BytesIO()
            sf.write(buf, wav, SR, format='WAV')
            wav_bytes = buf.getvalue()
            return {
                "audio": {"path": "aug_" + os.path.basename(audio_path), "bytes": wav_bytes},
                "text": example["text"],
                "duration": round(len(wav) / SR, 4),
            }

        wav_aug = np.clip(wav_aug, -1.0, 1.0)

        buf = io.BytesIO()
        sf.write(buf, wav_aug, SR, format='WAV')
        wav_bytes = buf.getvalue()

        return {
            "audio": {"path": "augmented_" + os.path.basename(audio_path), "bytes": wav_bytes},
            "text":  example["text"],
            "duration": round(len(wav_aug) / SR, 4),
        }
    except Exception as e:
        import traceback
        with open("aug_worker_error.log", "a") as f:
            f.write(f"Error on {audio_path}: {e}\n{traceback.format_exc()}\n")
        print(f"[WARNING] Augment failed: {e}, using original")
        buf = io.BytesIO()
        sf.write(buf, wav, SR, format='WAV')
        wav_bytes = buf.getvalue()
        return {
            "audio": {"path": "aug_" + os.path.basename(audio_path), "bytes": wav_bytes},
            "text": example["text"],
            "duration": round(len(wav) / SR, 4),
        }


# ─────────────────────────────────────────
# 5. Main
# ─────────────────────────────────────────

def main():
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