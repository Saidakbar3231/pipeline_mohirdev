import os
import io
import sys
import logging
from typing import Optional
from dataclasses import dataclass

import numpy as np
import torch
from datasets import load_dataset, load_from_disk, concatenate_datasets, Audio, Features, Value, Dataset
from transformers import (
    WhisperFeatureExtractor,
    WhisperProcessor,
    WhisperTokenizer
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("mapping")


def _safe_under(target_path, *allowed_roots):
    """commonpath-based sandbox check (mirrors app._safe_under /
    pipeline_extensions._safe_under). Standalone copy so this script has no
    import dependency on the Flask app. Rejects dataset/output paths that escape
    the allowed roots. Returns (abs_path, ok)."""
    if not target_path:
        return None, False
    try:
        abs_target = os.path.realpath(os.path.abspath(target_path))
    except Exception:
        return None, False
    for root in allowed_roots:
        if not root:
            continue
        try:
            abs_root = os.path.realpath(os.path.abspath(root))
        except Exception:
            continue
        try:
            if os.path.commonpath([abs_target, abs_root]) == abs_root:
                return abs_target, True
        except ValueError:
            continue
    return abs_target, False


@dataclass
class Config:
    # Model
    model_name: str = os.environ.get("MODEL_NAME", "openai/whisper-large-v3")
    language: str = "Uzbek"
    task: str = "transcribe"

    # Dataset
    output_dir: str = os.environ.get("OUTPUT_DIR", "")
    hf_token: str = os.environ.get("HF_TOKEN", "")

    def __post_init__(self):
        if not self.output_dir:
            model_slug = self.model_name.split("/")[-1].replace("-", "_")
            self.output_dir = f"full_mapping_dataset_{model_slug}"

    # Audio
    sampling_rate: int = 16000
    min_audio_duration: float = 1.0


DS_NAMES = [d.strip() for d in os.environ.get("DS_NAMES", "").split(",") if d.strip()]
SR = 16000


def _to_mono_1d(arr) -> np.ndarray:
    """Collapse any audio array to 1-D float32 mono (matches augmentation.to_mono_1d)."""
    wav = np.asarray(arr, dtype=np.float32)
    if wav.ndim == 1:
        return wav
    if wav.ndim == 2:
        # soundfile returns (samples, channels); average channels down to mono.
        if wav.shape[1] in (1, 2) and wav.shape[1] <= wav.shape[0]:
            return wav.mean(axis=1)
        if wav.shape[0] in (1, 2):
            return wav.mean(axis=0)
        return wav.mean(axis=1)
    return wav.flatten()


def _load_wav(audio_data) -> Optional[np.ndarray]:
    """Load a 1-D float32 mono waveform at SR from an HF Audio(decode=False) value.

    Bytes-first: after Dataset.save_to_disk() the audio is embedded as in-memory
    bytes and `audio["path"]` is rewritten to a bare basename that no longer
    exists on disk (the exact failure that silently broke augmentation.py). So the
    embedded bytes are authoritative; a real filesystem path is only a fallback.

    Returns None ONLY when the audio is truly unreadable (no usable bytes/array
    AND no existing path) — never merely because `path` is a basename.
    """
    import soundfile as sf
    import librosa

    if isinstance(audio_data, dict):
        raw = audio_data.get("bytes")
        if raw:
            try:
                wav, sr = sf.read(io.BytesIO(raw))
            except Exception as e:
                # Corrupt/undecodable blob — fall through to path, then None.
                # Counted as a per-row failure in main(), never silently kept.
                log.debug("Embedded audio bytes undecodable: %s", e)
            else:
                wav = _to_mono_1d(wav)
                if sr != SR:
                    wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
                return wav.astype(np.float32)

        arr = audio_data.get("array")
        if arr is not None:
            wav = _to_mono_1d(arr)
            sr = audio_data.get("sampling_rate", SR)
            if sr != SR:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
            return wav.astype(np.float32)

        path = audio_data.get("path")
    else:
        path = audio_data if isinstance(audio_data, str) else None

    if path and os.path.exists(path):
        wav, _ = librosa.load(path, sr=SR)
        return _to_mono_1d(wav).astype(np.float32)

    return None


class WhisperDatasetPreprocessor:
    def __init__(self, config: Config):
        self.config = config
        # The processor/extractor/tokenizer are small config+vocab files (NOT the
        # multi-GB acoustic weights). A load failure here is fatal and must be
        # loud — never proceed to map() with a half-built preprocessor.
        try:
            self.feature_extractor = WhisperFeatureExtractor.from_pretrained(config.model_name)
            self.tokenizer = WhisperTokenizer.from_pretrained(
                config.model_name,
                language=config.language,
                task=config.task,
            )
            self.processor = WhisperProcessor.from_pretrained(
                config.model_name,
                language=config.language,
                task=config.task,
            )
        except Exception as e:
            log.error(
                "Failed to load Whisper processor for model_name=%r: %s",
                config.model_name, e,
            )
            log.error(
                "Check the model id, network access, and HF_TOKEN for gated models."
            )
            sys.exit(2)

    def prepare_features(self, batch: dict) -> dict:
        wav = _load_wav(batch.get("audio"))
        if wav is None or len(wav) == 0:
            # Unreadable / empty audio — mark failed (null tensors) instead of
            # silently emitting a row with no features. Dropped in main().
            log.debug("Unreadable audio, marking failed")
            batch["input_features"] = None
            batch["labels"] = None
            batch["map_status"] = "failed"
            return batch

        batch["input_features"] = self.feature_extractor(
            wav,
            sampling_rate=SR,
        ).input_features[0]
        batch["labels"] = self.tokenizer(batch.get("text") or "").input_ids
        batch["map_status"] = "ok"
        return batch


def load_and_concat(config: Config) -> Dataset:
    if not DS_NAMES:
        log.error("DS_NAMES is empty — nothing to map. Set DS_NAMES env var.")
        sys.exit(2)

    all_ds = []
    for name in DS_NAMES:
        log.info("Yuklanmoqda: %s", name)
        # A value that looks like a local path must stay inside the project tree.
        # HF hub ids (org/name, no OS separators, not on disk) are left as-is.
        looks_local = (os.sep in name) or ("/" in name) or ("\\" in name) or os.path.exists(name)
        if looks_local and os.path.exists(name):
            abs_name, ok = _safe_under(name, ".")
            if not ok:
                log.error("Dataset path escapes project root, refusing: %s", name)
                sys.exit(2)
            ds = load_from_disk(abs_name)
        elif os.path.exists(name):
            ds = load_from_disk(name)
        else:
            ds = load_dataset(
                name,
                token=config.hf_token if config.hf_token else None,
                split="train",
            )

        if "transcription" in ds.column_names:
            ds = ds.rename_column("transcription", "text")

        if "text" not in ds.column_names or "audio" not in ds.column_names:
            log.error(
                "Dataset %s is missing required column(s); has %s",
                name, ds.column_names,
            )
            sys.exit(2)

        # Ortiqcha ustunlarni olib tashlash (masalan, duration)
        cols_to_remove = [c for c in ds.column_names if c not in ["audio", "text"]]
        if cols_to_remove:
            ds = ds.remove_columns(cols_to_remove)

        # Audio faqat 16000 bo'lishi kerak
        ds = ds.cast_column("audio", Audio(sampling_rate=config.sampling_rate, decode=False))
        all_ds.append(ds)

    log.info("Datasetlar birlashtirilmoqda...")
    if not all_ds:
        log.error("No datasets found to map!")
        sys.exit(2)
    return concatenate_datasets(all_ds)


def main():
    config = Config()

    # Validate output location stays inside the project tree.
    abs_out, ok_out = _safe_under(config.output_dir, ".")
    if not ok_out:
        log.error("OUTPUT_DIR escapes project root, refusing: %s", config.output_dir)
        sys.exit(2)
    config.output_dir = abs_out

    # 1. Load
    dataset = load_and_concat(config)
    n_in = len(dataset)
    if n_in == 0:
        log.error("Loaded dataset is empty — nothing to map.")
        sys.exit(2)

    log.info("#" * 50)

    # 2. Preprocessor (model load is fatal-on-failure, see __init__)
    log.info("Preprocessor yaratilmoqda... (model=%s)", config.model_name)
    preprocessor = WhisperDatasetPreprocessor(config)

    # 3. Map features — plain Python loop, NOT dataset.map().
    #    On Python 3.14 the datasets library still routes through multiprocess +
    #    dill internally even at num_proc=1, and dill cannot pickle the bound
    #    `preprocessor.prepare_features` method (TypeError: _Pickler.
    #    _batch_setitems() missing 'obj'). Iterating in-process sidesteps that
    #    entirely. Only successfully-mapped ("ok") rows are kept, so the rebuilt
    #    Dataset has a clean, uniform schema (no null input_features to confuse
    #    Arrow type inference).
    log.info("Dataset tayyorlanmoqda... (%d samples in)", n_in)
    processed = []
    n_failed = 0
    for i, sample in enumerate(dataset):
        try:
            result = preprocessor.prepare_features(sample)
        except Exception as e:
            log.warning("Sample %d failed: %s", i, e)
            n_failed += 1
            continue
        if result.get("map_status") != "ok":
            # Unreadable/empty audio flagged inside prepare_features.
            n_failed += 1
            continue
        result.pop("map_status", None)
        processed.append(result)

    n_ok = len(processed)
    log.info("Mapped %d/%d samples (failed: %d)", n_ok, n_in, n_failed)
    if n_ok == 0:
        log.error(
            "0 of %d samples were mapped — every row had unreadable audio. "
            "Refusing to save an empty dataset.", n_in,
        )
        sys.exit(2)

    dataset = Dataset.from_list(processed)

    # 5. Save
    log.info("Dataset saqlanmoqda: %s", config.output_dir)
    dataset.save_to_disk(config.output_dir)
    log.info("DONE: Tayyor! (%d samples saved)", len(dataset))


if __name__ == "__main__":
    main()
