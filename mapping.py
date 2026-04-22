import os
from typing import Optional
from dataclasses import dataclass

import torch
from datasets import load_dataset, load_from_disk, concatenate_datasets, Audio, Features, Value, Dataset
from transformers import (
    WhisperFeatureExtractor,
    WhisperProcessor,
    WhisperTokenizer
)


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
NUM_PROC = int(os.environ.get("NUM_PROC", "20"))


class WhisperDatasetPreprocessor:
    def __init__(self, config: Config):
        self.config = config
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(config.model_name)
        self.tokenizer = WhisperTokenizer.from_pretrained(
            config.model_name,
            language=config.language,
            task=config.task
        )
        self.processor = WhisperProcessor.from_pretrained(
            config.model_name,
            language=config.language,
            task=config.task
        )

    def prepare_features(self, batch: dict) -> dict:
        import io
        import soundfile as sf
        import numpy as np

        audio_data = batch["audio"]
        
        # Manually decode to bypass torchcodec issue
        if isinstance(audio_data, dict) and audio_data.get("bytes"):
            wav, sr = sf.read(io.BytesIO(audio_data["bytes"]))
        elif isinstance(audio_data, dict) and audio_data.get("array") is not None:
            wav = audio_data["array"]
            sr = audio_data.get("sampling_rate", 16000)
        else:
            import librosa
            path = audio_data if isinstance(audio_data, str) else audio_data.get("path")
            wav, sr = librosa.load(path, sr=16000)

        batch["input_features"] = self.feature_extractor(
            wav,
            sampling_rate=sr
        ).input_features[0]
        batch["labels"] = self.tokenizer(batch["text"]).input_ids
        return batch


def load_and_concat(config: Config) -> Dataset:
    all_ds = []
    for name in DS_NAMES:
        print(f"Yuklanmoqda: {name}")
        if os.path.exists(name):
            ds = load_from_disk(name)
        else:
            ds = load_dataset(name, token=config.hf_token if config.hf_token else None, split="train")
            
        if "transcription" in ds.column_names:
            ds = ds.rename_column("transcription", "text")
            
        # Ortiqcha ustunlarni olib tashlash (masalan, duration)
        cols_to_remove = [c for c in ds.column_names if c not in ["audio", "text"]]
        if cols_to_remove:
            ds = ds.remove_columns(cols_to_remove)
            
        # Audio faqat 16000 bo'lishi kerak
        from datasets import Audio
        ds = ds.cast_column("audio", Audio(sampling_rate=config.sampling_rate, decode=False))
        all_ds.append(ds)

    print("Datasetlar birlashtirilmoqda...")
    if not all_ds:
        raise ValueError("No datasets found to map!")
    return concatenate_datasets(all_ds)


def main():
    config = Config()
    
    # 1. Load
    dataset = load_and_concat(config)
    
    print("#" * 50)

    # 2. Preprocessor
    print("Preprocessor yaratilmoqda...")
    preprocessor = WhisperDatasetPreprocessor(config)

    # 3. Map features
    print("Dataset tayyorlanmoqda...")
    dataset = dataset.map(
        preprocessor.prepare_features,
        num_proc=NUM_PROC
    )

    # 4. Save
    print(f"Dataset saqlanmoqda: {config.output_dir}")
    dataset.save_to_disk(config.output_dir)
    print("DONE: Tayyor!")


if __name__ == "__main__":
    main()
