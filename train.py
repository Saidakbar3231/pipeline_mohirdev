import torch
import evaluate
from datasets import load_from_disk
from functools import partial
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
import os

# 1. Global constants
MAX_LABEL_LENGTH = 440

# 2. Global functions (must be top-level for multiprocessing)
def filter_long_labels(batch):
    return [len(label) < MAX_LABEL_LENGTH for label in batch["labels"]]

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # Split inputs and labels
        input_features = [
            {"input_features": feature["input_features"]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Remove BOS token if present
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

def main():
    os.environ['WANDB_API_KEY'] = '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'

    # Timeoutni 2 soatga (7200 sekund) uzaytirish
    os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    os.environ["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] = "7200" # 2 soat

    # Load Dataset
    model_name_or_path = os.environ.get("MODEL_NAME", "openai/whisper-large-v3")
    
    from datasets import load_from_disk, concatenate_datasets
    
    DS_DIRS = [d.strip() for d in os.environ.get("DS_DIRS", "full_mapping_dataset,full_mapping_dataset_v2").split(",") if d.strip()]

    loaded_ds = []
    for d in DS_DIRS:
        if os.path.exists(d):
            loaded_ds.append(load_from_disk(d))
        else:
            print(f"Skipping {d} as it does not exist.")

    if not loaded_ds:
        raise ValueError("No datasets loaded! Check DS_DIRS.")

    # Concatenate
    combined = concatenate_datasets(loaded_ds)

    # Remove columns
    cols_to_remove = [c.strip() for c in os.environ.get("COLS_TO_REMOVE", "").split(",") if c.strip() and c.strip() in combined.column_names]
    if cols_to_remove:
        combined = combined.remove_columns(cols_to_remove)

    # Shuffle (set seed for reproducibility)
    common_voice = combined.shuffle(seed=42)
    print(common_voice)

    NUM_PROC = int(os.environ.get("NUM_PROC", "4"))

    common_voice = common_voice.filter(
        filter_long_labels,
        batched=True,
        batch_size=1000,
        num_proc=NUM_PROC
    )
    print(common_voice)

    TEST_SPLIT = float(os.environ.get("TRAIN_TEST_SPLIT", "0.01"))
    common_voice = common_voice.train_test_split(TEST_SPLIT)
    print(common_voice)

    hf_token = os.environ.get("HF_TOKEN", None)

    # Initialize processor
    processor = WhisperProcessor.from_pretrained(
        model_name_or_path, language="uzbek", task="transcribe", token=hf_token
    )

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # Set up metrics
    metric = evaluate.load("wer")
    normalizer = BasicTextNormalizer()

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
        wer_ortho = 100 * metric.compute(predictions=pred_str, references=label_str)
        pred_str_norm = [normalizer(pred) for pred in pred_str]
        label_str_norm = [normalizer(label) for label in label_str]
        valid_indices = [i for i in range(len(label_str_norm)) if len(label_str_norm[i]) > 0]
        pred_str_norm = [pred_str_norm[i] for i in valid_indices]
        label_str_norm = [label_str_norm[i] for i in valid_indices]
        wer = 100 * metric.compute(predictions=pred_str_norm, references=label_str_norm)
        return {"wer_ortho": wer_ortho, "wer": wer}

    # Initialize model
    model = WhisperForConditionalGeneration.from_pretrained(
        model_name_or_path, attn_implementation="sdpa", token=hf_token
    )
    # Ensure model is strictly float32 to avoid mixed precision issues when fp16 is False
    model.to(torch.float32)

    model.config.use_cache = False
    model.config.dropout = 0.01
    model.config.attention_dropout = 0.1
    model.config.activation_dropout = 0.1

    model.generate = partial(
        model.generate, language="uzbek", task="transcribe", use_cache=False
    )

    # Check if GPU is available to determine mixed precision
    use_fp16 = torch.cuda.is_available()

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=os.environ.get("OUTPUT_DIR", "./whisper-large-dv_v2"),
        per_device_train_batch_size=int(os.environ.get("BATCH_SIZE", "16")),
        gradient_accumulation_steps=int(os.environ.get("GRAD_ACCUM", "1")),
        learning_rate=float(os.environ.get("LR", "1e-6")),
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=50,
        num_train_epochs=int(os.environ.get("EPOCHS", "7")),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        fp16=False,
        weight_decay=0.001,
        fp16_full_eval=False,
        eval_strategy="steps",
        per_device_eval_batch_size=int(os.environ.get("BATCH_SIZE", "16")),
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=20000,
        eval_steps=20000,
        logging_steps=25,
        report_to=["wandb"],
        load_best_model_at_end=True,
        save_total_limit=3,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
        ddp_timeout=int(os.environ.get("DDP_TIMEOUT", "7200")),
        dataloader_num_workers=int(os.environ.get("DATALOADER_WORKERS", "2")),
        dataloader_pin_memory=os.environ.get("DATALOADER_PIN_MEMORY", "True").lower() == "true",
        ddp_find_unused_parameters=os.environ.get("DDP_FIND_UNUSED_PARAMETERS", "False").lower() == "true",
    )

    from transformers import EarlyStoppingCallback

    # Initialize trainer
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=common_voice["train"],
        eval_dataset=common_voice["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    resume = os.environ.get("RESUME_CHECKPOINT", "True").lower() == "true"
    
    # Check if there is actually a valid checkpoint to resume from
    if resume and os.path.exists(training_args.output_dir):
        from transformers.trainer_utils import get_last_checkpoint
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None:
            resume = False
    elif not os.path.exists(training_args.output_dir):
        resume = False

    trainer.train(resume_from_checkpoint=resume)

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()
