# MohirDev Pipeline — CLAUDE.md

## Project Purpose

MohirDev Pipeline is a Flask web tool for building Uzbek speech datasets. It ingests audio from YouTube, HuggingFace datasets, local folders, or pre-diarized JSON files; performs VAD chunking and optional noise reduction; transcribes via either the Aisha/MohirDev async STT API (Path 1) or Gemini 2.5 Flash (Path 2), or skips transcription entirely for JSON sources with existing text (Path 3); applies audio + text filters and text normalization; exports to JSONL/CSV; then optionally augments with MUSAN noise, maps with Whisper, fine-tunes Whisper, and pushes the final dataset to HuggingFace Hub.

---

## Architecture Overview — Data Flow

```
Sources
  YouTube URL         → downloader.download_youtube()       → audio items []
  HuggingFace dataset → downloader.download_from_huggingface()
  Local folder        → app._run_pipeline() inline scan
  JSON + diarization  → json_processor.extract_chunks_from_json()

audio items [{file, file_name, source, source_url}]
        ↓
  audio_utils.vad_chunk()          VAD split, noise reduce (pydub + noisereduce)
        ↓
  segments [{file, file_name, duration, ...}]
        ↓
  ┌──────────────────────────┬─────────────────────────────┬──────────────────────────┐
  │  Yo'l 1 (Path 1)         │  Yo'l 2 (Path 2)            │  Yo'l 3 (Path 3)         │
  │  stt_v1.process_segment  │  stt_v2.process_segment_v2  │  json_processor.         │
  │  ├─ filter_audio()       │  ├─ filter_audio()           │  process_json_python_    │
  │  ├─ Aisha API async poll │  ├─ gemini_utils.            │  filter()                │
  │  ├─ filter_text_v1()     │  │   transcribe_audio_gemini │  (no STT, no Gemini)     │
  │  └─ polish_text_local()  │  ├─ filter_text_v2()         │                          │
  │     (num→word, dedup,    │  └─ gemini_utils.            │                          │
  │      fillers removed)    │      score_transcription()   │                          │
  └──────────────────────────┴─────────────────────────────┴──────────────────────────┘
        ↓
  app._apply_norm_filters()     apostrophe, num→word, dedup, brackets, HTML, cyrillic…
        ↓
  exporter.append_jsonl() / _save_csv_filtered()
  outputs/YYYY-MM-DD/metadata/metadata_{v1|v2}_HHMMSS.{jsonl,csv}
  outputs/YYYY-MM-DD/audios/audio_HHMMSS/*.wav
  + .audiodir sidecar (pairs metadata file → audio dir for HF push)
        ↓
  [Section 5] pipeline_extensions.api_augmentation_start()
    worker:  HF Dataset → save_to_disk(birlashtirilgan_dataset)
             → subprocess: augmentation.py (MUSAN + audiomentations)
             → birlashtirilgan_dataset_augmented
        ↓
  [Section 6] pipeline_extensions.api_validation_start()
    → subprocess: validate_augmented.py (V1/V3/V4/V5/V7 checks)
        ↓
  [Section 7] pipeline_extensions.api_mapping_start()
    → subprocess: mapping.py  (Whisper forced-align)
        ↓
  [Section 8] pipeline_extensions.api_training_start()
    → subprocess: train.py  (Whisper fine-tune)
        ↓
  [Section 9] hf_pusher.push_jsonl() / push_csv()
    → HuggingFace Hub  (datasets.push_to_hub)
```

---

## Key Commands

```bash
# Run dev server (port 7861, kills existing listener on Windows)
python app.py

# Docker
docker-compose up --build

# Run augmentation standalone (env vars required)
INPUT_DS=birlashtirilgan_dataset OUTPUT_DS=birlashtirilgan_dataset_augmented \
  NUM_PROC=16 AUG_PROB=1.0 MUSAN_PATH=./musan python augmentation.py

# Run validation standalone
python validate_augmented.py \
  --output-ds birlashtirilgan_dataset_augmented \
  --input-audio-dir ./outputs/YYYY-MM-DD/audios/audio_HHMMSS \
  --musan-path ./musan \
  --auto-filter-failed

# Run mapping / training (triggered via UI, but can call directly with env vars)
MODEL_NAME=openai/whisper-large-v3 OUTPUT_DIR=full_mapping_dataset_v2 \
  DS_NAMES=birlashtirilgan_dataset_augmented HF_TOKEN=hf_... python mapping.py

python train.py   # reads DS_DIRS, OUTPUT_DIR, EPOCHS, LR, BATCH_SIZE, etc.
```

---

## Module Responsibilities

| Module | Responsibility |
|---|---|
| `app.py` | Flask server, `_run_pipeline()` main orchestration in background thread, global `_state` dict, pipeline stop/resume, `_apply_norm_filters()`, all `/api/*` routes |
| `pipeline_extensions.py` | Flask Blueprint for sections 5–8: augmentation (test/continue/full/skip modes), mapping, training, validation; each runs as a subprocess with streaming stdout; also houses MUSAN integrity check |
| `config.py` | All thresholds (V1/V2 duration, SNR, silence, word count), API URLs, directory names |
| `downloader.py` | YouTube via yt-dlp, HuggingFace streaming (handles config/split/audio-column auto-detect, retry), JSON URL download; all yield `{file, file_name, source, source_url}` items |
| `json_processor.py` | Three JSON processing paths: `process_json_python` (use existing text), `process_json_gemini` (retranscribe with Gemini), `process_json_python_filter` (Path 3 / Yo'l 3 with style detection) |
| `exporter.py` | `append_jsonl`, `save_all` (JSONL+CSV), `.audiodir` sidecar read/write, `save_report` |
| `hf_pusher.py` | `push_jsonl` / `push_csv` → HuggingFace Hub via `datasets.push_to_hub` |
| `gemini_utils.py` | Gemini client singleton, `transcribe_audio_gemini` (dynamic prompt from filter toggles), `score_transcription` (1–5), `determine_status_v2`, `polish_text` (Yo'l 1 Gemini variant, rarely used) |
| `audio_utils.py` | `vad_chunk` (pydub split_on_silence → min/max duration enforcement), `reduce_noise` (noisereduce), `to_mono_16k` |
| `stt_v1.py` | Path 1: Aisha async STT (submit+poll), `polish_text_local` (filler removal, elongated vowel stripping, stutter dedup, num→Uzbek word with placeholder protection, Jaccard+LCS+Run dedup-half), `_is_valid_transcription`, `_compute_local_change_ratio` (synonym-aware) |
| `stt_v2.py` | Path 2: thin wrapper calling filter_audio → gemini_utils.transcribe → filter_text_v2 → score → determine_status |
| `filter_audio.py` | `filter_audio` (SNR+silence+duration+language gate), `detect_music` (HPSS+RMS-CV+spectral contrast, majority vote on 3 chunks), `detect_multiple_speakers` (MFCC+log-F0+DBSCAN, temporal coherence), `filter_text_v1/v2`, `compute_snr/silence_ratio/repeat_ratio` |
| `filter_text.py` | Re-exports everything from `filter_audio.py`; exists only as a semantic alias |
| `augmentation.py` | MUSAN quarantine filter, `to_mono_1d`, process-local pipeline cache, `apply_augment` (TimeStretch, PitchShift, OneOf[MUSAN noise sources], LowPassFilter, Gain); `aug_status` in {augmented, skip_prob, skip_long, failed, invalid_input} |
| `validate_augmented.py` | Standalone validator: V1 waveform-diff check, V3 status distribution, V4 schema, V5 full decode, V7 MUSAN preflight; emits `RESULT_JSON:` line for machine parsing; exit codes 0/2/3 |
| `tab_review.py` | Manual review tab routes (registered into main app) |
| `mapping.py` | Whisper forced-align for HF datasets (read via env vars) |
| `train.py` | Whisper fine-tuning (read via env vars) |

---

## Important Conventions

**Naming**
- Segment result dicts always carry: `file_name`, `transcription`, `status`, `pipeline`, `snr_score`, `silence_ratio`, `source`, `source_url`, `duration`
- Status values: `"approved"`, `"pending"`, `"rejected"`, `"filtered"` — `"filtered"` is the only one excluded from output JSONL/CSV
- Audio item dicts: `{file, file_name, source, source_url}` — `file` is absolute path
- Output dirs: `outputs/YYYY-MM-DD/metadata/` and `outputs/YYYY-MM-DD/audios/audio_HHMMSS/`
- `.audiodir` sidecar files: every metadata file gets a same-stem `.audiodir` containing the absolute path of its audio folder

**Path security**
- Every file path from user input goes through `_safe_under(path, *allowed_roots)` before use
- Uses `os.path.commonpath` (not `startswith`) to prevent sibling-prefix bypass
- `api_delete` is restricted to `OUTPUT_DIR` only

**Pipeline state**
- `_state` global dict in `app.py` is the single source of truth for the running pipeline; mutated only inside `thread_fn`
- `_resume` dict stores checkpoint when Stop is requested; next Start checks `_resume.get("remaining_segments")`
- `_stop_requested` bool is checked per-segment in the main loop

**Filters vs normalization**
- Audio filters run before STT; text filters run on raw STT output; norm filters (`_apply_norm_filters`) run on polished output in `app.py` — they can both clean and drop (return `filtered=True`)
- Music/multi-speaker detection results are metadata flags only (`has_background_music`, `has_multiple_speakers`); they do NOT by themselves drop segments

**Augmentation modes**
- `test`: processes first 100 samples, writes manifest `.aug_test_manifest.json`, saves to `birlashtirilgan_dataset_test_out`
- `continue`: loads manifest, filters out already-processed rows (`load_from_cache_file=False` required), saves to `birlashtirilgan_dataset_cont_out`, then concatenates with test output → `birlashtirilgan_dataset_augmented`
- `full`: processes everything → `birlashtirilgan_dataset_augmented`
- `skip`: no augmentation, just copies data as HF dataset with `aug_status=skip_prob`

**Subprocess pattern**
- Augmentation, mapping, and training run as `subprocess.Popen` with `env` overrides; stdout is streamed into `*_state["log"]`
- `pipeline_extensions.run_subprocess()` is the generic helper; validation has its own inline version that parses `RESULT_JSON:` line

**Number conversion**
- `_num_words_uz` in `stt_v1.py` is the canonical implementation (uses placeholder protection)
- `_num_words_uz` in `app.py` is a simpler duplicate used only inside `_apply_norm_filters`
- Phone numbers, decimals, dates, times, years (1500–2049) are protected from conversion

---

## Known TODOs / Incomplete Parts

1. **Dead function definition** — `downloader.py:216-228`: there is a first `download_from_huggingface` with only a docstring body; the real implementation starts at line 229 and shadows it. The first definition is unreachable dead code.

2. **Debug print statements left in production** — never removed:
   - `pipeline_extensions.py:276` — `print(f"[DEBUG] augmentation worker start mode={mode!r}...")`
   - `pipeline_extensions.py:360` — `print("[DEBUG] ENTER TEST", flush=True)`
   - `pipeline_extensions.py:417` — `print("[DEBUG] ENTER CONTINUE", flush=True)`
   - `pipeline_extensions.py:465-469` — multi-line `print(f"[DEBUG] used_files defined=...")`
   - `augmentation.py:229` — `print("[DEBUG] Successfully ran pipeline augmentation for file:", audio_path)`

3. **`validate_augmented.py` V1 check** requires `--input-audio-dir` pointing to the pre-augmentation audio; it is silently skipped when not provided. The UI does not auto-populate this path.

4. **`mapping.py` and `train.py`** are invoked via env vars only; their internal implementation is not reviewed here (not in the read list).

5. **`tab_review.py`** manual review routes are not in the read list; their implementation is unknown.

6. **Single-user assumption** — `_state` is a module-level global; concurrent pipeline runs from two browser sessions will corrupt each other's state.

---

## Environment Variables

**Required for operation (copy `.env.example` → `.env`):**

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | `""` | Google AI Studio key — required for Yo'l 2 (Gemini STT) |
| `STT_API_KEY` | `""` | MohirDev/Aisha STT key — required for Yo'l 1; can also be entered in UI per-run |
| `ACCESS_PASSWORD` | `mohirdev2026` | Web UI login password |
| `APP_SECRET_KEY` | `aisha-pipeline-secret-2026` | Flask session signing key — change in production |

**Optional / set by pipeline at runtime:**

| Variable | Default | Purpose |
|---|---|---|
| `STT_API_URL` | `https://back.aisha.group/api/v2/stt/post/` | Aisha STT POST endpoint |
| `GEMINI_MODEL_STT` | `gemini-2.5-flash` | Gemini model for STT (Yo'l 2) |
| `GEMINI_MODEL_POLISH` | `gemini-2.5-flash` | Gemini model for scoring |
| `NUM_PROC` | `16` | CPU workers for `datasets.map` in augmentation |
| `AUG_PROB` | `1.0` | Probability each sample is augmented (0.0–1.0) |
| `INPUT_DS` | `birlashtirilgan_dataset` | HF dataset input path for augmentation.py |
| `OUTPUT_DS` | `birlashtirilgan_dataset_augmented` | HF dataset output path for augmentation.py |
| `MUSAN_PATH` | `./musan` | Root of the MUSAN noise corpus (~8 GB, auto-downloaded if absent) |
| `INPUT_AUDIO_DIR` | `""` | Pre-augmentation audio dir for validate_augmented.py V1 check |
| `HF_TOKEN` | `""` | HuggingFace token for private datasets / push |
| `MODEL_NAME` | `openai/whisper-large-v3` | Whisper model for mapping/training |
| `OUTPUT_DIR` | varies | Output dir for mapping and training |
| `DS_NAMES` | `""` | Comma-separated dataset names/paths for mapping |
| `DS_DIRS` | `""` | Comma-separated dataset dirs for training |
| `WANDB_API_KEY` | `""` | Weights & Biases logging for training |
| `EPOCHS`, `LR`, `BATCH_SIZE`, `GRAD_ACCUM` | see train.py | Training hyperparameters |
