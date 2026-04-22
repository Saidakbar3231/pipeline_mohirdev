import os
import sys
import json
import threading
import subprocess
import traceback
import tarfile
import urllib.request
from flask import Blueprint, request, jsonify, send_file
from datasets import Dataset, Audio

extensions_bp = Blueprint('extensions', __name__)

# STATES
aug_state = {"running": False, "log": "", "done": False, "error": ""}
map_state = {"running": False, "log": "", "done": False, "error": ""}
train_state = {"running": False, "log": "", "done": False, "error": ""}

def run_subprocess(cmd, env, state_obj):
    state_obj["running"] = True
    state_obj["done"] = False
    state_obj["log"] = f"Running command: {' '.join(cmd)}\n"
    state_obj["error"] = ""
    try:
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        for line in iter(process.stdout.readline, ''):
            state_obj["log"] += line
            
        process.wait()
        if process.returncode != 0:
            state_obj["error"] = f"Process failed with code {process.returncode}"
    except Exception as e:
        state_obj["error"] = str(e)
        state_obj["log"] += f"\nError: {traceback.format_exc()}"
    finally:
        state_obj["running"] = False
        state_obj["done"] = True

@extensions_bp.route('/api/task_status')
def api_task_status():
    return jsonify({
        "augmentation": aug_state,
        "mapping": map_state,
        "training": train_state
    })

# AUGMENTATION
@extensions_bp.route('/api/augmentation/start', methods=['POST'])
def api_augmentation_start():
    if aug_state["running"]:
        return jsonify({"error": "Augmentation is already running"}), 400
        
    data = request.json or {}
    num_proc = str(data.get("num_proc", 16))
    skip = data.get("skip", False)
    mode = data.get("mode", "full")
    aug_prob = str(data.get("aug_prob", 1.0))
    last_jsonl = data.get("last_jsonl", "")
    last_audio_dir = data.get("last_audio_dir", "")
    
    if not last_jsonl or not last_audio_dir:
        return jsonify({"error": "Missing input dataset from Section 4"}), 400
        
    def worker():
        aug_state["running"] = True
        aug_state["done"] = False
        aug_state["log"] = f"Starting Preparation (Mode: {mode})...\n"
        aug_state["error"] = ""
        try:
            # 1. Check MUSAN (only if not skipping)
            if not skip:
                if not os.path.exists("./musan") or not os.path.exists("./musan/noise"):
                    aug_state["log"] += "Downloading MUSAN dataset (this may take 10-15 minutes)...\n"
                    musan_tar = "musan.tar.gz"
                    
                    # Delete incomplete file if it exists
                    if os.path.exists(musan_tar):
                        try:
                            with tarfile.open(musan_tar) as tar:
                                pass # Check if valid
                        except Exception:
                            os.remove(musan_tar)
                            
                    if not os.path.exists(musan_tar):
                        urllib.request.urlretrieve("https://www.openslr.org/resources/17/musan.tar.gz", musan_tar)
                    aug_state["log"] += "Extracting MUSAN...\n"
                    with tarfile.open(musan_tar) as tar:
                        tar.extractall(path=".")
            
            # 2. Build HF dataset from jsonl + audio
            aug_state["log"] += "Building HF Dataset from last section's output...\n"
            records = []
            with open(last_jsonl, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip(): continue
                    item = json.loads(line)
                    fn = item.get("file_name", "")
                    fp = os.path.join(last_audio_dir, fn)
                    if os.path.exists(fp):
                        records.append({
                            "audio": os.path.abspath(fp),
                            "text": item.get("transcription", ""),
                            "duration": float(item.get("duration", 0.0))
                        })
            if not records:
                raise ValueError("No valid audio files found.")
                
            ds = Dataset.from_list(records)
            ds = ds.cast_column("audio", Audio(sampling_rate=16000, decode=False))
            
            if skip:
                output_ds_path = "birlashtirilgan_dataset_augmented"
                aug_state["log"] += f"Skipping augmentation. Saving straight to {output_ds_path}...\n"
                ds.save_to_disk(output_ds_path)
                aug_state["log"] += "\nPreparation complete (Skipped augmentation)!\n"
            else:
                if mode == "test":
                    ds = ds.select(range(min(100, len(ds))))
                    input_ds_path = "birlashtirilgan_dataset_test_in"
                    output_ds_path = "birlashtirilgan_dataset_test_out"
                elif mode == "continue":
                    if len(ds) > 100:
                        ds = ds.select(range(100, len(ds)))
                    else:
                        aug_state["log"] += "Dataset is less than 100 items, nothing to continue.\n"
                        aug_state["done"] = True
                        aug_state["running"] = False
                        return
                    input_ds_path = "birlashtirilgan_dataset_cont_in"
                    output_ds_path = "birlashtirilgan_dataset_cont_out"
                else:
                    input_ds_path = "birlashtirilgan_dataset"
                    output_ds_path = "birlashtirilgan_dataset_augmented"

                aug_state["log"] += f"Saving base dataset to {input_ds_path}...\n"
                ds.save_to_disk(input_ds_path)
                
                # 3. Run augmentation.py
                env = os.environ.copy()
                env["NUM_PROC"] = num_proc
                env["AUG_PROB"] = aug_prob
                env["INPUT_DS"] = input_ds_path
                env["OUTPUT_DS"] = output_ds_path
                env["MUSAN_PATH"] = "./musan"
                
                aug_state["log"] += "Running augmentation.py...\n"
                cmd = [sys.executable, "augmentation.py"]
                
                process = subprocess.Popen(
                    cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
                )
                for line in process.stdout:
                    aug_state["log"] += line
                process.wait()
                if process.returncode != 0:
                    aug_state["error"] = f"Augmentation failed with code {process.returncode}"
                else:
                    if mode == "continue":
                        aug_state["log"] += "\nConcatenating test and continue datasets...\n"
                        try:
                            from datasets import load_from_disk, concatenate_datasets
                            ds_test = load_from_disk("birlashtirilgan_dataset_test_out")
                            ds_cont = load_from_disk("birlashtirilgan_dataset_cont_out")
                            ds_final = concatenate_datasets([ds_test, ds_cont])
                            ds_final.save_to_disk("birlashtirilgan_dataset_augmented")
                            aug_state["log"] += "Concatenation complete! Final dataset is at birlashtirilgan_dataset_augmented\n"
                        except Exception as e:
                            aug_state["error"] = f"Concatenation error: {e}"
                            
                    elif mode == "test":
                        aug_state["log"] += "\nTest completed! Saved to birlashtirilgan_dataset_test_out. You can review and then 'Continue'.\n"
                    else:
                        aug_state["log"] += "\nFull Augmentation complete!\n"
                
        except Exception as e:
            aug_state["error"] = str(e)
            aug_state["log"] += f"\nError: {traceback.format_exc()}"
        finally:
            aug_state["running"] = False
            aug_state["done"] = True

    threading.Thread(target=worker, daemon=True).start()
    return jsonify({"status": "started"})

# MAPPING
@extensions_bp.route('/api/mapping/start', methods=['POST'])
def api_mapping_start():
    if map_state["running"]:
        return jsonify({"error": "Mapping is already running"}), 400
        
    data = request.json or {}
    env = os.environ.copy()
    env["MODEL_NAME"] = data.get("model_name", "openai/whisper-large-v3")
    env["OUTPUT_DIR"] = data.get("output_dir", "full_mapping_dataset_v2")
    env["HF_TOKEN"] = data.get("hf_token", "")
    
    ds_names = data.get("ds_names", "")
    names_list = [x.strip() for x in ds_names.split(",") if x.strip()]
    
    # Automatically include local augmented dataset if it exists
    if os.path.exists("birlashtirilgan_dataset_augmented"):
        names_list.append("birlashtirilgan_dataset_augmented")
        
    env["DS_NAMES"] = ",".join(names_list)
    env["NUM_PROC"] = str(data.get("num_proc", 20))
    
    cmd = [sys.executable, "mapping.py"]
    threading.Thread(target=run_subprocess, args=(cmd, env, map_state), daemon=True).start()
    return jsonify({"status": "started"})

# TRAINING
@extensions_bp.route('/api/training/start', methods=['POST'])
def api_training_start():
    if train_state["running"]:
        return jsonify({"error": "Training is already running"}), 400
        
    data = request.json or {}
    env = os.environ.copy()
    if data.get("wandb_api_key"):
        env["WANDB_API_KEY"] = data.get("wandb_api_key")
    if data.get("hf_token"):
        env["HF_TOKEN"] = data.get("hf_token")
    env["MODEL_NAME"] = data.get("model_name_or_path", "openai/whisper-large-v3")
    env["DS_DIRS"] = data.get("ds_dirs", "full_mapping_dataset,full_mapping_dataset_v2")
    env["COLS_TO_REMOVE"] = data.get("columns_to_remove", "")
    env["NUM_PROC"] = str(data.get("num_proc", 4))
    env["TRAIN_TEST_SPLIT"] = str(data.get("train_test_split", 0.01))
    env["OUTPUT_DIR"] = data.get("output_dir", "./whisper-large-dv_v2")
    env["BATCH_SIZE"] = str(data.get("per_device_train_batch_size", 16))
    env["GRAD_ACCUM"] = str(data.get("gradient_accumulation_steps", 1))
    env["LR"] = str(data.get("learning_rate", 1e-6))
    env["EPOCHS"] = str(data.get("num_train_epochs", 7))
    env["DDP_TIMEOUT"] = str(data.get("ddp_timeout", 7200))
    env["DATALOADER_WORKERS"] = str(data.get("dataloader_num_workers", 2))
    env["DATALOADER_PIN_MEMORY"] = str(data.get("dataloader_pin_memory", True))
    env["DDP_FIND_UNUSED_PARAMETERS"] = str(data.get("ddp_find_unused_parameters", False))
    env["RESUME_CHECKPOINT"] = str(data.get("resume_from_checkpoint", True))
    
    cmd = [sys.executable, "train.py"]
    threading.Thread(target=run_subprocess, args=(cmd, env, train_state), daemon=True).start()
    return jsonify({"status": "started"})

# AUDIO ENDPOINT FOR HF DATASETS (Augmented review)
@extensions_bp.route('/api/hf_dataset_audio')
def api_hf_dataset_audio():
    dataset_path = request.args.get("dataset_path", "birlashtirilgan_dataset_augmented")
    idx = int(request.args.get("idx", 0))
    try:
        from datasets import load_from_disk
        import soundfile as sf
        ds = load_from_disk(dataset_path)
        if idx < 0 or idx >= len(ds):
            return "Index out of bounds", 404
            
        item = ds[idx]
        
        # Audio might be a dict with bytes, array, or a string path
        audio_data = item.get("audio")
        if isinstance(audio_data, dict):
            if audio_data.get("bytes"):
                import time
                tmp_dir = "uploads_tmp"
                os.makedirs(tmp_dir, exist_ok=True)
                unique_id = int(time.time() * 1000)
                tmp_path = os.path.join(tmp_dir, f"hf_audio_{idx}_{unique_id}.wav")
                with open(tmp_path, "wb") as f:
                    f.write(audio_data["bytes"])
                return send_file(os.path.abspath(tmp_path), mimetype="audio/wav", max_age=0)
            elif "array" in audio_data and audio_data["array"] is not None:
                import time
                audio_array = audio_data["array"]
                sr = audio_data.get("sampling_rate", 16000)
                tmp_dir = "uploads_tmp"
                os.makedirs(tmp_dir, exist_ok=True)
                unique_id = int(time.time() * 1000)
                tmp_path = os.path.join(tmp_dir, f"hf_audio_{idx}_{unique_id}.wav")
                sf.write(tmp_path, audio_array, sr)
                return send_file(os.path.abspath(tmp_path), mimetype="audio/wav", max_age=0)
            else:
                # It's a path
                audio_path = audio_data.get("path")
                if audio_path and os.path.exists(audio_path):
                    return send_file(os.path.abspath(audio_path), mimetype="audio/wav")
                else:
                    return "Audio file not found or invalid format", 404
        else:
            # It's a path string
            audio_path = str(audio_data)
            if audio_path and os.path.exists(audio_path):
                return send_file(os.path.abspath(audio_path), mimetype="audio/wav")
            else:
                return "Audio file not found", 404
    except Exception as e:
        return str(e), 500

@extensions_bp.route('/api/hf_dataset_review')
def api_hf_dataset_review():
    dataset_path = request.args.get("dataset_path", "birlashtirilgan_dataset_augmented")
    idx = int(request.args.get("idx", 0))
    try:
        from datasets import load_from_disk
        ds = load_from_disk(dataset_path)
        if len(ds) == 0:
            return jsonify({"error": "Dataset is empty"}), 404
            
        idx = max(0, min(idx, len(ds) - 1))
        item = ds[idx]
        
        return jsonify({
            "idx": idx,
            "total": len(ds),
            "text": item.get("text", ""),
            "duration": item.get("duration", 0),
            "audio_url": f"/api/hf_dataset_audio?dataset_path={dataset_path}&idx={idx}"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
