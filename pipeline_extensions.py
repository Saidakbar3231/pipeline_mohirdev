import os
import sys
import json
import threading
import subprocess
import traceback
import tarfile
import functools
import urllib.request
from flask import Blueprint, request, jsonify, send_file
from datasets import Dataset, Audio


def _safe_under(target_path, *allowed_roots):
    """commonpath-based sandbox check (see app._safe_under). Used to reject
    dataset-supplied audio paths that escape the project tree."""
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


# Module-level callables for HF Dataset.filter / .map. Defining these at
# module scope (instead of as nested closures inside worker()) keeps them
# trivially picklable when datasets dispatches across worker processes.
def _safe_file_name_filter(used_files, x):
    """Keep rows whose file_name is a non-empty string AND not in used_files."""
    if not isinstance(x, dict):
        return False
    name = x.get("file_name")
    if not isinstance(name, str) or not name:
        return False
    return name not in used_files


def _make_safe_filter(used_files):
    """
    Closure factory for Dataset.filter.

    Even `frozenset` serializes to bytes whose ordering depends on insertion
    order / hash seed, which can destabilize HF Datasets' fingerprint and lead
    to stale-cache hits. A `tuple(sorted(...))` is byte-for-byte deterministic
    across runs, processes, and Python versions.
    """
    used_files = tuple(sorted(used_files))
    # O(1) membership lookup, derived from the already-sorted tuple so the
    # canonical deterministic structure (the tuple) remains the source of truth.
    used_lookup = set(used_files)

    def _f(x):
        # Defensive: HF Datasets normally yields dict rows, but a malformed
        # row or an unexpected wrapper type must not raise — keep it.
        if not isinstance(x, dict):
            return True
        name = x.get("file_name")
        # Preserve rows with invalid/missing names — only the manifest decides drops.
        if not isinstance(name, str) or not name.strip():
            return True
        return name not in used_lookup

    return _f


def _mark_skip(x):
    """Tag a row with aug_status=skip_prob; preserves all original keys."""
    if not isinstance(x, dict):
        return {"aug_status": "skip_prob"}
    out = dict(x)
    out["aug_status"] = "skip_prob"
    return out

extensions_bp = Blueprint('extensions', __name__)

# STATES
aug_state = {"running": False, "log": "", "done": False, "error": ""}
map_state = {"running": False, "log": "", "done": False, "error": ""}
train_state = {"running": False, "log": "", "done": False, "error": ""}
validation_state = {
    "running": False,
    "done": False,
    "log": "",
    "error": "",
    "status": None,
    "result": None,
}

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
        "training": train_state,
        "validation": validation_state,
    })


# VALIDATION
@extensions_bp.route('/api/validation/start', methods=['POST'])
def api_validation_start():
    if validation_state["running"]:
        return jsonify({"error": "Validation is already running"}), 400

    data = request.json or {}
    auto_filter_failed = bool(data.get("auto_filter_failed", False))
    strict_augmented_only = bool(data.get("strict_augmented_only", False))

    def worker():
        validation_state["running"] = True
        validation_state["done"] = False
        validation_state["log"] = ""
        validation_state["error"] = ""
        validation_state["status"] = None
        validation_state["result"] = None

        cmd = [sys.executable, "validate_augmented.py"]
        if auto_filter_failed:
            cmd.append("--auto-filter-failed")
        if strict_augmented_only:
            cmd.append("--strict-augmented-only")

        validation_state["log"] = f"Running: {' '.join(cmd)}\n"

        full_stdout_lines = []
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            for line in iter(process.stdout.readline, ''):
                full_stdout_lines.append(line)
                validation_state["log"] += line
            process.wait()

            # Extract the RESULT_JSON: line emitted by validate_augmented.py.
            for line in reversed(full_stdout_lines):
                stripped = line.strip()
                if stripped.startswith("RESULT_JSON:"):
                    try:
                        payload = json.loads(stripped[len("RESULT_JSON:"):].strip())
                        validation_state["result"] = payload
                        validation_state["status"] = payload.get("status")
                    except Exception as parse_err:
                        validation_state["error"] = f"RESULT_JSON parse error: {parse_err}"
                    break

            if process.returncode not in (0, 2, 3):
                validation_state["error"] = (
                    validation_state["error"]
                    or f"Validator exited with code {process.returncode}"
                )
        except Exception as e:
            validation_state["error"] = str(e)
            validation_state["log"] += f"\nError: {traceback.format_exc()}"
        finally:
            validation_state["running"] = False
            validation_state["done"] = True

    threading.Thread(target=worker, daemon=True).start()
    return jsonify({"status": "started"})

# AUGMENTATION

# Subdirectories of MUSAN that augmentation.py's get_pipeline() consumes.
# Order matches the four AddBackgroundNoise stanzas in OneOf(...).
REQUIRED_MUSAN_SUBDIRS = (
    "noise/free-sound",
    "noise/sound-bible",
    "speech/librivox",
    "music/jamendo",
)


def check_musan_integrity(musan_path, min_files=50):
    """
    Preflight check for MUSAN: verify each required subdirectory exists
    and contains at least `min_files` .wav files. Files inside a
    `_quarantine/` subtree (populated by augmentation.filter_musan_files)
    are excluded, since AddBackgroundNoise will not see them.

    Returns:
        (ok, details) — details is {subdir: int_count} for every required
        subdir. ok is True only when ALL subdirs meet the threshold.
    """
    details = {}
    ok = True
    for sub in REQUIRED_MUSAN_SUBDIRS:
        full = os.path.join(musan_path, sub)
        count = 0
        if os.path.isdir(full):
            for root, _, files in os.walk(full):
                # Quarantined files are invisible to AddBackgroundNoise.
                if "_quarantine" in root.replace("\\", "/").split("/"):
                    continue
                count += sum(1 for f in files if f.endswith(".wav"))
        details[sub] = count
        if count < min_files:
            ok = False
    return ok, details


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

    # Noise loudness (0-100%). Non-numeric → 400; out-of-range → clamped to
    # [0, 100] (augmentation.py also clamps defensively). Default 50 keeps
    # existing callers that send no noise_level working unchanged.
    try:
        noise_level = float(data.get("noise_level", 50.0))
    except (TypeError, ValueError):
        return jsonify({"error": "noise_level raqam bo'lishi kerak (0-100)"}), 400
    noise_level = max(0.0, min(100.0, noise_level))
    
    if not last_jsonl or not last_audio_dir:
        return jsonify({"error": "Missing input dataset from Section 4"}), 400

    abs_jsonl, ok_jsonl = _safe_under(last_jsonl, ".")
    abs_audio_dir, ok_audio = _safe_under(last_audio_dir, ".")
    if not ok_jsonl or not ok_audio:
        return jsonify({"error": "Ruxsat yo'q (last_jsonl/last_audio_dir cwd dan tashqarida)"}), 403
    last_jsonl = abs_jsonl
    last_audio_dir = abs_audio_dir

    def worker():
        import time as _time

        aug_state["running"] = True
        aug_state["done"] = False
        aug_state["log"] = f"Starting Preparation (Mode: {mode})...\n"
        aug_state["error"] = ""
        try:
            # Pre-flight: warn if project lives inside OneDrive.
            cwd_abs = os.path.abspath(".")
            if "onedrive" in cwd_abs.lower():
                aug_state["log"] += (
                    "[WARNING] Project is inside OneDrive — this may cause issues with large files "
                    "(file locking, sync corruption, EOFError on musan.tar.gz). "
                    "Consider moving to C:\\Projects\\ or similar.\n"
                )

            # 1. Check MUSAN (only if not skipping)
            if not skip:
                musan_ready = (
                    os.path.isdir("./musan/music")
                    and os.path.isdir("./musan/noise")
                    and os.path.isdir("./musan/speech")
                )
                if musan_ready:
                    aug_state["log"] += "[INFO] MUSAN directory already exists with expected subdirs — skipping download.\n"
                else:
                    musan_tar = "musan.tar.gz"
                    musan_url = "https://www.openslr.org/resources/17/musan.tar.gz"

                    # Report current tar size so the user can see if it is truncated.
                    if os.path.exists(musan_tar):
                        existing_mb = os.path.getsize(musan_tar) / (1024 * 1024)
                        aug_state["log"] += (
                            f"[INFO] Found existing {musan_tar} — size: {existing_mb:.1f} MB\n"
                        )

                    # Validate any existing tar — delete if corrupt/incomplete.
                    if os.path.exists(musan_tar):
                        aug_state["log"] += f"[INFO] Checking {musan_tar} integrity...\n"
                        try:
                            with tarfile.open(musan_tar, "r:gz") as _t:
                                _t.getmembers()   # forces full index read; raises on truncation
                            aug_state["log"] += "[INFO] Existing tar is valid.\n"
                        except Exception as tar_err:
                            aug_state["log"] += (
                                f"[WARN] {musan_tar} is corrupt ({tar_err}).\n"
                            )
                            # Check whether the file is still growing (OneDrive mid-sync).
                            size_before = os.path.getsize(musan_tar)
                            aug_state["log"] += "[INFO] Waiting 2s to check if file is still being written...\n"
                            _time.sleep(2)
                            size_after = os.path.getsize(musan_tar)
                            if size_after != size_before:
                                aug_state["error"] = (
                                    f"musan.tar.gz is still being written "
                                    f"({size_before/(1024*1024):.1f} MB → {size_after/(1024*1024):.1f} MB). "
                                    "Wait for OneDrive sync to finish, then retry."
                                )
                                aug_state["log"] += f"[ERROR] {aug_state['error']}\n"
                                return
                            # File is not growing — try to delete it.
                            aug_state["log"] += "Deleting corrupt tar and re-downloading...\n"
                            try:
                                os.remove(musan_tar)
                            except PermissionError as rm_err:
                                aug_state["error"] = (
                                    "Cannot delete musan.tar.gz — file is locked (OneDrive sync?). "
                                    "Please pause OneDrive sync or move the project outside OneDrive."
                                )
                                aug_state["log"] += f"[ERROR] {aug_state['error']}\n"
                                raise RuntimeError(aug_state["error"]) from rm_err
                            except OSError as rm_err:
                                aug_state["error"] = f"Cannot delete musan.tar.gz: {rm_err}"
                                aug_state["log"] += f"[ERROR] {aug_state['error']}\n"
                                return

                    if not os.path.exists(musan_tar):
                        aug_state["log"] += (
                            f"Downloading MUSAN dataset from {musan_url}\n"
                            "(this may take 10-15 minutes for an 8 GB file)...\n"
                        )
                        try:
                            import requests as _req
                            CHUNK = 8 * 1024 * 1024   # 8 MB
                            downloaded = 0
                            last_logged_mb = 0
                            with _req.get(musan_url, stream=True, timeout=60) as resp:
                                resp.raise_for_status()
                                total = int(resp.headers.get("content-length", 0))
                                total_mb = total / (1024 * 1024) if total else 0
                                with open(musan_tar, "wb") as fout:
                                    for chunk in resp.iter_content(chunk_size=CHUNK):
                                        if not chunk:
                                            continue
                                        fout.write(chunk)
                                        downloaded += len(chunk)
                                        done_mb = downloaded // (1024 * 1024)
                                        if done_mb >= last_logged_mb + 100:
                                            last_logged_mb = done_mb
                                            pct = f" ({done_mb/total_mb*100:.0f}%)" if total_mb else ""
                                            aug_state["log"] += (
                                                f"  Downloaded {done_mb} MB{pct}\n"
                                            )
                            aug_state["log"] += f"  Download complete ({downloaded // (1024*1024)} MB).\n"
                        except Exception as dl_err:
                            aug_state["error"] = f"MUSAN download failed: {dl_err}"
                            aug_state["log"] += f"[ERROR] {aug_state['error']}\n"
                            return

                    # Verify tar integrity before extraction.
                    aug_state["log"] += "[INFO] Verifying tar integrity before extraction...\n"
                    try:
                        with tarfile.open(musan_tar, "r:gz") as _t:
                            _t.getmembers()
                    except Exception as verify_err:
                        aug_state["error"] = (
                            f"Downloaded {musan_tar} is still corrupt: {verify_err}. "
                            "Try deleting musan.tar.gz and re-running."
                        )
                        aug_state["log"] += f"[ERROR] {aug_state['error']}\n"
                        return

                    aug_state["log"] += "Extracting MUSAN...\n"
                    try:
                        with tarfile.open(musan_tar, "r:gz") as tar:
                            tar.extractall(path=".")
                    except Exception as ex_err:
                        aug_state["error"] = f"Extraction failed: {ex_err}"
                        aug_state["log"] += f"[ERROR] {aug_state['error']}\n"
                        return
                    aug_state["log"] += "[INFO] MUSAN extraction complete.\n"
            
            # 2. Build HF dataset from jsonl + audio
            aug_state["log"] += "Building HF Dataset from last section's output...\n"
            records = []
            skipped_no_name = 0
            with open(last_jsonl, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip(): continue
                    item = json.loads(line)
                    fn = item.get("file_name", "") or ""
                    fp = os.path.join(last_audio_dir, fn) if fn else ""
                    # Guarantee a non-empty file_name on every record. If both
                    # are missing we cannot identify the row downstream — drop it.
                    if not fn and not fp:
                        skipped_no_name += 1
                        continue
                    safe_name = fn if fn else os.path.basename(fp)
                    if not isinstance(safe_name, str) or not safe_name.strip():
                        skipped_no_name += 1
                        continue
                    if os.path.exists(fp):
                        records.append({
                            "audio": os.path.abspath(fp),
                            "text": item.get("transcription", ""),
                            "duration": float(item.get("duration", 0.0)),
                            "file_name": safe_name,
                        })
            if skipped_no_name:
                aug_state["log"] += (
                    f"[WARN] Skipped {skipped_no_name} record(s) with missing file_name\n"
                )
            if not records:
                raise ValueError("No valid audio files found.")
                
            ds = Dataset.from_list(records)
            ds = ds.cast_column("audio", Audio(sampling_rate=16000, decode=False))
            
            if skip:
                output_ds_path = "birlashtirilgan_dataset_skip"
                aug_state["log"] += f"Skipping augmentation. Saving straight to {output_ds_path}...\n"
                try:
                    ds = ds.map(_mark_skip)
                except Exception as e:
                    aug_state["error"] = f"Skip-mode map failed: {e}"
                    aug_state["log"] += (
                        f"[ERROR] {aug_state['error']}\n{traceback.format_exc()}"
                    )
                    return
                if ds is None or (ds.num_rows if hasattr(ds, "num_rows") else len(ds)) == 0:
                    aug_state["error"] = "Skip-mode produced empty dataset"
                    aug_state["log"] += f"[ERROR] {aug_state['error']}\n"
                    return
                try:
                    ds.save_to_disk(output_ds_path)
                except Exception as e:
                    aug_state["error"] = f"save_to_disk failed: {e}"
                    aug_state["log"] += (
                        f"[ERROR] {aug_state['error']}\n{traceback.format_exc()}"
                    )
                    return
                aug_state["log"] += "\nPreparation complete (Skipped augmentation)!\n"
            else:
                if mode == "test":
                    print("[DEBUG] ENTER TEST", flush=True)
                    ds_len = ds.num_rows if hasattr(ds, "num_rows") else len(ds)
                    aug_state["log"] += f"Before slice: {ds_len} rows\n"
                    if ds_len <= 0:
                        aug_state["error"] = "Cannot slice: dataset is empty"
                        aug_state["log"] += f"[ERROR] {aug_state['error']}\n"
                        return
                    try:
                        ds = ds.select(range(min(100, ds_len)))
                    except Exception as e:
                        aug_state["error"] = f"select() failed: {e}"
                        aug_state["log"] += (
                            f"[ERROR] {aug_state['error']}\n{traceback.format_exc()}"
                        )
                        return
                    sliced_len = ds.num_rows if hasattr(ds, "num_rows") else len(ds)
                    aug_state["log"] += f"After slice: {sliced_len} rows\n"
                    if ds is None or sliced_len <= 0:
                        aug_state["error"] = "Sliced dataset is empty"
                        aug_state["log"] += f"[ERROR] {aug_state['error']}\n"
                        return
                    # Deduplicate, drop any None/empty defensively, then write
                    # atomically so a crash mid-write cannot leave a partial
                    # manifest that 'continue' would happily load.
                    file_names = list({
                        x.get("file_name")
                        for x in ds
                        if isinstance(x.get("file_name"), str) and x.get("file_name").strip()
                    })
                    if not file_names:
                        aug_state["error"] = "No valid file_name values to write to manifest"
                        aug_state["log"] += f"[ERROR] {aug_state['error']}\n"
                        return
                    manifest_path = ".aug_test_manifest.json"
                    tmp_manifest = manifest_path + ".tmp"
                    try:
                        with open(tmp_manifest, "w") as f:
                            json.dump(file_names, f)
                        os.replace(tmp_manifest, manifest_path)
                    except Exception as e:
                        aug_state["error"] = f"Failed to write manifest: {e}"
                        aug_state["log"] += (
                            f"[ERROR] {aug_state['error']}\n{traceback.format_exc()}"
                        )
                        if os.path.exists(tmp_manifest):
                            try:
                                os.remove(tmp_manifest)
                            except Exception:
                                pass
                        return
                    aug_state["log"] += (
                        f"Saved manifest with {len(file_names)} unique files "
                        f"to {manifest_path}\n"
                    )
                    input_ds_path = "birlashtirilgan_dataset_test_in"
                    output_ds_path = "birlashtirilgan_dataset_test_out"
                elif mode == "continue":
                    print("[DEBUG] ENTER CONTINUE", flush=True)
                    manifest_path = ".aug_test_manifest.json"
                    if not os.path.exists(manifest_path):
                        aug_state["error"] = "Manifest missing. Run test mode first."
                        aug_state["log"] += f"[ERROR] {aug_state['error']}\n"
                        return
                    try:
                        with open(manifest_path) as f:
                            data = json.load(f)
                    except Exception as e:
                        aug_state["error"] = f"Manifest corrupted: {e}"
                        aug_state["log"] += (
                            f"[ERROR] {aug_state['error']}\n{traceback.format_exc()}"
                        )
                        return
                    if not isinstance(data, list):
                        aug_state["error"] = "Manifest invalid format (not a list)"
                        aug_state["log"] += f"[ERROR] {aug_state['error']}\n"
                        return
                    used_files = set(
                        x for x in data
                        if isinstance(x, str) and x.strip()
                    )
                    if len(used_files) == 0:
                        aug_state["error"] = "Manifest empty"
                        aug_state["log"] += f"[ERROR] {aug_state['error']}\n"
                        return
                    aug_state["log"] += (
                        f"Loaded manifest with {len(used_files)} valid entries\n"
                    )

                    # Duplicate detection — warn only, never crash. Duplicates
                    # in the source dataset don't break filtering, but they
                    # signal upstream drift the user should know about.
                    all_names = [
                        x.get("file_name")
                        for x in ds
                        if x.get("file_name")
                    ]
                    if len(all_names) != len(set(all_names)):
                        aug_state["log"] += (
                            "[WARN] Duplicate file_name detected in dataset\n"
                        )

                    # Closure factory + frozenset + load_from_cache_file=False
                    # together neutralize HF Datasets' fingerprint-based cache,
                    # which can otherwise return stale filtered shards keyed on
                    # a non-deterministic pickle of `set` (PYTHONHASHSEED).
                    print(
                        f"[DEBUG] used_files defined={('used_files' in locals())} "
                        f"size={len(used_files)} "
                        f"sample={list(used_files)[:3]}",
                        flush=True,
                    )
                    before_count = ds.num_rows if hasattr(ds, "num_rows") else len(ds)
                    aug_state["log"] += f"Before filter: {before_count} rows\n"
                    try:
                        try:
                            ds = ds.filter(
                                _make_safe_filter(used_files),
                                load_from_cache_file=False,
                            )
                        except TypeError:
                            # Older datasets versions reject load_from_cache_file
                            # on .filter. Default cache reuse is the exact
                            # nondeterminism we are guarding against, so refuse
                            # to fall through silently.
                            aug_state["error"] = (
                                "Installed `datasets` version does not support "
                                "load_from_cache_file=False on .filter — refusing "
                                "to run with default cache (would reintroduce "
                                "stale-shard reuse). Upgrade `datasets`."
                            )
                            aug_state["log"] += f"[ERROR] {aug_state['error']}\n"
                            return
                    except Exception as e:
                        aug_state["error"] = f"Filter failed: {e}"
                        aug_state["log"] += (
                            f"[ERROR] {aug_state['error']}\n{traceback.format_exc()}"
                        )
                        return
                    if ds is None:
                        aug_state["error"] = "Filter returned None"
                        aug_state["log"] += f"[ERROR] {aug_state['error']}\n"
                        return
                    after_count = ds.num_rows if hasattr(ds, "num_rows") else len(ds)
                    aug_state["log"] += f"After filter: {after_count} rows\n"
                    aug_state["log"] += (
                        f"Filtered dataset: {before_count} → {after_count}\n"
                    )
                    if after_count <= 0:
                        aug_state["log"] += "No remaining rows after filtering\n"
                        return
                    input_ds_path = "birlashtirilgan_dataset_cont_in"
                    output_ds_path = "birlashtirilgan_dataset_cont_out"
                else:
                    input_ds_path = "birlashtirilgan_dataset"
                    output_ds_path = "birlashtirilgan_dataset_augmented"

                aug_state["log"] += f"Saving base dataset to {input_ds_path}...\n"
                if ds is None or (ds.num_rows if hasattr(ds, "num_rows") else len(ds)) == 0:
                    aug_state["error"] = "Cannot save: dataset is empty"
                    aug_state["log"] += f"[ERROR] {aug_state['error']}\n"
                    return
                try:
                    ds.save_to_disk(input_ds_path)
                except Exception as e:
                    aug_state["error"] = f"save_to_disk failed: {e}"
                    aug_state["log"] += (
                        f"[ERROR] {aug_state['error']}\n{traceback.format_exc()}"
                    )
                    return
                
                # 3. Run augmentation.py
                env = os.environ.copy()
                env["NUM_PROC"] = num_proc
                env["AUG_PROB"] = aug_prob
                env["INPUT_DS"] = input_ds_path
                env["OUTPUT_DS"] = output_ds_path
                # Honor a caller-provided MUSAN_PATH (env var or shell export);
                # only fall back to the bundled default when nothing is set.
                env.setdefault("MUSAN_PATH", "./musan")

                aug_state["log"] += f"[INFO] Using MUSAN_PATH={env['MUSAN_PATH']}\n"

                # Preflight: verify MUSAN is complete BEFORE spawning the
                # subprocess. Without this, audiomentations silently no-ops
                # per sample and a broken run looks identical to a healthy one.
                ok, stats = check_musan_integrity(env["MUSAN_PATH"])
                stats_block = "".join(f"  - {k}: {v}\n" for k, v in stats.items())
                if not ok:
                    aug_state["log"] += "[ERROR] MUSAN integrity check failed\n"
                    aug_state["log"] += stats_block
                    aug_state["error"] = "MUSAN integrity check failed — augmentation aborted"
                    return
                aug_state["log"] += "[INFO] MUSAN integrity OK\n"
                aug_state["log"] += stats_block

                aug_state["log"] += "Running augmentation.py...\n"
                cmd = [sys.executable, "augmentation.py", "--noise-level", str(noise_level)]
                
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
                            ds_final = ds_final.shuffle(seed=42)
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

    raw_output_dir = data.get("output_dir", "full_mapping_dataset_v2")
    abs_out, ok_out = _safe_under(raw_output_dir, ".")
    if not ok_out:
        return jsonify({"error": "Ruxsat yo'q (output_dir cwd dan tashqarida)"}), 403
    env["OUTPUT_DIR"] = abs_out
    env["HF_TOKEN"] = data.get("hf_token", "")

    ds_names = data.get("ds_names", "")
    names_list = []
    for raw in (x.strip() for x in ds_names.split(",") if x.strip()):
        # HF hub IDs (org/name) are not file paths and contain no separators
        # we'd misclassify; only sandbox values that look like local paths.
        if os.sep in raw or "/" in raw or "\\" in raw or os.path.exists(raw):
            abs_n, ok_n = _safe_under(raw, ".")
            if not ok_n:
                return jsonify({"error": f"Ruxsat yo'q (ds_names dan tashqarida: {raw})"}), 403
            names_list.append(abs_n)
        else:
            names_list.append(raw)

    # Automatically include local augmented dataset if it exists
    if os.path.exists("birlashtirilgan_dataset_augmented"):
        names_list.append(os.path.abspath("birlashtirilgan_dataset_augmented"))

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

    raw_ds_dirs = data.get("ds_dirs", "full_mapping_dataset,full_mapping_dataset_v2")
    safe_ds_dirs = []
    for raw in (x.strip() for x in raw_ds_dirs.split(",") if x.strip()):
        abs_d, ok_d = _safe_under(raw, ".")
        if not ok_d:
            return jsonify({"error": f"Ruxsat yo'q (ds_dirs cwd dan tashqarida: {raw})"}), 403
        safe_ds_dirs.append(abs_d)
    env["DS_DIRS"] = ",".join(safe_ds_dirs)

    env["COLS_TO_REMOVE"] = data.get("columns_to_remove", "")
    env["NUM_PROC"] = str(data.get("num_proc", 4))
    env["TRAIN_TEST_SPLIT"] = str(data.get("train_test_split", 0.01))

    raw_train_out = data.get("output_dir", "./whisper-large-dv_v2")
    abs_train_out, ok_train_out = _safe_under(raw_train_out, ".")
    if not ok_train_out:
        return jsonify({"error": "Ruxsat yo'q (output_dir cwd dan tashqarida)"}), 403
    env["OUTPUT_DIR"] = abs_train_out
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
    # Dataset paths and rows that came from disk are untrusted input — a
    # poisoned `audio.path` could otherwise read e.g. C:\Windows\... via
    # send_file. Allowed root: project cwd (covers both birlashtirilgan_*
    # datasets and uploads_tmp/, which lives under cwd).
    _, ds_ok = _safe_under(dataset_path, ".")
    if not ds_ok:
        return jsonify({"error": "Ruxsat yo'q (dataset_path cwd dan tashqarida)"}), 403
    try:
        from datasets import load_from_disk
        import soundfile as sf
        ds = load_from_disk(dataset_path)
        if idx < 0 or idx >= len(ds):
            return jsonify({"error": "Index out of bounds"}), 404

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
                # Server-generated path; still confirm it stays in cwd.
                abs_tmp, ok = _safe_under(tmp_path, ".")
                if not ok:
                    return jsonify({"error": "Ruxsat yo'q"}), 403
                return send_file(abs_tmp, mimetype="audio/wav", max_age=0)
            elif "array" in audio_data and audio_data["array"] is not None:
                import time
                audio_array = audio_data["array"]
                sr = audio_data.get("sampling_rate", 16000)
                tmp_dir = "uploads_tmp"
                os.makedirs(tmp_dir, exist_ok=True)
                unique_id = int(time.time() * 1000)
                tmp_path = os.path.join(tmp_dir, f"hf_audio_{idx}_{unique_id}.wav")
                sf.write(tmp_path, audio_array, sr)
                abs_tmp, ok = _safe_under(tmp_path, ".")
                if not ok:
                    return jsonify({"error": "Ruxsat yo'q"}), 403
                return send_file(abs_tmp, mimetype="audio/wav", max_age=0)
            else:
                # Path supplied by the dataset → strictly untrusted.
                audio_path = audio_data.get("path")
                abs_audio, ok = _safe_under(audio_path, ".")
                if not ok:
                    return jsonify({"error": "Ruxsat yo'q (audio path cwd dan tashqarida)"}), 403
                if not os.path.exists(abs_audio):
                    return jsonify({"error": "Audio file not found"}), 404
                return send_file(abs_audio, mimetype="audio/wav")
        else:
            # Path string supplied by the dataset → strictly untrusted.
            audio_path = str(audio_data) if audio_data is not None else ""
            abs_audio, ok = _safe_under(audio_path, ".")
            if not ok:
                return jsonify({"error": "Ruxsat yo'q (audio path cwd dan tashqarida)"}), 403
            if not os.path.exists(abs_audio):
                return jsonify({"error": "Audio file not found"}), 404
            return send_file(abs_audio, mimetype="audio/wav")
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@extensions_bp.route('/api/hf_dataset_review')
def api_hf_dataset_review():
    dataset_path = request.args.get("dataset_path", "birlashtirilgan_dataset_augmented")
    idx = int(request.args.get("idx", 0))
    _, ds_ok = _safe_under(dataset_path, ".")
    if not ds_ok:
        return jsonify({"error": "Ruxsat yo'q (dataset_path cwd dan tashqarida)"}), 403
    try:
        from datasets import load_from_disk
        ds = load_from_disk(dataset_path)
        if len(ds) == 0:
            return jsonify({"error": "Dataset is empty"}), 404

        idx = max(0, min(idx, len(ds) - 1))
        item = ds[idx]

        from urllib.parse import quote
        return jsonify({
            "idx": idx,
            "total": len(ds),
            "text": item.get("text", ""),
            "duration": item.get("duration", 0),
            "audio_url": f"/api/hf_dataset_audio?dataset_path={quote(dataset_path, safe='')}&idx={idx}"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
