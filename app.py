"""
app.py — Flask web server (Gradio o'rniga)
Ishga tushirish: python app.py
Brauzerda: http://127.0.0.1:7861
"""

import os
import json
import threading
import zipfile
import shutil
import time
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify, send_file, render_template, abort, session, redirect, url_for
from werkzeug.utils import secure_filename
from pipeline_extensions import extensions_bp

from config import (
    V1_DURATION_MIN, V1_DURATION_MAX, V1_SILENCE_MAX,
    V1_WORD_MIN, V1_REPEAT_MAX,
    V2_DURATION_MIN, V2_DURATION_MAX,
    V2_SCORE_AUTO_APPROVE, V2_SCORE_AUTO_REJECT,
    SEGMENTS_DIR, DOWNLOAD_DIR, OUTPUT_DIR,
)

app = Flask(__name__)
app.register_blueprint(extensions_bp)

from tab_review import register_review_routes
register_review_routes(app)
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024  # 2 GB (folder uploads)
app.secret_key = os.getenv("APP_SECRET_KEY", "aisha-pipeline-secret-2026")

# ── PAROL HIMOYASI ─────────────────────────────────────────────
ACCESS_PASSWORD = os.getenv("ACCESS_PASSWORD", "mohirdev2026")
# ──────────────────────────────────────────────────────────────

UPLOAD_TMP = "uploads_tmp"
os.makedirs(UPLOAD_TMP, exist_ok=True)
os.makedirs(SEGMENTS_DIR, exist_ok=True)
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _safe_under(target_path, *allowed_roots):
    """
    Path-traversal guard. Returns (abs_target, True) iff `target_path` resolves
    inside one of `allowed_roots`. Uses os.path.commonpath rather than
    str.startswith so sibling-prefix bypass (`/srv/app` vs `/srv/app-secrets`)
    is rejected. All inputs are normalized via abspath/realpath first.
    """
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
            # Different drives on Windows → commonpath raises; treat as outside.
            continue
    return abs_target, False

# ═══════════════════════════════════════════════════════════════
# GLOBAL STATE
# ═══════════════════════════════════════════════════════════════
_state = {
    "running": False,
    "stopped": False,
    "log": [],
    "stats_v1": {"total": 0, "approved": 0, "pending": 0, "rejected": 0, "filtered": 0},
    "stats_v2": {"total": 0, "approved": 0, "pending": 0, "rejected": 0, "filtered": 0},
    "done": False,
    "notify": "",
    "output_files": [],
    "results_preview": [],
    "v_results": [],
    "v_review_idx": 0,
    "progress_current": 0,
    "progress_total": 0,
    "progress_stage": "",
}
_stop_requested = False
_resume = {}


def _log(msg: str):
    _state["log"].append(msg)
    try:
        print(msg)
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════
# PIPELINE
# ═══════════════════════════════════════════════════════════════
def _run_pipeline(params: dict, json_file_path: str = None):
    global _state, _stop_requested, _resume
    _stop_requested = False
    resume_data = dict(_resume) if _resume.get("remaining_segments") else None
    _resume.clear()
    _state = {
        "running": True, "stopped": False, "log": [], "done": False,
        "stats_v1": {"total": 0, "approved": 0, "pending": 0, "rejected": 0, "filtered": 0},
        "stats_v2": {"total": 0, "approved": 0, "pending": 0, "rejected": 0, "filtered": 0},
        "notify": "", "output_files": [],
        "v_results": [], "v_review_idx": 0,
        "last_jsonl": "", "last_csv": "", "last_audio_dir": "",
        "progress_current": 0, "progress_total": 0, "progress_stage": "Tayyorlanmoqda...",
    }

    def thread_fn():
        try:
            import gemini_utils
            from downloader import download_youtube, download_from_huggingface
            from audio_utils import vad_chunk
            from exporter import save_report, append_jsonl, save_all

            pipeline_choice = params.get("pipeline_choice") or "Yo'l 2 — Gemini STT"
            source_type     = params.get("source_type") or "Local papka"
            gemini_api_key  = params.get("gemini_api_key") or ""
            aisha_api_key   = params.get("aisha_api_key") or ""
            yt_url          = params.get("yt_url") or ""
            hf_name         = params.get("hf_name") or ""
            hf_config       = params.get("hf_config") or ""
            hf_split        = params.get("hf_split") or "train"
            hf_audio_col    = params.get("hf_audio_col") or "audio"
            hf_dataset_token= params.get("hf_dataset_token") or ""
            local_dir       = params.get("local_dir") or ""
            noise_reduce    = params.get("noise_reduce", True)
            noise_strength  = float(params.get("noise_strength") or 0.75)
            out_name_v1     = params.get("out_name_v1") or "metadata_v1"
            out_name_v2     = params.get("out_name_v2") or "metadata_v2"
            out_name_v3     = params.get("out_name_v3") or "metadata_v3"
            write_mode      = params.get("write_mode") or "new"
            fmt_jsonl       = params.get("fmt_jsonl", True)
            fmt_csv         = params.get("fmt_csv", True)
            selected_cols   = params.get("selected_cols") or \
                ["file_name", "transcription", "duration", "source", "status", "reason", "source_url"]
            dur_min = float(params.get("dur_min") or V1_DURATION_MIN)
            dur_max = float(params.get("dur_max") or V1_DURATION_MAX)

            do_v1 = "Yo'l 1" in pipeline_choice
            do_v2 = "Yo'l 2" in pipeline_choice
            do_v3 = "Yo'l 3" in pipeline_choice

            if do_v2 and gemini_api_key and gemini_api_key.strip():
                try:
                    from google import genai
                    gemini_utils._client = genai.Client(api_key=gemini_api_key.strip())
                    _log("✅ Model API key o'rnatildi")
                except Exception as e:
                    _log(f"⚠️ API key xato: {e}")

            # Yo'l 1 filtrlari — UI togglelaridan o'qiladi (Yo'l 2 bilan bir xil pattern)
            v1_filters = {
                "snr_min":      15.0 if params.get("filter_noisy", False) else 0.0,
                "duration_min": dur_min, "duration_max": dur_max,
                "silence_max":  V1_SILENCE_MAX if params.get("filter_silence", True) else 100.0,
                "word_min":     V1_WORD_MIN,
                "repeat_max":   V1_REPEAT_MAX,
                "check_language": False,
                "check_mixed":            params.get("filter_latin_only", True),
                "check_noise":            params.get("filter_no_noise_tags", True),
                "filter_background_music": params.get("filter_music", True),
                "filter_multiple_speakers": params.get("filter_multi", True),
            }
            # Yo'l 1: lokal normalizatsiya uchun sozlamalar (Gemini yo'q)
            v1_polish = {
                "normalize_numbers": params.get("filter_num_words",  True),
                "fix_punctuation":   params.get("filter_capitalize", True),
            }
            v2_filters = {
                # NOTE (duration wiring): Yo'l 2 intentionally uses the shared
                # `dur_min`/`dur_max` (UI value, defaulting to V1_DURATION_MIN=3.0)
                # — NOT config.V2_DURATION_MIN (5.0). This is deliberate: the VAD
                # stage already enforces a 3.0s floor, so 3.0 is the single source
                # of truth for the minimum and V2_DURATION_MIN stays unused here.
                # Do NOT "fix" this to V2_DURATION_MIN unless you also intend to
                # raise the effective V2 minimum from 3.0s to 5.0s.
                "duration_min": dur_min, "duration_max": dur_max,
                "filter_background_music":  params.get("filter_music", True),
                "filter_multiple_speakers": params.get("filter_multi", True),
                "filter_noisy":             params.get("filter_noisy", False),
                "filter_long_silence":      params.get("filter_silence", True),
                "filter_capitalize":        params.get("filter_capitalize", True),
                "filter_num_to_words":      params.get("filter_num_words", True),
                "filter_latin_only":        params.get("filter_latin_only", True),
                "filter_no_translate":      params.get("filter_no_translate", True),
                "filter_no_noise_tags":     params.get("filter_no_noise_tags", True),
                "filter_no_repeat_prompt":  params.get("filter_no_repeat_prompt", True),
                # Gemini sifat bahosi chegaralari — config.py yagona manba (single
                # source of truth). Shu yerda uzatiladi, stt_v2 hardcoded fallbackga
                # tushmaydi: config.py ni o'zgartirish endi pipeline'ga ta'sir qiladi.
                "score_approve_min":        V2_SCORE_AUTO_APPROVE,
                "score_reject_max":         V2_SCORE_AUTO_REJECT,
            }

            # Metadata normalizatsiya filtrlari
            norm_filters = {
                "capitalize":    params.get("norm_capitalize",    True),
                "num_words":     params.get("norm_num_words",     True),
                "apostrophe":    params.get("norm_apostrophe",    True),
                "duplicate":     params.get("norm_duplicate",     True),
                "punct":         params.get("norm_punct",         False),
                "quotes":        params.get("norm_quotes",        True),
                "sentence_case": params.get("norm_sentence_case", True),
                "double_space":  params.get("norm_double_space",  True),
                "clean_json":    params.get("norm_clean_json",    True),
                "ellipsis":      params.get("norm_ellipsis",      False),
                "dash":          params.get("norm_dash",          False),
                "cyrillic":      params.get("norm_cyrillic",      True),
                "brackets":      params.get("norm_brackets",      True),
                "html":          params.get("norm_html",          True),
                "only_digits":   params.get("norm_only_digits",   True),
                "broken_hyphen": params.get("norm_broken_hyphen", False),
                "multi_comma":   params.get("norm_multi_comma",   True),
            }

            now    = datetime.now()
            ts     = now.strftime("%H%M%S")
            date_s = now.strftime("%Y-%m-%d")
            is_new = write_mode == "new"

            date_dir       = os.path.join("outputs", date_s)
            meta_dir       = os.path.join(date_dir, "metadata")
            audio_out_dir  = os.path.join(date_dir, "audios", f"audio_{ts}")
            os.makedirs(meta_dir, exist_ok=True)
            os.makedirs(audio_out_dir, exist_ok=True)

            sfx      = f"_{ts}" if is_new else ""
            v1_out_j = os.path.join(meta_dir, f"{out_name_v1}{sfx}.jsonl")
            v1_out_c = os.path.join(meta_dir, f"{out_name_v1}{sfx}.csv")
            v2_out_j = os.path.join(meta_dir, f"{out_name_v2}{sfx}.jsonl")
            v2_out_c = os.path.join(meta_dir, f"{out_name_v2}{sfx}.csv")
            all_v1, all_v2 = [], []

            if resume_data:
                segments      = resume_data["remaining_segments"]
                all_v1        = resume_data.get("all_v1", [])
                all_v2        = resume_data.get("all_v2", [])
                do_v1         = resume_data["do_v1"]
                do_v2         = resume_data["do_v2"]
                fmt_jsonl     = resume_data["fmt_jsonl"]
                fmt_csv       = resume_data["fmt_csv"]
                selected_cols = resume_data["selected_cols"]
                v1_filters    = resume_data["v1_filters"]
                v2_filters    = resume_data["v2_filters"]
                norm_filters  = resume_data["norm_filters"]
                v1_polish     = resume_data["v1_polish"]
                v1_out_j      = resume_data["v1_out_j"]
                v1_out_c      = resume_data["v1_out_c"]
                v2_out_j      = resume_data["v2_out_j"]
                v2_out_c      = resume_data["v2_out_c"]
                audio_out_dir = resume_data["audio_out_dir"]
                _state["stats_v1"]         = dict(resume_data["stats_v1"])
                _state["stats_v2"]         = dict(resume_data["stats_v2"])
                _state["progress_total"]   = resume_data["progress_total"]
                _state["progress_current"] = resume_data["progress_done"]
                _state["progress_stage"]   = "Pipeline davom etmoqda..."
                _log(f"▶️ Davom ettirilmoqda — {len(segments)} ta segment qoldi "
                     f"({resume_data['progress_done']}/{resume_data['progress_total']})")
            else:
                if os.path.exists(SEGMENTS_DIR):
                    shutil.rmtree(SEGMENTS_DIR)
                os.makedirs(SEGMENTS_DIR, exist_ok=True)

            # ── V3 ──────────────────────────────────────────────
            if do_v3 and not resume_data:
                if not json_file_path:
                    _log("❌ JSON fayl yuklanmagan"); return
                try:
                    import random
                    from json_processor import process_json_python_filter
                    now_v3   = datetime.now()
                    ts_v3    = now_v3.strftime("%H%M%S")
                    date_v3  = now_v3.strftime("%Y-%m-%d")

                    # Chiqish papkasi — sana bo'yicha
                    date_dir_v3  = os.path.join("outputs", date_v3)
                    meta_dir_v3  = os.path.join(date_dir_v3, "metadata")
                    os.makedirs(meta_dir_v3, exist_ok=True)

                    v3_out_dir = os.path.join(date_dir_v3, out_name_v3.strip() or "v3")
                    os.makedirs(v3_out_dir, exist_ok=True)

                    _state["progress_stage"]   = "Yo'l 3 filtrlash..."
                    _state["progress_total"]   = 1
                    _state["progress_current"] = 0
                    all_results = process_json_python_filter(
                        json_path=json_file_path, output_dir=v3_out_dir, log_cb=_log)
                    _state["progress_current"] = 1

                    total_v3 = len(all_results)
                    _log(f"📊 Jami {total_v3} ta chunk yaratildi")

                    # Statistika
                    _state["stats_v1"]["total"]    = total_v3
                    _state["stats_v1"]["approved"] = total_v3
                    _state["stats_v1"]["filtered"] = 0

                    # Random 100 ta review uchun
                    if total_v3 > 100:
                        review_sample = random.sample(all_results, 100)
                        _log(f"🎲 Review uchun {total_v3} dan 100 ta random tanlandi")
                    else:
                        review_sample = all_results

                    _state["v_results"]     = review_sample
                    _state["v_review_idx"]  = 0

                    # JSONL fayl yo'lini topish (json_processor ichida yaratiladi)
                    jsonl_files = []
                    for root, dirs, files in os.walk(v3_out_dir):
                        for fn in files:
                            if fn.endswith(".jsonl"):
                                jsonl_files.append(os.path.join(root, fn))

                    # Audio papkani topish
                    audio_dirs = []
                    for root, dirs, files in os.walk(v3_out_dir):
                        for d in dirs:
                            if d.startswith("audio_v3_"):
                                audio_dirs.append(os.path.join(root, d))

                    audio_dir_v3 = audio_dirs[-1] if audio_dirs else v3_out_dir
                    jsonl_v3     = jsonl_files[-1] if jsonl_files else ""

                    # Output files yangilash
                    out_files_v3 = [p for p in jsonl_files if os.path.exists(p)]
                    _state["output_files"]   = out_files_v3
                    _state["last_jsonl"]     = jsonl_v3
                    _state["last_csv"]       = ""
                    _state["last_audio_dir"] = os.path.abspath(audio_dir_v3)

                    # Fayl hajmini log qilish
                    if jsonl_v3 and os.path.exists(jsonl_v3):
                        size_kb = os.path.getsize(jsonl_v3) / 1024
                        _log(f"📄 Saqlandi: {jsonl_v3}  ({size_kb:.1f} KB, {total_v3} ta yozuv)")

                    _state["notify"] = (
                        f"✅ {date_v3}/{out_name_v3}/dataset_v3_{ts_v3}.jsonl\n"
                        f"🎵 Audiolar: {os.path.basename(audio_dir_v3)}/\n"
                        f"📊 Jami: {total_v3} ta chunk"
                    )
                    _log("🎉 V3 pipeline yakunlandi.")
                except Exception as e:
                    import traceback
                    _log(f"❌ V3 xato: {e}\n{traceback.format_exc()}")
                return

            if not resume_data:
                # ── Audio yuklash ─────────────────────────────────
                _log("🎵 Audio yuklash boshlandi...")
                _state["progress_stage"] = "Audio yuklanmoqda..."
                audio_items = []

                if source_type == "YouTube URL":
                    if not yt_url.strip(): _log("❌ YouTube URL kiritilmagan"); return
                    for item in download_youtube(yt_url.strip(), _log, run_id=ts):
                        audio_items.append(item)

                elif source_type == "JSON URL fayl":
                    if not json_file_path: _log("❌ JSON fayl yuklanmagan"); return
                    try:
                        from json_processor import extract_chunks_from_json
                        json_chunk_dir = os.path.join(OUTPUT_DIR, "json_chunks", ts)
                        for item in extract_chunks_from_json(json_file_path, json_chunk_dir, log_cb=_log):
                            audio_items.append(item)
                    except Exception as e:
                        _log(f"❌ JSON xato: {e}"); return

                elif source_type == "HuggingFace Dataset":
                    if not hf_name.strip(): _log("❌ Dataset nomi kiritilmagan"); return
                    for item in download_from_huggingface(
                        hf_name.strip(), split=hf_split or "train",
                        audio_column=hf_audio_col or "audio",
                        hf_token=hf_dataset_token or None,
                        config=hf_config or None,
                        progress_cb=_log):
                        audio_items.append(item)

                elif source_type == "Local papka":
                    raw_path = os.path.normpath(local_dir.strip().strip('"').strip("'"))
                    if not raw_path: _log("❌ Path kiritilmagan"); return
                    if os.path.isfile(raw_path):
                        if raw_path.lower().endswith((".wav",".mp3",".ogg",".flac",".m4a")):
                            audio_items.append({"file": raw_path, "file_name": os.path.basename(raw_path),
                                                "source": "local", "source_url": raw_path})
                        else: _log(f"❌ Audio fayl emas: {raw_path}"); return
                    elif os.path.isdir(raw_path):
                        for f in sorted(os.listdir(raw_path)):
                            if f.lower().endswith((".wav",".mp3",".ogg",".flac",".m4a")):
                                fp = os.path.join(raw_path, f)
                                audio_items.append({"file": fp, "file_name": f, "source": "local", "source_url": fp})
                        if not audio_items: _log(f"❌ Papkada audio topilmadi: {raw_path}"); return
                    else: _log(f"❌ Topilmadi: {raw_path}"); return

                _log(f"📁 {len(audio_items)} ta audio fayl topildi")

                # ── VAD ───────────────────────────────────────────
                _log("✂️ VAD qirqilmoqda...")
                _state["progress_stage"] = "Audio bo'linmoqda (VAD)..."
                segments = []
                for item in audio_items:
                    try:
                        segs = vad_chunk(item, min_sec=dur_min, max_sec=dur_max,
                                         noise_reduce=noise_reduce, noise_strength=noise_strength,
                                         log=_log)
                        segments.extend(segs)
                    except Exception as e:
                        _log(f"  ⚠️ {item.get('file_name','?')}: {e}")
                _log(f"📊 Jami {len(segments)} ta segment")
                _state["progress_total"]   = len(segments)
                _state["progress_current"] = 0
                _state["progress_stage"]   = "Pipeline ishlamoqda..."

            # ── Pipeline ─────────────────────────────────────────
            for i, seg in enumerate(segments, 1):
                if _stop_requested:
                    _resume.update({
                        "remaining_segments": segments[i - 1:],
                        "all_v1": all_v1,
                        "all_v2": all_v2,
                        "stats_v1": dict(_state["stats_v1"]),
                        "stats_v2": dict(_state["stats_v2"]),
                        "do_v1": do_v1, "do_v2": do_v2,
                        "fmt_jsonl": fmt_jsonl, "fmt_csv": fmt_csv,
                        "selected_cols": selected_cols,
                        "v1_filters": v1_filters, "v2_filters": v2_filters,
                        "norm_filters": norm_filters,
                        "v1_polish": v1_polish,
                        "v1_out_j": v1_out_j, "v1_out_c": v1_out_c,
                        "v2_out_j": v2_out_j, "v2_out_c": v2_out_c,
                        "audio_out_dir": audio_out_dir,
                        "progress_done":  _state["progress_current"],
                        "progress_total": _state["progress_total"],
                    })
                    _state["stopped"] = True
                    remaining = _state["progress_total"] - _state["progress_current"]
                    _log(f"⏹ Pipeline to'xtatildi — {remaining} ta segment qoldi. "
                         f"Davom ettirish uchun '▶ Davom ettirish' tugmasini bosing.")
                    return
                _state["progress_current"] = i
                fname = seg.get("file_name", "?")

                if do_v1:
                    from stt_v1 import process_segment_v1
                    _state["stats_v1"]["total"] += 1
                    try:
                        res = process_segment_v1(seg.copy(), v1_filters, v1_polish, api_key=aisha_api_key)
                        # Norm filtri qo'llash
                        if res.get("status") != "filtered" and res.get("transcription"):
                            cleaned, drop, nreason = _apply_norm_filters(res["transcription"], norm_filters)
                            if drop:
                                res["status"] = "filtered"
                                res["reason"] = f"norm: {nreason}"
                                _log(f"[NORM_REJECT] {fname} → reason: {nreason}")
                            else:
                                res["transcription"] = cleaned
                        st = res["status"]
                        _state["stats_v1"][st] = _state["stats_v1"].get(st, 0) + 1
                        all_v1.append(res)
                        if st != "filtered":
                            _state["v_results"].append(res)
                        if fmt_jsonl:
                            _append_jsonl_filtered(res, v1_out_j, selected_cols)
                        _log(f"[V1 {i}/{len(segments)}] {fname} → {st}" +
                             (f" ({res.get('reason','')})" if st == "filtered" else ""))
                    except Exception as e:
                        _log(f"[V1] ❌ {fname}: {e}")

                if do_v2:
                    from stt_v2 import process_segment_v2
                    _state["stats_v2"]["total"] += 1
                    try:
                        res = process_segment_v2(seg.copy(), v2_filters, log=_log)
                        # Norm filtri qo'llash
                        if res.get("status") != "filtered" and res.get("transcription"):
                            cleaned, drop, nreason = _apply_norm_filters(res["transcription"], norm_filters)
                            if drop:
                                res["status"] = "filtered"
                                res["reason"] = f"norm: {nreason}"
                                _log(f"[NORM_REJECT] {fname} → reason: {nreason}")
                            else:
                                res["transcription"] = cleaned
                        st = res["status"]
                        _state["stats_v2"][st] = _state["stats_v2"].get(st, 0) + 1
                        all_v2.append(res)
                        if st != "filtered":
                            _state["v_results"].append(res)
                        if fmt_jsonl:
                            _append_jsonl_filtered(res, v2_out_j, selected_cols)
                        _log(f"[V2 {i}/{len(segments)}] {fname} → {st}" +
                             (f" ({res.get('reason','')})" if st == "filtered" else ""))
                    except Exception as e:
                        _log(f"[V2] ❌ {fname}: {e}")

            # ── CSV ──────────────────────────────────────────────
            if do_v1 and fmt_csv and all_v1:
                _save_csv_filtered(all_v1, v1_out_c, selected_cols)
            if do_v2 and fmt_csv and all_v2:
                _save_csv_filtered(all_v2, v2_out_c, selected_cols)

            # ── Audio copy — faqat transcription bo'lganlar ──────
            _log(f"💾 Audiolar saqlanmoqda → {audio_out_dir}")
            all_results = (all_v1 if do_v1 else []) + (all_v2 if do_v2 else [])
            audio_count  = 0
            skip_count   = 0
            for res in all_results:
                src    = res.get("file","")
                fn     = res.get("file_name","")
                status = res.get("status","")
                transcription = res.get("transcription","") or ""

                if not src or not fn:
                    continue

                # Filtered yoki bo'sh transcription → audio o'chiriladi
                if status == "filtered" or not transcription.strip():
                    if os.path.exists(src):
                        try: os.remove(src)
                        except: pass
                    skip_count += 1
                    continue

                # Approved/pending — saqlash
                if os.path.exists(src):
                    dst = os.path.join(audio_out_dir, fn)
                    try:
                        shutil.copy2(src, dst)
                        res["file"] = os.path.abspath(dst)
                        audio_count += 1
                    except Exception as e:
                        _log(f"  ⚠️ {fn}: {e}")

            _log(f"✅ {audio_count} ta audio saqlandi, {skip_count} ta o'chirildi (filtered/bo'sh)")

            # .audiodir sidecar
            audio_abs = os.path.abspath(audio_out_dir)
            for p in ([v1_out_j, v1_out_c] if do_v1 else []) + ([v2_out_j, v2_out_c] if do_v2 else []):
                if os.path.exists(p):
                    with open(os.path.splitext(p)[0] + ".audiodir", "w") as f:
                        f.write(audio_abs)

            save_report({"pipeline": pipeline_choice, "source": source_type,
                         "total_segments": len(segments),
                         "v1": _state["stats_v1"], "v2": _state["stats_v2"]})

            out_files = [p for p in [v1_out_j, v1_out_c, v2_out_j, v2_out_c] if os.path.exists(p)]
            _state["output_files"] = out_files
            total_segs  = len(all_v1) + len(all_v2)
            saved_count = sum(
                1 for r in (all_v1 + all_v2)
                if r.get("status") != "filtered"
                and (r.get("transcription") or "").strip()
            )
            notify_name = v1_out_j if do_v1 else v2_out_j

            # HF Push uchun avtomatik to'ldirish
            _state["last_jsonl"]     = (v1_out_j if do_v1 and os.path.exists(v1_out_j)
                                        else v2_out_j if os.path.exists(v2_out_j) else "")
            _state["last_csv"]       = (v1_out_c if do_v1 and os.path.exists(v1_out_c)
                                        else v2_out_c if os.path.exists(v2_out_c) else "")
            _state["last_audio_dir"] = os.path.abspath(audio_out_dir)

            # Fayl hajmlarini ko'rsatish — saqlangan yozuvlar soni
            for p in out_files:
                size_kb = os.path.getsize(p) / 1024
                _log(f"📄 Saqlandi: {p}  ({size_kb:.1f} KB, {saved_count} ta yozuv)")

            _state["notify"] = (
                f"✅ {date_s}/metadata/{os.path.basename(notify_name)}\n"
                f"🎵 {date_s}/audios/audio_{ts}/\n"
                f"📊 Jami: {total_segs} ta segment | ✅ Saqlangan: {saved_count} ta"
            )
            _log("🎉 Pipeline tugadi!")

        except Exception as e:
            import traceback
            _log(f"❌ Kritik xato: {e}\n{traceback.format_exc()}")
        finally:
            _state["running"] = False
            if not _state.get("stopped"):
                _state["done"] = True

    threading.Thread(target=thread_fn, daemon=True).start()


def _num_words_uz(text: str) -> str:
    """Matn ichidagi butun sonlarni o'zbek tilida so'zga aylantiradi."""
    import re
    _ones  = ["", "bir", "ikki", "uch", "to'rt", "besh", "olti", "yetti", "sakkiz", "to'qqiz"]
    _teens = ["o'n", "o'n bir", "o'n ikki", "o'n uch", "o'n to'rt", "o'n besh",
              "o'n olti", "o'n yetti", "o'n sakkiz", "o'n to'qqiz"]
    _tens  = ["", "o'n", "yigirma", "o'ttiz", "qirq", "ellik", "oltmish", "yetmish", "sakson", "to'qson"]

    def _n2w(n: int) -> str:
        if n == 0:   return "nol"
        if n < 0:    return "minus " + _n2w(-n)
        parts = []
        if n >= 1_000_000:
            parts.append(_n2w(n // 1_000_000) + " million"); n %= 1_000_000
        if n >= 1_000:
            m = n // 1_000
            parts.append(("bir ming" if m == 1 else _n2w(m) + " ming")); n %= 1_000
        if n >= 100:
            h = n // 100
            parts.append(("bir yuz" if h == 1 else _ones[h] + " yuz")); n %= 100
        if n >= 20:
            parts.append(_tens[n // 10])
            if n % 10: parts.append(_ones[n % 10])
        elif n >= 10:
            parts.append(_teens[n - 10])
        elif n > 0:
            parts.append(_ones[n])
        return " ".join(parts)

    def _replace(m):
        try:    return _n2w(int(m.group()))
        except: return m.group()

    return re.sub(r'\b\d+\b', _replace, text)


def _apply_norm_filters(text: str, norm: dict) -> tuple:
    """
    Transcription matnini normalizatsiya qiladi.
    Returns: (cleaned_text, filtered: bool, reason: str)
    filtered=True bo'lsa segment o'chirilsin; reason aniq sababni beradi
    (UI logidagi [NORM_REJECT] satri uchun).
    """
    import re

    if not text or not text.strip():
        return text, True, "empty text"

    t = text

    # 3. Apostrofni to'g'rilash: ʻ ʼ → '
    if norm.get("apostrophe"):
        t = t.replace("ʻ", "'").replace("ʼ", "'").replace("\u02bb", "'").replace("\u2019", "'")

    # 2. Raqamlarni so'z bilan yozish
    if norm.get("num_words"):
        t = _num_words_uz(t)

    # 8. Qo'sh bo'shliqlarni o'chirish
    if norm.get("double_space"):
        t = re.sub(r" {2,}", " ", t).strip()

    # 9. JSON artifactlarni tozalash
    if norm.get("clean_json"):
        t = re.sub(r"\{['\"]transcription['\"]\s*:\s*['\"](.+?)['\"]\}", r"\1", t)
        t = re.sub(r"\{.*?\}", "", t).strip()

    # 13. Bracket artifactlarni tozalash
    if norm.get("brackets"):
        t = re.sub(r"\[.*?\]", "", t).strip()

    # 14. HTML teglarni tozalash
    if norm.get("html"):
        t = re.sub(r"<[^>]+>", "", t).strip()

    # 4. Takrorlangan matnni o'chirish
    if norm.get("duplicate"):
        half = len(t) // 2
        if half > 10 and t[:half].strip() == t[half:].strip():
            t = t[:half].strip()

    # 17. Ko'p vergullarni tozalash
    if norm.get("multi_comma"):
        t = re.sub(r",{2,}", ",", t)

    # 16. Singan tirani to'g'rilash
    if norm.get("broken_hyphen"):
        t = re.sub(r"(\w)-\s+(\w)", r"\1-\2", t)

    # 11. Defisni tirega almashtirish
    if norm.get("dash"):
        t = re.sub(r" - ", " — ", t)

    # 10. Uch nuqta → ellipsis
    if norm.get("ellipsis"):
        t = t.replace("...", "…")

    # 6. Qo'shtirnoqlarni to'g'rilash
    if norm.get("quotes"):
        t = t.replace("\u201c", '"').replace("\u201d", '"')
        t = t.replace("\u2018", "'").replace("\u2019", "'")

    # 1. Birinchi harfni katta bilan yozish
    if norm.get("capitalize") and t:
        t = t[0].upper() + t[1:]

    # 7. Gap ichida kichik harfni to'g'rilash
    if norm.get("sentence_case"):
        t = re.sub(r"([.!?]\s+)([a-záéíóúA-Z])", lambda m: m.group(1) + m.group(2).upper(), t)

    # 5. Tinish belgisi qo'shish
    if norm.get("punct") and t and t[-1] not in ".!?…":
        t = t + "."

    # 15. Faqat raqam → filtr
    if norm.get("only_digits") and re.fullmatch(r"[\d\s.,]+", t.strip()):
        return t, True, "only digits"

    # 12. Kirill harflar → filtr
    if norm.get("cyrillic") and re.search(r"[а-яёА-ЯЁ]", t):
        return t, True, "cyrillic detected"

    if not t.strip():
        return t, True, "empty after normalization"

    return t, False, "ok"


def _process_with_ref_text(segment: dict, ref_text: str,
                            filters: dict, polish_opts: dict,
                            pipeline_name: str) -> dict:
    """
    JSON manbaidan kelgan segment uchun:
    reference_text = JSON dagi tayyor matn → to'g'ridan transcription.
    STT ham, Gemini ham chaqirilmaydi — matn allaqachon tayyor.
    Faqat uzunlik va matn filtri qilinadi.
    """
    from filter_text import filter_text_v1
    from config import V1_DURATION_MIN, V1_DURATION_MAX, V1_WORD_MIN, V1_REPEAT_MAX

    result = {
        **segment,
        "pipeline":      pipeline_name,
        "status":        "pending",
        "original_text": ref_text,
        "transcription": ref_text,   # JSON matn = transcription
    }

    # 1. Uzunlik filtri
    duration = segment.get("duration", 0)
    dur_min  = filters.get("duration_min", V1_DURATION_MIN)
    dur_max  = filters.get("duration_max", V1_DURATION_MAX)
    if duration < dur_min:
        return {**result, "status": "filtered", "reason": f"juda qisqa ({duration:.1f}s)"}
    if duration > dur_max:
        return {**result, "status": "filtered", "reason": f"juda uzun ({duration:.1f}s)"}

    # 2. Bo'sh matn tekshiruvi
    if not ref_text or not ref_text.strip():
        return {**result, "status": "filtered", "reason": "bo'sh matn"}

    # 3. Minimal so'z soni
    words = ref_text.strip().split()
    word_min = filters.get("word_min", 2)
    if len(words) < word_min:
        return {**result, "status": "filtered", "reason": f"kam so'z ({len(words)})"}

    # ✅ Filtrdan o'tdi — approved
    result["status"] = "approved"
    return result


def _append_jsonl_filtered(entry: dict, path: str, selected_cols: list):
    # Filtered yoki bo'sh transcription — yozmaymiz
    if entry.get("status") == "filtered":
        return
    transcription = entry.get("transcription", "") or ""
    if not transcription.strip():
        return
    row = {k: (entry.get(k) if entry.get(k) is not None else "") for k in selected_cols}
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _save_csv_filtered(entries: list, path: str, selected_cols: list):
    import csv
    # Faqat transcription bo'lgan va filtered bo'lmagan yozuvlar
    clean = [
        e for e in entries
        if e.get("status") != "filtered"
        and (e.get("transcription") or "").strip()
    ]
    if not clean:
        return
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=selected_cols, extrasaction="ignore")
        writer.writeheader()
        for e in clean:
            writer.writerow({k: e.get(k, "") for k in selected_cols})


# ═══════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        if request.form.get('password') == ACCESS_PASSWORD:
            session['logged_in'] = True
            return redirect('/')
        error = "Parol noto\u02bcg\u02bcri!"
    err_html = f"<div class=\'err\'>&#9888; {error}</div>" if error else ""
    return render_template("login.html", err_html=err_html)



@app.route('/logout')
def logout():
    session.clear()
    return redirect('/login')

@app.before_request
def require_login():
    if request.endpoint in ('login', 'static'):
        return
    if not session.get('logged_in'):
        return redirect('/login')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/stop', methods=['POST'])
def api_stop():
    global _stop_requested
    if not _state["running"]:
        return jsonify({"error": "Pipeline ishlamayapti"}), 400
    _stop_requested = True
    return jsonify({"ok": True, "msg": "To'xtatilmoqda..."})


@app.errorhandler(413)
def _payload_too_large(e):
    limit_mb = app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)
    return jsonify({
        "error": f"Yuklangan fayllar hajmi {limit_mb} MB chegarasidan oshib ketdi. "
                 f"Kamroq fayl tanlang yoki serverning MAX_CONTENT_LENGTH sozlamasini oshiring."
    }), 413


@app.route('/api/start', methods=['POST'])
def api_start():
    if _state["running"]:
        return jsonify({"error": "Pipeline allaqachon ishlayapti"}), 400

    # Explicit schema-based parse. Heuristic int/float casting (the prior
    # `'.' in val → float else int`) silently mutated string config — e.g.
    # `out_name="v1.2"` would become the float 1.2 and break path joins.
    BOOL_KEYS = {
        "noise_reduce", "filter_music", "filter_multi", "filter_noisy",
        "filter_silence", "filter_capitalize", "filter_num_words",
        "filter_latin_only", "filter_no_translate", "filter_no_noise_tags",
        "filter_no_repeat_prompt",
        "norm_capitalize", "norm_num_words", "norm_apostrophe", "norm_duplicate",
        "norm_punct", "norm_quotes", "norm_sentence_case", "norm_double_space",
        "norm_clean_json", "norm_ellipsis", "norm_dash", "norm_cyrillic",
        "norm_brackets", "norm_html", "norm_only_digits", "norm_broken_hyphen",
        "norm_multi_comma",
        "fmt_jsonl", "fmt_csv",
    }
    FLOAT_KEYS = {"dur_min", "dur_max", "noise_strength"}
    INT_KEYS = set()  # add explicit int fields here when introduced

    def _coerce(key, raw):
        if raw == "":
            return None
        if key in BOOL_KEYS:
            return raw == "true"
        if key in FLOAT_KEYS:
            try:
                return float(raw)
            except (ValueError, TypeError):
                return None
        if key in INT_KEYS:
            try:
                return int(raw)
            except (ValueError, TypeError):
                return None
        return raw  # all other fields stay as strings

    params = {key: _coerce(key, request.form[key]) for key in request.form}

    if 'selected_cols' in request.form:
        try:
            params['selected_cols'] = json.loads(request.form['selected_cols'])
        except Exception:
            params['selected_cols'] = ["file_name","transcription","duration","source","status","reason","source_url"]

    json_file_path = None
    if 'json_file' in request.files:
        f = request.files['json_file']
        if f and f.filename:
            safe_name = secure_filename(f.filename) or "upload.json"
            save_path = os.path.join(UPLOAD_TMP, safe_name)
            abs_save, ok = _safe_under(save_path, UPLOAD_TMP)
            if not ok:
                return jsonify({"error": "Yuklash uchun ruxsat yo'q"}), 403
            f.save(abs_save)
            json_file_path = abs_save

    # Local folder picker: browser cannot expose absolute paths, so it sends
    # the audio files themselves. Save them to a fresh temp directory and
    # treat that as the local_dir for the pipeline.
    local_files = request.files.getlist('local_files')
    if local_files:
        import tempfile
        tmp_dir = tempfile.mkdtemp(prefix='local_upload_', dir=UPLOAD_TMP)
        saved = 0
        skipped = 0
        for f in local_files:
            if not f or not f.filename:
                skipped += 1
                continue
            name = os.path.basename(f.filename)
            if not name.lower().endswith((".wav", ".mp3", ".ogg", ".flac", ".m4a")):
                skipped += 1
                continue
            f.save(os.path.join(tmp_dir, name))
            saved += 1
        print(f"[api_start] Uploaded local folder: {saved} audio saved, {skipped} skipped → {tmp_dir}")
        if saved > 0:
            params['local_dir'] = tmp_dir
        else:
            # All uploaded files were rejected — surface a clear error
            return jsonify({
                "error": "Yuklangan faylllar ichida qo'llab-quvvatlanadigan audio topilmadi "
                         "(.wav .mp3 .ogg .flac .m4a)."
            }), 400

    _run_pipeline(params, json_file_path)
    return jsonify({"status": "started"})


@app.route('/api/progress')
def api_progress():
    files_info = []
    for p in _state.get("output_files", []):
        if os.path.exists(p):
            size = os.path.getsize(p)
            files_info.append({"path": p, "name": os.path.basename(p),
                                "size": f"{size/1024:.1f} KB"})
    if not files_info and os.path.exists(OUTPUT_DIR):
        for root, dirs, fnames in os.walk(OUTPUT_DIR):
            for fn in sorted(fnames):
                if fn.endswith((".jsonl",".csv")) and not fn.endswith(".audiodir"):
                    p = os.path.join(root, fn)
                    size = os.path.getsize(p)
                    files_info.append({"path": p, "name": fn, "size": f"{size/1024:.1f} KB"})

    cur   = _state.get("progress_current", 0)
    total = _state.get("progress_total", 0)
    pct   = round(cur / total * 100) if total > 0 else (100 if _state["done"] else 0)

    return jsonify({
        "log":           "\n".join(_state["log"][-60:]),
        "stats_v1":      _state["stats_v1"],
        "stats_v2":      _state["stats_v2"],
        "notify":        _state.get("notify",""),
        "files":         files_info,
        "running":       _state["running"],
        "done":          _state["done"],
        "stopped":       _state.get("stopped", False),
        "can_resume":    bool(_resume.get("remaining_segments")),
        "review_total":  len(_state.get("v_results",[])),
        "last_jsonl":    _state.get("last_jsonl",""),
        "last_csv":      _state.get("last_csv",""),
        "last_audio_dir":_state.get("last_audio_dir",""),
        "progress_pct":  pct,
        "progress_cur":  cur,
        "progress_total":total,
        "progress_stage":_state.get("progress_stage",""),
    })


@app.route('/api/review')
def api_review():
    direction = int(request.args.get("dir", 0))
    results = _state.get("v_results", [])
    if not results:
        return jsonify({"error": "Natijalar yo'q"}), 404
    idx = max(0, min(_state.get("v_review_idx", 0) + direction, len(results) - 1))
    _state["v_review_idx"] = idx
    item = results[idx]
    audio_path = item.get("file", "")
    audio_url = f"/api/audio?file={audio_path}" if audio_path and os.path.exists(audio_path) else None
    meta = {k: item[k] for k in ["file_name","status","reason","duration","snr_score",
                                   "silence_ratio","source","pipeline"] if item.get(k) not in (None, "")}
    return jsonify({
        # Invariant: backend always returns 0-based idx. Frontend formats +1.
        "idx": idx, "total": len(results),
        "transcription": item.get("transcription") or item.get("text") or "",
        "audio_url": audio_url, "file_name": item.get("file_name",""), "meta": meta,
    })


@app.route('/api/audio')
def api_audio():
    file_path = request.args.get("file","")
    if not file_path or not os.path.exists(file_path): abort(404)
    abs_path, ok = _safe_under(file_path, ".")
    if not ok: abort(403)
    return send_file(abs_path, mimetype="audio/wav")


@app.route('/api/download')
def api_download():
    file_path = request.args.get("file","")
    if not file_path or not os.path.exists(file_path): abort(404)
    abs_path, ok = _safe_under(file_path, ".")
    if not ok: abort(403)
    return send_file(abs_path, as_attachment=True, download_name=os.path.basename(abs_path))


@app.route('/api/zip', methods=['POST'])
def api_zip():
    # Oxirgi pipeline audio papkasini ZIP qilish
    body = (request.get_json(silent=True) or {}) if request.is_json else {}
    audio_dir = body.get("audio_dir", "") if isinstance(body, dict) else ""
    wav_files = []
    if audio_dir:
        abs_dir, ok = _safe_under(audio_dir, ".")
        if not ok:
            return jsonify({"error": "Ruxsat yo'q (audio_dir cwd dan tashqarida)"}), 403
        if os.path.exists(abs_dir):
            for fn in sorted(os.listdir(abs_dir)):
                if fn.lower().endswith(".wav"):
                    wav_files.append(os.path.join(abs_dir, fn))
    if not wav_files:
        for root, dirs, files in os.walk(OUTPUT_DIR):
            for fn in files:
                if fn.lower().endswith(".wav"):
                    wav_files.append(os.path.join(root, fn))
    if not wav_files and os.path.exists(SEGMENTS_DIR):
        wav_files = [os.path.join(SEGMENTS_DIR,f) for f in os.listdir(SEGMENTS_DIR)
                     if f.lower().endswith(".wav")]
    if not wav_files:
        return jsonify({"error": "Audio fayl topilmadi"}), 404
    zip_path = os.path.join(OUTPUT_DIR, f"audio_{int(time.time())}.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in wav_files:
            zf.write(f, os.path.basename(f))
    size_mb = round(os.path.getsize(zip_path)/1024/1024, 1)
    return jsonify({"path": zip_path, "count": len(wav_files),
                    "size_mb": size_mb,
                    "name": os.path.basename(zip_path),
                    "download_url": f"/api/download?file={zip_path}"})


@app.route('/api/delete', methods=['POST'])
def api_delete():
    """Fayl yoki papkani o'chirish — faqat OUTPUT_DIR ichida."""
    data = request.json or {}
    path = data.get("path", "")
    if not path:
        return jsonify({"error": "path berilmagan"}), 400
    # Restrict deletes to OUTPUT_DIR. cwd-wide delete (prior behavior) would
    # let any logged-in caller wipe source files / configs / datasets.
    abs_path, ok = _safe_under(path, OUTPUT_DIR)
    if not ok:
        return jsonify({"error": "Ruxsat yo'q (faqat outputs/ ichidagi fayllar)"}), 403
    try:
        if os.path.isfile(abs_path):
            os.remove(abs_path)
            # .audiodir sidecar ham o'chirish
            sidecar = os.path.splitext(abs_path)[0] + ".audiodir"
            if os.path.exists(sidecar):
                os.remove(sidecar)
        elif os.path.isdir(abs_path):
            shutil.rmtree(abs_path)
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/audio-zips')
def api_audio_zips():
    """outputs/ ichidagi barcha ZIP fayllar ro'yxati."""
    zips = []
    if os.path.exists(OUTPUT_DIR):
        for root, dirs, files in os.walk(OUTPUT_DIR):
            for fn in sorted(files):
                if fn.lower().endswith(".zip"):
                    p = os.path.join(root, fn)
                    size = os.path.getsize(p)
                    zips.append({
                        "path": p,
                        "name": fn,
                        "size": f"{size/1024/1024:.1f} MB" if size > 1024*1024 else f"{size/1024:.1f} KB"
                    })
    return jsonify(zips)



@app.route('/api/hf/upload', methods=['POST'])
def api_hf_upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file"}), 400
    f = request.files['file']
    if not f.filename:
        return jsonify({"error": "No filename"}), 400
    safe_name = secure_filename(f.filename)
    if not safe_name:
        return jsonify({"error": "Invalid filename"}), 400
    save_path = os.path.join(UPLOAD_TMP, safe_name)
    abs_save, ok = _safe_under(save_path, UPLOAD_TMP)
    if not ok:
        return jsonify({"error": "Ruxsat yo'q"}), 403
    f.save(abs_save)
    return jsonify({"path": abs_save, "name": safe_name})


@app.route('/api/hf/push', methods=['POST'])
def api_hf_push():
    data = request.json or {}
    try:
        from hf_pusher import push_jsonl, push_csv
        fn = push_jsonl if data.get("mode","jsonl") == "jsonl" else push_csv
        result = fn(data.get("token",""), data.get("org",""), data.get("repo",""),
                    data.get("file_path",""), data.get("audio_dir",""), data.get("private",True))
        return jsonify({"result": result})
    except Exception as e:
        # Invariant: failure returns {"error": ...}. Never put failures under
        # the success key — frontend cannot disambiguate otherwise.
        return jsonify({"error": f"Xato: {e}"}), 500


@app.route('/api/audiodir')
def api_audiodir():
    fpath = request.args.get("file","")
    if not fpath: return jsonify({"dir": ""})
    sidecar = os.path.splitext(fpath)[0] + ".audiodir"
    abs_sidecar, ok = _safe_under(sidecar, OUTPUT_DIR)
    if not ok:
        return jsonify({"dir": ""})
    if not os.path.exists(abs_sidecar):
        return jsonify({"dir": ""})
    try:
        with open(abs_sidecar) as fh:
            return jsonify({"dir": fh.read().strip()})
    except Exception:
        return jsonify({"dir": ""})


@app.route('/api/outputs/list')
def api_outputs_list():
    files = []
    if os.path.exists(OUTPUT_DIR):
        for root, dirs, fnames in os.walk(OUTPUT_DIR):
            for fn in sorted(fnames):
                if fn.endswith((".jsonl",".csv")) and not fn.endswith(".audiodir"):
                    files.append({"path": os.path.join(root, fn), "name": fn})
    return jsonify(files)


# ═══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    # Kill existing port 7861 on Windows
    import platform
    if platform.system() == "Windows":
        import subprocess
        try:
            out = subprocess.check_output('netstat -ano | findstr :7861', shell=True).decode()
            for line in out.strip().split('\n'):
                if 'LISTENING' in line:
                    pid = line.strip().split()[-1]
                    subprocess.call(['taskkill','/F','/PID',pid],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

    print("\n" + "="*52)
    print("  MohirDev Pipeline - Flask Interface")
    print("  http://127.0.0.1:7861")
    print("="*52 + "\n")
    app.run(host='0.0.0.0', port=7861, debug=False, threaded=True)
