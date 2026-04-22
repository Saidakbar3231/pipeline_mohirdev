"""
tab_review.py
─────────────────────────────────────────────────────────────────
Natijalarni ko'rib chiqish va tasdiqlash moduli.
  - JSONL metadata faylini yuklaydi
  - Har bir segmentni audio + transkript bilan ko'rsatadi
  - Tasdiqlash / Rad etish / Tahrirlash imkoniyati
  - Tasdiqlangan segmentlarni HuggingFace ga yuklash

Gradio o'chirildi — Flask API endpointlar orqali ishlaydi.
app.py ga qo'shish uchun:
    from tab_review import register_review_routes
    register_review_routes(app)
"""

import json
import os
from pathlib import Path

from hf_pusher import push_jsonl
from exporter import read_audiodir_sidecar


# ════════════════════════════════════════════════════════════════
# STATE  (server-side, xotira ichida)
# ════════════════════════════════════════════════════════════════

_review_state = {
    "segments":     [],
    "decisions":    {},
    "edited_texts": {},
    "current_idx":  0,
    "jsonl_path":   "",
    "audio_dir":    "",
}


def _empty_state():
    return {
        "segments":     [],
        "decisions":    {},
        "edited_texts": {},
        "current_idx":  0,
        "jsonl_path":   "",
        "audio_dir":    "",
    }


# ════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════

def _load_jsonl(path: str) -> list:
    """JSONL fayldan segmentlar ro'yxatini yuklaydi."""
    segments = []
    if not path or not os.path.exists(path):
        return segments
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    segments.append(json.loads(line))
                except Exception:
                    pass
    return segments


def _find_audio(segment: dict, audio_dir: str) -> str | None:
    """Segment uchun audio faylni topadi."""
    fname = segment.get("file_name", "")
    abs_p = segment.get("file", "")

    # 1. Absolut yo'l
    if abs_p and os.path.exists(abs_p):
        return abs_p

    # 2. audio_dir + file_name
    if fname and audio_dir:
        p = os.path.join(audio_dir, fname)
        if os.path.exists(p):
            return p

    # 3. Standart papkalar
    for folder in ["segments", "outputs", "downloads"]:
        if fname:
            p = os.path.join(folder, fname)
            if os.path.exists(p):
                return p

    return None


def _get_stats(state: dict) -> dict:
    segs      = state.get("segments", [])
    decisions = state.get("decisions", {})
    total     = len(segs)
    approved  = sum(1 for v in decisions.values() if v in ("approved", "edited"))
    rejected  = sum(1 for v in decisions.values() if v == "rejected")
    pending   = total - approved - rejected
    pct       = round(approved / total * 100) if total else 0
    return {
        "total": total, "approved": approved,
        "rejected": rejected, "pending": pending, "pct": pct
    }


def _render_segment(state: dict) -> dict:
    """Hozirgi segmentni dict sifatida qaytaradi."""
    segs  = state.get("segments", [])
    idx   = state.get("current_idx", 0)
    total = len(segs)

    if total == 0:
        return {
            "idx": 0, "total": 0,
            "transcription": "",
            "audio_url": None,
            "file_name": "",
            "meta": {},
            "decision": "pending",
            "stats": _get_stats(state),
        }

    seg   = segs[idx]
    fname = seg.get("file_name", "")
    text  = state["edited_texts"].get(fname) or seg.get("transcription", seg.get("original_text", ""))
    audio_p = _find_audio(seg, state.get("audio_dir", ""))

    meta = {}
    for k in ["duration", "snr_score", "silence_ratio", "aisha_score",
              "gemini_score", "status", "pipeline", "reason", "source"]:
        v = seg.get(k)
        if v not in (None, ""):
            meta[k] = v

    return {
        "idx":          idx + 1,
        "total":        total,
        "transcription": text,
        "audio_url":    f"/api/audio?file={audio_p}" if audio_p and os.path.exists(audio_p) else None,
        "file_name":    fname,
        "meta":         meta,
        "decision":     state["decisions"].get(fname, "pending"),
        "stats":        _get_stats(state),
    }


# ════════════════════════════════════════════════════════════════
# ACTIONS
# ════════════════════════════════════════════════════════════════

def action_load(jsonl_path: str, audio_dir: str) -> dict:
    """JSONL faylni yuklaydi."""
    global _review_state
    path = jsonl_path.strip().strip('"').strip("'")
    adir = audio_dir.strip().strip('"').strip("'") if audio_dir else ""

    segments = _load_jsonl(path)
    if not segments:
        return {"error": f"JSONL fayl topilmadi yoki bo'sh: {path}"}

    if not adir:
        adir = read_audiodir_sidecar(path)

    _review_state = {
        "segments":     segments,
        "decisions":    {},
        "edited_texts": {},
        "current_idx":  0,
        "jsonl_path":   path,
        "audio_dir":    adir,
    }
    return _render_segment(_review_state)


def action_navigate(direction: int) -> dict:
    """Oldingi / Keyingi segment."""
    global _review_state
    segs  = _review_state.get("segments", [])
    idx   = _review_state.get("current_idx", 0)
    total = len(segs)
    if total == 0:
        return {"error": "Segment yuklanmagan"}
    new_idx = max(0, min(idx + direction, total - 1))
    _review_state["current_idx"] = new_idx
    return _render_segment(_review_state)


def action_approve(edited_text: str = "") -> dict:
    """Hozirgi segmentni tasdiqlaydi."""
    global _review_state
    segs = _review_state.get("segments", [])
    if not segs:
        return {"error": "Segment yuklanmagan"}

    idx   = _review_state["current_idx"]
    fname = segs[idx].get("file_name", "")
    orig  = segs[idx].get("transcription", segs[idx].get("original_text", ""))

    if edited_text and edited_text.strip() and edited_text.strip() != orig.strip():
        _review_state["decisions"][fname]    = "edited"
        _review_state["edited_texts"][fname] = edited_text.strip()
        segs[idx]["transcription"]           = edited_text.strip()
    else:
        _review_state["decisions"][fname] = "approved"

    # Keyingi segmentga o'tish
    if idx < len(segs) - 1:
        _review_state["current_idx"] += 1

    return _render_segment(_review_state)


def action_reject() -> dict:
    """Hozirgi segmentni rad etadi."""
    global _review_state
    segs = _review_state.get("segments", [])
    if not segs:
        return {"error": "Segment yuklanmagan"}

    idx   = _review_state["current_idx"]
    fname = segs[idx].get("file_name", "")
    _review_state["decisions"][fname] = "rejected"

    if idx < len(segs) - 1:
        _review_state["current_idx"] += 1

    return _render_segment(_review_state)


def action_save() -> dict:
    """Tasdiqlangan segmentlarni yangi JSONL ga saqlaydi."""
    global _review_state
    segs      = _review_state.get("segments", [])
    decisions = _review_state.get("decisions", {})

    if not segs:
        return {"error": "Hech qanday segment yuklanmagan!"}

    approved_segs = []
    for seg in segs:
        fname = seg.get("file_name", "")
        d     = decisions.get(fname, "pending")
        if d == "rejected":
            continue
        if fname in _review_state.get("edited_texts", {}):
            seg = {**seg, "transcription": _review_state["edited_texts"][fname]}
        approved_segs.append(seg)

    if not approved_segs:
        return {"error": "Tasdiqlangan segment yo'q!"}

    base_path    = _review_state.get("jsonl_path", "outputs/metadata.jsonl")
    stem         = Path(base_path).stem
    out_path     = str(Path(base_path).parent / f"{stem}_reviewed.jsonl")

    with open(out_path, "w", encoding="utf-8") as f:
        for seg in approved_segs:
            f.write(json.dumps(seg, ensure_ascii=False) + "\n")

    # .audiodir sidecar
    if _review_state.get("audio_dir"):
        sidecar = os.path.splitext(out_path)[0] + ".audiodir"
        with open(sidecar, "w", encoding="utf-8") as f:
            f.write(_review_state["audio_dir"])

    total    = len(segs)
    approved = len(approved_segs)
    rejected = total - approved

    return {
        "ok":       True,
        "path":     out_path,
        "approved": approved,
        "rejected": rejected,
        "total":    total,
        "message":  f"Saqlandi: {out_path}\nTasdiqlangan: {approved} ta\nRad etilgan: {rejected} ta",
    }


def action_push_hf(hf_token: str, hf_org: str, hf_repo: str,
                   hf_private: bool = True) -> dict:
    """Ko'rib chiqilgan faylni HuggingFace ga yuklaydi."""
    global _review_state

    save_result = action_save()
    if "error" in save_result:
        return save_result

    reviewed_path = save_result["path"]
    if not os.path.exists(reviewed_path):
        return {"error": f"Reviewed fayl topilmadi: {reviewed_path}"}

    result = push_jsonl(
        hf_token=hf_token,
        org=hf_org,
        repo_name=hf_repo,
        jsonl_path=reviewed_path,
        audio_dir=_review_state.get("audio_dir", ""),
        private=hf_private,
    )
    return {"ok": True, "result": save_result["message"] + "\n\n" + result}


# ════════════════════════════════════════════════════════════════
# FLASK ROUTES — app.py ga register qilish uchun
# ════════════════════════════════════════════════════════════════

def register_review_routes(app):
    """
    Flask app ga review endpointlarini qo'shadi.
    app.py da:
        from tab_review import register_review_routes
        register_review_routes(app)
    """
    from flask import request, jsonify

    @app.route("/api/review/load", methods=["POST"])
    def api_review_load():
        data       = request.json or {}
        jsonl_path = data.get("jsonl_path", "")
        audio_dir  = data.get("audio_dir", "")
        result     = action_load(jsonl_path, audio_dir)
        return jsonify(result)

    @app.route("/api/review/navigate", methods=["POST"])
    def api_review_navigate():
        data      = request.json or {}
        direction = int(data.get("direction", 0))
        return jsonify(action_navigate(direction))

    @app.route("/api/review/approve", methods=["POST"])
    def api_review_approve():
        data        = request.json or {}
        edited_text = data.get("edited_text", "")
        return jsonify(action_approve(edited_text))

    @app.route("/api/review/reject", methods=["POST"])
    def api_review_reject():
        return jsonify(action_reject())

    @app.route("/api/review/save", methods=["POST"])
    def api_review_save():
        return jsonify(action_save())

    @app.route("/api/review/push_hf", methods=["POST"])
    def api_review_push_hf():
        data = request.json or {}
        return jsonify(action_push_hf(
            hf_token  = data.get("token", ""),
            hf_org    = data.get("org", ""),
            hf_repo   = data.get("repo", ""),
            hf_private= data.get("private", True),
        ))

    @app.route("/api/review/state")
    def api_review_state():
        return jsonify(_render_segment(_review_state))
