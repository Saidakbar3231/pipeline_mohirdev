"""
hf_pusher.py
─────────────────────────────────────────────────────────────────
HuggingFace ga JSONL yoki CSV metadata + audio push qiladi.
"""

import json
import csv
import os
from pathlib import Path

try:
    from config import OUTPUT_V1_JSONL, OUTPUT_V1_CSV, OUTPUT_V2_JSONL, OUTPUT_V2_CSV
except ImportError:
    OUTPUT_V1_JSONL = "outputs/metadata_v1.jsonl"
    OUTPUT_V1_CSV   = "outputs/metadata_v1.csv"
    OUTPUT_V2_JSONL = "outputs/metadata_v2.jsonl"
    OUTPUT_V2_CSV   = "outputs/metadata_v2.csv"


def get_output_files() -> list:
    """outputs/ papkasidagi mavjud JSONL va CSV fayllar ro'yxati."""
    candidates = [OUTPUT_V1_JSONL, OUTPUT_V1_CSV, OUTPUT_V2_JSONL, OUTPUT_V2_CSV]
    return [p for p in candidates if Path(p).exists()]


def _clean_path(p: str) -> str:
    """
    Foydalanuvchi kiritgan yo'ldan ortiqcha belgilarni tozalaydi:
      - bosh/oxirdagi bo'shliqlar
      - qo'shtirnoqlar: "C:\..." → C:\...
      - apostrof: 'C:\...' → C:\...
    """
    if not p:
        return p
    p = p.strip()
    # Ikki tomondan qo'shtirnoq yoki apostrof
    if (p.startswith('"') and p.endswith('"')) or \
       (p.startswith("'") and p.endswith("'")):
        p = p[1:-1]
    return p.strip()


def push_jsonl(
    hf_token: str,
    org: str,
    repo_name: str,
    jsonl_path: str,
    audio_dir: str,
    private: bool = True,
) -> str:

    if not hf_token or not hf_token.strip():
        return "❌ HF Token kiritilmagan!"
    if not org or not org.strip():
        return "❌ Org nomi kiritilmagan!"
    if not repo_name or not repo_name.strip():
        return "❌ Repo nomi kiritilmagan!"
    if not jsonl_path or not Path(jsonl_path).exists():
        return f"❌ JSONL fayl topilmadi: {jsonl_path}"

    # Audio papka
    if audio_dir and audio_dir.strip():
        audio_base = Path(_clean_path(audio_dir))
    else:
        audio_base = Path(jsonl_path).parent
        segments = Path("segments")
        if segments.exists():
            audio_base = segments.resolve()

    if not audio_base.exists():
        return f"❌ Audio papka topilmadi: {audio_base}"

    # JSONL o'qish
    audio_paths, transcriptions, missing = [], [], []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except Exception as e:
                return f"❌ JSONL {i}-qatorda xato: {e}"
            fname = record.get("file_name", "")
            text  = record.get("transcription", "")
            if not fname:
                continue
            full_path = audio_base / fname
            if full_path.exists():
                audio_paths.append(str(full_path))
                transcriptions.append(text)
            else:
                missing.append(str(full_path))

    if not audio_paths:
        msg = f"❌ Audio fayllar topilmadi!\n   Audio papka: {audio_base}\n"
        if missing:
            msg += "   Qidirilgan (namuna):\n"
            for p in missing[:3]:
                msg += f"     • {p}\n"
        return msg

    try:
        from datasets import Dataset, Audio as HFAudio
    except ImportError:
        return "❌ 'datasets' o'rnatilmagan: pip install datasets"

    repo_id = f"{org.strip()}/{repo_name.strip()}"
    log = [f"📂 {len(audio_paths)} ta audio topildi"]
    if missing:
        log.append(f"⚠️  {len(missing)} ta fayl o'tkazib yuborildi")

    try:
        ds = Dataset.from_dict({"audio": audio_paths, "transcription": transcriptions})
        ds = ds.cast_column("audio", HFAudio(sampling_rate=16000))
        log.append(f"✅ Dataset yaratildi: {len(ds)} ta yozuv")
    except Exception as e:
        return "\n".join(log) + f"\n❌ Dataset xatosi: {e}"

    try:
        log.append(f"🚀 Push → {repo_id} ...")
        ds.push_to_hub(repo_id, split="train", token=hf_token.strip(), private=private)
        log.append("✅ Muvaffaqiyatli!")
        log.append(f"🔗 https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        return "\n".join(log) + f"\n❌ Push xatosi: {e}"

    return "\n".join(log)


def push_csv(
    hf_token: str,
    org: str,
    repo_name: str,
    csv_path: str,
    audio_dir: str,
    private: bool = True,
) -> str:

    if not hf_token or not hf_token.strip():
        return "❌ HF Token kiritilmagan!"
    if not org or not org.strip():
        return "❌ Org nomi kiritilmagan!"
    if not repo_name or not repo_name.strip():
        return "❌ Repo nomi kiritilmagan!"
    if not csv_path or not Path(csv_path).exists():
        return f"❌ CSV fayl topilmadi: {csv_path}"

    # Audio papka
    if audio_dir and audio_dir.strip():
        audio_base = Path(_clean_path(audio_dir))
    else:
        audio_base = Path(csv_path).parent
        segments = Path("segments")
        if segments.exists():
            audio_base = segments.resolve()

    if not audio_base.exists():
        return f"❌ Audio papka topilmadi: {audio_base}"

    # CSV o'qish
    audio_paths, transcriptions, missing = [], [], []
    with open(csv_path, "r", encoding="utf-8") as f:
        try:
            rows = list(csv.DictReader(f))
        except Exception as e:
            return f"❌ CSV o'qishda xato: {e}"

    if not rows:
        return "❌ CSV fayl bo'sh!"

    for record in rows:
        fname = record.get("file_name", "")
        text  = record.get("transcription", "")
        if not fname:
            continue
        full_path = audio_base / fname
        if full_path.exists():
            audio_paths.append(str(full_path))
            transcriptions.append(text)
        else:
            missing.append(str(full_path))

    if not audio_paths:
        msg = f"❌ Audio fayllar topilmadi!\n   Audio papka: {audio_base}\n"
        if missing:
            msg += "   Qidirilgan (namuna):\n"
            for p in missing[:3]:
                msg += f"     • {p}\n"
        return msg

    try:
        from datasets import Dataset, Audio as HFAudio
    except ImportError:
        return "❌ 'datasets' o'rnatilmagan: pip install datasets"

    repo_id = f"{org.strip()}/{repo_name.strip()}"
    log = [f"📂 {len(audio_paths)} ta audio topildi"]
    if missing:
        log.append(f"⚠️  {len(missing)} ta fayl o'tkazib yuborildi")

    try:
        ds = Dataset.from_dict({"audio": audio_paths, "transcription": transcriptions})
        ds = ds.cast_column("audio", HFAudio(sampling_rate=16000))
        log.append(f"✅ Dataset yaratildi: {len(ds)} ta yozuv")
    except Exception as e:
        return "\n".join(log) + f"\n❌ Dataset xatosi: {e}"

    try:
        log.append(f"🚀 Push → {repo_id} ...")
        ds.push_to_hub(repo_id, split="train", token=hf_token.strip(), private=private)
        log.append("✅ Muvaffaqiyatli!")
        log.append(f"🔗 https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        return "\n".join(log) + f"\n❌ Push xatosi: {e}"

    return "\n".join(log)
