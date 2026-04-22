"""
downloader.py
─────────────────────────────────────────────────────────────────
3 ta manbadan audio yuklab oladi:
  1. YouTube URL
  2. JSON fayl ichidagi URL lar
  3. HuggingFace dataset
"""

import os
import json
import shutil
import subprocess
from pathlib import Path
from typing import Generator

from config import DOWNLOAD_DIR, SAMPLE_RATE

os.makedirs(DOWNLOAD_DIR, exist_ok=True)


# ════════════════════════════════════════════════════════════════
# 1. YOUTUBE
# ════════════════════════════════════════════════════════════════

def download_youtube(url: str, progress_cb=None,
                     run_id: str = None) -> Generator[dict, None, None]:
    """
    YouTube URL dan audio yuklab oladi.

    Optional: agar loyiha ildizida 'cookies.txt' mavjud bo'lsa,
    authenticated quota uchun ishlatiladi (429 bypass).
    Brauzer cookies avtomatik ishlatilmaydi (Windows DPAPI muammosi).
    """
    from datetime import datetime

    rid     = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(DOWNLOAD_DIR, "youtube", rid)
    os.makedirs(out_dir, exist_ok=True)

    if shutil.which("yt-dlp") is None:
        raise RuntimeError("yt-dlp topilmadi. O'rnatish: pip install -U yt-dlp")

    template = os.path.join(out_dir, "%(id)s.%(ext)s")

    cmd = [
        "yt-dlp",
        "--extract-audio",
        "--audio-format", "wav",
        "--audio-quality", "0",
        "--postprocessor-args", f"-ar {SAMPLE_RATE} -ac 1",
        "--output", template,
        # Non-JS clients — don't need a JS runtime
        "--extractor-args", "youtube:player_client=default,tv,ios,android_vr",
        # Modern UA to avoid naive bot checks
        "--user-agent",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        # Retry on transient errors
        "--retries", "3",
        "--fragment-retries", "3",
        "--retry-sleep", "5",
        "--no-playlist" if "list=" not in url else "--yes-playlist",
    ]

    # Optional cookies.txt — if user exported it manually
    cookies_txt = os.path.abspath("cookies.txt")
    if os.path.exists(cookies_txt):
        cmd[1:1] = ["--cookies", cookies_txt]
        if progress_cb:
            progress_cb(f"🔐 cookies.txt topildi — authenticated yuklash")

    cmd.append(url)

    if progress_cb:
        progress_cb(f"YouTube yuklanmoqda: {url}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        stderr = result.stderr or ""
        s = stderr.lower()
        # Classified, friendly error messages
        if "429" in stderr or "too many requests" in s:
            raise RuntimeError(
                "YouTube rate-limit (HTTP 429). IP vaqtincha bloklangan.\n"
                "Yechimlar:\n"
                "  • 30–60 daqiqa kutib qayta urinib ko'ring\n"
                "  • VPN yoki mobil hotspot orqali boshqa IP ishlating\n"
                "  • cookies.txt faylini loyiha ildiziga joylashtiring "
                "(Chrome extension: 'Get cookies.txt LOCALLY')"
            )
        if "no supported javascript" in s or "javascript runtime" in s:
            raise RuntimeError(
                "yt-dlp JavaScript runtime talab qiladi.\n"
                "  1) Node.js o'rnating: https://nodejs.org (LTS)\n"
                "  2) yt-dlp ni yangilang: pip install -U yt-dlp"
            )
        if "sign in to confirm" in s or "age-restricted" in s:
            raise RuntimeError(
                "Video yoshga cheklangan. cookies.txt fayli kerak bo'ladi."
            )
        if "video unavailable" in s or "private video" in s:
            raise RuntimeError("Video mavjud emas yoki mintaqaviy cheklangan.")
        if "members-only" in s:
            raise RuntimeError("Bu video faqat a'zolar uchun (members-only).")
        raise RuntimeError(f"yt-dlp xato: {stderr[:400]}")

    files = sorted(Path(out_dir).glob("*.wav"))
    if not files:
        raise RuntimeError(
            "yt-dlp muvaffaqiyat qaytardi, lekin WAV fayl yaratilmadi."
        )
    for f in files:
        yield {
            "file":       str(f),
            "source":     "youtube",
            "source_url": url,
            "file_name":  f.name,
        }


# ════════════════════════════════════════════════════════════════
# 2. JSON URL
# ════════════════════════════════════════════════════════════════

def download_from_json(json_path: str, url_field: str = "url",
                       progress_cb=None) -> Generator[dict, None, None]:
    """
    JSON fayl ichidagi URL lardan audio yuklab oladi.

    JSON formatlari qo'llab-quvvatlanadi:
      [{"url": "http://..."}, ...]
      {"data": [{"audio_url": "http://..."}, ...]}
      {"url": "http://..."}  (bitta)
    """
    import requests

    out_dir = os.path.join(DOWNLOAD_DIR, "json")
    os.makedirs(out_dir, exist_ok=True)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Turli formatlarni qo'llab-quvvatlash
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        # dict bo'lsa — ichidan list topamiz
        for v in data.values():
            if isinstance(v, list):
                items = v
                break
        else:
            items = [data]
    else:
        raise ValueError("JSON formati qo'llab-quvvatlanmaydi")

    total = len(items)
    for i, item in enumerate(items, 1):
        # URL ni topish — turli field nomlar
        audio_url = None
        for field in [url_field, "url", "audio_url", "link", "audio", "path"]:
            if field in item:
                audio_url = item[field]
                break

        if not audio_url:
            if progress_cb:
                progress_cb(f"[{i}/{total}] URL topilmadi, o'tkazildi")
            continue

        if progress_cb:
            progress_cb(f"[{i}/{total}] Yuklanmoqda: {audio_url[:60]}...")

        try:
            fname = f"json_{i:04d}_{Path(audio_url).stem}.wav"
            out_path = os.path.join(out_dir, fname)

            if os.path.exists(out_path):
                yield {"file": out_path, "source": "json",
                       "source_url": audio_url, "file_name": fname}
                continue

            r = requests.get(audio_url, timeout=60, stream=True)
            r.raise_for_status()

            # Vaqtinchalik fayl saqlab, keyin WAV ga o'tkazamiz
            tmp_path = out_path + ".tmp"
            with open(tmp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

            # ffmpeg bilan WAV ga o'tkazish
            _convert_to_wav(tmp_path, out_path)
            os.remove(tmp_path)

            yield {
                "file":       out_path,
                "source":     "json",
                "source_url": audio_url,
                "file_name":  fname,
                "meta":       {k: v for k, v in item.items()
                               if k not in [url_field, "url", "audio_url"]}
            }

        except Exception as e:
            if progress_cb:
                progress_cb(f"  ❌ Xato: {e}")


# ════════════════════════════════════════════════════════════════
# 3. HUGGINGFACE
# ════════════════════════════════════════════════════════════════

def download_from_huggingface(dataset_name: str, split: str = "train",
                               audio_column: str = "audio",
                               text_column: str = None,
                               max_samples: int = None,
                               progress_cb=None) -> Generator[dict, None, None]:
    """
    HuggingFace dataset dan audio yuklab oladi.

    Misol:
      dataset_name = "mozilla-foundation/common_voice_11_0"
      split = "train"
      audio_column = "audio"
    """
def download_from_huggingface(dataset_name: str, split: str = "train",
                               audio_column: str = "audio",
                               text_column: str = None,
                               max_samples: int = None,
                               hf_token: str = None,
                               config: str = None,
                               progress_cb=None) -> Generator[dict, None, None]:
    """
    HuggingFace dataset dan audio yuklab oladi.
    Public va private datasetlarni qo'llab-quvvatlaydi.
    """
    try:
        from datasets import load_dataset, Audio
    except ImportError:
        raise RuntimeError("datasets kutubxonasi topilmadi. O'rnatish: pip install datasets")

    # URL berilgan bo'lsa — faqat dataset nomini olish
    dataset_name = dataset_name.strip()
    if "huggingface.co" in dataset_name:
        if "/datasets/" in dataset_name:
            dataset_name = dataset_name.split("/datasets/", 1)[1]
        elif "huggingface.co/" in dataset_name:
            dataset_name = dataset_name.split("huggingface.co/", 1)[1]
        dataset_name = dataset_name.split("?")[0].rstrip("/")

    # /viewer, /tree, /blob kabi qo'shimchalarni olib tashlash
    # mrmuminov/uzbek_voice/viewer → mrmuminov/uzbek_voice
    hf_suffixes = ["/viewer", "/tree", "/blob", "/resolve", "/raw"]
    for suffix in hf_suffixes:
        if suffix in dataset_name:
            dataset_name = dataset_name.split(suffix)[0]
    # Faqat 2 qism bo'lishi kerak: org/name
    parts = dataset_name.strip("/").split("/")
    if len(parts) > 2:
        dataset_name = "/".join(parts[:2])

    if progress_cb:
        progress_cb(f"📦 Dataset: {dataset_name}" + (" (private)" if hf_token else " (public)"))

    # Papka nomi uchun xavfsiz nom
    safe_name = dataset_name.replace("/", "_").replace(":", "_")
    out_dir = os.path.join(DOWNLOAD_DIR, "huggingface", safe_name)
    os.makedirs(out_dir, exist_ok=True)

    if progress_cb:
        progress_cb(f"HuggingFace yuklanmoqda: {dataset_name} [{split}]")

    # token parametri — private dataset uchun
    load_kwargs = {"split": split, "streaming": True}
    if hf_token:
        load_kwargs["token"] = hf_token
    if config:
        load_kwargs["name"] = config

    try:
        ds = load_dataset(dataset_name, **load_kwargs)
        try:
            ds = ds.cast_column(audio_column, Audio(decode=False))
        except Exception:
            pass
    except Exception as e1:
        err_str = str(e1)

        # Config kerak bo'lsa — xato logga chiqariladi, foydalanuvchi kiritsin
        if "Config name is missing" in err_str or "pick one among" in err_str:
            import re
            configs = re.findall(r"'([a-z_]+)'", err_str)
            msg = f"❌ Bu dataset config talab qiladi!\n   Mavjud configlar: {configs}\n   UI da 'Config/Subset' maydoniga birini kiriting (masalan: uz yoki en)"
            raise RuntimeError(msg)

        # Split noto'g'ri bo'lsa — mavjud splitlarni topamiz
        if "Unknown split" in err_str or "Bad split" in err_str:
            try:
                from datasets import get_dataset_split_names
                avail_kwargs = {"token": hf_token} if hf_token else {}
                if config:
                    avail_kwargs["config_name"] = config
                available = get_dataset_split_names(dataset_name, **avail_kwargs)
                if available:
                    if progress_cb:
                        progress_cb(f"⚠️ '{split}' split yo'q. Mavjud: {available}. '{available[0]}' ishlatiladi.")
                    load_kwargs["split"] = available[0]
                    ds = load_dataset(dataset_name, **load_kwargs)
                    try:
                        ds = ds.cast_column(audio_column, Audio(decode=False))
                    except Exception:
                        pass
                else:
                    raise RuntimeError(f"Dataset yuklanmadi: {e1}")
            except RuntimeError:
                raise
            except Exception as e2:
                raise RuntimeError(f"Dataset yuklanmadi: {e2}")
        else:
            try:
                load_kwargs_no_stream = {k: v for k, v in load_kwargs.items() if k != "streaming"}
                ds = load_dataset(dataset_name, **load_kwargs_no_stream)
                try:
                    ds = ds.cast_column(audio_column, Audio(decode=False))
                except Exception:
                    pass
            except Exception as e2:
                raise RuntimeError(f"Dataset yuklanmadi: {e2}")

    if progress_cb:
        progress_cb(f"⏳ Dataset yuklanmoqda, iltimos kuting...")

    count = 0
    first_item = None
    retry_count = 0
    max_retries = 3

    while retry_count <= max_retries:
        try:
          for i, item in enumerate(ds):
            if max_samples and count >= max_samples:
                break

            # Birinchi elementda dataset strukturasini ko'rsatamiz
            if i == 0:
                first_item = item
                keys = list(item.keys())
                if progress_cb:
                    progress_cb(f"📋 Dataset ustunlari: {keys}")

                # Audio ustunni avtomatik topish
                audio_cols = [k for k, v in item.items()
                              if isinstance(v, dict) and ("bytes" in v or "array" in v or "path" in v)]
                if not audio_cols:
                    audio_cols = [k for k in keys
                                  if any(x in k.lower() for x in ["audio","speech","wav","sound","voice","recording"])]

                if audio_cols and audio_column not in keys:
                    audio_column = audio_cols[0]
                    if progress_cb:
                        progress_cb(f"🔍 Audio ustun avtomatik topildi: '{audio_column}'")
                elif audio_column not in keys:
                    if progress_cb:
                        progress_cb(f"⚠️ '{audio_column}' ustun yo'q. Mavjud: {keys}")
                    return

            audio_data = item.get(audio_column)
            if audio_data is None:
                continue

            fname = f"hf_{i:06d}.wav"
            out_path = os.path.join(out_dir, fname)

            if progress_cb and i % 10 == 0:
                progress_cb(f"HuggingFace: {i} ta yuklanmoqda...")

            try:
                if not os.path.exists(out_path):
                    if isinstance(audio_data, dict):
                        if "bytes" in audio_data and audio_data["bytes"]:
                            tmp = out_path + ".tmp"
                            with open(tmp, "wb") as f:
                                f.write(audio_data["bytes"])
                            _convert_to_wav(tmp, out_path)
                            try: os.remove(tmp)
                            except: pass
                        elif "array" in audio_data:
                            import soundfile as sf
                            import numpy as np
                            arr = np.array(audio_data["array"])
                            sr  = audio_data.get("sampling_rate", SAMPLE_RATE)
                            sf.write(out_path, arr, sr)
                        elif "path" in audio_data and audio_data["path"] and os.path.exists(audio_data["path"]):
                            _convert_to_wav(audio_data["path"], out_path)
                        else:
                            if progress_cb:
                                progress_cb(f"  ⚠️ [{i}] Audio ma'lumot topilmadi, o'tkazildi")
                            continue
                    elif isinstance(audio_data, str) and os.path.exists(audio_data):
                        _convert_to_wav(audio_data, out_path)
                    elif isinstance(audio_data, bytes):
                        tmp = out_path + ".tmp"
                        with open(tmp, "wb") as f:
                            f.write(audio_data)
                        _convert_to_wav(tmp, out_path)
                        try: os.remove(tmp)
                        except: pass
                    else:
                        continue

                result = {
                    "file":       out_path,
                    "source":     "huggingface",
                    "source_url": f"{dataset_name}/{split}/{i}",
                    "file_name":  fname,
                }

                if text_column and text_column in item:
                    result["reference_text"] = item[text_column]

                yield result
                count += 1

            except Exception as e:
                if progress_cb:
                    progress_cb(f"  ❌ [{i}] Xato: {e}")
          # Loop tugadi — chiqamiz
          break

        except Exception as loop_err:
            err_msg = str(loop_err)
            if "client has been closed" in err_msg or "Connection" in err_msg or "timeout" in err_msg.lower():
                retry_count += 1
                if retry_count <= max_retries:
                    if progress_cb:
                        progress_cb(f"⚠️ Ulanish uzildi, qayta urinish {retry_count}/{max_retries}...")
                    import time as _time
                    _time.sleep(3)
                    # Datasetni qayta yuklaymiz
                    try:
                        ds = load_dataset(dataset_name, **load_kwargs)
                        try:
                            ds = ds.cast_column(audio_column, Audio(decode=False))
                        except Exception:
                            pass
                    except Exception:
                        pass
                else:
                    if progress_cb:
                        progress_cb(f"⚠️ Ulanish muammosi. {count} ta yuklanib bo'lindi.")
                    break
            else:
                raise


# ════════════════════════════════════════════════════════════════
# YORDAMCHI
# ════════════════════════════════════════════════════════════════

def _convert_to_wav(src: str, dst: str):
    """ffmpeg orqali istalgan audio formatni WAV 16kHz mono ga o'tkazadi."""
    cmd = [
        "ffmpeg", "-y", "-i", src,
        "-ar", str(SAMPLE_RATE),
        "-ac", "1",
        "-f", "wav",
        dst
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg xato: {result.stderr.decode()[:200]}")
