"""
gemini_utils.py
─────────────────────────────────────────────────────────────────
Gemini bilan ishlash:
  - Yo'l 1: Matn tozalash (muharrir)
  - Yo'l 2: Audio STT
  - Yo'l 2: Sifat baholash (1-5)
"""

import os
import base64
from google import genai

from config import GEMINI_API_KEY, GEMINI_MODEL_STT, GEMINI_MODEL_POLISH

_client = None

def get_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


# ════════════════════════════════════════════════════════════════
# YO'L 1 — MATN TOZALASH
# ════════════════════════════════════════════════════════════════

POLISH_PROMPT = """Sen O'zbek tili transkripsiya muharririsan.
Quyidagi xom transkripsiyani to'g'irla:

1. Yozma sonlar → raqam: "bir ming besh yuz" → "1500"
2. Telefon: "nol nol etti besh" → "+998 75..."
3. Kompaniya nomlari to'g'ri imlo
4. Grammatik xatolar va tushirib qoldirilgan so'zlar
5. Tinish belgilari qo'y
6. Agar to'g'ri bo'lsa — AYNAN qaytар

Faqat tozalangan matnni yaz. Izoh yo'q."""


def polish_text(raw_text: str,
                normalize_numbers: bool = True,
                fix_spelling: bool = True,
                fix_punctuation: bool = True,
                transliterate_ru: bool = False) -> str:
    """Yo'l 1 uchun: STT natijasini Gemini orqali tozalaydi."""
    prompt_parts = ["Sen O'zbek tili transkripsiya muharririsan.\nQuyidagi xom transkripsiyani to'g'irla:\n"]

    if normalize_numbers:
        prompt_parts.append("- Yozma sonlar → raqam: 'bir ming' → '1000'")
    if fix_spelling:
        prompt_parts.append("- Imlo xatolarni tuzat")
    if fix_punctuation:
        prompt_parts.append("- Tinish belgilarini qo'y")
    if transliterate_ru:
        prompt_parts.append("- Rus so'zlarini o'zbek transliteratsiyasiga o'tkaz")

    prompt_parts.append("\nFaqat tozalangan matnni yaz. Izoh yo'q.\n\nMatn: " + raw_text)

    response = get_client().models.generate_content(
        model=GEMINI_MODEL_POLISH,
        contents="\n".join(prompt_parts)
    )
    return response.text.strip()


# ════════════════════════════════════════════════════════════════
# YO'L 2 — GEMINI AUDIO STT
# ════════════════════════════════════════════════════════════════

def _build_stt_prompt(filters: dict) -> str:
    """Filter togglelaridan dinamik Gemini STT prompti yaratadi."""
    lines = [
        "Bu audio faylni o'zbek tilida transkripsiya qil.",
        "Faqat aytilgan so'zlarni yoz — izoh, tushuntirish yo'q.",
    ]
    if filters.get("filter_no_translate", True):
        lines.append("Tarjima qilma — audiodagi tildagidek yoz.")
    if filters.get("filter_capitalize", True):
        lines.append("Har bir gapning birinchi harfini katta harf bilan yoz.")
    if filters.get("filter_num_to_words", True):
        lines.append("Raqamlarni so'z bilan yoz (masalan: 5 → besh, 100 → yuz).")
    if filters.get("filter_latin_only", True):
        lines.append("Faqat lotin yozuvida yoz — kirill harflarni ishlatma.")
    if filters.get("filter_no_noise_tags", True):
        lines.append("Kulgu, shovqin, musiqa kabi tavsiflovchi teglarni yozma.")
    if filters.get("filter_no_repeat_prompt", True):
        lines.append("Bu ko'rsatmalar matnini transcription ichiga yozma.")
    if filters.get("filter_background_music", True):
        lines.append("Agar orqa fonda musiqa yoki kuchli fon shovqini bor bo'lsa — matn o'rniga faqat [MUSIQA_BOR] deb yoz.")
    if filters.get("filter_multiple_speakers", True):
        lines.append("Agar bir nechta kishi gapirsa — matn o'rniga faqat [KO'P_OVOZ] deb yoz.")
    lines.append("Agar audio tushunarsiz yoki haddan tashqari shovqinli bo'lsa — [TUSHUNARSIZ] deb yoz.")
    return "\n".join(lines)


_KNOWN_TAGS = ("MUSIQA_BOR", "KO'P_OVOZ", "TUSHUNARSIZ")


def _extract_filter_tags(text: str) -> list:
    """Gemini javobidan maxsus teglarni ajratib oladi."""
    tags = []
    for tag in _KNOWN_TAGS:
        if f"[{tag}]" in text:
            tags.append(tag)
    return tags


def transcribe_audio_gemini(file_path: str, filters: dict = None) -> tuple:
    """Yo'l 2 uchun: Audio faylni Gemini orqali transkripsiya qiladi.

    Returns: (text: str, filter_tags: list[str])
      text        — tozalangan transcription matni
      filter_tags — aniqlangan maxsus teglar (MUSIQA_BOR / KO'P_OVOZ / TUSHUNARSIZ)
    """
    filters = filters or {}
    prompt = _build_stt_prompt(filters)

    with open(file_path, "rb") as f:
        audio_bytes = f.read()
    audio_b64 = base64.b64encode(audio_bytes).decode()

    response = get_client().models.generate_content(
        model=GEMINI_MODEL_STT,
        contents=[{
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": "audio/wav", "data": audio_b64}}
            ]
        }]
    )
    text = response.text.strip()
    filter_tags = _extract_filter_tags(text)
    return text, filter_tags


# ════════════════════════════════════════════════════════════════
# YO'L 2 — SIFAT BAHOLASH (1-5)
# ════════════════════════════════════════════════════════════════

SCORE_PROMPT = """Quyidagi transkripsiyani baholab, FAQAT bitta raqam yoz (1-5):

5 — Mukammal: to'liq, aniq, grammatik to'g'ri
4 — Yaxshi: kichik xatolar bor, lekin tushunarli
3 — O'rtacha: ba'zi so'zlar noaniq yoki tushib qolgan
2 — Yomon: ko'p xato, qisman tushunarli
1 — Yaroqsiz: tushunarsiz, juda ko'p xato

Faqat raqam yoz, boshqa hech narsa yozma.

Transkripsiya: """


def score_transcription(text: str) -> int:
    """Yo'l 2 uchun: Gemini o'z transkripsiyasini baholaydi (1-5)."""
    try:
        response = get_client().models.generate_content(
            model=GEMINI_MODEL_POLISH,
            contents=SCORE_PROMPT + text
        )
        score_text = response.text.strip()
        score = int(score_text[0])
        return max(1, min(5, score))
    except Exception:
        return 3   # default: o'rtacha


def determine_status_v2(score: int,
                         auto_approve_min: int = 4,
                         auto_reject_max: int = 2) -> str:
    """Sifat bahosiga qarab status belgilaydi."""
    if score >= auto_approve_min:
        return "approved"
    elif score <= auto_reject_max:
        return "rejected"
    else:
        return "pending"
