"""
stt_v1.py — MohirDev API async pipeline
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Yo'l 1: MohirDev STT + lokal matn normalizatsiya.

GEMINI CHAQIRILMAYDI — to'liq offline.
Yagona tashqi chaqiruv: MohirDev STT API (audio → raw text).

Pipeline oqimi:
  Audio → filter_audio → MohirDev STT → filter_text_v1
        → polish_text_local → _is_valid_transcription → status
"""

import os
import re
import time
import requests
from pathlib import Path

from config import (
    STT_API_KEY, STT_LANGUAGE,
    V1_SNR_MIN, V1_DURATION_MIN, V1_DURATION_MAX,
    V1_SILENCE_MAX, V1_WORD_MIN, V1_REPEAT_MAX,
)
from filter_audio import filter_audio
from filter_text import filter_text_v1


# ════════════════════════════════════════════════════════════════
# AISHA STT API
# ════════════════════════════════════════════════════════════════

STT_POST_URL  = "https://back.aisha.group/api/v2/stt/post/"
STT_GET_URL   = "https://back.aisha.group/api/v2/stt/get/"
POLL_INTERVAL = 3
POLL_TIMEOUT  = 300


def submit_audio(file_path: str, api_key: str) -> str:
    headers = {"x-api-key": api_key}
    fname   = Path(file_path).name
    with open(file_path, "rb") as f:
        files = {"audio": (fname, f, "audio/wav")}
        data  = {"title": fname, "has_diarization": "false", "language": STT_LANGUAGE}
        r = requests.post(STT_POST_URL, headers=headers,
                          files=files, data=data, timeout=60)
    r.raise_for_status()
    result = r.json()
    job_id = result.get("id") or result.get("job_id") or result.get("task_id") or ""
    if not job_id:
        raise ValueError(f"job_id topilmadi: {result}")
    return str(job_id)


def poll_result(job_id: str, api_key: str) -> str:
    headers = {"x-api-key": api_key}
    url     = f"{STT_GET_URL}{job_id}/"
    elapsed = 0
    while elapsed < POLL_TIMEOUT:
        time.sleep(POLL_INTERVAL)
        elapsed += POLL_INTERVAL
        r    = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        data = r.json()
        status = str(data.get("status") or "").upper()
        if status == "SUCCESS":
            text = (data.get("transcript") or
                    data.get("text") or
                    data.get("transcription") or "")
            return str(text).strip()
        if status in ("FAILED", "ERROR"):
            raise ValueError(f"MohirDev API xato: {data}")
    raise TimeoutError(f"job_id={job_id} {POLL_TIMEOUT}s da natija kelmadi")


def call_stt_api(file_path: str, api_key: str = "") -> str:
    key = api_key or STT_API_KEY
    if not key:
        raise ValueError("MohirDev API key kiritilmagan!")
    return poll_result(submit_audio(file_path, key), key)


# ════════════════════════════════════════════════════════════════
# PLACEHOLDER TIZIMI
# Random salt: placeholder matnda tasodifiy uchrab qolishdan himoya qiladi.
# os.urandom(6) → 12 hex belgi → 16^12 ≈ 281 trillion kombinatsiya.
# Null-byte yondashuvidan farqli: encoding-xavfsiz, JSON-xavfsiz.
# ════════════════════════════════════════════════════════════════

_PH_SALT = os.urandom(6).hex()   # modul yuklanishida bir marta, keyin o'zgarmaydi


def _ph(idx: int) -> str:
    """Collision-proof placeholder: __<salt><04d_idx>__"""
    return f"__{_PH_SALT}{idx:04d}__"


# ════════════════════════════════════════════════════════════════
# PRE-COMPILED PATTERNS  (modul darajasida — batch uchun bir marta)
# ════════════════════════════════════════════════════════════════

# ARTIFACTS
_RE_JSON_ARTIFACT = re.compile(
    r"\{['\"]transcription['\"]\s*:\s*['\"](.+?)['\"]\}"
    r"|\{[^}]{0,200}\}",
    re.DOTALL,
)
_RE_STT_TAGS = re.compile(
    r"\[\s*(?:MUSIQA_BOR|KO[''']P_OVOZ|TUSHUNARSIZ|SHOVQIN|GUDOK|MUSIC|NOISE)\s*\]",
    re.IGNORECASE,
)
_RE_HTML = re.compile(r"</?[a-zA-Z][^>]{0,60}>")

# FILLERS — inglizcha
_RE_EN_FILLER = re.compile(
    r"(?<!\w)(mm+|uh+|ah+|eh+|um+|hmm+|hm+|ugh+|er+|eeh+)(?!\w)", re.I
)
# FILLERS — cho'zilgan unlilar: "aaaa", "eeee", "oooo", "iiii", "uuuu"
# STT ko'pincha tutilish paytidagi tovushlarni shu tarzda transkriptsiya qiladi.
# Shart: kamida 3 ta bir xil unli ketma-ket VA so'z chegarasida.
_RE_ELONGATED_VOWEL = re.compile(
    r"(?<!\w)([aeiouyAEIOUY])\1{2,}(?!\w)", re.I
)
# FILLERS — unli + cho'zilgan h: "ahhh", "ohhhh", "ehhh"
_RE_ELONGATED_MIXED = re.compile(
    r"(?<!\w)[aeiouyAEIOUY]h{3,}(?!\w)", re.I
)
# FILLERS — o'zbekcha
_RE_UZ_FILLER = re.compile(
    r"(?<!\w)(he-?he|ha-?ha|i-?i|o-?o|a-?a)(?!\w)"
    r"|(?<![a-zA-Z''])xo'sh(?!\w)",
    re.I,
)
# FILLERS — o'zbek nutqida uchraydigan rus so'zlari
_RE_RU_FILLER = re.compile(
    r"(?<!\w)(ну|вот|эм+|э-э+|короче|значит|ладно)(?!\w)", re.I
)
# STUTTER: bir xil so'zning 3+ ketma-ket takrori ("bu bu bu" → "bu")
_RE_STUTTER = re.compile(r"\b(\w{2,})(?:\s+\1){2,}\b", re.I)

# FORMATTING
_RE_MULTI_SPACE  = re.compile(r" {2,}")
_RE_MULTI_COMMA  = re.compile(r",{2,}")
_RE_BROKEN_HYP   = re.compile(r"(\w)-\s+(\w)")
_RE_SENT_CASE    = re.compile(r"([.!?]\s+)([a-zA-Z])")

# VALIDATION
_RE_ONLY_DIGITS   = re.compile(r"^[\d\s.,]+$")
_RE_ONLY_SYMBOLS  = re.compile(r"^[\W\s]+$")
_RE_CYRILLIC      = re.compile(r"[а-яёА-ЯЁ]")
_RE_LATIN_ALPHA   = re.compile(r"[a-zA-Z]")
_RE_STRIP_PUNCT   = re.compile(r"[.,!?;:'\"«»\-]+")

# NUMBER — saqlanishi kerak bo'lgan formatlar
_RE_PHONE     = re.compile(r"\+?\d[\d\s\-()]{6,}\d")
_RE_DECIMAL   = re.compile(r"\b\d+[.,]\d+\b")
_RE_DATE_FULL = re.compile(r"\b\d{1,2}[.\-/]\d{1,2}[.\-/]\d{2,4}\b")
_RE_TIME_FMT  = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")
_RE_YEAR_FMT  = re.compile(r"\b(?:1[5-9]\d{2}|20[0-4]\d)\b")
_RE_ORDINAL   = re.compile(r"\b(\d+)-(chi|nchi|inchi|unchi|inci)\b", re.I)


# ════════════════════════════════════════════════════════════════
# O'ZBEK SINONIM NORMALIZATSIYASI (faqat change_ratio uchun)
# ════════════════════════════════════════════════════════════════
# Bu lug'at FAQAT _compute_local_change_ratio ichida ishlatiladi.
# Maqsad: bir xil ma'noli, biroq boshqacha yozilgan so'zlarni
# taqqoslashda "o'zgarish" deb hisoblamaslik.
#
# Qoidalar:
#   - Faqat bir tomonlama (canonical → variant emas, variant → canonical)
#   - Faqat ma'no farq qilmaydigan juftlar
#   - Matnning o'zini o'zgartirmaydi (faqat token setini hisoblashda)
_RATIO_SYNONYMS: dict[str, str] = {
    # O'zbek dialekt: h / x almashinishi
    "xam":    "ham",
    "xar":    "har",
    "xech":   "hech",
    "xamma":  "hamma",
    "xali":   "hali",
    # Qisqartirilgan va to'liq shakl
    "yani":   "ya'ni",
    "dema":   "demak",
    "keyin":  "keyin",    # o'zi canonical, lekin "kyin" variant
    "kyin":   "keyin",
    # Raqam so'z ekvivalentlari (raqamlar allaqachon _num_words_uz orqali normallanadi,
    # bu yerda qo'shimcha xavfsizlik sifatida)
    "nul":    "nol",
    "zero":   "nol",
    # Umumiy affirmativ variantlar
    "xop":    "xo'p",
    "hop":    "xo'p",
    "mayle":  "mayli",
    "mayo":   "mayli",
    # STT tez-tez xato qiladigan juftlar
    "bilann":  "bilan",
    "uchun":   "uchun",   # canonical
    "uchin":   "uchun",
    "uchinn":  "uchun",
}


# ════════════════════════════════════════════════════════════════
# RAQAM → SO'Z
# ════════════════════════════════════════════════════════════════

_ORDINAL_MAP = {
    1: "birinchi",  2: "ikkinchi",   3: "uchinchi",
    4: "to'rtinchi", 5: "beshinchi", 6: "oltinchi",
    7: "yettinchi", 8: "sakkizinchi", 9: "to'qqizinchi",
    10: "o'ninchi",
}

_NUM_ONES  = ["", "bir", "ikki", "uch", "to'rt", "besh", "olti", "yetti", "sakkiz", "to'qqiz"]
_NUM_TEENS = ["o'n", "o'n bir", "o'n ikki", "o'n uch", "o'n to'rt", "o'n besh",
              "o'n olti", "o'n yetti", "o'n sakkiz", "o'n to'qqiz"]
_NUM_TENS  = ["", "o'n", "yigirma", "o'ttiz", "qirq", "ellik",
              "oltmish", "yetmish", "sakson", "to'qson"]


def _int_to_uz(n: int) -> str:
    if n == 0:  return "nol"
    if n < 0:   return "minus " + _int_to_uz(-n)
    if n < 10:  return _NUM_ONES[n]
    if n < 20:  return _NUM_TEENS[n - 10]
    if n < 100: return _NUM_TENS[n // 10] + ("" if n % 10 == 0 else " " + _NUM_ONES[n % 10])
    if n < 1_000:
        r = n % 100
        return _NUM_ONES[n // 100] + " yuz" + ("" if r == 0 else " " + _int_to_uz(r))
    if n < 1_000_000:
        r = n % 1000
        return _int_to_uz(n // 1000) + " ming" + ("" if r == 0 else " " + _int_to_uz(r))
    if n < 1_000_000_000:
        r = n % 1_000_000
        return _int_to_uz(n // 1_000_000) + " million" + ("" if r == 0 else " " + _int_to_uz(r))
    return str(n)


def _num_words_uz(text: str) -> str:
    """
    Matndagi sonlarni o'zbek so'ziga aylantiradi.

    Saqlanadi: telefon, kasr (5.5), sana (14.03.2024), vaqt (14:30), yil (1990–2049).
    Konvertatsiya: tartib sonlar (1-chi → birinchi), oddiy butun sonlar.

    Placeholder tizimi: random salt bilan collision-proof (_ph funksiyasi).
    """
    protected: dict[str, str] = {}
    idx = 0

    def _save(m: re.Match) -> str:
        nonlocal idx
        key = _ph(idx)
        protected[key] = m.group()
        idx += 1
        return key

    # Muhim tartib: aniqroq naqsh avval saqlansin
    t = _RE_PHONE.sub(_save, text)
    t = _RE_DATE_FULL.sub(_save, t)
    t = _RE_TIME_FMT.sub(_save, t)
    t = _RE_YEAR_FMT.sub(_save, t)
    t = _RE_DECIMAL.sub(_save, t)

    def _repl_ord(m: re.Match) -> str:
        n = int(m.group(1))
        return _ORDINAL_MAP.get(n, _int_to_uz(n) + "-nchi")

    t = _RE_ORDINAL.sub(_repl_ord, t)

    def _repl_int(m: re.Match) -> str:
        try:
            n = int(m.group())
            return m.group() if n > 999_999_999 else _int_to_uz(n)
        except ValueError:
            return m.group()

    t = re.sub(r"\b\d+\b", _repl_int, t)

    for key, val in protected.items():
        t = t.replace(key, val)

    return t


# ════════════════════════════════════════════════════════════════
# DEDUPLICATION: LCS + CONTIGUOUS RUN
# ════════════════════════════════════════════════════════════════

def _lcs_ratio(a: list, b: list) -> float:
    """
    Longest Common Subsequence / max(len(a), len(b)).
    O(m*n) lekin m, n <= 20 — batch uchun amaliy tez.
    """
    m, n = len(a), len(b)
    if not m or not n:
        return 0.0
    dp = [0] * (n + 1)
    for ai in a:
        prev = 0
        for j, bj in enumerate(b):
            temp      = dp[j + 1]
            dp[j + 1] = prev + 1 if ai == bj else max(dp[j + 1], dp[j])
            prev      = temp
    return dp[n] / max(m, n)


def _run_ratio(a: list, b: list) -> float:
    """
    Longest Common CONTIGUOUS run / max(len(a), len(b)).

    LCS ga qo'shimcha: bir xil so'zlar TAR TI BDA ham bir joyda
    kelishi kerakligini tekshiradi.

    Misol:
      a = [A, B, C, D],  b = [D, C, B, A]
      LCS  = 2/4 = 0.50  (A va D har ikkalasida bor)
      Run  = 1/4 = 0.25  (eng uzun ketma-ket mos = 1)
      → Run tekshiruvi to'liq teskari tartibni aniqroq rad etadi.
    """
    m, n = len(a), len(b)
    if not m or not n:
        return 0.0
    prev = [0] * (n + 1)
    best = 0
    for i in range(m):
        curr = [0] * (n + 1)
        for j in range(n):
            if a[i] == b[j]:
                curr[j + 1] = prev[j] + 1
                if curr[j + 1] > best:
                    best = curr[j + 1]
        prev = curr
    return best / max(m, n)


def _dedup_half(
    text: str,
    jaccard_threshold: float = 0.82,
    lcs_threshold:     float = 0.70,
    run_threshold:     float = 0.40,
) -> str:
    """
    Takrorlangan ikkinchi yarmni aniqlaydi.

    UCHTA shart bir vaqtda o'tilgandagina o'chirish amalga oshadi:
      1. Jaccard >= 0.82  — ikki yarmda bir xil so'zlar mavjud
      2. LCS    >= 0.70   — so'zlar umumiy tartibini saqlaydi
      3. Run    >= 0.40   — kamida 2/5 ta so'z ketma-ket moslik beradi

    Uchta shart kombinatsiyasi quyidagi edge-caseni hal qiladi:
      "A B C D E. A B C D E." → J=1.0 LCS=1.0 Run=1.0 → o'chirish ✓
      "A B C D E. E D C B A." → J=1.0 LCS=0.2 Run=0.2 → saqlash ✓
      "A B C D E. A B E D C." → J=1.0 LCS=0.6 Run=0.4 → chegara → Run hal qiladi
    """
    words = text.split()
    half  = len(words) // 2
    if half < 4:
        return text

    def _clean(w: str) -> str:
        return _RE_STRIP_PUNCT.sub("", w).lower()

    first_c  = [_clean(w) for w in words[:half] if _clean(w)]
    second_c = [_clean(w) for w in words[half:] if _clean(w)]
    if not first_c or not second_c:
        return text

    f_set   = set(first_c)
    s_set   = set(second_c)
    union   = f_set | s_set
    jaccard = len(f_set & s_set) / len(union) if union else 0.0
    if jaccard < jaccard_threshold:
        return text

    if _lcs_ratio(first_c, second_c) < lcs_threshold:
        return text

    if _run_ratio(first_c, second_c) < run_threshold:
        return text

    return " ".join(words[:half]).strip()


# ════════════════════════════════════════════════════════════════
# ARALASH TIL ANIQLASH (so'z darajasida)
# ════════════════════════════════════════════════════════════════

def _cyrillic_word_ratio(text: str) -> float:
    """
    Kirill-dominant so'zlar ulushini hisoblaydi.

    Belgi-darajasidagi nisbatdan ustunligi:
      Belgi-daraja: "Moskva" → 6 kirill / 6 alpha = 100% → rad etadi (noto'g'ri)
      So'z-daraja:  "Moskva" → 1 kirill so'z / 4 umumiy so'z = 25% → qabul qiladi

    So'z "kirill-dominant" hisoblanadi agar uning alfanumerik
    belgilarining > 50% i kirill bo'lsa.
    """
    words = text.split()
    if not words:
        return 0.0
    cyr_dominant = 0
    for w in words:
        alpha = [c for c in w if c.isalpha()]
        if not alpha:
            continue
        cyr = sum(1 for c in alpha if _RE_CYRILLIC.match(c))
        if cyr / len(alpha) > 0.5:
            cyr_dominant += 1
    return cyr_dominant / len(words)


# ════════════════════════════════════════════════════════════════
# NORMALIZATSIYA ASOSIY FUNKSIYA
# ════════════════════════════════════════════════════════════════

def polish_text_local(
    text: str,
    normalize_numbers: bool = True,
    fix_punctuation:   bool = True,
    fix_apostrophe:    bool = True,
    remove_artifacts:  bool = True,
) -> str:
    """
    Yo'l 1 uchun lokal matn normalizatsiya — Gemini yo'q, API yo'q.

      1.  JSON artifact va STT teglari
      2.  HTML teglari
      3.  Apostrofni standartlashtirish
      4.  Inglizcha filler
      5.  Cho'zilgan unlilar:  aaaa, eeee, ahhhhh
      6.  O'zbekcha filler
      7.  Rus filler
      8.  Stutter (3+)
      9.  Takrorlangan yarmi  (Jaccard + LCS + Run)
     10.  Ko'p vergul + singan tira
     11.  Qo'sh bo'shliqlar
     12.  Raqam → so'z
     13.  Birinchi harf katta
     14.  Gap ichida katta harf
    """
    if not text or not text.strip():
        return text

    t = text.strip()

    if remove_artifacts:
        def _json_repl(m: re.Match) -> str:
            return m.group(1).strip() if m.group(1) else ""
        t = _RE_JSON_ARTIFACT.sub(_json_repl, t).strip()
        t = _RE_STT_TAGS.sub("", t).strip()
        t = _RE_HTML.sub("", t).strip()

    if fix_apostrophe:
        for src, dst in [("ʻ","'"), ("ʼ","'"), ("'","'"),
                         ("'","'"), ("`","'"), ("´","'")]:
            t = t.replace(src, dst)

    # Filler va cho'zilgan tovushlar
    t = _RE_EN_FILLER.sub("", t)
    t = _RE_ELONGATED_VOWEL.sub("", t)     # aaaa, eeee, oooo
    t = _RE_ELONGATED_MIXED.sub("", t)     # ahhhhh, ohhhh
    t = _RE_UZ_FILLER.sub("", t)
    t = _RE_RU_FILLER.sub("", t)
    t = _RE_STUTTER.sub(r"\1", t)

    t = _dedup_half(t)

    t = _RE_MULTI_COMMA.sub(",", t)
    t = _RE_BROKEN_HYP.sub(r"\1-\2", t)
    t = _RE_MULTI_SPACE.sub(" ", t).strip()

    if normalize_numbers:
        t = _num_words_uz(t)

    if fix_punctuation and t:
        t = t[0].upper() + t[1:]

    if fix_punctuation:
        t = _RE_SENT_CASE.sub(lambda m: m.group(1) + m.group(2).upper(), t)

    return t.strip()


# ════════════════════════════════════════════════════════════════
# VALIDATSIYA
# ════════════════════════════════════════════════════════════════

def _is_valid_transcription(
    text: str,
    word_min:       int  = 3,
    check_cyrillic: bool = False,
) -> tuple[bool, str]:
    """
    Post-normalizatsiya matn sifatini tekshiradi.

    So'z soni istisno:
      "Ha.", "Yo'q.", "Xo'p, mayli." → ruxsat (>= 4 harf + terminal)

    Kirill tekshiruvi — so'z darajasida:
      Belgi-darajasidagi nisbat o'rniga so'z-darajasidagi nisbat ishlatiladi.
      Sabab: "Moskva", "Putin" kabi rus otlari o'zbek matnida normal.
      Rad etish chegarasi: kirill-dominant so'zlar >= 30%.
    """
    if not text or not text.strip():
        return False, "bosh matn"

    t     = text.strip()
    words = t.split()
    n     = len(words)

    if n < word_min:
        clean_chars = len(re.sub(r"[\s.,!?;:\-]", "", t))
        if clean_chars >= 4 and t[-1] in ".!?":
            pass
        else:
            return False, f"kam so'z ({n} < {word_min})"

    if _RE_ONLY_DIGITS.fullmatch(t):
        return False, "faqat raqam"

    if _RE_ONLY_SYMBOLS.fullmatch(t):
        return False, "faqat belgilar"

    if check_cyrillic:
        cyr_ratio = _cyrillic_word_ratio(t)
        if cyr_ratio >= 0.30:
            return False, f"kirill so'zlar ko'p ({cyr_ratio:.0%})"

    return True, "ok"


# ════════════════════════════════════════════════════════════════
# O'ZGARISH NISBATI
# ════════════════════════════════════════════════════════════════

def _compute_local_change_ratio(original: str, polished: str) -> float:
    """
    Sinonim va raqam ekvivalentlarini hisobga oladigan o'zgarish nisbati.

    Normalizatsiya qadamlari (taqqoslashdan oldin, matnni o'zgartirmasdan):
      1. _num_words_uz: "5" → "besh" → ikkala matnda "besh" → 0% o'zgarish
      2. _RATIO_SYNONYMS: "xam" → "ham" → dialekt farqi hisoblanmaydi
      3. Tinish belgilari striplash: "dunyo." == "dunyo"
      4. Kichik harfga o'tkazish
    """
    def _normalize_tokens(text: str) -> set:
        normed = _num_words_uz(text)
        tokens = set()
        for w in normed.split():
            cleaned = _RE_STRIP_PUNCT.sub("", w).lower()
            if cleaned:
                tokens.add(_RATIO_SYNONYMS.get(cleaned, cleaned))
        return tokens

    orig_tok     = _normalize_tokens(original)
    polished_tok = _normalize_tokens(polished)

    if not orig_tok:
        return 100.0

    changed = orig_tok.symmetric_difference(polished_tok)
    denom   = max(len(orig_tok), len(polished_tok), 1)

    return round(len(changed) / denom * 100.0, 1)


# ════════════════════════════════════════════════════════════════
# Yo'l 1 ASOSIY FUNKSIYA
# ════════════════════════════════════════════════════════════════

def process_segment_v1(
    segment:     dict,
    filters:     dict,
    polish_opts: dict,
    api_key:     str = "",
) -> dict:
    """
    Yo'l 1: MohirDev STT + lokal normalizatsiya.

    ╔══════════════════════════════════════════════╗
    ║  GEMINI CHAQIRILMAYDI — to'liq offline       ║
    ║  Yagona tashqi: MohirDev STT HTTP so'rovi        ║
    ╚══════════════════════════════════════════════╝

    Status mantigi:
      change_ratio < 25%  → approved
      change_ratio 25–50% → pending  (qo'l tekshiruv tavsiya)
      change_ratio >= 50% → pending  (original matn saqlanadi)
    """
    result = {**segment, "pipeline": "aisha_stt", "status": "pending"}

    # ── 0. Lokal audio kontent tahlili — teglashtirish, filtrlash emas ─
    # Musiqa yoki ko'p ovoz aniqlansa segment O'CHIRILMAYDI.
    # Natija metadataga flag sifatida qo'shiladi va transcription davom etadi.
    if filters.get("filter_background_music", True):
        from filter_audio import detect_music
        result["has_background_music"] = detect_music(segment["file"])

    if filters.get("filter_multiple_speakers", True):
        from filter_audio import detect_multiple_speakers
        result["has_multiple_speakers"] = detect_multiple_speakers(segment["file"])

    ok, reason = filter_audio(
        segment,
        snr_min=filters.get("snr_min", V1_SNR_MIN),
        duration_min=filters.get("duration_min", V1_DURATION_MIN),
        duration_max=filters.get("duration_max", V1_DURATION_MAX),
        silence_max=filters.get("silence_max", V1_SILENCE_MAX),
        check_language=filters.get("check_language", False),
    )
    result["snr_score"]     = segment.get("snr_score")
    result["silence_ratio"] = segment.get("silence_ratio")

    if not ok:
        return {**result, "status": "filtered", "reason": f"audio: {reason}"}

    try:
        raw_text = call_stt_api(segment["file"], api_key=api_key)
    except Exception as e:
        return {**result, "status": "filtered", "reason": f"STT xato: {e}"}

    if not raw_text:
        return {**result, "status": "filtered", "reason": "STT bo'sh natija"}

    result["original_text"] = raw_text

    ok, reason = filter_text_v1(
        raw_text,
        word_min=filters.get("word_min", V1_WORD_MIN),
        repeat_max=filters.get("repeat_max", V1_REPEAT_MAX),
        check_noise=filters.get("check_noise", True),
        check_mixed=filters.get("check_mixed", False),
    )
    if not ok:
        return {**result, "status": "filtered", "reason": f"matn: {reason}"}

    polished = polish_text_local(
        raw_text,
        normalize_numbers=polish_opts.get("normalize_numbers", True),
        fix_punctuation=polish_opts.get("fix_punctuation", True),
        fix_apostrophe=True,
        remove_artifacts=True,
    )

    if not polished or not polished.strip():
        polished = raw_text

    valid, reason = _is_valid_transcription(
        polished,
        word_min=filters.get("word_min", V1_WORD_MIN),
        check_cyrillic=filters.get("check_mixed", False),
    )
    if not valid:
        return {**result, "status": "filtered", "reason": f"post-norm: {reason}"}

    result["transcription"] = polished

    change_pct = _compute_local_change_ratio(raw_text, polished)
    result["change_ratio"] = change_pct

    if change_pct < 25.0:
        result["status"] = "approved"
    elif change_pct < 50.0:
        result["status"] = "pending"
    else:
        result["transcription"] = raw_text
        result["status"]        = "pending"
        result["reason"]        = f"normalizatsiya ko'p o'zgartirdi ({change_pct:.0f}%)"

    return result
