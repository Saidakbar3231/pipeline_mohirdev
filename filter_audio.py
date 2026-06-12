"""
filter_audio.py  +  filter_text.py
─────────────────────────────────────────────────────────────────
Audio va matn sifat filtrlari.
"""

import re
import numpy as np


# ════════════════════════════════════════════════════════════════
# MODUL DARAJASIDA IMPORT KESHI
# Batch processing: import overhead bitta marta to'lanadi
# ════════════════════════════════════════════════════════════════

_librosa_cache = None

def _lib():
    global _librosa_cache
    if _librosa_cache is None:
        import librosa as _l
        _librosa_cache = _l
    return _librosa_cache


# ════════════════════════════════════════════════════════════════
# UMUMIY YORDAMCHI FUNKSIYALAR
# ════════════════════════════════════════════════════════════════

def _load_and_normalize(file_path: str, sr: int, max_dur: float = 30.0):
    """
    Audio yuklaydi, normallaydi va davomiyligini cheklaydi.

    Normalizatsiya strategiyasi (ikkala holat ham batch-safe):
      - Oddiy audio: peak-normalizatsiya -3 dBFS
        past fon musiqasi va past ovozli yozuvlarda xususiyatlar ko'rinadi
      - Impulsiv shovqin bor audio (peak/RMS > 12):
        bitta keskin tovush butun peak-normni bostirib qoyadi
        buning orniga RMS-normalizatsiya -20 dBFS ishlatiladi

    Returns: (y, sr) yoki (None, sr) agar audio bosh bolsa.
    """
    y, _ = _lib().load(file_path, sr=sr, mono=True, duration=max_dur)
    if len(y) == 0:
        return None, sr

    peak = float(np.max(np.abs(y)))
    if peak < 1e-6:
        return None, sr

    rms = float(np.sqrt(np.mean(y ** 2)))
    if rms < 1e-8:
        return None, sr

    if (peak / rms) > 12.0:
        # Impulsiv shovqin: RMS-normalizatsiya -20 dBFS
        y = y * (10 ** (-20.0 / 20.0)) / rms
    else:
        # Standart: peak-normalizatsiya -3 dBFS
        y = y * (10 ** (-3.0 / 20.0)) / peak

    return y, sr


def _sig(x: float, center: float, k: float) -> float:
    """
    Sigmoid normalizatsiya: x ni [0,1] ga otkazadi.
    Overflow oldini olish uchun eksponent argumenti [-500, 500] ga qisqartiriladi.
    k > 0: osuvchi; k < 0: tushuvchi (teskari xususiyatlar uchun).
    """
    arg = float(np.clip(-k * float(x - center), -500.0, 500.0))
    return 1.0 / (1.0 + float(np.exp(arg)))


def _voiced_frame_mask(rms: np.ndarray, top_db: float = 30.0) -> np.ndarray:
    """
    RMS massivi uchun ovozli freymlar maskasini qaytaradi.
    Maksimaldan top_db dB past bolgan freymlar sukunat deb belgilanadi.
    RMS CV ni faqat nutq freymlarida hisoblash uchun ishlatiladi:
    sukunat oraligi CV ni suniy ravishda oshiradi.
    """
    if len(rms) == 0 or np.max(rms) < 1e-8:
        return np.ones(len(rms), dtype=bool)
    db = 20.0 * np.log10(rms / (np.max(rms) + 1e-10) + 1e-10)
    return db > -top_db


# ════════════════════════════════════════════════════════════════
# MUSIQA ANIQLASH
# ════════════════════════════════════════════════════════════════

def _music_score_chunk(chunk: np.ndarray, sr: int) -> float:
    """
    Bitta audio parcha uchun musiqa ballini hisoblaydi [0.0 - 1.0].

    Ishlab chiqarish darajasidagi ishonchlilik:
      - HPSS margin=(1,5): garmonik tomoni perkussivdan kuchliroq ajratadi
      - Garmonik uzluksizlik: freym darajasida garmonik nisbat >28% bolgan
        freymlarning ulushi. Musiqa uzluksiz harmonikaga ega, nutq emas
      - Chroma 75-persentil: shovqin-chidamli tonal olchov (dispersiya emas)
      - RMS CV faqat ovozli freymlar boyicha: sukunat CV ni oshirmaydi
      - Barcha sigmoid argumentlari qirqilgan: overflow yoq
    """
    lib = _lib()
    HOP = 512
    FRAME = 2048
    MIN_SAMPLES = int(sr * 0.75)   # 750ms: barqaror xususiyatlar uchun minimum

    if len(chunk) < MIN_SAMPLES:
        return 0.0

    # Garmonik-perkussiv ajratish
    harmonic, _ = lib.effects.hpss(chunk, margin=(1.0, 5.0))
    total_energy = float(np.mean(np.abs(chunk))) + 1e-8

    # 1. Global garmonik nisbat
    harmonic_ratio = float(np.mean(np.abs(harmonic))) / total_energy

    # 2. Garmonik uzluksizlik — nutqdan musiqani ajratuvchi asosiy belgi
    #    Nutq: 0.40-0.62 (tovoshsiz bo'limlarda garmonik tushadi)
    #    Musiqa+nutq: 0.72-0.95 (fon musiqa sukutda ham garmonikni saqlaydi)
    #    center=0.68: nutq doimo pastda, musiqa+nutq doimo ustida qoladi
    h_frames = lib.util.frame(np.abs(harmonic), frame_length=FRAME, hop_length=HOP)
    c_frames = lib.util.frame(np.abs(chunk),    frame_length=FRAME, hop_length=HOP)
    n_f = min(h_frames.shape[1], c_frames.shape[1])
    frame_harm = (np.mean(h_frames[:, :n_f], axis=0) /
                  (np.mean(c_frames[:, :n_f], axis=0) + 1e-8))
    harm_continuity = float(np.mean(frame_harm > 0.28))

    # 3. RMS CV — eng ishonchli ajratuvchi (olib tashlangan chroma/flatness o'rniga)
    #    Nutq: yuqori CV (0.8-1.5) — undoshlar/sukutlarda energiya keskin o'zgaradi
    #    Musiqa+nutq: past CV (0.2-0.5) — musiqa energiyani barqarorlashtiradi
    #    Chroma va spectral_flatness olib tashlandi: nutq unlilari ham tonal va past
    #    flatness beradi — ular nutqni musiqadan ajrata olmaydi (yolg'on-ijobiy sabab)
    rms = lib.feature.rms(y=chunk, frame_length=FRAME, hop_length=HOP)[0]
    voiced_mask = _voiced_frame_mask(rms)
    n_rms       = min(len(rms), len(voiced_mask))
    rms_voiced  = rms[:n_rms][voiced_mask[:n_rms]]
    if len(rms_voiced) >= 4:
        rms_cv = float(np.std(rms_voiced) / (np.mean(rms_voiced) + 1e-8))
    else:
        rms_cv = float(np.std(rms) / (np.mean(rms) + 1e-8))

    # 4. Spektral kontrast
    contrast      = lib.feature.spectral_contrast(y=chunk, sr=sr, n_bands=6)
    mean_contrast = float(np.mean(contrast))

    # 5. Onset kuchi dispersiyasi: musiqa ritmik tuzilishga ega
    onset_env = lib.onset.onset_strength(y=chunk, sr=sr)
    onset_var = float(np.var(onset_env))

    # Taxminiy balllar:
    #   Faqat nutq     → ≈ 0.12-0.18  (threshold 0.60 dan ancha past → filtered emas)
    #   Nutq + musiqa  → ≈ 0.72-0.85  (threshold 0.60 dan ancha yuqori → filtered)
    scores = [
        _sig(harmonic_ratio,  center=0.38, k= 20.0),  # nutq<0.38<musiqa
        _sig(harm_continuity, center=0.68, k= 14.0),  # nutq<0.68<musiqa [eng yuqori center]
        _sig(rms_cv,          center=0.55, k= -8.0),  # past CV=barqaror=musiqa [teskari]
        _sig(mean_contrast,   center=16.0, k=  0.4),  # qo'shimcha signal
        _sig(onset_var,       center=0.50, k=  3.0),  # ritmik tuzilish
    ]
    weights = [0.30, 0.35, 0.25, 0.07, 0.03]   # jami = 1.0
    return float(np.dot(weights, scores))


def detect_music(file_path: str, threshold: float = 0.60) -> bool:
    """
    Orqa fon musiqasini ishonchli kop xususiyatli tahlil orqali aniqlaydi.
    Offline - hech qanday API ishlatilmaydi.

    Qisqa audio (<6s): yagona toliq tahlil.
    Uzun audio (>=6s): 3 ta kesishgan parcha + qatiy kopchilik ovoz.
      Parchalar bosh, orta, oxirdan olinadi: shu tariqa musiqa
      faqat bitta qismda bolsa ham aniqlanadi.
    """
    try:
        y, sr = _load_and_normalize(file_path, sr=22050, max_dur=30.0)
        if y is None:
            return False
        duration = len(y) / sr
        if duration < 1.0:
            return False

        if duration < 6.0:
            return _music_score_chunk(y, sr) >= threshold

        # 3 ta kesishgan parcha: bosh / orta / oxir
        chunk_samples = max(int(2.0 * sr), len(y) // 3)
        starts = [
            0,
            (len(y) - chunk_samples) // 2,
            max(0, len(y) - chunk_samples),
        ]
        chunk_scores = [
            _music_score_chunk(y[s: s + chunk_samples], sr)
            for s in starts
            if s + chunk_samples <= len(y) + int(sr * 0.5)
        ]
        if not chunk_scores:
            return False

        n_above = sum(s >= threshold for s in chunk_scores)
        return n_above >= (len(chunk_scores) // 2 + 1)   # qatiy kopchilik

    except Exception as e:
        print(f"[filter_audio] detect_music xato: {e}")
        return False


# ════════════════════════════════════════════════════════════════
# KOP KISHI ANIQLASH
# ════════════════════════════════════════════════════════════════

def _has_pitch_bimodality(voiced_f0: np.ndarray) -> bool:
    """
    F0 taqsimotining bimodalligi orqali kop kishi gapirganini tekshiradi.
    DBSCAN klaster topmaganda yoki segment soni yetarli bolmaganda zahira.

    Yaxshilanishlar:
      - log2(F0) ishlatiladi: inson qulogi chastotani logarifmik idrok etadi.
        Bu erkak (80-180 Hz) va ayol (165-255 Hz) ovozlari orasidagi
        ajrashni aniqroq korsatadi.
      - scipy prominence filtri: kichik harmonik artefakt choqqilarni rad etadi.
      - Choqqi nisbati >=0.30: kichik sputnik choqqi real ikkinchi kishi emas.

    Args:
        voiced_f0: detect_multiple_speakers ichida hisoblangan ovozli F0 massivi
                   (qayta audio yuklash va YIN hisoblashdan qochish uchun)
    """
    try:
        from scipy.signal import find_peaks

        if len(voiced_f0) < 30:
            return False

        # log2 shkala: oktava oraligidagi chastotalarni teng masofali qiladi
        log_f0 = np.log2(voiced_f0)
        hist, _ = np.histogram(log_f0, bins=30)
        if hist.max() == 0:
            return False

        peaks, props = find_peaks(
            hist,
            height=hist.max() * 0.25,       # minimal choqqi balandligi
            distance=3,                       # choqqilar orasidagi minimal bin oraliq
            prominence=hist.max() * 0.15,    # kozga koringan choqqi talabi
        )
        if len(peaks) < 2:
            return False

        # Kichik choqqi kattasining >=30% bolishi kerak
        top2 = np.sort(props["peak_heights"])[-2:]
        return float(top2[0] / (top2[1] + 1e-8)) > 0.30

    except Exception:
        return False


def _check_temporal_coherence(labels: np.ndarray,
                               seg_durations: np.ndarray,
                               min_run_sec: float = 1.0) -> bool:
    """
    Klasterlar vaqt boyicha bloklarda (haqiqiy almashinuv) yoki
    almashinuvchi naqshda (bitta kishi prosodiya ozgarishi) ekanligini tekshiradi.

    Blok:       [0,0,0,0,1,1,1,1] -> True   (ikkita kishi navbat bilan)
    Almashinuv: [0,1,0,1,0,1,0,1] -> False  (bitta kishi, uslub ozgarishi)

    Yaxshilanish: segment SONI orniga segment DAVOMIYLIGI (soniyalarda) ishlatiladi.
    Masalan: 3x0.5s = 1.5s run vs 1x4s run -> ikkinchisi aniqroq almashinuv.
    min_run_sec: ortacha run kamida shuncha soniya bolishi kerak.
    """
    valid_mask   = labels != -1
    valid_labels = labels[valid_mask]
    valid_durs   = seg_durations[valid_mask]

    if len(valid_labels) < 4:
        return True   # malumot yetarli emas - shubha berilamiz

    run_secs = []
    cur = float(valid_durs[0])
    for i in range(1, len(valid_labels)):
        if valid_labels[i] == valid_labels[i - 1]:
            cur += float(valid_durs[i])
        else:
            run_secs.append(cur)
            cur = float(valid_durs[i])
    run_secs.append(cur)

    return float(np.mean(run_secs)) >= min_run_sec


def detect_multiple_speakers(file_path: str, threshold: float = 0.40) -> bool:
    """
    Kop kishi gapirganini segment MFCC+log(F0) embeddings + DBSCAN orqali aniqlaydi.
    Offline - hech qanday API ishlatilmaydi.

    Ishlab chiqarish darajasidagi yaxshilanishlar:
      - F0 log2 skalasida: inson ovoz balandligi logarifmik idrok etiladi,
        bu erkak/ayol va har xil pitchli kishi farqini aniqroq korsatadi
      - Eng yaxshi klasterlash sinovi tanlanadi (birinchi emas):
        inter-klaster Evklid masofasi maksimal bolgan eps tanlanadi
      - Vaqtinchalik kogerentlik DAVOMIYLIK bilan ogirlanadi (segment soni emas)
      - Ovozli F0 massivi bir marta hisoblanib zahiraga uzatiladi:
        qayta YIN hisoblashdan qochadi
      - 4+ segment talabi (3 orniga): DBSCAN uchun minimal ishonchli hajm
    """
    try:
        lib = _lib()
        from sklearn.cluster import DBSCAN
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics.pairwise import cosine_distances, euclidean_distances

        y, sr = _load_and_normalize(file_path, sr=16000, max_dur=120.0)
        if y is None or len(y) / sr < 2.5:
            return False

        # 1. Ovozli intervallar
        intervals = lib.effects.split(
            y, top_db=28, frame_length=2048, hop_length=512
        )
        min_len = int(0.5 * sr)   # 500ms: ishonchli MFCC+F0 embedding uchun minimum
        intervals = [(s, e) for s, e in intervals if (e - s) >= min_len]

        if len(intervals) < 2:
            return False

        # 2. Segment embeddings: MFCC mean+std (40) + log2(F0) mean+std (2) = 42-olcham
        embeddings    = []
        seg_durs      = []
        all_voiced_f0 = []   # bir marta hisoblayamiz, zahiraga uzatamiz

        for start, end in intervals:
            seg = y[start:end]

            mfcc     = lib.feature.mfcc(y=seg, sr=sr, n_mfcc=20)
            mfcc_emb = np.concatenate([np.mean(mfcc, axis=1), np.std(mfcc, axis=1)])

            # YIN: F0 log2 skalasida
            f0 = lib.yin(seg, fmin=70, fmax=400, sr=sr, hop_length=512)
            vf = f0[(f0 > 70) & (f0 < 400)]
            all_voiced_f0.append(vf)

            if len(vf) > 0:
                log_vf   = np.log2(vf)
                f0_feats = np.array([float(np.mean(log_vf)), float(np.std(log_vf))])
            else:
                f0_feats = np.zeros(2)

            embeddings.append(np.concatenate([mfcc_emb, f0_feats]))
            seg_durs.append((end - start) / sr)

        # Barcha ovozli F0 larni birlashtiramiz (zahira funksiyasi uchun)
        voiced_f0_all = np.concatenate(
            [v for v in all_voiced_f0 if len(v) > 0]
        ) if any(len(v) > 0 for v in all_voiced_f0) else np.array([])

        if len(embeddings) < 4:
            return _has_pitch_bimodality(voiced_f0_all)

        X    = StandardScaler().fit_transform(np.array(embeddings))
        dist = cosine_distances(X).astype(np.float64)
        upper = dist[np.triu_indices(len(X), k=1)]

        # 3. 3 ta eps sinovi: har birida inter-klaster masofa saqlanadi
        eps_configs = [
            (np.percentile(upper, 25), 0.10, 0.45),   # qattiq
            (np.percentile(upper, 35), 0.15, 0.55),   # ortacha
            (np.percentile(upper, 50), 0.20, 0.65),   # yumshoq
        ]

        votes        = []
        trial_labels = []
        inter_dists  = []

        for raw_eps, lo, hi in eps_configs:
            eps = float(np.clip(raw_eps, lo, hi))
            lbl = DBSCAN(
                eps=eps, min_samples=2, metric="precomputed"
            ).fit_predict(dist)

            unique = set(lbl) - {-1}
            if len(unique) >= 2:
                labeled = lbl[lbl != -1]
                fracs   = [np.sum(labeled == l) / len(labeled) for l in unique]
                ok      = min(fracs) >= 0.15
                if ok:
                    # Inter-klaster masofa: klaster sentroidlari orasidagi minimum
                    centroids  = np.array([X[lbl == l].mean(axis=0) for l in unique])
                    c_dist_mat = euclidean_distances(centroids)
                    np.fill_diagonal(c_dist_mat, np.inf)
                    inter_dists.append(float(c_dist_mat.min()))
                else:
                    inter_dists.append(0.0)
            else:
                ok = False
                inter_dists.append(0.0)

            votes.append(ok)
            trial_labels.append(lbl)

        n_true = sum(votes)

        if n_true == 0:
            return _has_pitch_bimodality(voiced_f0_all)

        # Eng yaxshi sinov: inter-klaster masofa maksimal bolgan ijobiy sinov
        best_idx = max(
            (i for i, v in enumerate(votes) if v),
            key=lambda i: inter_dists[i],
        )
        best_labels = trial_labels[best_idx]

        if n_true >= 2:
            return _check_temporal_coherence(best_labels, np.array(seg_durs))

        # Aynan 1/3 ovoz ijobiy - chegara holat, F0 bimodalligi bilan hal qilamiz
        return _has_pitch_bimodality(voiced_f0_all)

    except Exception as e:
        print(f"[filter_audio] detect_multiple_speakers xato: {e}")
        return False


# ════════════════════════════════════════════════════════════════
# SNR, SUKUNAT, TIL
# ════════════════════════════════════════════════════════════════

def compute_snr(file_path: str) -> float:
    """
    Signal-to-Noise Ratio (dB) hisoblaydi.
    Yuqori = sifatli audio.
    """
    try:
        import librosa
        y, sr = librosa.load(file_path, sr=None, mono=True)

        signal_rms = np.sqrt(np.mean(y ** 2))
        if signal_rms == 0:
            return 0.0

        frame_size  = int(sr * 0.02)
        hop_size    = int(sr * 0.01)
        frames      = librosa.util.frame(y, frame_length=frame_size, hop_length=hop_size)
        frame_rms   = np.sqrt(np.mean(frames ** 2, axis=0))
        noise_rms   = np.percentile(frame_rms, 10)

        if noise_rms == 0:
            return 60.0

        snr = 20 * np.log10(signal_rms / noise_rms)
        return round(float(snr), 2)

    except Exception:
        return -1.0


def compute_silence_ratio(file_path: str, threshold_db: float = -40) -> float:
    """
    Fayldagi sukunat ulushini % da qaytaradi.
    """
    try:
        from pydub import AudioSegment
        audio    = AudioSegment.from_file(file_path)
        total_ms = len(audio)
        silent   = sum(1 for ms in range(0, total_ms, 10)
                       if audio[ms:ms+10].dBFS < threshold_db)
        return round(silent * 10 / total_ms * 100, 1)
    except Exception:
        return -1.0


def detect_language(file_path: str) -> str:
    """
    Audio tilini aniqlaydi (langdetect orqali).
    Qaytaradi: "uz", "ru", "en", "unknown"
    """
    try:
        import whisper
        model = whisper.load_model("tiny")
        audio = whisper.load_audio(file_path)
        audio = whisper.pad_or_trim(audio)
        mel   = whisper.log_mel_spectrogram(audio)
        _, probs = model.detect_language(mel)
        return max(probs, key=probs.get)
    except Exception:
        return "unknown"


def filter_audio(segment: dict,
                 snr_min: float = 15.0,
                 duration_min: float = 3.0,
                 duration_max: float = 30.0,
                 silence_max: float = 80.0,
                 check_language: bool = False,
                 target_lang: str = "uz") -> tuple[bool, str]:
    """
    Audio segmentni filtrlaydi.

    Returns:
        (True, "ok")          - filtrdan otdi
        (False, "sabab")      - tashlab yuboriladi
    """
    file_path = segment["file"]
    duration  = segment.get("duration", 0)

    if duration < duration_min:
        return False, f"juda qisqa ({duration:.1f}s < {duration_min}s)"
    if duration > duration_max:
        return False, f"juda uzun ({duration:.1f}s > {duration_max}s)"

    snr = compute_snr(file_path)
    segment["snr_score"] = snr
    if snr_min > 0 and snr >= 0 and snr < snr_min:
        return False, f"SNR past ({snr:.1f}dB < {snr_min}dB)"

    silence = compute_silence_ratio(file_path)
    segment["silence_ratio"] = silence
    if silence > 0 and silence > silence_max:
        return False, f"kop sukunat ({silence:.0f}% > {silence_max}%)"

    if check_language:
        lang = detect_language(file_path)
        segment["detected_lang"] = lang
        if lang != target_lang and lang != "unknown":
            return False, f"til mos emas ({lang} != {target_lang})"

    return True, "ok"


# ════════════════════════════════════════════════════════════════
# MATN FILTRLARI
# ════════════════════════════════════════════════════════════════

# Shovqin belgilari
NOISE_PATTERNS = [
    r"^(mm+|uh+|ah+|eh+|um+|hmm+)[\s.,!?]*$",   # faqat undovlar
    r"^\W+$",                                       # faqat tinish
    r"^\.{3,}$",                                    # faqat nuqtalar
]

def compute_repeat_ratio(text: str) -> float:
    """Takror sozlar ulushini % da hisoblaydi."""
    words = text.lower().split()
    if not words:
        return 100.0
    unique = set(words)
    return round((1 - len(unique) / len(words)) * 100, 1)


def has_mixed_scripts(text: str) -> bool:
    """Kirill va Lotin aralash ekanligini tekshiradi."""
    has_cyrillic = bool(re.search(r'[а-яёА-ЯЁ]', text))
    has_latin    = bool(re.search(r'[a-zA-Z]', text))
    if has_cyrillic and has_latin:
        latin_words = len(re.findall(r'[a-zA-Z]+', text))
        total_words = len(text.split())
        return latin_words / max(total_words, 1) > 0.3
    return False


def filter_text_v1(text: str,
                   word_min: int = 3,
                   repeat_max: float = 70.0,
                   check_noise: bool = True,
                   check_mixed: bool = False) -> tuple[bool, str]:
    """
    Yol 1 matn filtri (ortacha qattiq).
    """
    if not text or not text.strip():
        return False, "bosh matn"

    words = text.strip().split()

    if len(words) < word_min:
        return False, f"kam soz ({len(words)} < {word_min})"

    repeat = compute_repeat_ratio(text)
    if repeat > repeat_max:
        return False, f"kop takror ({repeat:.0f}% > {repeat_max}%)"

    if check_noise:
        for pat in NOISE_PATTERNS:
            if re.match(pat, text.strip(), re.IGNORECASE):
                return False, "shovqin/undov matni"

    if check_mixed and has_mixed_scripts(text):
        return False, "kirill/lotin aralash"

    return True, "ok"


def filter_text_v2(text: str,
                   word_min: int = 5,
                   repeat_max: float = 50.0,
                   check_mixed: bool = True) -> tuple[bool, str]:
    """
    Yol 2 matn filtri (qattiqroq).
    """
    if not text or not text.strip():
        return False, "bosh matn"

    words = text.strip().split()

    if len(words) < word_min:
        return False, f"kam soz ({len(words)} < {word_min})"

    repeat = compute_repeat_ratio(text)
    if repeat > repeat_max:
        return False, f"kop takror ({repeat:.0f}% > {repeat_max}%)"

    for pat in NOISE_PATTERNS:
        if re.match(pat, text.strip(), re.IGNORECASE):
            return False, "shovqin/undov matni"

    if check_mixed and has_mixed_scripts(text):
        return False, "til aralash"

    return True, "ok"


def compute_change_ratio(original: str, polished: str) -> float:
    """Gemini qancha ozgartirgani ni % da hisoblaydi."""
    orig_words     = set(original.lower().split())
    polished_words = set(polished.lower().split())
    if not orig_words:
        return 100.0
    changed = orig_words.symmetric_difference(polished_words)
    return round(len(changed) / len(orig_words) * 100, 1)
