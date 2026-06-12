// ─────────────────────────────────────────────
// TOAST SYSTEM (replaces native alert)
// ─────────────────────────────────────────────
(function _initToastRoot(){
  if (document.getElementById('toast-root')) return;
  var r = document.createElement('div');
  r.id = 'toast-root';
  r.setAttribute('aria-live', 'polite');
  r.setAttribute('aria-atomic', 'true');
  // Defer insertion until body exists
  if (document.body) document.body.appendChild(r);
  else document.addEventListener('DOMContentLoaded', function(){ document.body.appendChild(r); });
})();

function toast(message, type, duration){
  type = type || 'info';
  duration = duration || 4200;
  var root = document.getElementById('toast-root');
  if (!root) { root = document.createElement('div'); root.id = 'toast-root'; document.body.appendChild(root); }
  var el = document.createElement('div');
  el.className = 'toast toast--' + type;
  el.setAttribute('role', type === 'error' ? 'alert' : 'status');

  var ico = document.createElement('span');
  ico.className = 'toast__icon';
  ico.textContent = type === 'error' ? '⚠' : (type === 'success' ? '✓' : (type === 'warning' ? '!' : 'ℹ'));
  var body = document.createElement('span');
  body.className = 'toast__body';
  body.textContent = String(message);
  var close = document.createElement('button');
  close.className = 'toast__close';
  close.type = 'button';
  close.setAttribute('aria-label', 'Close');
  close.textContent = '×';

  el.appendChild(ico); el.appendChild(body); el.appendChild(close);
  root.appendChild(el);

  // Trigger enter animation
  requestAnimationFrame(function(){ el.classList.add('toast--show'); });

  function dismiss(){
    el.classList.remove('toast--show');
    el.classList.add('toast--leaving');
    setTimeout(function(){ if (el.parentNode) el.parentNode.removeChild(el); }, 220);
  }
  close.addEventListener('click', dismiss);
  var t = setTimeout(dismiss, duration);
  el.addEventListener('mouseenter', function(){ clearTimeout(t); });
  el.addEventListener('mouseleave', function(){ t = setTimeout(dismiss, 1800); });
}

// Override native alert — keeps every existing `alert(...)` call working,
// but renders as a styled toast instead of a native modal.
(function _overrideAlert(){
  var _nativeAlert = window.alert;
  window.alert = function(message){
    var msg = String(message == null ? '' : message);
    // In this app, alert() is always used for issue notifications.
    // Promote by prefix/keyword; default to "error".
    var lower = msg.toLowerCase();
    var type = 'error';
    if (msg.indexOf('✅') === 0 || /\b(success|done)\b/i.test(lower)) type = 'success';
    else if (msg.indexOf('⚠') === 0 || /\b(warn|warning)\b/i.test(lower)) type = 'warning';
    else if (msg.indexOf('ℹ') === 0) type = 'info';
    try { toast(msg, type, 5000); }
    catch (e) { _nativeAlert(msg); }
  };
})();

// ─────────────────────────────────────────────
// EYE TOGGLE
// ─────────────────────────────────────────────
function toggleEye(inputId, btn) {
  const inp = document.getElementById(inputId);
  if (inp.type === 'password') {
    inp.type = 'text';
    btn.textContent = '🙈';
    btn.classList.add('open');
  } else {
    inp.type = 'password';
    btn.textContent = '👁';
    btn.classList.remove('open');
  }
}

// ─────────────────────────────────────────────
// LANGUAGE DICTIONARIES (complete i18n)
// ─────────────────────────────────────────────
var LANG = {
  uz: {
    lang_name: "O'zbek",
    // Sidebar tabs
    tab_1: "1 · Pipeline va Manba",
    tab_2: "2 · Filtrlar",
    tab_3: "3 · Natija Formati",
    tab_4: "4 · Ishga tushur",
    tab_5: "5 · HuggingFace Push",
    tab_6: "6 · Augmentation",
    tab_7: "7 · Mapping",
    tab_8: "8 · Training",
    // Header
    site_sub: "Speech Dataset Tool",
    logout: "Chiqish",
    // Tab 1 Pipeline
    pipeline_title: "Pipeline tanlang",
    pipeline_p1: "Yo'l 1 — MohirDev API",
    pipeline_p2: "Yo'l 2 — Gemini STT",
    pipeline_p3: "Yo'l 3 — Python Filter",
    key_gemini: "Model API Key",
    key_gemini_ph: "AIza... (har safar kiriting)",
    key_aisha: "MohirDev STT API Key (Yo'l 1)",
    key_aisha_ph: "x-api-key qiymatini kiriting...",
    // Source
    source_title: "Audio manba",
    source_yt: "▶ YouTube URL",
    source_json: "📄 JSON URL fayl",
    source_hf: "🤗 HuggingFace",
    source_local: "📁 Local papka",
    yt_url: "YouTube URL",
    yt_url_ph: "https://youtube.com/watch?v=...",
    json_upload: "JSON fayl yuklang",
    json_drop: "Faylni bu yerga tashlang yoki bosing",
    json_empty: "Fayl tanlanmagan",
    hf_name_label: "Dataset nomi yoki URL",
    hf_name_ph: "org/dataset-name  yoki  https://huggingface.co/datasets/...",
    hf_config: "Config / Subset",
    hf_config_opt: "(ixtiyoriy)",
    hf_config_ph: "uz, en, ...",
    hf_token_label: "HuggingFace Token",
    hf_token_private: "(private dataset uchun)",
    hf_token_ph: "hf_xxxx... (public uchun bo'sh qoldiring)",
    local_title: "Audio fayllarni tanlang",
    local_click: "Audio fayllarni tanlash uchun bosing",
    local_hint: "Bir yoki bir nechta fayl tanlang (.wav .mp3 .ogg .flac .m4a)",
    uploading: "Audio fayllar yuklanmoqda...",
    // Tab 2
    filter_title: "Shovqin Kamaytirish",
    nr_label: "Noise Reduction",
    nr_desc: "Har bir audio segmentdan orqa fon shovqinini kamaytiradi",
    nr_strength: "Kuch (0.1 = yengil · 1.0 = maksimal)",
    chunk_title: "Audio Chunk Uzunligi",
    dur_min: "Minimal uzunlik (s)",
    dur_max: "Maksimal uzunlik (s)",
    smart_title: "Model Aqlli Filtrlari",
    f_music: "Fon musiqasini aniqlash",
    f_music_d: "Orqa fonda musiqa bor bo'lsa metadataga belgilanadi, audio yo'qolmaydi",
    f_multi: "Ko'p ovozni aniqlash",
    f_multi_d: "Bir nechta kishi gapirsa metadataga belgilanadi, audio yo'qolmaydi",
    f_noisy: "Juda shovqinli audiolarni o'chirish",
    f_noisy_d: "Atrofdagi shovqin juda baland bo'lsa filtrlanadi",
    f_silence: "Uzoq sukut bo'lagini o'chirish",
    f_silence_d: "Audioning ko'p qismi sukunat bo'lsa filtrlanadi",
    norm_title: "Matnni Normallashtirish qoidalari",
    n_cap: "Birinchi harfni katta bilan yozish",
    n_cap_d: "Har bir gapning birinchi so'zi doim katta harf bilan boshlanadi",
    n_numw: "1→bir · Raqamlarni so'z bilan yozish",
    n_numw_d: "Barcha raqamlar tekst holatida yoziladi",
    n_latin: "Faqat lotin yozuvida bo'lsin",
    n_latin_d: "Boshqa tildagi yozuvlar lotin alifbosiga o'tkaziladi",
    n_notrans: "Tarjima qilmaslik",
    n_notrans_d: "Asl audio tili holatida transcription qilinadi",
    n_nonoise: "Shovqin teglarini o'chirish",
    n_nonoise_d: "Kulgu, it ovozi va shunga o'xshash teglarni qo'shmaslik",
    n_noprompt: "Buyruqlarni qayta yozmaslik",
    n_noprompt_d: "Model ko'rsatmalarni transcription ichiga yozmasin",
    meta_title: "Yo'l 2 — Gemini STT metadata filtri",
    meta_desc: "Metadatadagi xato va nomaqbul matnlarni tozalash. Yoqilgan holatda pipeline davomida avtomatik qo'llaniladi.",
    m1: "① Birinchi harfni katta bilan yozish",
    m1_d: "Har bir gapning birinchi so'zi katta harf bilan boshlanadi.",
    m2: "② Raqamlarni so'z bilan yozish",
    m2_d: "Raqamlar tekst holatida yoziladi.",
    m3: "③ Apostrofni to'g'rilash",
    m3_d: "ʻ va ʼ → standart ' apostrofi.",
    m4: "④ Takrorlangan matnni o'chirish",
    m4_d: "Matn ikki marta yozilgan bo'lsa biri o'chiriladi.",
    m5: "⑤ Tinish belgisi qo'shish",
    m5_d: "Gap oxirida . ! ? bo'lmasa nuqta qo'yiladi.",
    m6: "⑥ Qo'shtirnoqlarni to'g'rilash",
    m6_d: "\" \" va ' ' → standart \" belgisi.",
    m7: "⑦ Gap ichida kichik harfni to'g'rilash",
    m7_d: ". ! ? dan keyin katta harf.",
    m8: "⑧ Qo'sh bo'shliqlarni o'chirish",
    m8_d: "Ikki+ bo'shliq → bitta.",
    m9: "⑨ JSON artifactlarni tozalash",
    m9_d: "{'transcription': '...'} qoldiqlari tozalanadi.",
    m10: "⑩ Uch nuqta → ellipsis",
    m10_d: "'...' o'rniga '…' belgisi.",
    m11: "⑪ Defisni tirega almashtirish",
    m11_d: "' - ' → '—'.",
    m12: "⑫ Kirill harflar → filtr",
    m12_d: "Kirill harfli segment o'chiriladi.",
    m13: "⑬ Bracket artifactlarni tozalash",
    m13_d: "[{json...}] qoldiqlari tozalanadi.",
    m14: "⑭ HTML teglarni tozalash",
    m14_d: "<tag> belgilar o'chiriladi.",
    m15: "⑮ Faqat raqam → filtr",
    m15_d: "Faqat raqamli matn ('497347') o'chiriladi.",
    m16: "⑯ Singan tirani to'g'rilash",
    m16_d: "'so'z- so'z' → 'so'z-so'z'.",
    m17: "⑰ Ko'p vergullarni tozalash",
    m17_d: "',,,,' → ','.",
    cols_title: "Metadata Ustunlarini Tanlash",
    col_always: "(har doim)",
    // Tab 3
    format_title: "Fayl nomlari",
    out_v1: "→ Yo'l 1 (MohirDev API) chiqish fayli",
    out_v2: "→ Yo'l 2 (Gemini STT) chiqish fayli",
    out_v3: "→ Yo'l 3 (Python Filter) chiqish fayli",
    out_name_label: "Metadata fayl nomi",
    write_title: "Yozish rejimi",
    wm_new: "📄 Yangi fayl yaratish",
    wm_append: "➕ Mavjudga qo'shish",
    wm_hint: "Append: bir nechta URL dan olingan metadatalar bitta faylga yig'iladi",
    format_title2: "Format",
    // Tab 4
    start_btn: "▶ Ishga tushur",
    status_ready: "Tayyor",
    status_preparing: "Tayyorlanmoqda...",
    status_running: "Ishlayapti...",
    status_done: "✅ Tugadi!",
    status_error: "Xato",
    status_session: "Sessiya tugadi",
    stats_title: "Statistika",
    stat_yol1: "YOʼL 1",
    stat_yol2: "YOʼL 2",
    stat_total: "Jami",
    stat_approved: "Approved",
    stat_pending: "Pending",
    stat_rejected: "Rejected",
    stat_filtered: "Filtered",
    log_title: "Log",
    files_meta_title: "Tayyor fayllar — Metadata",
    files_none: "Hali fayl yo'q",
    files_zip_title: "Audio fayllar — ZIP",
    zip_btn: "🗜 Audio fayllarni ZIP qilish",
    zip_none: "Hali ZIP yo'q",
    results_title: "Natijalarni Ko'rish",
    prev_btn: "◀ Oldingi",
    next_btn: "Keyingi ▶",
    transcript_label: "Transkript",
    no_results: "Natijalar yo'q...",
    download_btn: "⬇ Yuklab olish",
    delete_title: "O'chirish",
    delete_confirm: "O'chirishni tasdiqlaysizmi?",
    folder_found: "ta audio fayl topildi",
    // Tab 5
    hf_settings: "HuggingFace Sozlamalari",
    hf_token_label2: "HF Token",
    hf_org_label: "Org nomi",
    hf_repo_label: "Repo nomi",
    hf_private_lbl: "🔒 Private repo",
    hf_auto_badge: "✅ Pipeline natijasi avtomatik yuklandi",
    hf_jsonl_lbl: "JSONL fayl",
    hf_csv_lbl: "CSV fayl",
    hf_audio_lbl: "Audio papka",
    hf_warn: "⚠️ Avval \"4 · Ishga tushur\" tabida pipeline ishga tushiring. Pipeline tugagach bu yerga avtomatik to'ldiriladi.",
    push_format_title: "Push formati",
    push_jsonl_btn: "📄 JSONL bilan push",
    push_csv_btn: "📊 CSV bilan push",
    push_btn_jsonl2: "🚀 HuggingFace ga push qilish (JSONL)",
    push_btn_csv2: "🚀 HuggingFace ga push qilish (CSV)",
    push_ready: "Tayyor",
    // Tab 6
    aug_title: "Dataset Augmentation",
    aug_cpu: "CPU yadro (num_proc)",
    aug_prob_lbl: "Augmentatsiya ehtimolligi (%)",
    aug_prob_hint: "— Necha foiz audioga shovqin qo'shilsin",
    aug_skip_lbl: "Augmentatsiyani o'tkazib yuborish (Skip)",
    aug_skip_d: "Shovqin qo'shmasdan shunchaki HF Dataset formatiga o'tkazish",
    aug_test_btn: "🧪 100 tasida Test",
    aug_continue_btn: "▶ Qolganini davom ettirish",
    aug_full_btn: "⚙ Barchasini Augment",
    aug_review_title: "Augmented Natijani Ko'rish",
    aug_text_lbl: "Matn",
    // Tab 7
    map_title: "HuggingFace Mapping",
    map_model_lbl: "Model nomi",
    map_output_lbl: "Chiqish papkasi (Output Dir)",
    map_token_lbl: "HuggingFace Token",
    map_ds_lbl: "HuggingFace Dataset nomlari (vergul bilan)",
    map_cpu_lbl: "CPU yadro (num_proc)",
    map_start_btn: "⚙ Mappingni boshlash",
    // Tab 8
    train_title: "Model Training",
    train_wandb_lbl: "W&B API Key",
    train_hf_lbl: "HuggingFace Token (HF_TOKEN)",
    train_model_lbl: "Model",
    train_ds_lbl: "Mapped Datasetlar (vergul bilan)",
    train_cpu_lbl: "CPU (num_proc)",
    train_split_lbl: "Train/Test split",
    train_output_lbl: "Output Dir",
    train_batch_lbl: "Batch Size",
    train_grad_lbl: "Grad Accumulation",
    train_lr_lbl: "Learning Rate",
    train_epochs_lbl: "Epochs",
    train_ddpt_lbl: "DDP Timeout",
    train_workers_lbl: "Dataloader Workers",
    train_pin_lbl: "Pin Memory",
    train_ddpu_lbl: "DDP Find Unused",
    train_resume_lbl: "Resume Checkpoint",
    train_start_btn: "🚀 Trainingni boshlash",
    // Alerts
    alert_aisha: "MohirDev STT API kalitini kiriting (Yo'l 1 uchun).",
    alert_gemini: "Gemini API kalitini kiriting (Yo'l 2 uchun).",
    alert_yturl: "YouTube URL kiriting.",
    alert_hfname: "HuggingFace dataset nomini kiriting.",
    alert_jsonfile: "JSON fayl tanlang.",
    alert_localdir: "Audio papkasini tanlang.",
    alert_session: "Sessiya tugadi — qayta kiring.",
  },
  ru: {
    lang_name: "Русский",
    tab_1: "1 · Пайплайн и Источник",
    tab_2: "2 · Фильтры",
    tab_3: "3 · Формат результата",
    tab_4: "4 · Запуск",
    tab_5: "5 · HuggingFace Push",
    tab_6: "6 · Аугментация",
    tab_7: "7 · Маппинг",
    tab_8: "8 · Обучение",
    site_sub: "Speech Dataset Tool",
    logout: "Выйти",
    pipeline_title: "Выберите пайплайн",
    pipeline_p1: "Путь 1 — MohirDev API",
    pipeline_p2: "Путь 2 — Gemini STT",
    pipeline_p3: "Путь 3 — Python Filter",
    key_gemini: "API-ключ модели",
    key_gemini_ph: "AIza... (вводите каждый раз)",
    key_aisha: "MohirDev STT API-ключ (Путь 1)",
    key_aisha_ph: "Введите значение x-api-key...",
    source_title: "Источник аудио",
    source_yt: "▶ YouTube URL",
    source_json: "📄 JSON файл со ссылками",
    source_hf: "🤗 HuggingFace",
    source_local: "📁 Локальная папка",
    yt_url: "YouTube URL",
    yt_url_ph: "https://youtube.com/watch?v=...",
    json_upload: "Загрузите JSON файл",
    json_drop: "Перетащите файл сюда или нажмите",
    json_empty: "Файл не выбран",
    hf_name_label: "Название датасета или URL",
    hf_name_ph: "org/dataset-name или https://huggingface.co/datasets/...",
    hf_config: "Config / Subset",
    hf_config_opt: "(опционально)",
    hf_config_ph: "uz, en, ...",
    hf_token_label: "HuggingFace Token",
    hf_token_private: "(для приватного датасета)",
    hf_token_ph: "hf_xxxx... (для публичного оставьте пустым)",
    local_title: "Выберите аудио файлы",
    local_click: "Нажмите, чтобы выбрать аудиофайлы",
    local_hint: "Выберите один или несколько файлов (.wav .mp3 .ogg .flac .m4a)",
    uploading: "Аудиофайлы загружаются...",
    filter_title: "Шумоподавление",
    nr_label: "Noise Reduction",
    nr_desc: "Уменьшает фоновый шум в каждом аудио сегменте",
    nr_strength: "Сила (0.1 = мягко · 1.0 = максимум)",
    chunk_title: "Длина аудио чанков",
    dur_min: "Минимальная длина (с)",
    dur_max: "Максимальная длина (с)",
    smart_title: "Умные фильтры модели",
    f_music: "Определять фоновую музыку",
    f_music_d: "Если на фоне музыка — помечается в метаданных, аудио не удаляется",
    f_multi: "Определять несколько голосов",
    f_multi_d: "Если несколько голосов — помечается в метаданных, аудио не удаляется",
    f_noisy: "Удалять очень шумные аудио",
    f_noisy_d: "Фильтруется, если фоновый шум слишком громкий",
    f_silence: "Удалять длинные участки тишины",
    f_silence_d: "Фильтруется, если большая часть аудио — тишина",
    norm_title: "Правила нормализации текста",
    n_cap: "Писать первую букву с заглавной",
    n_cap_d: "Первое слово каждого предложения всегда с заглавной буквы",
    n_numw: "1→один · Писать числа словами",
    n_numw_d: "Все числа записываются текстом",
    n_latin: "Только латиница",
    n_latin_d: "Текст на других языках переводится в латиницу",
    n_notrans: "Не переводить",
    n_notrans_d: "Транскрипция сохраняется на языке оригинала",
    n_nonoise: "Удалять теги шума",
    n_nonoise_d: "Не добавлять теги вроде смеха, лая и т. п.",
    n_noprompt: "Не повторять инструкции",
    n_noprompt_d: "Модель не должна писать инструкции внутри транскрипции",
    meta_title: "Yo'l 2 — Gemini STT metadata filtri",
    meta_desc: "Очистка ошибочных и неподходящих текстов в метаданных. При включении применяется автоматически во время пайплайна.",
    m1: "① Писать первую букву с заглавной",
    m1_d: "Первое слово каждого предложения с заглавной.",
    m2: "② Писать числа словами",
    m2_d: "Числа записываются текстом.",
    m3: "③ Исправить апостроф",
    m3_d: "ʻ и ʼ → стандартный ' апостроф.",
    m4: "④ Удалить повторяющийся текст",
    m4_d: "Если текст записан дважды, один удаляется.",
    m5: "⑤ Добавить знаки препинания",
    m5_d: "Если в конце нет . ! ? — ставится точка.",
    m6: "⑥ Исправить кавычки",
    m6_d: "\" \" и ' ' → стандартные \" кавычки.",
    m7: "⑦ Исправить регистр после знаков",
    m7_d: "Заглавная буква после . ! ?",
    m8: "⑧ Убрать двойные пробелы",
    m8_d: "Два и больше пробелов → один.",
    m9: "⑨ Очистить JSON артефакты",
    m9_d: "Остатки {'transcription': '...'} очищаются.",
    m10: "⑩ Три точки → многоточие",
    m10_d: "'...' → '…'",
    m11: "⑪ Заменить дефис на тире",
    m11_d: "' - ' → '—'.",
    m12: "⑫ Кириллица → фильтр",
    m12_d: "Сегмент с кириллицей удаляется.",
    m13: "⑬ Очистить артефакты скобок",
    m13_d: "Остатки [{json...}] очищаются.",
    m14: "⑭ Очистить HTML теги",
    m14_d: "Символы <tag> удаляются.",
    m15: "⑮ Только числа → фильтр",
    m15_d: "Текст только из цифр ('497347') удаляется.",
    m16: "⑯ Исправить разорванный дефис",
    m16_d: "'so'z- so'z' → 'so'z-so'z'.",
    m17: "⑰ Убрать лишние запятые",
    m17_d: "',,,,' → ','.",
    cols_title: "Выбор колонок метаданных",
    col_always: "(всегда)",
    format_title: "Имя файла",
    out_v1: "→ Путь 1 (MohirDev API) файл вывода",
    out_v2: "→ Путь 2 (Gemini STT) файл вывода",
    out_v3: "→ Путь 3 (Python Filter) файл вывода",
    out_name_label: "Имя файла метаданных",
    write_title: "Режим записи",
    wm_new: "📄 Создать новый файл",
    wm_append: "➕ Дополнить существующий",
    wm_hint: "Append: метаданные из нескольких URL объединяются в один файл",
    format_title2: "Формат",
    start_btn: "▶ Запустить",
    status_ready: "Готов",
    status_preparing: "Подготовка...",
    status_running: "Выполняется...",
    status_done: "✅ Готово!",
    status_error: "Ошибка",
    status_session: "Сессия истекла",
    stats_title: "Статистика",
    stat_yol1: "ПУТЬ 1",
    stat_yol2: "ПУТЬ 2",
    stat_total: "Всего",
    stat_approved: "Одобрено",
    stat_pending: "В ожидании",
    stat_rejected: "Отклонено",
    stat_filtered: "Отфильтр.",
    log_title: "Лог",
    files_meta_title: "Готовые файлы — Метаданные",
    files_none: "Пока нет файлов",
    files_zip_title: "Аудио файлы — ZIP",
    zip_btn: "🗜 Упаковать аудио в ZIP",
    zip_none: "Пока нет ZIP",
    results_title: "Просмотр результатов",
    prev_btn: "◀ Предыдущий",
    next_btn: "Следующий ▶",
    transcript_label: "Транскрипт",
    no_results: "Результатов нет...",
    download_btn: "⬇ Скачать",
    delete_title: "Удалить",
    delete_confirm: "Подтверждаете удаление?",
    folder_found: "аудио файлов найдено",
    hf_settings: "Настройки HuggingFace",
    hf_token_label2: "HF Token",
    hf_org_label: "Название организации",
    hf_repo_label: "Название репозитория",
    hf_private_lbl: "🔒 Приватный репозиторий",
    hf_auto_badge: "✅ Результат пайплайна загружен автоматически",
    hf_jsonl_lbl: "JSONL файл",
    hf_csv_lbl: "CSV файл",
    hf_audio_lbl: "Папка аудио",
    hf_warn: "⚠️ Сначала запустите пайплайн на вкладке «4 · Запуск». После завершения данные заполнятся автоматически.",
    push_format_title: "Формат push",
    push_jsonl_btn: "📄 Push в JSONL",
    push_csv_btn: "📊 Push в CSV",
    push_btn_jsonl2: "🚀 Push на HuggingFace (JSONL)",
    push_btn_csv2: "🚀 Push на HuggingFace (CSV)",
    push_ready: "Готов",
    aug_title: "Аугментация датасета",
    aug_cpu: "CPU ядра (num_proc)",
    aug_prob_lbl: "Вероятность аугментации (%)",
    aug_prob_hint: "— Какой процент аудио получит шум",
    aug_skip_lbl: "Пропустить аугментацию (Skip)",
    aug_skip_d: "Без добавления шума — просто конвертировать в формат HF Dataset",
    aug_test_btn: "🧪 Тест на 100 элементах",
    aug_continue_btn: "▶ Продолжить с остальных",
    aug_full_btn: "⚙ Аугментировать все",
    aug_review_title: "Просмотр результата аугментации",
    aug_text_lbl: "Текст",
    map_title: "HuggingFace Маппинг",
    map_model_lbl: "Название модели",
    map_output_lbl: "Выходная папка (Output Dir)",
    map_token_lbl: "HuggingFace Token",
    map_ds_lbl: "Названия HuggingFace датасетов (через запятую)",
    map_cpu_lbl: "CPU ядра (num_proc)",
    map_start_btn: "⚙ Запустить маппинг",
    train_title: "Обучение модели",
    train_wandb_lbl: "W&B API Key",
    train_hf_lbl: "HuggingFace Token (HF_TOKEN)",
    train_model_lbl: "Модель",
    train_ds_lbl: "Маппинг датасеты (через запятую)",
    train_cpu_lbl: "CPU (num_proc)",
    train_split_lbl: "Train/Test split",
    train_output_lbl: "Output Dir",
    train_batch_lbl: "Batch Size",
    train_grad_lbl: "Grad Accumulation",
    train_lr_lbl: "Learning Rate",
    train_epochs_lbl: "Эпохи",
    train_ddpt_lbl: "DDP Timeout",
    train_workers_lbl: "Dataloader Workers",
    train_pin_lbl: "Pin Memory",
    train_ddpu_lbl: "DDP Find Unused",
    train_resume_lbl: "Resume Checkpoint",
    train_start_btn: "🚀 Запустить обучение",
    alert_aisha: "Введите ключ MohirDev STT API (для Пути 1).",
    alert_gemini: "Введите Gemini API ключ (для Пути 2).",
    alert_yturl: "Введите YouTube URL.",
    alert_hfname: "Введите название HuggingFace датасета.",
    alert_jsonfile: "Выберите JSON файл.",
    alert_localdir: "Выберите папку с аудио.",
    alert_session: "Сессия истекла — войдите снова.",
  },
  en: {
    lang_name: "English",
    tab_1: "1 · Pipeline & Source",
    tab_2: "2 · Filters",
    tab_3: "3 · Output Format",
    tab_4: "4 · Run",
    tab_5: "5 · HuggingFace Push",
    tab_6: "6 · Augmentation",
    tab_7: "7 · Mapping",
    tab_8: "8 · Training",
    site_sub: "Speech Dataset Tool",
    logout: "Sign out",
    pipeline_title: "Select Pipeline",
    pipeline_p1: "Path 1 — MohirDev API",
    pipeline_p2: "Path 2 — Gemini STT",
    pipeline_p3: "Path 3 — Python Filter",
    key_gemini: "Model API Key",
    key_gemini_ph: "AIza... (enter each time)",
    key_aisha: "MohirDev STT API Key (Path 1)",
    key_aisha_ph: "Enter x-api-key value...",
    source_title: "Audio Source",
    source_yt: "▶ YouTube URL",
    source_json: "📄 JSON URL File",
    source_hf: "🤗 HuggingFace",
    source_local: "📁 Local Folder",
    yt_url: "YouTube URL",
    yt_url_ph: "https://youtube.com/watch?v=...",
    json_upload: "Upload JSON file",
    json_drop: "Drop file here or click",
    json_empty: "No file selected",
    hf_name_label: "Dataset name or URL",
    hf_name_ph: "org/dataset-name  or  https://huggingface.co/datasets/...",
    hf_config: "Config / Subset",
    hf_config_opt: "(optional)",
    hf_config_ph: "uz, en, ...",
    hf_token_label: "HuggingFace Token",
    hf_token_private: "(for private dataset)",
    hf_token_ph: "hf_xxxx... (leave empty for public)",
    local_title: "Select audio files",
    local_click: "Click to select audio files",
    local_hint: "Pick one or more files (.wav .mp3 .ogg .flac .m4a)",
    uploading: "Uploading audio files...",
    filter_title: "Noise Reduction",
    nr_label: "Noise Reduction",
    nr_desc: "Reduces background noise from each audio segment",
    nr_strength: "Strength (0.1 = light · 1.0 = maximum)",
    chunk_title: "Audio Chunk Length",
    dur_min: "Min length (s)",
    dur_max: "Max length (s)",
    smart_title: "Smart Model Filters",
    f_music: "Detect background music",
    f_music_d: "If background music found — flagged in metadata, audio is kept",
    f_multi: "Detect multiple speakers",
    f_multi_d: "If multiple speakers found — flagged in metadata, audio is kept",
    f_noisy: "Remove very noisy audio",
    f_noisy_d: "Filtered if ambient noise is too loud",
    f_silence: "Remove long silence segments",
    f_silence_d: "Filtered if most of audio is silence",
    norm_title: "Text Normalization Rules",
    n_cap: "Capitalize first letter",
    n_cap_d: "First word of every sentence always starts with uppercase",
    n_numw: "1→one · Write numbers as words",
    n_numw_d: "All numbers are written as text",
    n_latin: "Latin script only",
    n_latin_d: "Other scripts are converted to Latin",
    n_notrans: "Don't translate",
    n_notrans_d: "Transcription stays in original audio language",
    n_nonoise: "Remove noise tags",
    n_nonoise_d: "Don't include tags like laughter, barking, etc.",
    n_noprompt: "Don't repeat prompts",
    n_noprompt_d: "Model shouldn't include instructions in transcription",
    meta_title: "Yo'l 2 — Gemini STT Metadata Filter",
    meta_desc: "Clean up errors and unwanted text in metadata. When enabled, applies automatically during pipeline.",
    m1: "① Capitalize first letter",
    m1_d: "First word of every sentence starts with uppercase.",
    m2: "② Write numbers as words",
    m2_d: "Numbers are written as text.",
    m3: "③ Fix apostrophe",
    m3_d: "ʻ and ʼ → standard ' apostrophe.",
    m4: "④ Remove duplicate text",
    m4_d: "If text is written twice, one is removed.",
    m5: "⑤ Add punctuation",
    m5_d: "Adds a period if sentence doesn't end with . ! ?",
    m6: "⑥ Fix quotes",
    m6_d: "\" \" and ' ' → standard \" quotes.",
    m7: "⑦ Fix case after sentence end",
    m7_d: "Uppercase letter after . ! ?",
    m8: "⑧ Remove double spaces",
    m8_d: "Two+ spaces → one.",
    m9: "⑨ Clean JSON artifacts",
    m9_d: "Residues of {'transcription': '...'} are cleaned.",
    m10: "⑩ Three dots → ellipsis",
    m10_d: "'...' → '…'",
    m11: "⑪ Replace hyphen with dash",
    m11_d: "' - ' → '—'.",
    m12: "⑫ Cyrillic chars → filter",
    m12_d: "Segment with Cyrillic is removed.",
    m13: "⑬ Clean bracket artifacts",
    m13_d: "Residues of [{json...}] are cleaned.",
    m14: "⑭ Clean HTML tags",
    m14_d: "<tag> characters are removed.",
    m15: "⑮ Digits only → filter",
    m15_d: "Digit-only text ('497347') is removed.",
    m16: "⑯ Fix broken hyphen",
    m16_d: "'so'z- so'z' → 'so'z-so'z'.",
    m17: "⑰ Clean multiple commas",
    m17_d: "',,,,' → ','.",
    cols_title: "Select Metadata Columns",
    col_always: "(always)",
    format_title: "File name",
    out_v1: "→ Path 1 (MohirDev API) output file",
    out_v2: "→ Path 2 (Gemini STT) output file",
    out_v3: "→ Path 3 (Python Filter) output file",
    out_name_label: "Metadata filename",
    write_title: "Write Mode",
    wm_new: "📄 Create new file",
    wm_append: "➕ Append to existing",
    wm_hint: "Append: metadata from multiple URLs is merged into one file",
    format_title2: "Format",
    start_btn: "▶ Run",
    status_ready: "Ready",
    status_preparing: "Preparing...",
    status_running: "Running...",
    status_done: "✅ Done!",
    status_error: "Error",
    status_session: "Session expired",
    stats_title: "Statistics",
    stat_yol1: "PATH 1",
    stat_yol2: "PATH 2",
    stat_total: "Total",
    stat_approved: "Approved",
    stat_pending: "Pending",
    stat_rejected: "Rejected",
    stat_filtered: "Filtered",
    log_title: "Log",
    files_meta_title: "Ready files — Metadata",
    files_none: "No files yet",
    files_zip_title: "Audio files — ZIP",
    zip_btn: "🗜 Zip audio files",
    zip_none: "No ZIP yet",
    results_title: "View Results",
    prev_btn: "◀ Previous",
    next_btn: "Next ▶",
    transcript_label: "Transcript",
    no_results: "No results...",
    download_btn: "⬇ Download",
    delete_title: "Delete",
    delete_confirm: "Confirm deletion?",
    folder_found: "audio files found",
    hf_settings: "HuggingFace Settings",
    hf_token_label2: "HF Token",
    hf_org_label: "Organization name",
    hf_repo_label: "Repository name",
    hf_private_lbl: "🔒 Private repo",
    hf_auto_badge: "✅ Pipeline result loaded automatically",
    hf_jsonl_lbl: "JSONL file",
    hf_csv_lbl: "CSV file",
    hf_audio_lbl: "Audio folder",
    hf_warn: "⚠️ First run the pipeline on the \"4 · Run\" tab. Data will be filled automatically after completion.",
    push_format_title: "Push format",
    push_jsonl_btn: "📄 Push as JSONL",
    push_csv_btn: "📊 Push as CSV",
    push_btn_jsonl2: "🚀 Push to HuggingFace (JSONL)",
    push_btn_csv2: "🚀 Push to HuggingFace (CSV)",
    push_ready: "Ready",
    aug_title: "Dataset Augmentation",
    aug_cpu: "CPU cores (num_proc)",
    aug_prob_lbl: "Augmentation probability (%)",
    aug_prob_hint: "— What percent of audio gets noise",
    aug_skip_lbl: "Skip augmentation",
    aug_skip_d: "Just convert to HF Dataset format without adding noise",
    aug_test_btn: "🧪 Test on 100 items",
    aug_continue_btn: "▶ Continue remaining",
    aug_full_btn: "⚙ Augment all",
    aug_review_title: "View Augmented Result",
    aug_text_lbl: "Text",
    map_title: "HuggingFace Mapping",
    map_model_lbl: "Model name",
    map_output_lbl: "Output folder (Output Dir)",
    map_token_lbl: "HuggingFace Token",
    map_ds_lbl: "HuggingFace dataset names (comma-separated)",
    map_cpu_lbl: "CPU cores (num_proc)",
    map_start_btn: "⚙ Start mapping",
    train_title: "Model Training",
    train_wandb_lbl: "W&B API Key",
    train_hf_lbl: "HuggingFace Token (HF_TOKEN)",
    train_model_lbl: "Model",
    train_ds_lbl: "Mapped datasets (comma-separated)",
    train_cpu_lbl: "CPU (num_proc)",
    train_split_lbl: "Train/Test split",
    train_output_lbl: "Output Dir",
    train_batch_lbl: "Batch Size",
    train_grad_lbl: "Grad Accumulation",
    train_lr_lbl: "Learning Rate",
    train_epochs_lbl: "Epochs",
    train_ddpt_lbl: "DDP Timeout",
    train_workers_lbl: "Dataloader Workers",
    train_pin_lbl: "Pin Memory",
    train_ddpu_lbl: "DDP Find Unused",
    train_resume_lbl: "Resume Checkpoint",
    train_start_btn: "🚀 Start training",
    alert_aisha: "Enter MohirDev STT API key (for Path 1).",
    alert_gemini: "Enter Gemini API key (for Path 2).",
    alert_yturl: "Enter YouTube URL.",
    alert_hfname: "Enter HuggingFace dataset name.",
    alert_jsonfile: "Select a JSON file.",
    alert_localdir: "Select audio folder.",
    alert_session: "Session expired — please log in again.",
  }
};

var currentLang = localStorage.getItem('lang') || 'uz';

// Helper: translate a key using currently-active language
function T(key, fallback) {
  var dict = LANG[currentLang] || LANG.uz;
  return (dict[key] !== undefined ? dict[key] : (fallback !== undefined ? fallback : key));
}

// Flag SVG map (reused in setLang and dropdown)
var FLAG_SVG = {
  uz: `<svg width="20" height="14" viewBox="0 0 20 14" xmlns="http://www.w3.org/2000/svg" style="border-radius:2px;display:block"><rect width="20" height="4.67" fill="#0099B5"/><rect y="4.67" width="20" height="4.67" fill="#FFFFFF"/><rect y="9.33" width="20" height="4.67" fill="#1EB53A"/><rect y="4.27" width="20" height="0.8" fill="#CE1126"/><rect y="8.93" width="20" height="0.8" fill="#CE1126"/><circle cx="3.2" cy="2.33" r="1.5" fill="#FFFFFF"/><circle cx="3.9" cy="2.33" r="1.5" fill="#0099B5"/></svg>`,
  ru: `<svg width="20" height="14" viewBox="0 0 20 14" xmlns="http://www.w3.org/2000/svg" style="border-radius:2px;display:block"><rect width="20" height="4.67" fill="#fff"/><rect y="4.67" width="20" height="4.67" fill="#0039A6"/><rect y="9.33" width="20" height="4.67" fill="#D52B1E"/></svg>`,
  en: `<svg width="20" height="14" viewBox="0 0 20 14" xmlns="http://www.w3.org/2000/svg" style="border-radius:2px;display:block"><rect width="20" height="14" fill="#012169"/><path d="M0 0L20 14M20 0L0 14" stroke="#fff" stroke-width="2.8"/><path d="M0 0L20 14M20 0L0 14" stroke="#C8102E" stroke-width="1.6"/><path d="M10 0V14M0 7H20" stroke="#fff" stroke-width="4"/><path d="M10 0V14M0 7H20" stroke="#C8102E" stroke-width="2.4"/></svg>`
};

function setLang(lang) {
  currentLang = lang;
  localStorage.setItem('lang', lang);

  var t = LANG[lang];
  if (!t) return;

  // Flag + name
  var flagEl = document.getElementById('lang-flag');
  var nameEl = document.getElementById('lang-name');
  if (flagEl) flagEl.innerHTML = FLAG_SVG[lang] || '';
  if (nameEl) nameEl.textContent = t.lang_name || lang;

  // Active lang option
  ['uz', 'ru', 'en'].forEach(function (l) {
    var b = document.getElementById('lang-' + l);
    if (b) b.classList.toggle('active', l === lang);
  });

  // Close dropdown
  var dd = document.getElementById('lang-dropdown');
  if (dd) dd.classList.remove('open');

  // data-i18n → textContent
  document.querySelectorAll('[data-i18n]').forEach(function (el) {
    var key = el.getAttribute('data-i18n');
    if (t[key] !== undefined) el.textContent = t[key];
  });

  // data-i18n-placeholder → placeholder
  document.querySelectorAll('[data-i18n-placeholder]').forEach(function (el) {
    var key = el.getAttribute('data-i18n-placeholder');
    if (t[key] !== undefined) el.placeholder = t[key];
  });

  // data-i18n-title → title attribute
  document.querySelectorAll('[data-i18n-title]').forEach(function (el) {
    var key = el.getAttribute('data-i18n-title');
    if (t[key] !== undefined) el.title = t[key];
  });

  // data-i18n-aria → aria-label attribute
  document.querySelectorAll('[data-i18n-aria]').forEach(function (el) {
    var key = el.getAttribute('data-i18n-aria');
    if (t[key] !== undefined) el.setAttribute('aria-label', t[key]);
  });

  // Refresh empty file-list / zip-list messages if currently shown
  var fl = document.getElementById('file-list');
  if (fl && fl.children.length === 1 && fl.firstElementChild.classList.contains('empty-list')) {
    fl.firstElementChild.textContent = T('files_none');
  }
  var zl = document.getElementById('zip-list');
  if (zl && zl.children.length === 1 && zl.firstElementChild.classList.contains('empty-list')) {
    zl.firstElementChild.textContent = T('zip_none');
  }

  // Document lang attr (for screen readers + CSS hooks)
  document.documentElement.setAttribute('lang', lang);
}

// ─────────────────────────────────────────────
// LOCAL AUDIO FILE SELECT
// Works for both:
//   - individual file picks (Ctrl+click multi-select in dialog)
//   - whole folders (if input has webkitdirectory)
// Collects valid audio files → uploaded on Start.
// ─────────────────────────────────────────────
function onLocalFolderSelect(inp) {
  if (!inp.files || inp.files.length === 0) return;

  const audioExts = ['.wav', '.mp3', '.ogg', '.flac', '.m4a'];
  const audioFiles = Array.from(inp.files).filter(f =>
    audioExts.some(ext => f.name.toLowerCase().endsWith(ext))
  );

  const label = document.getElementById('local-drop-label');
  const sub   = document.getElementById('local-drop-sub');
  const drop  = document.getElementById('local-drop');
  const dir   = document.getElementById('local-dir');

  // Folder name detection (works for webkitdirectory mode)
  const firstFile  = inp.files[0];
  const relPath    = firstFile.webkitRelativePath || '';
  const folderName = relPath.includes('/') ? relPath.split('/')[0] : '';

  if (audioFiles.length === 0) {
    // Nothing usable — warn clearly
    const picked = folderName || (inp.files.length + ' ta fayl');
    if (label) label.textContent = `⚠️ ${picked} — audio fayl topilmadi`;
    if (sub)   sub.textContent   = `Qo'llab-quvvatlanadigan formatlar: .wav .mp3 .ogg .flac .m4a`;
    if (drop)  drop.style.borderColor = 'var(--warning)';
    window._localAudioFiles = null;
    if (dir) dir.value = '';
    return;
  }

  // Build summary
  const totalBytes = audioFiles.reduce((s, f) => s + (f.size || 0), 0);
  const mb         = (totalBytes / 1024 / 1024).toFixed(1);
  const firstFew   = audioFiles.slice(0, 3).map(f => f.name).join(', ');
  const more       = audioFiles.length > 3 ? `, +${audioFiles.length - 3} ta…` : '';
  const title      = folderName
    ? `📁 ${folderName}  ·  ${audioFiles.length} ta audio  ·  ${mb} MB`
    : `🎵 ${audioFiles.length} ta audio  ·  ${mb} MB`;

  if (label) label.textContent = title;
  if (sub)   sub.textContent   = firstFew + more;
  if (drop)  drop.style.borderColor = 'var(--p)';
  if (dir && !dir.value.trim()) dir.value = folderName || 'selected_files';
  window._localAudioFiles = audioFiles;

  // Warn if near upload limit
  if (totalBytes > 1.8 * 1024 * 1024 * 1024) {
    toast(`Jami ${mb} MB — 2 GB chegarasidan yuqori. Kamroq fayl tanlang.`, 'warning', 6000);
  }
}

// ─────────────────────────────────────────────
// TAB SWITCHING
// ─────────────────────────────────────────────
function switchTab(idx) {
  document.querySelectorAll('.tab-btn').forEach((b, i) => b.classList.toggle('active', i === idx));
  document.querySelectorAll('.app .tab-content').forEach((c, i) => c.classList.toggle('active', i === idx));
}

// ─────────────────────────────────────────────
// SOURCE RADIO → show/hide source inputs
// ─────────────────────────────────────────────
function updateSourceUI(val) {
  document.getElementById('src-yt').style.display    = val === 'YouTube URL' ? 'flex' : 'none';
  document.getElementById('src-json').style.display  = val === 'JSON URL fayl' ? '' : 'none';
  document.getElementById('src-hf').style.display    = val === 'HuggingFace Dataset' ? '' : 'none';
  document.getElementById('src-local').style.display = val === 'Local papka' ? '' : 'none';
}

// ─────────────────────────────────────────────
// HF PANEL SWITCHING
// ─────────────────────────────────────────────
function switchHFTab(mode) {
  document.getElementById('hf-jsonl-panel').style.display = mode === 'jsonl' ? '' : 'none';
  document.getElementById('hf-csv-panel').style.display   = mode === 'csv'   ? '' : 'none';
  document.getElementById('hf-tab-jsonl-btn').style.borderColor = mode === 'jsonl' ? 'var(--p)' : 'var(--border-strong)';
  document.getElementById('hf-tab-jsonl-btn').style.color       = mode === 'jsonl' ? 'var(--p)' : 'var(--text-subtle)';
  document.getElementById('hf-tab-csv-btn').style.borderColor   = mode === 'csv'   ? 'var(--p)' : 'var(--border-strong)';
  document.getElementById('hf-tab-csv-btn').style.color         = mode === 'csv'   ? 'var(--p)' : 'var(--text-subtle)';
}

// ─────────────────────────────────────────────
// STATUS BADGE
// ─────────────────────────────────────────────
function setStatus(cls, text) {
  const badge = document.getElementById('status-badge');
  badge.className = 'status-badge' + (cls ? ' ' + cls : '');
  document.getElementById('status-text').textContent = text;
}

// ─────────────────────────────────────────────
// FILE LIST RENDER
// ─────────────────────────────────────────────
function renderFileList(files) {
  const fl = document.getElementById('file-list');
  if (!fl) return;
  if (!files || files.length === 0) {
    fl.innerHTML = `<div class="empty-list">${T('files_none')}</div>`;
    return;
  }
  fl.innerHTML = '';
  files.forEach(f => {
    const safeP = f.path.replace(/\\/g, '\\\\').replace(/'/g, "\\'");
    fl.innerHTML += `
      <div class="file-item">
        <div><div class="file-item-name">${f.name}</div><div class="file-item-size">${f.size}</div></div>
        <div style="display:flex;gap:6px;align-items:center">
          <a class="file-item-dl" href="/api/download?file=${encodeURIComponent(f.path)}" download="${f.name}">${T('download_btn')}</a>
          <button class="file-item-del" onclick="deleteFile('${safeP}')" title="${T('delete_title')}">🗑</button>
        </div>
      </div>`;
  });
}

// ─────────────────────────────────────────────
// LANGUAGE DROPDOWN
// ─────────────────────────────────────────────
function toggleLangDropdown() {
  var dd = document.getElementById('lang-dropdown');
  var trigger = document.getElementById('lang-trigger');
  if (!dd || !trigger) return;
  if (dd.classList.contains('open')) {
    dd.classList.remove('open');
    return;
  }
  var rect = trigger.getBoundingClientRect();
  dd.style.left = rect.left + 'px';
  dd.style.top  = (rect.top - 8) + 'px';
  dd.style.transform = 'translateY(-100%)';
  dd.classList.add('open');
}

document.addEventListener('click', function (e) {
  if (!e.target.closest('.lang-wrap')) {
    var dd = document.getElementById('lang-dropdown');
    if (dd) dd.classList.remove('open');
  }
});

// ─────────────────────────────────────────────
// THEME
// ─────────────────────────────────────────────
function toggleTheme() {
  var isDark = document.documentElement.getAttribute('data-theme') === 'dark';
  var newTheme = isDark ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', newTheme);
  localStorage.setItem('theme', newTheme);
  var icon = document.getElementById('theme-icon');
  if (icon) icon.textContent = newTheme === 'dark' ? '☀️' : '🌙';
}

// ─────────────────────────────────────────────
// DOM READY
// ─────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', function () {
  var jf = document.getElementById('json-file');
  if (jf) jf.addEventListener('change', function () {
    document.getElementById('json-fname').textContent = this.files[0]?.name || T('json_empty');
  });
});
