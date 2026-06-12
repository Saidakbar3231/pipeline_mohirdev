let _pollTimer = null;
let _taskPollTimer = null;
let _state_last_audio_dir = '';
var _state_last_jsonl_for_aug = "";
let _augReviewIdx = 0;
let _augReviewDataset = 'birlashtirilgan_dataset_augmented';
// True once the review panel has successfully rendered at least one row. Proves
// the dataset exists and is readable, so navigation stays enabled even when the
// live task status no longer reports done=true (e.g. after a server restart).
// Reset to false whenever a new augmentation run starts.
let _augReviewLoaded = false;
let _hfJsonlPath = '', _hfCsvPath = '';

// Last-observed augmentation status, written by updateTaskStatus and consulted
// by doAugReview. Prevents the FE from fetching review endpoints while the BE
// is still producing the dataset (or has errored out producing it).
let _augStatus = { running: false, done: false, error: '' };

// Monotonic generation counter. Each new task-status polling cycle bumps it;
// in-flight responses from prior cycles are dropped so a delayed reply cannot
// repaint stale state over a fresh task.
let _taskPollGen = 0;
// Same pattern for the pipeline progress poller.
let _progressPollGen = 0;

// ─── IN-FLIGHT GUARDS (prevent double-click / duplicate backend jobs) ──
const _inflight = { aug: false, map: false, train: false, push: false };
function _lock(key, onButtons) {
  if (_inflight[key]) return false;
  _inflight[key] = true;
  (onButtons || []).forEach(id => { const b = _get(id); if (b) b.disabled = true; });
  return true;
}
function _unlock(key, onButtons) {
  _inflight[key] = false;
  (onButtons || []).forEach(id => { const b = _get(id); if (b) b.disabled = false; });
}

// ─── SAFE DOM HELPERS ─────────────────────────────
function _get(id) {
  return document.getElementById(id);
}
function _setDisplay(id, show) {
  const el = _get(id);
  if (!el) return;
  // "block" forces show even when CSS sets display:none (e.g. #notify-box)
  el.style.display = show ? "block" : "none";
}
function _setText(id, text) {
  const el = _get(id);
  if (el) el.textContent = text;
}
function _val(id) {
  const el = _get(id);
  if (!el) {
    if (typeof console !== "undefined") console.warn(`[api.js] _val: #${id} not found`);
    return '';
  }
  return el.value;
}
function _checked(id, fallback) {
  const el = _get(id);
  if (!el) {
    if (typeof console !== "undefined") console.warn(`[api.js] _checked: #${id} not found`);
    return fallback ?? false;
  }
  return el.checked;
}

// ─── PIPELINE FILTER VISIBILITY ───────────────────
function _applyPipelineFilters(v) {
  const p = v.includes("Yo'l 1") ? "yol1"
           : v.includes("Yo'l 2") ? "yol2"
           : "yol3";
  document.body.dataset.pipeline = p;
}

// ─── PIPELINE RADIO → show/hide rows ──────────────
document.querySelectorAll('input[name="pipeline"]').forEach(r => r.addEventListener('change', function() {
  const v = this.value;
  _applyPipelineFilters(v);
  _setDisplay('row-gemini-key', v.includes("Yo'l 2"));
  _setDisplay('row-aisha-key',  v.includes("Yo'l 1"));

  if (v.includes("Yo'l 3")) {
    const s2 = _get('s2');
    if (s2) s2.checked = true;
    updateSourceUI('JSON URL fayl');
    document.querySelectorAll('#source-radio .radio-item').forEach(ri => {
      const inp = ri.querySelector('input');
      if (!inp) return;
      ri.style.opacity = inp.value === 'JSON URL fayl' ? '1' : '0.3';
      inp.disabled = inp.value !== 'JSON URL fayl';
    });
  } else {
    document.querySelectorAll('#source-radio .radio-item').forEach(ri => {
      ri.style.opacity = '1';
      const inp = ri.querySelector('input');
      if (inp) inp.disabled = false;
    });
  }
  _setDisplay('out-v1-wrap', v.includes("Yo'l 1"));
  _setDisplay('out-v2-wrap', v.includes("Yo'l 2"));
  _setDisplay('out-v3-wrap', v.includes("Yo'l 3"));

  const isV3 = v.includes("Yo'l 3");
  _setText('stats-label-v1',
    v.includes("Yo'l 1") ? "YOʼL 1" : v.includes("Yo'l 2") ? "YOʼL 2" : "YOʼL 3");
  _setDisplay('stats-block-v2', !isV3);
}));

// ─── SOURCE RADIO ─────────────────────────────────
document.querySelectorAll('input[name="source"]').forEach(r => r.addEventListener('change', function() {
  updateSourceUI(this.value);
}));

// Sync UI to HTML-default :checked radios on initial load
// (change events don't auto-fire, so stats label / src panels can start stale)
(function _syncOnLoad() {
  const p = document.querySelector('input[name="pipeline"]:checked');
  if (p) p.dispatchEvent(new Event('change'));
  const s = document.querySelector('input[name="source"]:checked');
  if (s) s.dispatchEvent(new Event('change'));
})();

// ─── MODEL SELECT CHANGE HANDLERS ─────────────────
(function() {
  const mapSel = _get('map-model-name');
  if (mapSel) mapSel.addEventListener('change', function() {
    const modelSlug = this.value.split('/').pop().replace(/-/g, '_');
    const out = _get('map-output-dir');
    if (out) out.value = `full_mapping_dataset_${modelSlug}`;
  });

  const trainSel = _get('train-model');
  if (trainSel) trainSel.addEventListener('change', function() {
    const modelSlug = this.value.split('/').pop().replace(/-/g, '_');
    const ds = _get('train-ds-dirs');
    const out = _get('train-output');
    if (ds) ds.value = `full_mapping_dataset_${modelSlug}`;
    if (out) out.value = `./whisper_${modelSlug}_dv_v2`;
  });
})();

document.addEventListener('DOMContentLoaded', function() {
  // Pipeline filter visibility — sahifa yuklanishida boshlang'ich holat
  const initPipeline = document.querySelector('input[name="pipeline"]:checked');
  if (initPipeline) _applyPipelineFilters(initPipeline.value);

  const m = _get('map-model-name');
  if (m) m.dispatchEvent(new Event('change'));
  const t = _get('train-model');
  if (t) t.dispatchEvent(new Event('change'));
});

// ─── BUILD FORM DATA ──────────────────────────────
function buildParams() {
  const pipeline = document.querySelector('input[name="pipeline"]:checked')?.value || '';
  const source   = document.querySelector('input[name="source"]:checked')?.value || '';
  const writeMode = document.querySelector('input[name="write-mode"]:checked')?.value || 'new';

  const cols = [];
  const colMap = {
    'col-file-name':'file_name','col-transcription':'transcription',
    'col-duration':'duration','col-source':'source'
  };
  for (const [id, col] of Object.entries(colMap)) {
    if (_checked(id, false)) cols.push(col);
  }

  const fd = new FormData();
  fd.append('pipeline_choice', pipeline);
  fd.append('source_type', source);
  fd.append('gemini_api_key', _val('gemini-api-key'));
  fd.append('aisha_api_key',  _val('aisha-api-key'));
  fd.append('yt_url',         _val('yt-url'));
  fd.append('hf_name',        _val('hf-name'));
  fd.append('hf_config',      _val('hf-config'));
  fd.append('hf_split',       _val('hf-split'));
  fd.append('hf_dataset_token', _val('hf-dataset-token'));
  // Local folder: if the user picked a folder via the browser picker,
  // upload the audio files directly (browsers can't expose absolute paths).
  if (window._localAudioFiles && window._localAudioFiles.length > 0) {
    window._localAudioFiles.forEach(f => fd.append('local_files', f, f.name));
    fd.append('local_dir', '__uploaded__');  // sentinel; backend replaces with temp dir
  } else {
    fd.append('local_dir', _val('local-dir'));
  }
  fd.append('dur_min',        _val('dur-min'));
  fd.append('dur_max',        _val('dur-max'));
  fd.append('noise_reduce',   _checked('noise-reduce', true));
  fd.append('noise_strength', _val('noise-strength'));
  fd.append('filter_music',    _checked('filter-music', true));
  fd.append('filter_multi',    _checked('filter-multi', true));
  fd.append('filter_noisy',    _checked('filter-noisy', false));
  fd.append('filter_silence',  _checked('filter-silence', true));
  fd.append('filter_capitalize',       _checked('filter-capitalize', true));
  fd.append('filter_num_words',        _checked('filter-num-words', true));
  fd.append('filter_latin_only',       _checked('filter-latin-only', true));
  fd.append('filter_no_translate',     _checked('filter-no-translate', true));
  fd.append('filter_no_noise_tags',    _checked('filter-no-noise-tags', true));
  fd.append('filter_no_repeat_prompt', _checked('filter-no-repeat-prompt', true));
  fd.append('norm_capitalize',    _checked('norm-capitalize', true));
  fd.append('norm_num_words',     _checked('norm-num-words', true));
  fd.append('norm_apostrophe',    _checked('norm-apostrophe', true));
  fd.append('norm_duplicate',     _checked('norm-duplicate', true));
  fd.append('norm_punct',         _checked('norm-punct', false));
  fd.append('norm_quotes',        _checked('norm-quotes', true));
  fd.append('norm_sentence_case', _checked('norm-sentence-case', true));
  fd.append('norm_double_space',  _checked('norm-double-space', true));
  fd.append('norm_clean_json',    _checked('norm-clean-json', true));
  fd.append('norm_ellipsis',      _checked('norm-ellipsis', false));
  fd.append('norm_dash',          _checked('norm-dash', false));
  fd.append('norm_cyrillic',      _checked('norm-cyrillic', true));
  fd.append('norm_brackets',      _checked('norm-brackets', true));
  fd.append('norm_html',          _checked('norm-html', true));
  fd.append('norm_only_digits',   _checked('norm-only-digits', true));
  fd.append('norm_broken_hyphen', _checked('norm-broken-hyphen', false));
  fd.append('norm_multi_comma',   _checked('norm-multi-comma', true));
  const _outName = _val('out-name') || 'metadata';
  fd.append('out_name_v1', _outName);
  fd.append('out_name_v2', _outName);
  fd.append('out_name_v3', _outName);
  fd.append('write_mode',  writeMode);
  fd.append('fmt_jsonl',   _checked('fmt-jsonl', true));
  fd.append('fmt_csv',     _checked('fmt-csv', true));
  fd.append('selected_cols', JSON.stringify(cols));

  const jsonFileInp = _get('json-file');
  if (jsonFileInp && jsonFileInp.files && jsonFileInp.files[0]) fd.append('json_file', jsonFileInp.files[0]);

  return fd;
}

// ─── START PIPELINE ───────────────────────────────
function _validatePipelineInputs() {
  const pipeline = document.querySelector('input[name="pipeline"]:checked')?.value || '';
  const source   = document.querySelector('input[name="source"]:checked')?.value || '';
  const geminiKey = (_val('gemini-api-key') || '').trim();
  const aishaKey  = (_val('aisha-api-key')  || '').trim();

  if (pipeline.includes("Yo'l 1") && !aishaKey) { alert(T('alert_aisha')); return false; }
  if (pipeline.includes("Yo'l 2") && !geminiKey) { alert(T('alert_gemini')); return false; }

  // Source-specific required inputs
  if (source === 'YouTube URL' && !(_val('yt-url') || '').trim()) {
    alert(T('alert_yturl')); return false;
  }
  if (source === 'HuggingFace Dataset' && !(_val('hf-name') || '').trim()) {
    alert(T('alert_hfname')); return false;
  }
  if (source === 'JSON URL fayl') {
    const jf = _get('json-file');
    if (!jf || !jf.files || !jf.files[0]) { alert(T('alert_jsonfile')); return false; }
  }
  if (source === 'Local papka') {
    const hasUpload = window._localAudioFiles && window._localAudioFiles.length > 0;
    const hasPath   = (_val('local-dir') || '').trim();
    if (!hasUpload && !hasPath) { alert(T('alert_localdir')); return false; }
  }
  return true;
}

async function startPipeline() {
  if (!_validatePipelineInputs()) return;

  const btn = _get('start-btn');
  if (btn) { btn.disabled = true; btn.textContent = '▶ Ishga tushur'; }
  const stopBtn = _get('stop-btn');
  if (stopBtn) { stopBtn.style.display = 'none'; stopBtn.disabled = false; stopBtn.textContent = '⏹ To\'xtatish'; }
  setStatus('running', T('status_preparing'));
  _setDisplay('notify-box', false);
  _setText('log-box', '');
  _setDisplay('progress-wrap', true);
  const pb = _get('progress-bar');
  if (pb) pb.style.width = '0%';
  _setText('progress-pct', '0%');
  _setText('progress-stage', T('status_preparing'));
  // Reset HF-push card so stale paths from previous run aren't shown
  _setDisplay('hf-auto-info', false);
  _setDisplay('hf-no-pipeline', true);
  ['hf-jsonl-path','hf-csv-path','hf-audio-path'].forEach(id => {
    const el = _get(id); if (el) el.value = '';
  });
  // Reset stats chips so previous run's numbers don't linger
  ['total','approved','pending','rejected','filtered'].forEach(k => {
    _setText(`s1-${k}`, 0);
    _setText(`s2-${k}`, 0);
  });
  _setText('rev-idx', '0 / 0');

  try {
    const fd = buildParams();
    // If uploading local audio files, show an interim status while bytes transfer
    const isUploading = window._localAudioFiles && window._localAudioFiles.length > 0;
    if (isUploading) {
      setStatus('running', T('uploading') || 'Yuklanmoqda...');
      _setText('progress-stage', T('uploading') || 'Yuklanmoqda...');
    }
    const r = await fetch('/api/start', {method:'POST', body: fd});
    let d = null;
    try { d = await r.json(); } catch (_) { d = null; }
    if (!r.ok || !d || d.error) {
      alert((d && d.error) || ('HTTP ' + r.status));
      if (btn) btn.disabled = false;
      setStatus('', T('status_ready'));
      return;
    }
    setStatus('running', T('status_running'));
    startPolling();
  } catch(e) {
    alert(T('status_error') + ': ' + e);
    if (btn) btn.disabled = false;
    setStatus('', T('status_error'));
  }
}

// ─── STOP PIPELINE ────────────────────────────────
async function stopPipeline() {
  const stopBtn  = _get('stop-btn');
  const startBtn = _get('start-btn');
  if (stopBtn) { stopBtn.disabled = true; stopBtn.textContent = 'To\'xtatilmoqda...'; }
  try {
    const r = await fetch('/api/stop', { method: 'POST' });
    let d = null;
    try { d = await r.json(); } catch (_) { d = null; }
    if (!r.ok || !d || d.error) {
      alert((d && d.error) || ('HTTP ' + r.status));
      if (stopBtn) { stopBtn.disabled = false; stopBtn.textContent = '⏹ To\'xtatish'; }
    }
  } catch(e) {
    if (stopBtn) { stopBtn.disabled = false; stopBtn.textContent = '⏹ To\'xtatish'; }
  }
}

// ─── POLLING ──────────────────────────────────────
function startPolling() {
  if (_pollTimer) clearInterval(_pollTimer);
  _progressPollGen++;
  _pollTimer = setInterval(pollProgress, 2000);
}

async function pollProgress() {
  const myGen = _progressPollGen;
  try {
    const r = await fetch('/api/progress');
    if (myGen !== _progressPollGen) return;
    // Session expired / auth redirect → response is HTML, not JSON
    if (!r.ok || !(r.headers.get('content-type') || '').includes('application/json')) {
      clearInterval(_pollTimer);
      _progressPollGen++;
      const btn = _get('start-btn'); if (btn) btn.disabled = false;
      setStatus('', T('status_session'));
      alert(T('alert_session'));
      window.location.href = '/login';
      return;
    }
    const d = await r.json();
    if (myGen !== _progressPollGen) return;
    if (d && d.error) return;

    _setText('log-box', d.log || '');
    const lb = _get('log-box');
    if (lb) lb.scrollTop = 999999;

    const pct   = d.progress_pct || 0;
    const stage = d.progress_stage || '';
    const cur   = d.progress_cur  || 0;
    const total = d.progress_total || 0;
    if (d.running || (d.done && pct > 0)) {
      _setDisplay('progress-wrap', true);
      const pb = _get('progress-bar');
      if (pb) pb.style.width = pct + '%';
      _setText('progress-pct', pct + '%');
      _setText('progress-stage', stage);
      _setText('progress-count', total > 1 ? `${cur} / ${total} ta segment` : '');
    }

    const sv1 = d.stats_v1 || {};
    const sv2 = d.stats_v2 || {};
    ['total','approved','pending','rejected','filtered'].forEach(k => {
      _setText(`s1-${k}`, sv1[k]||0);
      _setText(`s2-${k}`, sv2[k]||0);
    });

    // Mirror the audio dir as soon as the backend reports it. Previously this
    // assignment lived inside the d.done branch, leaving startAugmentation
    // unable to find the dir if the user navigated to the augmentation tab
    // mid-pipeline.
    if (d.last_audio_dir) _state_last_audio_dir = d.last_audio_dir;
    if (d.last_jsonl)     _state_last_jsonl_for_aug = d.last_jsonl;

    if (d.notify) {
      _setText('notify-box', d.notify);
      _setDisplay('notify-box', true);
    }

    renderFileList(d.files || []);

    // Show/hide stop button based on running state
    const stopBtn  = _get('stop-btn');
    const startBtn = _get('start-btn');
    if (stopBtn) stopBtn.style.display = d.running ? 'inline-flex' : 'none';
    if (stopBtn && d.running) { stopBtn.disabled = false; stopBtn.textContent = '⏹ To\'xtatish'; }

    // Stopped (not running, not done) — show Continue button
    if (d.stopped && d.can_resume && !d.running) {
      clearInterval(_pollTimer);
      _progressPollGen++;
      if (startBtn) {
        startBtn.disabled = false;
        startBtn.textContent = '▶ Davom ettirish';
      }
      setStatus('', T('status_stopped') || 'To\'xtatildi');
    }

    if (d.done && !d.running) {
      clearInterval(_pollTimer);
      _progressPollGen++;
      if (startBtn) { startBtn.disabled = false; startBtn.textContent = '▶ Ishga tushur'; }
      setStatus('done', T('status_done'));
      const pb = _get('progress-bar');
      if (pb) pb.style.width = '100%';
      _setText('progress-pct', '100%');
      _setText('progress-stage', T('status_done'));
      if (d.review_total > 0) doReview(0);
      if (d.last_audio_dir) _state_last_audio_dir = d.last_audio_dir;

      if (d.last_jsonl || d.last_csv) {
        _state_last_jsonl_for_aug = d.last_jsonl;
        _setDisplay('hf-no-pipeline', false);
        _setDisplay('hf-auto-info', true);
        const j = _get('hf-jsonl-path');
        const c = _get('hf-csv-path');
        const a = _get('hf-audio-path');
        if (j && d.last_jsonl)    j.value = d.last_jsonl;
        if (c && d.last_csv)      c.value = d.last_csv;
        if (a && d.last_audio_dir)a.value = d.last_audio_dir;
      }
    }
  } catch(e) {
    console.error('Poll error:', e);
  }
}

// ─── REVIEW ───────────────────────────────────────
async function doReview(dir) {
  try {
    const r = await fetch(`/api/review?dir=${dir}`);
    if (!r.ok) { console.log('Review: no results'); return; }
    const d = await r.json();
    if (d.error) { console.log('Review error:', d.error); return; }
    // Backend returns 0-based idx; FE owns +1 display.
    _setText('rev-idx', `${(d.idx || 0) + 1} / ${d.total}`);
    const rt = _get('rev-text');
    if (rt) rt.value = d.transcription || '';
    _setText('rev-audio-name', d.file_name || '');

    const audioEl = _get('rev-audio');
    if (audioEl) {
      if (d.audio_url) {
        // Cache-bust so a new pipeline run that overwrites the same file path
        // is not masked by the browser's HTTP cache.
        const sep = d.audio_url.includes('?') ? '&' : '?';
        audioEl.src = d.audio_url + sep + 't=' + Date.now();
        audioEl.style.display = '';
        audioEl.load();
      } else {
        audioEl.style.display = 'none';
      }
    }

    const metaLines = Object.entries(d.meta||{}).map(([k,v]) => `${k}: ${v}`);
    _setText('rev-meta', metaLines.join('\n'));
  } catch(e) {
    console.error('Review error:', e);
  }
}

// ─── ZIP & DELETE ─────────────────────────────────
async function makeZip() {
  _setText('zip-status', '⏳ ZIP yaratilmoqda...');
  try {
    const r = await fetch('/api/zip', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({audio_dir: _state_last_audio_dir})
    });
    let d = null;
    try { d = await r.json(); } catch (_) { d = null; }
    if (!r.ok || !d || d.error) {
      _setText('zip-status', '❌ ' + ((d && d.error) || ('HTTP ' + r.status)));
      return;
    }
    _setText('zip-status', `✅ ${d.count} ta fayl, ${d.size_mb} MB`);
    loadZipList();
  } catch(e) {
    _setText('zip-status', '❌ ' + T('status_error') + ': ' + e);
  }
}

async function loadZipList() {
  try {
    const r = await fetch('/api/audio-zips');
    if (!r.ok) { console.warn('[api.js] loadZipList: HTTP', r.status); return; }
    const zips = await r.json();
    const zl = _get('zip-list');
    if (!zl) return;
    if (!zips || zips.length === 0) {
      zl.innerHTML = `<div class="empty-list">${T('zip_none')}</div>`;
      return;
    }
    zl.innerHTML = '';
    zips.forEach(z => {
      const safeP = z.path.replace(/\\/g,'\\\\').replace(/'/g,"\\'");
      zl.innerHTML += `
        <div class="file-item">
          <div><div class="file-item-name">🗜️ ${z.name}</div><div class="file-item-size">${z.size}</div></div>
          <div style="display:flex;gap:6px;align-items:center">
            <a class="file-item-dl" href="/api/download?file=${encodeURIComponent(z.path)}" download="${z.name}">⬇ Yuklab olish</a>
            <button class="file-item-del" onclick="deleteFile('${safeP}')" title="O'chirish">🗑</button>
          </div>
        </div>`;
    });
  } catch(e) { console.warn('[api.js] loadZipList failed:', e); }
}

async function deleteFile(path) {
  if (!confirm('O\'chirishni tasdiqlaysizmi?\n' + path)) return;
  try {
    const r = await fetch('/api/delete', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({path})
    });
    let d = null;
    try { d = await r.json(); } catch (_) { d = null; }
    if (!r.ok || !d || d.error) {
      alert('Xato: ' + ((d && d.error) || ('HTTP ' + r.status)));
      return;
    }
    loadZipList();
    refreshFileLists();
  } catch(e) { alert(T('status_error') + ': ' + e); }
}

async function refreshFileLists() {
  try {
    const r = await fetch('/api/progress');
    if (!r.ok) return;
    let d = null;
    try { d = await r.json(); } catch (_) { return; }
    if (!d || d.error) return;
    renderFileList(d.files || []);
  } catch(e) {}
}

// ─── HF PUSH ──────────────────────────────────────
async function doPush(mode) {
  if (!_lock('push')) return;
  const statusEl = _get(`hf-${mode}-status`);
  const pushBtns = document.querySelectorAll('[onclick^="doPush("]');
  pushBtns.forEach(b => b.disabled = true);
  if (statusEl) statusEl.textContent = '⏳ Push qilinmoqda...';

  const filePath = mode === 'jsonl' ? _val('hf-jsonl-path') : _val('hf-csv-path');
  const audioDir = _val('hf-audio-path');

  const bail = (msg) => {
    if (statusEl) statusEl.textContent = msg;
    pushBtns.forEach(b => b.disabled = false);
    _unlock('push');
  };

  if (!filePath)              return bail('❌ Avval pipeline ishga tushiring!');
  if (!_val('hf-token'))      return bail('❌ HF Token kiritilmagan!');
  if (!_val('hf-org'))        return bail('❌ Org nomi kiritilmagan!');
  if (!_val('hf-repo'))       return bail('❌ Repo nomi kiritilmagan!');

  const payload = {
    mode,
    token:     _val('hf-token'),
    org:       _val('hf-org'),
    repo:      _val('hf-repo'),
    file_path: filePath,
    audio_dir: audioDir,
    private:   _checked('hf-private', true),
  };

  try {
    const r = await fetch('/api/hf/push', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload)
    });
    let d = null;
    try { d = await r.json(); } catch (_) { d = null; }
    if (!r.ok || !d || d.error) {
      const msg = (d && d.error) || ('HTTP ' + r.status);
      if (statusEl) statusEl.textContent = '❌ ' + msg;
      return;
    }
    if (statusEl) statusEl.textContent = d.result || T('status_done');
  } catch(e) {
    if (statusEl) statusEl.textContent = '❌ ' + e;
  } finally {
    pushBtns.forEach(b => b.disabled = false);
    _unlock('push');
  }
}

// ─── AUGMENTATION, MAPPING, TRAINING ──────────────
async function startValidation() {
  try {
    const res = await fetch('/api/validation/start', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        auto_filter_failed: document.getElementById('auto-filter')?.checked || false,
        strict_augmented_only: document.getElementById('strict-only')?.checked || false
      })
    });
    let d = null;
    try { d = await res.json(); } catch (_) { d = null; }
    if (!res.ok || !d || d.error) {
      alert((d && d.error) || ('HTTP ' + res.status));
      return;
    }
    const sec = _get('val-log-section'); if (sec) sec.style.display = '';
    const badge = _get('val-status-badge'); if (badge) badge.style.display = '';
    pollTaskStatus();
  } catch(e) { alert((typeof T==='function'?T('status_error'):'Error') + ': ' + e); }
}

async function startAugmentation(mode = 'full') {
  if (!_lock('aug')) return;
  _augReviewIdx = 0;
  _augReviewLoaded = false;
  const desiredDataset = (mode === 'test')
    ? 'birlashtirilgan_dataset_test_out'
    : 'birlashtirilgan_dataset_augmented';
  try {
    const res = await fetch('/api/augmentation/start', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        mode: mode,
        aug_prob: (function(){ const n = parseFloat(_val('aug-prob')); return (Number.isFinite(n) ? n : 0) / 100.0; })(),
        num_proc: _val('aug-num-proc'),
        skip: _checked('aug-skip-checkbox', false),
        // Number.isFinite guard (not `|| 50`) so a literal 0% reaches the backend
        // as 0 (barely-audible noise) instead of being coerced back to 50.
        noise_level: (function(){ const n = parseFloat(document.getElementById('augNoiseLevel')?.value); return Number.isFinite(n) ? n : 50.0; })(),
        last_jsonl: _state_last_jsonl_for_aug || _val('hf-jsonl-path'),
        last_audio_dir: _state_last_audio_dir || _val('hf-audio-path')
      })
    });
    let d = null;
    try { d = await res.json(); } catch (_) { d = null; }
    if (!res.ok || !d || d.error) {
      alert((d && d.error) || ('HTTP ' + res.status));
      return;
    }
    // only mutate review dataset AFTER backend confirms success
    _augReviewDataset = desiredDataset;
    // Reset status snapshot so prior `done` flag from a previous run does
    // not let doAugReview hit a half-built dataset before the new worker
    // reports running=true.
    _augStatus = { running: true, done: false, error: '' };
    _setAugReviewButtonsEnabled(false);
    pollTaskStatus();
  } catch(e) { alert(T('status_error') + ': ' + e); }
  finally { _unlock('aug'); }
}

async function startMapping() {
  if (!_lock('map')) return;
  try {
    const res = await fetch('/api/mapping/start', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        model_name: _val('map-model-name'),
        output_dir: _val('map-output-dir'),
        hf_token:   _val('map-hf-token'),
        ds_names:   _val('map-ds-names'),
        num_proc:   _val('map-num-proc')
      })
    });
    let d = null;
    try { d = await res.json(); } catch (_) { d = null; }
    if (!res.ok || !d || d.error) {
      alert((d && d.error) || ('HTTP ' + res.status));
      return;
    }
    pollTaskStatus();
  } catch(e) { alert(T('status_error') + ': ' + e); }
  finally { _unlock('map'); }
}

async function startTraining() {
  if (!_lock('train')) return;
  try {
    const res = await fetch('/api/training/start', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        wandb_api_key: _val('train-wandb-key'),
        hf_token:      _val('train-hf-token'),
        model_name_or_path: _val('train-model'),
        ds_dirs:       _val('train-ds-dirs'),
        num_proc:      _val('train-num-proc'),
        train_test_split: _val('train-split'),
        output_dir:    _val('train-output'),
        per_device_train_batch_size: _val('train-batch'),
        gradient_accumulation_steps: _val('train-grad-acc'),
        learning_rate: _val('train-lr'),
        num_train_epochs: _val('train-epochs'),
        ddp_timeout: _val('train-ddp-timeout'),
        dataloader_num_workers: _val('train-workers'),
        dataloader_pin_memory: _checked('train-pin-mem', true),
        ddp_find_unused_parameters: _checked('train-ddp-unused', false),
        resume_from_checkpoint: _checked('train-resume', true)
      })
    });
    let d = null;
    try { d = await res.json(); } catch (_) { d = null; }
    if (!res.ok || !d || d.error) {
      alert((d && d.error) || ('HTTP ' + res.status));
      return;
    }
    pollTaskStatus();
  } catch(e) { alert(T('status_error') + ': ' + e); }
  finally { _unlock('train'); }
}

function pollTaskStatus() {
  // Bump generation before installing a new interval. updateTaskStatus
  // captures the current generation at request issue and discards stale
  // responses whose generation no longer matches.
  _taskPollGen++;
  if (!_taskPollTimer) _taskPollTimer = setInterval(updateTaskStatus, 2000);
}

async function updateTaskStatus() {
  // Capture the generation at request time. If the timer has been cleared
  // and re-created (new task started) before this fetch resolves, the gen
  // will no longer match and we drop the stale response.
  const myGen = _taskPollGen;
  try {
    const r = await fetch('/api/task_status');
    if (myGen !== _taskPollGen) return;
    if (!r.ok) return;
    const d = await r.json();
    if (myGen !== _taskPollGen) return;
    if (d.error) return;

    const aug = d.augmentation || {};
    // Snapshot for doAugReview's function guard, plus button enable state.
    _augStatus = {
      running: !!aug.running,
      done:    !!aug.done,
      error:   aug.error || ''
    };
    // A fresh run is producing a new dataset — invalidate any prior load so the
    // panel reloads from idx 0 when this run finishes.
    if (_augStatus.running) _augReviewLoaded = false;
    const _augReady = _augStatus.done && !_augStatus.running && !_augStatus.error;
    _setAugReviewButtonsEnabled(_augReviewLoaded || _augReady);
    // Auto-load the first result the moment augmentation finishes cleanly, so
    // the user never has to click a nav button to see anything.
    if (_augReady && !_augReviewLoaded) {
      loadAugReviewInitial();
    }
    const augBadge = _get('aug-status-badge');
    if (augBadge) {
      if (aug.running)      { augBadge.className = 'status-badge running'; _setText('aug-status-text', T('status_running')); }
      else if (aug.done)    { augBadge.className = 'status-badge done';    _setText('aug-status-text', aug.error ? T('status_error') : T('status_done')); }
      else                  { augBadge.className = 'status-badge';         _setText('aug-status-text', T('status_ready')); }
    }
    let logText = (aug.log || '') + (aug.error ? "\nError: " + aug.error : "");
    if (logText.length > 200000) {
      logText = "[...truncated...]\n" + logText.slice(-200000);
    }
    _setText('aug-log-box', logText);
    const alb = _get('aug-log-box');
    if (alb) {
      const atBottom = alb.scrollTop + alb.clientHeight >= alb.scrollHeight - 10;
      if (atBottom) alb.scrollTop = alb.scrollHeight;
    }

    const map = d.mapping || {};
    const mapBadge = _get('map-status-badge');
    if (mapBadge) {
      if (map.running)      { mapBadge.className = 'status-badge running'; _setText('map-status-text', T('status_running')); }
      else if (map.done)    { mapBadge.className = 'status-badge done';    _setText('map-status-text', map.error ? T('status_error') : T('status_done')); }
      else                  { mapBadge.className = 'status-badge';         _setText('map-status-text', T('status_ready')); }
    }
    _setText('map-log-box', (map.log || '') + (map.error ? "\nError: " + map.error : ""));
    const mlb = _get('map-log-box'); if (mlb) mlb.scrollTop = 999999;

    const train = d.training || {};
    const trainBadge = _get('train-status-badge');
    if (trainBadge) {
      if (train.running)    { trainBadge.className = 'status-badge running'; _setText('train-status-text', T('status_running')); }
      else if (train.done)  { trainBadge.className = 'status-badge done';    _setText('train-status-text', train.error ? T('status_error') : T('status_done')); }
      else                  { trainBadge.className = 'status-badge';         _setText('train-status-text', T('status_ready')); }
    }
    _setText('train-log-box', (train.log || '') + (train.error ? "\nError: " + train.error : ""));
    const tlb = _get('train-log-box'); if (tlb) tlb.scrollTop = 999999;

    const val = d.validation || {};
    const valBadge = _get('val-status-badge');
    const valText  = _get('val-status-text');
    if (valBadge && (val.running || val.done)) {
      valBadge.style.display = '';
      const sec = _get('val-log-section'); if (sec) sec.style.display = '';
      let cls = 'status-badge';
      let label = 'Validation: idle';
      if (val.running) {
        cls = 'status-badge running';
        label = 'Validation: running';
      } else if (val.done) {
        const s = (val.status || '').toLowerCase();
        if (s === 'pass')          { cls = 'status-badge done';    label = 'Validation: PASS'; }
        else if (s === 'fail')     { cls = 'status-badge';         label = 'Validation: FAIL'; }
        else if (s === 'critical') { cls = 'status-badge';         label = 'Validation: CRITICAL'; }
        else                       { cls = 'status-badge';         label = 'Validation: ' + (val.error || 'unknown'); }
      }
      valBadge.className = cls;
      if (valText) valText.textContent = label;
      // Severity color override (pass=green, fail=orange, critical=red)
      const sev = (val.status || '').toLowerCase();
      if (val.done) {
        if (sev === 'pass')          valBadge.style.background = '#1e7e34';
        else if (sev === 'fail')     valBadge.style.background = '#d97706';
        else if (sev === 'critical') valBadge.style.background = '#b91c1c';
        else                         valBadge.style.background = '';
      } else {
        valBadge.style.background = '';
      }
    }
    let valLog = (val.log || '') + (val.error ? "\nError: " + val.error : "");
    if (val.result) {
      valLog += "\n--- RESULT_JSON ---\n" + JSON.stringify(val.result, null, 2);
    }
    if (valLog.length > 200000) {
      valLog = "[...truncated...]\n" + valLog.slice(-200000);
    }
    _setText('val-log-box', valLog);
    const vlb = _get('val-log-box');
    if (vlb) {
      const atBottom = vlb.scrollTop + vlb.clientHeight >= vlb.scrollHeight - 10;
      if (atBottom) vlb.scrollTop = vlb.scrollHeight;
    }

    if (!aug.running && !map.running && !train.running && !val.running) {
      clearInterval(_taskPollTimer);
      _taskPollTimer = null;
      // Bump generation so any in-flight updateTaskStatus from a prior loop
      // is treated as stale when the next pollTaskStatus() is called.
      _taskPollGen++;
    }
  } catch(e) {}
}

// Fetch a single review row and paint it into the panel. Pure fetch+render —
// no index mutation, no guards — so it can be shared by both the navigation
// handler and the automatic initial load. Returns {ok, error, data}.
async function _fetchAndRenderAug(idx) {
  try {
    const res = await fetch(`/api/hf_dataset_review?dataset_path=${encodeURIComponent(_augReviewDataset)}&idx=${idx}`);
    let d = null;
    try { d = await res.json(); } catch (_) { d = null; }
    if (!res.ok || !d || d.error) {
      return { ok: false, error: (d && d.error) || ('HTTP ' + res.status), data: d };
    }
    _setText('aug-rev-idx', `${(d.idx || 0) + 1} / ${d.total}`);
    const rt = _get('aug-rev-text');
    if (rt) rt.value = d.text || '';

    const audioEl = _get('aug-rev-audio');
    if (audioEl) {
      if (d.audio_url) {
        const sep = d.audio_url.includes('?') ? '&' : '?';
        audioEl.src = d.audio_url + sep + 't=' + Date.now();
        audioEl.style.display = '';
        audioEl.load();
      } else {
        audioEl.style.display = 'none';
      }
    }
    return { ok: true, data: d };
  } catch(e) {
    console.log(e);
    return { ok: false, error: String(e) };
  }
}

async function doAugReview(dir) {
  // Function guard: never read review endpoints while the worker is actively
  // (re)writing the dataset — that could return half-built or stale rows.
  // Once we've successfully rendered a row (_augReviewLoaded), the dataset is
  // proven present, so navigation stays allowed even if the live status no
  // longer reports done=true (e.g. after a server restart).
  if (_augStatus.running) return;
  if (!_augReviewLoaded && (!_augStatus.done || _augStatus.error)) return;

  _augReviewIdx += dir;
  if (_augReviewIdx < 0) _augReviewIdx = 0;

  const r = await _fetchAndRenderAug(_augReviewIdx);
  if (!r.ok) {
    if (r.data && r.data.error) alert(r.data.error);
    // Roll back a forward step that overran the end so idx stays in range.
    if (_augReviewIdx > 0 && dir > 0) _augReviewIdx -= 1;
    return;
  }
  _augReviewLoaded = true;
}

// Automatically populate the panel with the first result (idx 0). Called on
// augmentation completion AND on page load: relies on the endpoint's own
// success/error response to detect whether a completed dataset exists on disk,
// so it stays silent (no alert, no state change) when there is nothing yet.
async function loadAugReviewInitial() {
  if (_augReviewLoaded) return;
  _augReviewIdx = 0;
  const r = await _fetchAndRenderAug(0);
  if (r.ok) {
    _augReviewLoaded = true;
    _setAugReviewButtonsEnabled(true);
  }
}

// Toggle the augmentation review buttons' enabled state from the latest
// status snapshot. UI guard mirrors the function guard in doAugReview.
// Selector strategy is robust to authoring style: matches inline-onclick
// buttons, an explicit data-attribute opt-in, and a stable container scope.
function _setAugReviewButtonsEnabled(enabled) {
  const seen = new Set();
  const collect = (sel) => document.querySelectorAll(sel).forEach(b => seen.add(b));
  collect('[onclick^="doAugReview("]');
  collect('[data-aug-review]');
  // Container-scoped fallback — any <button> inside #aug-review-controls
  // counts as a review control regardless of how its handler is attached.
  collect('#aug-review-controls button, #aug-review-controls [role="button"]');
  seen.forEach(b => { b.disabled = !enabled; });
}

// ─── INIT ─────────────────────────────────────────
loadZipList();

(function() {
  var savedTheme = localStorage.getItem('theme') || 'light';
  document.documentElement.setAttribute('data-theme', savedTheme);
  var icon = _get('theme-icon');
  if (icon) icon.textContent = savedTheme === 'dark' ? '☀️' : '🌙';

  setLang(currentLang);

  (async function checkOnLoad() {
    try {
      const r = await fetch('/api/progress');
      const d = await r.json();
      if (d.running) {
        const btn = _get('start-btn'); if (btn) btn.disabled = true;
        setStatus('running', T('status_running'));
        _setDisplay('progress-wrap', true);
        switchTab(3);
        _pollTimer = setInterval(pollProgress, 2000);
      } else if (d.done && (d.log || '').length > 0) {
        _setText('log-box', d.log || '');
        const lb = _get('log-box'); if (lb) lb.scrollTop = 999999;
        _setDisplay('progress-wrap', true);
        const pb = _get('progress-bar'); if (pb) pb.style.width = '100%';
        _setText('progress-pct', '100%');
        _setText('progress-stage', T('status_done'));
        setStatus('done', T('status_done'));
        renderFileList(d.files || []);
        if (d.notify) {
          _setText('notify-box', d.notify);
          _setDisplay('notify-box', true);
        }
        if (d.last_jsonl || d.last_csv) {
          _setDisplay('hf-no-pipeline', false);
          _setDisplay('hf-auto-info', true);
          const j = _get('hf-jsonl-path'); if (j && d.last_jsonl) j.value = d.last_jsonl;
          const c = _get('hf-csv-path');   if (c && d.last_csv)   c.value = d.last_csv;
          const a = _get('hf-audio-path'); if (a && d.last_audio_dir) a.value = d.last_audio_dir;
        }
      }

      pollTaskStatus();
    } catch(e) {}
    // Populate the augmented-review panel on load if a completed dataset
    // already exists on disk. Silent no-op (no alert) when there is none yet.
    loadAugReviewInitial();
  })();
})();
