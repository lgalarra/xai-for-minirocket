(() => {
  // Global toggle for UI animations (can be set later from the HTML UI).
  window.ANIMATE_TRANSITIONS = true;

  const extraForLabel = 30;

  function syncHeaderHeight() {
    const h = document.querySelector('header')?.offsetHeight || 0;
    document.documentElement.style.setProperty('--headerH', `${h}px`);
    if (document.body) document.body.style.paddingTop = `${h}px`;
  }
  syncHeaderHeight();
  window.addEventListener('resize', syncHeaderHeight);

  // Always start with the Summary modal closed.
  // This also fixes cases where the browser restores the previous DOM state (bfcache).
  function forceHideSummaryOverlay() {
    const o = document.getElementById('peSummaryOverlay');
    if (o) o.hidden = true;
  }
  forceHideSummaryOverlay();
  window.addEventListener('pageshow', forceHideSummaryOverlay);
  const _hdr = document.querySelector('header');
  if (_hdr && 'ResizeObserver' in window) {
    new ResizeObserver(syncHeaderHeight).observe(_hdr);
  }

  const PORT = 3000;
  if (typeof window.inProduction !== 'boolean') {
    window.inProduction = !(location.hostname === "localhost" || location.hostname === "127.0.0.1");
  }
  const inProduction = window.inProduction;

  const API_ORIGIN = inProduction ? "" : `http://localhost:${PORT}`;

  console.log(API_ORIGIN);


  function withApiOrigin(url) {
    if (!API_ORIGIN || !url) return url;
    if (url.startsWith('/api/') || url.startsWith('/output/')) return API_ORIGIN + url;
    return url;
  }

  function generateMiniRocketKernels() {
    const ks = [];
    let idx = 0;
    for (let a = 0; a < 9; a++) {
      for (let b = a + 1; b < 9; b++) {
        for (let c = b + 1; c < 9; c++) {
          const w = Array(9).fill(-1);
          w[a] = 2;
          w[b] = 2;
          w[c] = 2;
          ks.push({
            id: `K${String(idx).padStart(2, '0')}`,
            weights: w,
            pos2: [a, b, c],
            desc: describeKernel([a, b, c])
          });
          idx++;
        }
      }
    }
    return ks;
  }

  function describeKernel([a, b, c]) {
    const gaps = [b - a, c - b];
    const span = c - a;
    const clustered = (span <= 2);
    const mid = clustered && a >= 2 && c <= 6;
    const edge = clustered && (a <= 1 || c >= 7);
    const spread = (gaps[0] >= 2 && gaps[1] >= 2 && span >= 6);
    const lead = (a === 0 || a === 1);
    const trail = (c === 8 || c === 7);
    if (clustered && mid) return 'Local bump contrast (centered)';
    if (clustered && edge) return 'Edge-local contrast (near boundary)';
    if (spread) return 'Wide contrast (distributed taps)';
    if (lead && trail) return 'Two-sided contrast (both ends)';
    if (lead) return 'Leading-edge contrast (early taps)';
    if (trail) return 'Trailing-edge contrast (late taps)';
    if (span <= 4) return 'Mid-range contrast (moderate span)';
    return 'Mixed contrast (no clear archetype)';
  }

  function parseSeriesCSV(text) {
    const rows = d3.csvParseRows(text);
    const data = rows.map(r => ({
      t: +r[0],
      y: +r[1]
    })).filter(d => Number.isFinite(d.y));
    const tOk = data.length && data.every(d => Number.isFinite(d.t));
    return data.map((d, i) => ({
      t: tOk ? d.t : i,
      y: d.y
    }));
  }

  function parseSeriesRowCSV(text) {
    const rows = d3.csvParseRows(text);
    if (!rows || !rows.length) return [];
    if (rows.length > 1 && rows[0].length >= 2) return parseSeriesCSV(text);
    const r = rows[0];
    const vals = r.slice(1).map(x => +x).filter(Number.isFinite);
    return vals.map((y, t) => ({
      t,
      y
    }));
  }

  function parseIndexValueCSV(text, valueCol = 1) {
    const rows = d3.csvParseRows(text);
    const out = [];
    rows.forEach(r => {
      if (!r || r.length <= valueCol) return;
      const t = +r[0];
      const v = +r[valueCol];
      if (Number.isFinite(t) && Number.isFinite(v)) out.push({
        t,
        v
      });
    });
    return out;
  }

  function parseIndexValueMapCSV(text) {
    const rows = d3.csvParseRows(text);
    const map = new Map();
    rows.forEach(r => {
      if (!r || r.length < 2) return;
      const i = +r[0];
      const v = +r[1];
      if (Number.isFinite(i) && Number.isFinite(v)) map.set(i, v);
    });
    return map;
  }

  function parseKernelMaskCSV(text, kLen = 9) {
    const map = parseIndexValueMapCSV(text);
    const w = new Array(kLen).fill(0);
    for (let i = 0; i < kLen; i++) {
      if (map.has(i)) w[i] = map.get(i);
    }
    return w;
  }

  function parseBetaCSV(text) {
    const rows = d3.csvParseRows(text);
    const out = new Array(rows.length);
    for (let i = 0; i < rows.length; i++) {
      const r = rows[i];
      const raw = (r && r.length > 1) ? String(r[1]).trim() : "";
      out[i] = raw;
    }
    return out;
  }

  function parseBetaRowCSV(text) {
    const rows = d3.csvParseRows(text);
    if (!rows || !rows.length) return [];
    if (rows.length > 1 || (rows[0] && rows[0].length === 2)) return parseBetaCSV(text);
    const r = rows[0];
    return r.slice(1).map(v => String(v).trim());
  }

  function parseDecimalParts(raw) {
    const s0 = (raw == null) ? "" : String(raw).trim();
    if (!s0) return null;
    let s = s0;
    let sign = 1;
    if (s[0] === '-') {
      sign = -1;
      s = s.slice(1);
    } else if (s[0] === '+') {
      s = s.slice(1);
    }
    let exp = 0;
    const eIdx = s.search(/e|E/);
    if (eIdx >= 0) {
      const base = s.slice(0, eIdx);
      const ePart = s.slice(eIdx + 1);
      const eVal = parseInt(ePart, 10);
      if (Number.isFinite(eVal)) exp += eVal;
      s = base;
    }
    const dotIdx = s.indexOf('.');
    if (dotIdx >= 0) {
      const intPart = s.slice(0, dotIdx);
      const fracPart = s.slice(dotIdx + 1);
      exp -= fracPart.length;
      s = (intPart + fracPart);
    }
    s = s.replace(/[^0-9]/g, '').replace(/^0+/, '');
    if (!s) return {
      sign: 1,
      digits: '0',
      exp: 0,
      isZero: true,
      raw: s0
    };
    const tz = s.match(/0+$/);
    if (tz && tz[0].length) {
      const k = tz[0].length;
      s = s.slice(0, -k);
      exp += k;
    }
    return {
      sign,
      digits: s,
      exp,
      isZero: false,
      raw: s0
    };
  }

  function sciExp10(p) {
    return p.exp + p.digits.length - 1;
  }

  function cmpParts(a, b) {
    if (!a && !b) return 0;
    if (!a) return -1;
    if (!b) return 1;
    const aZero = !!a.isZero || a.digits === '0';
    const bZero = !!b.isZero || b.digits === '0';
    if (aZero && bZero) return 0;
    if (aZero) return (b.sign < 0) ? 1 : -1;
    if (bZero) return (a.sign < 0) ? -1 : 1;
    if (a.sign !== b.sign) return a.sign < b.sign ? -1 : 1;
    const sa = sciExp10(a),
      sb = sciExp10(b);
    if (sa !== sb) {
      if (a.sign > 0) return sa < sb ? -1 : 1;
      return sa < sb ? 1 : -1;
    }
    const L = Math.max(a.digits.length, b.digits.length);
    const da = a.digits.padEnd(L, '0');
    const db = b.digits.padEnd(L, '0');
    if (da === db) return 0;
    return (a.sign > 0) ? (da < db ? -1 : 1) : (da < db ? 1 : -1);
  }

  function mantissaApprox(p) {
    const K = Math.min(18, p.digits.length);
    const head = parseFloat(p.digits.slice(0, K));
    if (!Number.isFinite(head) || head === 0) return 0;
    return head / Math.pow(10, K - 1);
  }

  function partsToScaledNumber(p, baseSciExp) {
    if (!p || p.isZero) return 0;
    const diff = sciExp10(p) - baseSciExp;
    if (diff > 308) return p.sign * Number.POSITIVE_INFINITY;
    if (diff < -324) return 0;
    const m = mantissaApprox(p);
    return p.sign * m * Math.pow(10, diff);
  }

  function computeBetaStats(betaRawArr) {
    const parts = (betaRawArr || []).map(parseDecimalParts);
    let minP = null,
      maxP = null;
    let hasPos = false,
      hasNeg = false;
    for (const p of parts) {
      if (!p || p.isZero) continue;
      if (p.sign > 0) hasPos = true;
      if (p.sign < 0) hasNeg = true;
      if (!minP || cmpParts(p, minP) < 0) minP = p;
      if (!maxP || cmpParts(p, maxP) > 0) maxP = p;
    }
    return {
      parts,
      minP,
      maxP,
      hasPos,
      hasNeg
    };
  }

  function isFeatureBetaFile(file) {
    return /feature[-_](\d+)\.csv$/i.test(String(file || ''));
  }

  function parseFeatureIdFromBetaFilename(file) {
    const m = String(file || '').match(/feature[-_](\d+)\.csv$/i);
    return m ? parseInt(m[1], 10) : null;
  }

  function betaVariantLabel(file, isBase = false) {
    if (isBase) return 'Original β';
    const fidx = parseFeatureIdFromBetaFilename(file);
    if (Number.isFinite(fidx)) return `Feature #${fidx} β`;
    return String(file || 'β');
  }

  function findBaseBetaFile(files, cfg, instanceId) {
    if (!files || !cfg || instanceId == null) return null;
    const prefix = `betas_backpropagated_explanations_ref_policy_${cfg.refPolicy}_instance_${instanceId}`;
    return (files || []).find(f => String(f || '').startsWith(prefix) && /\.csv$/i.test(String(f || '')) && !isFeatureBetaFile(f)) || null;
  }

  function findFeatureBetaFiles(files, cfg, instanceId) {
    if (!files || !cfg || instanceId == null) return [];
    const prefix = `betas_backpropagated_explanations_ref_policy_${cfg.refPolicy}_instance_${instanceId}`;
    return (files || []).filter(f => {
      const s = String(f || '');
      return s.startsWith(prefix) && /\.csv$/i.test(s) && isFeatureBetaFile(s);
    });
  }

  function formatCliCommand(args) {
    return `python3 ${(args || []).map(a => (/[^\w\-\.\/]/.test(String(a)) ? JSON.stringify(String(a)) : String(a))).join(' ')}`;
  }

  function convolve(seriesY, weights, dilation) {
    const n = seriesY.length;
    const k = weights.length;
    const center = Math.floor(k / 2);
    const receptive = (k - 1) * dilation + 1;
    const tMin = 0;
    const tMax = Math.max(0, n - 1);
    const resp = new Array(n);
    for (let t = 0; t < n; t++) {
      let s = 0;
      for (let i = 0; i < k; i++) {
        const idx = t + (i - center) * dilation;
        const x = (idx >= 0 && idx < n) ? seriesY[idx] : 0;
        s += weights[i] * x;
      }
      resp[t] = {
        t,
        v: s
      };
    }
    return {
      resp,
      tMin,
      tMax,
      receptive,
      center
    };
  }

  function computeFootprint(resp, threshold, weights, dilation, n) {
    const fp = new Array(n).fill(0);
    if (!resp || resp.length === 0) return fp;
    const center = Math.floor(weights.length / 2);
    for (const d of resp) {
      const a = Math.max(0, d.v - threshold);
      if (a <= 0) continue;
      for (let i = 0; i < weights.length; i++) {
        const idx = d.t + (i - center) * dilation;
        if (idx < 0 || idx >= n) continue;
        fp[idx] += a * Math.abs(weights[i]);
      }
    }
    let mx = 0;
    for (let i = 0; i < fp.length; i++) mx = Math.max(mx, fp[i]);
    if (mx > 0)
      for (let i = 0; i < fp.length; i++) fp[i] /= mx;
    return fp;
  }

  function quantile(arr, q) {
    const xs = [...arr].sort((a, b) => a - b);
    if (xs.length === 0) return 0;
    const p = (xs.length - 1) * q;
    const lo = Math.floor(p),
      hi = Math.ceil(p);
    if (lo === hi) return xs[lo];
    const w = p - lo;
    return xs[lo] * (1 - w) + xs[hi] * w;
  }

  function clamp(v, a, b) {
    return Math.max(a, Math.min(b, v));
  }

  function fmt(x) {
    if (x == null || !Number.isFinite(x)) return '—';
    const ax = Math.abs(x);
    if (ax >= 1000) return x.toFixed(0);
    if (ax >= 10) return x.toFixed(2);
    if (ax >= 1) return x.toFixed(3);
    return x.toFixed(8);
  }

  const REFERENCE_POLICY_LABELS = {
    opposite_class_farthest_instance: 'Opposite farthest',
    opposite_class_closest_instance: 'Opposite closest',
    opposite_class_medoid: 'Opposite medoid',
    opposite_class_centroid: 'Opposite centroid',
    global_medoid: 'Global medoid',
    global_centroid: 'Global centroid'
  };

  function getReferencePolicyLabel(policy) {
    const key = String(policy || '').trim();
    if (!key) return 'Current reference';
    if (REFERENCE_POLICY_LABELS[key]) return REFERENCE_POLICY_LABELS[key];
    if (el && el.peRefPolicy) {
      const match = Array.from(el.peRefPolicy.options || []).find(opt => String(opt.value) === key);
      if (match && String(match.textContent || '').trim()) return String(match.textContent).trim();
    }
    return key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
  }

  function extractReferencePolicyFromMetadataFile(filename, instanceId) {
    const safeId = String(instanceId).replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const rx = new RegExp(`^metadata_ref_policy_(.+)_instance_${safeId}\\.json$`, 'i');
    const m = String(filename || '').match(rx);
    return m ? m[1] : null;
  }

  function clearReferencePolicySelect(placeholder) {
    if (!el.referencePolicySelect) return;
    el.referencePolicySelect.innerHTML = '';
    const opt = document.createElement('option');
    opt.value = '';
    opt.textContent = placeholder || 'Current reference';
    el.referencePolicySelect.appendChild(opt);
    el.referencePolicySelect.disabled = true;
  }

  function updateReferencePolicySelect(files, instanceId, preferredPolicy) {
    if (!el.referencePolicySelect) return [];
    const discovered = [];
    const seen = new Set();
    (files || []).forEach(file => {
      const policy = extractReferencePolicyFromMetadataFile(file, instanceId);
      if (!policy || seen.has(policy)) return;
      seen.add(policy);
      discovered.push(policy);
    });
    const preferred = String(preferredPolicy || '').trim();
    if (preferred && !seen.has(preferred)) {
      discovered.push(preferred);
      seen.add(preferred);
    }
    const canonicalOrder = [
      'opposite_class_farthest_instance',
      'opposite_class_closest_instance',
      'opposite_class_medoid',
      'opposite_class_centroid',
      'global_medoid',
      'global_centroid'
    ];
    discovered.sort((a, b) => {
      const ia = canonicalOrder.indexOf(a);
      const ib = canonicalOrder.indexOf(b);
      const oa = ia === -1 ? Number.POSITIVE_INFINITY : ia;
      const ob = ib === -1 ? Number.POSITIVE_INFINITY : ib;
      return d3.ascending(oa, ob) || String(a).localeCompare(String(b));
    });
    el.referencePolicySelect.innerHTML = '';
    if (!discovered.length) {
      const opt = document.createElement('option');
      opt.value = '';
      opt.textContent = 'Current reference';
      el.referencePolicySelect.appendChild(opt);
      el.referencePolicySelect.disabled = true;
      return discovered;
    }
    discovered.forEach(policy => {
      const opt = document.createElement('option');
      opt.value = policy;
      opt.textContent = getReferencePolicyLabel(policy);
      el.referencePolicySelect.appendChild(opt);
    });
    const selected = discovered.includes(preferred) ? preferred : discovered[0];
    el.referencePolicySelect.value = selected;
    el.referencePolicySelect.disabled = false;
    return discovered;
  }

  function syncActiveTabToCurrentLoad(cfg, instanceId) {
    const active = tabState.tabs.find(t => t.id === tabState.activeId);
    if (!active) return;
    active.cfg = { ...cfg };
    active.instanceId = instanceId;
    active.key = tabKey(cfg, instanceId);
    active.label = String(instanceId);
    renderTabs();
  }

  function resetFeatureSelectionUI() {
    stopSweep();
    state.selectedFeatureIdx = null;
    if (el.featureIdx) el.featureIdx.value = '';
    if (el.featureInfo) el.featureInfo.textContent = 'α = — · emb = —';
    if (el.seriesInfo) el.seriesInfo.textContent = '—';
    if (el.summary) el.summary.textContent = '';
    if (el.selectedKernelDesc) el.selectedKernelDesc.textContent = '';
    try {
      d3.select(el.selectedKernelPlot).selectAll('*').remove();
    } catch (e) { }
    try {
      d3.select(el.convChart).selectAll('*').remove();
    } catch (e) { }
    renderInstanceDetails();
    renderKernelPreview();
    renderKernelList();
    syncRetropropButtonState();
  }

  const state = {
    kernels: generateMiniRocketKernels(),
    selectedId: 'K00',
    kernelOverride: null,
    kernelOverrideKey: null,
    series: [],
    referenceSeries: [],
    referencePath: '',
    dilation: 1,
    align: 0,
    threshold: 0,
    betaVariants: new Map(),
    activeBetaKey: null,
    betaRaw: [],
    betaStats: null,
    zoomTransform: d3.zoomIdentity,
    featureZoomTransform: d3.zoomIdentity,
    isFeatureZooming: false,
    features: [],
    selectedFeatureIdx: null,
    hoveredFeatureIdx: null,
    // Embedding panel ordering (visual only; does not change feature indices)
    featureSortKey: 'index', // 'index' | 'beta' | 'alpha'
    featureSortDir: 'asc',   // 'asc' | 'desc'
    featureOrder: null,      // Array<fidx> in display order
    featurePos: null,        // Map<fidx, rank>
    featureOrderSig: null,
    animateFeatureSortNext: false,
    footprint: null,
    sweepActive: false,
    sweepTimer: null,
    animateTapsNext: false,
    animateKernelTransitionNext: false,
    convLayout: null,
    alignLocked: false,
    isZooming: false,
    // NEW for perf
    seriesVersion: 0,
    convCache: new Map(),

    // Summary modal caches
    summaryCache: {
      // per-instance α maps loaded from disk (key: sig|id)
      alphaMapCache: new Map(),

      rangeSig: null,
      rangeIds: null,
      rangeAlphaMaps: null,
      rangeErrors: null,

      listSig: null,
      listIds: null,
      listAlphaMaps: null,
      listErrors: null,

      classFlipKey: null,
      classFlipFile: null,
      classFlipTopN: null,
      classFlipFeatureIds: null
    },

    // runtime caches for fast alignment & throttling
    sX: null,
    sY: null,
    sInnerW: 0,
    sInnerH: 0,
    __alignRAF: null,
    __pendingAlignPx: 0,
    __alignBounds: null,
    __alignX: null,

    // Track which instance's series is currently previewed (before Load)
    previewedInstanceId: null,

  };

  const el = {
    status: document.querySelector('#status'),
    seriesInfo: document.querySelector('#seriesInfo'),
    kernelSearch: document.querySelector('#kernelSearch'),
    kernelList: d3.select('#kernelList'),
    summary: document.querySelector('#summary'),
    selectedKernelDesc: document.querySelector('#selectedKernelDesc'),
    selectedKernelPlot: document.querySelector('#selectedKernelPlot'),
    selectedKernelDesc: document.querySelector('#selectedKernelDesc'),
    pickerBtn: document.querySelector('#kernelPickerBtn'),
    dropdown: document.querySelector('#kernelDropdown'),
    resetZoom: document.querySelector('#resetZoom'),
    dilationLabel: document.querySelector('#dilationLabel'),
    dilationBounds: document.querySelector('#dilationBounds'),
    dilationFill: document.querySelector('#dilationFill'),
    dilationMarker: document.querySelector('#dilationMarker'),
    dilationSlider: document.querySelector('#dilationSlider'),
    thresholdLabel: document.querySelector('#thresholdLabel'),
    thresholdBounds: document.querySelector('#thresholdBounds'),
    thresholdFill: document.querySelector('#thresholdFill'),
    thresholdMarker: document.querySelector('#thresholdMarker'),
    thresholdSlider: document.querySelector('#thresholdSlider'),
    align: document.querySelector('#align'),
    alignLabel: document.querySelector('#alignLabel'),
    sweepBtn: document.querySelector('#sweepBtn'),
    betaFileSelect: document.querySelector('#betaFileSelect'),
    referencePolicySelect: document.querySelector('#referencePolicySelect'),
    showReference: document.querySelector('#showReference'),
    retropropFeatureBtn: document.querySelector('#retropropFeatureBtn'),
    retroProgressWrap: document.querySelector('#retroProgressWrap'),
    retroProgressText: document.querySelector('#retroProgressText'),
    retroProgressPct: document.querySelector('#retroProgressPct'),
    retroProgressBar: document.querySelector('#retroProgressBar'),
    retroProgressFillBar: document.querySelector('#retroProgressFillBar'),
    retroProgressLog: document.querySelector('#retroProgressLog'),
    featureIdx: document.querySelector('#featureIdx'),
    goFeature: document.querySelector('#goFeature'),
    featureInfo: document.querySelector('#featureInfo'),
    featureTooltip: document.querySelector('#featureTooltip'),
    embSortKey: document.querySelector('#embSortKey'),
    embSortDir: document.querySelectorAll('input[name=\"embSortDir\"]'),
    peDataset: document.querySelector('#peDataset'),
    peLabelSelect: document.querySelector('#peLabel'),
    peModel: document.querySelector('#peModel'),
    peExplainer: document.querySelector('#peExplainer'),
    peRefPolicy: document.querySelector('#peRefPolicy'),
    peStart: document.querySelector('#peStart'),
    peEnd: document.querySelector('#peEnd'),
    peStartNum: document.querySelector('#peStartNum'),
    peEndNum: document.querySelector('#peEndNum'),
    peRangeFill: document.querySelector('#peRangeFill'),
    peRangeLabel: document.querySelector('#peRangeLabel'),
    peInstance: document.querySelector('#peInstance'),
    peTopT: document.querySelector('#peTopT'),
    peTopTField: document.querySelector('#peTopTField'),
    peRunBtn: document.querySelector('#peRunBtn'),
    peLoadBtn: document.querySelector('#peLoadBtn'),
    peCopyBtn: document.querySelector('#peCopyBtn'),
    peCmd: document.querySelector('#peCmd'),
    peProgressWrap: document.querySelector('#peProgressWrap'),
    peProgressText: document.querySelector('#peProgressText'),
    peProgressPct: document.querySelector('#peProgressPct'),
    peProgressBar: document.querySelector('#peProgressBar'),
    peProgressFillBar: document.querySelector('#peProgressFillBar'),
    peLog: document.querySelector('#peLog'),
    peOpenTabBtn: document.querySelector('#peOpenTabBtn'),
    instanceTabs: document.querySelector('#instanceTabs'),
    tpProgress: document.querySelector('#tpProgress'),
    tpProgressText: document.querySelector('#tpProgressText'),
    emptyState: document.querySelector('#emptyState'),
    peRunStatus: document.querySelector('#peRunStatus'),

    // Instance details (left sidebar)
    instanceDetailsCard: document.querySelector('#instanceDetailsCard'),
    instanceDetailsHint: document.querySelector('#instanceDetailsHint'),
    instanceMetaGrid: document.querySelector('#instanceMetaGrid'),
    instanceMetaBadges: document.querySelector('#instanceMetaBadges'),
    instanceSlopeChart: document.querySelector('#instanceSlopeChart'),
    instanceEmbeddingPct: document.querySelector('#instanceEmbeddingPct'),
    instanceMetaFootnote: document.querySelector('#instanceMetaFootnote'),


    // Summary modal (Top embeddings)
    peSummaryBtn: document.querySelector('#peSummaryBtn'),
    peSummaryOverlay: document.querySelector('#peSummaryOverlay'),
    peSummaryModal: document.querySelector('#peSummaryModal'),
    peSummaryHeader: document.querySelector('#peSummaryHeader'),
    peSummaryCloseBtn: document.querySelector('#peSummaryCloseBtn'),
    peSummaryRefreshBtn: document.querySelector('#peSummaryRefreshBtn'),
    peSummarySubtitle: document.querySelector('#peSummarySubtitle'),
    peSummaryStatus: document.querySelector('#peSummaryStatus'),
    peSummaryBiasStats: document.querySelector('#peSummaryBiasStats'),
    peSummaryScopeSelect: document.querySelector('#peSummaryScopeSelect'),
    peSummaryRangeControls: document.querySelector('#peSummaryRangeControls'),
    peSummaryStartId: document.querySelector('#peSummaryStartId'),
    peSummaryEndId: document.querySelector('#peSummaryEndId'),
    peSummaryLoadRangeBtn: document.querySelector('#peSummaryLoadRangeBtn'),
    peSummaryListControls: document.querySelector('#peSummaryListControls'),
    peSummaryInstanceList: document.querySelector('#peSummaryInstanceList'),
    peSummaryLoadListBtn: document.querySelector('#peSummaryLoadListBtn'),
    peSummaryTopK: document.querySelector('#peSummaryTopK'),
    peSummaryTopKLabel: document.querySelector('#peSummaryTopKLabel'),
    peSummaryShowDilated: document.querySelector('#peSummaryShowDilated'),
    peSummaryKernelSort: document.querySelector('#peSummaryKernelSort'),
    peSummaryMatrixWrap: document.querySelector('#peSummaryMatrixWrap'),
    peSummaryMatrixSvg: document.querySelector('#peSummaryMatrixSvg'),
    peSummaryKernelStrip: document.querySelector('#peSummaryKernelStrip'),

    // Generate-instance popover
    peAdvancedBtn: document.querySelector('#peAdvancedBtn'),
    peAdvancedPopover: document.querySelector('#peAdvancedPopover'),
    peAdvancedCloseBtn: document.querySelector('#peAdvancedCloseBtn'),
  };


  // Instance Details: use a single SVG (slope + contribution bar) instead of a separate embedding widget.
  (function ensureSingleInstanceDetailsSVG() {
    const slope = el.instanceSlopeChart;
    const embed = el.instanceEmbeddingPct;

    if (embed && embed.parentElement) {
      const parent = embed.parentElement;

      // If the slope chart and embedding widget were siblings in a two-column layout,
      // removing the embedding widget can leave an empty column. We force a single-column layout
      // only in the common case where the parent only held these two elements.
      const parentHadTwoChildren = (parent.childElementCount === 2) && slope && parent.contains(slope);
      try { embed.remove(); } catch (e) { embed.style.display = 'none'; }

      if (parentHadTwoChildren) {
        parent.style.display = 'block';
        parent.style.gridTemplateColumns = '1fr';
      }
    }

    if (slope) {
      slope.style.display = 'block';
      slope.style.width = '100%';
      slope.style.maxWidth = '100%';
    }

    // Keep the reference but mark as unused; the UI is now rendered in the SVG above.
    el.instanceEmbeddingPct = null;
  })();


  function chart(svgSel, margin) {
    const svg = d3.select(svgSel);
    const g = svg.append('g');
    const m = margin || {
      top: 16,
      right: 18,
      bottom: 26,
      left: 44
    };

    function resize() {
      const w = svg.node().clientWidth;
      const h = svg.node().clientHeight;
      g.attr('transform', `translate(${m.left},${m.top})`);
      return {
        w,
        h,
        innerW: w - m.left - m.right,
        innerH: h - m.top - m.bottom,
        m
      };
    }
    return {
      svg,
      g,
      m,
      resize
    };
  }
  const seriesChart = chart('#seriesChart', {
    top: 16,
    right: 18,
    bottom: 26,
    left: 44
  });
  const convChart = chart('#convChart', {
    top: 16,
    right: 18,
    bottom: 26,
    left: 44
  });
  const featureChart = chart('#featureOverview', {
    top: 16,
    right: 18,
    bottom: 26,
    left: 44
  });

  const fG = featureChart.g;
  const fTopG = fG.append('g');
  const fBotG = fG.append('g');
  const fAxes = {
    x: fG.append('g'),
    y: fG.append('g')
  };
  const fEmbedPath = fTopG.append('path').attr('fill', 'none').attr('stroke', 'rgba(37,99,235,.75)').attr('stroke-width', 1.8);
  const fEmbedBarsG = fTopG.append('g');
  const fHeatG = fBotG.append('g');
  const fSep = fG.append('line').attr('stroke', 'rgba(17,24,39,.10)').attr('stroke-width', 1);
  const fSel = fG.append('line').attr('stroke', 'rgba(17,24,39,.55)').attr('stroke-width', 1.5);
  const fHover = fG.append('line').attr('stroke', 'rgba(17,24,39,.20)').attr('stroke-width', 1.25).attr('stroke-dasharray', '4 4');
  const fOverlay = fG.append('rect').attr('fill', 'transparent').style('cursor', 'crosshair');


  const sG = seriesChart.g;
  const sAxes = {
    x: sG.append('g'),
    y: sG.append('g')
  };
  const sOverlay = sG.insert('rect', ':first-child').attr('fill', 'transparent').style('cursor', 'grab');
  const sFootG = sG.append('g').attr('class', 'footprintLayer');
  const sBetaG = sG.append('g').attr('class', 'betaHeatmapLayer');
  const sRefPath = sG.append('path').attr('class', 'refPath').attr('fill', 'none').attr('stroke', 'rgba(99,102,241,.85)').attr('stroke-width', 1.75).attr('stroke-dasharray', '6 4').attr('opacity', 0);
  const sPath = sG.append('path').attr('fill', 'none').attr('stroke', 'rgba(37,99,235,.85)').attr('stroke-width', 2);
  const sTapsG = sG.append('g').attr('class', 'tapsLayer');
  const sHandleG = sG.append('g').attr('class', 'handleLayer');

  const cG = convChart.g;
  const cAxes = {
    x: cG.append('g'),
    y: cG.append('g')
  };
  const cArea = cG.append('path').attr('stroke', 'none').attr('fill', 'rgba(16,185,129,.18)');
  const cPath = cG.append('path').attr('fill', 'none').attr('stroke', 'rgba(16,185,129,.75)').attr('stroke-width', 2);
  const cThr = cG.append('line').attr('stroke', 'rgba(245,158,11,.95)').attr('stroke-width', 1.5).attr('stroke-dasharray', '6 4');
  const cAlign = cG.append('line').attr('stroke', 'rgba(245,158,11,.40)').attr('stroke-width', 2);
  const cOverlay = cG.append('rect').attr('fill', 'transparent').style('cursor', 'crosshair');

  function styleAxes(g) {
    g.selectAll('.domain').attr('stroke', 'rgba(17,24,39,.20)');
    g.selectAll('.tick line').attr('stroke', 'rgba(17,24,39,.12)');
    g.selectAll('.tick text').attr('fill', 'rgba(17,24,39,.70)');
  }

  function parseFeatureMetaCSV(text) {
    const rows = d3.csvParse(text);
    return rows.map(r => ({
      fidx: +r.fidx,
      alpha: +r.alpha,
      embedding: +r.embedding,
      kernel_index: +r.kernel_index,
      kernel_id: r.kernel_id,
      dilation: +r.dilation,
      threshold: +r.threshold,
      triplet: r.triplet_str,
      kernel_str: r.kernel_str
    })).filter(d => Number.isFinite(d.fidx));
  }

  function parseAlphaCSV(text) {
    const rows = d3.csvParseRows(text);
    const map = new Map();
    rows.forEach(r => {
      const f = +r[0];
      const a = +r[1];
      if (Number.isFinite(f) && Number.isFinite(a)) map.set(f, a);
    });
    return map;
  }

  function parseEmbeddingCSV(text) {
    const rows = d3.csvParseRows(text || ""),
      map = new Map();
    let countPair = 0;
    rows.forEach(r => {
      const f = +r[0],
        v = +r[1];
      if (Number.isFinite(f) && Number.isFinite(v)) {
        map.set(f, v);
        countPair += 1;
      }
    });
    if (countPair > 0) return map;
    if (!rows.length) return map;
    const isIndexRow = (r) => {
      if (!r || r.length < 4) return false;
      let start = 0;
      if (!Number.isFinite(+r[0])) start = 1;
      let ok = 0;
      for (let i = start; i < Math.min(r.length, start + 12); i++) {
        if (+r[i] === (i - start)) ok += 1;
      }
      return ok >= Math.min(6, Math.max(0, Math.min(r.length - start, 12)));
    };
    if (rows.length >= 2 && isIndexRow(rows[0])) {
      const vals = rows[1].map(x => +x);
      for (let i = 0; i < vals.length; i++) {
        if (Number.isFinite(vals[i])) map.set(i, vals[i]);
      }
      return map;
    }
    const vals0 = rows[0].map(x => +x);
    const start0 = Number.isFinite(vals0[0]) ? 0 : 1;
    for (let i = start0; i < vals0.length; i++) {
      if (Number.isFinite(vals0[i])) map.set(i - start0, vals0[i]);
    }
    return map;
  }


  function buildFeatureMetaFromSignals(alphaMap, embMap) {
    const ids = new Set();

    try {
      if (alphaMap && typeof alphaMap.forEach === 'function') {
        alphaMap.forEach((_, k) => {
          const f = +k;
          if (Number.isFinite(f)) ids.add(f);
        });
      }
    } catch (_) { }

    try {
      if (embMap && typeof embMap.forEach === 'function') {
        embMap.forEach((_, k) => {
          const f = +k;
          if (Number.isFinite(f)) ids.add(f);
        });
      }
    } catch (_) { }

    // Fallback to model caches if needed
    if (!ids.size) {
      try {
        if (peState && peState.dilations && typeof peState.dilations.forEach === 'function') {
          peState.dilations.forEach((_, k) => {
            const f = +k;
            if (Number.isFinite(f)) ids.add(f);
          });
        }
      } catch (_) { }

      try {
        if (peState && peState.biases && typeof peState.biases.forEach === 'function') {
          peState.biases.forEach((_, k) => {
            const f = +k;
            if (Number.isFinite(f)) ids.add(f);
          });
        }
      } catch (_) { }
    }

    const fidxs = Array.from(ids).sort((a, b) => a - b);

    const kernelCount = (state && state.kernels && state.kernels.length) ? state.kernels.length : 84;

    return fidxs.map(fidx => {
      const kIdx = ((fidx % kernelCount) + kernelCount) % kernelCount;
      const kId = `K${String(kIdx).padStart(2, '0')}`;
      const dil = peDilationForFeature(fidx);
      const thr = peBiasForFeature(fidx);

      return {
        fidx,
        alpha: (alphaMap && alphaMap.has && alphaMap.has(fidx)) ? +alphaMap.get(fidx) : 0,
        embedding: (embMap && embMap.has && embMap.has(fidx)) ? +embMap.get(fidx) : 0,
        kernel_index: kIdx,
        kernel_id: kId,
        dilation: Number.isFinite(+dil) ? +dil : 1,
        threshold: Number.isFinite(+thr) ? +thr : 0,
        triplet: '',
        kernel_str: ''
      };
    });
  }

  function mergeFeatureSignals(features, alphaMap, embMap) {
    return (features || []).map(d => ({
      ...d,
      alpha: (alphaMap && alphaMap.has(d.fidx)) ? alphaMap.get(d.fidx) : (Number.isFinite(d.alpha) ? d.alpha : 0),
      embedding: (embMap && embMap.has(d.fidx)) ? embMap.get(d.fidx) : (Number.isFinite(d.embedding) ? d.embedding : 0)
    }));
  }
  // ---- Embedding panel sorting (visual only) ----
  function invalidateFeatureDisplayOrder() {
    state.featureOrder = null;
    state.featurePos = null;
    state.featureOrderSig = null;
  }

  function featureByFidx(fidx) {
    if (!state.features || state.features.length === 0) return null;
    const i = Math.round(+fidx);
    if (Number.isFinite(i) && i >= 0 && i < state.features.length) {
      const d = state.features[i];
      if (d && +d.fidx === i) return d;
    }
    // Fallback (handles missing or non-contiguous feature sets)
    return state.features.find(d => +d.fidx === +fidx) || null;
  }

  function getFeatureDisplayOrder() {
    const n = (state.features && state.features.length) ? state.features.length : 0;
    const key = state.featureSortKey || 'index';
    const dir = state.featureSortDir || 'asc';
    const sig = `${key}|${dir}|${n}`;
    if (state.featureOrder && state.featureOrderSig === sig && state.featurePos) return state.featureOrder;

    const ord = new Array(n);
    for (let i = 0; i < n; i++) ord[i] = state.features[i].fidx;

    const getVal = (fidx) => {
      const d = featureByFidx(fidx);
      if (!d) return null;
      if (key === 'alpha') return Number.isFinite(+d.alpha) ? +d.alpha : null;
      if (key === 'beta') return Number.isFinite(+d.embedding) ? +d.embedding : null; // beta == embedding here
      // index
      return Number.isFinite(+d.fidx) ? +d.fidx : null;
    };

    ord.sort((a, b) => {
      if (key === 'index') return d3.ascending(+a, +b);

      const va = getVal(a);
      const vb = getVal(b);

      const aBad = (va == null || !Number.isFinite(va));
      const bBad = (vb == null || !Number.isFinite(vb));
      if (aBad && bBad) return d3.ascending(+a, +b);
      if (aBad) return 1;
      if (bBad) return -1;

      const cmp = d3.ascending(va, vb) || d3.ascending(+a, +b);
      return cmp;
    });

    if (dir === 'desc') ord.reverse();

    const pos = new Map();
    for (let i = 0; i < ord.length; i++) pos.set(ord[i], i);

    state.featureOrder = ord;
    state.featurePos = pos;
    state.featureOrderSig = sig;
    return ord;
  }

  function featureDisplayPos(fidx) {
    const ord = getFeatureDisplayOrder();
    if (state.featurePos && state.featurePos.has(fidx)) return state.featurePos.get(fidx);
    // Fall back to identity if needed
    return +fidx;
  }


  function featureFocusTransformForFidx(fidx, zoomK) {
    const n = (state.features && state.features.length) ? state.features.length : 0;
    if (!n) return d3.zoomIdentity;
    const dims = featureChart.resize();
    const innerW = Math.max(1, dims.innerW || 1);
    const xBaseFeat = d3.scaleLinear().domain([0, n - 1]).range([0, innerW]);
    const rank = clamp(featureDisplayPos(fidx), 0, Math.max(0, n - 1));
    const k = clamp(Number.isFinite(+zoomK) ? +zoomK : 6, 1, 200);
    const txDesired = (innerW / 2) - (xBaseFeat(rank) * k);
    const tx = clamp(txDesired, innerW * (1 - k), 0);
    return d3.zoomIdentity.translate(tx, 0).scale(k);
  }

  function focusFeatureInEmbeddingPanel(fidx, opts = {}) {
    const n = (state.features && state.features.length) ? state.features.length : 0;
    const featureId = Number.isFinite(+fidx) ? Math.round(+fidx) : null;
    if (!n || !Number.isFinite(featureId)) return;
    state.featureZoomTransform = featureFocusTransformForFidx(featureId, opts.zoomK);
    state.hoveredFeatureIdx = featureId;
    state.__featureHoverIdx = featureId;
    if (state.__featureFocusPulseTimer) clearTimeout(state.__featureFocusPulseTimer);
    state.__featureFocusPulseTimer = setTimeout(() => {
      if (state.hoveredFeatureIdx !== featureId) return;
      state.hoveredFeatureIdx = null;
      state.__featureHoverIdx = null;
      renderFeatureCharts();
    }, 900);
  }


  async function fetchFirst(paths) {
    let lastErr = null;
    for (const p of paths) {
      try {
        const res = await fetch(p, {
          cache: 'no-store'
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return await res.text();
      } catch (e) {
        lastErr = e;
      }
    }
    throw lastErr || new Error('Fetch failed');
  }

  function setFeatures(features) {
    state.features = (features || []).slice().sort((a, b) => d3.ascending(a.fidx, b.fidx));
    invalidateFeatureDisplayOrder();
    state.featureZoomTransform = d3.zoomIdentity;
    if (el.featureIdx) el.featureIdx.max = Math.max(0, state.features.length - 1);
    renderFeatureCharts();
    if (el.peSummaryBtn) el.peSummaryBtn.disabled = false;
    if (el.featureInfo) el.featureInfo.textContent = 'α = — · emb = —';
    if (el.seriesInfo) el.seriesInfo.textContent = '';
    try {
      d3.select(el.convChart).selectAll('*').remove();
    } catch (e) { }
    syncRetropropButtonState();
  }

  function featureByIdx(fidx) {
    if (!state.features || state.features.length === 0) return null;
    const i = clamp(Math.round(fidx), 0, state.features.length - 1);
    return state.features[i];
  }

  function updateFeatureHeader(d) {
    if (!d) return;
    if (el.featureInfo) el.featureInfo.textContent = `α = ${fmt(d.alpha)} · emb = ${fmt(d.embedding)}`;
  }

  function applyFeatureToKernelExplorer(d) {
    if (!d) return;

    const fidx = +d.fidx;

    // Update dilation + threshold from model-level CSVs (dilations.csv, biases.csv).
    state.dilation = peDilationForFeature(fidx);
    updateDilationLimits();

    state.threshold = peBiasForFeature(fidx);
    updateThresholdDisplay(state.thresholdRange?.min, state.thresholdRange?.max);

    // Load the kernel mask from: /output/<dataset>/<model>/base_mask_feature_<fidx>.csv
    const cfg = peState ? peState.lastRun : null;
    const reqKey = `${cfg ? (cfg.dataset + '|' + cfg.model) : ''}|${fidx}`;
    state.kernelOverrideKey = reqKey;

    const applyPack = (pack) => {
      state.animateKernelTransitionNext = true;
      const matchId = kernelIdFromWeights(pack.weights);
      if (matchId) {
        state.selectedId = matchId;
        state.kernelOverride = { id: matchId, weights: pack.weights, pos2: pack.pos2, desc: pack.desc };
      } else {
        state.kernelOverride = { id: (state.selectedId || 'KFILE'), weights: pack.weights, pos2: pack.pos2, desc: pack.desc };
      }

      fixAlignBounds();
      renderKernelPreview();
      renderKernelList();
      renderSeriesAndConv();
    };

    // If we don't have an output context yet, fall back to metadata immediately.
    if (!cfg) {
      state.animateKernelTransitionNext = true;
      state.kernelOverride = null;
      if (d.kernel_id) state.selectedId = d.kernel_id;
      fixAlignBounds();
      renderKernelPreview();
      renderKernelList();
      renderSeriesAndConv();
      return;
    }

    // Apply cached mask synchronously (avoids a transient "wrong kernel" while the file loads).
    const cacheKey = `${cfg.dataset}|${cfg.model}|${fidx}`;
    if (peState && peState.kernelMaskCache && peState.kernelMaskCache.has(cacheKey)) {
      const pack = peState.kernelMaskCache.get(cacheKey);
      if (pack && state.kernelOverrideKey === reqKey) {
        applyPack(pack);
        return;
      }
    }

    // Keep the current kernel visible while loading; update only when the correct mask arrives.
    peLoadKernelMaskForFeature(fidx).then(pack => {
      if (!pack) throw new Error('no mask');
      if (state.kernelOverrideKey !== reqKey) return;
      applyPack(pack);
    }).catch(() => {
      if (state.kernelOverrideKey !== reqKey) return;
      state.animateKernelTransitionNext = true;
      state.kernelOverride = null;
      if (d.kernel_id) state.selectedId = d.kernel_id;
      fixAlignBounds();
      renderKernelPreview();
      renderKernelList();
      renderSeriesAndConv();
    });
  }


  function selectFeature(fidx, opts) {
    const d = featureByIdx(fidx);
    if (!d) return;
    stopSweep();
    state.selectedFeatureIdx = d.fidx;
    if (el.featureIdx) el.featureIdx.value = d.fidx;
    updateFeatureHeader(d);
    if (opts && opts.focusEmbedding) {
      focusFeatureInEmbeddingPanel(d.fidx, {
        zoomK: (opts && Number.isFinite(+opts.embeddingZoomK)) ? +opts.embeddingZoomK : 8
      });
    }
    // Keep the instance details card in sync with the currently selected embedding.
    renderInstanceDetails();
    renderFeatureCharts();
    applyFeatureToKernelExplorer(d);
    if (!(opts && opts.silentTooltip)) hideFeatureTooltip();
    syncRetropropButtonState();
  }

  function showFeatureTooltip(clientX, clientY, d) {
    const tip = el.featureTooltip;
    const card = document.querySelector('#featureCard');
    if (!tip || !d || !card) return;
    const r = card.getBoundingClientRect();
    const x = clientX - r.left;
    const y = clientY - r.top;
    tip.innerHTML = `#${d.fidx}<br>α=${fmt(d.alpha)}<br>emb=${fmt(d.embedding)}`;
    tip.style.left = `${x + 12}px`;
    tip.style.top = `${y + 12}px`;
    tip.style.opacity = 1;
    tip.style.transform = 'translateY(0px)';
  }

  function hideFeatureTooltip() {
    const tip = el.featureTooltip;
    if (!tip) return;
    tip.style.opacity = 0;
    tip.style.transform = 'translateY(-4px)';
  }

  function renderFeatureCharts() {
    if (!state.features || state.features.length === 0) return;
    renderFeatureOverview();
  }


  function renderFeatureOverview() {
    const {
      innerW,
      innerH
    } = featureChart.resize();
    const gap = 10,
      stripH = 22;
    const topH = Math.max(40, innerH - stripH - gap);
    const n = state.features.length;
    const ord = getFeatureDisplayOrder();
    // Zoom-based navigation (wheel to zoom, drag to pan), matching the convolution series plot.
    const xBaseFeat = d3.scaleLinear().domain([0, n - 1]).range([0, innerW]);
    if (!state.featureZoomTransform) state.featureZoomTransform = d3.zoomIdentity;
    const xZ = state.featureZoomTransform.rescaleX(xBaseFeat);
    const step = (n > 1) ? (xZ(1) - xZ(0)) : innerW;
    const bandW = Math.max(1, Math.abs(step));

    const doAnimSort = !!window.ANIMATE_TRANSITIONS && !!state.animateFeatureSortNext && (typeof gsap !== 'undefined');
    const sortDur = 0.55;
    const sortEase = 'power2.out';

    function fidxFromPx(px) {
      const v = xZ.invert(px);
      const r = clamp(Math.round(v), 0, n - 1);
      const fidx = ord && ord.length ? ord[r] : r;
      return (fidx == null) ? r : fidx;
    }

    fTopG.attr('transform', 'translate(0,0)');
    fBotG.attr('transform', `translate(0,${topH + gap})`);
    fSep.attr('x1', 0).attr('x2', innerW).attr('y1', topH + gap / 2).attr('y2', topH + gap / 2).attr('opacity', 1);
    const y = d3.scaleLinear().domain(d3.extent(state.features, d => d.embedding)).nice().range([topH, 0]);

    // Render embeddings as a bar chart (instead of a line chart).
    fEmbedPath.attr('d', '').attr('opacity', 0);

    const yZero = y(0);
    const nBars = state.features.length;
    const barW0 = Math.max(0.8, bandW * 0.9);

    const bars = fEmbedBarsG.selectAll('rect.embedBar').data(state.features, d => d.fidx);
    const barsEnter = bars.enter().append('rect')
      .attr('class', 'embedBar')
      .attr('fill', 'rgba(125,169,165,.35)')
      .attr('stroke', 'rgba(125,169,165,.75)')
      .attr('stroke-width', 0.8);

    const barsMerged = barsEnter.merge(bars);

    barsMerged.each(function (d) {
      const node = this;
      const xNew = xZ(featureDisplayPos(d.fidx)) - barW0 / 2;
      const yNew = Math.min(y(d.embedding), yZero);
      const hNew = Math.max(0, Math.abs(y(d.embedding) - yZero));

      // y/height/width are stable across sorts; animate only x (and keep widths correct under zoom).
      d3.select(node)
        .attr('width', barW0)
        .attr('y', yNew)
        .attr('height', hNew);

      if (doAnimSort) {
        gsap.killTweensOf(node);
        gsap.to(node, { duration: sortDur, ease: sortEase, attr: { x: xNew } });
      } else {
        d3.select(node).attr('x', xNew);
      }
    });

    bars.exit().remove();
    fAxes.x.attr('transform', `translate(0,${innerH})`);
    fAxes.y.attr('transform', 'translate(0,0)');
    const xTick = (v) => {
      const i = clamp(Math.round(v), 0, n - 1);
      const f = (ord && ord.length) ? ord[i] : i;
      return (f == null) ? '' : String(f);
    };
    fAxes.x.call(d3.axisBottom(xZ).ticks(6).tickFormat(xTick).tickSizeOuter(0));
    fAxes.y.call(d3.axisLeft(y).ticks(4).tickSizeOuter(0));
    styleAxes(fG);
    const aVals = state.features.map(d => d.alpha);
    const minA = d3.min(aVals);
    const maxA = d3.max(aVals);
    const maxAbs = Math.max(Math.abs(minA || 0), Math.abs(maxA || 0), 1e-12);
    let color;
    if ((minA || 0) < 0 && (maxA || 0) > 0) {
      color = d3.scaleDiverging().domain([-maxAbs, 0, maxAbs]).interpolator(d3.interpolateRdBu);
    } else {
      color = d3.scaleSequential().domain([0, maxAbs]).interpolator(d3.interpolateBlues);
    }
    const rects = fHeatG.selectAll('rect.cell').data(state.features, d => d.fidx);

    const rectsEnter = rects.enter()
      .append('rect')
      .attr('class', 'cell')
      .attr('y', 0)
      .attr('height', stripH);

    const rectsMerged = rectsEnter.merge(rects);

    rectsMerged.each(function (d) {
      const node = this;
      const xNew = xZ(featureDisplayPos(d.fidx)) - bandW / 2;
      const wNew = bandW + 0.6;

      const v = d.alpha;
      const fillNew = (() => {
        if ((minA || 0) < 0 && (maxA || 0) > 0) return color(v);
        return color(Math.abs(v));
      })();

      d3.select(node)
        .attr('width', wNew)
        .attr('fill', fillNew);

      if (doAnimSort) {
        gsap.killTweensOf(node);
        gsap.to(node, { duration: sortDur, ease: sortEase, attr: { x: xNew } });
      } else {
        d3.select(node).attr('x', xNew);
      }
    });

    rects.exit().remove();
    const y0 = 0,
      y1 = topH + gap + stripH;

    if (state.selectedFeatureIdx != null) {
      const sx = xZ(featureDisplayPos(state.selectedFeatureIdx));
      fSel.attr('y1', y0).attr('y2', y1).attr('opacity', 1);

      const n0 = fSel.node();
      if (doAnimSort && n0) {
        gsap.killTweensOf(n0);
        gsap.to(n0, { duration: sortDur, ease: sortEase, attr: { x1: sx, x2: sx } });
      } else {
        fSel.attr('x1', sx).attr('x2', sx);
      }
    } else {
      fSel.attr('opacity', 0);
    }

    if (state.hoveredFeatureIdx != null) {
      const hx = xZ(featureDisplayPos(state.hoveredFeatureIdx));
      fHover.attr('y1', y0).attr('y2', y1).attr('opacity', 1);

      const n1 = fHover.node();
      if (doAnimSort && n1) {
        gsap.killTweensOf(n1);
        gsap.to(n1, { duration: sortDur, ease: sortEase, attr: { x1: hx, x2: hx } });
      } else {
        fHover.attr('x1', hx).attr('x2', hx);
      }
    } else {
      fHover.attr('opacity', 0);
    }

    // Throttle feature overview re-render to animation frames (performance). Also avoid re-rendering
    // when the hovered feature index hasn't changed.
    function scheduleFeatureOverviewRender() {
      if (state.__featureOverviewRAF) return;
      state.__featureOverviewRAF = requestAnimationFrame(() => {
        state.__featureOverviewRAF = null;
        renderFeatureOverview();
      });
    }




    fOverlay.attr('x', 0).attr('y', 0).attr('width', innerW).attr('height', y1)
      .style('cursor', 'grab');

    // Attach/refresh zoom on the embedding bar chart (x only).
    if (!featureZoom) {
      featureZoom = d3.zoom().scaleExtent([1, 200])
        .on('start', () => {
          state.isFeatureZooming = true;
          fOverlay.style('cursor', 'grabbing');
        })
        .on('zoom', (ev) => {
          if (syncingFeatureZoom) return;
          const t = d3.zoomIdentity.translate(ev.transform.x, 0).scale(ev.transform.k);
          state.featureZoomTransform = t;
          if (featureZoomRAF) cancelAnimationFrame(featureZoomRAF);
          featureZoomRAF = requestAnimationFrame(() => {
            featureZoomRAF = null;
            renderFeatureOverview();
          });
        })
        .on('end', () => {
          state.isFeatureZooming = false;
          fOverlay.style('cursor', 'grab');
        });
    }
    featureZoom.extent([
      [0, 0],
      [innerW, y1]
    ]).translateExtent([
      [0, 0],
      [innerW, y1]
    ]);
    syncingFeatureZoom = true;
    fOverlay.call(featureZoom).call(featureZoom.transform, state.featureZoomTransform);
    syncingFeatureZoom = false;
    fOverlay.selectAll('title').data([0]).join('title').text('Scroll to zoom. Drag to pan.');

    fOverlay
      .on('mousemove', (ev) => {
        if (state.isFeatureZooming) return;
        const [px] = d3.pointer(ev, fOverlay.node());
        const fidx = fidxFromPx(px);
        const prev = state.__featureHoverIdx;
        state.hoveredFeatureIdx = fidx;
        state.hoveredFeaturePx = px;
        if (prev !== fidx) {
          state.__featureHoverIdx = fidx;
          scheduleFeatureOverviewRender();
        }
        showFeatureTooltip(ev.clientX, ev.clientY, featureByFidx(fidx));
      })
      .on('mouseleave', () => {
        state.hoveredFeatureIdx = null;
        state.hoveredFeaturePx = null;
        state.__featureHoverIdx = null;
        scheduleFeatureOverviewRender();
        hideFeatureTooltip();
      })
      .on('click', (ev) => {
        if (state.isFeatureZooming) return;
        const [px] = d3.pointer(ev, fOverlay.node());
        const fidx = fidxFromPx(px);
        selectFeature(fidx);
      });


    // Only animate one render pass after a sort change (avoid animating on hover/zoom).
    if (state.animateFeatureSortNext) state.animateFeatureSortNext = false;

  }

  function getKernel() {
    if (state.kernelOverride && state.kernelOverride.weights) return state.kernelOverride;
    return state.kernels.find(k => k.id === state.selectedId) || state.kernels[0];
  }

  function setSelected(id) {
    state.kernelOverride = null;
    state.kernelOverrideKey = null;
    state.selectedId = id;
    fixAlignBounds();
    renderKernelPreview();
    renderKernelList();
    renderSeriesAndConv();
    closeDropdown();
    closeAdvancedPopover();
    if (typeof gsap !== 'undefined') {
      gsap.fromTo(el.selectedKernelPlot, {
        opacity: 0.65
      }, {
        opacity: 1,
        duration: 0.22,
        ease: 'power2.out'
      });
    }
  }

  function gsapToPromise(target, vars) {
    return new Promise(resolve => {
      if (typeof gsap === 'undefined') {
        resolve();
        return;
      }
      const v = Object.assign({}, vars, {
        onComplete: resolve
      });
      gsap.to(target, v);
    });
  }
  async function peFade(target, opacity, duration) {
    if (!target) return;
    await gsapToPromise(target, {
      opacity,
      duration: duration ?? 0.16,
      ease: 'power2.out'
    });
  }

  function toggleDropdown() {
    const isOpen = el.dropdown.classList.contains('open');
    if (isOpen) closeDropdown();
    else openDropdown();
  }

  function openDropdown() {
    el.dropdown.classList.add('open');
    el.pickerBtn.setAttribute('aria-expanded', 'true');
    setTimeout(() => el.kernelSearch && el.kernelSearch.focus(), 0);
  }

  function closeDropdown() {
    el.dropdown.classList.remove('open');
    el.pickerBtn.setAttribute('aria-expanded', 'false');
  }
  document.addEventListener('click', (ev) => {
    const card = document.querySelector('#kernelPickerCard');
    if (!card) return;
    if (!card.contains(ev.target)) closeDropdown();
  });
  el.pickerBtn.addEventListener('click', toggleDropdown);
  el.pickerBtn.addEventListener('keydown', (ev) => {
    if (ev.key === 'Enter' || ev.key === ' ') {
      ev.preventDefault();
      toggleDropdown();
    }
    if (ev.key === 'Escape') closeDropdown();
  });


  function drawKernelMini(svgNode, weights) {
    if (!svgNode) return;
    const svg = d3.select(svgNode);

    // Animate only the main (selected) kernel preview when a kernel change triggered this render.
    // Using GSAP here because d3 transitions can be easy to miss when multiple panels re-render.
    const isMainPreview = (svgNode === el.selectedKernelPlot);
    const animate = !!window.ANIMATE_TRANSITIONS && !!state.animateKernelTransitionNext && isMainPreview && (typeof gsap !== 'undefined');
    const animDur = 0.5;
    const animEase = 'power2.out';

    const bboxW = svgNode.getBoundingClientRect().width;
    const w = bboxW > 0 ? bboxW : (svgNode.clientWidth || 320);
    const h = svgNode.clientHeight || 118;

    svg.attr('width', '100%').attr('height', h);

    const m = { left: 28, right: 10, top: 10, bottom: 32 };
    const innerW = Math.max(10, w - m.left - m.right);
    const innerH = Math.max(10, h - m.top - m.bottom);

    let root = svg.select('g.km-root');
    const first = root.empty();
    if (first) {
      svg.selectAll('*').remove();
      root = svg.append('g').attr('class', 'km-root');
      root.append('g').attr('class', 'km-x');
      root.append('g').attr('class', 'km-y');
      root.append('g').attr('class', 'km-stems');
      root.append('path').attr('class', 'km-line');
      root.append('g').attr('class', 'km-dots');
    }

    root.attr('transform', `translate(${m.left},${m.top})`);

    const x = d3.scaleLinear().domain([0, 8]).range([0, innerW]);
    const y = d3.scaleLinear().domain([-1.5, 2.5]).range([innerH, 0]);

    // Place the x-axis at the minimum y-value (bottom of chart)
    const axisY = innerH;

    const gx = root.select('g.km-x').attr('transform', `translate(0,${axisY})`);
    const gy = root.select('g.km-y');

    gx.call(d3.axisBottom(x).ticks(9).tickFormat(d3.format('d')).tickSizeOuter(0));
    gy.call(d3.axisLeft(y).ticks(4).tickFormat(d3.format('d')).tickSizeOuter(0));

    [gx, gy].forEach(ax => {
      ax.selectAll('.domain').attr('stroke', 'rgba(17,24,39,.20)');
      ax.selectAll('.tick line').attr('stroke', 'rgba(17,24,39,.12)');
      ax.selectAll('.tick text')
        .attr('fill', 'rgba(17,24,39,.70)')
        .attr('font-family', 'var(--mono)')
        .attr('font-size', 10);
    });

    const pts = weights.map((w, i) => ({ w, i }));

    const stems = root.select('g.km-stems')
      .selectAll('line.stem')
      .data(pts, d => d.i);

    stems.join(
      enter => enter.append('line')
        .attr('class', 'stem')
        .attr('stroke', 'rgba(17,24,39,.12)')
        .attr('stroke-width', 1)
        .attr('x1', d => x(d.i))
        .attr('x2', d => x(d.i))
        .attr('y1', axisY)
        .attr('y2', axisY),
      update => update,
      exit => exit.remove()
    ).each(function (d) {
      const node = this;
      const attrs = { x1: x(d.i), x2: x(d.i), y1: axisY, y2: y(d.w) };
      if (animate) {
        gsap.killTweensOf(node);
        gsap.to(node, { duration: animDur, ease: animEase, attr: attrs });
      } else {
        d3.select(node)
          .attr('x1', attrs.x1).attr('x2', attrs.x2)
          .attr('y1', attrs.y1).attr('y2', attrs.y2);
      }
    });

    const line = d3.line().x((d, i) => x(i)).y(d => y(d));
    const linePath = root.select('path.km-line')
      .attr('fill', 'none')
      .attr('stroke', 'rgba(37,99,235,.85)')
      .attr('stroke-width', 2);
    const newD = line(weights);
    const curD = linePath.attr('d');
    if (animate && linePath.node() && curD) {
      const pNode = linePath.node();
      gsap.killTweensOf(pNode);
      gsap.to(pNode, { duration: animDur, ease: animEase, attr: { d: newD } });
    } else {
      linePath.attr('d', newD);
    }

    const dots = root.select('g.km-dots')
      .selectAll('circle.dot')
      .data(pts, d => d.i);

    const dotsEnter = dots.enter()
      .append('circle')
      .attr('class', 'dot')
      .attr('r', 4.6)
      .attr('stroke', 'rgba(17,24,39,.18)')
      .attr('stroke-width', 1)
      .attr('cx', d => x(d.i))
      .attr('cy', axisY);

    dotsEnter.append('title');

    const merged = dotsEnter.merge(dots);
    merged.select('title').text(d => `x=${d.i}, w=${d.w}`);

    merged.call(sel => {
      const fill = (d) => (Number.isFinite(d.w) && d.w > 0) ? 'rgba(16,185,129,.85)' : 'rgba(239,68,68,.70)';
      sel.each(function (d) {
        const node = this;
        const attrs = { cx: x(d.i), cy: y(d.w), fill: fill(d) };
        if (animate) {
          gsap.killTweensOf(node);
          // Animate both position and fill color.
          gsap.to(node, { duration: animDur, ease: animEase, attr: attrs });
        } else {
          d3.select(node)
            .attr('cx', attrs.cx)
            .attr('cy', attrs.cy)
            .attr('fill', attrs.fill);
        }
      });
    });

    dots.exit().remove();
  }

  function renderKernelPreview() {
    if (state.selectedFeatureIdx == null) {
      try {
        d3.select(el.selectedKernelPlot).selectAll('*').remove();
      } catch (e) { }
      return;
    }
    const k = getKernel();
    el.selectedKernelDesc.textContent = k.desc;
    drawKernelMini(el.selectedKernelPlot, k.weights);
  }

  function renderKernelList() {
    el.kernelList.selectAll('*').remove();
    const q = (el.kernelSearch?.value || '').trim().toLowerCase();
    let ks = state.kernels.slice();
    if (q) {
      ks = ks.filter(k => k.id.toLowerCase().includes(q) || k.desc.toLowerCase().includes(q));
    }
    const items = el.kernelList.selectAll('.kitem').data(ks, d => d.id).enter().append('div').attr('class', d => 'kitem' + (d.id === state.selectedId ? ' active' : '')).on('click', (_, d) => setSelected(d.id));
    const head = items.append('div').attr('class', 'khead');
    head.append('div').attr('class', 'kid').text(d => d.id);
    head.append('div').attr('class', 'ktag').text(d => d.desc);
    items.each(function (d) {
      const wrap = d3.select(this);
      const kp = wrap.append('svg').attr('class', 'kernelMini compact').style('height', '118px').node();
      drawKernelMini(kp, d.weights);
    });
  }

  let xBase = null,
    seriesZoom = null,
    convZoom = null,
    syncingZoom = false;
  let zoomRAF = null;
  let featureZoom = null,
    syncingFeatureZoom = false;
  let featureZoomRAF = null;

  function redrawConvFromCacheOnly() {
    if (!lastConv) return;

    const {
      innerW,
      innerH
    } = convChart.resize();
    cOverlay.attr('x', 0).attr('y', 0).attr('width', innerW).attr('height', innerH);

    const x = currentXScale(innerW);
    state.convX = x;
    state.convLayout = {
      tMin: lastConv.tMin,
      tMax: lastConv.tMax,
      innerW,
      innerH
    };
    initConvOverlayHandlers();

    const y = d3.scaleLinear().domain(d3.extent(lastConv.resp, d => d.v)).nice().range([innerH, 0]);
    const line = d3.line().x(d => x(d.t)).y(d => y(d.v));
    const area = d3.area().x(d => x(d.t)).y0(y(state.threshold)).y1(d => y(Math.max(d.v, state.threshold)));

    cPath.attr('d', line(lastConv.resp));
    cArea.attr('d', area(lastConv.resp));

    cThr
      .attr('x1', x(lastConv.tMin)).attr('x2', x(lastConv.tMax))
      .attr('y1', y(state.threshold)).attr('y2', y(state.threshold));

    updateConvAlignLine();

    cAxes.x.call(d3.axisBottom(x).ticks(6).tickSizeOuter(0));
    cAxes.y.call(d3.axisLeft(y).ticks(5).tickSizeOuter(0));
    styleAxes(cG);
  }

  function attachZoom() {
    const {
      innerW: sW,
      innerH: sH
    } = seriesChart.resize();
    const {
      innerW: cW,
      innerH: cH
    } = convChart.resize();
    sOverlay.attr('x', 0).attr('y', 0).attr('width', sW).attr('height', sH);
    cOverlay.attr('x', 0).attr('y', 0).attr('width', cW).attr('height', cH);

    function onZoom(ev, from) {
      if (!xBase) return;
      if (syncingZoom) return;
      state.zoomTransform = ev.transform;
      if (zoomRAF) cancelAnimationFrame(zoomRAF);
      syncingZoom = true;
      const target = (from === 'series') ? convZoom : seriesZoom;
      const overlay = (from === 'series') ? cOverlay : sOverlay;
      if (target) overlay.call(target.transform, state.zoomTransform);
      syncingZoom = false;
      zoomRAF = requestAnimationFrame(() => {
        renderSeries();
        redrawConvFromCacheOnly();
        zoomRAF = null;
      });
    }
    seriesZoom = d3.zoom().scaleExtent([1, 200]).extent([
      [0, 0],
      [sW, sH]
    ]).translateExtent([
      [0, 0],
      [sW, sH]
    ])
      .on('start', () => {
        state.isZooming = true;
        sOverlay.style('cursor', 'grabbing');
      })
      .on('zoom', (ev) => onZoom(ev, 'series'))
      .on('end', () => {
        state.isZooming = false;
        sOverlay.style('cursor', 'grab');
      });
    convZoom = d3.zoom().scaleExtent([1, 200]).extent([
      [0, 0],
      [cW, cH]
    ]).translateExtent([
      [0, 0],
      [cW, cH]
    ])
      .on('start', () => {
        state.isZooming = true;
        cOverlay.style('cursor', 'grabbing');
      })
      .on('zoom', (ev) => onZoom(ev, 'conv'))
      .on('end', () => {
        state.isZooming = false;
        cOverlay.style('cursor', 'grab');
      });
    sOverlay.style('cursor', 'grab').call(seriesZoom).call(seriesZoom.transform, state.zoomTransform);
    cOverlay.style('cursor', 'grab').call(convZoom).call(convZoom.transform, state.zoomTransform);
    sOverlay.selectAll('title').data([0]).join('title').text('Scroll to zoom. Drag to pan. (Synced with convolution view.)');
    cOverlay.selectAll('title').data([0]).join('title').text('Scroll to zoom. Drag to pan. Move to preview alignment; click to lock. Shift+click or Esc to unlock.');
  }

  function resetZoom() {
    state.zoomTransform = d3.zoomIdentity;
    if (seriesZoom) sOverlay.call(seriesZoom.transform, d3.zoomIdentity);
    if (convZoom) cOverlay.call(convZoom.transform, d3.zoomIdentity);
    renderSeriesAndConv();
  }

  function currentXScale(innerW) {
    const domain0 = d3.extent(state.series, d => d.t);
    let [t0, t1] = domain0;
    const k = getKernel();
    const kLen = (k && k.weights) ? k.weights.length : 9;
    const center = Math.floor(kLen / 2);
    const leftSteps = center * state.dilation;
    const rightSteps = (kLen - 1 - center) * state.dilation;
    let dt = 1;
    if (state.series.length > 1) {
      const span = t1 - t0;
      dt = Number.isFinite(span) ? (span / (state.series.length - 1)) : 1;
      if (!Number.isFinite(dt) || dt === 0) dt = 1;
    }
    t0 = t0 - leftSteps * dt;
    t1 = t1 + rightSteps * dt;
    xBase = d3.scaleLinear().domain([t0, t1]).range([0, innerW]);
    return state.zoomTransform.rescaleX(xBase);
  }

  function renderFootprintOverlay(x, innerW, innerH) {
    sFootG.selectAll('*').remove();
    const fp = state.footprint;
    if (!fp || fp.length === 0) return;
    const bandH = 14;
    const y0 = innerH - bandH; // slice to visible range
    const tVis0 = x.invert(0),
      tVis1 = x.invert(innerW);
    const i0 = d3.bisector(d => d.t).left(state.series, tVis0);
    const i1 = d3.bisector(d => d.t).right(state.series, tVis1);
    const start = Math.max(0, i0 - 1),
      end = Math.min(state.series.length, i1 + 1);
    const data = [];
    for (let i = start; i < end; i++) {
      const v = fp[i];
      if (!Number.isFinite(v) || v <= 0) continue;
      const t0 = state.series[i].t;
      const t1 = (i < state.series.length - 1) ? state.series[i + 1].t : (t0 + 1);
      const x0 = x(t0);
      const x1 = x(t1);
      data.push({
        x: x0,
        w: Math.max(1, x1 - x0),
        v
      });
    }
    sFootG.selectAll('rect.fp').data(data).enter().append('rect').attr('class', 'fp').attr('x', d => d.x).attr('y', y0).attr('width', d => d.w).attr('height', bandH).attr('fill', d => `rgba(245,158,11,${0.06 + 0.55 * d.v})`);
    sFootG.append('rect').attr('x', 0).attr('y', y0).attr('width', innerW).attr('height', bandH).attr('fill', 'none').attr('stroke', 'rgba(17,24,39,.10)');
  }

  function renderBetaHeatmap(x, innerW) {
    sBetaG.selectAll('*').remove();
    const stats = state.betaStats;
    if (!stats || !stats.parts || !stats.minP || !stats.maxP) return;
    const stripH = 12;
    const y0 = -stripH - 2;
    const minSci = sciExp10(stats.minP);
    const maxSci = sciExp10(stats.maxP);
    const baseSci = (maxSci - minSci <= 300) ? minSci : maxSci;
    const minScaled = partsToScaledNumber(stats.minP, baseSci);
    const maxScaled = partsToScaledNumber(stats.maxP, baseSci);
    if (!Number.isFinite(minScaled) || !Number.isFinite(maxScaled)) return;
    const maxAbs = Math.max(Math.abs(minScaled), Math.abs(maxScaled));
    let color;
    if (!Number.isFinite(maxAbs) || maxAbs === 0) {
      const c = d3.interpolateRdBu(0.5);
      color = () => c;
    } else {
      color = d3.scaleDiverging().domain([-maxAbs, 0, maxAbs]).interpolator(d3.interpolateRdBu);
    }
    const tVis0 = x.invert(0),
      tVis1 = x.invert(innerW);
    const i0 = clamp(Math.floor(tVis0), 0, stats.parts.length - 1);
    const i1 = clamp(Math.ceil(tVis1), 0, stats.parts.length - 1);
    const cells = [];
    const start = Math.min(i0, i1),
      end = Math.max(i0, i1);
    for (let i = start; i <= end; i++) {
      const p = stats.parts[i];
      if (!p) continue;
      const t0 = (i < state.series.length) ? state.series[i].t : i;
      const t1 = (i < state.series.length - 1) ? state.series[i + 1].t : (t0 + 1);
      const x0 = x(t0);
      const x1 = x(t1);
      const w = Math.max(1, x1 - x0);
      const vScaled = partsToScaledNumber(p, baseSci);
      if (!Number.isFinite(vScaled)) continue;
      cells.push({
        x: x0,
        w,
        v: vScaled,
        raw: p.raw
      });
    }
    sBetaG.selectAll('rect.beta').data(cells).enter().append('rect').attr('class', 'beta').attr('x', d => d.x).attr('y', y0).attr('width', d => d.w).attr('height', stripH).attr('fill', d => color(d.v)).attr('opacity', 0.92).append('title').text(d => `β=${d.raw}`);
    sBetaG.append('rect').attr('x', 0).attr('y', y0).attr('width', innerW).attr('height', stripH).attr('fill', 'none').attr('stroke', 'rgba(17,24,39,.10)');
  }

  function renderSeries() {
    const {
      innerW,
      innerH
    } = seriesChart.resize();
    sOverlay.attr('x', 0).attr('y', 0).attr('width', innerW).attr('height', innerH);
    if (seriesZoom) {
      seriesZoom.extent([
        [0, 0],
        [innerW, innerH]
      ]).translateExtent([
        [0, 0],
        [innerW, innerH]
      ]);
    }
    sAxes.x.attr('transform', `translate(0,${innerH})`);
    sAxes.y.attr('transform', 'translate(0,0)');
    const x = currentXScale(innerW);
    const drawRef = !!(el.showReference && el.showReference.checked);
    const yVals = state.series.map(d => d.y);
    if (drawRef && state.referenceSeries && state.referenceSeries.length) {
      for (const d of state.referenceSeries) {
        if (Number.isFinite(d.y)) yVals.push(d.y);
      }
    }
    const y = d3.scaleLinear().domain(d3.extent(yVals)).nice().range([innerH, 0]);
    const line = d3.line().x(d => x(d.t)).y(d => y(d.y));
    sPath.attr('d', line(state.series));
    if (drawRef && state.referenceSeries && state.referenceSeries.length) {
      sRefPath.attr('d', line(state.referenceSeries)).attr('opacity', 1);
    } else {
      sRefPath.attr('d', '').attr('opacity', 0);
    }
    sAxes.x.call(d3.axisBottom(x).ticks(6).tickSizeOuter(0));
    sAxes.y.call(d3.axisLeft(y).ticks(5).tickSizeOuter(0));
    styleAxes(sG);
    renderBetaHeatmap(x, innerW);
    renderFootprintOverlay(x, innerW, innerH);
    updateTapsAndHandle(x, y, innerW, innerH);
    // Cache scales and chart dimensions for fast alignment redraws
    state.sX = x;
    state.sY = y;
    state.sInnerW = innerW;
    state.sInnerH = innerH;
  }

  function renderSeriesWithReferenceMorph(fromReferenceSeries, toReferenceSeries) {
    const {
      innerW,
      innerH
    } = seriesChart.resize();
    sOverlay.attr('x', 0).attr('y', 0).attr('width', innerW).attr('height', innerH);
    if (seriesZoom) {
      seriesZoom.extent([
        [0, 0],
        [innerW, innerH]
      ]).translateExtent([
        [0, 0],
        [innerW, innerH]
      ]);
    }
    sAxes.x.attr('transform', `translate(0,${innerH})`);
    sAxes.y.attr('transform', 'translate(0,0)');
    const x = currentXScale(innerW);
    const drawRef = !!(el.showReference && el.showReference.checked);
    const yVals = state.series.map(d => d.y);
    [fromReferenceSeries, toReferenceSeries].forEach(series => {
      if (!drawRef || !series || !series.length) return;
      for (const d of series) {
        if (Number.isFinite(d.y)) yVals.push(d.y);
      }
    });
    const y = d3.scaleLinear().domain(d3.extent(yVals)).nice().range([innerH, 0]);
    const line = d3.line().x(d => x(d.t)).y(d => y(d.y));
    sPath.attr('d', line(state.series));
    if (drawRef && toReferenceSeries && toReferenceSeries.length) {
      const fromSeries = (fromReferenceSeries && fromReferenceSeries.length) ? fromReferenceSeries : toReferenceSeries;
      sRefPath.interrupt()
        .attr('d', line(fromSeries))
        .attr('opacity', 1)
        .transition()
        .duration(320)
        .ease(d3.easeCubicInOut)
        .attr('d', line(toReferenceSeries))
        .attr('opacity', 1);
    } else {
      sRefPath.interrupt().attr('d', '').attr('opacity', 0);
    }
    sAxes.x.call(d3.axisBottom(x).ticks(6).tickSizeOuter(0));
    sAxes.y.call(d3.axisLeft(y).ticks(5).tickSizeOuter(0));
    styleAxes(sG);
    renderBetaHeatmap(x, innerW);
    renderFootprintOverlay(x, innerW, innerH);
    updateTapsAndHandle(x, y, innerW, innerH);
    state.sX = x;
    state.sY = y;
    state.sInnerW = innerW;
    state.sInnerH = innerH;
  }

  function getTaps() {
    const k = getKernel();
    const weights = (k && k.weights) ? k.weights : Array(9).fill(0);
    const n = state.series.length;
    const kLen = weights.length;
    const center = Math.floor(kLen / 2);
    const taps = [];

    // Estimate dt for padded timestamps when the series has an explicit t axis.
    const dt = (n > 1) ? ((state.series[n - 1].t - state.series[0].t) / (n - 1)) : 1;
    const t0 = (n ? state.series[0].t : 0);
    const t1 = (n ? state.series[n - 1].t : (t0 + (n - 1) * dt));

    for (let i = 0; i < kLen; i++) {
      const idx = state.align + (i - center) * state.dilation;
      const inRange = (idx >= 0 && idx < n);
      let t, yv;
      if (inRange) {
        t = state.series[idx].t;
        yv = state.series[idx].y;
      } else if (idx < 0) {
        t = t0 + idx * dt;
        yv = 0;
      } else {
        t = t1 + (idx - (n - 1)) * dt;
        yv = 0;
      }
      taps.push({
        i,
        idx,
        t,
        y: yv,
        w: weights[i],
        padded: !inRange
      });
    }
    return taps;
  }

  const __knobDrag = d3.drag()
    .on('start', () => {
      stopSweep();
      const k = sHandleG.select('circle.knob');
      if (typeof gsap !== 'undefined' && k && k.node()) gsap.to(k.node(), {
        scale: 1.08,
        duration: 0.12
      });
    })
    .on('drag', (ev) => {
      if (!state.sX || state.series.length === 0) return;
      const innerW = state.sInnerW || 0;
      const px = clamp(ev.x, 0, innerW);
      const tVal = state.sX.invert(px);
      const i = d3.bisector(d => d.t).left(state.series, tVal);
      let idx;
      if (i <= 0) idx = 0;
      else if (i >= state.series.length) idx = state.series.length - 1;
      else {
        const tL = state.series[i - 1].t;
        const tR = state.series[i].t;
        idx = (Math.abs(tVal - tL) <= Math.abs(tR - tVal)) ? (i - 1) : i;
      }
      setAlign(idx, {
        light: true
      });
    })
    .on('end', () => {
      const k = sHandleG.select('circle.knob');
      if (typeof gsap !== 'undefined' && k && k.node()) gsap.to(k.node(), {
        scale: 1.0,
        duration: 0.12
      });
    });

  function updateTapsAndHandle(x, y, innerW, innerH) {
    if (!x || !y) return;

    const taps = getTaps();

    // Color taps by kernel weight (restore original red/green semantics):
    // negative weight -> red, positive weight -> green (e.g., -2 red, +1 green).
    const __pos = (a) => `rgba(16,185,129,${a})`;
    const __neg = (a) => `rgba(239,68,68,${a})`;
    const __zero = (a) => `rgba(17,24,39,${a})`;
    const __wColor = (w, aPos, aNeg, aZero) => {
      if (!Number.isFinite(w) || w === 0) return __zero(aZero);
      return (w > 0) ? __pos(aPos) : __neg(aNeg);
    };


    const doAnim = !!window.ANIMATE_TRANSITIONS && !!state.animateKernelTransitionNext;
    const tr = doAnim ? d3.transition().duration(320).ease(d3.easeCubicInOut) : null;

    const tapLines = sTapsG.selectAll('line.tap').data(taps, d => d.i);
    const tapLinesMerged = tapLines.join(
      enter => enter.append('line')
        .attr('class', 'tap')
        .attr('stroke-width', 2),
      update => update,
      exit => exit.remove()
    )
      .attr('stroke', d => d.padded ? __wColor(d.w, 0.5, 0.5, 0.08) : __wColor(d.w, 0.5, 0.5, 0.18));

    if (doAnim) {
      tapLinesMerged.transition(tr)
        .attr('x1', d => x(d.t)).attr('x2', d => x(d.t))
        .attr('y1', d => y(d.y)).attr('y2', innerH);
    } else {
      tapLinesMerged
        .attr('x1', d => x(d.t)).attr('x2', d => x(d.t))
        .attr('y1', d => y(d.y)).attr('y2', innerH);
    }

    const tapCircles = sTapsG.selectAll('circle.tap').data(taps, d => d.i);
    const tapCirclesEnter = tapCircles.enter()
      .append('circle')
      .attr('class', 'tap')
      .attr('r', 6)
      .attr('stroke-width', 1.25)
      .style('cursor', 'default')
      .attr('cx', d => x(d.t))
      .attr('cy', d => y(d.y));
    tapCirclesEnter.append('title');

    const tapCirclesMerged = tapCirclesEnter.merge(tapCircles);
    tapCirclesMerged.select('title').text(d => `tap ${d.i} · w=${d.w}`);

    if (doAnim) {
      tapCirclesMerged.transition(tr)
        .attr('cx', d => x(d.t))
        .attr('cy', d => y(d.y))
        .attr('fill', d => d.padded ? __wColor(d.w, 0.16, 0.14, 0.10) : __wColor(d.w, 0.80, 0.70, 0.45))
        .attr('stroke', d => d.padded ? __wColor(d.w, 0.24, 0.22, 0.16) : __wColor(d.w, 1.0, 0.95, 0.70));
    } else {
      tapCirclesMerged
        .attr('cx', d => x(d.t))
        .attr('cy', d => y(d.y))
        .attr('fill', d => d.padded ? __wColor(d.w, 0.16, 0.14, 0.10) : __wColor(d.w, 0.80, 0.70, 0.45))
        .attr('stroke', d => d.padded ? __wColor(d.w, 0.24, 0.22, 0.16) : __wColor(d.w, 1.0, 0.95, 0.70));
    }

    tapCircles.exit().remove();    // Handle and knob
    const n = state.series.length;
    const dt = (n > 1) ? ((state.series[n - 1].t - state.series[0].t) / (n - 1)) : 1;
    const t0 = (n ? state.series[0].t : 0);
    const t1 = (n ? state.series[n - 1].t : (t0 + (n - 1) * dt));

    let handleT;
    if (state.align >= 0 && state.align < n) handleT = state.series[state.align].t;
    else if (state.align < 0) handleT = t0 + state.align * dt;
    else handleT = t1 + (state.align - (n - 1)) * dt;

    const hx = x(handleT);

    const hLine = sHandleG.selectAll('line.handle').data([0]);
    hLine.join(enter => enter.append('line').attr('class', 'handle'))
      .attr('x1', hx).attr('x2', hx)
      .attr('y1', 0).attr('y2', innerH)
      .attr('stroke', 'rgba(245,158,11,.70)')
      .attr('stroke-width', 2);

    const knob = sHandleG.selectAll('circle.knob').data([0]);
    const knobEnter = knob.enter()
      .append('circle')
      .attr('class', 'knob')
      .attr('cy', 10)
      .attr('r', 8)
      .attr('fill', 'rgba(245,158,11,.85)')
      .attr('stroke', 'rgba(17,24,39,.18)')
      .attr('stroke-width', 1)
      .style('cursor', 'ew-resize')
      .call(__knobDrag);
    knobEnter.append('title').text('Drag to change alignment (t)');

    knobEnter.merge(knob)
      .attr('cx', hx);

    knob.exit().remove();

    if (state.animateTapsNext) {
      state.animateTapsNext = false;
      animateTapBuild();
    }
  }

  function renderSeriesFastAlign() {
    if (!state.sX || !state.sY) return;
    updateTapsAndHandle(state.sX, state.sY, state.sInnerW, state.sInnerH);
  }

  function updateAlignLabel() {
    if (!el.alignLabel) return;
    el.alignLabel.textContent = state.alignLocked ? (`t = ${state.align} (locked)`) : (`t = ${state.align}`);
  }


  // RAF-throttled mousemove -> alignment update for the convolution overlay.
  // Uses cached conv x-scale from the most recent render, so alignment updates are cheap.
  function initConvOverlayHandlers() {
    if (state.__convOverlayInit) return;
    state.__convOverlayInit = true;

    cOverlay
      .on('mousemove', (ev) => {
        if (state.sweepActive || state.alignLocked || state.isZooming || ev.buttons) return;
        if (!state.convX || !state.convLayout) return;

        const [px] = d3.pointer(ev, cOverlay.node());
        state.__pendingAlignPx = px;

        if (state.__alignRAF) return;
        state.__alignRAF = requestAnimationFrame(() => {
          state.__alignRAF = null;
          if (!state.convX || !state.convLayout) return;
          const {
            tMin,
            tMax
          } = state.convLayout;
          const t = clamp(Math.round(state.convX.invert(state.__pendingAlignPx)), tMin, tMax);
          setAlign(t, {
            light: true
          });
        });
      })
      .on('click', (ev) => {
        if (state.sweepActive || state.isZooming) return;
        if (!state.convX || !state.convLayout) return;

        if (ev.shiftKey) {
          state.alignLocked = false;
          updateAlignLabel();
          return;
        }
        state.alignLocked = true;
        updateAlignLabel();

        const [px] = d3.pointer(ev, cOverlay.node());
        const {
          tMin,
          tMax
        } = state.convLayout;
        const t = clamp(Math.round(state.convX.invert(px)), tMin, tMax);
        setAlign(t, {
          light: true,
          animateTaps: true
        });
      });
  }

  function updateConvAlignLine() {
    if (!state.convLayout || !state.convX) return;
    const {
      innerH
    } = state.convLayout;
    const hx = state.convX(state.align);
    cAlign
      .attr('x1', hx).attr('x2', hx)
      .attr('y1', 0).attr('y2', innerH)
      .attr('opacity', 1);
  }

  function animateTapBuild() {
    const circles = sTapsG.selectAll('circle.tap').nodes();
    if (!circles.length) return;
    circles.forEach((node, i) => {
      if (typeof gsap !== 'undefined') {
        gsap.killTweensOf(node);
        gsap.to(node, {
          attr: {
            r: 9.0
          },
          duration: 0.10,
          yoyo: true,
          repeat: 1,
          delay: i * 0.035,
          ease: 'power1.inOut'
        });
      }
    });
  }

  function setAlign(newAlign, opts) {
    if (state.series.length === 0) return;
    const maxAlign = el.align ? (+el.align.max) : Math.max(0, state.series.length - 1);
    const idx = clamp(Math.round(newAlign), 0, maxAlign);
    if (idx === state.align) return;

    state.align = idx;
    if (el.align) el.align.value = state.align;

    if (opts && opts.animateTaps) state.animateTapsNext = true;
    updateAlignLabel();

    if (opts && opts.light) {
      renderSeriesFastAlign();
      updateConvAlignLine();
    } else {
      renderSeriesAndConv();
    }
  }

  let lastConv = null;

  function renderConv() {
    const k = getKernel();
    const seriesY = state.series.map(d => d.y);

    // Optional: load convolution files computed by the Python pipeline (when available).
    let fromFile = null;
    const featureIdx = (el.featureIdx ? (+el.featureIdx.value) : 0);

    if (peState && peState.lastRun && peState.instanceId != null) {
      const key = `${peState.instanceId}:${featureIdx}`;

      // Only trigger a (cached) load when the feature changes.
      if (state.__convFromFileKey !== key) {
        state.__convFromFileKey = key;
        state.convFromFile = null;

        peLoadConvForFeature(featureIdx)
          .then(pack => {
            if (!pack) return;
            if (state.__convFromFileKey !== key) return;
            state.convFromFile = pack.resp;
            // Re-render once with the loaded file data.
            renderConv();
          })
          .catch(() => { });
      } else if (state.convFromFile) {
        fromFile = state.convFromFile;
      }
    }

    // Compute convolution (cached).
    const cacheKey = `${k.id}|${state.dilation}|${state.seriesVersion}`;
    let computed = state.convCache.get(cacheKey);
    if (!computed) {
      computed = convolve(seriesY, k.weights, state.dilation);
      state.convCache.set(cacheKey, computed);
    }

    const resp = (fromFile && fromFile.length) ? fromFile : computed.resp;
    const tMin = 0;
    const tMax = (resp && resp.length) ? resp[resp.length - 1].t : computed.tMax;
    const receptive = computed.receptive;

    lastConv = {
      resp,
      tMin,
      tMax,
      receptive
    };

    // Align bounds follow the available response range.
    state.align = clamp(state.align, tMin, tMax);
    if (el.align) el.align.value = state.align;
    updateAlignLabel();

    // Threshold display range (robust to outliers).
    const values = resp.map(d => d.v);
    const lo = quantile(values, 0.01),
      hi = quantile(values, 0.99);
    const span = Math.max(1e-6, hi - lo);
    const pad = 0.20 * span;
    let thrMin = lo - pad,
      thrMax = hi + pad;
    if (Number.isFinite(state.threshold)) {
      thrMin = Math.min(thrMin, state.threshold);
      thrMax = Math.max(thrMax, state.threshold);
    }
    state.thresholdRange = {
      min: thrMin,
      max: thrMax
    };
    updateThresholdDisplay(thrMin, thrMax);

    const {
      innerW,
      innerH
    } = convChart.resize();
    cOverlay.attr('x', 0).attr('y', 0).attr('width', innerW).attr('height', innerH);
    if (convZoom) {
      convZoom.extent([
        [0, 0],
        [innerW, innerH]
      ]).translateExtent([
        [0, 0],
        [innerW, innerH]
      ]);
    }
    cAxes.x.attr('transform', `translate(0,${innerH})`);
    cAxes.y.attr('transform', 'translate(0,0)');

    // Cache the conv layout + x scale for fast alignment updates.
    const x = currentXScale(innerW);
    state.convX = x;
    state.convLayout = {
      tMin,
      tMax,
      innerW,
      innerH
    };

    initConvOverlayHandlers();

    const y = d3.scaleLinear().domain(d3.extent(resp, d => d.v)).nice().range([innerH, 0]);
    const line = d3.line().x(d => x(d.t)).y(d => y(d.v));
    const area = d3.area()
      .x(d => x(d.t))
      .y0(y(state.threshold))
      .y1(d => y(Math.max(d.v, state.threshold)));

    cPath.attr('d', line(resp));
    cArea.attr('d', area(resp));

    cThr
      .attr('x1', x(tMin)).attr('x2', x(tMax))
      .attr('y1', y(state.threshold)).attr('y2', y(state.threshold));

    updateConvAlignLine();

    // Footprint computed from the current response.
    state.footprint = computeFootprint(resp, state.threshold, k.weights, state.dilation, state.series.length);

    cAxes.x.call(d3.axisBottom(x).ticks(6).tickSizeOuter(0));
    cAxes.y.call(d3.axisLeft(y).ticks(5).tickSizeOuter(0));
    styleAxes(cG);
  }

  function renderSummary() {
    if (!lastConv) return;
    const {
      resp,
      tMin,
      tMax,
      receptive
    } = lastConv;
    const vals = resp.map(d => d.v);
    const ppv = vals.length ? (vals.filter(v => v > state.threshold).length / vals.length) : 0;
    const peaks = [...resp].map(d => ({
      ...d,
      av: Math.abs(d.v)
    })).sort((a, b) => d3.descending(a.av, b.av)).slice(0, 3).sort((a, b) => d3.ascending(a.t, b.t));
    const peaksStr = peaks.map(d => `t=${d.t} (v=${fmt(d.v)})`).join(' · ');
    el.summary.innerHTML = `Support: <span class="kbd">t ∈ [${tMin}, ${tMax}]</span> · Receptive field: <span class="kbd">${receptive}</span> · PPV(v > thr): <span class="kbd">${(ppv * 100).toFixed(1)}%</span> · Top |v|: <span class="kbd">${peaksStr || '—'}</span>`;
  }

  function cssVar(name, fallback) {
    try {
      const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
      return v || fallback;
    } catch (_) {
      return fallback;
    }
  }

  function fmtProb01(x) {
    const v = +x;
    if (!Number.isFinite(v)) return '—';
    return d3.format('.2f')(clamp(v, 0, 1));
  }

  function predToClass(pred) {
    const p = +pred;
    if (!Number.isFinite(p)) return 'neutral';
    return (p === 1) ? 'good' : 'bad';
  }


  function drawInstanceSlope(svgEl, meta, alpha) {
    if (!svgEl) return;

    // console.log("alpha: " + alpha);

    const W = 500;
    const H = 150;
    const top = 26;
    const bottom = 20;

    const svg = d3.select(svgEl);
    svg.attr('viewBox', `0 0 ${W} ${H}`);
    svg.selectAll('*').remove();

    // If metadata is missing, keep the minimal alpha-only signal (same semantics as before).
    if (!meta) {
      const g0 = svg.append('g');
      const textMuted = 'rgba(31,41,55,.70)';
      return;
    }

    const refPRaw = +meta.reference_predicted_probability;
    const instPRaw = +meta.instance_predicted_probability;
    const refP = Number.isFinite(refPRaw) ? clamp(refPRaw, 0, 1) : 0;
    const instP = Number.isFinite(instPRaw) ? clamp(instPRaw, 0, 1) : 0;
    const refPred = meta.reference_prediction;
    const instPred = meta.instance_prediction;

    const x0 = 60;
    const x1 = 170;
    const xDelta = 245;
    const xFirstRect = 300;

    const y = d3.scaleLinear().domain([0, 1]).range([H - bottom, top]).clamp(true);

    const axisStroke = '#a7a7a7';
    const textMuted = 'rgba(31,41,55,.70)';
    const lineStroke = 'rgb(44,44,44)';
    const referenceColor = 'rgba(99,102,241,.99)';
    const instanceColor = 'rgba(37,99,235,.99)';
    const valueLabelColor = 'rgba(31,41,55,.92)';

    const delta = instP - refP;
    const deltaAbs = Math.abs(delta);
    const ratio = (deltaAbs > 1e-12 && alpha != null && Number.isFinite(+alpha))
      ? clamp(Math.abs(+alpha) / deltaAbs, 0, 1)
      : 0;

    // const contribColor = (delta > 0)
    //   ? 'rgb(16,185,129)'
    //   : (delta < 0)
    //     ? 'rgb(239,68,68)'
    //     : 'rgba(31,41,55,0.55)';

    const contribColor = 'rgba(31,41,55,0.99)';

    const aVals = state.features.map(d => d.alpha);
    const minA = d3.min(aVals);
    const maxA = d3.max(aVals);
    const maxAbs = Math.max(Math.abs(minA || 0), Math.abs(maxA || 0), 1e-12);
    let colorScale;
    if ((minA || 0) < 0 && (maxA || 0) > 0) {
      colorScale = d3.scaleDiverging().domain([-maxAbs, 0, maxAbs]).interpolator(d3.interpolateRdBu);
    } else {
      colorScale = d3.scaleSequential().domain([0, maxAbs]).interpolator(d3.interpolateBlues);
    }
    const alphaColor = (alpha != null && Number.isFinite(+alpha)) ? colorScale(alpha) : 'rgba(31,41,55,0.55)';

    const alphaLabelColor = (alpha != null && Number.isFinite(+alpha) && +alpha > 0)
      ? 'rgb(37,99,235)'
      : (+alpha < 0)
        ? 'rgb(239,68,68)'
        : 'rgba(31,41,55,0.55)';

    const stroke = 'rgba(31,41,55,0.18)';
    const bg = 'rgba(31,41,55,0.06)';

    const fmt2 = d3.format('.2f');
    const deltaLabel = fmt2(deltaAbs);
    const alphaLabel = (alpha != null && Number.isFinite(+alpha)) ? `${fmt2(+alpha)}` : null;
    

    const g = svg.append('g');

    // Axis lines
    g.append('line').attr('x1', x0).attr('x2', x0).attr('y1', top).attr('y2', H - bottom)
      .attr('stroke', axisStroke).attr('stroke-width', 1.5);
    g.append('line').attr('x1', x1).attr('x2', x1).attr('y1', top).attr('y2', H - bottom)
      .attr('stroke', axisStroke).attr('stroke-width', 1.5);

    // Minimal ticks
    const ticks = [0, 1];

    // Left axis ticks
    g.selectAll('line.tickL').data(ticks).enter().append('line')
      .attr('class', 'tickL')
      .attr('x1', x0 - 6).attr('x2', x0)
      .attr('y1', d => y(d)).attr('y2', d => y(d))
      .attr('stroke', axisStroke).attr('stroke-width', 1.5);
    g.selectAll('text.tickLab').data(ticks).enter().append('text')
      .attr('x', x0 - 10)
      .attr('y', d => y(d) + 4)
      .attr('text-anchor', 'end')
      .attr('font-family', cssVar('--mono', 'monospace'))
      .attr('font-size', 11)
      .attr('fill', textMuted)
      .text(d => d3.format('.0f')(d));

    // Right axis ticks
    g.selectAll('line.tickR').data(ticks).enter().append('line')
      .attr('class', 'tickR')
      .attr('x1', x1).attr('x2', x1 + 6)
      .attr('y1', d => y(d)).attr('y2', d => y(d))
      .attr('stroke', axisStroke).attr('stroke-width', 1);
    g.selectAll('text.tickLabR').data(ticks).enter().append('text')
      .attr('x', x1 + 10)
      .attr('y', d => y(d) + 4)
      .attr('text-anchor', 'start')
      .attr('font-family', cssVar('--mono', 'monospace'))
      .attr('font-size', 11)
      .attr('fill', textMuted)
      .text(d => d3.format('.0f')(d));

    // Horizontal dotted line at the minimum probability value
    const minP = Math.min(refP, instP);
    const minY = y(minP);
    g.append('line')
      .attr('x1', x0).attr('x2', xDelta + 6)
      .attr('y1', minY).attr('y2', minY)
      .attr('stroke', 'rgba(17,24,39,.35)')
      .attr('stroke-width', 1)
      .attr('stroke-dasharray', '4 3')
      .attr('opacity', 0.7);

    // Horizontal dotted line at the maximum probability value
    const maxP = Math.max(refP, instP);
    const maxY = y(maxP);
    const maxPx = (maxP === refP) ? x0 : x1;
    g.append('line')
      .attr('x1', maxPx).attr('x2', xDelta + 6)
      .attr('y1', maxY).attr('y2', maxY)
      .attr('stroke', 'rgba(17,24,39,.35)')
      .attr('stroke-width', 1)
      .attr('stroke-dasharray', '4 3')
      .attr('opacity', 0.7);

    // Titles
    g.append('text').attr('x', x0).attr('y', 12).attr('text-anchor', 'middle')
      .attr('font-weight', 600).attr('font-size', 18).attr('fill', 'rgba(31,41,55,.92)')
      .text('Reference');
    g.append('text').attr('x', x1).attr('y', 12).attr('text-anchor', 'middle')
      .attr('font-weight', 600).attr('font-size', 18).attr('fill', 'rgba(31,41,55,.92)')
      .text('Instance');

    // Slope
    g.append('line')
      .attr('x1', x0).attr('y1', y(refP))
      .attr('x2', x1).attr('y2', y(instP))
      .attr('stroke', lineStroke)
      .attr('stroke-width', 2.2)
      .attr('opacity', 0.9);

    // Endpoints
    g.append('circle').attr('cx', x0).attr('cy', y(refP)).attr('r', 5)
      .attr('fill', referenceColor)
      .attr('stroke', darkenColor(referenceColor, 1.25))
      .attr('stroke-width', 2);
    g.append('circle').attr('cx', x1).attr('cy', y(instP)).attr('r', 5)
      .attr('fill', instanceColor)
      .attr('stroke', darkenColor(instanceColor, 1.25))
      .attr('stroke-width', 2);

    // Value labels
    g.append('text')
      .attr('x', x0 - 12).attr('y', y(refP) + 4)
      .attr('text-anchor', 'end')
      .attr('font-family', cssVar('--mono', 'monospace'))
      .attr('font-weight', 900)
      .attr('font-size', 18)
      .attr('fill', valueLabelColor)
      .text(fmtProb01(refP));
    g.append('text')
      .attr('x', x1 + 12).attr('y', y(instP) + 4)
      .attr('text-anchor', 'start')
      .attr('font-family', cssVar('--mono', 'monospace'))
      .attr('font-weight', 900)
      .attr('font-size', 18)
      .attr('fill', valueLabelColor)
      .text(fmtProb01(instP));

    // Pred labels
    g.append('text')
      .attr('x', x0).attr('y', H + 4)
      .attr('text-anchor', 'middle')
      .attr('font-weight', 600).attr('font-size', 18).attr('fill', 'rgba(31,41,55,.92)')
      .text(`Pred: ${Number.isFinite(+refPred) ? refPred : '—'}`);

    g.append('text')
      .attr('x', x1).attr('y', H + 4)
      .attr('text-anchor', 'middle')
      .attr('font-weight', 600).attr('font-size', 18).attr('fill', 'rgba(31,41,55,.92)')
      .text(`Pred: ${Number.isFinite(+instPred) ? instPred : '—'}`);

    // Δ bracket (difference between reference and instance probabilities)
    const yA = y(refP);
    const yB = y(instP);
    const yTop = Math.min(yA, yB);
    const yBottom = Math.max(yA, yB);
    const yMid = (yTop + yBottom) / 2;

    const tickLen = 12;

    g.append('line')
      .attr('x1', xDelta + 6).attr('x2', xDelta + 6)
      .attr('y1', yTop).attr('y2', yBottom)
      .attr('stroke', contribColor)
      .attr('stroke-width', 3)
      .attr('stroke-linecap', 'round');

    g.append('line')
      .attr('x1', xDelta).attr('x2', xDelta + tickLen)
      .attr('y1', yTop).attr('y2', yTop)
      .attr('stroke', contribColor)
      .attr('stroke-width', 3)
      .attr('stroke-linecap', 'round');

    g.append('line')
      .attr('x1', xDelta).attr('x2', xDelta + tickLen)
      .attr('y1', yBottom).attr('y2', yBottom)
      .attr('stroke', contribColor)
      .attr('stroke-width', 3)
      .attr('stroke-linecap', 'round');

    // White background for delta label
    g.append('rect')
      .attr('x', xDelta - 4).attr('y', yMid - 10)
      .attr('width', 16)
      .attr('height', 20)
      .attr('fill', 'white')
      .attr('stroke', 'none');

    g.append('text')
      .attr('x', xDelta + 5).attr('y', yMid + 5)
      .attr('text-anchor', 'middle')
      .attr('font-family', cssVar('--mono', 'monospace'))
      .attr('font-size', 16)
      .attr('fill', contribColor)
      .text(deltaLabel);

    // Embedding contribution bar (|α| relative to |Δ|)
    const rectWidth = 30;
    const barH = 150;
    const yBarTop = Math.round(top + (H - bottom - top - barH) / 2);
    const yBarBottom = yBarTop + barH;

    // Dotted guide lines: show that the right bar refers to the Δ range
    const xGuide0 = xDelta + tickLen + 4;
    const xGuide1 = xFirstRect - 2;
    g.append('line')
      .attr('x1', xGuide0).attr('y1', yTop)
      .attr('x2', xGuide1).attr('y2', yBarTop)
      .attr('stroke', 'rgba(17,24,39,.35)')
      .attr('stroke-width', 1)
      .attr('stroke-dasharray', '4 3')
      .attr('opacity', 0.7);

    g.append('line')
      .attr('x1', xGuide0).attr('y1', yBottom)
      .attr('x2', xGuide1).attr('y2', yBarBottom)
      .attr('stroke', 'rgba(17,24,39,.35)')
      .attr('stroke-width', 1)
      .attr('stroke-dasharray', '4 3')
      .attr('opacity', 0.7);

    g.append('rect')
      .attr('x', xFirstRect)
      .attr('y', yBarTop)
      .attr('width', rectWidth).attr('height', barH)
      .attr('fill', bg)
      .attr('stroke', stroke);

    // Horizontal line at 0.5 - reference probability within the bar
    const lineY = yBarTop + (barH * (1 - (0.5 - refP) / deltaAbs));
    g.append('line')
      .attr('x1', xFirstRect).attr('x2', xFirstRect + rectWidth)
      .attr('y1', lineY).attr('y2', lineY)
      .attr('stroke', 'rgba(17,24,39,.40)')
      .attr('stroke-width', 1);

    // Label showing the reference probability value (0.5 marker)
    g.append('text')
      .attr('x', xFirstRect + rectWidth / 2).attr('y', lineY - 4)
      .attr('text-anchor', 'middle')
      .attr('font-family', cssVar('--mono', 'monospace'))
      .attr('font-weight', 700)
      .attr('font-size', 12)
      .attr('fill', 'rgba(17,24,39,.70)')
      .text(fmtProb01(0.5 - refP));

    // Arrow showing the decision boundary (class change line)
    g.append('text')
      .attr('x', xFirstRect + rectWidth + 4).attr('y', lineY + 4)
      .attr('text-anchor', 'start')
      .attr('font-family', cssVar('--mono', 'monospace'))
      .attr('font-weight', 700)
      .attr('font-size', 16)
      .attr('fill', 'rgba(17,24,39,.60)')
      .text('←');

    // Label showing the decision boundary (class change line)
    g.append('text')
      .attr('x', xFirstRect + rectWidth + 16).attr('y', lineY - 2)
      .attr('text-anchor', 'start')
      .attr('font-family', cssVar('--mono', 'monospace'))
      .attr('font-weight', 700)
      .attr('font-size', 14)
      .attr('fill', 'rgba(17,24,39,.60)')
      .text('Class');

    g.append('text')
      .attr('x', xFirstRect + rectWidth + 16).attr('y', lineY + 12)
      .attr('text-anchor', 'start')
      .attr('font-family', cssVar('--mono', 'monospace'))
      .attr('font-weight', 700)
      .attr('font-size', 14)
      .attr('fill', 'rgba(17,24,39,.60)')
      .text('flip');

    const dlineLenght = 70;
    const secondDiffX = xFirstRect + rectWidth + dlineLenght;

    // Vertical line connecting the decision boundary to the base of the rectangle
    g.append('line')
      .attr('x1', secondDiffX).attr('x2', secondDiffX)
      .attr('y1', lineY).attr('y2', yBarBottom)
      .attr('stroke', contribColor)
      .attr('stroke-width', 3)
      .attr('stroke-linecap', 'round');

    g.append('line')
      .attr('x1', secondDiffX - tickLen / 2).attr('x2', secondDiffX + tickLen / 2)
      .attr('y1', lineY).attr('y2', lineY)
      .attr('stroke', contribColor)
      .attr('stroke-width', 3)
      .attr('stroke-linecap', 'round');

    g.append('line')
      .attr('x1', secondDiffX - tickLen / 2).attr('x2', secondDiffX + tickLen / 2)
      .attr('y1', yBarBottom).attr('y2', yBarBottom)
      .attr('stroke', contribColor)
      .attr('stroke-width', 3)
      .attr('stroke-linecap', 'round');

    // Horizontal dotted line at the decision boundary (0.5)
    g.append('line')
      .attr('x1', xFirstRect + rectWidth + 2).attr('x2', xFirstRect + rectWidth + dlineLenght)
      .attr('y1', lineY).attr('y2', lineY)
      .attr('stroke', 'rgba(17,24,39,.35)')
      .attr('stroke-width', 1)
      .attr('stroke-dasharray', '4 3')
      .attr('opacity', 0.7);

    // Horizontal dotted line at the base of the rectangle
    g.append('line')
      .attr('x1', xFirstRect + rectWidth + 2).attr('x2', xFirstRect + rectWidth + dlineLenght)
      .attr('y1', yBarBottom).attr('y2', yBarBottom)
      .attr('stroke', 'rgba(17,24,39,.35)')
      .attr('stroke-width', 1)
      .attr('stroke-dasharray', '4 3')
      .attr('opacity', 0.7);

    const startSecondRect = xFirstRect + rectWidth + dlineLenght + 30;

    g.append('rect')
      .attr('x', startSecondRect).attr('y', yBarTop)
      .attr('width', rectWidth).attr('height', barH)
      .attr('fill', bg)
      .attr('stroke', stroke);

    // Line connecting the secon difference to the top of the second rectangle
    g.append('line')
      .attr('x1', xFirstRect + rectWidth + dlineLenght + tickLen / 2).attr('x2', startSecondRect)
      .attr('y1', lineY).attr('y2', yBarTop)
      .attr('stroke', 'rgba(17,24,39,.35)')
      .attr('stroke-width', 1)
      .attr('stroke-dasharray', '4 3')
      .attr('opacity', 0.7);

    // Line connecting the secon difference to the bottom of the second rectangle
    g.append('line')
      .attr('x1', xFirstRect + rectWidth + dlineLenght + tickLen / 2).attr('x2', startSecondRect)
      .attr('y1', yBarBottom).attr('y2', yBarBottom)
      .attr('stroke', 'rgba(17,24,39,.35)')
      .attr('stroke-width', 1)
      .attr('stroke-dasharray', '4 3')
      .attr('opacity', 0.7);

    // Label at the bottom of the second rect showing the decision boundary value
    g.append('text')
      .attr('x', startSecondRect + rectWidth / 2).attr('y', yBarBottom + 16)
      .attr('text-anchor', 'middle')
      .attr('font-family', cssVar('--mono', 'monospace'))
      .attr('font-weight', 900)
      .attr('font-size', 18)
      .attr('fill', contribColor)
      .text('0');

    g.append('text')
      .attr('x', startSecondRect + rectWidth / 2).attr('y', yBarTop - 10)
      .attr('text-anchor', 'middle')
      .attr('font-family', cssVar('--mono', 'monospace'))
      .attr('font-weight', 900)
      .attr('font-size', 18)
      .attr('fill', contribColor)
      .text(fmtProb01(0.5 - refP));

    const fillH = barH * ratio;
    const yFill = yBarTop + (barH - fillH);

    g.append('rect')
      .attr('x', xFirstRect).attr('y', yFill)
      .attr('width', rectWidth)
      .attr('height', fillH)
      .attr('fill', alphaLabelColor);

    const classFlipAbs = Math.abs(0.5 - refP);
    const classFlipRatio = (classFlipAbs > 1e-12 && alpha != null && Number.isFinite(+alpha))
      ? clamp(Math.abs(+alpha) / classFlipAbs, 0, 1)
      : 0;
    const secondFillH = barH * classFlipRatio;
    const secondYFill = yBarTop + (barH - secondFillH);

    g.append('rect')
      .attr('x', startSecondRect).attr('y', secondYFill)
      .attr('width', rectWidth)
      .attr('height', secondFillH)
      .attr('fill', alphaLabelColor);

    g.append('text')
      .attr('x', xFirstRect + rectWidth / 2).attr('y', yBarBottom + 16)
      .attr('text-anchor', 'middle')
      .attr('font-family', cssVar('--mono', 'monospace'))
      .attr('font-weight', 900)
      .attr('font-size', 18)
      .text('0');

    g.append('text')
      .attr('x', xFirstRect + rectWidth / 2).attr('y', yBarTop - 10)
      .attr('text-anchor', 'middle')
      .attr('font-family', cssVar('--mono', 'monospace'))
      .attr('font-weight', 900)
      .attr('font-size', 18)
      .text(deltaLabel);

    if (!alphaLabel) {
      g.append('text')
        .attr('x', xFirstRect + rectWidth / 2).attr('y', yBarTop - 10)
        .attr('text-anchor', 'middle')
        .attr('font-family', cssVar('--mono', 'monospace'))
        .attr('font-weight', 800)
        .attr('font-size', 14)
        .attr('fill', 'rgba(31,41,55,0.70)')
        .text('—');
      return;
    }

    // Place alpha label near the top of the filled segment, clamped to stay readable.
    const yAlpha = clamp(yFill + 4, yBarTop + 10, yBarTop + barH - 6);

    g.append('text')
      .attr('x', xFirstRect + rectWidth + (startSecondRect - xFirstRect - rectWidth) / 2)      
      .attr('y', 5)
      .attr('text-anchor', 'middle')
      .attr('font-family', cssVar('--mono', 'monospace'))
      .attr('font-weight', 800)
      .attr('font-size', 20)
      .attr('fill', alphaLabelColor)
      .text("α =");

    g.append('text')
      .attr('x', xFirstRect + rectWidth + (startSecondRect - xFirstRect - rectWidth) / 2)      
      .attr('y', 25)
      .attr('text-anchor', 'middle')
      .attr('font-family', cssVar('--mono', 'monospace'))
      .attr('font-weight', 800)
      .attr('font-size', 20)
      .attr('fill', alphaLabelColor)
      .text(alphaLabel);
  }



  function renderInstanceDetails() {
    if (!el.instanceDetailsCard) return;

    // Alpha contribution (selected feature) shown relative to the ref→inst probability gap (Δ)
    const f = (state.selectedFeatureIdx != null) ? featureByIdx(state.selectedFeatureIdx) : null;

    // console.log("f:");
    // console.log(f);


    const alpha = (f && Number.isFinite(+f.alpha)) ? +f.alpha : null; // expected in probability units (e.g., 0.05 = 5pp)

    const meta = peState.instanceMeta;

    drawInstanceSlope(el.instanceSlopeChart, meta, alpha);

    if (!meta) {
      if (el.instanceDetailsHint) {
        el.instanceDetailsHint.style.display = 'block';
        el.instanceDetailsHint.textContent = (peState.instanceId == null)
          ? 'Select an instance to see reference vs instance metadata.'
          : 'No metadata_ref_policy_*.json file found for this instance.';
      }
      if (el.instanceMetaGrid) el.instanceMetaGrid.hidden = true;
      return;
    }

    if (el.instanceDetailsHint) el.instanceDetailsHint.style.display = 'none';
    if (el.instanceMetaGrid) el.instanceMetaGrid.hidden = false;

    if (el.instanceMetaBadges) {
      el.instanceMetaBadges.innerHTML = '';
      const mk = (text, cls) => {
        const s = document.createElement('span');
        s.className = `pill ${cls || 'neutral'}`;
        s.textContent = text;
        return s;
      };
      el.instanceMetaBadges.appendChild(mk(`label: ${meta.instance_label ?? '—'}`, 'neutral'));
      el.instanceMetaBadges.appendChild(mk(`ref pred: ${meta.reference_prediction ?? '—'} (p=${fmtProb01(meta.reference_predicted_probability)})`, predToClass(meta.reference_prediction)));
      el.instanceMetaBadges.appendChild(mk(`inst pred: ${meta.instance_prediction ?? '—'} (p=${fmtProb01(meta.instance_predicted_probability)})`, predToClass(meta.instance_prediction)));
    }

    if (el.instanceMetaFootnote) {
      const pol = (peState.lastRun && peState.lastRun.refPolicy) ? peState.lastRun.refPolicy : '—';
      const file = peState.instanceMetaFile ? peState.instanceMetaFile : '—';
      el.instanceMetaFootnote.textContent = `ref policy: ${pol} · file: ${file}`;
    }
  }

  function fixAlignBounds() {
    if (state.series.length === 0) return;
    const n = state.series.length;
    const tMax = Math.max(0, n - 1);
    el.align.min = 0;
    el.align.max = tMax;
    state.align = clamp(state.align, 0, tMax);
    el.align.value = state.align;
    updateAlignLabel();
  }

  function updateDilationLimits() {
    if (state.series.length === 0) return;
    const n = state.series.length;
    const maxD = Math.max(1, Math.min(256, Math.floor((n - 1) / 8)));
    state.maxDilation = maxD;
    state.dilation = clamp(state.dilation, 1, maxD);
    updateDilationDisplay();
    fixAlignBounds();
  }

  function updateDilationDisplay() {
    if (el.dilationLabel) el.dilationLabel.textContent = `d = ${state.dilation}`;
    const minD = 1,
      maxD = 256;
    const eff = Number.isFinite(state.maxDilation) ? state.maxDilation : maxD;
    if (el.dilationSlider) {
      el.dilationSlider.min = String(minD);
      el.dilationSlider.max = String(eff);
      el.dilationSlider.step = '1';
      el.dilationSlider.value = String(clamp(state.dilation, minD, eff));
      el.dilationSlider.disabled = (state.series.length === 0);
    }
    setRangeBar(el.dilationFill, el.dilationMarker, state.dilation, minD, eff);
  }

  function updateThresholdDisplay(minT, maxT) {
    if (el.thresholdLabel) el.thresholdLabel.textContent = `thr = ${fmt(state.threshold)}`;
    const ok = (Number.isFinite(minT) && Number.isFinite(maxT) && maxT > minT);
    if (el.thresholdBounds) el.thresholdBounds.textContent = ok ? `${fmt(minT)}–${fmt(maxT)}` : '—';
    if (el.thresholdSlider) {
      if (ok) {
        const span = Math.max(1e-12, maxT - minT);
        const step = Math.max(span / 800, 1e-6);
        el.thresholdSlider.min = String(minT);
        el.thresholdSlider.max = String(maxT);
        el.thresholdSlider.step = String(step);
        el.thresholdSlider.value = String(clamp(state.threshold, minT, maxT));
        el.thresholdSlider.disabled = (state.series.length === 0);
      } else {
        el.thresholdSlider.disabled = true;
      }
    }
    if (ok) setRangeBar(el.thresholdFill, el.thresholdMarker, state.threshold, minT, maxT);
  }

  function setRangeBar(fillNode, markerNode, value, min, max) {
    if (!fillNode || !markerNode) return;
    if (!Number.isFinite(value) || !Number.isFinite(min) || !Number.isFinite(max)) return;
    if (max <= min) return;
    const r = clamp((value - min) / (max - min), 0, 1);
    const pct = r * 100;
    fillNode.style.width = `${pct}%`;
    markerNode.style.left = `calc(${pct}% - 1px)`;
  }

  async function loadCSVViaFetch(path) {
    const res = await fetch(path, {
      cache: 'no-store'
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.text();
  }
  async function setReferenceSeries(path) {
    state.referencePath = path || '';
    state.referenceSeries = [];
    if (!state.referencePath) {
      if (state.series.length) renderSeries();
      return;
    }
    const fname = state.referencePath.split('/').pop();
    const candidates = Array.from(new Set([state.referencePath, fname, `data/${fname}`].filter(Boolean)));
    try {
      const text = await fetchFirst(candidates);
      state.referenceSeries = parseSeriesCSV(text);
    } catch (err) {
      console.warn('Could not load reference series:', err);
      state.referencePath = '';
      state.referenceSeries = [];
    }
    if (state.series.length) renderSeries();
  }

  function initReferenceToggle() {
    if (!el.showReference) return;
    el.showReference.checked = false;
    el.showReference.addEventListener('change', () => {
      renderSeries();
    });
  }

  function initEmbeddingSortControls() {
    if (!el.embSortKey || !el.embSortDir) return;

    // Initialize from state (so you can later bind a checkbox etc.)
    el.embSortKey.value = state.featureSortKey || 'index';
    const dir0 = state.featureSortDir || 'asc';
    try {
      el.embSortDir.forEach(r => { r.checked = (r.value === dir0); });
    } catch (e) { }

    const apply = () => {
      const key = (el.embSortKey && el.embSortKey.value) ? el.embSortKey.value : 'index';
      const dirEl = document.querySelector('input[name="embSortDir"]:checked');
      const dir = dirEl ? dirEl.value : 'asc';

      if (key === state.featureSortKey && dir === state.featureSortDir) return;

      state.featureSortKey = key;
      state.featureSortDir = dir;
      invalidateFeatureDisplayOrder();

      state.animateFeatureSortNext = true;
      renderFeatureCharts();
    };

    el.embSortKey.addEventListener('change', apply);
    try {
      el.embSortDir.forEach(r => r.addEventListener('change', apply));
    } catch (e) { }
  }





  function setTopBusy(on, text) { }

  function forceHideTopBusy() { }

  function clearVisuals() {
    try {
      d3.select(el.featureOverview).selectAll('*').remove();
    } catch (e) { }
    try {
      d3.select(el.seriesChart).selectAll('*').remove();
    } catch (e) { }
    try {
      d3.select(el.convChart).selectAll('*').remove();
    } catch (e) { }
    if (el.summary) el.summary.textContent = '';
    if (el.seriesInfo) el.seriesInfo.textContent = '';
    if (el.featureInfo) el.featureInfo.textContent = 'α = — · emb = —';
  }

  function clearToEmptyState() {
    clearVisuals();
    if (el.featureInfo) el.featureInfo.textContent = 'α = — · emb = —';
    if (el.seriesInfo) el.seriesInfo.textContent = '';
    if (el.summary) el.summary.textContent = '';
    if (el.emptyState) el.emptyState.style.display = 'block';
    document.body.classList.add('noInstance');
    if (el.peSummaryBtn) el.peSummaryBtn.disabled = true;
    try {
      peState.instanceMeta = null;
      peState.instanceMetaFile = null;
      peState.instanceId = null;
      renderInstanceDetails();
    } catch (e) { }
    if (el.featureInfo) el.featureInfo.textContent = 'α = — · emb = —';
    if (el.seriesInfo) el.seriesInfo.textContent = '';
    clearBetaVariants();
    clearReferencePolicySelect('Current reference');
    retroShowProgress(false);
    syncRetropropButtonState();
    if (el.status) el.status.textContent = 'No instance loaded';
  }

  function hideEmptyState() {
    if (el.emptyState) el.emptyState.style.display = 'none';
  }

  const tabState = {
    tabs: [],
    activeId: null,
    nextId: 1
  };

  function tabKey(cfg, instanceId) {
    return `${cfg.dataset}\n${cfg.model}\n${cfg.explainer}\n${cfg.label}\n${cfg.refPolicy}\n${instanceId}`;
  }

  function renderTabs() {
    if (!el.instanceTabs) return;
    el.instanceTabs.innerHTML = '';
    tabState.tabs.forEach(t => {
      const btn = document.createElement('button');
      btn.className = 'tabBtn' + (t.id === tabState.activeId ? ' active' : '');
      btn.type = 'button';
      btn.textContent = t.label;
      btn.addEventListener('click', () => activateTab(t.id));
      const close = document.createElement('button');
      close.className = 'tabClose';
      close.type = 'button';
      close.textContent = '×';
      close.title = 'Close';
      close.addEventListener('click', (ev) => {
        ev.stopPropagation();
        closeTab(t.id);
      });
      btn.appendChild(close);
      el.instanceTabs.appendChild(btn);
    });
  }

  function closeTab(id) {
    tabState.tabs = tabState.tabs.filter(t => t.id !== id);
    if (tabState.activeId === id) {
      tabState.activeId = tabState.tabs.length ? tabState.tabs[0].id : null;
      renderTabs();
      if (tabState.activeId) activateTab(tabState.activeId);
      else clearToEmptyState();
    } else {
      renderTabs();
    }
  }

  function ensureTab(cfg, instanceId, openNew) {
    const key = tabKey(cfg, instanceId);
    const label = String(instanceId);
    if (!openNew) {
      const active = tabState.tabs.find(t => t.id === tabState.activeId);
      if (active) {
        active.key = key;
        active.cfg = cfg;
        active.instanceId = instanceId;
        active.label = label;
        return active.id;
      }
    }
    const existing = tabState.tabs.find(t => t.key === key);
    if (existing) return existing.id;
    const id = tabState.nextId++;
    tabState.tabs.push({
      id,
      key,
      label,
      cfg,
      instanceId
    });
    tabState.activeId = id;
    return id;
  }
  async function activateTab(id) {
    const t = tabState.tabs.find(x => x.id === id);
    if (!t) return;
    tabState.activeId = id;
    renderTabs();
    if (el.peDataset) el.peDataset.value = t.cfg.dataset;
    if (el.peModel) el.peModel.value = t.cfg.model;
    if (el.peExplainer) el.peExplainer.value = t.cfg.explainer;
    if (el.peLabelSelect) el.peLabelSelect.value = t.cfg.label;
    if (el.peRefPolicy) el.peRefPolicy.value = t.cfg.refPolicy;
    peState.lastRun = t.cfg;
    setTopBusy(true, 'Loading instance…');
    try {
      await peRefreshAvailableInstances(t.cfg);
      if (el.peInstance) el.peInstance.value = String(t.instanceId);
      await peEnsureModelCaches(t.cfg);
      await peLoadInstance(t.cfg, t.instanceId);
      if (el.featureInfo) el.featureInfo.textContent = 'α = — · emb = —';
      if (el.seriesInfo) el.seriesInfo.textContent = '';
      try {
        d3.select(el.convChart).selectAll('*').remove();
      } catch (e) { }
      document.body.classList.remove('noInstance');
      hideEmptyState();
    } finally {
      setTopBusy(false, '');
    }
  }
  setInterval(() => {
    try {
      if (el.tpProgress && !el.tpProgress.hidden && __busyCount === 0) {
        forceHideTopBusy();
      }
    } catch (e) { }
  }, 8000);

  const peState = {
    lastRun: null,
    modelCacheKey: null,
    instanceId: null,
    biases: null,
    dilations: null,
    kernelMaskCache: new Map(),
    convCache: new Map(),
    lastBetaFile: null,
    fileList: null,
    availableConvFeatures: new Set(),

    // Instance metadata shown in #instanceDetailsCard
    instanceMeta: null,
    instanceMetaFile: null,
  };

  function clearBetaVariants() {
    state.betaVariants = new Map();
    state.activeBetaKey = null;
    state.betaRaw = [];
    state.betaStats = null;
    updateBetaVariantSelect();
  }

  function getDefaultBetaVariantKey() {
    return state.betaVariants.has('__beta_default__') ? '__beta_default__' : null;
  }

  function listBetaVariants() {
    const items = Array.from((state.betaVariants || new Map()).values());
    items.sort((a, b) => {
      if (!!a.isBase !== !!b.isBase) return a.isBase ? -1 : 1;
      const af = Number.isFinite(+a.featureId) ? +a.featureId : Number.POSITIVE_INFINITY;
      const bf = Number.isFinite(+b.featureId) ? +b.featureId : Number.POSITIVE_INFINITY;
      return d3.ascending(af, bf) || String(a.label || '').localeCompare(String(b.label || ''));
    });
    return items;
  }

  function updateBetaVariantSelect(preferredKey) {
    if (!el.betaFileSelect) return;
    const variants = listBetaVariants();
    el.betaFileSelect.innerHTML = '';
    if (!variants.length) {
      const opt = document.createElement('option');
      opt.value = '';
      opt.textContent = 'Original β';
      el.betaFileSelect.appendChild(opt);
      el.betaFileSelect.disabled = true;
      return;
    }
    variants.forEach(v => {
      const opt = document.createElement('option');
      opt.value = v.key;
      opt.textContent = v.label;
      el.betaFileSelect.appendChild(opt);
    });
    const activeKey = preferredKey || state.activeBetaKey || variants[0].key;
    el.betaFileSelect.value = variants.some(v => v.key === activeKey) ? activeKey : variants[0].key;
    el.betaFileSelect.disabled = false;
  }

  async function ensureBetaVariantLoaded(key) {
    const v = state.betaVariants.get(key);
    if (!v) return null;
    if (v.loaded) return v;
    if (!peState.lastRun || peState.instanceId == null || !v.file) return v;
    const cfg = peState.lastRun;
    const url = `/output/${cfg.dataset}/${cfg.model}/${cfg.explainer}/${cfg.label}/${peState.instanceId}/${v.file}`;
    const text = await peFetchText(url);
    const raw = parseBetaRowCSV(text);
    const next = {
      ...v,
      loaded: true,
      raw,
      stats: computeBetaStats(raw)
    };
    state.betaVariants.set(key, next);
    return next;
  }

  async function setActiveBetaVariant(key, opts = {}) {
    const variants = listBetaVariants();
    if (!variants.length) {
      state.activeBetaKey = null;
      state.betaRaw = [];
      state.betaStats = null;
      updateBetaVariantSelect();
      if (!opts.skipRender && state.series.length) renderSeries();
      return null;
    }
    const targetKey = (key && state.betaVariants.has(key)) ? key : variants[0].key;
    const v = await ensureBetaVariantLoaded(targetKey);
    state.activeBetaKey = targetKey;
    state.betaRaw = (v && v.raw) ? v.raw : [];
    state.betaStats = (v && v.stats) ? v.stats : null;
    updateBetaVariantSelect(targetKey);

    const shouldSyncFeature = !!(
      opts && opts.syncFeatureSelection &&
      state.features && state.features.length &&
      v && !v.isBase &&
      Number.isFinite(+v.featureId) &&
      !(opts && opts.skipFeatureSync) &&
      !(opts && opts.skipRender)
    );

    if (shouldSyncFeature) {
      selectFeature(Math.round(+v.featureId), {
        silentTooltip: true,
        focusEmbedding: true,
        embeddingZoomK: Number.isFinite(+opts.embeddingZoomK) ? +opts.embeddingZoomK : 8
      });
      return v;
    }

    if (!opts.skipRender && state.series.length) renderSeries();
    return v;
  }

  function findBetaVariantKeyForFeature(featureIdx) {
    const f = Number.isFinite(+featureIdx) ? Math.round(+featureIdx) : null;
    if (!Number.isFinite(f)) return null;
    const hit = listBetaVariants().find(v => !v.isBase && +v.featureId === f);
    return hit ? hit.key : null;
  }

  async function peRefreshBetaVariantsForInstance(cfg, instanceId, opts = {}) {
    if (!cfg || instanceId == null) {
      clearBetaVariants();
      return state.betaVariants;
    }

    let files = opts.files || null;
    if (!files) {
      const entries = await peListEntries(`output/${cfg.dataset}/${cfg.model}/${cfg.explainer}/${cfg.label}/${instanceId}`, 'files');
      files = entries.map(e => e.name);
    }
    peState.fileList = files;

    const prev = new Map(state.betaVariants || []);
    const next = new Map();
    const baseFile = opts.baseFile || findBaseBetaFile(files, cfg, instanceId) || null;

    if (baseFile) {
      const prevBase = prev.get('__beta_default__');
      next.set('__beta_default__', {
        ...(prevBase && prevBase.file === baseFile ? prevBase : {}),
        key: '__beta_default__',
        isBase: true,
        file: baseFile,
        label: betaVariantLabel(baseFile, true),
        featureId: null,
        loaded: !!(prevBase && prevBase.file === baseFile && prevBase.loaded)
      });
    }

    const backpropFiles = findFeatureBetaFiles(files, cfg, instanceId)
      .sort((a, b) => {
        const af = parseFeatureIdFromBetaFilename(a);
        const bf = parseFeatureIdFromBetaFilename(b);
        return d3.ascending(af, bf) || String(a).localeCompare(String(b));
      });

    backpropFiles.forEach(file => {
      const key = file;
      const prevV = prev.get(key);
      next.set(key, {
        ...(prevV || {}),
        key,
        isBase: false,
        file,
        label: betaVariantLabel(file, false),
        featureId: parseFeatureIdFromBetaFilename(file),
        loaded: !!(prevV && prevV.loaded)
      });
    });

    if (opts.baseText != null && baseFile && next.has('__beta_default__')) {
      const raw = parseBetaRowCSV(opts.baseText);
      next.set('__beta_default__', {
        ...next.get('__beta_default__'),
        loaded: true,
        raw,
        stats: computeBetaStats(raw)
      });
    }

    state.betaVariants = next;
    updateBetaVariantSelect(opts.preferredKey || state.activeBetaKey || getDefaultBetaVariantKey());
    return next;
  }

  function peArgsFromCfg(cfg) {
    const c = (cfg && cfg.propagate === 'yes') ? 'yes' : 'no';
    const tVal = (cfg && Number.isFinite(+cfg.t)) ? parseInt(cfg.t, 10) : NaN;
    const args = [
      'predict_and_explain.py',
      '-M', cfg ? cfg.model : 'LogisticRegression',
      '-D', cfg ? cfg.dataset : 'abnormal-heartbeat-c1',
      '-s', String(cfg ? cfg.start : 0),
      '-e', String(cfg ? cfg.end : 1),
      '-E', cfg ? cfg.explainer : 'shap',
      '-L', cfg ? cfg.label : 'predicted',
      '-r', cfg ? cfg.refPolicy : 'opposite_class_farthest_instance'
    ];
    if (c === 'yes') {
      args.push('-c', 'yes');
    } else if (Number.isFinite(tVal) && tVal > 0) {
      args.push('-t', String(tVal));
    } else {
      args.push('-c', 'no');
    }
    return args;
  }

  function retroSetLog(text) {
    if (!el.retroProgressLog) return;
    el.retroProgressLog.textContent = text || '';
    el.retroProgressLog.scrollTop = el.retroProgressLog.scrollHeight;
  }

  function retroShowProgress(show, modeText) {
    if (!el.retroProgressWrap) return;
    el.retroProgressWrap.hidden = !show;
    if (!show) return;
    if (el.retroProgressBar) el.retroProgressBar.classList.add('indeterminate');
    if (el.retroProgressText) el.retroProgressText.textContent = modeText || 'Running…';
    if (el.retroProgressPct) el.retroProgressPct.textContent = '…';
    retroSetLog('');
  }

  function getRequestedFeatureIdx() {
    const max = (state.features && state.features.length) ? (state.features.length - 1) : null;
    const typed = parseInt(el.featureIdx ? el.featureIdx.value : '', 10);
    let v = Number.isFinite(typed) ? typed : ((state.selectedFeatureIdx != null) ? +state.selectedFeatureIdx : NaN);
    if (!Number.isFinite(v)) return null;
    if (max != null) v = clamp(Math.round(v), 0, max);
    return v;
  }

  function syncRetropropButtonState() {
    if (!el.retropropFeatureBtn) return;
    const featureIdx = getRequestedFeatureIdx();
    el.retropropFeatureBtn.disabled = !(peState.lastRun && peState.instanceId != null && Number.isFinite(featureIdx));
  }

  async function retropropagateSelectedFeature() {
    if (!el.retropropFeatureBtn) return;
    const featureIdx = getRequestedFeatureIdx();
    const instanceId = peState.instanceId;
    const baseCfg = peState.lastRun;
    if (!baseCfg || instanceId == null || !Number.isFinite(featureIdx)) return;

    if (state.selectedFeatureIdx !== featureIdx) {
      selectFeature(featureIdx, { silentTooltip: true });
    }

    const cfg = {
      ...baseCfg,
      start: instanceId,
      end: instanceId + 1
    };
    const args = peArgsFromCfg(cfg);
    args.push('-f', String(featureIdx));

    retroShowProgress(true, `Retropropagating feature #${featureIdx}…`);
    if (el.retropropFeatureBtn) el.retropropFeatureBtn.disabled = true;
    retroSetLog(formatCliCommand(args));

    try {
      const res = await fetch(withApiOrigin('/api/run'), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ args })
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok || !data.ok) {
        const msg = (data && (data.error || data.stderr)) ? (data.error || data.stderr) : `Request failed (${res.status})`;
        throw new Error(String(msg));
      }

      const logs = [data.stdout || '', data.stderr || ''].filter(Boolean).join('\n');

      retroSetLog([formatCliCommand(args), logs].filter(Boolean).join('\n\n'));

      await peRefreshBetaVariantsForInstance(cfg, instanceId, {
        preferredKey: state.activeBetaKey
      });

      const featureKey = findBetaVariantKeyForFeature(featureIdx);
      if (!featureKey) {
        throw new Error(`The run finished, but no backpropagated beta file for feature ${featureIdx} was found.`);
      }

      await setActiveBetaVariant(featureKey);

      if (el.retroProgressText) el.retroProgressText.textContent = `Loaded Feature #${featureIdx} β`;
      if (el.retroProgressPct) el.retroProgressPct.textContent = '✓';
      if (el.retroProgressBar) el.retroProgressBar.classList.remove('indeterminate');
    } catch (err) {
      retroSetLog([formatCliCommand(args), String(err)].filter(Boolean).join('\n\n'));
      if (el.retroProgressText) el.retroProgressText.textContent = 'Failed';
      if (el.retroProgressPct) el.retroProgressPct.textContent = '×';
      if (el.retroProgressBar) el.retroProgressBar.classList.remove('indeterminate');
    } finally {
      syncRetropropButtonState();
    }
  }

  function peGetPropagateValue() {
    const sel = document.querySelector('input[name="pePropagate"]:checked');
    return sel ? sel.value : 'no';
  }

  function peSetLog(text) {
    if (!el.peLog) return;
    el.peLog.textContent = text || '';
    el.peLog.scrollTop = el.peLog.scrollHeight;
  }

  function peShowProgress(show, modeText) {
    if (!el.peProgressWrap) return;
    el.peProgressWrap.hidden = !show;
    if (show) {
      if (el.peProgressBar) el.peProgressBar.classList.add('indeterminate');
      if (el.peProgressPct) el.peProgressPct.textContent = '…';
      if (el.peProgressText) el.peProgressText.textContent = modeText || 'Running…';
      peSetLog('');
    }
  }

  function peClampInt(v, min, max, fb) {
    const n = parseInt(v, 10);
    if (!Number.isFinite(n)) return fb;
    return Math.max(min, Math.min(max, n));
  }

  function peNormalizeStartEnd() {
    if (!el.peStartNum || !el.peEndNum) return;
    let start = peClampInt(el.peStartNum.value, 0, 1_000_000_000, 0);
    let end = peClampInt(el.peEndNum.value, start + 1, 1_000_000_000, start + 1);
    if (end <= start) end = start + 1;
    el.peStartNum.value = start;
    el.peEndNum.value = end;
  }

  function peUpdateTEnable() {
    if (!el.peTopT) return;
    const c = peGetPropagateValue();
    const enable = (c === 'no');
    el.peTopT.disabled = !enable;
    if (!enable) el.peTopT.value = '';
  }

  function peBuildArgs() {
    const model = el.peModel ? el.peModel.value : 'LogisticRegression';
    const dataset = el.peDataset ? el.peDataset.value : 'abnormal-heartbeat-c1';
    const label = el.peLabelSelect ? el.peLabelSelect.value : 'predicted';
    const explainer = el.peExplainer ? el.peExplainer.value : 'shap';
    const refPolicy = el.peRefPolicy ? el.peRefPolicy.value : 'opposite_class_farthest_instance';
    const start = el.peStartNum ? parseInt(el.peStartNum.value, 10) : 0;
    const end = el.peEndNum ? parseInt(el.peEndNum.value, 10) : start + 1;
    const c = peGetPropagateValue();
    const tRaw = el.peTopT ? String(el.peTopT.value || '').trim() : '';
    const tVal = tRaw ? parseInt(tRaw, 10) : NaN;
    const cfg = {
      dataset,
      model,
      explainer,
      label,
      refPolicy,
      start,
      end,
      propagate: c,
      t: (Number.isFinite(tVal) ? tVal : null)
    };
    return {
      args: peArgsFromCfg(cfg),
      cfg
    };
  }

  function peUpdateCmdPreview() {
    if (!el.peCmd) return;
    const {
      args
    } = peBuildArgs();
    el.peCmd.textContent = formatCliCommand(args);
  }

  function peSetLoadGenerateAvailability(hasLoadableInstances, statusText = null) {
    if (el.peLoadBtn) el.peLoadBtn.disabled = !hasLoadableInstances;
    if (el.peOpenTabBtn) el.peOpenTabBtn.disabled = !hasLoadableInstances;
    if (el.peAdvancedBtn) {
      el.peAdvancedBtn.disabled = false;
      el.peAdvancedBtn.classList.toggle('attention', !hasLoadableInstances);
    }
    if (statusText != null && el.status) el.status.textContent = statusText;
  }

  function peHasLoadableReferenceArtifacts(files, cfg, instanceId) {
    if (!cfg || instanceId == null) return false;
    const safePolicy = String(cfg.refPolicy || '').replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const safeId = String(instanceId).replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const alphaRx = new RegExp(`^alphas_mr_explanations_ref_policy_${safePolicy}_instance_${safeId}\\.csv$`, 'i');
    return (files || []).some(f => alphaRx.test(String(f || '')));
  }

  async function peFilterInstanceIdsByReference(cfg, ids, baseDir) {
    const checks = await Promise.all((ids || []).map(async id => {
      try {
        const folder = `${baseDir}/${id}`;
        const entries = await peListEntries(folder, 'files');
        const files = entries.map(e => e.name);
        return peHasLoadableReferenceArtifacts(files, cfg, id) ? id : null;
      } catch (e) {
        return null;
      }
    }));
    return checks.filter(Number.isFinite).sort((a, b) => a - b);
  }

  function peSetInstanceAvailability(ids, opts = {}) {
    if (!el.peInstance) return [];
    const list = (ids || []).filter(Number.isFinite).sort((a, b) => a - b);
    const preferredInstanceId = Number.isFinite(+opts.preferredInstanceId) ? +opts.preferredInstanceId : null;
    const noInstancesLabel = opts.noInstancesLabel || '(no instances found for this combination)';
    const emptyStatusText = opts.emptyStatusText || 'No instances found for the current combination. Use Generate files.';

    el.peInstance.innerHTML = '';

    if (!list.length) {
      const opt = document.createElement('option');
      opt.value = '';
      opt.textContent = noInstancesLabel;
      el.peInstance.appendChild(opt);
      el.peInstance.disabled = true;
      peSetLoadGenerateAvailability(false, emptyStatusText);
      return list;
    }

    list.forEach(id => {
      const opt = document.createElement('option');
      opt.value = String(id);
      opt.textContent = String(id);
      el.peInstance.appendChild(opt);
    });

    const cur = (preferredInstanceId != null && list.includes(preferredInstanceId))
      ? preferredInstanceId
      : ((peState.instanceId != null && list.includes(peState.instanceId)) ? peState.instanceId : list[0]);

    el.peInstance.disabled = false;
    el.peInstance.value = String(cur);
    peSetLoadGenerateAvailability(true, `Selected instance ${cur} (click Load)`);
    return list;
  }

  function pePopulateInstanceSelect(start, end) {
    const ids = [];
    for (let i = start; i < end; i++) ids.push(i);
    peSetInstanceAvailability(ids, { preferredInstanceId: start });
  }
  async function peRefreshAvailableInstances(cfg) {
    if (!el.peInstance) return [];
    const baseDir = `output/${cfg.dataset}/${cfg.model}/${cfg.explainer}/${cfg.label}`;
    try {
      const entries = await peListEntries(baseDir, 'dirs');
      const ids = entries.map(e => e.name).filter(n => /^\d+$/.test(n)).map(n => parseInt(n, 10)).filter(Number.isFinite).sort((a, b) => a - b);
      const loadableIds = ids.length ? await peFilterInstanceIdsByReference(cfg, ids, baseDir) : [];
      peSetInstanceAvailability(loadableIds, {
        preferredInstanceId: peState.instanceId,
        noInstancesLabel: ids.length ? '(no files found for this reference policy)' : '(no instances found for this combination)',
        emptyStatusText: ids.length
          ? 'No files found for the selected reference policy. Use Generate files.'
          : 'No instances found for the current combination. Use Generate files.'
      });
      forceHideTopBusy();
      return loadableIds;
    } catch (e) {
      console.error('[peRefreshAvailableInstances] failed', {
        baseDir,
        error: e
      });
      forceHideTopBusy();
      peSetInstanceAvailability([], {
        noInstancesLabel: '(no output folder yet for this combination)',
        emptyStatusText: 'No output folder yet for this combination. Use Generate files.'
      });
      return [];
    }
  }
  async function peFetchText(url) {
    const r = await fetch(withApiOrigin(url));
    if (!r.ok) throw new Error(`HTTP ${r.status} for ${url}`);
    return await r.text();
  }

  async function peFetchJSON(url) {
    const r = await fetch(withApiOrigin(url));
    if (!r.ok) throw new Error(`HTTP ${r.status} for ${url}`);
    return await r.json();
  }
  async function peListEntries(dir, only = "") {
    const q = `/api/list?dir=${encodeURIComponent(dir)}` + (only ? `&only=${encodeURIComponent(only)}` : '');
    const r = await fetch(withApiOrigin(q));
    let j = null;
    try {
      j = await r.json();
    } catch (e) {
      const t = await r.text().catch(() => "");
      console.error('[peListEntries] Non-JSON response', {
        status: r.status,
        text: t.slice(0, 200)
      });
      throw new Error(`list failed (${r.status})`);
    }
    if (!r.ok || !j.ok) {
      console.error('[peListEntries] list failed', {
        status: r.status,
        body: j
      });
      throw new Error(j.error || `list failed (${r.status})`);
    }
    return j.entries || [];
  }

  function setSelectOptions(selectEl, values, preferred) {
    if (!selectEl) return;
    const cur = (preferred != null) ? String(preferred) : String(selectEl.value || '');
    selectEl.innerHTML = '';
    (values || []).forEach(v => {
      const opt = document.createElement('option');
      opt.value = String(v);
      opt.textContent = String(v);
      selectEl.appendChild(opt);
    });
    if ((values || []).includes(cur)) selectEl.value = cur;
  }
  async function peDiscoverOutputOptions() {
    if (!el.peDataset || !el.peModel || !el.peExplainer || !el.peLabelSelect) return;
    const dataset = el.peDataset.value;
    const model = el.peModel.value;
    try {
      const explEntries = await peListEntries(`output/${dataset}/${model}`, 'dirs');
      const explainers = explEntries.map(e => e.name).filter(Boolean).sort();
      if (explainers.length) setSelectOptions(el.peExplainer, explainers, el.peExplainer.value);
    } catch (e) { }
    const explainer = el.peExplainer.value || 'shap';
    try {
      const labEntries = await peListEntries(`output/${dataset}/${model}/${explainer}`, 'dirs');
      const labels = labEntries.map(e => e.name).filter(Boolean).sort();
      if (labels.length) setSelectOptions(el.peLabelSelect, labels, el.peLabelSelect.value);
    } catch (e) { }
  }
  async function peEnsureModelCaches(cfg) {
    const cacheKey = `${cfg.dataset}|${cfg.model}`;
    if (peState.biases && peState.dilations && peState.modelCacheKey === cacheKey) {
      return;
    }
    try {
      const [bText, dText] = await Promise.all([peFetchText(`/output/${cfg.dataset}/${cfg.model}/biases.csv`), peFetchText(`/output/${cfg.dataset}/${cfg.model}/dilations.csv`)]);
      peState.biases = parseIndexValueMapCSV(bText);
      peState.dilations = parseIndexValueMapCSV(dText);
      peState.modelCacheKey = cacheKey;
    } catch (e) {
      console.warn('Could not load biases/dilations:', e);
      peState.biases = null;
      peState.dilations = null;
      peState.modelCacheKey = null;
    }
  }

  function peBiasForFeature(fidx) {
    if (peState.biases && peState.biases.has(fidx)) return +peState.biases.get(fidx);
    return 0;
  }

  function peDilationForFeature(fidx) {
    if (peState.dilations && peState.dilations.has(fidx)) {
      const v = +peState.dilations.get(fidx);
      return Number.isFinite(v) ? Math.max(1, Math.round(v)) : 1;
    }
    return 1;
  }

  async function peLoadKernelMaskForFeature(featureIdx) {
    if (!peState.lastRun) return null;
    const cfg = peState.lastRun;
    const cacheKey = `${cfg.dataset}|${cfg.model}|${featureIdx}`;
    if (peState.kernelMaskCache && peState.kernelMaskCache.has(cacheKey)) return peState.kernelMaskCache.get(cacheKey);

    const url = `/output/${cfg.dataset}/${cfg.model}/base_mask_feature_${featureIdx}.csv`;
    try {
      const text = await peFetchText(url);
      const weights = parseKernelMaskCSV(text, 9);
      const pos2 = [];
      for (let i = 0; i < weights.length; i++) if (weights[i] === 2) pos2.push(i);
      const desc = (pos2.length === 3) ? describeKernel(pos2) : 'Custom mask';
      const pack = { weights, pos2, desc };
      if (peState.kernelMaskCache) peState.kernelMaskCache.set(cacheKey, pack);
      return pack;
    } catch (e) {
      return null;
    }
  }

  function kernelIdFromWeights(weights) {
    if (!weights || !weights.length) return null;
    for (const k of (state.kernels || [])) {
      if (!k || !k.weights || k.weights.length !== weights.length) continue;
      let ok = true;
      for (let i = 0; i < weights.length; i++) {
        if (+k.weights[i] !== +weights[i]) { ok = false; break; }
      }
      if (ok) return k.id;
    }
    return null;
  }


  async function peLoadInstance(cfg, instanceId, opts = {}) {
    peState.instanceId = instanceId;
    retroShowProgress(false);
    peState.convCache.clear();
    peState.instanceMeta = null;
    peState.instanceMetaFile = null;
    state.convFromFile = null;
    const folder = `output/${cfg.dataset}/${cfg.model}/${cfg.explainer}/${cfg.label}/${instanceId}`;
    const entries = await peListEntries(folder, 'files');
    const files = entries.map(e => e.name);
    peState.fileList = files;
    updateReferencePolicySelect(files, instanceId, cfg.refPolicy);
    peState.availableConvFeatures = new Set(files.filter(f => /^convolved_instance_\d+_feature_\d+\.csv$/i.test(String(f || ''))).map(f => {
      const m = f.match(/\_feature\_(\d+)\.csv$/);
      return m ? parseInt(m[1], 10) : null;
    }).filter(Number.isFinite));
    const betasPrefix = `betas_backpropagated_explanations_ref_policy_${cfg.refPolicy}_instance_${instanceId}`;
    const betasFile = findBaseBetaFile(files, cfg, instanceId);
    const instanceFile = `instance_${instanceId}.csv`;
    const refFile = `reference_ref_policy_${cfg.refPolicy}_for_instance_${instanceId}.csv`;
    const alphasFile = `alphas_mr_explanations_ref_policy_${cfg.refPolicy}_instance_${instanceId}.csv`;
    const mrExact = `mr_instance_${instanceId}.csv`;
    const mrCandidates = (files || []).filter(f => {
      const fl = String(f || '').toLowerCase();
      if (fl === mrExact.toLowerCase()) return true;
      if (fl.startsWith(`mr_instance_${instanceId}`) && fl.endsWith('.csv')) return true;
      // Fallback for alternative per-instance embedding filenames (e.g., MR_subion_*_instance_<id>*.csv)
      if ((/^mr_/i.test(f) || /^MR_/i.test(f)) && new RegExp(`instance_${instanceId}.*\.csv$`, 'i').test(f)) return true;
      return false;
    }).sort((a, b) => (a.length - b.length) || String(a).localeCompare(String(b)));
    const mrFile = mrCandidates.length ? mrCandidates[0] : null;
    const seriesUrl = `/output/${cfg.dataset}/${cfg.model}/${cfg.explainer}/${cfg.label}/${instanceId}/${instanceFile}`;
    const refUrl = `/output/${cfg.dataset}/${cfg.model}/${cfg.explainer}/${cfg.label}/${instanceId}/${refFile}`;
    const alphaUrl = `/output/${cfg.dataset}/${cfg.model}/${cfg.explainer}/${cfg.label}/${instanceId}/${alphasFile}`;
    const betaUrl = betasFile ? `/output/${cfg.dataset}/${cfg.model}/${cfg.explainer}/${cfg.label}/${instanceId}/${betasFile}` : null;
    const mrUrl = mrFile ? `/output/${cfg.dataset}/${cfg.model}/${cfg.explainer}/${cfg.label}/${instanceId}/${mrFile}` : null;

    // Optional per-instance metadata file: metadata_ref_policy_<policy>_instance_<id>.json
    const metaJsonExact = `metadata_ref_policy_${cfg.refPolicy}_instance_${instanceId}.json`;
    const metaJsonFile = (files || []).find(f => String(f).toLowerCase() === metaJsonExact.toLowerCase())
      || (files || []).find(f => String(f).toLowerCase().startsWith(`metadata_ref_policy_${String(cfg.refPolicy).toLowerCase()}_instance_`) && String(f).toLowerCase().endsWith(`${instanceId}.json`))
      || (files || []).find(f => String(f).toLowerCase().startsWith('metadata_ref_policy_') && String(f).toLowerCase().includes(`instance_${instanceId}.json`));
    const metaJsonUrl = metaJsonFile ? `/output/${cfg.dataset}/${cfg.model}/${cfg.explainer}/${cfg.label}/${instanceId}/${metaJsonFile}` : null;
    // Log which data files are being rendered for this instance (clickable URLs in console)
    try {
      const toLink = (u) => {
        if (!u) return u;
        try {
          if (typeof withApiOrigin === 'function') return withApiOrigin(u);
          return new URL(u, location.href).href;
        } catch (_) {
          return u;
        }
      };

      console.groupCollapsed(`[peLoadInstance] rendering files · instance ${instanceId}`);
      console.log('folder:', folder);
      console.log('series:', toLink(seriesUrl));
      console.log('alphas:', toLink(alphaUrl));
      console.log('embeddings:', mrUrl ? toLink(mrUrl) : '(missing)');
      console.log('reference:', toLink(refUrl));
      console.log('betas:', betaUrl ? toLink(betaUrl) : '(missing)');
      console.groupEnd();
    } catch (e) { }

    const featureSvg = document.getElementById('featureOverview');
    const seriesSvg = document.getElementById('seriesChart');
    const convSvg = document.getElementById('convChart');
    const morphReference = !!(opts && opts.referenceMorphFrom && el.showReference && el.showReference.checked);
    await Promise.all([
      peFade(featureSvg, 0, 0.12),
      morphReference ? Promise.resolve() : peFade(seriesSvg, 0, 0.12),
      peFade(convSvg, 0, 0.12)
    ]);
    const metaJsonPromise = metaJsonUrl ? peFetchJSON(metaJsonUrl).catch(() => null) : Promise.resolve(null);
    const [seriesText, alphaText, mrText, instMetaJson] = await Promise.all([
      peFetchText(seriesUrl),
      peFetchText(alphaUrl),
      mrUrl ? peFetchText(mrUrl) : Promise.resolve(''),
      metaJsonPromise
    ]);
    let refText = null;
    try {
      refText = await peFetchText(refUrl);
    } catch (e) {
      console.warn('No reference file:', e);
    }
    let betaText = null;
    if (betaUrl) {
      try {
        betaText = await peFetchText(betaUrl);
      } catch (e) {
        console.warn('Could not load betas:', e);
      }
    }

    // console.log('metaText:', metaText);
    // console.log('meta:', meta);


    const alphaMap = parseAlphaCSV(alphaText);
    const embMap = parseEmbeddingCSV(mrText);
    if (mrUrl && embMap.size === 0) {
      console.warn('mr_instance file loaded but produced 0 values. Check CSV format:', mrUrl);
    }

    const meta = buildFeatureMetaFromSignals(alphaMap, embMap);

    // Store & render per-instance metadata (if present)
    if (instMetaJson && typeof instMetaJson === 'object') {
      peState.instanceMeta = instMetaJson;
      peState.instanceMetaFile = metaJsonFile || null;
    } else {
      peState.instanceMeta = null;
      peState.instanceMetaFile = metaJsonFile || null;
    }

    setFeatures(mergeFeatureSignals(meta, alphaMap, embMap));
    if (opts && opts.resetFeatureSelection) resetFeatureSelectionUI();

    renderInstanceDetails();
    await peFade(featureSvg, 1, 0.18);

    // If the series is already previewed for this instance, skip re-parsing and re-setting it.
    const alreadyPreviewed = (state.previewedInstanceId === instanceId && state.series && state.series.length > 0);
    if (!alreadyPreviewed) {
      setSeries(parseSeriesRowCSV(seriesText), {
        deferRender: true
      });
    }
    state.previewedInstanceId = instanceId;

    const nextReferenceSeries = refText ? parseSeriesRowCSV(refText) : null;
    if (nextReferenceSeries) {
      state.referenceSeriesMap = new Map();
      state.referenceSeriesMap.set(cfg.refPolicy, nextReferenceSeries);
      state.referenceSeries = state.referenceSeriesMap.get(cfg.refPolicy);
    } else {
      state.referenceSeriesMap = new Map();
      state.referenceSeries = null;
    }
    clearBetaVariants();
    peState.lastBetaFile = betasFile || null;
    await peRefreshBetaVariantsForInstance(cfg, instanceId, {
      files,
      baseFile: peState.lastBetaFile,
      baseText: betaText || null
    });
    if (getDefaultBetaVariantKey()) {
      await setActiveBetaVariant(getDefaultBetaVariantKey(), { skipRender: true });
    } else {
      state.betaRaw = [];
      state.betaStats = null;
      updateBetaVariantSelect();
    }
    // Remove instancePreview class so heatmaps/overlays become visible
    document.body.classList.remove('instancePreview');
    renderConv();
    if (morphReference && nextReferenceSeries && Array.isArray(opts.referenceMorphFrom) && opts.referenceMorphFrom.length) {
      renderSeriesWithReferenceMorph(opts.referenceMorphFrom, nextReferenceSeries);
    } else {
      renderSeries();
    }
    renderSummary();
    if (!morphReference) await peFade(seriesSvg, 1, 0.18);
    await peFade(convSvg, 1, 0.18);
    syncActiveTabToCurrentLoad(cfg, instanceId);
    renderKernelList();
    renderKernelPreview();
    syncRetropropButtonState();
    console.log(`Loaded output: ${cfg.dataset} · ${cfg.model} · ${cfg.explainer} · ${cfg.label} · instance ${instanceId}`);
  }
  async function peLoadConvForFeature(featureIdx) {
    if (!peState.lastRun || peState.instanceId == null) return null;
    if (peState.availableConvFeatures && !peState.availableConvFeatures.has(featureIdx)) return null;
    const cfg = peState.lastRun;
    const instanceId = peState.instanceId;
    const key = `${instanceId}:${featureIdx}`;
    if (peState.convCache.has(key)) return peState.convCache.get(key);
    const base = `/output/${cfg.dataset}/${cfg.model}/${cfg.explainer}/${cfg.label}/${instanceId}`;
    const convUrl = `${base}/convolved_instance_${instanceId}_feature_${featureIdx}.csv`;
    try {
      try {
        const toLink = (u) => {
          if (!u) return u;
          try {
            if (typeof withApiOrigin === 'function') return withApiOrigin(u);
            return new URL(u, location.href).href;
          } catch (_) {
            return u;
          }
        };
        console.log(`[peLoadConvForFeature] rendering files · instance ${instanceId} · feature ${featureIdx}`, {
          convUrl: toLink(convUrl)
        });
      } catch (e) { }
    } catch (e) { }
    try {
      setTopBusy(true, 'Loading convolution…');
      let __cText;
      try {
        __cText = await peFetchText(convUrl);
      } finally {
        setTopBusy(false, '');
      }
      const resp = parseIndexValueCSV(__cText, 1).map(d => ({
        t: d.t,
        v: d.v
      }));
      const pack = {
        resp
      };
      peState.convCache.set(key, pack);
      return pack;
    } catch (e) {
      return null;
    }
  }
  async function peRun() {
    if (!el.peRunBtn) return;
    peNormalizeStartEnd();
    peUpdateCmdPreview();
    const built = peBuildArgs();
    const args = built.args;
    const cfg = built.cfg;
    peShowProgress(true, 'Running Predict & Explain…');
    if (el.peRunStatus) el.peRunStatus.textContent = 'running…';
    el.peRunBtn.disabled = true;
    if (el.peLoadBtn) el.peLoadBtn.disabled = true;
    if (el.peOpenTabBtn) el.peOpenTabBtn.disabled = true;
    if (el.peInstance) el.peInstance.disabled = true;
    try {
      const res = await fetch(withApiOrigin('/api/run'), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          args
        })
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok || !data.ok) {
        const msg = (data && (data.error || data.stderr)) ? (data.error || data.stderr) : `Request failed (${res.status})`;
        peSetLog(String(msg));
        if (el.peProgressText) el.peProgressText.textContent = 'Failed';
        if (el.peProgressPct) el.peProgressPct.textContent = '×';
        if (el.peProgressBar) el.peProgressBar.classList.remove('indeterminate');
        return;
      }
      peState.lastRun = cfg;
      await peRefreshAvailableInstances(cfg);
      await peEnsureModelCaches(cfg);
      peSetLog([data.stdout || '', data.stderr || ''].filter(Boolean).join('\n'));
      if (el.peProgressText) el.peProgressText.textContent = 'Done (loaded first instance)';
      if (el.peProgressPct) el.peProgressPct.textContent = '✓';
      if (el.peProgressBar) el.peProgressBar.classList.remove('indeterminate');
      await peLoadInstance(cfg, cfg.start);
    } catch (err) {
      peSetLog(String(err));
      if (el.peProgressText) el.peProgressText.textContent = 'Failed';
      if (el.peProgressPct) el.peProgressPct.textContent = '×';
      if (el.peProgressBar) el.peProgressBar.classList.remove('indeterminate');
    } finally {
      el.peRunBtn.disabled = false;
      if (el.peLoadBtn) el.peLoadBtn.disabled = false;
      if (el.peOpenTabBtn) el.peOpenTabBtn.disabled = false;
      if (el.peInstance) el.peInstance.disabled = false;
    }
  }
  async function peLoadSelectedInstance(openNew = false) {
    const built = peBuildArgs();
    const cfg = built.cfg;
    if (!el.peInstance) return;
    const id = parseInt(el.peInstance.value, 10);
    if (!Number.isFinite(id)) return;
    peState.lastRun = cfg;
    const tabId = ensureTab(cfg, id, openNew);
    renderTabs();
    await activateTab(tabId);
  }

  async function switchLoadedReferencePolicy(nextPolicy) {
    const policy = String(nextPolicy || '').trim();
    if (!policy || !peState.lastRun || peState.instanceId == null) return;
    const currentPolicy = String(peState.lastRun.refPolicy || '').trim();
    if (policy === currentPolicy) return;
    const prevReference = Array.isArray(state.referenceSeries) ? state.referenceSeries.map(d => ({ ...d })) : null;
    const nextCfg = {
      ...peState.lastRun,
      refPolicy: policy
    };
    peState.lastRun = nextCfg;
    if (el.peRefPolicy && Array.from(el.peRefPolicy.options || []).some(opt => String(opt.value) === policy)) {
      el.peRefPolicy.value = policy;
    }
    setTopBusy(true, 'Switching reference…');
    try {
      await peEnsureModelCaches(nextCfg);
      await peLoadInstance(nextCfg, peState.instanceId, {
        resetFeatureSelection: true,
        referenceMorphFrom: (el.showReference && el.showReference.checked) ? prevReference : null
      });
    } finally {
      setTopBusy(false, '');
    }
  }

  function peCopy() {
    if (!el.peCmd) return;
    const text = el.peCmd.textContent || '';
    if (!text) return;
    navigator.clipboard.writeText(text).catch(() => { });
  }

  // ---- New: fetch ALL instance IDs for a dataset (across all model/explainer/label combos) ----
  async function peRefreshAllDatasetInstances(dataset) {
    if (!el.peInstance || !dataset) return [];
    try {
      const r = await fetch(withApiOrigin(`/api/dataset-instances?dataset=${encodeURIComponent(dataset)}`));
      const j = await r.json();
      if (!j.ok || !j.ids || !j.ids.length) {
        el.peInstance.innerHTML = '<option value="">(no instances found)</option>';
        el.peInstance.disabled = true;
        return [];
      }
      const ids = j.ids;
      const prev = el.peInstance.value;
      el.peInstance.innerHTML = '';
      ids.forEach(id => {
        const opt = document.createElement('option');
        opt.value = String(id);
        opt.textContent = String(id);
        el.peInstance.appendChild(opt);
      });
      // Restore previous selection if still valid
      if (ids.includes(+prev)) el.peInstance.value = String(prev);
      el.peInstance.disabled = false;
      return ids;
    } catch (e) {
      console.warn('[peRefreshAllDatasetInstances] failed', e);
      el.peInstance.innerHTML = '<option value="">(error loading instances)</option>';
      el.peInstance.disabled = true;
      return [];
    }
  }

  // ---- New: immediate series preview when an instance is selected ----
  async function pePreviewInstanceSeries(dataset, instanceId) {
    if (!dataset || instanceId == null || !Number.isFinite(+instanceId)) return;
    // Try to find instance_N.csv in any model/explainer/label subfolder
    const models = el.peModel ? Array.from(el.peModel.options).map(o => o.value) : [];
    const explainers = el.peExplainer ? Array.from(el.peExplainer.options).map(o => o.value) : [];
    const labels = el.peLabelSelect ? Array.from(el.peLabelSelect.options).map(o => o.value) : [];
    const instanceFile = `instance_${instanceId}.csv`;
    let seriesText = null;
    // Try current selection first, then any combination
    const currentModel = el.peModel ? el.peModel.value : '';
    const currentExplainer = el.peExplainer ? el.peExplainer.value : '';
    const currentLabel = el.peLabelSelect ? el.peLabelSelect.value : '';
    const combos = [];
    if (currentModel && currentExplainer && currentLabel) {
      combos.push([currentModel, currentExplainer, currentLabel]);
    }
    for (const m of models) {
      for (const e of explainers) {
        for (const l of labels) {
          if (m === currentModel && e === currentExplainer && l === currentLabel) continue;
          combos.push([m, e, l]);
        }
      }
    }
    for (const [m, e, l] of combos) {
      try {
        const url = `/output/${dataset}/${m}/${e}/${l}/${instanceId}/${instanceFile}`;
        const text = await peFetchText(url);
        if (text && text.trim()) { seriesText = text; break; }
      } catch (_) { /* try next */ }
    }
    if (!seriesText) {
      console.warn(`[pePreviewInstanceSeries] No instance_${instanceId}.csv found for dataset ${dataset}`);
      return;
    }
    // Set instancePreview mode: show series but hide heatmaps/overlays
    document.body.classList.add('instancePreview');
    document.body.classList.remove('noInstance');
    state.previewedInstanceId = +instanceId;
    setSeries(parseSeriesRowCSV(seriesText), { deferRender: true });
    // Clear overlays
    state.betaRaw = [];
    state.betaStats = null;
    state.footprint = null;
    state.referenceSeries = null;
    // Render the plain series (no heatmaps, no taps)
    renderSeries();
    attachZoom();
  }

  // ---- New: check Load availability for the current secondary-param combination ----
  async function peCheckLoadAvailability() {
    if (!el.peInstance || !el.peInstance.value) return;
    const instanceId = parseInt(el.peInstance.value, 10);
    if (!Number.isFinite(instanceId)) return;
    try {
      const built = peBuildArgs();
      const cfg = built.cfg;
      peState.lastRun = cfg;
      const baseDir = `output/${cfg.dataset}/${cfg.model}/${cfg.explainer}/${cfg.label}/${instanceId}`;
      const entries = await peListEntries(baseDir, 'files').catch(() => []);
      const files = entries.map(e => e.name);
      const hasLoadable = peHasLoadableReferenceArtifacts(files, cfg, instanceId);
      peSetLoadGenerateAvailability(hasLoadable,
        hasLoadable ? `Instance ${instanceId} ready (click Load)` : `No data for this combination — use Generate files`);
    } catch (e) {
      peSetLoadGenerateAvailability(false, 'No data for this combination — use Generate files');
    }
  }

  function peInitUI() {
    if (!el.peRunBtn) return;
    if (!el.peStartNum || !el.peEndNum) return;
    peNormalizeStartEnd();
    peUpdateTEnable();
    peUpdateCmdPreview();

    // On startup, populate instance dropdown and discover output options
    const startDataset = el.peDataset ? el.peDataset.value : '';
    peRefreshAllDatasetInstances(startDataset).then(() => {
      peDiscoverOutputOptions().then(() => {
        peCheckLoadAvailability();
      }).catch(() => {});
    }).catch(() => {});

    // Dataset change → refresh ALL instances for new dataset, then preview
    if (el.peDataset) {
      el.peDataset.addEventListener('change', async () => {
        peUpdateCmdPreview();
        await peRefreshAllDatasetInstances(el.peDataset.value);
        await peDiscoverOutputOptions().catch(() => {});
        // Auto-preview first instance
        if (el.peInstance && el.peInstance.value) {
          await pePreviewInstanceSeries(el.peDataset.value, parseInt(el.peInstance.value, 10));
        }
        await peCheckLoadAvailability();
      });
    }

    // Instance change → immediate series preview
    if (el.peInstance) {
      el.peInstance.addEventListener('change', async () => {
        const v = el.peInstance.value;
        const dataset = el.peDataset ? el.peDataset.value : '';
        if (v && dataset) {
          await pePreviewInstanceSeries(dataset, parseInt(v, 10));
        }
        await peCheckLoadAvailability();
      });
    }

    // Secondary parameters (Label, Model, Explainer, RefPolicy) → only update Load/Generate availability
    [el.peLabelSelect, el.peModel, el.peExplainer, el.peRefPolicy].forEach(n => {
      if (!n) return;
      n.addEventListener('change', async () => {
        peUpdateCmdPreview();
        await peDiscoverOutputOptions().catch(() => {});
        await peCheckLoadAvailability();
      });
    });

    document.querySelectorAll('input[name="pePropagate"]').forEach(r => {
      r.addEventListener('change', () => {
        peUpdateTEnable();
        peUpdateCmdPreview();
      });
    });
    if (el.peTopT) el.peTopT.addEventListener('input', peUpdateCmdPreview);
    if (el.peRunBtn) el.peRunBtn.addEventListener('click', peRun);
    if (el.peCopyBtn) el.peCopyBtn.addEventListener('click', peCopy);
    if (el.peLoadBtn) el.peLoadBtn.addEventListener('click', () => peLoadSelectedInstance(false));
    if (el.peOpenTabBtn) el.peOpenTabBtn.addEventListener('click', () => peLoadSelectedInstance(true));

    if (el.peLoadBtn) el.peLoadBtn.disabled = true;
    if (el.peOpenTabBtn) el.peOpenTabBtn.disabled = true;
  }

  function setSeries(data, opts) {
    state.series = data || [];
    state.seriesVersion++;
    state.zoomTransform = d3.zoomIdentity;
    const tMin = d3.min(state.series, d => d.t);
    const tMax = d3.max(state.series, d => d.t);
    const yMin = d3.min(state.series, d => d.y);
    const yMax = d3.max(state.series, d => d.y);
    if (el.seriesInfo) {
      el.seriesInfo.innerHTML = `<b>Samples:</b> ${state.series.length} · <b>t:</b> [${fmt(tMin)}, ${fmt(tMax)}] · <b>y:</b> [${fmt(yMin)}, ${fmt(yMax)}]`;
    }
    if (el.status) {
      el.status.textContent = `Series loaded (N=${state.series.length})`;
    } else {
      console.log(`Series loaded (N=${state.series.length})`);
    }
    updateDilationLimits();
    state.align = 0;
    if (el.align) el.align.value = 0;
    updateAlignLabel();
    state.threshold = 0;
    attachZoom();
    if (!(opts && opts.deferRender)) {
      renderAll();
    }
  }

  function renderAll() {
    renderKernelPreview();
    renderKernelList();
    renderFeatureCharts();
    renderSeriesAndConv();
  }

  function renderSeriesAndConv() {
    if (state.series.length === 0) return;
    const __animKernel = state.animateKernelTransitionNext;
    renderConv();
    renderSeries();
    renderSummary();
    if (__animKernel) state.animateKernelTransitionNext = false;
  }

  function startSweep() {
    if (state.series.length === 0 || !lastConv) return;
    stopSweep();
    state.sweepActive = true;
    if (el.sweepBtn) {
      el.sweepBtn.classList.add('toggled');
      el.sweepBtn.textContent = 'Stop';
    }
    if (el.align) el.align.disabled = true;
    const tMin = lastConv.tMin;
    const tMax = lastConv.tMax;
    const stepMs = 90;
    let lastStep = performance.now();
    const tick = (now) => {
      if (!state.sweepActive) return;
      if (now - lastStep >= stepMs) {
        lastStep = now;
        const next = (state.align >= tMax) ? tMin : (state.align + 1);
        setAlign(next, {
          light: true,
          animateTaps: true
        });
      }
      state.sweepRaf = requestAnimationFrame(tick);
    };
    state.sweepRaf = requestAnimationFrame(tick);
  }

  function stopSweep() {
    state.sweepActive = false;
    if (state.sweepTimer) {
      clearInterval(state.sweepTimer);
      state.sweepTimer = null;
    }
    if (state.sweepRaf) {
      cancelAnimationFrame(state.sweepRaf);
      state.sweepRaf = null;
    }
    if (el.sweepBtn) {
      el.sweepBtn.classList.remove('toggled');
      el.sweepBtn.textContent = 'Sweep';
    }
    if (el.align) el.align.disabled = false;
  }

  function toggleSweep() {
    state.sweepActive ? stopSweep() : startSweep();
  }

  el.kernelSearch.addEventListener('input', renderKernelList);
  if (el.dilationSlider) {
    el.dilationSlider.addEventListener('input', () => {
      if (state.series.length === 0) return;
      stopSweep();
      state.dilation = +el.dilationSlider.value;
      updateDilationLimits();
      renderSeriesAndConv();
    });
  }
  if (el.thresholdSlider) {
    el.thresholdSlider.addEventListener('input', () => {
      if (state.series.length === 0) return;
      stopSweep();
      state.threshold = +el.thresholdSlider.value;
      renderSeriesAndConv();
    });
  }
  if (el.align) {
    el.align.addEventListener('input', () => {
      stopSweep();
      setAlign(+el.align.value, {
        light: true
      });
    });
  }
  if (el.sweepBtn) {
    el.sweepBtn.addEventListener('click', () => toggleSweep());
  }

  window.addEventListener('keydown', (ev) => {
    if (state.series.length === 0) return;
    const step = ev.shiftKey ? state.dilation : 1;
    const minAlign = el.align ? 0 : (lastConv ? lastConv.tMin : 0);
    const maxAlign = el.align ? (+el.align.max) : (lastConv ? lastConv.tMax : Math.max(0, state.series.length - 1));
    if (ev.key === 'ArrowLeft') {
      stopSweep();
      setAlign(state.align - step, {
        light: true
      });
    }
    if (ev.key === 'ArrowRight') {
      stopSweep();
      setAlign(state.align + step, {
        light: true
      });
    }
    if (ev.key === 'Escape') {
      state.alignLocked = false;
      updateAlignLabel();
      closeDropdown();
    }
  });

  el.resetZoom.addEventListener('click', resetZoom);
  if (el.goFeature && el.featureIdx) {
    const go = () => {
      const max = (state.features && state.features.length) ? state.features.length - 1 : 923;
      const v = clamp(parseInt(el.featureIdx.value, 10) || 0, 0, max);
      selectFeature(v);
    };
    el.goFeature.addEventListener('click', go);
    el.featureIdx.addEventListener('input', syncRetropropButtonState);
    el.featureIdx.addEventListener('keydown', (ev) => {
      if (ev.key === 'Enter') go();
    });
  }
  if (el.betaFileSelect) {
    el.betaFileSelect.addEventListener('change', () => {
      const key = el.betaFileSelect.value;
      setActiveBetaVariant(key, {
        syncFeatureSelection: true,
        embeddingZoomK: 8
      }).catch(err => {
        console.warn('Could not switch beta file:', err);
      });
    });
  }
  if (el.referencePolicySelect) {
    el.referencePolicySelect.addEventListener('change', () => {
      switchLoadedReferencePolicy(el.referencePolicySelect.value).catch(err => {
        console.warn('Could not switch reference policy:', err);
      });
    });
  }
  if (el.retropropFeatureBtn) {
    el.retropropFeatureBtn.addEventListener('click', () => {
      retropropagateSelectedFeature().catch(err => {
        retroSetLog(String(err));
        if (el.retroProgressText) el.retroProgressText.textContent = 'Failed';
        if (el.retroProgressPct) el.retroProgressPct.textContent = '×';
        if (el.retroProgressBar) el.retroProgressBar.classList.remove('indeterminate');
        syncRetropropButtonState();
      });
    });
  }

  function darkenColor(rgbaStr, factor) {
    const match = rgbaStr.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)(?:,\s*([\d.]+))?\)/);
    if (!match) return rgbaStr;
    const r = Math.max(0, Math.floor(parseInt(match[1], 10) * factor));
    const g = Math.max(0, Math.floor(parseInt(match[2], 10) * factor));
    const b = Math.max(0, Math.floor(parseInt(match[3], 10) * factor));
    const a = match[4] ? parseFloat(match[4]) : 1;
    return `rgba(${r},${g},${b},${a})`;
  }

  // --- Generate new instance files popover --------------------------------------
  const advancedUI = { open: false };

  function positionAdvancedPopover() {
    if (!el.peAdvancedPopover || !el.peAdvancedBtn) return;
    try {
      const btnR = el.peAdvancedBtn.getBoundingClientRect();
      const pop = el.peAdvancedPopover;
      const headerH = document.querySelector('header')?.offsetHeight || 0;

      // If the popover is still hidden, width/height can be 0; ensure it's measurable.
      const w = pop.offsetWidth || 420;
      const h = pop.offsetHeight || 260;

      let left = btnR.left;
      let top = btnR.bottom + 8;

      left = clamp(left, 8, Math.max(8, window.innerWidth - w - 8));
      top = clamp(top, headerH + 6, Math.max(headerH + 6, window.innerHeight - h - 8));

      pop.style.left = `${left}px`;
      pop.style.top = `${top}px`;
    } catch (e) { }
  }

  function openAdvancedPopover() {
    if (!el.peAdvancedPopover || !el.peAdvancedBtn) return;
    el.peAdvancedPopover.hidden = false;
    advancedUI.open = true;
    el.peAdvancedBtn.setAttribute('aria-expanded', 'true');
    positionAdvancedPopover();
  }

  function closeAdvancedPopover() {
    if (!el.peAdvancedPopover || !el.peAdvancedBtn) return;
    el.peAdvancedPopover.hidden = true;
    advancedUI.open = false;
    el.peAdvancedBtn.setAttribute('aria-expanded', 'false');
  }

  function toggleAdvancedPopover() {
    advancedUI.open ? closeAdvancedPopover() : openAdvancedPopover();
  }

  function initAdvancedPopover() {
    if (!el.peAdvancedBtn || !el.peAdvancedPopover) return;

    // Safety: always start closed.
    el.peAdvancedPopover.hidden = true;
    advancedUI.open = false;
    el.peAdvancedBtn.setAttribute('aria-expanded', 'false');

    el.peAdvancedBtn.addEventListener('click', (ev) => {
      ev.preventDefault();
      ev.stopPropagation();
      toggleAdvancedPopover();
    });

    if (el.peAdvancedCloseBtn) {
      el.peAdvancedCloseBtn.addEventListener('click', (ev) => {
        ev.preventDefault();
        closeAdvancedPopover();
      });
    }

    // Close on outside click
    document.addEventListener('mousedown', (ev) => {
      if (!advancedUI.open) return;
      const t = ev.target;
      if (el.peAdvancedPopover.contains(t) || el.peAdvancedBtn.contains(t)) return;
      closeAdvancedPopover();
    });

    window.addEventListener('resize', () => {
      if (advancedUI.open) positionAdvancedPopover();
    });

    // Escape always closes this popover (even when no instance is loaded)
    window.addEventListener('keydown', (ev) => {
      if (ev.key === 'Escape' && advancedUI.open) closeAdvancedPopover();
    });
  }




  // --- Summary modal (Top embeddings) ---------------------------------------------
  const summaryUI = {
    open: false,
    dragging: false,
    dragOffX: 0,
    dragOffY: 0,
    selectedFidx: null,
    requestScrollToKernel: false,
    refreshToken: 0,

    // Feature-matrix sorting inside the Summary modal
    sortKey: 'attribution',   // 'feature' | 'attribution' | 'threshold' | 'dilation'
    sortDir: 'desc',       // 'asc' | 'desc'
    lastPickedRows: null,
    lastMaxAbsAlpha: 1
  };

  function cfgSig(cfg) {
    if (!cfg) return '';
    return `${cfg.dataset}|${cfg.model}|${cfg.explainer}|${cfg.label}|${cfg.refPolicy}`;
  }

  function getCurrentCfgForSummary() {
    try {
      const built = peBuildArgs();
      return built ? built.cfg : (peState ? peState.lastRun : null);
    } catch (e) {
      return (peState ? peState.lastRun : null);
    }
  }

  function modalSetStatus(text) {
    if (el.peSummaryStatus) el.peSummaryStatus.textContent = text || '—';
  }

  function modalSetSubtitle(text) {
    if (el.peSummarySubtitle) el.peSummarySubtitle.textContent = text || '—';
  }

  function updateSummaryScopeControlsVisibility(scope) {
    if (el.peSummaryRangeControls) el.peSummaryRangeControls.style.display = (scope === 'range') ? 'block' : 'none';
    if (el.peSummaryListControls) el.peSummaryListControls.style.display = (scope === 'list') ? 'block' : 'none';
  }

  function summaryScopeFromUI() {
    const v = el.peSummaryScopeSelect ? String(el.peSummaryScopeSelect.value || '') : '';
    return v || 'active';
  }

  function normalizeInt(v, fb = 0) {
    const n = parseInt(v, 10);
    return Number.isFinite(n) ? n : fb;
  }

  function mean(arr) {
    if (!arr || !arr.length) return NaN;
    let s = 0, k = 0;
    for (const x of arr) {
      if (!Number.isFinite(x)) continue;
      s += x; k += 1;
    }
    return k ? (s / k) : NaN;
  }


  function fmtSci(x) {
    if (x == null || !Number.isFinite(x)) return '—';
    const ax = Math.abs(x);
    // Scientific notation for very small / very large values (useful for thresholds)
    if (ax > 0 && ax < 1e-4) return x.toExponential(2);
    if (ax >= 1e5) return x.toExponential(2);
    // Otherwise, keep a compact fixed format
    if (ax >= 1000) return x.toFixed(0);
    if (ax >= 10) return x.toFixed(2);
    if (ax >= 1) return x.toFixed(3);
    if (ax >= 0.01) return x.toFixed(5);
    return x.toFixed(8);
  }

  function activeAlphaMap() {
    const m = new Map();
    (state.features || []).forEach(d => {
      const f = +d.fidx;
      const a = +d.alpha;
      if (Number.isFinite(f) && Number.isFinite(a)) m.set(f, a);
    });
    return m;
  }

  async function loadAlphaMapForInstance(cfg, instanceId) {
    if (!cfg || instanceId == null) return null;
    const sig = cfgSig(cfg);
    const key = `${sig}|${instanceId}`;
    if (state.summaryCache && state.summaryCache.alphaMapCache && state.summaryCache.alphaMapCache.has(key)) {
      return state.summaryCache.alphaMapCache.get(key);
    }
    const file = `alphas_mr_explanations_ref_policy_${cfg.refPolicy}_instance_${instanceId}.csv`;
    const url = `/output/${cfg.dataset}/${cfg.model}/${cfg.explainer}/${cfg.label}/${instanceId}/${file}`;
    const text = await peFetchText(url);
    const map = parseAlphaCSV(text);
    if (state.summaryCache && state.summaryCache.alphaMapCache) state.summaryCache.alphaMapCache.set(key, map);
    return map;
  }

  function parseInstanceListSpec(spec) {
    const s0 = (spec == null) ? '' : String(spec);
    const s = s0.replace(/\s+/g, '').trim();
    if (!s) return [];
    const parts = s.split(',').map(x => x.trim()).filter(Boolean);
    const out = new Set();
    parts.forEach(tok => {
      const m1 = tok.match(/^(\d+)$/);
      if (m1) { out.add(parseInt(m1[1], 10)); return; }
      const m2 = tok.match(/^(\d+)[\-–](\d+)$/);
      if (m2) {
        let a = parseInt(m2[1], 10), b = parseInt(m2[2], 10);
        if (!Number.isFinite(a) || !Number.isFinite(b)) return;
        if (b < a) [a, b] = [b, a];
        for (let i = a; i <= b; i++) out.add(i);
      }
    });
    return Array.from(out).filter(Number.isFinite).sort((a, b) => a - b);
  }

  function findClassFlipFile(files, instanceId) {
    if (!files || instanceId == null) return null;
    const id = String(instanceId);
    const reId = new RegExp(`(?:^|[^0-9])${id}(?:[^0-9]|$)`);
    const candidates = (files || []).filter(f => {
      const s = String(f || '');
      if (!/selected[_-]?features/i.test(s)) return false;
      if (!/top/i.test(s)) return false;
      if (!/\.csv$/i.test(s)) return false;
      return reId.test(s);
    }).sort((a, b) => (a.length - b.length) || String(a).localeCompare(String(b)));
    for (const f of candidates) {
      const m = String(f).match(/top[_-]?(\d+)/i);
      const topN = m ? parseInt(m[1], 10) : NaN;
      return { file: f, topN: Number.isFinite(topN) ? topN : null };
    }
    return null;
  }

  function parseSelectedFeaturesCSV(text) {
    const rows = d3.csvParseRows(text || '');
    const ids = [];
    rows.forEach(r => {
      if (!r || r.length < 2) return;
      const v = parseInt(String(r[1]).trim(), 10);
      if (Number.isFinite(v)) ids.push(v);
    });
    return ids;
  }

  async function ensureClassFlipFeatureIds(cfg, instanceId) {
    if (!cfg || instanceId == null) return null;
    const sig = cfgSig(cfg);
    const key = `${sig}|${instanceId}`;

    if (state.summaryCache.classFlipKey === key &&
      state.summaryCache.classFlipFeatureIds &&
      state.summaryCache.classFlipTopN) {
      return {
        file: state.summaryCache.classFlipFile,
        topN: state.summaryCache.classFlipTopN,
        featureIds: state.summaryCache.classFlipFeatureIds
      };
    }

    const info = findClassFlipFile(peState ? peState.fileList : null, instanceId);
    if (!info || !info.file || !info.topN) return null;

    const url = `/output/${cfg.dataset}/${cfg.model}/${cfg.explainer}/${cfg.label}/${instanceId}/${info.file}`;
    const text = await peFetchText(url);
    const ids = Array.from(new Set(parseSelectedFeaturesCSV(text).filter(Number.isFinite)));

    state.summaryCache.classFlipKey = key;
    state.summaryCache.classFlipFile = info.file;
    state.summaryCache.classFlipTopN = info.topN;
    state.summaryCache.classFlipFeatureIds = ids;

    return { file: info.file, topN: info.topN, featureIds: ids };
  }

  function updateSummaryScopeSelectOptions() {
    const sel = el.peSummaryScopeSelect;
    if (!sel) return;

    const activeId = (peState && peState.instanceId != null) ? peState.instanceId : null;

    const optActive = sel.querySelector('option[value="active"]');
    if (optActive) optActive.textContent = `active instance (${activeId != null ? activeId : '—'})`;

    const optClass = sel.querySelector('option[value="classflip"]');
    const info = findClassFlipFile(peState ? peState.fileList : null, activeId);

    if (optClass) {
      if (info && info.topN) {
        optClass.disabled = false;
        optClass.textContent = `ClassFlip (Top${info.topN})`;
      } else {
        optClass.disabled = true;
        optClass.textContent = 'ClassFlip (Top—)';
      }
    }

    if (sel.value === 'classflip' && optClass && optClass.disabled) sel.value = 'active';
  }

  async function loadAlphaMapsForIds(cfg, ids) {
    const sig = cfgSig(cfg);
    const uniq = Array.from(new Set((ids || []).filter(Number.isFinite))).sort((a, b) => a - b);
    const base = `/output/${cfg.dataset}/${cfg.model}/${cfg.explainer}/${cfg.label}`;
    const refPolicy = cfg.refPolicy;

    const limit = 6;
    let done = 0;
    const alphaMaps = [];
    const errors = [];

    async function worker(id) {
      const file = `alphas_mr_explanations_ref_policy_${refPolicy}_instance_${id}.csv`;
      const url = `${base}/${id}/${file}`;
      try {
        const text = await peFetchText(url);
        const m = parseAlphaCSV(text);
        alphaMaps.push({ id, map: m });
        const key = `${sig}|${id}`;
        if (state.summaryCache && state.summaryCache.alphaMapCache) state.summaryCache.alphaMapCache.set(key, m);
      } catch (e) {
        errors.push({ id, error: String(e) });
      } finally {
        done += 1;
        modalSetStatus(`Loading α… ${done}/${uniq.length}`);
      }
    }

    modalSetStatus(uniq.length ? `Loading α for ${uniq.length} instance(s)…` : 'No instance ids.');

    const queue = uniq.slice();
    const runners = new Array(Math.min(limit, queue.length)).fill(0).map(async () => {
      while (queue.length) await worker(queue.shift());
    });
    await Promise.all(runners);

    return { sig, ids: uniq, alphaMaps, errors };
  }

  function kernelIdForFeature(fidx) {
    const meta = featureByFidx(fidx);
    if (!meta) return 'K?';
    if (meta.kernel_id) return String(meta.kernel_id);
    if (meta.kernel_index != null && Number.isFinite(+meta.kernel_index)) return `K${String(+meta.kernel_index).padStart(2, '0')}`;
    return 'K?';
  }

  function kernelDescForId(kernelId) {
    const kObj = (state.kernels || []).find(k => k && k.id === kernelId);
    return kObj ? (kObj.desc || '') : '';
  }

  function weightsForKernelId(kernelId) {
    const kObj = (state.kernels || []).find(k => k && k.id === kernelId);
    return kObj ? kObj.weights : Array(9).fill(0);
  }

  function computeModeDilation(rows) {
    const counts = new Map();
    rows.forEach(r => {
      const d = +r.dilation;
      if (!Number.isFinite(d)) return;
      counts.set(d, (counts.get(d) || 0) + 1);
    });
    let bestD = 1;
    let bestC = -1;
    counts.forEach((c, d) => {
      if (c > bestC || (c === bestC && d < bestD)) {
        bestC = c; bestD = d;
      }
    });
    return bestD;
  }

  function alphaFill(alpha, maxAbs) {
    const v = +alpha;
    const ma = Math.max(1e-9, +maxAbs || 1e-9);
    const t = Math.min(1, Math.abs(v) / ma);
    const a = 0.08 + 0.75 * t;
    if (v > 0) return `rgba(37,99,235,${a})`;
    if (v < 0) return `rgba(239,68,68,${a})`;
    return `rgba(17,24,39,${0.08})`;
  }

  function barFillByCount(count, maxCount) {
    const c = +count;
    const m = Math.max(1, +maxCount || 1);
    const t = Math.min(1, c / m);
    const a = 0.10 + 0.75 * t;
    return `rgba(37,99,235,${a})`;
  }

  function sortSummaryRows(rows) {
    const key = summaryUI.sortKey || 'feature';
    const dir = summaryUI.sortDir || 'asc';
    const arr = (rows || []).slice();

    const val = (r) => {
      if (!r) return null;
      if (key === 'feature') return +r.fidx;
      if (key === 'attribution') return +r.meanAbs; // absolute magnitude
      if (key === 'threshold') return +r.bias;
      if (key === 'dilation') return +r.dilation;
      return +r.fidx;
    };

    arr.sort((a, b) => {
      const va = val(a);
      const vb = val(b);
      const aBad = (va == null || !Number.isFinite(va));
      const bBad = (vb == null || !Number.isFinite(vb));
      if (aBad && bBad) return d3.ascending(+a.fidx, +b.fidx);
      if (aBad) return 1;
      if (bBad) return -1;
      const cmp = d3.ascending(va, vb) || d3.ascending(+a.fidx, +b.fidx);
      return cmp;
    });

    if (dir === 'desc') arr.reverse();
    return arr;
  }

  function setSummarySort(key) {
    if (!key) return;
    if (summaryUI.sortKey === key) {
      summaryUI.sortDir = (summaryUI.sortDir === 'asc') ? 'desc' : 'asc';
    } else {
      summaryUI.sortKey = key;
      summaryUI.sortDir = (key === 'feature') ? 'asc' : 'desc';
    }

    const st = el.peSummaryMatrixWrap ? el.peSummaryMatrixWrap.scrollTop : 0;
    const rows = summaryUI.lastPickedRows ? summaryUI.lastPickedRows.slice() : [];
    const maxAbs = Number.isFinite(+summaryUI.lastMaxAbsAlpha) ? +summaryUI.lastMaxAbsAlpha : 1;
    renderSummaryMatrix(sortSummaryRows(rows), maxAbs);
    if (el.peSummaryMatrixWrap) el.peSummaryMatrixWrap.scrollTop = st;
  }


  function scrollKernelPanelToCenter(kernelId) {
    if (!kernelId || !el.peSummaryKernelStrip) return;
    const host = el.peSummaryKernelStrip;
    const rows = host.querySelectorAll('.krow');
    let target = null;
    rows.forEach(r => {
      if (target) return;
      const kid = r.dataset ? r.dataset.kernelId : null;
      if (kid === kernelId) target = r;
    });
    if (!target) return;

    const hostRect = host.getBoundingClientRect();
    const tarRect = target.getBoundingClientRect();
    const curTop = host.scrollTop;
    const offsetTop = (tarRect.top - hostRect.top) + curTop;
    const desired = offsetTop - (host.clientHeight / 2) + (target.clientHeight / 2);
    const nextTop = clamp(desired, 0, Math.max(0, host.scrollHeight - host.clientHeight));

    try { host.scrollTo({ top: nextTop, behavior: 'smooth' }); }
    catch (_) { host.scrollTop = nextTop; }
  }

  function clearSummarySelection() {
    summaryUI.selectedFidx = null;
    summaryUI.requestScrollToKernel = false;
    applySummaryHighlights();
  }

  function applySummaryHighlights() {
    const fidx = summaryUI.selectedFidx;

    // Matrix rows
    if (el.peSummaryMatrixSvg) {
      const sel = d3.select(el.peSummaryMatrixSvg).selectAll('g.matrixRow');
      sel.classed('dim', d => (fidx != null && +d.fidx !== +fidx))
        .classed('active', d => (fidx != null && +d.fidx === +fidx));
    }

    // Kernel rows
    let selKernel = null;
    if (fidx != null) selKernel = kernelIdForFeature(fidx);

    if (el.peSummaryKernelStrip) {
      const rows = el.peSummaryKernelStrip.querySelectorAll('.krow');
      rows.forEach(r => {
        const kid = r.dataset ? r.dataset.kernelId : null;
        const isHit = (selKernel != null && kid === selKernel);
        r.classList.toggle('active', !!isHit);
        r.classList.toggle('dim', (selKernel != null && !isHit));
      });
    }

    if (summaryUI.requestScrollToKernel && selKernel) {
      summaryUI.requestScrollToKernel = false;
      scrollKernelPanelToCenter(selKernel);
    }
  }

  function drawKernelMiniShared(svgNode, weights, dilation, showDilated, sharedXMax) {
    if (!svgNode) return;
    const svg = d3.select(svgNode);
    svg.selectAll('*').remove();

    const W = 240;
    const H = 96;
    svg.attr('viewBox', `0 0 ${W} ${H}`);

    const m = { left: 16, right: 10, top: 8, bottom: 22 };
    const innerW = W - m.left - m.right;
    const innerH = H - m.top - m.bottom;

    const g = svg.append('g').attr('transform', `translate(${m.left},${m.top})`);

    const kLen = (weights && weights.length) ? weights.length : 9;
    const d = Math.max(1, Math.round(+dilation || 1));
    const xMax = showDilated ? Math.max(1, +sharedXMax || (kLen - 1) * d) : (kLen - 1);
    const x = d3.scaleLinear().domain([0, xMax]).range([0, innerW]);
    const y = d3.scaleLinear().domain([-1.5, 2.5]).range([innerH, 0]);

    const axisY = innerH;

    // Axes (min/max + a couple of reference ticks)
    // X axis: 0 .. xMax
    g.append('line')
      .attr('x1', 0).attr('x2', innerW)
      .attr('y1', axisY).attr('y2', axisY)
      .attr('stroke', 'rgba(17,24,39,.20)');

    const xTickVals = (() => {
      const numTicks = showDilated ? 5 : 9;
      const out = [];
      const step = Math.max(1, Math.floor(xMax / (numTicks - 1)));
      for (let i = 0; i <= xMax; i += step) {
        out.push(i);
      }
      if (out[out.length - 1] !== xMax) {
        out.push(xMax);
      }
      return out;
    })();

    g.selectAll('line.xTick').data(xTickVals).enter().append('line')
      .attr('class', 'xTick')
      .attr('x1', v => x(v)).attr('x2', v => x(v))
      .attr('y1', axisY).attr('y2', axisY + 4)
      .attr('stroke', 'rgba(17,24,39,.22)');

    g.selectAll('text.xTickLab').data(xTickVals).enter().append('text')
      .attr('class', 'xTickLab')
      .attr('x', v => x(v))
      .attr('y', axisY + 14)
      .attr('text-anchor', v => (v === 0 ? 'start' : (v === xMax ? 'end' : 'middle')))
      .attr('font-family', cssVar('--mono', 'monospace'))
      .attr('font-size', 8.5)
      .attr('fill', 'rgba(17,24,39,.55)')
      .text(v => String(v));

    // Y reference ticks (-1, 0, 2)
    const yTickVals = [-1, 0, 2];
    g.selectAll('line.yTick').data(yTickVals).enter().append('line')
      .attr('class', 'yTick')
      .attr('x1', -4).attr('x2', 0)
      .attr('y1', v => y(v)).attr('y2', v => y(v))
      .attr('stroke', 'rgba(17,24,39,.22)');

    g.selectAll('text.yTickLab').data(yTickVals).enter().append('text')
      .attr('class', 'yTickLab')
      .attr('x', -6)
      .attr('y', v => y(v) + 3)
      .attr('text-anchor', 'end')
      .attr('font-family', cssVar('--mono', 'monospace'))
      .attr('font-size', 8.5)
      .attr('fill', 'rgba(17,24,39,.55)')
      .text(v => String(v));

    const pts = (weights || []).map((w, i) => ({ w: +w, i, x: showDilated ? (i * d) : i }));

    g.selectAll('line.stem').data(pts).enter().append('line')
      .attr('class', 'stem')
      .attr('x1', p => x(p.x)).attr('x2', p => x(p.x))
      .attr('y1', axisY).attr('y2', p => y(p.w))
      .attr('stroke', 'rgba(17,24,39,.12)')
      .attr('stroke-width', 1);

    const line = d3.line().x(p => x(p.x)).y(p => y(p.w));
    g.append('path')
      .attr('fill', 'none')
      .attr('stroke', 'rgba(37,99,235,.85)')
      .attr('stroke-width', 2)
      .attr('d', line(pts));

    g.selectAll('circle.dot').data(pts).enter().append('circle')
      .attr('class', 'dot')
      .attr('cx', p => x(p.x))
      .attr('cy', p => y(p.w))
      .attr('r', 4.0)
      .attr('fill', p => (p.w > 0 ? 'rgba(16,185,129,.85)' : 'rgba(239,68,68,.70)'))
      .attr('stroke', 'rgba(17,24,39,.18)')
      .attr('stroke-width', 1);
  }

  function renderSummaryMatrix(rows, maxAbsAlpha) {
    const svgEl = el.peSummaryMatrixSvg;
    if (!svgEl) return;
    const svg = d3.select(svgEl);
    svg.selectAll('*').remove();

    const wrapW = el.peSummaryMatrixWrap ? el.peSummaryMatrixWrap.clientWidth : 860;

    const col = {
      idW: 40, // feature id width
      alphaW: 320, // attribution width
      thrW: 320,
      dW: 160,
      gap: 62 // space between columns
    };
    const m = { left: 10, right: 10, top: 10, bottom: 10 };
    const W = Math.max(860, Math.min(1040, wrapW || 900));
    const innerW = W - m.left - m.right;

    const x0 = 0;
    const xId = x0;
    const xA = xId + col.idW + col.gap;
    const xThr = xA + col.alphaW + col.gap;
    const xD = xThr + col.thrW + col.gap;
    const usable = xD + col.dW;

    const rowH = 22;
    const headerH = 26;
    const H = m.top + m.bottom + headerH + rowH * (rows ? rows.length : 0) + 10;

    svg.attr('viewBox', `0 0 ${W} ${H}`);

    const g = svg.append('g').attr('transform', `translate(${m.left},${m.top})`);

    const toggleSelect = (fidx) => {
      const f = +fidx;
      if (!Number.isFinite(f)) return;

      if (summaryUI.selectedFidx != null && +summaryUI.selectedFidx === f) {
        clearSummarySelection();
        return;
      }
      summaryUI.selectedFidx = f;
      summaryUI.requestScrollToKernel = true;
      applySummaryHighlights();
    };

    // background click clears selection
    g.append('rect')
      .attr('x', 0).attr('y', 0)
      .attr('width', innerW).attr('height', H)
      .attr('fill', 'transparent')
      .style('pointer-events', 'all')
      .on('click', () => { clearSummarySelection(); });

    const thrVals = (rows || []).map(r => +r.bias).filter(Number.isFinite);
    let thrExtent = d3.extent(thrVals);
    if (!thrExtent[0] && thrExtent[0] !== 0) thrExtent = [-1, 1];
    if (thrExtent[0] === thrExtent[1]) {
      const v = thrExtent[0];
      thrExtent = [v - 1, v + 1];
    }
    const maxAbsThr = Math.max(Math.abs(thrExtent[0]), Math.abs(thrExtent[1]), 1e-6);

    const dilVals = (rows || []).map(r => +r.dilation).filter(Number.isFinite);
    const maxDil = Math.max(1, d3.max(dilVals) || 1);

    const alphaBuffer = 40;
    const xAlpha = d3.scaleLinear().domain([-maxAbsAlpha, maxAbsAlpha]).range([xA, xA + col.alphaW - alphaBuffer]).nice();

    const thrBuffer = 40;
    const xThrScale = d3.scaleLinear().domain([-maxAbsThr, maxAbsThr]).range([xThr, xThr + col.thrW - thrBuffer]).nice();

    const dilBuffer = 40;
    const xDil = d3.scaleLinear().domain([0, maxDil]).range([xD, xD + col.dW - dilBuffer]).nice();

    // headers (click to sort asc/desc)
    const header = g.append('g').attr('transform', `translate(0,0)`);
    const hStyle = (sel) => sel
      .attr('font-family', cssVar('--sans', 'sans-serif'))
      .attr('font-size', 11)
      .attr('font-weight', 900)
      .attr('fill', 'rgba(17,24,39,.72)');

    const arrowFor = (key) => {
      if (summaryUI.sortKey !== key) return '';
      return (summaryUI.sortDir === 'asc') ? ' ▲' : ' ▼';
    };

    function headerCell(x, w, label, key) {
      const hg = header.append('g')
        .attr('transform', `translate(${x},0)`)
        .style('cursor', 'pointer')
        .on('click', (ev) => {
          ev.stopPropagation();
          setSummarySort(key);
        });

      hg.append('rect')
        .attr('x', 0).attr('y', 0)
        .attr('width', w).attr('height', headerH)
        .attr('fill', 'transparent');

      hStyle(hg.append('text')
        .attr('x', w / 2)
        .attr('y', 16)
        .attr('text-anchor', 'middle')
        .text(label + arrowFor(key)));
    }

    headerCell(xId, col.idW, 'Feature #', 'feature');
    headerCell(xA, col.alphaW - alphaBuffer, 'Attribution', 'attribution');
    headerCell(xThr, col.thrW - thrBuffer, 'Threshold', 'threshold');
    headerCell(xD, col.dW - dilBuffer, 'Dilation', 'dilation');

    // Mini x-axes (show min/max for each column)
    const axG = g.append('g').attr('class', 'matrixMiniAxes');
    const axY = headerH - 3;

    // Draw a mini axis with ticks and min/max labels
    function drawMiniAxis(scale, xStart, w, tickVals, labelMin, labelMax) {
      // baseline
      axG.append('line')
        .attr('x1', xStart).attr('x2', xStart + w)
        .attr('y1', axY).attr('y2', axY)
        .attr('stroke', 'rgba(17,24,39,.14)');

      // ticks
      (tickVals || []).forEach(v => {
        const px = scale(v);
        axG.append('line')
          .attr('x1', px).attr('x2', px)
          .attr('y1', axY - 2).attr('y2', axY + 2)
          .attr('stroke', 'rgba(17,24,39,.16)');
      });

      // min/max labels
      axG.append('text')
        .attr('x', xStart)
        .attr('y', axY - 5)
        .attr('text-anchor', 'middle')
        .attr('font-family', cssVar('--mono', 'monospace'))
        .attr('font-size', 9)
        .attr('fill', 'rgba(17,24,39,.55)')
        .text(labelMin);

      axG.append('text')
        .attr('x', xStart + w)
        .attr('y', axY - 5)
        .attr('text-anchor', 'middle')
        .attr('font-family', cssVar('--mono', 'monospace'))
        .attr('font-size', 9)
        .attr('fill', 'rgba(17,24,39,.55)')
        .text(labelMax);
    }

    const aDom = xAlpha.domain();
    const tDom = xThrScale.domain();
    const dDom = xDil.domain();

    drawMiniAxis(
      xAlpha,
      xA,
      col.alphaW - alphaBuffer,
      [aDom[0], 0, aDom[1]],
      d3.format('.3f')(fmtSci(aDom[0])),
      d3.format('.3f')(fmtSci(aDom[1]))
    );

    drawMiniAxis(
      xThrScale,
      xThr,
      col.thrW - thrBuffer,
      [tDom[0], 0, tDom[1]],
      d3.format('.3f')(fmtSci(tDom[0])),
      d3.format('.3f')(fmtSci(tDom[1]))
    );

    drawMiniAxis(
      xDil,
      xD,
      col.dW - dilBuffer,
      [dDom[0], (dDom[0] + dDom[1]) / 2, dDom[1]],
      dDom[0].toFixed(0),
      dDom[1].toFixed(0)
    );


    // axis zero lines for alpha + thr
    g.append('line').attr('x1', xAlpha(0)).attr('x2', xAlpha(0)).attr('y1', headerH).attr('y2', H)
      .attr('stroke', 'rgba(17,24,39,.16)');
    g.append('line').attr('x1', xThrScale(0)).attr('x2', xThrScale(0)).attr('y1', headerH).attr('y2', H)
      .attr('stroke', 'rgba(17,24,39,.12)');

    const rowsG = g.append('g').attr('transform', `translate(0,${headerH})`);

    const rowSel = rowsG.selectAll('g.matrixRow')
      .data(rows || [], d => d.fidx)
      .enter().append('g')
      .attr('class', 'matrixRow')
      .attr('transform', (d, i) => `translate(0,${i * rowH})`);

    // Clicking outside bars (row background) clears selection
    rowSel.append('rect')
      .attr('class', 'rowBg')
      .attr('x', 0).attr('y', 0)
      .attr('width', Math.max(innerW, usable + 6)).attr('height', rowH - 2)
      .attr('rx', 6).attr('ry', 6)
      .attr('fill', 'transparent')
      .style('pointer-events', 'all')
      .style('cursor', 'default')
      .on('click', (ev) => {
        ev.stopPropagation();
        clearSummarySelection();
      });

    // Feature id label (click toggles selection)
    rowSel.append('text')
      .attr('x', xId)
      .attr('y', 15)
      .attr('font-family', cssVar('--mono', 'monospace'))
      .attr('font-size', 14)
      .attr('fill', 'rgba(17,24,39,.78)')
      .style('cursor', 'pointer')
      .text(d => `${d.fidx}`)
      .on('click', (ev, d) => {
        ev.stopPropagation();
        toggleSelect(d.fidx);
      });

    // Attribution bars (click toggles selection)
    rowSel.append('rect')
      .attr('class', 'aBar')
      .attr('x', d => (d.meanAlpha >= 0 ? xAlpha(0) : xAlpha(d.meanAlpha)))
      .attr('y', 4)
      .attr('height', rowH - 10)
      .attr('width', d => Math.max(1, Math.abs(xAlpha(d.meanAlpha) - xAlpha(0))))
      .attr('fill', d => alphaFill(d.meanAlpha, maxAbsAlpha))
      .attr('stroke', 'rgba(17,24,39,.10)')
      .style('cursor', 'pointer')
      .on('click', (ev, d) => {
        ev.stopPropagation();
        toggleSelect(d.fidx);
      });

    // Attribution value labels
    rowSel.append('text')
      .attr('class', 'aVal')
      .attr('y', 15)
      .attr('font-family', cssVar('--mono', 'monospace'))
      .attr('font-size', 12)
      .attr('fill', 'rgba(17,24,39,.55)')
      .style('pointer-events', 'none')
      .attr('text-anchor', d => {
        const end = xAlpha(+d.meanAlpha);
        if (+d.meanAlpha >= 0) return (end > (xA + col.alphaW - 24)) ? 'end' : 'start';
        return (end < (xA + 24)) ? 'start' : 'end';
      })
      .attr('x', d => {
        const v = +d.meanAlpha;
        const end = xAlpha(v);
        const pad = 4;
        if (v >= 0) return Math.min(end + pad, xA + col.alphaW - 2);
        return Math.max(end - pad, xA + 2);
      })
      .text(d => fmtSci(+d.meanAlpha));

    // Threshold bars (click toggles selection)
    rowSel.append('rect')
      .attr('class', 'thrBar')
      .attr('x', d => (d.bias >= 0 ? xThrScale(0) : xThrScale(d.bias)))
      .attr('y', 4)
      .attr('height', rowH - 10)
      .attr('width', d => Math.max(1, Math.abs(xThrScale(d.bias) - xThrScale(0))))
      .attr('fill', d => {
        const ma = maxAbsThr;
        const t = Math.min(1, Math.abs(+d.bias) / ma);
        const a = 0.08 + 0.70 * t;
        return `rgba(245,158,11,${a})`;
      })
      .attr('stroke', 'rgba(17,24,39,.10)')
      .style('cursor', 'pointer')
      .on('click', (ev, d) => {
        ev.stopPropagation();
        toggleSelect(d.fidx);
      });

    // Threshold value labels
    rowSel.append('text')
      .attr('class', 'thrVal')
      .attr('y', 15)
      .attr('font-family', cssVar('--mono', 'monospace'))
      .attr('font-size', 12)
      .attr('fill', 'rgba(17,24,39,.55)')
      .style('pointer-events', 'none')
      .attr('text-anchor', d => {
        const end = xThrScale(+d.bias);
        if (+d.bias >= 0) return (end > (xThr + col.thrW - 24)) ? 'end' : 'start';
        return (end < (xThr + 24)) ? 'start' : 'end';
      })
      .attr('x', d => {
        const v = +d.bias;
        const end = xThrScale(v);
        const pad = 4;
        if (v >= 0) return Math.min(end + pad, xThr + col.thrW - 2);
        return Math.max(end - pad, xThr + 2);
      })
      .text(d => fmtSci(+d.bias));

    // Dilation bars (click toggles selection)
    rowSel.append('rect')
      .attr('class', 'dilBar')
      .attr('x', xDil(0))
      .attr('y', 4)
      .attr('height', rowH - 10)
      .attr('width', d => Math.max(1, xDil(+d.dilation) - xDil(0)))
      .attr('fill', d => {
        const t = Math.min(1, (+d.dilation) / maxDil);
        const a = 0.10 + 0.65 * t;
        return `rgba(16,185,129,${a})`;
      })
      .attr('stroke', 'rgba(17,24,39,.10)')
      .style('cursor', 'pointer')
      .on('click', (ev, d) => {
        ev.stopPropagation();
        toggleSelect(d.fidx);
      });

    // Dilation value labels
    rowSel.append('text')
      .attr('class', 'dilVal')
      .attr('x', d => Math.min(xDil(+d.dilation) + 4, xD + col.dW - 2))
      .attr('y', 15)
      .attr('font-family', cssVar('--mono', 'monospace'))
      .attr('font-size', 12)
      .attr('fill', 'rgba(17,24,39,.55)')
      .style('pointer-events', 'none')
      .attr('text-anchor', d => (xDil(+d.dilation) > (xD + col.dW - 24)) ? 'end' : 'start')
      .text(d => String(Math.round(+d.dilation)));

    rowSel.append('title').text(d =>
      `#${d.fidx}
Attribution=${fmt(d.meanAlpha)}
Threshold=${fmtSci(d.bias)}
Dilation=${d.dilation}
${d.kernelId}`
    );

    applySummaryHighlights();
  }

  function renderKernelStrip(kernelRows, thrBinGen, thrBinsTemplate, maxThrBinCount, maxDilCount, showDilated) {
    const host = el.peSummaryKernelStrip;
    if (!host) return;
    host.innerHTML = '';

    if (!kernelRows || !kernelRows.length) {
      const p = document.createElement('div');
      p.className = 'muted';
      p.textContent = 'No kernel data.';
      host.appendChild(p);
      return;
    }

    const sharedXMax = (() => {
      if (!showDilated) return 8;
      let m = 8;
      kernelRows.forEach(r => {
        const d = Math.max(1, Math.round(+r.modeDilation || 1));
        m = Math.max(m, (9 - 1) * d);
      });
      return m;
    })();

    kernelRows.forEach(r => {
      const row = document.createElement('div');
      row.className = 'krow';
      row.dataset.kernelId = r.kernelId;

      const left = document.createElement('div');
      left.className = 'krowLeft';

      const meta = document.createElement('div');
      meta.className = 'krowMeta';

      const idWrap = document.createElement('div');
      idWrap.className = 'krowIdWrap';

      const kid = document.createElement('div');
      kid.className = 'krowId';
      kid.textContent = String(r.kernelId || 'K?');

      const desc = document.createElement('div');
      desc.className = 'krowDesc';
      desc.textContent = r.desc || '';

      idWrap.appendChild(kid);
      idWrap.appendChild(desc);

      const cnt = document.createElement('div');
      cnt.className = 'krowCount';
      cnt.textContent = `×${r.count}`;

      meta.appendChild(idWrap);
      meta.appendChild(cnt);

      const ksvg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
      ksvg.classList.add('krowKernelSvg');

      left.appendChild(meta);
      left.appendChild(ksvg);

      const right = document.createElement('div');
      right.className = 'krowRight';

      // Threshold histogram (global bins)
      // Rendered in the exact same style as the dilation chart below, but using numeric ranges (bins).
      const thrBox = document.createElement('div');
      thrBox.className = 'kdist';
      const thrTitle = document.createElement('div');
      thrTitle.className = 'kdistTitle';
      thrTitle.textContent = 'Threshold Frequencies';
      thrBox.appendChild(thrTitle);

      const thrVals = (r.thrVals || []).filter(Number.isFinite);
      const thrBins = thrBinGen(thrVals);
      // Ensure same bin count as template
      const thrCounts = (thrBinsTemplate || []).map((b, i) => (thrBins[i] ? thrBins[i].length : 0));

      const thrEntries = thrCounts.map((c, i) => {
        const b = thrBinsTemplate[i];
        const x0 = (b && b.x0 != null) ? b.x0 : (thrBins[i] ? thrBins[i].x0 : null);
        const x1 = (b && b.x1 != null) ? b.x1 : (thrBins[i] ? thrBins[i].x1 : null);
        const label = (x0 != null && x1 != null) ? `${fmtSci(x0)}..${fmtSci(x1)}` : `bin ${i + 1}`;
        return { i, c: +c, x0, x1, label };
      });

      if (!thrEntries.length) {
        const p = document.createElement('div');
        p.className = 'muted';
        p.style.fontSize = '11px';
        p.textContent = 'No threshold data.';
        thrBox.appendChild(p);
      } else {
        const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        svg.classList.add('kdistSvg');
        svg.style.display = 'block';
        svg.style.width = '100%';
        svg.style.marginTop = '4px';

        // Match the dilation chart layout and styling (horizontal bars + shared baseline).
        const W = 240;
        const rowH = 26;
        const barH = 20;
        const topPad = 6;
        const bottomPad = 6;

        // Threshold labels are ranges, so we need a wider label area than for dilations.
        const leftLabelW = 92;
        const axisPad = 8;
        const axisX = leftLabelW + axisPad;

        const rightPad = 40;
        const barW = Math.max(10, W - axisX - rightPad);

        const H = topPad + bottomPad + rowH * thrEntries.length;


        svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
        svg.setAttribute('preserveAspectRatio', 'xMinYMin meet');

        const s = d3.select(svg);
        const g = s.append('g').attr('transform', `translate(0,${topPad})`);

        const x = d3.scaleLinear()
          .domain([0, Math.max(1, maxThrBinCount)])
          .range([0, barW - 15]);

        const thrMid = Math.max(0, Math.round(maxThrBinCount / 2));
        const tickVals = [thrMid, maxThrBinCount].filter(v => Number.isFinite(v) && v > 0);
        const y1 = -2;
        const y2 = rowH * thrEntries.length - (rowH - barH) / 2 + 2;


        // Grid lines (vertical, aligned to the right edge of the bars)
        // g.selectAll('line.thrGrid').data(tickVals).enter().append('line')
        //   .attr('class', 'thrGrid')
        //   .attr('x1', v => axisX + x(v) + 12 + extraForLabel)
        //   .attr('x2', v => axisX + x(v) + 12 + extraForLabel)
        //   .attr('y1', y1)
        //   .attr('y2', y2)
        //   .attr('stroke', 'rgba(17,24,39,.10)');

        // Baseline
        g.append('line')
          .attr('x1', axisX + 12 + extraForLabel)
          .attr('x2', axisX + 12 + extraForLabel)
          .attr('y1', y1)
          .attr('y2', y2)
          .attr('stroke', 'rgba(17,24,39,.85)')
          .attr('stroke-width', 2);

        const rowsG = g.selectAll('g.thrRow')
          .data(thrEntries)
          .enter().append('g')
          .attr('class', 'thrRow')
          .attr('transform', (d, i) => `translate(0,${i * rowH})`);



        rowsG.append('text')
          .attr('x', axisX + 10 + extraForLabel)
          .attr('y', rowH / 2 + 8)
          .attr('text-anchor', 'end')
          .attr('font-family', cssVar('--mono', 'monospace'))
          .attr('font-size', 20)
          .attr('font-weight', 'bold')
          .attr('fill', 'rgba(17,24,39,.72)')
          .text(d => {
            const fmtShort = (v) => {
              if (v == null || !Number.isFinite(v)) return '—';
              // const ax = Math.abs(v);
              // if (ax > 0 && ax < 1e-3) return v.toExponential(1);
              // if (ax >= 1e4) return v.toExponential(1);
              // if (ax >= 1) return v.toFixed(2);
              // if (ax >= 0.01) return v.toFixed(3);
              // return v.toFixed(4);
              return v.toFixed(2);
            };
            if (d.x0 != null && d.x1 != null) return `${fmtShort(d.x0)}  ${fmtShort(d.x1)}`;
            return d.label;
          })
          .append('title')
          .text(d => {
            const fmtShort = (v) => {
              if (v == null || !Number.isFinite(v)) return '—';
              const ax = Math.abs(v);
              if (ax > 0 && ax < 1e-3) return v.toExponential(1);
              if (ax >= 1e4) return v.toExponential(1);
              if (ax >= 1) return v.toFixed(2);
              if (ax >= 0.01) return v.toFixed(3);
              return v.toFixed(4);
            };
            const label = (d.x0 != null && d.x1 != null) ? `${fmtShort(d.x0)} .. ${fmtShort(d.x1)}` : d.label;
            return `range: ${label}`;
          });

        // Bars (same style as dilation bars, but horizontal and using a shared scale across kernels)
        rowsG.append('rect')
          .attr('x', axisX + 12 + extraForLabel)
          .attr('y', (rowH - barH) / 2)
          .attr('height', barH)
          .attr('rx', 3)
          .attr('ry', 3)
          .attr('width', d => {
            const w = x(d.c);
            return (d.c > 0) ? Math.max(2, w) : 0;
          })
          .attr('fill', d => barFillByCount(d.c, Math.max(1, maxThrBinCount)))
          .attr('stroke', 'rgba(17,24,39,.10)');

        // Count labels (same style as dilation counts, but positioned on the right side of the bars)
        rowsG.append('text')
          .attr('y', rowH / 2 + 6)
          .attr('font-family', cssVar('--mono', 'monospace'))
          .attr('font-size', 20)
          .attr('fill', 'rgba(17,24,39,.72)')
          .attr('text-anchor', 'start')
          .attr('x', d => {
            const w = Math.max(2, x(d.c));
            const at = axisX + w;
            return 10 + ((at > W - rightPad - 10) ? (at - 4) : (at + 4)) + extraForLabel;
          })
          .text(d => d.c > 0 ? String(Math.round(d.c)) : '');

        thrBox.appendChild(svg);
      }

      // Dilation frequencies (horizontal bar chart; shared scale across kernels)
      const dilBox = document.createElement('div');
      dilBox.className = 'kdist';
      const dilTitle = document.createElement('div');
      dilTitle.className = 'kdistTitle';
      dilTitle.textContent = 'Dilation Frequencies';
      dilBox.appendChild(dilTitle);

      const dilMid = Math.max(0, Math.round(maxDilCount / 2));

      const dilEntries = Array.from((r.dilCounts || new Map()).entries())
        .map(([k, v]) => ({ d: +k, c: +v }))
        .filter(e => Number.isFinite(e.d) && Number.isFinite(e.c) && e.d >= 1 && e.c >= 0)
        .sort((a, b) => d3.ascending(a.d, b.d));

      if (!dilEntries.length) {
        const p = document.createElement('div');
        p.className = 'muted';
        p.style.fontSize = '11px';
        p.textContent = 'No dilation data.';
        dilBox.appendChild(p);
      } else {
        const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        svg.classList.add('kdistSvg');
        svg.style.display = 'block';
        svg.style.width = '100%';
        svg.style.marginTop = '4px';

        // Layout tuned to match a standard horizontal bar chart:
        // dilation labels on the left, a common vertical axis line, and bars starting from that axis.
        const W = 240;
        const rowH = 26;
        const barH = 20;
        const topPad = 6;
        const bottomPad = 6;

        const leftLabelW = 22;   // room for dilation labels (e.g., 256)
        const axisPad = 8;       // gap between labels and the axis line
        const axisX = leftLabelW + axisPad;

        const rightPad = 40;     // room for count labels
        const barW = W - axisX - rightPad;

        const H = topPad + bottomPad + rowH * dilEntries.length;

        svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
        svg.setAttribute('preserveAspectRatio', 'xMinYMin meet');
        svg.style.height = `${H}px`;

        const s = d3.select(svg);
        const g = s.append('g').attr('transform', `translate(0,${topPad})`);

        const x = d3.scaleLinear()
          .domain([0, Math.max(1, maxDilCount)])
          .range([0, barW - 15]);

        // Optional shared grid lines for easier comparison
        const tickVals = [dilMid, maxDilCount].filter(v => Number.isFinite(v) && v > 0);

        const y1 = -2;
        const y2 = rowH * dilEntries.length - (rowH - barH) / 2 + 2;

        g.selectAll('line.dilGrid').data(tickVals).enter().append('line')
          .attr('class', 'dilGrid')
          .attr('x1', v => axisX + x(v) + 12)
          .attr('x2', v => axisX + x(v) + 12)
          .attr('y1', y1)
          .attr('y2', y2)
          .attr('stroke', 'rgba(17,24,39,.10)');

        // Common vertical axis line (baseline)
        g.append('line')
          .attr('x1', axisX + 12)
          .attr('x2', axisX + 12)
          .attr('y1', y1)
          .attr('y2', y2)
          .attr('stroke', 'rgba(17,24,39,.85)')
          .attr('stroke-width', 2);

        const rowsG = g.selectAll('g.dilRow')
          .data(dilEntries)
          .enter().append('g')
          .attr('class', 'dilRow')
          .attr('transform', (d, i) => `translate(0,${i * rowH})`);

        // Dilation labels
        rowsG.append('text')
          .attr('x', axisX + 10)
          .attr('y', rowH / 2 + 8)
          .attr('text-anchor', 'end')
          .attr('font-family', cssVar('--mono', 'monospace'))
          .attr('font-size', 20)
          .attr('font-weight', 'bold')
          .attr('fill', 'rgba(17,24,39,.72)')
          .text(d => String(Math.round(d.d)));

        // Bars
        rowsG.append('rect')
          .attr('x', axisX + 12)
          .attr('y', (rowH - barH) / 2)
          .attr('height', barH)
          .attr('rx', 3)
          .attr('ry', 3)
          .attr('width', d => {
            const w = x(d.c);
            return (d.c > 0) ? Math.max(2, w) : 0;
          })
          .attr('fill', d => barFillByCount(d.c, Math.max(1, maxDilCount)))
          .attr('stroke', 'rgba(17,24,39,.10)');

        // Frequency labels (at bar end)
        rowsG.append('text')
          .attr('y', rowH / 2 + 6)
          .attr('font-family', cssVar('--mono', 'monospace'))
          .attr('font-size', 20)
          .attr('fill', 'rgba(17,24,39,.72)')
          .attr('text-anchor', 'start')
          .attr('x', d => {
            const w = Math.max(2, x(d.c));
            const at = axisX + w;
            return 20 + ((at > W - rightPad - 10) ? (at - 4) : (at + 4));
          })
          .text(d => String(Math.round(d.c)));

        dilBox.appendChild(svg);
      }
      right.appendChild(thrBox);
      right.appendChild(dilBox);

      row.appendChild(left);
      row.appendChild(right);

      host.appendChild(row);

      // draw kernel after insert
      drawKernelMiniShared(ksvg, r.weights, r.modeDilation, showDilated, sharedXMax);
    });

    applySummaryHighlights();
  }

  async function refreshSummaryModal(token) {
    if (!summaryUI.open) return;
    if (token != null && token !== summaryUI.refreshToken) return;

    updateSummaryScopeSelectOptions();

    const scope = summaryScopeFromUI();
    updateSummaryScopeControlsVisibility(scope);

    const cfg = getCurrentCfgForSummary();
    if (!cfg) {
      modalSetStatus('No configuration available.');
      modalSetSubtitle('—');
      renderSummaryMatrix([], 1);
      renderKernelStrip([], d3.bin(), [], 1, 1, false);
      return;
    }

    const sig = cfgSig(cfg);
    const activeId = (peState && peState.instanceId != null) ? peState.instanceId : null;

    // TopK slider behavior
    let topK = el.peSummaryTopK ? Math.max(1, parseInt(el.peSummaryTopK.value, 10) || 15) : 15;
    if (el.peSummaryTopKLabel) el.peSummaryTopKLabel.textContent = String(topK);

    const showDilated = !!(el.peSummaryShowDilated && el.peSummaryShowDilated.checked);
    const kernelSort = el.peSummaryKernelSort ? String(el.peSummaryKernelSort.value || 'freq_desc') : 'freq_desc';

    let alphaMaps = [];
    let instIds = [];
    let subtitle = '';

    if (scope === 'active') {
      if (activeId != null) {
        alphaMaps = [{ id: activeId, map: activeAlphaMap() }];
        instIds = [activeId];
        subtitle = `active instance ${activeId}`;
      } else {
        subtitle = 'no instance';
      }
    } else if (scope === 'tabs') {
      const tabs = (tabState && tabState.tabs) ? tabState.tabs : [];
      const matching = tabs.filter(t => t && t.cfg && cfgSig(t.cfg) === sig && t.instanceId != null);
      instIds = matching.map(t => t.instanceId);
      subtitle = `${instIds.length} open tab(s)`;

      if (instIds.length) {
        modalSetStatus(`Loading α for ${instIds.length} tab(s)…`);
        const maps = [];
        for (const id of instIds) {
          try {
            const m = await loadAlphaMapForInstance(cfg, id);
            maps.push({ id, map: m });
          } catch (e) {
            // skip missing
          }
        }
        alphaMaps = maps;
      }
    } else if (scope === 'range') {
      // Must be loaded explicitly
      if (state.summaryCache.rangeSig === sig && state.summaryCache.rangeAlphaMaps && state.summaryCache.rangeAlphaMaps.length) {
        alphaMaps = state.summaryCache.rangeAlphaMaps.slice();
        instIds = alphaMaps.map(p => p.id);
        subtitle = `range: ${instIds.length} instance(s)`;
      } else {
        subtitle = 'range: not loaded';
      }
    } else if (scope === 'list') {
      if (state.summaryCache.listSig === sig && state.summaryCache.listAlphaMaps && state.summaryCache.listAlphaMaps.length) {
        alphaMaps = state.summaryCache.listAlphaMaps.slice();
        instIds = alphaMaps.map(p => p.id);
        subtitle = `list: ${instIds.length} instance(s)`;
      } else {
        subtitle = 'list: not loaded';
      }
    } else if (scope === 'classflip') {
      if (activeId == null) {
        subtitle = 'ClassFlip: no instance';
      } else {
        const cf = await ensureClassFlipFeatureIds(cfg, activeId);
        if (!cf || !cf.featureIds || !cf.featureIds.length) {
          subtitle = 'ClassFlip: file not found';
        } else {
          subtitle = `ClassFlip Top${cf.topN} (instance ${activeId})`;
          // force topK slider to reflect TopN and disable
          if (el.peSummaryTopK) {
            el.peSummaryTopK.value = String(Math.max(1, cf.topN));
            el.peSummaryTopK.disabled = true;
            topK = Math.max(1, cf.topN);
          }
          if (el.peSummaryTopKLabel) el.peSummaryTopKLabel.textContent = String(topK);

          alphaMaps = [{ id: activeId, map: activeAlphaMap() }];
          instIds = [activeId];
        }
      }
    }

    // Re-enable slider if not classflip
    if (scope !== 'classflip' && el.peSummaryTopK) el.peSummaryTopK.disabled = false;

    modalSetSubtitle(subtitle);

    if (!state.features || !state.features.length) {
      modalSetStatus('Load an instance first.');
      renderSummaryMatrix([], 1);
      renderKernelStrip([], d3.bin(), [], 1, 1, showDilated);
      return;
    }

    if (!alphaMaps.length) {
      modalSetStatus((scope === 'range' || scope === 'list') ? 'Click Load to fetch data.' : 'No α data in this scope.');
      renderSummaryMatrix([], 1);
      renderKernelStrip([], d3.bin(), [], 1, 1, showDilated);
      return;
    }

    // Select features
    let pickedFidx = null;

    if (scope === 'classflip' && activeId != null) {
      const cf = await ensureClassFlipFeatureIds(cfg, activeId);
      pickedFidx = (cf && cf.featureIds) ? cf.featureIds.slice(0, topK) : [];
    } else {
      const rows = [];
      for (const d of state.features) {
        const fidx = +d.fidx;
        if (!Number.isFinite(fidx)) continue;
        const vals = [];
        alphaMaps.forEach(p => {
          if (p.map && p.map.has(fidx)) vals.push(+p.map.get(fidx));
        });
        if (!vals.length) continue;
        const mA = mean(vals);
        const mAbs = mean(vals.map(v => Math.abs(v)));
        rows.push({ fidx, meanAlpha: mA, meanAbs: mAbs, n: vals.length });
      }
      rows.sort((a, b) => d3.descending(a.meanAbs, b.meanAbs));
      pickedFidx = rows.slice(0, Math.min(topK, rows.length)).map(r => r.fidx);
    }

    // Build picked rows with attributes
    const pickedRows = [];
    const absA = [];
    pickedFidx.forEach(fidx => {
      const vals = [];
      alphaMaps.forEach(p => {
        if (p.map && p.map.has(fidx)) vals.push(+p.map.get(fidx));
      });
      if (!vals.length) return;
      const meanAlpha = mean(vals);
      const meanAbs = mean(vals.map(v => Math.abs(v)));
      const kernelId = kernelIdForFeature(fidx);
      const dilation = peDilationForFeature(fidx);
      const bias = peBiasForFeature(fidx);
      pickedRows.push({
        fidx,
        meanAlpha,
        meanAbs,
        n: vals.length,
        kernelId,
        dilation,
        bias,
        desc: kernelDescForId(kernelId)
      });
      absA.push(Math.abs(meanAlpha));
    });

    const maxAbsAlpha = Math.max(1e-9, d3.max(absA) || 1e-9);

    // Bias stats overall
    const biasVals = pickedRows.map(d => d.bias).filter(Number.isFinite);
    const bMin = d3.min(biasVals), bMax = d3.max(biasVals), bMean = d3.mean(biasVals);
    if (el.peSummaryBiasStats) {
      el.peSummaryBiasStats.textContent = biasVals.length
        ? `thr in selection: min=${fmt(bMin)} · mean=${fmt(bMean)} · max=${fmt(bMax)}`
        : '';
    }

    // Render matrix (sortable)
    summaryUI.lastPickedRows = pickedRows.slice();
    summaryUI.lastMaxAbsAlpha = maxAbsAlpha;
    renderSummaryMatrix(sortSummaryRows(pickedRows), maxAbsAlpha);

    // Group by kernel
    const byKernel = new Map();
    pickedRows.forEach(r => {
      const k = r.kernelId || 'K?';
      if (!byKernel.has(k)) byKernel.set(k, []);
      byKernel.get(k).push(r);
    });

    const allThr = pickedRows.map(r => r.bias).filter(Number.isFinite);
    let thrExtent = d3.extent(allThr);
    if (!thrExtent[0] && thrExtent[0] !== 0) thrExtent = [-1, 1];
    if (thrExtent[0] === thrExtent[1]) {
      const v = thrExtent[0];
      thrExtent = [v - 1, v + 1];
    }

    const thrBinCount = 6; // global bins (option 1)
    const thrBinGen = d3.bin().domain(thrExtent).thresholds(thrBinCount);
    const thrBinsTemplate = thrBinGen(allThr);

    // Precompute global maxima for bar scaling
    let maxThrBinCount = 1;
    let maxDilCount = 1;

    const kernelRows = Array.from(byKernel.entries()).map(([kernelId, rows]) => {
      const dilCounts = new Map();
      rows.forEach(rr => {
        const d = +rr.dilation;
        if (!Number.isFinite(d)) return;
        dilCounts.set(d, (dilCounts.get(d) || 0) + 1);
        maxDilCount = Math.max(maxDilCount, dilCounts.get(d));
      });

      const thrVals = rows.map(rr => +rr.bias).filter(Number.isFinite);
      const bins = thrBinGen(thrVals);
      bins.forEach(b => { maxThrBinCount = Math.max(maxThrBinCount, b.length); });

      return {
        kernelId,
        desc: kernelDescForId(kernelId),
        count: rows.length,
        weights: weightsForKernelId(kernelId),
        modeDilation: computeModeDilation(rows),
        thrVals,
        dilCounts
      };
    });

    kernelRows.sort((a, b) => {
      const c = (kernelSort === 'freq_asc') ? d3.ascending(a.count, b.count) : d3.descending(a.count, b.count);
      return c || String(a.kernelId).localeCompare(String(b.kernelId));
    });

    // Render kernel panel
    renderKernelStrip(kernelRows, thrBinGen, thrBinsTemplate, maxThrBinCount, maxDilCount, showDilated);

    modalSetStatus(`Computed on ${instIds.length} instance(s), selection=${pickedRows.length}.`);
  }

  function refreshSummaryModalSafe() {
    const token = ++summaryUI.refreshToken;
    return refreshSummaryModal(token);
  }

  async function openSummaryModal() {
    if (!el.peSummaryOverlay || !el.peSummaryModal) return;
    el.peSummaryOverlay.hidden = false;
    summaryUI.open = true;

    // Center if first open
    try {
      if (!el.peSummaryModal.style.left && !el.peSummaryModal.style.top) {
        const mw = el.peSummaryModal.offsetWidth;
        const mh = el.peSummaryModal.offsetHeight;
        const left = Math.max(8, (window.innerWidth - mw) / 2);
        const top = Math.max(8, (window.innerHeight - mh) / 2);
        el.peSummaryModal.style.left = `${left}px`;
        el.peSummaryModal.style.top = `${top}px`;
      }
    } catch (_) { }

    // Range defaults based on current instance
    try {
      const cur = (peState && peState.instanceId != null) ? peState.instanceId : 0;
      if (el.peSummaryStartId) el.peSummaryStartId.value = String(Math.max(0, cur));
      if (el.peSummaryEndId) el.peSummaryEndId.value = String(Math.max(0, cur));
    } catch (e) { }

    // Slider max based on feature count
    try {
      const n = (state.features && state.features.length) ? state.features.length : 0;
      const maxK = Math.max(1, Math.min(200, n || 50));
      if (el.peSummaryTopK) {
        el.peSummaryTopK.max = String(maxK);
        const v0 = Math.min(maxK, Math.max(1, parseInt(el.peSummaryTopK.value, 10) || 15));
        el.peSummaryTopK.value = String(v0);
        if (el.peSummaryTopKLabel) el.peSummaryTopKLabel.textContent = String(v0);
      }
    } catch (e) { }

    updateSummaryScopeSelectOptions();
    updateSummaryScopeControlsVisibility(summaryScopeFromUI());
    modalSetStatus('Ready.');
    await refreshSummaryModalSafe();
  }

  function closeSummaryModal() {
    if (!el.peSummaryOverlay) return;
    el.peSummaryOverlay.hidden = true;
    summaryUI.open = false;
    summaryUI.dragging = false;
  }

  function initSummaryModal() {
    if (!el.peSummaryBtn || !el.peSummaryOverlay || !el.peSummaryModal) return;

    // Safety: ensure the overlay starts closed (some browsers/extensions may restore DOM state)
    el.peSummaryOverlay.hidden = true;
    summaryUI.open = false;
    summaryUI.dragging = false;

    el.peSummaryBtn.addEventListener('click', async () => {
      if (!state.features || !state.features.length) return;
      await openSummaryModal();
    });

    if (el.peSummaryCloseBtn) el.peSummaryCloseBtn.addEventListener('click', closeSummaryModal);
    if (el.peSummaryRefreshBtn) el.peSummaryRefreshBtn.addEventListener('click', refreshSummaryModalSafe);

    if (el.peSummaryOverlay) {
      el.peSummaryOverlay.addEventListener('mousedown', (ev) => {
        if (ev.target === el.peSummaryOverlay) closeSummaryModal();
      });
    }

    // Dragging
    if (el.peSummaryHeader) {
      el.peSummaryHeader.addEventListener('mousedown', (ev) => {
        if (ev.button !== 0) return;
        summaryUI.dragging = true;
        const r = el.peSummaryModal.getBoundingClientRect();
        summaryUI.dragOffX = ev.clientX - r.left;
        summaryUI.dragOffY = ev.clientY - r.top;
        ev.preventDefault();
      });
    }

    window.addEventListener('mousemove', (ev) => {
      if (!summaryUI.dragging || !summaryUI.open) return;
      const mw = el.peSummaryModal.offsetWidth;
      const mh = el.peSummaryModal.offsetHeight;
      const vw = window.innerWidth;
      const vh = window.innerHeight;
      let left = ev.clientX - summaryUI.dragOffX;
      let top = ev.clientY - summaryUI.dragOffY;
      left = clamp(left, 8, Math.max(8, vw - mw - 8));
      top = clamp(top, 8, Math.max(8, vh - mh - 8));
      el.peSummaryModal.style.left = `${left}px`;
      el.peSummaryModal.style.top = `${top}px`;
    });

    window.addEventListener('mouseup', () => {
      summaryUI.dragging = false;
    });

    // UI controls
    if (el.peSummaryScopeSelect) {
      el.peSummaryScopeSelect.addEventListener('change', () => {
        updateSummaryScopeSelectOptions();
        updateSummaryScopeControlsVisibility(summaryScopeFromUI());
        refreshSummaryModalSafe();
      });
    }

    if (el.peSummaryTopK) el.peSummaryTopK.addEventListener('input', () => refreshSummaryModalSafe());
    if (el.peSummaryShowDilated) el.peSummaryShowDilated.addEventListener('change', () => refreshSummaryModalSafe());
    if (el.peSummaryKernelSort) el.peSummaryKernelSort.addEventListener('change', () => refreshSummaryModalSafe());

    if (el.peSummaryLoadRangeBtn) {
      el.peSummaryLoadRangeBtn.addEventListener('click', async () => {
        const cfg = getCurrentCfgForSummary();
        if (!cfg) { modalSetStatus('No configuration available.'); return; }
        const s = normalizeInt(el.peSummaryStartId ? el.peSummaryStartId.value : 0, 0);
        const e = normalizeInt(el.peSummaryEndId ? el.peSummaryEndId.value : 0, 0);
        const a = Math.min(s, e), b = Math.max(s, e);
        const ids = [];
        for (let i = a; i <= b; i++) ids.push(i);
        const res = await loadAlphaMapsForIds(cfg, ids);
        state.summaryCache.rangeSig = res.sig;
        state.summaryCache.rangeIds = res.ids;
        state.summaryCache.rangeAlphaMaps = res.alphaMaps;
        state.summaryCache.rangeErrors = res.errors;
        refreshSummaryModalSafe();
      });
    }

    if (el.peSummaryLoadListBtn) {
      el.peSummaryLoadListBtn.addEventListener('click', async () => {
        const cfg = getCurrentCfgForSummary();
        if (!cfg) { modalSetStatus('No configuration available.'); return; }
        const spec = el.peSummaryInstanceList ? String(el.peSummaryInstanceList.value || '') : '';
        const ids = parseInstanceListSpec(spec);
        if (!ids.length) { modalSetStatus('No valid instance ids.'); return; }
        const res = await loadAlphaMapsForIds(cfg, ids);
        state.summaryCache.listSig = res.sig;
        state.summaryCache.listIds = res.ids;
        state.summaryCache.listAlphaMaps = res.alphaMaps;
        state.summaryCache.listErrors = res.errors;
        refreshSummaryModalSafe();
      });
    }

    // Disable by default (enabled when instance is loaded)
    if (el.peSummaryBtn) el.peSummaryBtn.disabled = (!state.features || state.features.length === 0);
  }


  initReferenceToggle();
  initEmbeddingSortControls();
  initSummaryModal();
  initAdvancedPopover();
  renderKernelPreview();
  renderKernelList();
  clearToEmptyState();
  forceHideTopBusy();
  peInitUI();
  forceHideTopBusy();
  syncRetropropButtonState();
  let resizeRaf = null;
  window.addEventListener('resize', () => {
    if (resizeRaf) cancelAnimationFrame(resizeRaf);
    resizeRaf = requestAnimationFrame(() => {
      resizeRaf = null;
      renderAll();
    });
  });

  // ---- Resizable panels with drag handles ----
  (function initResizeHandles() {
    const handles = document.querySelectorAll('.resize-handle');
    const MIN_H = 80;
    const MAX_RATIO = 0.8;
    const panelVarMap = {
      'timeSeriesPanelContent': '--panelH-ts',
      'embeddingsPanelContent': '--panelH-emb',
      'convPanelContent': '--panelH-conv'
    };
    const renderMap = {
      'timeSeriesPanelContent': () => { renderSeries(); if (seriesZoom) attachZoom(); },
      'embeddingsPanelContent': () => { renderFeatureCharts(); },
      'convPanelContent': () => { renderConv(); redrawConvFromCacheOnly(); }
    };

    handles.forEach(handle => {
      let startY = 0, startH = 0, aboveEl = null, aboveId = '';
      let dragRaf = null;

      function onPointerDown(e) {
        aboveId = handle.dataset.above || '';
        aboveEl = document.getElementById(aboveId);
        if (!aboveEl) return;
        startY = e.clientY;
        startH = aboveEl.offsetHeight;
        document.addEventListener('pointermove', onPointerMove);
        document.addEventListener('pointerup', onPointerUp);
        document.body.style.cursor = 'ns-resize';
        document.body.style.userSelect = 'none';
        e.preventDefault();
      }

      function onPointerMove(e) {
        if (!aboveEl) return;
        if (dragRaf) cancelAnimationFrame(dragRaf);
        dragRaf = requestAnimationFrame(() => {
          dragRaf = null;
          const delta = e.clientY - startY;
          const maxH = window.innerHeight * MAX_RATIO;
          const newH = Math.max(MIN_H, Math.min(maxH, startH + delta));
          const cssVar = panelVarMap[aboveId];
          if (cssVar) {
            document.documentElement.style.setProperty(cssVar, `${newH}px`);
          }
          const renderFn = renderMap[aboveId];
          if (renderFn) renderFn();
        });
      }

      function onPointerUp() {
        document.removeEventListener('pointermove', onPointerMove);
        document.removeEventListener('pointerup', onPointerUp);
        document.body.style.cursor = '';
        document.body.style.userSelect = '';
        if (dragRaf) { cancelAnimationFrame(dragRaf); dragRaf = null; }
        // Final render
        const renderFn = renderMap[aboveId];
        if (renderFn) renderFn();
      }

      handle.addEventListener('pointerdown', onPointerDown);
    });
  })();
})();


function saveWebPageAsSVGForDownload() {
  // Collect all SVG elements from the page
  const svgElements = document.querySelectorAll('svg');
  if (!svgElements.length) {
    console.warn('No SVG elements found on the page');
    return;
  }

  // Create a new SVG container
  const container = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  container.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
  container.setAttribute('width', window.innerWidth);
  container.setAttribute('height', window.innerHeight);
  container.setAttribute('viewBox', `0 0 ${window.innerWidth} ${window.innerHeight}`);

  // Add a white background
  const bg = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
  bg.setAttribute('width', '100%');
  bg.setAttribute('height', '100%');
  bg.setAttribute('fill', 'white');
  container.appendChild(bg);

  // Clone and append each SVG element
  svgElements.forEach(svg => {
    const rect = svg.getBoundingClientRect();
    const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    g.setAttribute('transform', `translate(${rect.left},${rect.top})`);

    // Clone children (not the SVG itself to avoid nesting issues)
    Array.from(svg.childNodes).forEach(child => {
      g.appendChild(child.cloneNode(true));
    });

    container.appendChild(g);
  });

  // Serialize and download
  const svgString = new XMLSerializer().serializeToString(container);
  const blob = new Blob([svgString], { type: 'image/svg+xml' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'webpage.svg';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
