(() => {
  function syncHeaderHeight() {
    const h = document.querySelector('header')?.offsetHeight || 0;
    document.documentElement.style.setProperty('--headerH', `${h}px`);
    if (document.body) document.body.style.paddingTop = `${h}px`;
  }
  syncHeaderHeight();
  window.addEventListener('resize', syncHeaderHeight);
  const _hdr = document.querySelector('header');
  if (_hdr && 'ResizeObserver' in window) {
    new ResizeObserver(syncHeaderHeight).observe(_hdr);
  }

  const PORT = 3000;
  const inProduction = !(location.hostname === "localhost" || location.hostname === "127.0.0.1");
  const API_ORIGIN = inProduction ? "" : ((location.port === String(PORT)) ? "" : `${location.protocol}//${location.hostname}:${PORT}`);

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

  const state = {
    kernels: generateMiniRocketKernels(),
    selectedId: 'K00',
    series: [],
    referenceSeries: [],
    referencePath: '',
    dilation: 1,
    align: 0,
    threshold: 0,
    zoomTransform: d3.zoomIdentity,
    featureZoomTransform: d3.zoomIdentity,
    isFeatureZooming: false,
    features: [],
    selectedFeatureIdx: null,
    hoveredFeatureIdx: null,
    footprint: null,
    sweepActive: false,
    sweepTimer: null,
    animateTapsNext: false,
    convLayout: null,
    alignLocked: false,
    isZooming: false,
    // NEW for perf
    seriesVersion: 0,
    convCache: new Map(),

    // runtime caches for fast alignment & throttling
    sX: null,
    sY: null,
    sInnerW: 0,
    sInnerH: 0,
    __alignRAF: null,
    __pendingAlignPx: 0,
    __alignBounds: null,
    __alignX: null,

  };

  const el = {
    status: document.querySelector('#status'),
    seriesInfo: document.querySelector('#seriesInfo'),
    kernelSearch: document.querySelector('#kernelSearch'),
    kernelList: d3.select('#kernelList'),
    summary: document.querySelector('#summary'),
    selectedKernelId: document.querySelector('#selectedKernelId'),
    selectedKernelDesc: document.querySelector('#selectedKernelDesc'),
    selectedKernelPlot: document.querySelector('#selectedKernelPlot'),
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
    showReference: document.querySelector('#showReference'),
    featureIdx: document.querySelector('#featureIdx'),
    goFeature: document.querySelector('#goFeature'),
    featureInfo: document.querySelector('#featureInfo'),
    featureTooltip: document.querySelector('#featureTooltip'),
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
  const sFootG = sG.append('g'); // RESTORED
  const sBetaG = sG.append('g'); // RESTORED
  const sRefPath = sG.append('path').attr('fill', 'none').attr('stroke', 'rgba(99,102,241,.85)').attr('stroke-width', 1.75).attr('stroke-dasharray', '6 4').attr('opacity', 0);
  const sPath = sG.append('path').attr('fill', 'none').attr('stroke', 'rgba(37,99,235,.85)').attr('stroke-width', 2);
  const sTapsG = sG.append('g');
  const sHandleG = sG.append('g');

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

  function mergeFeatureSignals(features, alphaMap, embMap) {
    return (features || []).map(d => ({
      ...d,
      alpha: (alphaMap && alphaMap.has(d.fidx)) ? alphaMap.get(d.fidx) : (Number.isFinite(d.alpha) ? d.alpha : 0),
      embedding: (embMap && embMap.has(d.fidx)) ? embMap.get(d.fidx) : (Number.isFinite(d.embedding) ? d.embedding : 0)
    }));
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
    state.featureZoomTransform = d3.zoomIdentity;
    if (el.featureIdx) el.featureIdx.max = Math.max(0, state.features.length - 1);
    renderFeatureCharts();
    if (el.featureInfo) el.featureInfo.textContent = '';
    if (el.seriesInfo) el.seriesInfo.textContent = '';
    try {
      d3.select(el.convChart).selectAll('*').remove();
    } catch (e) { }
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
    state.dilation = +d.dilation;
    updateDilationLimits();
    state.selectedId = d.kernel_id;
    fixAlignBounds();
    renderKernelPreview();
    renderKernelList();
    state.threshold = +d.threshold;
    updateThresholdDisplay(state.thresholdRange?.min, state.thresholdRange?.max);
    renderSeriesAndConv();
  }

  function selectFeature(fidx, opts) {
    const d = featureByIdx(fidx);
    if (!d) return;
    stopSweep();
    state.selectedFeatureIdx = d.fidx;
    if (el.featureIdx) el.featureIdx.value = d.fidx;
    updateFeatureHeader(d);
    // Keep the instance details card in sync with the currently selected embedding.
    renderInstanceDetails();
    renderFeatureCharts();
    applyFeatureToKernelExplorer(d);
    if (!(opts && opts.silentTooltip)) hideFeatureTooltip();
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
    // Zoom-based navigation (wheel to zoom, drag to pan), matching the convolution series plot.
    const xBaseFeat = d3.scaleLinear().domain([0, n - 1]).range([0, innerW]);
    if (!state.featureZoomTransform) state.featureZoomTransform = d3.zoomIdentity;
    const xZ = state.featureZoomTransform.rescaleX(xBaseFeat);
    const step = (n > 1) ? (xZ(1) - xZ(0)) : innerW;
    const bandW = Math.max(1, Math.abs(step));

    function idxFromPx(px) {
      const v = xZ.invert(px);
      return clamp(Math.round(v), 0, n - 1);
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
    bars.enter().append('rect')
      .attr('class', 'embedBar')
      .attr('fill', 'rgba(125,169,165,.35)')
      .attr('stroke', 'rgba(125,169,165,.75)')
      .attr('stroke-width', 0.8)
      .merge(bars)
      .attr('x', d => xZ(d.fidx) - barW0 / 2)
      .attr('width', barW0)
      .attr('y', d => Math.min(y(d.embedding), yZero))
      .attr('height', d => Math.max(0, Math.abs(y(d.embedding) - yZero)));
    bars.exit().remove();
    fAxes.x.attr('transform', `translate(0,${innerH})`);
    fAxes.y.attr('transform', 'translate(0,0)');
    fAxes.x.call(d3.axisBottom(xZ).ticks(6).tickFormat(d3.format('d')).tickSizeOuter(0));
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
    rects.enter().append('rect').attr('class', 'cell').attr('y', 0).attr('height', stripH)
      .merge(rects)
      .attr('x', d => xZ(d.fidx) - bandW / 2)
      .attr('width', bandW + 0.6)
      .attr('fill', d => {
        const v = d.alpha;
        if ((minA || 0) < 0 && (maxA || 0) > 0) return color(v);
        return color(Math.abs(v));
      });
    rects.exit().remove();
    const y0 = 0,
      y1 = topH + gap + stripH;
    if (state.selectedFeatureIdx != null) {
      const sx = xZ(state.selectedFeatureIdx);
      fSel.attr('x1', sx).attr('x2', sx).attr('y1', y0).attr('y2', y1).attr('opacity', 1);
    } else {
      fSel.attr('opacity', 0);
    }
    if (state.hoveredFeatureIdx != null) {
      const hx = xZ(state.hoveredFeatureIdx);
      fHover.attr('x1', hx).attr('x2', hx).attr('y1', y0).attr('y2', y1).attr('opacity', 1);
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
        const idx = idxFromPx(px);
        const prev = state.__featureHoverIdx;
        state.hoveredFeatureIdx = idx;
        state.hoveredFeaturePx = px;
        if (prev !== idx) {
          state.__featureHoverIdx = idx;
          scheduleFeatureOverviewRender();
        }
        showFeatureTooltip(ev.clientX, ev.clientY, featureByIdx(idx));
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
        const idx = idxFromPx(px);
        selectFeature(idx);
      });


  }

  function getKernel() {
    return state.kernels.find(k => k.id === state.selectedId) || state.kernels[0];
  }

  function setSelected(id) {
    state.selectedId = id;
    fixAlignBounds();
    renderKernelPreview();
    renderKernelList();
    renderSeriesAndConv();
    closeDropdown();
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
    const bboxW = svgNode.getBoundingClientRect ? svgNode.getBoundingClientRect().width : 0;
    const w = bboxW > 0 ? bboxW : (svgNode.clientWidth || 320);
    const h = svgNode.clientHeight || 118;
    svg.attr('width', '100%').attr('height', h);
    svg.selectAll('*').remove();
    const m = {
      left: 28,
      right: 10,
      top: 10,
      bottom: 32
    };
    const innerW = Math.max(10, w - m.left - m.right);
    const innerH = Math.max(10, h - m.top - m.bottom);
    const g = svg.append('g').attr('transform', `translate(${m.left},${m.top})`);
    const x = d3.scaleLinear().domain([0, 8]).range([0, innerW]);
    const y = d3.scaleLinear().domain([-1.5, 2.5]).range([innerH, 0]);
    const gx = g.append('g').attr('transform', `translate(0,${innerH})`);
    const gy = g.append('g');
    gx.call(d3.axisBottom(x).ticks(9).tickFormat(d3.format('d')).tickSizeOuter(0));
    gy.call(d3.axisLeft(y).ticks(4).tickFormat(d3.format('d')).tickSizeOuter(0));
    [gx, gy].forEach(ax => {
      ax.selectAll('.domain').attr('stroke', 'rgba(17,24,39,.20)');
      ax.selectAll('.tick line').attr('stroke', 'rgba(17,24,39,.12)');
      ax.selectAll('.tick text').attr('fill', 'rgba(17,24,39,.70)').attr('font-family', 'var(--mono)').attr('font-size', 10);
    });
    const zeroY = y(0);
    g.selectAll('line.stem').data(weights.map((w, i) => ({
      w,
      i
    }))).enter().append('line').attr('class', 'stem').attr('x1', d => x(d.i)).attr('x2', d => x(d.i)).attr('y1', zeroY).attr('y2', d => y(d.w)).attr('stroke', 'rgba(17,24,39,.12)').attr('stroke-width', 1);
    const line = d3.line().x((d, i) => x(i)).y(d => y(d));
    g.append('path').attr('d', line(weights)).attr('fill', 'none').attr('stroke', 'rgba(37,99,235,.85)').attr('stroke-width', 2);
    g.selectAll('circle.dot').data(weights.map((w, i) => ({
      w,
      i
    }))).enter().append('circle').attr('class', 'dot').attr('cx', d => x(d.i)).attr('cy', d => y(d.w)).attr('r', 4.6).attr('fill', d => d.w === 2 ? 'rgba(16,185,129,.85)' : 'rgba(239,68,68,.70)').attr('stroke', 'rgba(17,24,39,.18)').attr('stroke-width', 1).append('title').text(d => `x=${d.i}, w=${d.w}`);
  }

  function renderKernelPreview() {
    if (state.selectedFeatureIdx == null) {
      try {
        d3.select(el.selectedKernelPlot).selectAll('*').remove();
      } catch (e) { }
      return;
    }
    const k = getKernel();
    el.selectedKernelId.textContent = k.id;
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


    const tapLines = sTapsG.selectAll('line.tap').data(taps, d => d.i);
    tapLines.join(
      enter => enter.append('line')
        .attr('class', 'tap')
        .attr('stroke-width', 2)
        .attr('stroke', d => d.padded ? __wColor(d.w, 0.5, 0.5, 0.08) : __wColor(d.w, 0.5, 0.5, 0.18)),
      update => update,
      exit => exit.remove()
    )
      .attr('x1', d => x(d.t)).attr('x2', d => x(d.t))
      .attr('y1', d => y(d.y)).attr('y2', innerH);

    const tapCircles = sTapsG.selectAll('circle.tap').data(taps, d => d.i);
    const tapCirclesEnter = tapCircles.enter()
      .append('circle')
      .attr('class', 'tap')
      .attr('r', 6)
      .attr('stroke-width', 1.25)
      .style('cursor', 'default');
    tapCirclesEnter.append('title');

    tapCirclesEnter.merge(tapCircles)
      .attr('cx', d => x(d.t))
      .attr('cy', d => y(d.y))
      .attr('fill', d => d.padded ? __wColor(d.w, 0.16, 0.14, 0.10) : __wColor(d.w, 0.80, 0.70, 0.45))
      .attr('stroke', d => d.padded ? __wColor(d.w, 0.24, 0.22, 0.16) : __wColor(d.w, 1.0, 0.95, 0.70))
      .select('title')
      .text(d => `tap ${d.i} · w=${d.w}`);

    tapCircles.exit().remove();

    // Handle and knob
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
        state.sigmaFromFile = null;

        peLoadConvForFeature(featureIdx)
          .then(pack => {
            if (!pack) return;
            if (state.__convFromFileKey !== key) return;
            state.convFromFile = pack.resp;
            state.sigmaFromFile = pack.sigma;
            // Re-render once with the loaded file data.
            renderConv();
          })
          .catch(() => { });
      } else if (state.convFromFile && state.sigmaFromFile) {
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
    const xBar = 300;

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
    const alphaLabel = (alpha != null && Number.isFinite(+alpha)) ? `α=${fmt2(+alpha)}` : null;

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

    // const arrow = (delta > 0) ? '↑' : (delta < 0) ? '↓' : '→';
    // g.append('text')
    //   .attr('x', xDelta + 10).attr('y', yMid + 6)
    //   .attr('text-anchor', 'start')
    //   .attr('font-family', cssVar('--mono', 'monospace'))
    //   .attr('font-weight', 900)
    //   .attr('font-size', 22)
    //   .attr('fill', contribColor)
    //   .text(arrow);

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
    const barW = 30;
    const barH = 150;
    const yBarTop = Math.round(top + (H - bottom - top - barH) / 2);
    const yBarBottom = yBarTop + barH;

    // Dotted guide lines: show that the right bar refers to the Δ range
    const xGuide0 = xDelta + tickLen + 4;
    const xGuide1 = xBar - 2;
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
      .attr('x', xBar).attr('y', yBarTop)
      .attr('width', barW).attr('height', barH)
      .attr('fill', bg)
      .attr('stroke', stroke);

    // Horizontal line at 0.5 - reference probability within the bar
    const lineY = yBarTop + (barH * (1 - (0.5 - refP) / deltaAbs));
    g.append('line')
      .attr('x1', xBar).attr('x2', xBar + barW)
      .attr('y1', lineY).attr('y2', lineY)
      .attr('stroke', 'rgba(17,24,39,.40)')
      .attr('stroke-width', 1);

    // Label showing the reference probability value (0.5 marker)
    g.append('text')
      .attr('x', xBar + barW / 2).attr('y', lineY - 4)
      .attr('text-anchor', 'middle')
      .attr('font-family', cssVar('--mono', 'monospace'))
      .attr('font-weight', 700)
      .attr('font-size', 12)
      .attr('fill', 'rgba(17,24,39,.70)')
      .text(fmtProb01(0.5 - refP));


    // Arrow showing the decision boundary (class change line)
    g.append('text')
      .attr('x', xBar + barW + 4).attr('y', lineY + 4)
      .attr('text-anchor', 'start')
      .attr('font-family', cssVar('--mono', 'monospace'))
      .attr('font-weight', 700)
      .attr('font-size', 16)
      .attr('fill', 'rgba(17,24,39,.60)')
      .text('←');

    // Label showing the decision boundary (class change line)
    g.append('text')
      .attr('x', xBar + barW + 16).attr('y', lineY - 2)
      .attr('text-anchor', 'start')
      .attr('font-family', cssVar('--mono', 'monospace'))
      .attr('font-weight', 700)
      .attr('font-size', 14)
      .attr('fill', 'rgba(17,24,39,.60)')
      .text('Class');

    g.append('text')
      .attr('x', xBar + barW + 16).attr('y', lineY + 12)
      .attr('text-anchor', 'start')
      .attr('font-family', cssVar('--mono', 'monospace'))
      .attr('font-weight', 700)
      .attr('font-size', 14)
      .attr('fill', 'rgba(17,24,39,.60)')
      .text('flip');

    const dlineLenght = 70;
    const secondDiffX = xBar + barW + dlineLenght;

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
      .attr('x1', xBar + barW + 2).attr('x2', xBar + barW + dlineLenght)
      .attr('y1', lineY).attr('y2', lineY)
      .attr('stroke', 'rgba(17,24,39,.35)')
      .attr('stroke-width', 1)
      .attr('stroke-dasharray', '4 3')
      .attr('opacity', 0.7);

    // Horizontal dotted line at the base of the rectangle
    g.append('line')
      .attr('x1', xBar + barW + 2).attr('x2', xBar + barW + dlineLenght)
      .attr('y1', yBarBottom).attr('y2', yBarBottom)
      .attr('stroke', 'rgba(17,24,39,.35)')
      .attr('stroke-width', 1)
      .attr('stroke-dasharray', '4 3')
      .attr('opacity', 0.7);


    const startSecondRect = xBar + barW + dlineLenght + 30;



    g.append('rect')
      .attr('x', startSecondRect).attr('y', yBarTop)
      .attr('width', barW).attr('height', barH)
      .attr('fill', bg)
      .attr('stroke', stroke);


    // Line connecting the secon difference to the top of the second rectangle
    g.append('line')
      .attr('x1', xBar + barW + dlineLenght + tickLen / 2).attr('x2', startSecondRect)
      .attr('y1', lineY).attr('y2', yBarTop)
      .attr('stroke', 'rgba(17,24,39,.35)')
      .attr('stroke-width', 1)
      .attr('stroke-dasharray', '4 3')
      .attr('opacity', 0.7);

    // Line connecting the secon difference to the bottom of the second rectangle
    g.append('line')
      .attr('x1', xBar + barW + dlineLenght + tickLen / 2).attr('x2', startSecondRect)
      .attr('y1', yBarBottom).attr('y2', yBarBottom)
      .attr('stroke', 'rgba(17,24,39,.35)')
      .attr('stroke-width', 1)
      .attr('stroke-dasharray', '4 3')
      .attr('opacity', 0.7);

    // Label at the bottom of the second rect showing the decision boundary value
    g.append('text')
      .attr('x', startSecondRect + barW / 2).attr('y', yBarBottom + 16)
      .attr('text-anchor', 'middle')
      .attr('font-family', cssVar('--mono', 'monospace'))
      .attr('font-weight', 900)
      .attr('font-size', 18)
      .attr('fill', contribColor)
      .text('0');

    g.append('text')
      .attr('x', startSecondRect + barW / 2).attr('y', yBarTop - 10)
      .attr('text-anchor', 'middle')
      .attr('font-family', cssVar('--mono', 'monospace'))
      .attr('font-weight', 900)
      .attr('font-size', 18)
      .attr('fill', contribColor)
      .text(fmtProb01(0.5 - refP));


    const fillH = barH * ratio;
    const yFill = yBarTop + (barH - fillH);

    g.append('rect')
      .attr('x', xBar).attr('y', yFill)
      .attr('width', barW).attr('height', fillH)
      .attr('fill', alphaLabelColor);

    g.append('text')
      .attr('x', xBar + barW / 2).attr('y', yBarBottom + 16)
      .attr('text-anchor', 'middle')
      .attr('font-family', cssVar('--mono', 'monospace'))
      .attr('font-weight', 900)
      .attr('font-size', 18)
      .text('0');

    g.append('text')
      .attr('x', xBar + barW / 2).attr('y', yBarTop - 10)
      .attr('text-anchor', 'middle')
      .attr('font-family', cssVar('--mono', 'monospace'))
      .attr('font-weight', 900)
      .attr('font-size', 18)
      .text(deltaLabel);

    if (!alphaLabel) {
      g.append('text')
        .attr('x', xBar + barW / 2).attr('y', yBarTop - 10)
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
      .attr('x', xBar + barW + 4).attr('y', yAlpha + 7)
      .attr('text-anchor', 'start')
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
    if (el.featureInfo) el.featureInfo.textContent = '';
  }

  function clearToEmptyState() {
    clearVisuals();
    if (el.featureInfo) el.featureInfo.textContent = '';
    if (el.seriesInfo) el.seriesInfo.textContent = '';
    if (el.summary) el.summary.textContent = '';
    if (el.emptyState) el.emptyState.style.display = 'block';
    document.body.classList.add('noInstance');
    try {
      peState.instanceMeta = null;
      peState.instanceMetaFile = null;
      peState.instanceId = null;
      renderInstanceDetails();
    } catch (e) { }
    if (el.featureInfo) el.featureInfo.textContent = '';
    if (el.seriesInfo) el.seriesInfo.textContent = '';
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
      if (el.featureInfo) el.featureInfo.textContent = '';
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
    instanceId: null,
    biases: null,
    dilations: null,
    convCache: new Map(),
    lastBetaFile: null,
    fileList: null,
    availableConvFeatures: new Set(),

    // Instance metadata shown in #instanceDetailsCard
    instanceMeta: null,
    instanceMetaFile: null,
  };

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
    const args = ['predict_and_explain.py', '-M', model, '-D', dataset, '-s', String(start), '-e', String(end), '-E', explainer, '-L', label, '-r', refPolicy];
    const c = peGetPropagateValue();
    const tRaw = el.peTopT ? String(el.peTopT.value || '').trim() : '';
    const tVal = tRaw ? parseInt(tRaw, 10) : NaN;
    if (c === 'yes') {
      args.push('-c', 'yes');
    } else {
      if (Number.isFinite(tVal) && tVal > 0) {
        args.push('-t', String(tVal));
      } else {
        args.push('-c', 'no');
      }
    }
    return {
      args,
      cfg: {
        dataset,
        model,
        explainer,
        label,
        refPolicy,
        start,
        end,
        propagate: c,
        t: (Number.isFinite(tVal) ? tVal : null)
      }
    };
  }

  function peUpdateCmdPreview() {
    if (!el.peCmd) return;
    const {
      args
    } = peBuildArgs();
    el.peCmd.textContent = `python3 ${args.map(a => (/[^\w\-\.\/]/.test(a) ? JSON.stringify(a) : a)).join(' ')}`;
  }

  function pePopulateInstanceSelect(start, end) {
    if (!el.peInstance) return;
    el.peInstance.innerHTML = '';
    for (let i = start; i < end; i++) {
      const opt = document.createElement('option');
      opt.value = String(i);
      opt.textContent = String(i);
      el.peInstance.appendChild(opt);
    }
    el.peInstance.disabled = false;
    if (el.peLoadBtn) el.peLoadBtn.disabled = false;
    if (el.peOpenTabBtn) el.peOpenTabBtn.disabled = false;
  }
  async function peRefreshAvailableInstances(cfg) {
    if (!el.peInstance) return;
    const baseDir = `output/${cfg.dataset}/${cfg.model}/${cfg.explainer}/${cfg.label}`;
    try {
      const entries = await peListEntries(baseDir, 'dirs');
      const ids = entries.map(e => e.name).filter(n => /^\d+$/.test(n)).map(n => parseInt(n, 10)).filter(Number.isFinite).sort((a, b) => a - b);
      el.peInstance.innerHTML = '';
      if (ids.length === 0) {
        const opt = document.createElement('option');
        opt.value = '';
        opt.textContent = '(no instances found)';
        el.peInstance.appendChild(opt);
        el.peInstance.disabled = true;
        if (el.peLoadBtn) el.peLoadBtn.disabled = true;
        if (el.peOpenTabBtn) el.peOpenTabBtn.disabled = true;
        return;
      }
      ids.forEach(id => {
        const opt = document.createElement('option');
        opt.value = String(id);
        opt.textContent = String(id);
        el.peInstance.appendChild(opt);
      });
      el.peInstance.disabled = false;
      if (el.peLoadBtn) el.peLoadBtn.disabled = false;
      if (el.peOpenTabBtn) el.peOpenTabBtn.disabled = false;
      const cur = peState.instanceId;
      const pick = (cur != null && ids.includes(cur)) ? cur : ids[0];
      el.peInstance.value = String(pick);
      forceHideTopBusy();
    } catch (e) {
      console.error('[peRefreshAvailableInstances] failed', {
        baseDir,
        error: e
      });
      el.peInstance.innerHTML = '';
      const opt = document.createElement('option');
      opt.value = '';
      opt.textContent = '(no output folder yet)';
      forceHideTopBusy();
      el.peInstance.appendChild(opt);
      el.peInstance.disabled = true;
      if (el.peLoadBtn) el.peLoadBtn.disabled = true;
      if (el.peOpenTabBtn) el.peOpenTabBtn.disabled = true;
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
    if (peState.biases && peState.dilations && peState.lastRun && peState.lastRun.dataset === cfg.dataset && peState.lastRun.model === cfg.model) {
      return;
    }
    try {
      const [bText, dText] = await Promise.all([peFetchText(`/output/${cfg.dataset}/${cfg.model}/biases.csv`), peFetchText(`/output/${cfg.dataset}/${cfg.model}/dilations.csv`)]);
      const bRows = d3.csvParseRows(bText);
      const dRows = d3.csvParseRows(dText);
      peState.biases = (bRows[0] ? bRows[0].slice(1).map(x => +x) : null);
      peState.dilations = (dRows[0] ? dRows[0].slice(1).map(x => +x) : null);
    } catch (e) {
      console.warn('Could not load biases/dilations:', e);
      peState.biases = null;
      peState.dilations = null;
    }
  }
  async function peLoadInstance(cfg, instanceId) {
    peState.instanceId = instanceId;
    peState.convCache.clear();
    peState.instanceMeta = null;
    peState.instanceMetaFile = null;
    state.convFromFile = null;
    state.sigmaFromFile = null;
    const folder = `output/${cfg.dataset}/${cfg.model}/${cfg.explainer}/${cfg.label}/${instanceId}`;
    const entries = await peListEntries(folder, 'files');
    const files = entries.map(e => e.name);
    peState.fileList = files;
    peState.availableConvFeatures = new Set(files.filter(f => f.startsWith('convolved_instance_') && f.includes('_feature_') && f.endsWith('.csv')).map(f => {
      const m = f.match(/\_feature\_(\d+)\.csv$/);
      return m ? parseInt(m[1], 10) : null;
    }).filter(Number.isFinite));
    const betasPrefix = `betas_backpropagated_explanations_ref_policy_${cfg.refPolicy}_instance_${instanceId}`;
    const betasFile = files.find(f => f.startsWith(betasPrefix) && f.endsWith('.csv'));
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
      console.log('feature_meta:', toLink('data/feature_meta_instance_22.csv'));
      console.groupEnd();
    } catch (e) { }

    const featureSvg = document.getElementById('featureOverview');
    const seriesSvg = document.getElementById('seriesChart');
    const convSvg = document.getElementById('convChart');
    await Promise.all([peFade(featureSvg, 0, 0.12), peFade(seriesSvg, 0, 0.12), peFade(convSvg, 0, 0.12)]);
    const metaJsonPromise = metaJsonUrl ? peFetchJSON(metaJsonUrl).catch(() => null) : Promise.resolve(null);
    const [seriesText, alphaText, metaText, mrText, instMetaJson] = await Promise.all([
      peFetchText(seriesUrl),
      peFetchText(alphaUrl),
      fetchFirst(["data/feature_meta_instance_22.csv"]),
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
    const meta = parseFeatureMetaCSV(metaText);
    const alphaMap = parseAlphaCSV(alphaText);
    const embMap = parseEmbeddingCSV(mrText);
    if (mrUrl && embMap.size === 0) {
      console.warn('mr_instance file loaded but produced 0 values. Check CSV format:', mrUrl);
    }

    // Store & render per-instance metadata (if present)
    if (instMetaJson && typeof instMetaJson === 'object') {
      peState.instanceMeta = instMetaJson;
      peState.instanceMetaFile = metaJsonFile || null;
    } else {
      peState.instanceMeta = null;
      peState.instanceMetaFile = metaJsonFile || null;
    }
    setFeatures(mergeFeatureSignals(meta, alphaMap, embMap));
    renderInstanceDetails();
    await peFade(featureSvg, 1, 0.18);
    setSeries(parseSeriesRowCSV(seriesText), {
      deferRender: true
    });
    if (refText) {
      state.referenceSeriesMap = new Map();
      state.referenceSeriesMap.set(cfg.refPolicy, parseSeriesRowCSV(refText));
      state.referenceSeries = state.referenceSeriesMap.get(cfg.refPolicy);
    } else {
      state.referenceSeriesMap = new Map();
      state.referenceSeries = null;
    }
    if (betaText) {
      state.betaRaw = parseBetaRowCSV(betaText);
      state.betaStats = computeBetaStats(state.betaRaw);
      peState.lastBetaFile = betasFile;
    } else {
      state.betaRaw = [];
      state.betaStats = null;
      peState.lastBetaFile = null;
    }
    renderConv();
    renderSeries();
    renderSummary();
    await peFade(seriesSvg, 1, 0.18);
    await peFade(convSvg, 1, 0.18);
    renderKernelList();
    renderKernelPreview();
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
    const sigmaUrl = `${base}/convolved_instance_after_sigma_instance_${instanceId}_feature_${featureIdx}.csv`;
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
          convUrl: toLink(convUrl),
          sigmaUrl: toLink(sigmaUrl)
        });
      } catch (e) { }
    } catch (e) { }
    try {
      setTopBusy(true, 'Loading convolution…');
      let __cText, __sText;
      try {
        const [cText, sText] = await Promise.all([peFetchText(convUrl), peFetchText(sigmaUrl)]);
        __cText = cText;
        __sText = sText;
      } finally {
        setTopBusy(false, '');
      }
      const cText = __cText,
        sText = __sText;
      const resp = parseIndexValueCSV(cText, 1).map(d => ({
        t: d.t,
        v: d.v
      }));
      const sigma = parseIndexValueCSV(sText, 1).map(d => ({
        t: d.t,
        v: d.v
      }));
      const pack = {
        resp,
        sigma
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

  function peCopy() {
    if (!el.peCmd) return;
    const text = el.peCmd.textContent || '';
    if (!text) return;
    navigator.clipboard.writeText(text).catch(() => { });
  }

  function peInitUI() {
    if (!el.peRunBtn) return;
    if (!el.peStartNum || !el.peEndNum) return;
    peNormalizeStartEnd();
    peUpdateTEnable();
    peUpdateCmdPreview();
    peDiscoverOutputOptions().then(() => {
      try {
        const built = peBuildArgs();
        peState.lastRun = built.cfg;
        peRefreshAvailableInstances(built.cfg);
      } catch (e) { }
    }).catch(() => { });
    [el.peDataset, el.peLabelSelect, el.peModel, el.peExplainer, el.peRefPolicy].forEach(n => {
      if (!n) return;
      n.addEventListener('change', async () => {
        peUpdateCmdPreview();
        await peDiscoverOutputOptions().catch(() => { });
        const built = peBuildArgs();
        peState.lastRun = built.cfg;
        await peRefreshAvailableInstances(built.cfg);
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
    if (el.peInstance) el.peInstance.addEventListener('change', () => {
      const v = el.peInstance.value;
      if (el.status) el.status.textContent = v ? `Selected instance ${v} (click Load)` : 'No instance selected';
    });
    if (el.peInstance) el.peInstance.disabled = true;
    if (el.peLoadBtn) el.peLoadBtn.disabled = true;
    if (el.peOpenTabBtn) el.peOpenTabBtn.disabled = true;
    try {
      const built = peBuildArgs();
      peState.lastRun = built.cfg;
      peRefreshAvailableInstances(built.cfg);
    } catch (e) { }
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
    renderConv();
    renderSeries();
    renderSummary();
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
    el.featureIdx.addEventListener('keydown', (ev) => {
      if (ev.key === 'Enter') go();
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

  initReferenceToggle();
  renderKernelPreview();
  renderKernelList();
  clearToEmptyState();
  forceHideTopBusy();
  peInitUI();
  forceHideTopBusy();
  let resizeRaf = null;
  window.addEventListener('resize', () => {
    if (resizeRaf) cancelAnimationFrame(resizeRaf);
    resizeRaf = requestAnimationFrame(() => {
      resizeRaf = null;
      renderAll();
    });
  });
})();