/* ARES live dashboard - hash router, polling, ticker KPIs, connection indicator and the "simulados" toggle. */
'use strict';

(() => {
  const { $, $$, esc, num, fmtInt, fmtCompact, fmtNum, api, store, debounce, emptyState } = U;
  const ROUTES = { '': 'live', progreso: 'progress', inteligencia: 'intel', liga: 'league', conocimiento: 'knowledge', run: 'run' };
  // Page data endpoints and their refresh periods (ms); /api/runs is the only 1 Hz poll.
  const PAGE_DATA = { progress: ['/api/progress?limit=300', 10000], intel: ['/api/intelligence', 15000],
    league: ['/api/league', 15000], knowledge: ['/api/knowledge', 20000] };
  const state = { page: 'live', runId: null, runs: [], summary: null, serverTime: 0, lastOk: 0, failures: 0,
    showDemo: store.get('showDemo', true), data: {}, dataAt: {}, loading: {}, history: null, historyFor: null, historyAt: 0 };

  // ----------------------------------------------------------------------------- routing
  function parseHash() {
    const parts = (location.hash || '#/').replace(/^#\/?/, '').split('/');
    const page = ROUTES[parts[0]] || 'live';
    return { page, runId: page === 'run' ? decodeURIComponent(parts.slice(1).join('/')) : null };
  }
  function route() {
    const r = parseHash();
    state.page = r.page; state.runId = r.runId;
    $$('.page').forEach((p) => p.classList.toggle('active', p.id === 'page-' + r.page));
    $$('.tabs a').forEach((a) => {
      const on = a.getAttribute('data-page') === r.page || (r.page === 'run' && a.getAttribute('data-page') === 'live');
      a.classList.toggle('active', on);
      if (on) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current');
    });
    if (r.page !== 'run') { L.frames.detach('detail'); const sec = $('#page-run'); sec.removeAttribute('data-run'); sec.innerHTML = ''; }
    else { state.history = null; state.historyFor = null; state.historyAt = 0; }
    U.tip.hide();
    document.title = 'ARES en vivo' + ({ progress: ' · Progreso', intel: ' · Inteligencia', league: ' · Liga', knowledge: ' · Conocimiento', run: ' · ' + (r.runId || '') }[r.page] || '');
    render();
    if (PAGE_DATA[r.page] && state.data[r.page]) renderPage(r.page);
    refreshPageData(true);
    window.scrollTo(0, 0);
  }

  // ----------------------------------------------------------------------------- data
  const visibleRuns = () => state.runs.filter((r) => state.showDemo || !r.demo);
  async function pollRuns() {
    try {
      const d = await api('/api/runs', 6000);
      state.runs = Array.isArray(d.runs) ? d.runs : [];
      state.summary = d.summary || null; state.serverTime = num(d.server_time) || 0;
      state.lastOk = Date.now(); state.failures = 0;
      render();
      if (state.page === 'run' && state.runs.some((r) => r.run_id === state.runId)) refreshHistory();
    } catch (e) {
      state.failures += 1;
    }
    renderConn();
    setTimeout(pollRuns, document.hidden ? 5000 : 1000);
  }
  async function refreshPageData(force) {
    const spec = PAGE_DATA[state.page]; if (!spec) return;
    const [url, period] = spec; const page = state.page;
    if (state.loading[page]) return;
    if (!force && Date.now() - (state.dataAt[page] || 0) < period) return;
    if (force && state.data[page] && Date.now() - (state.dataAt[page] || 0) < 2000) return;
    state.loading[page] = true;
    const sec = $('#page-' + page); if (!state.data[page]) sec.classList.add('loading');
    try {
      state.data[page] = await api(url, 15000);
      state.dataAt[page] = Date.now();
      renderPage(page);
    } catch (e) {
      state.dataAt[page] = Date.now() - period + 3000; // retry in ~3 s
      if (!state.data[page]) {
        const target = { progress: '#race-chart', intel: '#intel-tiles', league: '#league-body', knowledge: '#knowledge-body' }[page];
        if (target) $(target).innerHTML = emptyState('No se pudo cargar', String(e && e.message ? e.message : e));
      }
    } finally { state.loading[page] = false; sec.classList.remove('loading'); }
  }
  async function refreshHistory() {
    const id = state.runId; if (!id) return;
    if (state.historyFor === id && Date.now() - state.historyAt < 10000) return;
    state.historyAt = Date.now();
    try {
      const h = await api(`/api/history/${encodeURIComponent(id)}?limit=400`, 10000);
      if (state.runId === id) { state.history = h; state.historyFor = id; render(); }
    } catch (e) { /* keep the previous history */ }
  }

  // ----------------------------------------------------------------------------- rendering
  function renderTicker() {
    const runs = visibleRuns();
    const live = runs.filter((r) => !r.stale);
    let badges = 0, ms = 0, maps = 0, wins = 0, frames = 0, fps = 0, battling = 0;
    for (const r of runs) {
      badges += num(r.badges) || 0; ms += Array.isArray(r.milestones) ? r.milestones.length : (num(r.milestones) || 0);
      maps += num(r.maps_visited) || 0; wins += num(r.stats && r.stats.wins) || 0; frames += num(r.frames) || 0;
      if (!r.stale) { fps += num(r.fps) || 0; if (r.battle && (r.battle.me || r.battle.choice)) battling += 1; }
    }
    const real = live.filter((r) => !r.demo).length;
    const kpi = (v, label, t) => `<div class="kpi" data-tip="${esc(t)}"><b>${esc(v)}</b><span>${esc(label)}</span></div>`;
    $('#ticker').innerHTML = kpi(`${live.length}`, live.length === 1 ? 'en vivo' : 'en vivo', `${real} reales · ${live.length - real} simuladas · ${runs.length - live.length} inactivas`) +
      kpi(fmtInt(battling), 'en combate', 'Ejecuciones en vivo dentro de un combate ahora mismo') +
      kpi(fmtInt(badges), 'medallas', 'Suma de medallas de las ejecuciones visibles') +
      kpi(fmtInt(ms), 'hitos', 'Suma de hitos de historia completados') +
      kpi(fmtInt(maps), 'mapas', 'Suma de mapas distintos visitados') +
      kpi(fmtInt(wins), 'victorias', 'Combates ganados') +
      kpi(fmtCompact(frames), 'fotogramas', `Fotogramas emulados en total · ${fmtNum(fps, 0)} fps ahora`);
  }
  function renderConn() {
    const el = $('#conn'), txt = el.querySelector('.txt');
    const age = state.lastOk ? (Date.now() - state.lastOk) / 1000 : Infinity;
    const ok = state.failures === 0 && age < 5;
    el.classList.toggle('ok', ok); el.classList.toggle('err', !ok && state.lastOk !== 0 || state.failures > 2);
    txt.textContent = ok ? 'conectado' : state.lastOk ? `sin conexión · ${Math.round(age)} s` : (state.failures ? 'sin conexión' : 'conectando…');
    el.title = ok ? 'Datos actualizados cada segundo desde scripts.live_dashboard' : 'El servidor del panel no responde; reintentando';
  }
  function renderLive() {
    const runs = visibleRuns(), grid = $('#live-grid'), emptyEl = $('#live-empty');
    L.renderGrid(grid, runs);
    const hiddenDemo = state.runs.length - runs.length;
    if (!runs.length) {
      emptyEl.hidden = false;
      emptyEl.innerHTML = emptyState(hiddenDemo ? `Solo hay ${hiddenDemo} ejecuciones simuladas (ocultas)` : 'Ningún agente está publicando',
        'Cada ejecución publica artifacts/live/<run_id>/status.json. Lanza una campaña, el agente universal o la liga:',
        ['python -m scripts.run_campaign --game firered --resume', 'python -m scripts.ares_u_league train', 'python -m scripts.live_demo_publisher   (datos simulados)']);
    } else emptyEl.hidden = true;
  }
  function renderPage(page) {
    const rerender = () => renderPage(page);
    const d = state.data[page];
    if (page === 'progress') P.renderProgress(d, state.showDemo, rerender);
    else if (page === 'intel') P.renderIntel(d, state.showDemo, rerender);
    else if (page === 'league') P.renderLeague(d, state.showDemo, rerender);
    else if (page === 'knowledge') P.renderKnowledge(d, state.showDemo, rerender);
  }
  function render() {
    renderTicker();
    if (state.page === 'live') renderLive();
    else if (state.page === 'run') {
      const run = state.runs.find((r) => r.run_id === state.runId) || null;
      if (run || state.lastOk) P.renderRun($('#page-run'), run, state.historyFor === state.runId ? state.history : null);
    } else if (state.data[state.page]) {
      refreshPageData(false);
    } else refreshPageData(true);
  }

  // ----------------------------------------------------------------------------- wiring
  const demoBox = $('#show-demo');
  demoBox.checked = !!state.showDemo;
  demoBox.addEventListener('change', () => {
    state.showDemo = demoBox.checked; store.set('showDemo', state.showDemo);
    render(); if (PAGE_DATA[state.page] && state.data[state.page]) renderPage(state.page);
  });
  window.addEventListener('hashchange', route);
  window.addEventListener('resize', debounce(() => {
    if (PAGE_DATA[state.page] && state.data[state.page]) renderPage(state.page);
    if (state.page === 'run') { const s = $('#page-run'); s._histW = -1; render(); }
  }, 250));
  document.addEventListener('visibilitychange', () => { if (!document.hidden) { render(); refreshPageData(false); } });
  setInterval(() => { renderConn(); if (!document.hidden) refreshPageData(false); }, 1000);

  route();
  pollRuns();
})();
