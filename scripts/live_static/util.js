/* ARES live dashboard - shared helpers (formatting, escaping, fetch, tooltip, type chips). No dependencies. */
'use strict';

const U = (() => {
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
  const ESC = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };
  const esc = (s) => String(s === null || s === undefined ? '' : s).replace(/[&<>"']/g, (c) => ESC[c]);

  /** Finite number or null (accepts numeric strings). */
  const num = (v) => {
    if (v === null || v === undefined || v === '' || typeof v === 'boolean') return null;
    const n = typeof v === 'number' ? v : Number(v);
    return Number.isFinite(n) ? n : null;
  };
  const NF = new Intl.NumberFormat('es-ES');
  const fmtInt = (v) => { const n = num(v); return n === null ? '—' : NF.format(Math.round(n)); };
  const fmtNum = (v, d = 1) => {
    const n = num(v); if (n === null) return '—';
    return new Intl.NumberFormat('es-ES', { maximumFractionDigits: d, minimumFractionDigits: 0 }).format(n);
  };
  const fmtCompact = (v) => {
    const n = num(v); if (n === null) return '—';
    const a = Math.abs(n);
    if (a >= 1e9) return fmtNum(n / 1e9, 2) + ' G';
    if (a >= 1e6) return fmtNum(n / 1e6, a >= 1e7 ? 1 : 2) + ' M';
    if (a >= 1e4) return fmtNum(n / 1e3, a >= 1e5 ? 0 : 1) + ' k';
    return fmtNum(n, a < 10 ? 2 : (a < 100 ? 1 : 0));
  };
  const fmtPct = (v, d = 0) => { const n = num(v); return n === null ? '—' : fmtNum(n * 100, d) + ' %'; };
  const fmtDur = (seconds) => {
    const s = num(seconds); if (s === null) return '—';
    if (s < 60) return Math.round(s) + ' s';
    const m = Math.floor(s / 60);
    if (m < 60) return m + ' min ' + Math.round(s % 60) + ' s';
    const h = Math.floor(m / 60);
    if (h < 48) return h + ' h ' + (m % 60) + ' min';
    return Math.floor(h / 24) + ' d ' + (h % 24) + ' h';
  };
  const NOMINAL_FPS = { gb: 59.7275, gbc: 59.7275, gba: 59.7275, nds: 59.8261, sim: 60 };
  /** Emulated play time of `frames` frames on a platform. */
  const fmtPlay = (frames, platform) => {
    const f = num(frames); if (f === null) return '—';
    return fmtDur(f / (NOMINAL_FPS[platform] || 59.7275));
  };
  const ago = (s) => { const n = num(s); return n === null ? '' : (n < 2 ? 'ahora' : 'hace ' + fmtDur(n)); };

  const GAMES = {
    rojo: 'Pokémon Rojo', cristal: 'Pokémon Cristal', firered: 'Pokémon Rojo Fuego', emerald: 'Pokémon Esmeralda',
    platino: 'Pokémon Platino', heartgold: 'Pokémon Oro HeartGold', negro: 'Pokémon Negro', negro2: 'Pokémon Negro 2',
    stadium: 'ARES Stadium',
  };
  const gameColor = (game) => (GAMES[game] ? `var(--g-${game})` : 'var(--accent)');
  /** Resolved colour (needed inside SVG attributes, where var() works but we also want a fallback). */
  const cssVar = (name, fallback) => {
    try { return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback; } catch (e) { return fallback; }
  };
  const gameHex = (game) => cssVar(`--g-${game}`, '#3987e5') || '#3987e5';
  const runTitle = (run) => (run && run.meta && run.meta.title) || GAMES[run && run.game] || (run && run.game) || (run && run.run_id) || '¿?';
  const KIND = { campaign: 'campaña', universal: 'universal', league: 'liga', experiment: 'experimento' };
  const kindLabel = (k) => KIND[k] || k || '';
  const PLATFORM = { gb: 'GB', gbc: 'GBC', gba: 'GBA', nds: 'NDS', sim: 'simulador' };
  const genLabel = (run) => {
    const meta = (run && run.meta) || {};
    const gen = num(run.generation) || num(meta.generation);
    const plat = PLATFORM[run.platform || meta.platform] || (run.platform || meta.platform || '');
    return [gen ? 'Gen ' + gen : '', plat].filter(Boolean).join(' · ');
  };

  const TYPE_COLORS = {
    normal: '#a8a77a', fire: '#ee8130', water: '#6390f0', electric: '#f7d02c', grass: '#7ac74c', ice: '#96d9d6',
    fighting: '#c22e28', poison: '#a33ea1', ground: '#e2bf65', flying: '#a98ff3', psychic: '#f95587', bug: '#a6b91a',
    rock: '#b6a136', ghost: '#735797', dragon: '#6f35fc', dark: '#705746', steel: '#b7b7ce', fairy: '#d685ad',
  };
  const TYPE_ES = {
    normal: 'Normal', fire: 'Fuego', water: 'Agua', electric: 'Eléctrico', grass: 'Planta', ice: 'Hielo', fighting: 'Lucha',
    poison: 'Veneno', ground: 'Tierra', flying: 'Volador', psychic: 'Psíquico', bug: 'Bicho', rock: 'Roca', ghost: 'Fantasma',
    dragon: 'Dragón', dark: 'Siniestro', steel: 'Acero', fairy: 'Hada',
  };
  const DARK_TEXT = new Set(['normal', 'electric', 'grass', 'ice', 'ground', 'flying', 'bug', 'rock', 'steel', 'fairy', 'fire', 'water', 'psychic']);
  const typeKey = (t) => String(t || '').trim().toLowerCase();
  const typeName = (t) => TYPE_ES[typeKey(t)] || String(t || '');
  const typeChip = (t) => {
    const k = typeKey(t); if (!k) return '';
    const bg = TYPE_COLORS[k] || '#555';
    const fg = DARK_TEXT.has(k) ? '#161616' : '#fff';
    return `<span class="type" style="background:${bg};color:${fg}">${esc(typeName(k))}</span>`;
  };
  const typeChips = (types) => (Array.isArray(types) ? types : (types ? [types] : [])).map(typeChip).join('');

  /** Effectiveness multiplier -> {cls, text}. */
  const effInfo = (v) => {
    const n = num(v); if (n === null) return null;
    const x = '×' + fmtNum(n, 2);
    if (n === 0) return { cls: 'bad', text: x + ' no afecta' };
    if (n < 1) return { cls: 'warn', text: x + ' poco eficaz' };
    if (n > 1) return { cls: 'good', text: x + ' súper eficaz' };
    return { cls: '', text: x + ' neutro' };
  };

  /** fetch JSON with a timeout; throws on HTTP errors. */
  async function api(path, timeoutMs = 8000) {
    const ctl = new AbortController();
    const timer = setTimeout(() => ctl.abort(), timeoutMs);
    try {
      const r = await fetch(path, { cache: 'no-store', signal: ctl.signal });
      if (!r.ok) throw new Error(`HTTP ${r.status} en ${path}`);
      return await r.json();
    } finally { clearTimeout(timer); }
  }

  const store = {
    get(key, fallback) { try { const v = localStorage.getItem('ares.' + key); return v === null ? fallback : JSON.parse(v); } catch (e) { return fallback; } },
    set(key, value) { try { localStorage.setItem('ares.' + key, JSON.stringify(value)); } catch (e) { /* private mode */ } },
  };

  const debounce = (fn, ms) => { let t = 0; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; };

  /** Wilson score interval for a proportion p over n trials (95 %). */
  const wilson = (p, n, z = 1.96) => {
    if (!n || p === null) return null;
    const d = 1 + z * z / n, c = (p + z * z / (2 * n)) / d, w = z * Math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d;
    return [Math.max(0, c - w), Math.min(1, c + w)];
  };

  /** Text of an event entry (strings, or dicts published by other runners). */
  const eventText = (e) => {
    if (e === null || e === undefined) return '';
    if (typeof e !== 'object') return String(e);
    const text = e.text || e.message || e.msg || e.event || e.description || JSON.stringify(e);
    const f = num(e.frame !== undefined ? e.frame : e.frames);
    return (f !== null ? 'f' + fmtInt(f) + ' · ' : '') + text;
  };
  const isBadgeMilestone = (m) => /medalla|badge|gimnasio|gym|l[ií]der/i.test(String(m || ''));

  // ----------------------------------------------------------------------------------- tooltip
  const tip = (() => {
    let el = null;
    const node = () => (el = el || document.getElementById('tooltip'));
    function show(html, x, y) {
      const t = node(); if (!t) return;
      t.innerHTML = html; t.hidden = false;
      const r = t.getBoundingClientRect();
      let left = x + 14, top = y + 14;
      if (left + r.width > window.innerWidth - 8) left = Math.max(8, x - r.width - 14);
      if (top + r.height > window.innerHeight - 8) top = Math.max(8, y - r.height - 14);
      t.style.left = left + 'px'; t.style.top = top + 'px';
    }
    function hide() { const t = node(); if (t) t.hidden = true; }
    // Any element with data-tip (plain text) gets a tooltip.
    document.addEventListener('mouseover', (ev) => {
      const target = ev.target && ev.target.closest ? ev.target.closest('[data-tip]') : null;
      if (target) show(`<div>${esc(target.getAttribute('data-tip'))}</div>`, ev.clientX, ev.clientY);
    });
    document.addEventListener('mousemove', (ev) => {
      const target = ev.target && ev.target.closest ? ev.target.closest('[data-tip]') : null;
      if (target) show(`<div>${esc(target.getAttribute('data-tip'))}</div>`, ev.clientX, ev.clientY);
    });
    document.addEventListener('mouseout', (ev) => {
      const target = ev.target && ev.target.closest ? ev.target.closest('[data-tip]') : null;
      if (target && !(ev.relatedTarget && target.contains(ev.relatedTarget))) hide();
    });
    return { show, hide };
  })();
  const ttRow = (color, label, value) =>
    `<div class="tt-row">${color ? `<i class="tt-key" style="background:${color}"></i>` : ''}<b>${esc(value)}</b><span>${esc(label)}</span></div>`;

  const emptyState = (title, body, commands = []) =>
    `<div class="empty-state"><h2>${esc(title)}</h2><p>${esc(body)}</p>${commands.map((c) => `<div><code>${esc(c)}</code></div>`).join('')}</div>`;

  return {
    $, $$, esc, num, fmtInt, fmtNum, fmtCompact, fmtPct, fmtDur, fmtPlay, ago, GAMES, gameColor, gameHex, cssVar, runTitle,
    kindLabel, genLabel, TYPE_COLORS, TYPE_ES, typeKey, typeName, typeChip, typeChips, effInfo, api, store, debounce,
    wilson, eventText, isBadgeMilestone, tip, ttRow, emptyState, NOMINAL_FPS,
  };
})();
