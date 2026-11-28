/* ARES live dashboard - run cards (live grid) and the frame loader. */
'use strict';

const L = (() => {
  const { esc, num, fmtInt, fmtNum, fmtCompact, fmtPct, fmtPlay, ago, typeChips, effInfo, eventText, isBadgeMilestone } = U;

  // ------------------------------------------------------------------------------- fragments
  const badgeSvg = (on, i) => on
    ? `<svg viewBox="0 0 24 24" aria-hidden="true"><defs><linearGradient id="bg${i}" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#ffe08a"/><stop offset="1" stop-color="#c98500"/></linearGradient></defs>` +
      `<path d="M12 2l2.9 6.1 6.6.8-4.9 4.6 1.3 6.6L12 16.9 6.1 20.1l1.3-6.6L2.5 8.9l6.6-.8z" fill="url(#bg${i})" stroke="#fab219" stroke-width="1"/></svg>`
    : `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 2l2.9 6.1 6.6.8-4.9 4.6 1.3 6.6L12 16.9 6.1 20.1l1.3-6.6L2.5 8.9l6.6-.8z" fill="none" stroke="#383835" stroke-width="1.4"/></svg>`;
  function badgesHtml(run) {
    const n = Math.max(0, Math.min(8, Math.round(num(run.badges) || 0)));
    let s = '';
    for (let i = 0; i < 8; i++) s += `<span class="badge-slot" data-tip="${i < n ? 'Medalla ' + (i + 1) + ' conseguida' : 'Medalla ' + (i + 1) + ' pendiente'}">${badgeSvg(i < n, i)}</span>`;
    return s + `<span class="count">${n}/8</span>`;
  }
  const hpClass = (f) => (f <= 0.2 ? ' low' : f <= 0.5 ? ' mid' : '');
  function monHtml(m) {
    if (!m || typeof m !== 'object') return '';
    const hp = num(m.hp), max = num(m.max_hp);
    const f = hp !== null && max ? Math.max(0, Math.min(1, hp / max)) : null;
    const status = m.status ? `<span class="status-chip">${esc(String(m.status).toUpperCase())}</span>` : '';
    const moves = Array.isArray(m.moves) ? m.moves.map((mv) => (typeof mv === 'object' && mv ? mv.name : mv)).filter(Boolean) : [];
    const tipText = moves.length ? 'Movimientos: ' + moves.join(', ') : '';
    return `<div class="mon"${tipText ? ` data-tip="${esc(tipText)}"` : ''}><div class="name">${esc(m.species || '¿?')}<small>Nv ${esc(m.level !== undefined ? m.level : '?')}</small>${status}</div>` +
      `<div class="types">${typeChips(m.types)}</div>` +
      `<div class="hp"><div class="hpbar${f === null ? '' : hpClass(f)}"><i style="width:${f === null ? 0 : (f * 100).toFixed(1)}%"></i></div>` +
      `<span class="num">${hp === null ? '—' : fmtInt(hp)}/${max === null ? '—' : fmtInt(max)}</span></div></div>`;
  }
  function partyHtml(run) {
    const party = Array.isArray(run.party) ? run.party : [];
    if (!party.length) return '<div class="empty-note">Sin equipo publicado</div>';
    return party.slice(0, 6).map(monHtml).join('');
  }
  function fighterHtml(m, who, enemy) {
    if (!m || typeof m !== 'object') return `<div class="fighter${enemy ? ' enemy' : ''}"><div class="who">${who}</div><div class="name">—</div></div>`;
    const hp = num(m.hp), max = num(m.max_hp);
    const f = hp !== null && max ? Math.max(0, Math.min(1, hp / max)) : null;
    return `<div class="fighter${enemy ? ' enemy' : ''}"><div class="who">${who}</div>` +
      `<div class="name">${esc(m.species || '¿?')} <small>Nv ${esc(m.level !== undefined ? m.level : '?')}</small></div>` +
      `<div class="types">${typeChips(m.types)}</div>` +
      `<div class="hpbar${f === null ? '' : hpClass(f)}"><i style="width:${f === null ? 0 : (f * 100).toFixed(1)}%"></i></div></div>`;
  }
  /** Battle panel: fighters, chosen action, effectiveness, the brain's reason and score bars. */
  function battleHtml(run, maxScores = 6) {
    const b = run.battle;
    if (!b || typeof b !== 'object' || (!b.me && !b.enemy && !b.choice)) return '';
    const eff = effInfo(b.effectiveness);
    const turn = num(b.turn);
    const head = `<div class="battle-top">${fighterHtml(b.me, 'Nuestro', false)}<div class="vs">VS</div>${fighterHtml(b.enemy, b.is_wild ? 'Salvaje' : 'Rival', true)}</div>`;
    const choice = b.choice !== undefined && b.choice !== null ? String(b.choice) : '';
    let decision = `<div class="decision"><span class="label">Decisión${turn !== null ? ' · turno ' + fmtInt(turn) : ''}</span>` +
      `<span class="move">${esc(choice || '—')}</span>${eff ? `<span class="eff ${eff.cls}" data-tip="Multiplicador de tipo esperado del movimiento elegido">${esc(eff.text)}</span>` : ''}</div>`;
    if (b.reason) decision += `<div class="reason">${esc(b.reason)}</div>`;
    let scores = '';
    const entries = b.scores && typeof b.scores === 'object'
      ? (Array.isArray(b.scores) ? b.scores.map((s, i) => [s && s.name ? s.name : 'opción ' + (i + 1), num(s && typeof s === 'object' ? s.score : s)])
        : Object.entries(b.scores).map(([k, v]) => [k, num(v)])).filter((e) => e[1] !== null)
      : [];
    if (entries.length) {
      entries.sort((a, c) => c[1] - a[1]);
      const lo = Math.min(0, ...entries.map((e) => e[1])), hi = Math.max(...entries.map((e) => e[1]));
      const span = hi - lo || 1;
      const chosenName = choice && entries.some((e) => e[0] === choice) ? choice : entries[0][0];
      scores = '<div class="scores">' + entries.slice(0, maxScores).map(([k, v]) =>
        `<div class="score${k === chosenName ? ' chosen' : ''}"><span class="nm" title="${esc(k)}">${esc(k)}</span>` +
        `<div class="bar"><i style="width:${Math.max(2, (v - lo) / span * 100).toFixed(1)}%"></i></div><span class="v">${esc(fmtNum(v, 2))}</span></div>`).join('') + '</div>';
    }
    return `<div class="battle">${head}${decision}${scores}</div>`;
  }
  /** Stats grid; works for campaign/universal and league runs. */
  function statsHtml(run) {
    const st = run.stats && typeof run.stats === 'object' ? run.stats : {};
    const w = num(st.wins), l = num(st.losses), battles = num(st.battles);
    const wr = w !== null && l !== null && w + l > 0 ? w / (w + l) : null;
    const kn = run.knowledge && typeof run.knowledge === 'object' ? run.knowledge : null;
    const knItems = kn ? num(kn.items) : null;
    const cells = [];
    cells.push(['Fotogramas', fmtCompact(run.frames), 'juego ' + fmtPlay(run.frames, run.platform || (run.meta || {}).platform), 'Fotogramas emulados y tiempo de juego equivalente']);
    cells.push(['Velocidad', fmtNum(run.fps, 0) + ' fps', run.fps ? '×' + fmtNum((num(run.fps) || 0) / 60, 1) + ' tiempo real' : '', 'Fotogramas emulados por segundo de reloj']);
    if (run.kind !== 'league') {
      cells.push(['Mapas', fmtInt(run.maps_visited), 'visitados', 'Mapas distintos visitados (RAM)']);
      cells.push(['Hitos', fmtInt(Array.isArray(run.milestones) ? run.milestones.length : run.milestones), 'de la historia', 'Hitos de historia completados']);
    }
    cells.push(['Combates', fmtInt(battles), wr !== null ? fmtPct(wr) + ' victorias' : (w !== null ? fmtInt(w) + ' victorias' : ''), `Victorias ${fmtInt(w)} · derrotas ${fmtInt(l)}`]);
    if (st.stalls !== undefined || st.recoveries !== undefined)
      cells.push(['Atascos', fmtInt(st.stalls), fmtInt(st.recoveries) + ' recuperados', 'Episodios de atasco detectados y recuperados por el vigilante de progreso']);
    if (knItems !== null)
      cells.push(['Conocimiento', fmtInt(knItems), num(kn.inherited) !== null ? fmtInt(kn.inherited) + ' heredados' : 'elementos', 'Elementos de conocimiento (tabla de tipos, habilidades, …)']);
    const skip = new Set(['battles', 'wins', 'losses', 'stalls', 'recoveries']);
    for (const [k, v] of Object.entries(st)) {
      if (skip.has(k) || cells.length >= 10 || num(v) === null) continue;
      cells.push([k.replace(/_/g, ' '), fmtInt(v), '', 'Estadística publicada: ' + k]);
    }
    return cells.map(([label, value, sub, t]) => `<div class="stat" data-tip="${esc(t)}"><b>${esc(value)}</b><span>${esc(label)}</span>${sub ? `<em>${esc(sub)}</em>` : ''}</div>`).join('');
  }
  function timelineHtml(run, max = 6) {
    const ms = Array.isArray(run.milestones) ? run.milestones : [];
    if (!ms.length) return '<li class="empty-note">Ningún hito todavía</li>';
    const items = ms.map((m, i) => [i + 1, m]).reverse();
    let s = items.slice(0, max).map(([n, m]) => `<li class="${isBadgeMilestone(m) ? 'badge-ms' : ''}"><span class="n">${n}</span><span>${esc(m)}</span></li>`).join('');
    if (items.length > max) s += `<li><span></span><span class="more">y ${items.length - max} hitos anteriores</span></li>`;
    return s;
  }
  function eventsHtml(run, max = 8) {
    const ev = Array.isArray(run.events) ? run.events.slice().reverse() : [];
    if (!ev.length) return '<li class="empty-note">Sin eventos</li>';
    return ev.slice(0, max).map((e) => `<li>${esc(eventText(e))}</li>`).join('');
  }
  function leagueSideHtml(run) {
    const lg = run.league && typeof run.league === 'object' ? run.league : {};
    const champ = lg.champion && typeof lg.champion === 'object' ? lg.champion : null;
    const rivals = Array.isArray(lg.rivals) ? lg.rivals.slice().sort((a, b) => (num(b.elo) || 0) - (num(a.elo) || 0)) : [];
    let s = `<div><div class="label">Campeón actual</div><div class="goal">${champ ? esc(champ.name) + ' <small>' + fmtInt(champ.elo) + ' Elo</small>' : '—'}</div></div>`;
    if (rivals.length) s += `<div><div class="label">Elo</div>` + rivals.slice(0, 5).map((r) =>
      `<div class="score"><span class="nm">${esc(r.name)}</span><div class="bar"><i style="width:${Math.max(2, Math.min(100, ((num(r.elo) || 1500) - 1300) / 4)).toFixed(0)}%"></i></div><span class="v">${fmtInt(r.elo)}</span></div>`).join('') + '</div>';
    s += `<div class="empty-note">Generación ${fmtInt(lg.generation)} · ${fmtInt(lg.matches)} combates · <a href="#/liga">ver liga</a></div>`;
    return s;
  }
  function tagsHtml(run) {
    const live = !run.stale;
    let s = `<span class="tag game" style="--game:${U.gameColor(run.game)}" data-tip="${esc(U.kindLabel(run.kind))}">${esc(U.genLabel(run) || run.game || '')}</span>`;
    s += live ? '<span class="tag live" data-tip="Publicó hace menos de 30 s"><span class="pulse"></span>en vivo</span>'
      : `<span class="tag stale" data-tip="Sin publicar desde ${esc(ago(run.age_s))}">inactivo ${esc(ago(run.age_s))}</span>`;
    if (run.demo) s += '<span class="tag demo" data-tip="Datos simulados por scripts.live_demo_publisher, no es un agente real">simulado</span>';
    return s;
  }

  // ------------------------------------------------------------------------------- frame loader
  /**
   * Conditional GETs of /frame/<id>.png (If-None-Match) at a few fps for the images on screen; the img
   * only changes when the server ETag changes. Stale runs are refreshed only when /api/runs reports a new
   * frame_etag. One request in flight per image.
   */
  const frames = (() => {
    const slots = new Map(); // key -> {runId, img, etag, url, inflight, last, period, stale, hint}
    const visible = new Set();
    const io = 'IntersectionObserver' in window ? new IntersectionObserver((entries) => {
      for (const e of entries) { const k = e.target.getAttribute('data-slot'); if (e.isIntersecting) visible.add(k); else visible.delete(k); }
    }) : null;
    const strip = (t) => String(t || '').replace(/^W\//, '').replace(/"/g, '');
    function attach(key, runId, img, period) {
      let s = slots.get(key);
      if (s && s.img === img) { s.period = period; return s; }
      if (s) detach(key);
      img.setAttribute('data-slot', key);
      s = { runId, img, etag: '', url: '', inflight: false, last: 0, period, stale: false, hint: '', hasFrame: true };
      slots.set(key, s);
      if (io) io.observe(img); else visible.add(key);
      return s;
    }
    function detach(key) {
      const s = slots.get(key); if (!s) return;
      if (io) io.unobserve(s.img);
      visible.delete(key);
      if (s.url) URL.revokeObjectURL(s.url);
      slots.delete(key);
    }
    function update(key, run) {
      const s = slots.get(key); if (!s) return;
      s.stale = !!run.stale; s.hint = strip(run.frame_etag); s.hasFrame = run.has_frame !== false;
    }
    async function load(s) {
      s.inflight = true; s.last = performance.now();
      try {
        const headers = s.etag ? { 'If-None-Match': s.etag } : {};
        const r = await fetch(`/frame/${encodeURIComponent(s.runId)}.png`, { cache: 'no-store', headers });
        if (r.status === 200) {
          const tag = r.headers.get('ETag') || '';
          const blob = await r.blob();
          const url = URL.createObjectURL(blob);
          const old = s.url;
          s.img.onload = () => { if (old) URL.revokeObjectURL(old); s.img.classList.add('ready'); const ph = s.img.parentElement && s.img.parentElement.querySelector('.placeholder'); if (ph) ph.hidden = true; };
          s.img.src = url; s.url = url; s.etag = tag;
        }
      } catch (e) { /* network hiccup: try again next tick */ }
      finally { s.inflight = false; }
    }
    function tick() {
      if (document.hidden) return;
      const now = performance.now();
      for (const [key, s] of slots) {
        if (s.inflight || !s.hasFrame || !s.img.isConnected) continue;
        if (io && !visible.has(key)) continue;
        if (s.stale) { if (s.etag && strip(s.etag) === s.hint) continue; if (now - s.last < 2000) continue; }
        else if (now - s.last < s.period) continue;
        load(s);
      }
      for (const [key, s] of slots) if (!s.img.isConnected) detach(key);
    }
    setInterval(tick, 100);
    return { attach, detach, update };
  })();

  // ------------------------------------------------------------------------------- grid
  const cards = new Map(); // run_id -> {el, parts: {name: html}}
  function setPart(card, name, html) {
    if (card.parts[name] === html) return;
    card.parts[name] = html;
    const node = card.el.querySelector(`[data-part="${name}"]`);
    if (node) node.innerHTML = html;
  }
  function makeCard(run) {
    const el = document.createElement('article');
    el.className = 'run-card';
    el.setAttribute('data-run', run.run_id);
    el.innerHTML = `<div class="run-head"><div class="run-title"><h3 data-part="title"></h3><div class="agent" data-part="agent"></div></div>` +
      `<span data-part="tags" style="display:contents"></span>` +
      `<a class="icon-btn" href="#/run/${encodeURIComponent(run.run_id)}" aria-label="Ver en grande" data-tip="Ver esta partida en grande">` +
      `<svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true"><path d="M2 6V2h4M14 6V2h-4M2 10v4h4M14 10v4h-4" fill="none" stroke="currentColor" stroke-width="1.6"/></svg></a></div>` +
      `<div class="run-body"><div class="screen"><img alt="Pantalla del emulador"><div class="placeholder">Sin imagen todavía</div><div class="overlay" data-part="overlay"></div></div>` +
      `<div class="side" data-part="side"></div></div>` +
      `<div data-part="battle"></div><div class="stats" data-part="stats"></div>` +
      `<div class="run-foot"><div><div class="label">Hitos</div><ol class="timeline" data-part="timeline"></ol></div>` +
      `<div><div class="label">Eventos</div><ul class="events" data-part="events"></ul></div></div>`;
    const card = { el, parts: {} };
    frames.attach('grid:' + run.run_id, run.run_id, el.querySelector('img'), 300);
    return card;
  }
  function sideHtml(run) {
    if (run.kind === 'league') return `<div><div class="label">Objetivo</div><div class="goal${run.goal ? '' : ' empty'}">${esc(run.goal || 'Sin objetivo publicado')}</div></div>` + leagueSideHtml(run);
    return `<div><div class="label">Objetivo actual</div><div class="goal${run.goal ? '' : ' empty'}">${esc(run.goal || 'Sin objetivo publicado')}</div></div>` +
      `<div><div class="label">Medallas</div><div class="badges">${badgesHtml(run)}</div></div>` +
      `<div><div class="label">Equipo</div><div class="party">${partyHtml(run)}</div></div>`;
  }
  function fillCard(card, run) {
    card.el.style.setProperty('--game', U.gameColor(run.game));
    card.el.classList.toggle('stale', !!run.stale);
    const battle = battleHtml(run);
    card.el.classList.toggle('in-battle', !!battle);
    setPart(card, 'title', esc(U.runTitle(run)) + (run.kind ? ` <small>${esc(U.kindLabel(run.kind))}</small>` : ''));
    setPart(card, 'agent', esc(run.agent || run.run_id));
    setPart(card, 'tags', tagsHtml(run));
    setPart(card, 'overlay', `<span>${esc(fmtNum(run.fps, 0))} fps</span><span>${esc(fmtPlay(run.frames, run.platform || (run.meta || {}).platform))}</span>`);
    setPart(card, 'side', sideHtml(run));
    setPart(card, 'battle', battle);
    setPart(card, 'stats', statsHtml(run));
    setPart(card, 'timeline', timelineHtml(run));
    setPart(card, 'events', eventsHtml(run));
    const ph = card.el.querySelector('.placeholder');
    if (ph && run.has_frame === false) { ph.hidden = false; ph.textContent = 'Esta ejecución no publica imagen'; }
    frames.update('grid:' + run.run_id, run);
  }
  /** Render the live grid keyed by run_id (cards are reused so images do not flicker). */
  function renderGrid(grid, runs) {
    const seen = new Set();
    runs.forEach((run, i) => {
      seen.add(run.run_id);
      let card = cards.get(run.run_id);
      if (!card) { card = makeCard(run); cards.set(run.run_id, card); }
      fillCard(card, run);
      if (grid.children[i] !== card.el) grid.insertBefore(card.el, grid.children[i] || null);
    });
    for (const [id, card] of cards) if (!seen.has(id)) { card.el.remove(); frames.detach('grid:' + id); cards.delete(id); }
  }

  return { badgesHtml, partyHtml, battleHtml, statsHtml, timelineHtml, eventsHtml, tagsHtml, leagueSideHtml, renderGrid, frames };
})();
