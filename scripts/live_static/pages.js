/* ARES live dashboard - pages: run detail, progress, intelligence, league, knowledge. */
'use strict';

const P = (() => {
  const { $, esc, num, fmtInt, fmtNum, fmtCompact, fmtPct, fmtPlay, ago, runTitle, gameHex, emptyState, ttRow } = U;
  const PALETTE = ['#3987e5', '#d95926', '#199e70', '#9085e9', '#c98500', '#d55181', '#0ca30c', '#e66767', '#898781', '#96d9d6'];
  const METRIC = { milestones: 'Hitos', badges: 'Medallas', maps_visited: 'Mapas visitados' };
  const keepDemo = (rows, showDemo) => (rows || []).filter((r) => showDemo || !r.demo);

  /** Colour per run: the game's slot; a second run of the same game is drawn dashed. */
  function runStyles(runs) {
    const used = {}; const out = {};
    for (const r of runs) { const n = used[r.game] = (used[r.game] || 0) + 1; out[r.run_id] = { color: gameHex(r.game), dashed: n > 1 }; }
    return out;
  }
  const shortName = (r) => `${runTitle(r).replace(/^Pokémon /, '')}${r.kind && r.kind !== 'universal' ? ' (' + U.kindLabel(r.kind) + ')' : ''}${r.demo ? ' ·sim' : ''}`;
  const seg = (el, options, current, onPick) => {
    const buttons = options.map(([v, label]) => `<button type="button" data-v="${esc(v)}" class="${v === current ? 'on' : ''}" aria-pressed="${v === current}">${esc(label)}</button>`).join('');
    el.innerHTML = el.classList.contains('seg') ? buttons : `<div class="seg">${buttons}</div>`;
    el.querySelectorAll('button').forEach((b) => b.addEventListener('click', () => onPick(b.getAttribute('data-v'))));
  };
  const table = (head, rows) => `<table class="data"><thead><tr>${head.map(([h, n]) => `<th${n ? ' class="num"' : ''}>${esc(h)}</th>`).join('')}</tr></thead><tbody>` +
    rows.map((r) => `<tr>${r.map((c, i) => `<td${head[i] && head[i][1] ? ' class="num"' : ''}>${c}</td>`).join('')}</tr>`).join('') + '</tbody></table>';

  // ===================================================================================== run detail
  const detail = { runId: null, history: null, historyAt: 0 };
  function renderRun(section, run, history) {
    if (!run) {
      section.innerHTML = `<a class="back" href="#/">← Volver</a>` + emptyState('Ejecución no encontrada', 'No hay ninguna ejecución publicada con ese identificador (o está oculta por el filtro «simulados»).');
      return;
    }
    if (section.getAttribute('data-run') !== run.run_id) {
      section.setAttribute('data-run', run.run_id);
      section.innerHTML = `<div class="theatre-head"><a class="back" href="#/">← Todas</a><h1 data-p="title"></h1><span data-p="tags" style="display:contents"></span></div>` +
        `<div class="theatre"><div class="screen"><img alt="Pantalla del emulador"><div class="placeholder">Sin imagen todavía</div><div class="overlay" data-p="overlay"></div></div>` +
        `<div class="side" data-p="side"></div></div>` +
        `<div class="panel" style="margin-top:16px;padding:0;overflow:hidden"><div class="stats" data-p="stats" style="border:0"></div></div>` +
        `<div class="theatre-foot"><div class="panel"><h2>Progreso de esta partida</h2><div class="sub">Hitos, medallas y mapas frente a fotogramas</div><div class="legend" data-p="legend"></div><div class="chart" data-p="chart"></div></div>` +
        `<div class="panel"><h2>Hitos</h2><ol class="timeline" data-p="timeline"></ol></div>` +
        `<div class="panel"><h2>Eventos</h2><ul class="events" data-p="events"></ul></div></div>`;
      section._parts = {};
      L.frames.attach('detail', run.run_id, section.querySelector('img'), 200);
    }
    const set = (name, html) => { if (section._parts[name] === html) return; section._parts[name] = html; const n = section.querySelector(`[data-p="${name}"]`); if (n) n.innerHTML = html; };
    set('title', esc(runTitle(run)) + ` <small>${esc(run.run_id)}</small>`);
    set('tags', L.tagsHtml(run));
    set('overlay', `<span>${esc(fmtNum(run.fps, 0))} fps</span><span>${esc(fmtPlay(run.frames, run.platform || (run.meta || {}).platform))}</span>`);
    const goal = `<div><div class="label">Objetivo actual</div><div class="goal${run.goal ? '' : ' empty'}">${esc(run.goal || 'Sin objetivo publicado')}</div></div>`;
    set('side', run.kind === 'league' ? goal + L.leagueSideHtml(run)
      : goal + `<div><div class="label">Medallas</div><div class="badges">${L.badgesHtml(run)}</div></div>` + L.battleHtml(run, 10) +
        `<div><div class="label">Equipo</div><div class="party">${L.partyHtml(run)}</div></div>` +
        `<div class="empty-note">Agente: ${esc(run.agent || '—')} · publicado ${esc(ago(run.age_s))}</div>`);
    set('stats', L.statsHtml(run));
    set('timeline', L.timelineHtml(run, 40));
    set('events', L.eventsHtml(run, 20));
    L.frames.update('detail', run);
    if (history) {
      const key = String(history.total_rows) + ':' + (history.rows || []).length;
      if (section._histKey !== key || section._histW !== section.clientWidth) {
        section._histKey = key; section._histW = section.clientWidth;
        const rows = history.rows || [];
        const mk = (k, name, color) => ({ id: k, name, color, step: true, points: rows.map((r) => [r.frames, r[k]]) });
        const series = run.kind === 'league' ? [] : [mk('milestones', 'Hitos', '#3987e5'), mk('badges', 'Medallas', '#fab219'), mk('maps_visited', 'Mapas', '#0ca30c')];
        C.legend(section.querySelector('[data-p="legend"]'), series.map((s) => ({ name: s.name, color: s.color })));
        C.line(section.querySelector('[data-p="chart"]'), { series, height: 220, integerY: true, xLabel: 'fotogramas', emptyText: run.kind === 'league' ? 'La liga no publica progreso de historia' : 'Sin historial todavía' });
      }
    }
  }

  // ===================================================================================== progress
  const prog = { metric: U.store.get('progress.metric', 'milestones'), hidden: new Set(U.store.get('progress.hidden', [])) };
  function renderProgress(data, showDemo, rerender) {
    const runs = keepDemo(data && data.runs, showDemo).filter((r) => r.kind !== 'league');
    seg($('#progress-filters'), Object.entries(METRIC), prog.metric, (v) => { prog.metric = v; U.store.set('progress.metric', v); rerender(); });
    const chart = $('#race-chart'), legend = $('#race-legend');
    if (!runs.length) {
      legend.innerHTML = ''; $('#race-table').innerHTML = ''; $('#progress-multiples').innerHTML = '';
      $('#race-sub').textContent = '';
      chart.innerHTML = emptyState('Sin partidas con historial', 'Lanza una campaña o el agente universal; cada ejecución publica su history.jsonl.', ['python -m scripts.run_campaign --game firered --resume']);
      return;
    }
    const styles = runStyles(runs);
    $('#race-title').textContent = 'Carrera de progreso · ' + METRIC[prog.metric];
    $('#race-sub').textContent = `${runs.length} partidas · eje X: fotogramas emulados (60 ≈ 1 s de juego) · pulsa una partida para ocultarla`;
    C.toggles(legend, runs.map((r) => ({ id: r.run_id, name: shortName(r), color: styles[r.run_id].color, on: !prog.hidden.has(r.run_id) })), (id) => {
      if (prog.hidden.has(id)) prog.hidden.delete(id); else prog.hidden.add(id);
      U.store.set('progress.hidden', Array.from(prog.hidden)); rerender();
    });
    const series = runs.filter((r) => !prog.hidden.has(r.run_id)).map((r) => ({
      id: r.run_id, name: shortName(r), color: styles[r.run_id].color, dashed: styles[r.run_id].dashed, step: true,
      points: (r.rows || []).map((row) => [row.frames, row[prog.metric]]),
    }));
    C.line(chart, { series, height: 320, integerY: true, xLabel: 'fotogramas emulados', yFmt: (v) => fmtInt(v), emptyText: 'Todas las partidas están ocultas' });
    const last = (r) => (r.rows && r.rows.length ? r.rows[r.rows.length - 1] : {});
    $('#race-table').innerHTML = table([['Partida'], ['Tipo'], ['Fotogramas', 1], ['Tiempo de juego', 1], ['Medallas', 1], ['Hitos', 1], ['Mapas', 1], ['Fotogramas/hito', 1], ['Estado']],
      runs.map((r) => { const l = last(r); return [esc(shortName(r)), esc(U.kindLabel(r.kind)), fmtInt(l.frames), esc(fmtPlay(l.frames, (r.meta || {}).platform)), fmtInt(l.badges), fmtInt(l.milestones), fmtInt(l.maps_visited),
        num(l.milestones) ? fmtCompact(l.frames / l.milestones) : '—', r.stale ? 'inactiva' : 'en vivo']; }));
    const mult = $('#progress-multiples');
    const want = runs.map((r) => r.run_id).join('|');
    if (mult.getAttribute('data-runs') !== want) {
      mult.setAttribute('data-runs', want);
      mult.innerHTML = runs.map((r) => `<div class="panel" data-run="${esc(r.run_id)}"><h2><a href="#/run/${encodeURIComponent(r.run_id)}" class="back">${esc(shortName(r))}</a></h2><div class="sub"></div><div class="legend"></div><div class="chart"></div></div>`).join('');
    }
    for (const r of runs) {
      const panel = mult.querySelector(`[data-run="${CSS.escape(r.run_id)}"]`); if (!panel) continue;
      const rows = r.rows || []; const l = last(r);
      panel.querySelector('.sub').textContent = `${fmtInt(l.badges)} medallas · ${fmtInt(l.milestones)} hitos · ${fmtInt(l.maps_visited)} mapas · ${fmtPlay(l.frames, (r.meta || {}).platform)} de juego`;
      const s = [['milestones', 'Hitos', '#3987e5'], ['badges', 'Medallas', '#fab219'], ['maps_visited', 'Mapas', '#0ca30c']]
        .map(([k, name, color]) => ({ id: k, name, color, step: true, points: rows.map((row) => [row.frames, row[k]]) }));
      C.legend(panel.querySelector('.legend'), s.map((x) => ({ name: x.name, color: x.color })));
      C.line(panel.querySelector('.chart'), { series: s, height: 150, integerY: true, endLabels: false });
    }
  }

  // ===================================================================================== intelligence
  function renderIntel(data, showDemo) {
    const runs = keepDemo(data && data.runs, showDemo).filter((r) => r.kind !== 'league');
    const demoCount = runs.filter((r) => r.demo).length;
    $('#intel-demo').innerHTML = demoCount ? `<div class="demo-banner">${demoCount} de ${runs.length} ejecuciones son SIMULADAS (scripts.live_demo_publisher): sus números no miden ningún agente. Desmarca «simulados» para ver solo datos reales.</div>` : '';
    const tiles = $('#intel-tiles');
    if (!runs.length) {
      tiles.innerHTML = ''; $('#intel-narrative').hidden = true;
      for (const id of ['#intel-fpm', '#intel-wr', '#intel-stalls', '#intel-knowledge']) C.hbars($(id), { rows: [], emptyText: 'Sin ejecuciones todavía' });
      $('#intel-table').innerHTML = emptyState('Sin métricas', 'Aparecerán cuando una campaña o el agente universal publique su estado.', ['python -m scripts.run_campaign --game firered --resume']);
      $('#intel-wr-legend').innerHTML = ''; $('#intel-kn-legend').innerHTML = '';
      return;
    }
    const styles = runStyles(runs);
    const withFpm = runs.filter((r) => num(r.frames_per_milestone) !== null);
    const first = withFpm[0], latest = withFpm[withFpm.length - 1];
    const change = first && latest && first !== latest ? latest.frames_per_milestone / first.frames_per_milestone - 1 : null;
    let w = 0, l = 0; for (const r of runs) { w += num(r.wins) || 0; l += num(r.losses) || 0; }
    const sph = runs.filter((r) => num(r.stalls_per_hour) !== null);
    const meanSph = sph.length ? sph.reduce((a, r) => a + r.stalls_per_hour, 0) / sph.length : null;
    const lastKn = runs.filter((r) => num(r.knowledge) !== null).slice(-1)[0];
    const games = new Set(runs.map((r) => r.game));
    const tile = (label, value, d, cls, t) => `<div class="tile" data-tip="${esc(t)}"><div class="label">${esc(label)}</div><div class="v">${esc(value)}</div><div class="d ${cls || ''}">${esc(d || '')}</div></div>`;
    tiles.innerHTML =
      tile('Juegos jugados', fmtInt(games.size), `${runs.length} ejecuciones`, '', 'Juegos distintos con estado publicado') +
      tile('Fotogramas por hito', latest ? fmtCompact(latest.frames_per_milestone) : '—', change !== null ? `${change < 0 ? '↓' : '↑'} ${fmtPct(Math.abs(change))} frente al primer juego` : 'hace falta más de un juego', change === null ? '' : (change < 0 ? 'down-good' : 'up-bad'), 'Fotogramas emulados entre hitos de historia en el juego más reciente; menos es mejor') +
      tile('Victorias en combate', w + l ? fmtPct(w / (w + l)) : '—', `${fmtInt(w)} ganados · ${fmtInt(l)} perdidos`, '', 'Victorias / (victorias + derrotas), todas las ejecuciones visibles') +
      tile('Atascos por hora', meanSph !== null ? fmtNum(meanSph, 2) : '—', 'media por juego · menos es mejor', '', 'Episodios de atasco por hora de juego emulada') +
      tile('Conocimiento', lastKn ? fmtInt(lastKn.knowledge) : '—', lastKn ? `${fmtInt(lastKn.knowledge_inherited)} heredados · ${fmtInt(lastKn.knowledge_learned)} aprendidos (${U.runTitle(lastKn).replace(/^Pokémon /, '')})` : '', '', 'Elementos de conocimiento de la ejecución más reciente');
    const nar = $('#intel-narrative');
    if (change !== null) {
      nar.hidden = false;
      nar.innerHTML = `El primer juego medido (<b>${esc(runTitle(first))}</b>) necesitó <b>${esc(fmtCompact(first.frames_per_milestone))}</b> fotogramas por hito; el más reciente (<b>${esc(runTitle(latest))}</b>) necesita <b>${esc(fmtCompact(latest.frames_per_milestone))}</b>: ` +
        `<b>${esc(fmtPct(Math.abs(change)))} ${change < 0 ? 'menos' : 'más'}</b>.` + (demoCount ? ' <b>Incluye datos simulados.</b>' : '') +
        ' Los juegos tienen hitos de dificultad distinta, así que esta comparación es orientativa.';
    } else nar.hidden = true;
    const label = (r, i) => `${num(r.sequence) !== null ? r.sequence : i + 1}. ${shortName(r)}`;
    C.hbars($('#intel-fpm'), { rows: runs.map((r, i) => ({ label: label(r, i), value: r.frames_per_milestone, color: styles[r.run_id].color,
      tip: `<div class="tt-title">${esc(label(r, i))}</div>` + ttRow(styles[r.run_id].color, 'fotogramas por hito', fmtCompact(r.frames_per_milestone)) + ttRow('', 'hitos observados', fmtInt(r.milestones_observed)) + ttRow('', 'primer hito en', fmtCompact(r.first_milestone_frames)) })) });
    const wrRuns = runs.filter((r) => Array.isArray(r.win_curve) && r.win_curve.length);
    C.legend($('#intel-wr-legend'), wrRuns.map((r) => ({ name: shortName(r), color: styles[r.run_id].color, kind: styles[r.run_id].dashed ? 'dashed' : 'line' })));
    C.line($('#intel-wr'), { series: wrRuns.map((r) => ({ id: r.run_id, name: shortName(r), color: styles[r.run_id].color, dashed: styles[r.run_id].dashed,
      points: r.win_curve.map((p) => [p.frames, p.cumulative]) })), height: 240, yMin: 0, yMax: 1, yFmt: (v) => fmtPct(v), xLabel: 'fotogramas emulados', emptyText: 'Sin combates registrados todavía' });
    C.hbars($('#intel-stalls'), { rows: runs.map((r, i) => ({ label: label(r, i), value: r.stalls_per_hour, color: styles[r.run_id].color, xFmt: (v) => fmtNum(v, 2),
      tip: `<div class="tt-title">${esc(label(r, i))}</div>` + ttRow('', 'atascos', fmtInt(r.stalls)) + ttRow('', 'recuperados', fmtInt(r.recoveries)) + ttRow('', 'horas emuladas', fmtNum(r.emulated_hours, 2)) })),
      xFmt: (v) => fmtNum(v, 2) });
    C.legend($('#intel-kn-legend'), [{ name: 'heredado', color: '#898781', kind: 'box' }, { name: 'aprendido en el juego', color: '#3987e5', kind: 'box' }]);
    C.hbars($('#intel-knowledge'), { rows: runs.map((r, i) => ({ label: label(r, i), segments: [
      { value: r.knowledge_inherited, color: '#898781', name: 'heredado' }, { value: r.knowledge_learned, color: styles[r.run_id].color, name: 'aprendido' }], xFmt: fmtInt })), xFmt: (v) => fmtInt(v) });
    $('#intel-table').innerHTML = table([['#'], ['Partida'], ['Fotogramas', 1], ['Hitos', 1], ['Fot./hito', 1], ['Combates', 1], ['Victorias', 1], ['Atascos/h', 1], ['Heredado', 1], ['Aprendido', 1]],
      runs.map((r, i) => [esc(num(r.sequence) !== null ? r.sequence : i + 1), `<a class="back" href="#/run/${encodeURIComponent(r.run_id)}">${esc(shortName(r))}</a>`, fmtCompact(r.frames), fmtInt(r.milestones),
        fmtCompact(r.frames_per_milestone), fmtInt(r.battles), fmtPct(r.win_rate), fmtNum(r.stalls_per_hour, 2), fmtInt(r.knowledge_inherited), fmtInt(r.knowledge_learned)]));
  }

  // ===================================================================================== league
  const league = { source: U.store.get('league.source', 'report') };
  const TRAIT_ES = { agresividad: 'agresividad', cambios: 'cambios', estados: 'estados', mejoras: 'mejoras', riesgo: 'riesgo',
    aggression: 'agresividad', switching: 'cambios', status: 'estados', setup: 'mejoras', risk: 'riesgo' };
  function renderLeague(data, showDemo, rerender) {
    const body = $('#league-body'), src = $('#league-source');
    const sources = [];
    if (data && data.normalized) sources.push(['report', 'Informe de la liga', data.normalized, data]);
    for (const lv of keepDemo(data && data.live, showDemo)) if (lv.normalized) sources.push([lv.run_id, (lv.demo ? 'Simulada · ' : 'En vivo · ') + lv.run_id, lv.normalized, lv]);
    if (!sources.length) {
      src.innerHTML = '';
      body.innerHTML = emptyState('Todavía no hay informe de la liga', 'La página se rellena con artifacts/ares_u/league/report.json o con una liga que publique su estado en vivo.',
        ['python -m scripts.ares_u_league train', 'python -m scripts.ares_u_league report']);
      return;
    }
    if (!sources.some((s) => s[0] === league.source)) league.source = sources[0][0];
    seg(src, sources.map((s) => [s[0], s[1]]), league.source, (v) => { league.source = v; U.store.set('league.source', v); rerender(); });
    const [, , N, raw] = sources.find((s) => s[0] === league.source);
    const rivals = Array.isArray(N.rivals) ? N.rivals : [];
    const champ = N.champion || null;
    const vs = N.vs_e1;
    let vsHtml = '<div class="empty-note">Sin evaluación del campeón frente a E1.</div>';
    if (vs !== null && vs !== undefined) {
      const wr = typeof vs === 'object' ? num(vs.win_rate !== undefined ? vs.win_rate : vs.winrate) : num(vs);
      const n = typeof vs === 'object' ? num(vs.battles !== undefined ? vs.battles : vs.games) : null;
      if (wr !== null) {
        const ci = typeof vs === 'object' && Array.isArray(vs.ci) ? vs.ci.map(num) : U.wilson(wr, n);
        vsHtml = `<div><b style="font-size:20px">${esc(fmtPct(wr, 1))}</b> de victorias del campeón contra el cerebro E1${n ? ` en ${esc(fmtInt(n))} combates` : ''}` +
          `${ci ? ` · IC 95 % ${esc(fmtPct(ci[0], 1))} – ${esc(fmtPct(ci[1], 1))}` : ''}</div>` +
          `<div class="ci" data-tip="La línea central es el 50 %: a la derecha, el campeón supera a E1">${ci ? `<div class="range" style="left:${(ci[0] * 100).toFixed(1)}%;width:${((ci[1] - ci[0]) * 100).toFixed(1)}%"></div>` : ''}` +
          `<div class="mid"></div><div class="point" style="left:calc(${(wr * 100).toFixed(1)}% - 2px)"></div></div>` +
          (typeof vs === 'object' && vs.note ? `<div class="empty-note">${esc(vs.note)}</div>` : '');
      } else vsHtml = `<div class="empty-note">${esc(typeof vs === 'object' ? JSON.stringify(vs) : vs)}</div>`;
    }
    const meta = N.meta || {};
    const metaBits = Object.entries(meta).slice(0, 6).map(([k, v]) => `${esc(k)}: <b>${esc(typeof v === 'number' ? fmtInt(v) : v)}</b>`).join(' · ');
    const demoNote = raw && raw.demo ? '<div class="demo-banner">Liga SIMULADA por scripts.live_demo_publisher: no son resultados reales.</div>' : '';
    const fileNote = league.source === 'report' && data.report_path ? `<div class="empty-note">Fuente: ${esc(data.report_path)}${data.report_mtime ? ' · ' + esc(new Date(data.report_mtime * 1000).toLocaleString('es-ES')) : ''}</div>` : '';
    body.innerHTML = demoNote +
      `<div class="panels"><div class="panel"><h2>Campeón</h2><div class="champion" style="margin-top:10px"><div class="crown" aria-hidden="true">♛</div><div><div class="who">${esc(champ ? champ.name : '—')}</div>` +
      `<div class="empty-note">${champ && num(champ.elo) !== null ? fmtInt(champ.elo) + ' Elo' : ''}${champ && champ.inferred ? ' · deducido del Elo' : ''}</div></div></div>` +
      `<h2 style="margin-top:16px">Campeón frente a E1</h2>${vsHtml}<div class="empty-note" style="margin-top:8px">${metaBits}</div>${fileNote}</div>` +
      `<div class="panel"><h2>Tabla Elo</h2><div class="table-scroll" id="league-elo"></div></div>` +
      `<div class="panel span-all"><h2>Rivales y su estilo de combate</h2><div class="sub">Perfil de comportamiento medido (0 – 1)</div><div class="rivals" id="league-rivals"></div></div>` +
      `<div class="panel span-all"><h2>Evolución del Elo</h2><div class="legend" id="league-elo-legend"></div><div class="chart" id="league-elo-chart"></div></div>` +
      `<div class="panel span-all"><h2>Cara a cara</h2><div class="sub">Tasa de victorias de la fila contra la columna</div><div class="heat-legend" id="league-heat-legend"></div><div class="chart" id="league-heat"></div></div></div>`;
    $('#league-elo').innerHTML = rivals.length ? table([['#'], ['Rival'], ['Elo', 1], ['Combates', 1], ['Victorias', 1]],
      rivals.map((r, i) => [String(i + 1), esc(r.name) + (champ && champ.name === r.name ? ' ♛' : ''), fmtInt(r.elo), fmtInt(r.games), fmtPct(r.win_rate)])) : '<div class="empty-note">Sin rivales</div>';
    $('#league-rivals').innerHTML = rivals.map((r) => {
      const prof = r.profile && typeof r.profile === 'object' ? Object.entries(r.profile).filter(([, v]) => num(v) !== null) : [];
      return `<div class="rival"><h3>${esc(r.name)}<span>${fmtInt(r.elo)}</span></h3><p>${esc(r.description || 'Sin descripción')}</p>` +
        prof.map(([k, v]) => { const x = Math.max(0, Math.min(1, v > 1 ? v / 100 : v)); return `<div class="trait"><span>${esc(TRAIT_ES[k] || k.replace(/_/g, ' '))}</span><div class="bar"><i style="width:${(x * 100).toFixed(0)}%"></i></div><span class="v">${esc(fmtNum(v, 2))}</span></div>`; }).join('') + '</div>';
    }).join('') || '<div class="empty-note">Sin perfiles</div>';
    const colorOf = {}; rivals.forEach((r, i) => { colorOf[r.name] = PALETTE[i % PALETTE.length]; });
    const hist = N.elo_history;
    if (hist && Array.isArray(hist.series) && Array.isArray(hist.x)) {
      const series = hist.series.map((s, i) => ({ id: s.name, name: s.name, color: colorOf[s.name] || PALETTE[i % PALETTE.length],
        points: s.values.map((v, j) => [hist.x[j], v]).filter((p) => num(p[1]) !== null) }));
      C.legend($('#league-elo-legend'), series.map((s) => ({ name: s.name, color: s.color })));
      C.line($('#league-elo-chart'), { series, height: 260, xLabel: 'generación', yMin: Math.min(...series.flatMap((s) => s.points.map((p) => p[1]))) - 20, yFmt: (v) => fmtInt(v), xFmt: (v) => fmtInt(v) });
    } else C.line($('#league-elo-chart'), { series: [], height: 120, emptyText: 'El informe no incluye historial de Elo' });
    const M = N.matrix;
    if (M && Array.isArray(M.names) && Array.isArray(M.values)) {
      C.legend($('#league-heat-legend'), [{ name: '0 % (pierde siempre)', color: '#d03b3b', kind: 'box' }, { name: '50 %', color: '#2a2a27', kind: 'box' }, { name: '100 % (gana siempre)', color: '#3987e5', kind: 'box' }]);
      C.heat($('#league-heat'), { rows: M.names, cols: M.names, rowLabelWidth: 110, cell: (i, j) => {
        const v = num((M.values[i] || [])[j]);
        if (v === null) return i === j ? { fill: '#1a1a19', text: '—', textColor: '#898781' } : null;
        return { fill: C.ramp(v), text: fmtNum(v * 100, 0), tip: `<div class="tt-title">${esc(M.names[i])} contra ${esc(M.names[j])}</div>` + ttRow('', M.metric === 'win_rate' ? 'victorias de la fila' : M.metric, fmtPct(v, 1)) };
      } });
    } else C.heat($('#league-heat'), { rows: [], cols: [], emptyText: 'El informe no incluye la matriz de cara a cara' });
  }

  // ===================================================================================== knowledge
  const REAL = {
    normal: { rock: 0.5, steel: 0.5, ghost: 0 }, fire: { grass: 2, ice: 2, bug: 2, steel: 2, fire: 0.5, water: 0.5, rock: 0.5, dragon: 0.5 },
    water: { fire: 2, ground: 2, rock: 2, water: 0.5, grass: 0.5, dragon: 0.5 }, electric: { water: 2, flying: 2, electric: 0.5, grass: 0.5, dragon: 0.5, ground: 0 },
    grass: { water: 2, ground: 2, rock: 2, fire: 0.5, grass: 0.5, poison: 0.5, flying: 0.5, bug: 0.5, dragon: 0.5, steel: 0.5 },
    ice: { grass: 2, ground: 2, flying: 2, dragon: 2, fire: 0.5, water: 0.5, ice: 0.5, steel: 0.5 },
    fighting: { normal: 2, ice: 2, rock: 2, dark: 2, steel: 2, poison: 0.5, flying: 0.5, psychic: 0.5, bug: 0.5, ghost: 0 },
    poison: { grass: 2, poison: 0.5, ground: 0.5, rock: 0.5, ghost: 0.5, steel: 0 }, ground: { fire: 2, electric: 2, poison: 2, rock: 2, steel: 2, grass: 0.5, bug: 0.5, flying: 0 },
    flying: { grass: 2, fighting: 2, bug: 2, electric: 0.5, rock: 0.5, steel: 0.5 }, psychic: { fighting: 2, poison: 2, psychic: 0.5, steel: 0.5, dark: 0 },
    bug: { grass: 2, psychic: 2, dark: 2, fire: 0.5, fighting: 0.5, poison: 0.5, flying: 0.5, ghost: 0.5, steel: 0.5 },
    rock: { fire: 2, ice: 2, flying: 2, bug: 2, fighting: 0.5, ground: 0.5, steel: 0.5 }, ghost: { psychic: 2, ghost: 2, dark: 0.5, steel: 0.5, normal: 0 },
    dragon: { dragon: 2, steel: 0.5 }, dark: { psychic: 2, ghost: 2, fighting: 0.5, dark: 0.5, steel: 0.5 },
    steel: { ice: 2, rock: 2, fire: 0.5, water: 0.5, electric: 0.5, steel: 0.5 },
  };
  const GEN1 = { 'bug>poison': 2, 'poison>bug': 2, 'ghost>psychic': 0, 'ice>fire': 1 };
  /** Real multiplier, generations 1-5 (same table as scripts/live_dashboard.py real_multiplier). */
  function realMult(a, d, gen) {
    if (gen <= 1 && GEN1[a + '>' + d] !== undefined) return GEN1[a + '>' + d];
    const v = (REAL[a] || {})[d];
    return v === undefined ? 1 : v;
  }
  const MULT_FILL = (v) => (v === 0 ? '#0d0d0d' : v < 1 ? '#8a2f2f' : v > 1 ? '#1f7a3a' : '#2a2a27');
  const multText = (v) => (v === 0 ? '0' : v === 0.5 ? '½' : v === 0.25 ? '¼' : fmtNum(v, 2));
  const know = { source: U.store.get('knowledge.source', '') };
  function renderKnowledge(data, showDemo, rerender) {
    const body = $('#knowledge-body'), srcEl = $('#knowledge-sources');
    const sources = [];
    if (data && (data.universal_type_chart || data.universal_summary)) sources.push(['universal', 'Universal (knowledge.json)', data.universal_summary, data.universal_type_chart, data.universal_path]);
    if (data && (data.legacy_type_chart || data.legacy_summary)) sources.push(['legacy', 'Campañas Rojo Fuego / Esmeralda', data.legacy_summary, data.legacy_type_chart, data.legacy_path]);
    for (const r of keepDemo(data && data.runs, showDemo)) sources.push([r.run_id, `${r.demo ? 'Simulada · ' : ''}${runTitle(r).replace(/^Pokémon /, '')} (${r.run_id})`, r.summary, r.type_chart, null, r]);
    if (!sources.length) {
      srcEl.innerHTML = '';
      body.innerHTML = emptyState('Todavía no hay conocimiento guardado', 'Se rellena con artifacts/universal/knowledge.json, artifacts/campaigns/native/shared_knowledge.json o el campo knowledge de cada ejecución.',
        ['python -m scripts.run_campaign --game firered --resume']);
      return;
    }
    if (!sources.some((s) => s[0] === know.source)) know.source = sources[0][0];
    seg(srcEl, sources.map((s) => [s[0], s[1]]), know.source, (v) => { know.source = v; U.store.set('knowledge.source', v); rerender(); });
    const [, label, summary, tc, path, run] = sources.find((s) => s[0] === know.source);
    const cats = summary && summary.categories && typeof summary.categories === 'object' ? Object.entries(summary.categories) : [];
    const tile = (l, v, d) => `<div class="tile"><div class="label">${esc(l)}</div><div class="v">${esc(v)}</div><div class="d">${esc(d || '')}</div></div>`;
    const acc = tc && tc.checked ? tc.correct / tc.checked : null;
    body.innerHTML = (run && run.demo ? '<div class="demo-banner">Conocimiento SIMULADO por scripts.live_demo_publisher.</div>' : '') +
      `<div class="tiles">${tile('Elementos', fmtInt(summary && summary.items), summary && num(summary.inherited) !== null ? fmtInt(summary.inherited) + ' heredados' : esc(label))}` +
      tile('Celdas de la tabla de tipos', tc ? fmtInt(tc.known) : '—', tc ? `de ${fmtInt((tc.types || []).length ** 2)} posibles (Gen ${tc.generation || '?'})` : 'sin tabla de tipos') +
      tile('Aciertos frente a la tabla real', acc !== null ? fmtPct(acc, 1) : '—', tc ? `${fmtInt(tc.correct)} de ${fmtInt(tc.checked)} comprobadas` : '') +
      tile('Hechos de doble tipo', tc ? fmtInt(tc.dual_facts) : '—', tc && tc.dual_facts ? `${fmtInt(tc.dual_correct)} correctos` : '') + '</div>' +
      `<div class="panels"><div class="panel"><h2>Tabla de tipos aprendida</h2><div class="sub">Ataque (fila) contra defensa (columna); vacío = sin observar; borde rojo = distinto de la tabla real</div><div class="heat-legend" id="kn-legend"></div><div class="chart" id="kn-learned"></div></div>` +
      `<div class="panel"><h2>Tabla de tipos real</h2><div class="sub">Referencia de los juegos (Gen ${esc(tc && tc.generation ? tc.generation : 3)})</div><div class="heat-legend" id="kn-legend2"></div><div class="chart" id="kn-real"></div></div>` +
      `<div class="panel span-all"><h2>Contenido por categoría</h2>${path ? `<div class="sub">${esc(path)}</div>` : ''}<div class="chart" id="kn-cats"></div></div></div>`;
    const legendItems = [{ name: '×0', color: '#0d0d0d', kind: 'box' }, { name: '×½', color: '#8a2f2f', kind: 'box' }, { name: '×1', color: '#2a2a27', kind: 'box' }, { name: '×2', color: '#1f7a3a', kind: 'box' }];
    C.legend($('#kn-legend'), legendItems); C.legend($('#kn-legend2'), legendItems);
    if (tc && Array.isArray(tc.types) && tc.types.length) {
      const types = tc.types, gen = tc.generation || 3;
      const byKey = {}; for (const c of tc.cells || []) byKey[c.atk + '>' + c.def] = c;
      const names = types.map((t) => U.typeName(t));
      C.heat($('#kn-learned'), { rows: names, cols: names, maxCell: 30, rowLabelWidth: 78, cell: (i, j) => {
        const c = byKey[types[i] + '>' + types[j]]; if (!c || num(c.mult) === null) return null;
        const ok = c.correct !== false;
        return { fill: MULT_FILL(c.mult), text: multText(c.mult), stroke: ok ? null : '#ff6b6b',
          tip: `<div class="tt-title">${esc(names[i])} → ${esc(names[j])}</div>` + ttRow('', 'aprendido', '×' + fmtNum(c.mult, 2)) + ttRow('', 'real', c.real === null || c.real === undefined ? '—' : '×' + fmtNum(c.real, 2)) + ttRow('', ok ? 'coincide' : 'NO coincide', ok ? '✓' : '✗') + (c.source ? ttRow('', 'fuente', c.source) : '') };
      } });
      C.heat($('#kn-real'), { rows: names, cols: names, maxCell: 30, rowLabelWidth: 78, cell: (i, j) => {
        const v = realMult(types[i], types[j], gen); const learned = byKey[types[i] + '>' + types[j]];
        return { fill: MULT_FILL(v), text: v === 1 ? '' : multText(v), stroke: learned ? null : null,
          tip: `<div class="tt-title">${esc(names[i])} → ${esc(names[j])}</div>` + ttRow('', 'real', '×' + fmtNum(v, 2)) + ttRow('', learned ? 'aprendido' : 'sin aprender', learned ? '×' + fmtNum(learned.mult, 2) : '—') };
      } });
    } else {
      C.heat($('#kn-learned'), { rows: [], cols: [], emptyText: 'Esta fuente no contiene tabla de tipos' });
      C.heat($('#kn-real'), { rows: [], cols: [], emptyText: '—' });
    }
    C.hbars($('#kn-cats'), { rows: cats.map(([k, v], i) => ({ label: k.replace(/_/g, ' '), value: v, color: PALETTE[i % PALETTE.length] })), xFmt: (v) => fmtInt(v), emptyText: 'Sin categorías' });
  }

  return { renderRun, renderProgress, renderIntel, renderLeague, renderKnowledge, realMult, detail };
})();
