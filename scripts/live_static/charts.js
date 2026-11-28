/* ARES live dashboard - tiny hand-drawn SVG charts (line/step, horizontal bars, heatmap). No dependencies. */
'use strict';

const C = (() => {
  const { esc, num, fmtCompact, tip, ttRow } = U;
  const NS = 'http://www.w3.org/2000/svg';

  function niceStep(span, count) {
    const raw = span / Math.max(1, count);
    const p = Math.pow(10, Math.floor(Math.log10(raw)));
    const f = raw / p;
    return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 2.5 ? 2.5 : f <= 5 ? 5 : 10) * p;
  }
  /** Rounded domain and ticks covering [lo, hi]. */
  function nice(lo, hi, count = 5, integer = false) {
    if (!(hi > lo)) { hi = lo + 1; }
    let step = niceStep(hi - lo, count);
    if (integer) step = Math.max(1, Math.round(step));
    const a = Math.floor(lo / step) * step, b = Math.ceil(hi / step) * step;
    const ticks = [];
    for (let v = a; v <= b + step * 1e-9; v += step) ticks.push(Math.abs(v) < step * 1e-9 ? 0 : v);
    return { lo: a, hi: b, ticks };
  }
  const width = (el) => Math.max(260, Math.floor((el && el.clientWidth) || 600));
  const empty = (el, h, text) => {
    el.innerHTML = `<svg viewBox="0 0 ${width(el)} ${h}" height="${h}"><text class="empty" x="50%" y="${h / 2}" text-anchor="middle">${esc(text || 'Sin datos todavía')}</text></svg>`;
  };
  const lerp = (a, b, t) => a + (b - a) * t;
  const hex = (h) => { const s = h.replace('#', ''); return [0, 2, 4].map((i) => parseInt(s.substr(i, 2), 16)); };
  /** Colour from a diverging ramp (lo -> mid -> hi) for t in [0, 1]. */
  function ramp(t, lo = '#d03b3b', mid = '#2a2a27', hi = '#3987e5') {
    if (t === null || !Number.isFinite(t)) return 'transparent';
    t = Math.max(0, Math.min(1, t));
    const [a, b, u] = t < 0.5 ? [hex(lo), hex(mid), t * 2] : [hex(mid), hex(hi), (t - 0.5) * 2];
    return `rgb(${a.map((v, i) => Math.round(lerp(v, b[i], u))).join(',')})`;
  }

  /**
   * Line / step chart.
   * opts: series [{id, name, color, points: [[x, y]], dashed, step, area}], height, xFmt, yFmt, yMin, yMax,
   *       xLabel, yLabel, endLabels, integerY, tipTitle(x)
   */
  function line(el, opts) {
    const H = opts.height || 240;
    const series = (opts.series || []).map((s) => ({ ...s, points: (s.points || []).filter((p) => num(p[0]) !== null && num(p[1]) !== null) }))
      .filter((s) => s.points.length);
    if (!series.length) return empty(el, H, opts.emptyText);
    const W = width(el);
    const m = { l: 50, r: opts.endLabels === false ? 14 : 104, t: 10, b: opts.xLabel ? 40 : 28 };
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (const s of series) for (const [x, y] of s.points) { x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y); }
    if (opts.yMin !== undefined) y0 = opts.yMin; else y0 = Math.min(0, y0);
    if (opts.yMax !== undefined) y1 = Math.max(opts.yMax, opts.yMaxIsFloor ? y1 : opts.yMax);
    if (x1 <= x0) x1 = x0 + 1;
    const ny = nice(y0, y1 > y0 ? y1 : y0 + 1, 4, !!opts.integerY);
    if (opts.yMax !== undefined && !opts.yMaxIsFloor) { ny.hi = opts.yMax; ny.ticks = ny.ticks.filter((t) => t <= opts.yMax + 1e-9); }
    const nx = nice(x0, x1, Math.max(2, Math.floor((W - m.l - m.r) / 110)));
    const X = (x) => m.l + (x - x0) / (x1 - x0) * (W - m.l - m.r);
    const Y = (y) => m.t + (1 - (y - ny.lo) / (ny.hi - ny.lo)) * (H - m.t - m.b);
    const xFmt = opts.xFmt || fmtCompact, yFmt = opts.yFmt || fmtCompact;
    let g = '<g class="grid">';
    for (const t of ny.ticks) g += `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(t)}" y2="${Y(t)}"/>`;
    g += '</g><g class="axis">';
    for (const t of ny.ticks) g += `<text x="${m.l - 8}" y="${Y(t) + 4}" text-anchor="end">${esc(yFmt(t))}</text>`;
    for (const t of nx.ticks) if (t >= x0 - 1e-9 && t <= x1 + 1e-9) g += `<text x="${X(t)}" y="${H - m.b + 16}" text-anchor="middle">${esc(xFmt(t))}</text>`;
    g += '</g>';
    g += `<line class="base" x1="${m.l}" x2="${W - m.r}" y1="${Y(ny.lo)}" y2="${Y(ny.lo)}"/>`;
    if (opts.xLabel) g += `<text class="xlabel" x="${(m.l + W - m.r) / 2}" y="${H - 4}" text-anchor="middle">${esc(opts.xLabel)}</text>`;
    let paths = '';
    const labels = [];
    for (const s of series) {
      let d = '';
      s.points.forEach(([x, y], i) => {
        if (i === 0) d += `M${X(x).toFixed(1)},${Y(y).toFixed(1)}`;
        else if (s.step) d += `H${X(x).toFixed(1)}V${Y(y).toFixed(1)}`;
        else d += `L${X(x).toFixed(1)},${Y(y).toFixed(1)}`;
      });
      const last = s.points[s.points.length - 1];
      if (s.extendTo !== undefined && s.extendTo > last[0]) d += `H${X(Math.min(s.extendTo, x1)).toFixed(1)}`;
      const dash = s.dashed ? ' stroke-dasharray="5 4"' : '';
      if (s.area) paths += `<path class="area" fill="${s.color}" d="${d}V${Y(ny.lo)}H${X(s.points[0][0])}Z"/>`;
      paths += `<path class="line" stroke="${s.color}"${dash} d="${d}"/>`;
      labels.push({ y: Y(last[1]), text: s.name, color: s.color, x: X(s.extendTo !== undefined ? Math.min(s.extendTo, x1) : last[0]) });
    }
    let endl = '';
    if (opts.endLabels !== false) {
      labels.sort((a, b) => a.y - b.y);
      for (let i = 1; i < labels.length; i++) if (labels[i].y - labels[i - 1].y < 13) labels[i].y = labels[i - 1].y + 13;
      for (const l of labels) {
        const t = l.text.length > 15 ? l.text.slice(0, 14) + '…' : l.text;
        endl += `<circle cx="${l.x}" cy="${l.y}" r="0"/><text class="endlabel" x="${W - m.r + 8}" y="${l.y + 4}" fill="${l.color}">${esc(t)}</text>`;
      }
    }
    el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" height="${H}" role="img" aria-label="${esc(opts.ariaLabel || 'gráfico de líneas')}">${g}<g class="series">${paths}</g>${endl}` +
      `<line class="crosshair" x1="0" x2="0" y1="${m.t}" y2="${H - m.b}" visibility="hidden"/>` +
      `<rect x="${m.l}" y="${m.t}" width="${W - m.l - m.r}" height="${H - m.t - m.b}" fill="transparent"/></svg>`;
    const svg = el.querySelector('svg'), cross = svg.querySelector('.crosshair'), hit = svg.querySelector('rect');
    hit.addEventListener('mousemove', (ev) => {
      const r = svg.getBoundingClientRect();
      const px = (ev.clientX - r.left) * (W / r.width);
      const xv = x0 + (px - m.l) / (W - m.l - m.r) * (x1 - x0);
      cross.setAttribute('x1', px); cross.setAttribute('x2', px); cross.setAttribute('visibility', 'visible');
      const rows = [];
      for (const s of series) {
        let best = null;
        if (s.step) { for (const p of s.points) { if (p[0] <= xv) best = p; else break; } }
        else { let dist = Infinity; for (const p of s.points) { const dd = Math.abs(p[0] - xv); if (dd < dist) { dist = dd; best = p; } } }
        if (best && (s.step ? true : true)) rows.push([s.color, s.name, yFmt(best[1]), best[1]]);
      }
      rows.sort((a, b) => b[3] - a[3]);
      const title = opts.tipTitle ? opts.tipTitle(xv) : (opts.xLabel ? opts.xLabel + ': ' : '') + xFmt(xv);
      tip.show(`<div class="tt-title">${esc(title)}</div>` + rows.slice(0, 12).map((r) => ttRow(r[0], r[1], r[2])).join(''), ev.clientX, ev.clientY);
    });
    hit.addEventListener('mouseleave', () => { cross.setAttribute('visibility', 'hidden'); tip.hide(); });
  }

  /**
   * Horizontal bars (optionally stacked).
   * opts: rows [{label, color, value, segments: [{value, color, name}], tip, note}], xFmt, xMax, rowHeight, labelWidth
   */
  function hbars(el, opts) {
    const rows = (opts.rows || []).filter((r) => r);
    const rh = opts.rowHeight || 28;
    if (!rows.length) return empty(el, 120, opts.emptyText);
    const W = width(el), lw = Math.min(opts.labelWidth || 170, Math.floor(W * 0.4)), right = 70;
    const H = rows.length * rh + 24;
    const total = (r) => (r.segments ? r.segments.reduce((a, s) => a + (num(s.value) || 0), 0) : (num(r.value) || 0));
    const xMax = opts.xMax || Math.max(1e-9, ...rows.map(total));
    const nx = nice(0, xMax, Math.max(2, Math.floor((W - lw - right) / 120)));
    const hi = opts.xMax ? opts.xMax : nx.hi;
    const X = (v) => lw + v / hi * (W - lw - right);
    const xFmt = opts.xFmt || fmtCompact;
    let s = '<g class="grid">';
    for (const t of nx.ticks) if (t <= hi + 1e-9) s += `<line x1="${X(t)}" x2="${X(t)}" y1="0" y2="${rows.length * rh}"/>`;
    s += '</g><g class="axis">';
    for (const t of nx.ticks) if (t <= hi + 1e-9) s += `<text x="${X(t)}" y="${rows.length * rh + 16}" text-anchor="middle">${esc(xFmt(t))}</text>`;
    s += '</g>';
    rows.forEach((r, i) => {
      const y = i * rh, bh = Math.max(8, rh - 10);
      const label = r.label.length > 24 ? r.label.slice(0, 23) + '…' : r.label;
      s += `<text class="catlabel" x="${lw - 8}" y="${y + rh / 2 + 4}" text-anchor="end">${esc(label)}</text>`;
      const segs = r.segments || [{ value: r.value, color: r.color, name: r.label }];
      let acc = 0;
      segs.forEach((sg, j) => {
        const v = num(sg.value) || 0; if (v <= 0) return;
        const isLast = j === segs.length - 1;
        const x = X(acc), w = Math.max(1, X(acc + v) - x - (isLast ? 0 : 2));
        const tipHtml = r.tip || `<div class="tt-title">${esc(r.label)}</div>` + segs.map((q) => ttRow(q.color, q.name || '', xFmt(q.value))).join('');
        s += `<rect class="bar" data-i="${i}" x="${x}" y="${y + (rh - bh) / 2}" width="${w}" height="${bh}" rx="3" fill="${sg.color}"/>`;
        acc += v;
        r._tip = tipHtml;
      });
      const shown = r.note !== undefined ? r.note : (r.value === null || r.value === undefined) && !r.segments ? '—' : xFmt(total(r));
      s += `<text class="barlabel" x="${X(acc) + 6}" y="${y + rh / 2 + 4}">${esc(shown)}</text>`;
    });
    el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" height="${H}" role="img" aria-label="${esc(opts.ariaLabel || 'gráfico de barras')}">${s}</svg>`;
    el.querySelectorAll('rect.bar').forEach((b) => {
      const r = rows[Number(b.getAttribute('data-i'))];
      b.addEventListener('mousemove', (ev) => tip.show(r._tip, ev.clientX, ev.clientY));
      b.addEventListener('mouseleave', () => tip.hide());
    });
  }

  /**
   * Heatmap. opts: rows [labels], cols [labels], cell(r, c) -> {fill, text, tip} | null, cell size, rowLabelWidth.
   */
  function heat(el, opts) {
    const rows = opts.rows || [], cols = opts.cols || [];
    if (!rows.length || !cols.length) return empty(el, 120, opts.emptyText);
    const W = width(el);
    const lw = opts.rowLabelWidth || 96, top = opts.colLabelHeight || 74;
    const cs = Math.max(14, Math.min(opts.maxCell || 48, Math.floor((W - lw - 8) / cols.length)));
    const H = top + rows.length * cs + 4, Wd = lw + cols.length * cs + 4;
    let s = '';
    cols.forEach((c, j) => {
      const x = lw + j * cs + cs / 2;
      s += `<text class="catlabel" transform="translate(${x + 4},${top - 6}) rotate(-55)" font-size="11">${esc(c)}</text>`;
    });
    const tips = [];
    rows.forEach((r, i) => {
      s += `<text class="catlabel" x="${lw - 6}" y="${top + i * cs + cs / 2 + 4}" text-anchor="end" font-size="11">${esc(r)}</text>`;
      cols.forEach((c, j) => {
        const v = opts.cell(i, j);
        const x = lw + j * cs, y = top + i * cs;
        if (!v) { s += `<rect x="${x + 1}" y="${y + 1}" width="${cs - 2}" height="${cs - 2}" rx="3" fill="none" stroke="#2c2c2a"/>`; return; }
        const k = tips.push(v.tip || '') - 1;
        s += `<rect class="cell" data-k="${k}" x="${x + 1}" y="${y + 1}" width="${cs - 2}" height="${cs - 2}" rx="3" fill="${v.fill}"${v.stroke ? ` stroke="${v.stroke}" stroke-width="2"` : ''}/>`;
        if (v.text && cs >= 22) s += `<text x="${x + cs / 2}" y="${y + cs / 2 + 4}" text-anchor="middle" font-size="${cs >= 34 ? 11 : 9}" fill="${v.textColor || '#f0efec'}" pointer-events="none">${esc(v.text)}</text>`;
      });
    });
    el.innerHTML = `<svg viewBox="0 0 ${Math.max(W, Wd)} ${H}" height="${H}" style="max-width:${Math.max(W, Wd)}px" role="img" aria-label="${esc(opts.ariaLabel || 'mapa de calor')}">${s}</svg>`;
    el.querySelectorAll('rect.cell').forEach((b) => {
      const t = tips[Number(b.getAttribute('data-k'))];
      if (!t) return;
      b.addEventListener('mousemove', (ev) => tip.show(t, ev.clientX, ev.clientY));
      b.addEventListener('mouseleave', () => tip.hide());
    });
  }

  /** Legend made of toggle chips. items [{id, name, color, on}], onToggle(id). */
  function toggles(el, items, onToggle) {
    el.innerHTML = items.map((it) => `<button type="button" class="chip-toggle${it.on ? ' on' : ''}" data-id="${esc(it.id)}" aria-pressed="${it.on}">` +
      `<span class="key" style="background:${it.color}"></span>${esc(it.name)}</button>`).join('');
    el.querySelectorAll('button').forEach((b) => b.addEventListener('click', () => onToggle(b.getAttribute('data-id'))));
  }
  /** Static legend. items [{name, color, kind: 'line'|'box'|'dashed'}]. */
  function legend(el, items) {
    el.innerHTML = items.map((it) => `<span><i class="${it.kind === 'box' ? 'box' : it.kind === 'dashed' ? 'dashed' : ''}" style="${it.kind === 'dashed' ? 'color' : 'background'}:${it.color}"></i>${esc(it.name)}</span>`).join('');
  }

  return { nice, line, hbars, heat, toggles, legend, ramp, NS };
})();
