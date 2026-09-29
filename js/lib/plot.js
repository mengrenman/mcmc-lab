// A small canvas plotting layer: scales, axes, a handful of marks and a hover tooltip.
// Charts read their colors from CSS tokens, so they follow the light/dark theme.

let paletteCache = null;

export function palette() {
  if (paletteCache) return paletteCache;
  const cs = getComputedStyle(document.documentElement);
  const v = (name) => cs.getPropertyValue(name).trim();
  paletteCache = {
    surface: v('--surface'), surface2: v('--surface-2'), ink: v('--ink'), ink2: v('--ink-2'),
    muted: v('--muted'), grid: v('--grid'), axis: v('--axis'),
    s1: v('--s1'), s2: v('--s2'), s3: v('--s3'), exact: v('--exact'),
    good: v('--good'), bad: v('--bad'), seqHi: v('--seq-hi'),
    spinUp: v('--spin-up'), spinDown: v('--spin-down'), cluster: v('--cluster'),
    divNeg: v('--div-neg'), divMid: v('--div-mid'), divPos: v('--div-pos'),
  };
  return paletteCache;
}
window.addEventListener('themechange', () => { paletteCache = null; }, { capture: true });

export function hexToRgb(hex) {
  const h = hex.replace('#', '');
  const full = h.length === 3 ? h.split('').map((c) => c + c).join('') : h;
  const n = parseInt(full, 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

export function withAlpha(hex, alpha) {
  const [r, g, b] = hexToRgb(hex);
  return `rgba(${r},${g},${b},${alpha})`;
}

// ---------- Number formatting ----------

const MINUS = '−';

export function fmtNum(v, digits = 2) {
  if (v === Infinity) return '∞';
  if (!Number.isFinite(v)) return '—';
  if (Math.abs(v) < 0.5 * 10 ** -digits) v = 0; // avoid "-0.00"
  const s = v.toLocaleString('en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits });
  return s.replaceAll('-', MINUS);
}

/** Significant-figure formatting with thousands separators for large values. */
export function fmtSig(v, sig = 3) {
  if (v === Infinity) return '∞';
  if (!Number.isFinite(v)) return '—';
  if (v === 0) return '0';
  const a = Math.abs(v);
  if (a >= 1e5 || a < 1e-4) return v.toExponential(sig - 1).replace('e+', 'e').replaceAll('-', MINUS);
  const digits = Math.max(0, sig - 1 - Math.floor(Math.log10(a)));
  return fmtNum(v, Math.min(digits, 8));
}

export function fmtInt(v) {
  return Number.isFinite(v) ? Math.round(v).toLocaleString('en-US') : '—';
}

function fmtTick(v, step) {
  const a = Math.abs(v);
  if (a >= 1e6) return (v / 1e6).toLocaleString('en-US', { maximumFractionDigits: 1 }) + 'M';
  if (a >= 1e4) return (v / 1e3).toLocaleString('en-US', { maximumFractionDigits: 1 }) + 'k';
  const d = Math.max(0, Math.min(6, -Math.floor(Math.log10(step) + 1e-9)));
  return fmtNum(v, d);
}

function fmtLogTick(v) {
  if (v >= 1e6) return (v / 1e6) + 'M';
  if (v >= 1e3) return (v / 1e3) + 'k';
  if (v >= 1) return String(v);
  return String(Number(v.toPrecision(1)));
}

// ---------- Ticks ----------

function niceStep(span, count) {
  const raw = span / Math.max(1, count);
  const mag = 10 ** Math.floor(Math.log10(raw));
  const r = raw / mag;
  return (r < 1.5 ? 1 : r < 3 ? 2 : r < 7 ? 5 : 10) * mag;
}

function linearTicks(lo, hi, count) {
  if (!(hi > lo)) return { ticks: [lo], step: 1 };
  const step = niceStep(hi - lo, count);
  const ticks = [];
  for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + step * 1e-9; v += step) {
    ticks.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  }
  return { ticks, step };
}

function logTicks(lo, hi) {
  const a = Math.ceil(Math.log10(lo) - 1e-9), b = Math.floor(Math.log10(hi) + 1e-9);
  let ticks = [];
  const every = Math.max(1, Math.ceil((b - a + 1) / 7));
  for (let k = a; k <= b; k += every) ticks.push(10 ** k);
  if (ticks.length < 2) {
    ticks = [];
    for (let k = a - 1; k <= b; k++) for (const m of [1, 2, 5]) {
      const v = m * 10 ** k;
      if (v >= lo * (1 - 1e-9) && v <= hi * (1 + 1e-9)) ticks.push(v);
    }
  }
  return ticks;
}

// ---------- Plot ----------

export class Plot {
  /**
   * @param {HTMLElement|string} el  container (gets a canvas, crosshair and tooltip)
   * @param {object} opts  { height | height(w), square, maxHeight, grid: 'y'|'xy'|'none',
   *   x: {type, domain, label, ticks}, y: {...}, margin, draw(plot), hover(x, y, p) }
   */
  constructor(el, opts = {}) {
    this.el = typeof el === 'string' ? document.getElementById(el) : el;
    this.el.classList.add('plot');
    this.opts = { height: 220, grid: 'y', ...opts };
    this.x = { type: 'linear', domain: [0, 1], label: '', ticks: 6, ...opts.x };
    this.y = { type: 'linear', domain: [0, 1], label: '', ticks: 5, ...opts.y };
    this.baseMargin = { t: 10, r: 14, b: this.x.label ? 38 : 24, l: this.y.label ? 58 : 46, ...opts.margin };
    this.m = { ...this.baseMargin };
    this.draw = opts.draw || null;
    this.hover = opts.hover || null;

    this.canvas = document.createElement('canvas');
    this.ctx = this.canvas.getContext('2d');
    this.cross = document.createElement('div');
    this.cross.className = 'plot-cross';
    this.tip = document.createElement('div');
    this.tip.className = 'plot-tip';
    this.el.append(this.canvas, this.cross, this.tip);
    if (opts.label) this.canvas.setAttribute('aria-label', opts.label);
    this.canvas.setAttribute('role', 'img');

    this.w = 0;
    this.h = 0;
    this.pointer = null;
    this.ro = new ResizeObserver(() => this.resize());
    this.ro.observe(this.el);
    // Size once the calling module has finished setting up (draw callbacks may use its later bindings).
    queueMicrotask(() => this.resize());
    this.canvas.addEventListener('pointermove', (e) => {
      const r = this.canvas.getBoundingClientRect();
      this.pointer = { px: e.clientX - r.left, py: e.clientY - r.top };
      this.updateHover();
    });
    this.canvas.addEventListener('pointerleave', () => {
      this.pointer = null;
      this.updateHover();
    });
    window.addEventListener('themechange', () => this.redraw());
  }

  get iw() { return this.w - this.m.l - this.m.r; }
  get ih() { return this.h - this.m.t - this.m.b; }

  resize() {
    const w = this.el.clientWidth;
    if (!w) return;
    let h;
    this.m = { ...this.baseMargin };
    if (this.opts.square) {
      const maxH = this.opts.maxHeight ?? 420;
      const side = Math.max(60, Math.min(w - this.m.l - this.m.r, maxH - this.m.t - this.m.b));
      const extra = w - this.m.l - this.m.r - side;
      this.m.l += extra / 2;
      this.m.r += extra / 2;
      h = side + this.m.t + this.m.b;
    } else {
      h = typeof this.opts.height === 'function' ? this.opts.height(w) : this.opts.height;
    }
    const dpr = window.devicePixelRatio || 1;
    if (w === this.w && h === this.h && this.dpr === dpr) return;
    this.w = w;
    this.h = h;
    this.dpr = dpr;
    this.canvas.width = Math.round(w * dpr);
    this.canvas.height = Math.round(h * dpr);
    this.canvas.style.height = h + 'px';
    this.redraw();
  }

  setX(domain) { this.x.domain = domain; return this; }
  setY(domain) { this.y.domain = domain; return this; }

  sx(v) {
    const [a, b] = this.x.domain;
    const t = this.x.type === 'log' ? (Math.log10(v) - Math.log10(a)) / (Math.log10(b) - Math.log10(a)) : (v - a) / (b - a);
    return this.m.l + t * this.iw;
  }
  sy(v) {
    const [a, b] = this.y.domain;
    const t = this.y.type === 'log' ? (Math.log10(v) - Math.log10(a)) / (Math.log10(b) - Math.log10(a)) : (v - a) / (b - a);
    return this.m.t + this.ih - t * this.ih;
  }
  invx(px) {
    const [a, b] = this.x.domain;
    const t = (px - this.m.l) / this.iw;
    return this.x.type === 'log' ? 10 ** (Math.log10(a) + t * (Math.log10(b) - Math.log10(a))) : a + t * (b - a);
  }
  invy(py) {
    const [a, b] = this.y.domain;
    const t = (this.m.t + this.ih - py) / this.ih;
    return this.y.type === 'log' ? 10 ** (Math.log10(a) + t * (Math.log10(b) - Math.log10(a))) : a + t * (b - a);
  }

  redraw() {
    if (!this.w) return;
    const ctx = this.ctx;
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.clearRect(0, 0, this.w, this.h);
    this.c = palette();
    this.drawGrid();
    ctx.save();
    ctx.beginPath();
    ctx.rect(this.m.l, this.m.t, this.iw, this.ih);
    ctx.clip();
    if (this.draw) this.draw(this);
    ctx.restore();
    this.drawAxes();
    if (this.pointer) this.updateHover();
  }

  xTicks() {
    return this.x.type === 'log' ? { ticks: logTicks(...this.x.domain), log: true } : linearTicks(...this.x.domain, Math.max(2, Math.min(this.x.ticks, Math.floor(this.iw / 60))));
  }
  yTicks() {
    return this.y.type === 'log' ? { ticks: logTicks(...this.y.domain), log: true } : linearTicks(...this.y.domain, Math.max(2, Math.min(this.y.ticks, Math.floor(this.ih / 32))));
  }

  drawGrid() {
    const { ctx, c } = this;
    ctx.lineWidth = 1;
    ctx.strokeStyle = c.grid;
    if (this.opts.grid === 'none') return;
    ctx.beginPath();
    for (const v of this.yTicks().ticks) {
      const y = Math.round(this.sy(v)) + 0.5;
      ctx.moveTo(this.m.l, y);
      ctx.lineTo(this.m.l + this.iw, y);
    }
    if (this.opts.grid === 'xy') {
      for (const v of this.xTicks().ticks) {
        const x = Math.round(this.sx(v)) + 0.5;
        ctx.moveTo(x, this.m.t);
        ctx.lineTo(x, this.m.t + this.ih);
      }
    }
    ctx.stroke();
  }

  drawAxes() {
    const { ctx, c } = this;
    ctx.font = '11px system-ui, -apple-system, "Segoe UI", sans-serif';
    ctx.fillStyle = c.muted;
    ctx.strokeStyle = c.axis;
    ctx.lineWidth = 1;
    const base = Math.round(this.m.t + this.ih) + 0.5;
    ctx.beginPath();
    ctx.moveTo(this.m.l, base);
    ctx.lineTo(this.m.l + this.iw, base);
    ctx.stroke();

    const xt = this.xTicks();
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    for (const v of xt.ticks) {
      const x = this.sx(v);
      ctx.beginPath();
      ctx.moveTo(Math.round(x) + 0.5, base);
      ctx.lineTo(Math.round(x) + 0.5, base + 4);
      ctx.stroke();
      ctx.fillText(xt.log ? fmtLogTick(v) : fmtTick(v, xt.step), x, base + 6);
    }
    const yt = this.yTicks();
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (const v of yt.ticks) {
      ctx.fillText(yt.log ? fmtLogTick(v) : fmtTick(v, yt.step), this.m.l - 6, this.sy(v));
    }
    if (this.x.label) {
      ctx.textAlign = 'right';
      ctx.textBaseline = 'bottom';
      ctx.fillStyle = c.ink2;
      ctx.fillText(this.x.label, this.m.l + this.iw, this.h - 2);
    }
    if (this.y.label) {
      ctx.save();
      ctx.translate(12, this.m.t + this.ih / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillStyle = c.ink2;
      ctx.fillText(this.y.label, 0, 0);
      ctx.restore();
    }
  }

  // ---------- Marks (all in data coordinates) ----------

  /** Polyline through (xs[i], ys[i]); xs may be null for x = x0 + i * dx. Decimates dense data. */
  line(xs, ys, { color, width = 2, alpha = 1, n = ys.length, x0 = 0, dx = 1, dash = null } = {}) {
    if (n < 1) return;
    const ctx = this.ctx;
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.globalAlpha = alpha;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    if (dash) ctx.setLineDash(dash);
    ctx.beginPath();
    const X = xs ? (i) => xs[i] : (i) => x0 + i * dx;
    if (n > this.iw * 3) {
      let col = null, first = 0, last = 0, lo = 0, hi = 0;
      const flush = () => {
        ctx.lineTo(col, first);
        ctx.lineTo(col, lo);
        ctx.lineTo(col, hi);
        ctx.lineTo(col, last);
      };
      for (let i = 0; i < n; i++) {
        const yv = ys[i];
        if (!Number.isFinite(yv)) continue;
        const px = Math.round(this.sx(X(i)));
        const py = this.sy(yv);
        if (px !== col) {
          if (col === null) ctx.moveTo(px, py);
          else flush();
          col = px;
          first = last = lo = hi = py;
        } else {
          last = py;
          if (py < lo) lo = py;
          if (py > hi) hi = py;
        }
      }
      if (col !== null) flush();
    } else {
      let pen = false;
      for (let i = 0; i < n; i++) {
        const xv = X(i), yv = ys[i];
        if (!Number.isFinite(yv) || !Number.isFinite(xv)) { pen = false; continue; }
        const px = this.sx(xv), py = Math.max(-1e4, Math.min(1e4, this.sy(yv)));
        if (pen) ctx.lineTo(px, py);
        else { ctx.moveTo(px, py); pen = true; }
      }
    }
    ctx.stroke();
    ctx.restore();
  }

  /** Plot a function across the visible x-range. */
  fn(f, opts = {}) {
    const k = Math.max(2, Math.round(this.iw));
    const xs = new Float64Array(k + 1), ys = new Float64Array(k + 1);
    for (let i = 0; i <= k; i++) {
      const x = this.invx(this.m.l + (i / k) * this.iw);
      xs[i] = x;
      ys[i] = f(x);
    }
    this.line(xs, ys, opts);
  }

  /** Shaded area under a function (a light wash of its color). */
  area(f, { color, alpha = 0.1 } = {}) {
    const ctx = this.ctx;
    const k = Math.max(2, Math.round(this.iw / 2));
    const y0 = this.sy(Math.max(this.y.domain[0], 0));
    ctx.save();
    ctx.fillStyle = color;
    ctx.globalAlpha = alpha;
    ctx.beginPath();
    ctx.moveTo(this.m.l, y0);
    for (let i = 0; i <= k; i++) {
      const px = this.m.l + (i / k) * this.iw;
      const v = f(this.invx(px));
      ctx.lineTo(px, Number.isFinite(v) ? this.sy(v) : y0);
    }
    ctx.lineTo(this.m.l + this.iw, y0);
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  }

  /** Histogram-style bars: bar k spans edges[k]..edges[k+1] with height heights[k]. */
  bars(edges, heights, { color, gap = 2, radius = 3, alpha = 1 } = {}) {
    const ctx = this.ctx;
    const y0 = this.sy(0);
    ctx.save();
    ctx.fillStyle = color;
    ctx.globalAlpha = alpha;
    for (let k = 0; k < heights.length; k++) {
      if (!(heights[k] > 0)) continue;
      const x1 = this.sx(edges[k]), x2 = this.sx(edges[k + 1]);
      const w = x2 - x1;
      const g = w > 6 ? gap : w > 3 ? 1 : 0;
      const y = this.sy(heights[k]);
      const r = w - g > 8 ? Math.min(radius, y0 - y) : 0;
      ctx.beginPath();
      if (r > 0 && ctx.roundRect) ctx.roundRect(x1 + g / 2, y, w - g, y0 - y, [r, r, 0, 0]);
      else ctx.rect(x1 + g / 2, y, w - g, y0 - y);
      ctx.fill();
    }
    ctx.restore();
  }

  /** Thin columns centered on xs, growing from 0 (used for autocorrelation). */
  columns(xs, ys, { color, width = 4, n = ys.length } = {}) {
    const ctx = this.ctx;
    const y0 = this.sy(0);
    ctx.save();
    ctx.fillStyle = color;
    for (let i = 0; i < n; i++) {
      const x = this.sx(xs ? xs[i] : i);
      const y = this.sy(ys[i]);
      ctx.fillRect(x - width / 2, Math.min(y, y0), width, Math.abs(y0 - y) || 1);
    }
    ctx.restore();
  }

  /** Dots. `ring` draws a surface-colored ring so they read against lines. */
  points(xs, ys, { color, r = 3, alpha = 1, ring = false, n = ys.length, start = 0 } = {}) {
    const ctx = this.ctx;
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.fillStyle = color;
    if (!ring && r <= 2) {
      for (let i = start; i < n; i++) {
        ctx.fillRect(this.sx(xs[i]) - r, this.sy(ys[i]) - r, 2 * r, 2 * r);
      }
    } else {
      ctx.strokeStyle = this.c.surface;
      ctx.lineWidth = 2;
      for (let i = start; i < n; i++) {
        const x = this.sx(xs[i]), y = this.sy(ys[i]);
        if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
        ctx.beginPath();
        ctx.arc(x, y, r, 0, 2 * Math.PI);
        if (ring) ctx.stroke();
        ctx.fill();
      }
    }
    ctx.restore();
  }

  /** A single marker; `hollow` draws an outlined ring. */
  marker(x, y, { color, r = 5, hollow = false } = {}) {
    const ctx = this.ctx;
    const px = this.sx(x), py = this.sy(y);
    ctx.save();
    ctx.beginPath();
    ctx.arc(px, py, r, 0, 2 * Math.PI);
    ctx.lineWidth = 2;
    if (hollow) {
      ctx.strokeStyle = this.c.surface;
      ctx.lineWidth = 5;
      ctx.stroke();
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.stroke();
    } else {
      ctx.strokeStyle = this.c.surface;
      ctx.stroke();
      ctx.fillStyle = color;
      ctx.fill();
    }
    ctx.restore();
  }

  /** Vertical band across the full plot height. */
  vband(x0, x1, { color, alpha = 0.08 } = {}) {
    const ctx = this.ctx;
    ctx.save();
    ctx.fillStyle = color;
    ctx.globalAlpha = alpha;
    const a = this.sx(x0), b = this.sx(x1);
    ctx.fillRect(Math.min(a, b), this.m.t, Math.abs(b - a), this.ih);
    ctx.restore();
  }

  vline(x, { color, width = 1, alpha = 1, dash = null } = {}) {
    const ctx = this.ctx;
    const px = Math.round(this.sx(x)) + 0.5;
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.globalAlpha = alpha;
    if (dash) ctx.setLineDash(dash);
    ctx.beginPath();
    ctx.moveTo(px, this.m.t);
    ctx.lineTo(px, this.m.t + this.ih);
    ctx.stroke();
    ctx.restore();
  }

  hline(y, { color, width = 1, alpha = 1, dash = null } = {}) {
    const ctx = this.ctx;
    const py = Math.round(this.sy(y)) + 0.5;
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.globalAlpha = alpha;
    if (dash) ctx.setLineDash(dash);
    ctx.beginPath();
    ctx.moveTo(this.m.l, py);
    ctx.lineTo(this.m.l + this.iw, py);
    ctx.stroke();
    ctx.restore();
  }

  /** Vertical error bars. */
  errorBars(xs, ys, errs, { color, n = ys.length } = {}) {
    const ctx = this.ctx;
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      if (!Number.isFinite(errs[i])) continue;
      const x = this.sx(xs[i]);
      ctx.moveTo(x, this.sy(ys[i] - errs[i]));
      ctx.lineTo(x, this.sy(ys[i] + errs[i]));
    }
    ctx.stroke();
    ctx.restore();
  }

  text(x, y, str, { color, align = 'left', baseline = 'middle', dx = 0, dy = 0, weight = 400, size = 11 } = {}) {
    const ctx = this.ctx;
    ctx.save();
    ctx.font = `${weight} ${size}px system-ui, -apple-system, "Segoe UI", sans-serif`;
    ctx.fillStyle = color || this.c.ink2;
    ctx.textAlign = align;
    ctx.textBaseline = baseline;
    ctx.fillText(str, this.sx(x) + dx, this.sy(y) + dy);
    ctx.restore();
  }

  /** Draw an image (e.g. a density heatmap) stretched over a data-space rectangle. */
  image(img, x0, x1, y0, y1, { smooth = true, alpha = 1 } = {}) {
    const ctx = this.ctx;
    ctx.save();
    ctx.imageSmoothingEnabled = smooth;
    ctx.globalAlpha = alpha;
    const a = this.sx(x0), b = this.sx(x1), c = this.sy(y1), d = this.sy(y0);
    ctx.drawImage(img, a, c, b - a, d - c);
    ctx.restore();
  }

  // ---------- Hover ----------

  updateHover() {
    const p = this.pointer;
    const inside = p && p.px >= this.m.l - 2 && p.px <= this.m.l + this.iw + 2 && p.py >= this.m.t - 2 && p.py <= this.m.t + this.ih + 2;
    const res = inside && this.hover ? this.hover(this.invx(p.px), this.invy(p.py), p) : null;
    if (!res) {
      this.tip.style.display = 'none';
      this.cross.style.display = 'none';
      return;
    }
    if (res.x !== undefined && Number.isFinite(res.x)) {
      this.cross.style.display = 'block';
      this.cross.style.left = Math.round(this.sx(res.x)) + 'px';
      this.cross.style.top = this.m.t + 'px';
      this.cross.style.height = this.ih + 'px';
    } else {
      this.cross.style.display = 'none';
    }
    const tip = this.tip;
    tip.replaceChildren();
    if (res.title) {
      const t = document.createElement('div');
      t.className = 'tip-title';
      t.textContent = res.title;
      tip.append(t);
    }
    for (const row of res.rows || []) {
      const r = document.createElement('div');
      r.className = 'tip-row';
      if (row.color) {
        const k = document.createElement('span');
        k.className = 'key line';
        k.style.setProperty('--c', row.color);
        r.append(k);
      }
      const v = document.createElement('span');
      v.className = 'tip-value';
      v.textContent = row.value;
      const l = document.createElement('span');
      l.className = 'tip-label';
      l.textContent = row.label;
      r.append(v, l);
      tip.append(r);
    }
    tip.style.display = 'block';
    const tw = tip.offsetWidth, th = tip.offsetHeight;
    let left = p.px + 14;
    if (left + tw > this.w) left = p.px - tw - 14;
    let top = p.py - th - 10;
    if (top < 0) top = p.py + 14;
    tip.style.left = Math.max(0, left) + 'px';
    tip.style.top = top + 'px';
  }
}

/** Index of the element of the sorted array `xs` (first n entries) nearest to x. */
export function nearestIndex(xs, x, n = xs.length) {
  if (n <= 0) return -1;
  let lo = 0, hi = n - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (xs[mid] <= x) lo = mid;
    else hi = mid;
  }
  return Math.abs(xs[lo] - x) <= Math.abs(xs[hi] - x) ? lo : hi;
}

/** Fill a <ul class="legend"> with keys. items: [{label, color, kind: 'line'|'rect'|'dot'}]. */
export function legend(el, items) {
  const ul = typeof el === 'string' ? document.getElementById(el) : el;
  ul.replaceChildren();
  for (const it of items) {
    const li = document.createElement('li');
    const k = document.createElement('span');
    k.className = `key ${it.kind || 'line'}`;
    k.style.setProperty('--c', it.color);
    const t = document.createElement('span');
    t.textContent = it.label;
    li.append(k, t);
    ul.append(li);
  }
}

/**
 * Render a 2D density to an offscreen canvas: surface color where the density is 0,
 * rising to `hi` at the maximum. f(x, y) may be unnormalized.
 */
export function densityImage(f, [x0, x1], [y0, y1], res, lo, hi, gamma = 0.6) {
  const cv = document.createElement('canvas');
  cv.width = res;
  cv.height = res;
  const ctx = cv.getContext('2d');
  const img = ctx.createImageData(res, res);
  const vals = new Float64Array(res * res);
  let max = 0;
  for (let j = 0; j < res; j++) {
    const y = y1 - ((j + 0.5) / res) * (y1 - y0);
    for (let i = 0; i < res; i++) {
      const x = x0 + ((i + 0.5) / res) * (x1 - x0);
      const v = f(x, y);
      vals[j * res + i] = v;
      if (v > max) max = v;
    }
  }
  const a = hexToRgb(lo), b = hexToRgb(hi);
  for (let k = 0; k < vals.length; k++) {
    const t = max > 0 ? Math.pow(vals[k] / max, gamma) : 0;
    img.data[4 * k] = a[0] + t * (b[0] - a[0]);
    img.data[4 * k + 1] = a[1] + t * (b[1] - a[1]);
    img.data[4 * k + 2] = a[2] + t * (b[2] - a[2]);
    img.data[4 * k + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  return cv;
}
