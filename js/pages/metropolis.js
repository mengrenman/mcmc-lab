import { mountChrome, $, setText, bindRange, Runner, Pacer, throttle } from '../lib/ui.js';
import { Plot, legend, nearestIndex, fmtNum, fmtSig, fmtInt } from '../lib/plot.js';
import { RNG, randomSeed } from '../lib/random.js';
import { Trace, chainDiagnostics, jackknifeCurve, histogram } from '../lib/stats.js';
import { targets1D } from '../lib/targets.js';
import { Metropolis1D } from '../samplers/metropolis.js';

mountChrome();

const MAX_STEPS = 200000;
const BINS = 60;

const rng = new RNG();
let target = targets1D.gaussian;
const chain = new Metropolis1D(target, rng);
const trace = new Trace(MAX_STEPS + 1);
const pacer = new Pacer();
let domain = target.domain;
let diag = null, jack = null, hist = null;

const fmtSpeed = (v) => (v < 10 ? fmtNum(v, 1) : fmtInt(v)) + ' steps/s';
const stepSize = bindRange('mh-step', { log: true, format: (v) => fmtSig(v, 2), onInput: restart });
const start = bindRange('mh-start', { format: (v) => fmtNum(v, 1), onInput: restart });
const burn = bindRange('mh-burn', {
  format: fmtInt,
  onInput() {
    updateDiagnostics();
    tracePlot.redraw();
  },
});
const speed = bindRange('mh-speed', { log: true, format: fmtSpeed });

// ---------- Charts ----------

const densityPlot = new Plot('mh-density', {
  height: 210,
  x: { label: 'x' },
  draw(p) {
    const c = p.c;
    p.area(target.pdf, { color: c.exact, alpha: 0.08 });
    p.fn(target.pdf, { color: c.exact, width: 2 });
    const showProposal = chain.hasLast && (!runner.running || speed.get() <= 60);
    if (showProposal) {
      const L = chain.last;
      const cc = stepSize.get();
      p.vband(L.from - cc, L.from + cc, { color: c.s1, alpha: 0.1 });
      const col = L.accepted ? c.good : c.bad;
      p.vline(L.to, { color: col, alpha: 0.7 });
      p.marker(L.to, target.pdf(L.to), { color: col, r: 7, hollow: true });
    }
    p.vline(chain.x, { color: c.s1, width: 2 });
    p.marker(chain.x, target.pdf(chain.x), { color: c.s1, r: 5 });
  },
  hover(x) {
    return { x, title: `x = ${fmtNum(x, 2)}`, rows: [{ value: fmtSig(target.pdf(x), 3), label: 'p(x)', color: 'var(--exact)' }] };
  },
});
legend('mh-density-legend', [
  { label: 'target p(x)', color: 'var(--exact)', kind: 'line' },
  { label: 'current x', color: 'var(--s1)', kind: 'dot' },
  { label: 'accepted proposal', color: 'var(--good)', kind: 'dot' },
  { label: 'rejected proposal', color: 'var(--bad)', kind: 'dot' },
]);

const tracePlot = new Plot('mh-trace', {
  height: 190,
  x: { label: 'step' },
  draw(p) {
    const c = p.c;
    const b = burn.get();
    if (b > 0) p.vband(0, b, { color: c.ink, alpha: 0.06 });
    p.hline(target.mean, { color: c.exact, alpha: 0.6 });
    p.line(null, trace.a, { color: c.s1, width: trace.n > 2000 ? 1 : 1.5, n: trace.n });
    if (trace.n) p.marker(trace.n - 1, trace.last(), { color: c.s1, r: 4 });
  },
  hover(x) {
    if (!trace.n) return null;
    const i = Math.max(0, Math.min(trace.n - 1, Math.round(x)));
    return {
      x: i,
      title: i === 0 ? 'start' : `step ${fmtInt(i)}`,
      rows: [{ value: fmtNum(trace.a[i], 3), label: 'x', color: 'var(--s1)' }],
    };
  },
});

const histPlot = new Plot('mh-hist', {
  height: 210,
  x: { label: 'x' },
  draw(p) {
    const c = p.c;
    if (hist) p.bars(hist.edges, hist.heights, { color: c.s1, alpha: 0.85 });
    p.fn(target.pdf, { color: c.exact, width: 2 });
  },
  hover(x) {
    const [lo, hi] = target.domain;
    const w = (hi - lo) / BINS;
    const k = Math.floor((x - lo) / w);
    if (k < 0 || k >= BINS) return null;
    const a = lo + k * w;
    return {
      title: `x in [${fmtNum(a, 2)}, ${fmtNum(a + w, 2)})`,
      rows: [
        { value: hist ? fmtNum(hist.heights[k], 3) : '—', label: 'samples', color: 'var(--s1)' },
        { value: fmtNum(target.pdf(a + w / 2), 3), label: 'target', color: 'var(--exact)' },
      ],
    };
  },
});
legend('mh-hist-legend', [
  { label: 'kept samples', color: 'var(--s1)', kind: 'rect' },
  { label: 'target p(x)', color: 'var(--exact)', kind: 'line' },
]);

let acfShown = 0;
const acfPlot = new Plot('mh-acf', {
  height: 210,
  x: { label: 'lag t' },
  y: { domain: [-0.2, 1] },
  draw(p) {
    const c = p.c;
    if (!diag || !diag.rho) return;
    p.hline(0, { color: c.axis });
    const w = Math.max(1, Math.min(6, (p.iw / (acfShown + 1)) * 0.6));
    p.columns(null, diag.rho, { color: c.s1, width: w, n: acfShown + 1 });
    if (diag.window <= acfShown) p.vline(diag.window, { color: c.ink2 });
  },
  hover(x) {
    if (!diag || !diag.rho) return null;
    const t = Math.max(0, Math.min(acfShown, Math.round(x)));
    return { x: t, title: `lag ${t}`, rows: [{ value: fmtNum(diag.rho[t], 3), label: 'ρ(t)', color: 'var(--s1)' }] };
  },
});

const jackPlot = new Plot('mh-jack', {
  height: 200,
  x: { type: 'log', domain: [1, 1024], label: 'bin size w' },
  draw(p) {
    const c = p.c;
    if (!jack || !jack.sizes.length) return;
    if (diag && Number.isFinite(diag.error)) p.hline(diag.error, { color: c.s2, width: 1.5 });
    p.line(jack.sizes, jack.errors, { color: c.s1 });
    p.points(jack.sizes, jack.errors, { color: c.s1, r: 4, ring: true });
  },
  hover(x) {
    if (!jack || !jack.sizes.length) return null;
    const logs = jack.sizes.map(Math.log10);
    const i = nearestIndex(logs, Math.log10(x));
    const rows = [{ value: fmtSig(jack.errors[i], 3), label: 'jackknife error', color: 'var(--s1)' }];
    if (diag) rows.push({ value: fmtSig(diag.error, 3), label: 'σ √(τ/N)', color: 'var(--s2)' });
    return { x: jack.sizes[i], title: `bin size ${fmtInt(jack.sizes[i])}`, rows };
  },
});
legend('mh-jack-legend', [
  { label: 'jackknife error', color: 'var(--s1)', kind: 'line' },
  { label: 'σ √(τint / N) from the autocorrelation', color: 'var(--s2)', kind: 'line' },
]);

// ---------- State updates ----------

function computeDomain() {
  const [a, b] = target.domain;
  const x0 = start.get();
  return [Math.min(a, x0 - 0.5), Math.max(b, x0 + 0.5)];
}

function restart() {
  runner?.stop();
  chain.setTarget(target);
  chain.reset(start.get());
  trace.clear();
  trace.push(chain.x);
  pacer.reset();
  domain = computeDomain();
  const peak = Math.max(...Array.from({ length: 400 }, (_, i) => target.pdf(domain[0] + ((i + 0.5) / 400) * (domain[1] - domain[0]))));
  densityPlot.setX(domain).setY([0, peak * 1.12]);
  tracePlot.setY(domain);
  histPlot.setX(target.domain);
  $('mh-full').hidden = true;
  setText('mh-t-mean-exact', `exact ${fmtNum(target.mean, 2)}`);
  setText('mh-t-var-exact', `exact ${fmtNum(target.variance, 2)}`);
  setText('mh-t-seed', `seed ${rng.initialSeed}`);
  updateDiagnostics();
  drawLive();
}

function describeDecision() {
  const el = $('mh-decision');
  if (!chain.hasLast) {
    el.textContent = 'Press Step to make one proposal, or Run to watch the chain.';
    return;
  }
  const L = chain.last;
  const x = fmtNum(L.from, 3), xp = fmtNum(L.to, 3);
  const verdict = L.accepted
    ? '<span class="badge good">✓ Accepted</span> The chain moves to x′.'
    : `<span class="badge bad">✗ Rejected</span> The chain stays at x = ${x} and records it again.`;
  let reason;
  if (L.ratio === 0) reason = 'p(x′) = 0 because x′ is outside the support, so the move is rejected.';
  else if (L.ratio >= 1) reason = `p(x′)/p(x) = ${fmtSig(L.ratio, 3)} ≥ 1, so the move is always accepted.`;
  else reason = `p(x′)/p(x) = ${fmtNum(L.ratio, 3)}. Drew u = ${fmtNum(L.u, 3)}, which is ${L.u < L.ratio ? '<' : '≥'} ${fmtNum(L.ratio, 3)}.`;
  el.innerHTML = `Proposed x′ = ${xp} from x = ${x}. ${reason} ${verdict}`;
}

function drawLive() {
  densityPlot.redraw();
  tracePlot.setX([0, Math.max(50, trace.n - 1)]);
  tracePlot.redraw();
  describeDecision();
  setText('mh-t-steps', fmtInt(chain.steps));
  setText('mh-t-acc', chain.steps ? `${fmtNum(100 * chain.acceptance, 1)}%` : '—');
}

function updateDiagnostics() {
  const n = trace.n;
  const s = Math.min(burn.get(), n);
  diag = chainDiagnostics(trace.a, s, n);
  jack = n - s >= 64 ? jackknifeCurve(trace.a, s, n) : null;
  hist = n - s > 0 ? histogram(trace.a, target.domain[0], target.domain[1], BINS, s, n) : null;

  let top = 0;
  for (let x = target.domain[0]; x <= target.domain[1]; x += (target.domain[1] - target.domain[0]) / 300) top = Math.max(top, target.pdf(x));
  if (hist) for (const h of hist.heights) top = Math.max(top, h);
  histPlot.setY([0, top * 1.1]);
  histPlot.redraw();

  if (diag && diag.rho) {
    acfShown = Math.min(diag.rho.length - 1, Math.max(20, Math.min(300, Math.ceil(2 * diag.window))));
    let lo = 0;
    for (let t = 0; t <= acfShown; t++) lo = Math.min(lo, diag.rho[t]);
    acfPlot.setX([-0.5, acfShown + 0.5]).setY([Math.min(-0.1, lo * 1.1), 1.02]);
  }
  acfPlot.redraw();

  if (jack && jack.sizes.length) {
    let top2 = 0;
    for (const e of jack.errors) if (Number.isFinite(e)) top2 = Math.max(top2, e);
    if (diag && Number.isFinite(diag.error)) top2 = Math.max(top2, diag.error);
    jackPlot.setX([0.8, Math.max(8, jack.sizes[jack.sizes.length - 1] * 1.25)]).setY([0, top2 * 1.15 || 1]);
  }
  jackPlot.redraw();

  if (diag) {
    // Until the window converges, tau (and so the error bar) is a lower bound and ESS an upper bound.
    const approx = diag.converged ? '' : '≳ ';
    setText('mh-t-mean', `${fmtNum(diag.mean, 3)} ± ${approx}${fmtSig(diag.error, 2)}`);
    setText('mh-t-var', fmtNum(diag.variance, 3));
    setText('mh-t-tau', approx + fmtSig(diag.tau, 3));
    setText('mh-t-window', diag.converged ? `window W = ${fmtInt(diag.window)}` : 'window not reached: run longer');
    setText('mh-t-ess', (diag.converged ? '' : '≲ ') + fmtInt(diag.ess));
    setText('mh-t-kept', `from ${fmtInt(diag.n)} kept samples`);
  } else {
    for (const id of ['mh-t-mean', 'mh-t-var', 'mh-t-tau', 'mh-t-ess']) setText(id, '—');
    setText('mh-t-window', '');
    setText('mh-t-kept', n <= s ? 'still inside the burn-in' : '');
  }
}
const diagnosticsSoon = throttle(updateDiagnostics, 300);

function advance(k) {
  const c = stepSize.get();
  for (let i = 0; i < k; i++) {
    chain.step(c);
    trace.push(chain.x);
    if (trace.full) {
      $('mh-full').hidden = false;
      return false;
    }
  }
  return true;
}

const runner = new Runner({
  button: 'mh-run',
  root: 'lab',
  tick(dt) {
    const k = pacer.take(speed.get(), dt, 20000);
    if (k === 0) return true;
    const ok = advance(k);
    drawLive();
    diagnosticsSoon();
    return ok;
  },
  onChange(running) {
    if (!running) {
      drawLive();
      updateDiagnostics();
    }
  },
});

$('mh-once').addEventListener('click', () => {
  runner.stop();
  advance(1);
  drawLive();
  updateDiagnostics();
});
$('mh-reset').addEventListener('click', () => {
  rng.seed(randomSeed());
  restart();
});
$('mh-target').addEventListener('change', (e) => {
  target = targets1D[e.target.value];
  start.setBounds(...target.startRange);
  start.set(target.start);
  restart();
});

restart();
