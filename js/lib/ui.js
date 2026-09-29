// Page chrome (header, theme toggle) and small helpers for controls and animation.

const PAGES = [
  ['index.html', 'Home'],
  ['monte-carlo.html', 'Monte Carlo'],
  ['metropolis.html', 'Metropolis'],
  ['hmc-gibbs.html', 'HMC & Gibbs'],
  ['bayes.html', 'Bayesian'],
  ['ising.html', 'Ising model'],
];

const THEME_KEY = 'mcmc-lab-theme';

function storedTheme() {
  try { return localStorage.getItem(THEME_KEY); } catch { return null; }
}

function effectiveTheme() {
  const t = document.documentElement.dataset.theme;
  if (t === 'light' || t === 'dark') return t;
  return matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

const SUN = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>';
const MOON = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/></svg>';
const LOGO = '<svg width="22" height="22" viewBox="0 0 24 24" aria-hidden="true"><path d="M3 18c3-9 5-12 7-6s4 4 5-2 3-6 6 1" fill="none" stroke="var(--s1)" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/><circle cx="21" cy="11" r="2.2" fill="var(--s2)"/></svg>';

export function mountChrome() {
  const here = location.pathname.split('/').pop() || 'index.html';
  const header = document.createElement('header');
  header.className = 'site-header';
  const inner = document.createElement('div');
  inner.className = 'inner';
  const brand = document.createElement('a');
  brand.className = 'brand';
  brand.href = 'index.html';
  brand.innerHTML = LOGO + '<span>MCMC Lab</span>';
  const nav = document.createElement('nav');
  nav.className = 'site-nav';
  nav.setAttribute('aria-label', 'Modules');
  for (const [href, label] of PAGES.slice(1)) {
    const a = document.createElement('a');
    a.href = href;
    a.textContent = label;
    if (href === here) a.setAttribute('aria-current', 'page');
    nav.append(a);
  }
  const toggle = document.createElement('button');
  toggle.className = 'theme-toggle';
  toggle.type = 'button';
  const paint = () => {
    const dark = effectiveTheme() === 'dark';
    toggle.innerHTML = dark ? SUN : MOON;
    toggle.setAttribute('aria-label', dark ? 'Switch to light theme' : 'Switch to dark theme');
    toggle.title = toggle.getAttribute('aria-label');
  };
  toggle.addEventListener('click', () => {
    const next = effectiveTheme() === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem(THEME_KEY, next); } catch { /* storage unavailable */ }
    paint();
    window.dispatchEvent(new Event('themechange'));
  });
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
    if (!storedTheme()) {
      paint();
      window.dispatchEvent(new Event('themechange'));
    }
  });
  paint();
  inner.append(brand, nav, toggle);
  header.append(inner);
  document.body.prepend(header);

  const footer = document.createElement('footer');
  footer.className = 'site-footer';
  footer.innerHTML = '<div class="inner">MCMC Lab — interactive companion to <em>MCMC from Scratch</em> (Hanada &amp; Matsuura, Springer 2022). Every sampler here is written from scratch in plain JavaScript. <a href="tests.html">Sampler checks</a></div>';
  document.body.append(footer);
}

export const $ = (id) => document.getElementById(id);

export function setText(id, text) {
  const el = $(id);
  if (el && el.textContent !== text) el.textContent = text;
}

/**
 * Wire an <input type="range"> to its <output for="...">.
 * With `log: true` the slider's min/max/value are log10 of the actual value.
 */
export function bindRange(id, { format = String, log = false, onInput } = {}) {
  const input = $(id);
  const out = document.querySelector(`output[for="${id}"]`);
  const get = () => (log ? 10 ** Number(input.value) : Number(input.value));
  const show = () => { if (out) out.textContent = format(get()); };
  input.addEventListener('input', () => {
    show();
    onInput?.(get());
  });
  show();
  return {
    get,
    set(v) { input.value = log ? Math.log10(v) : v; show(); },
    setBounds(min, max) { input.min = log ? Math.log10(min) : min; input.max = log ? Math.log10(max) : max; },
    input,
  };
}

/**
 * requestAnimationFrame loop with play/pause. tick(dt) runs each frame while playing;
 * return false from it to stop. Frames are skipped while the lab is scrolled out of view.
 */
export class Runner {
  constructor({ tick, button, root, onChange }) {
    this.tick = tick;
    this.button = typeof button === 'string' ? $(button) : button;
    this.onChange = onChange;
    this.running = false;
    this.visible = true;
    this.last = 0;
    this.frame = this.frame.bind(this);
    if (root && 'IntersectionObserver' in window) {
      new IntersectionObserver((entries) => {
        this.visible = entries[0].isIntersecting;
      }).observe(typeof root === 'string' ? $(root) : root);
    }
    this.button?.addEventListener('click', () => this.toggle());
    this.paint();
  }
  paint() {
    if (!this.button) return;
    this.button.textContent = this.running ? 'Pause' : 'Run';
    this.button.setAttribute('aria-pressed', String(this.running));
  }
  start() {
    if (this.running) return;
    this.running = true;
    this.last = performance.now();
    requestAnimationFrame(this.frame);
    this.paint();
    this.onChange?.(true);
  }
  stop() {
    if (!this.running) return;
    this.running = false;
    this.paint();
    this.onChange?.(false);
  }
  toggle() { this.running ? this.stop() : this.start(); }
  frame(now) {
    if (!this.running) return;
    const dt = Math.min(0.1, (now - this.last) / 1000);
    this.last = now;
    if (this.visible && this.tick(dt) === false) {
      this.stop();
      return;
    }
    requestAnimationFrame(this.frame);
  }
}

/** Run fn at most once per `ms` milliseconds (trailing call guaranteed). */
export function throttle(fn, ms) {
  let last = 0, timer = null;
  return (...args) => {
    const now = performance.now();
    const wait = ms - (now - last);
    if (wait <= 0) {
      last = now;
      fn(...args);
    } else if (!timer) {
      timer = setTimeout(() => {
        timer = null;
        last = performance.now();
        fn(...args);
      }, wait);
    }
  };
}

/** Accumulates fractional work across frames: rate per second -> whole units this frame. */
export class Pacer {
  constructor() { this.acc = 0; }
  take(rate, dt, max = Infinity) {
    this.acc += rate * dt;
    const k = Math.min(max, Math.floor(this.acc));
    this.acc -= k;
    if (this.acc > 1) this.acc = Math.min(this.acc, 1);
    return k;
  }
  reset() { this.acc = 0; }
}
