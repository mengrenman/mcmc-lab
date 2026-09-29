# MCMC Lab

An interactive web app for learning Markov chain Monte Carlo, built as a companion to
*MCMC from Scratch* by Masanori Hanada and So Matsuura (Springer, 2022). Every sampler is
written from scratch in plain JavaScript. There is no build step and there are no dependencies.

## Run it

Browsers refuse to load JavaScript modules from `file://` URLs, so serve the folder locally:

```bash
python3 serve.py
```

Then open <http://127.0.0.1:8000/>. `serve.py` is a small wrapper around Python's built-in
server that turns off caching, so edits to the JavaScript show up on reload. Plain
`python3 -m http.server` works too, but it may serve stale modules while you edit.

## Modules

| Page | Book | What you can do |
|---|---|---|
| `monte-carlo.html` | ch. 2 | Estimate π with darts and watch the error fall like 1/√N. Compare uniform and importance sampling, including a proposal with infinite variance. |
| `metropolis.html` | ch. 3–4 | Step through accept/reject decisions on three targets. Read the trace, histogram, autocorrelation, τ_int, ESS and jackknife error bars. |
| `hmc-gibbs.html` | ch. 5 | Animate HMC leapfrog trajectories and their energy. Race Metropolis, Gibbs and HMC on a correlated Gaussian or a banana, with ESS per iteration and per unit of work. |
| `bayes.html` | §6.1 | Combine a prior with the likelihood of coin tosses and watch the posterior update. Sample it with Metropolis and check against the exact answer. Infer the mean and width of a Gaussian in 2D. |
| `ising.html` | §6.2 | Run the 2D Ising model with Metropolis, heat-bath or Wolff updates. Scan temperatures to measure ⟨\|m\|⟩, ⟨e⟩ and critical slowing down against Onsager's exact results. |
| `optimization.html` | §6.3 | Watch simulated annealing leave half its walkers in the wrong valley of the book's double well, then let replica exchange fix it. Solve the traveling salesman problem with replica exchange, against brute force and greedy descent. |
| `lattice.html` | §6.4 | The algorithms of lattice QCD on the 2D U(1) toy model: gauge fields on links, Wilson loops and the area law, topological freezing, and two quark flavors via pseudofermion HMC with a conjugate-gradient solver. |
| `tests.html` | — | Statistical and exact checks of every sampler (see below). |

Chart conventions used throughout: a gray line is the exact answer, blue/orange/aqua are
simulation output, and green ✓ / red ✗ mark accepted and rejected proposals.

## Layout

```
css/style.css          design tokens (light and dark) and layout
js/lib/random.js       seeded xoshiro128** generator, Box–Muller normals
js/lib/stats.js        FFT autocorrelation, τ_int with Sokal's window, ESS, jackknife, histograms
js/lib/plot.js         small canvas plotting layer with hover tooltips
js/lib/targets.js      target densities and their exact moments
js/lib/bayes.js        coin priors and posteriors on a grid, Gaussian (μ, σ) posterior, log Γ
js/lib/ui.js           header, theme toggle, sliders, animation loop
js/samplers/           metropolis.js, hmc.js, gibbs.js, ising.js, annealing.js, tsp.js,
                       gauge.js, schwinger.js (no DOM code)
js/pages/              one script per page
js/tests.js            the checks behind tests.html
```

The samplers do not touch the DOM, so they can be reused or tested on their own.

## Checks

`tests.html` runs 50 checks in the browser: RNG moments, FFT autocorrelation against a direct
sum, τ_int of an AR(1) process against its exact value, sampler moments against exact
expectations (each within 4 of its own autocorrelation-aware error bars), leapfrog
reversibility and ε² energy scaling, Ising energy bookkeeping, Onsager's exact results,
agreement between the three Ising algorithms, Bayesian posteriors against closed forms and
brute-force integration, replica exchange against exact equilibrium and brute-force TSP optima,
and the lattice code against exact torus results. The Schwinger-model pseudofermion HMC is
checked against an independent exact-determinant simulation, and its quark force against
finite differences. The first run uses fixed seeds, and the button
reruns with fresh ones.

One of those checks exists because of a bug it caught. An earlier Wolff "sweep" stopped as soon
as N spins had flipped. That stopping rule depends on the state (the cluster that crosses the line
tends to be large), and it biased ⟨e⟩ by about 9 error bars. A sweep is now a fixed number of
clusters, calibrated after each temperature change.

## Coverage

Every chapter of the book from 2 to 6. Section 6.4 (lattice QCD) runs on the two-dimensional U(1)
toy model, not full four-dimensional SU(3) QCD, using the same algorithms: HMC, pseudofermions
and conjugate gradient.
