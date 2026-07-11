# Architecture

## Goal & scope

Solve and visualize the **single-electron** time-dependent Schrödinger equation
(TDSE) in atomic units:

```
i ∂ψ/∂t = H ψ,    H = -½ ∇² + V(r, t)
```

First deliverable: a **Gaussian wavepacket** evolving in real time (free space,
then a potential), with `|ψ(r,t)|²` rendered as an electron cloud.

**Out of scope (by design):** multiple electrons, Hartree/HF/DFT, the many-body
wavefunction. The exact `N`-electron wavefunction lives in `3N` dimensions, so a
grid costs `M^(3N)` — at `M=50` that is ~250 GB for `N=2` and ~31 PB for `N=3`.
Direct FDM dies at ~2 electrons; "arbitrary electrons" would mean a different,
research-grade mean-field (DFT) project. We stay single-electron on purpose.

## The seams

```
                 +-------------------+
 tests/ -------->|   sesolver_core   |   pure physics: no Qt, no GPU,
 (gtest)         |   fully testable  |   fully unit-tested
                 +-------------------+
                    ^             ^
                    |             |
 +---------------------+      +--------------------------+
 |       ses_vk        |<-----|      app/ Qt shell       |
 | framework-free      |      | window + input + UI +    |
 | Vulkan compute AND  |      | ONE blit of the rendered |
 | rendering           |      | image ("Humble Object")  |
 +---------------------+      +--------------------------+
           ^
           |
 sesolver_vkcheck  (zero-Qt GPU oracle, runs inside ctest)
```

- **`core` depends on nothing** (no Qt, no GPU, no windowing). Every behavior
  has an analytic or golden oracle and is unit-tested, test-first.
- **`ses_vk`** (`app/src/vk_*.hpp`) is the GPU engine and scene renderer:
  framework-free Vulkan on top of `core` plus vendored infrastructure only
  (volk, VMA, VkFFT). It contains zero Qt — proven by `sesolver_vkcheck`,
  which links no Qt and drives every kernel and engine path against the CPU
  double core inside ctest.
- **The Qt shell is thin and replaceable**: window, input, widgets/UI, and one
  fullscreen-triangle blit of the renderer's image. Only raw Khronos handles
  cross the shell <-> `ses_vk` seam, so swapping Qt for another windowing
  layer touches the shell alone. The shell holds **no domain logic**; anything
  worth testing is pushed down into `core` as pure data/geometry first (e.g.
  marching-cubes vertex generation, transfer-function math, camera matrices).
- Dependency direction points **inward**: `app → ses_vk → core`,
  `tests → core`. `core` never points outward.

## Reuse boundary (purist reinvention)

The boundary applies to THIRD-PARTY libraries only. The **C++ standard
library is always fair game** (user decision): `ses::Complex` is an alias of
`std::complex` (built with `-fcx-limited-range` so multiply/divide use the
naive formulas the exact-value test oracles pin), and `std::vector`,
`<cmath>`, `<numbers>`, `<random>` etc. are used freely.

Hand-written (the learning lives here):
- vector/matrix/camera math (no stdlib equivalent until C++26 `<linalg>`)
- FFT — 1D radix-2 → N-D on the CPU (the tested truth), plus hand-rolled
  compute-shader line-FFT kernels kept as a verified GPU alternative
- grid, complex field, finite-difference / spectral operators
- time propagation (split-operator Fourier method, real and imaginary time)
- potentials (harmonic, regularized bare Coulomb)
- visualization geometry/color math (marching cubes, transfer functions)
- all Vulkan rendering/compute logic: the raymarched volume renderer, the HDR
  post chain, and every pipeline, descriptor, and barrier in `ses_vk`

Reused (pure plumbing, not the learning target):
- **Qt 6** — window, input, widgets/UI, and the one-blit presentation only
- **GoogleTest** — test harness
- **Vendored Vulkan infrastructure** — volk (loader), VMA (allocation),
  VkFFT (the production GPU FFT), glslang (offline GLSL → SPIR-V bake)

VkFFT is the one deliberate exception to "hand-roll the FFT": the CPU FFT
remains hand-written and IS the truth oracle, the hand-rolled GPU line-FFT
kernels remain verified as the learning artifact, and VkFFT is production
plumbing chosen for speed.

Explicitly *not* reused: GLM (we hand-roll math), GLFW/SDL (Qt is the shell),
FFTW (the CPU FFT is hand-rolled), Eigen/BLAS/LAPACK.

## Numerical decisions

- **Atomic units** (ℏ = mₑ = e = 1) everywhere; SI only at the I/O boundary.
- **Complex fields** are first-class (TDSE is inherently complex).
- **Propagator: split-operator (Fourier).** One step of `exp(-iHΔt)` is split as
  `exp(-iVΔt/2) · exp(-iTΔt) · exp(-iVΔt/2)`, with the kinetic factor applied in
  k-space (diagonal there) — via the hand-written FFT on the CPU truth path,
  via VkFFT on the GPU. Unitary and norm-preserving.
- **Never** `-ffast-math` / `/fp:fast` — it breaks NaN handling and bitwise
  reproducibility that the tests rely on.
- **Regularize the bare Coulomb `-Z/r`** rather than softening it: the single
  cell on the nucleus (where `-Z/r` = -∞) takes the analytic cell average
  `-Z·C/h` (`C = ∫1/r` over the unit cube ≈ 2.380), while every other cell keeps
  the exact `-Z/r`. This yields the textbook hydrogen spectrum (`E(1s) = -13.6
  eV`, exact `2s = 2p` degeneracy); the older soft-Coulomb `-Z/√(r²+a²)` rounded
  the whole well and pushed `E(1s)` up to -9 eV. The radial solves feed bare
  `-Z/r` directly (their grid `r = (i+1)h` never hits 0).

## Validation strategy

Validation is the spine, not a phase. Each numerical layer ships with an
analytic oracle as its red test:
- FFT: linearity, `IFFT(FFT(x)) = x`, Parseval, known transforms of δ and cosines.
- Free propagation: a 1D Gaussian wavepacket disperses analytically —
  `σ(t) = σ₀·√(1 + (t/(2σ₀²))²)` (atomic units, mₑ=1), center moves at `k₀`.
  Norm stays 1.
- Bound dynamics: harmonic-oscillator coherent state oscillates rigidly;
  stationary states (later, via imaginary time) match `E` and shape.
- Every discretization carries a **convergence-order** test (error ~ `h^p`).
- **GPU kernels**: the GPU runs fp32 for display speed; the CPU double core
  stays THE tested truth. `sesolver_vkcheck` compares every kernel and engine
  path against the CPU core element-by-element with fp32 tolerances (each
  check documents its oracle and tolerance rationale), fails on any Vulkan
  validation-layer error, and is registered with ctest (label `gpu`). Every
  kernel lands together with its vkcheck comparison — the GPU analogue of
  red/green. The demo arcs regress headlessly via the app's `--selftest-*`
  flags, and `--dump-frame` / `--dump-frame-near` verify the render path.

## Build topology

`CMakeLists.txt` (root) → `core/` (always) → `tests/` (if `SES_BUILD_TESTS`) →
`bench/` (if `SES_BUILD_BENCH`, the `sesolver_bench` micro-benchmark) →
`app/` (if `SES_BUILD_APP` **and** Qt6 found). The app and bench are optional
so the TDD loop never requires a Qt toolchain.
