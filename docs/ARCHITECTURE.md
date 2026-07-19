# Architecture

## Goal & scope

Solve and visualize the **single-electron** time-dependent Schrödinger equation
(TDSE) in atomic units:

```
i ∂ψ/∂t = H ψ,    H = ½(-i∇ - A)² + V(r, t)
```

The app renders `|ψ(r,t)|²` in real time — as a volume-raymarched **electron
cloud** for the 3D scenes, as a phasor curve + phase-hued shadow band for the
1D scenes, and as a face-on plane for the 2D lattice scenes. Fifteen scenes
(see [README](../README.md#scenes)) cover the hydrogen atom, traps, tunneling,
molecules, interference (double slit / Aharonov–Bohm), and solid-state physics
(Landau levels, Bloch oscillations, the quantum corral, a quantum dot).

**Out of scope (by design):** multiple electrons, Hartree/HF/DFT, the many-body
wavefunction. The exact `N`-electron wavefunction lives in `3N` dimensions, so a
grid costs `M^(3N)` — at `M=50` that is ~250 GB for `N=2` and ~31 PB for `N=3`.
Direct FDM dies at ~2 electrons; "arbitrary electrons" would mean a different,
research-grade mean-field (DFT) project. We stay single-electron on purpose.

## The seams

```
                 +-------------------+
 tests/ -------->|   sesolver_core   |   pure physics: no GUI, no GPU,
 (gtest)         |   fully testable  |   fully unit-tested
                 +-------------------+
                           ^
                 +-------------------+
                 | solver/ ses_solver|   Schrodinger GPGPU library
                 | volk+VMA+Slang    |   (sesolver_vkcheck = windowless
                 +-------------------+    GPU oracle, runs inside ctest)
                    ^             ^
 +---------------------+   +----------------------+
 |    viz/ ses_viz     |   | scenario/ ses_scenario|
 | raw-Vulkan renderer |   | physics orchestration |
 | + swapchain present |   | (directors, arcs)     |
 +---------------------+   +----------------------+
            ^                     ^
 +--------------------------------------+
 |  app/ SDL3 shell + ImGui panel       |   pure presentation
 |  ("Humble Object")                   |
 +--------------------------------------+
```

- **`core` depends on nothing** (no GUI, no GPU, no windowing). Every behavior
  has an analytic or golden oracle and is unit-tested, test-first. Everything
  is C++20 named modules (`ses.*`) — no project header files anywhere.
- **`solver/`** (`ses_solver`, modules `ses.vk.device/compute/engine/...`) is
  the Schrodinger GPGPU library: framework-free Vulkan 1.3 on top of `core`
  plus vendored infrastructure only (volk, VMA, VkFFT, offline-baked Slang
  kernels). It links no windowing at all — proven by `sesolver_vkcheck`,
  which drives every kernel and engine path against the CPU double core
  inside ctest, on machines with no SDL installed.
- **`viz/`** (`ses_viz`, `ses.vk.render/present/...`) is the raw-Vulkan scene
  renderer (volume raymarch, HDR post chain, overlays, marker spheres) and
  the swapchain presenter. It rides the solver's device and knows nothing
  about SDL or ImGui (the presenter takes an opaque UI-record callback).
- **`scenario/`** (`ses_scenario`, `ses.scenario.*`) is everything a demo IS:
  the `ScenarioDirector` seam, the atom model, all scene directors, and the
  headless selftest arcs. Physics orchestration only — no UI.
- **The shell is thin and replaceable**: SDL3 gives the window, input, and the
  Vulkan surface; the shell creates the device (the same
  `DeviceContext::create` path vkcheck exercises) and drives `ses.vk.present`
  (viz) for the swapchain + fullscreen present pass, with the Dear ImGui
  panel riding that pass. Only raw Khronos handles and CPU meshes cross the
  seams. The shell holds **no domain logic**; anything worth testing is
  pushed down into `core` as pure data/geometry first.
- Dependency direction points **inward**:
  `app → scenario/viz → solver → core`, `tests → core + scenario`. `core`
  never points outward.

## Scene architecture

The app owns exactly **one `ScenarioDirector`** (picked by `--scene=` or the
panel combo). Scene-specific controls are reached only through capability
accessors (`hydrogen()`, `tunnel()`, `slit()`, `landau()`, ...) that return
`nullptr` unless implemented — no down-casts anywhere. Scene keys funnel
through `handle_key(char)`.

Three director families implement the seam:

| Family | Scenes | Compute |
|---|---|---|
| `BaseDirector` | hydrogen, 3D trap, 3D tunnel, H₂⁺, benzene | GPU engine, 256³, CPU double truth + fp32 GPU mirror under one sync invariant (`cpu_is_truth_`) |
| `Line1DDirector` | six 1D scenes (HO, tunnel, double well, Pöschl–Teller, Morse, Bloch) | CPU double, 64k-point grids (Bloch 4096), overlay-polyline display |
| `Lattice2DDirectorBase` (+ two standalone directors) | corral, qdot (base); double slit, Landau (standalone) | CPU Peierls lattice on one z-plane, displayed by replicating into a thin volume slab |

Contracts every family obeys:

- **Pacing:** `tick()` supplies at most ONE tick's worth of steps
  (`pending = min(pending + per_tick, per_tick)`) — catch-up ticks drop
  instead of bundling; `run_frame()` consumes all pending once per paint.
  The Time-scale dial multiplies steps per frame (`dt` untouched, clamp
  1..16) and is the ONLY pacing multiplier — no mode may add a hidden one.
- **Real time = x1:** `ScenarioDirector::set_real_time()` is a non-virtual
  NVI wrapper — `do_set_real_time()` then `set_time_scale(1)` — so every
  route back to real time structurally clears the dial.
- **Scene props are renderer-only hints** (marker spheres, overlay curves,
  boot camera); physics never reads them.

## Time dependence & gauge fields

Time-dependent and magnetic terms enter without disturbing the tested static
Strang tables:

- **Laser (dipole drive):** `ses.drive` wraps time-dependent half-kicks
  `exp(-i E₀ cos(ωt)(ε̂·r) dt/2)` AROUND the untouched split-operator
  tables — global O(dt²) survives. Carrier and amplitude come from the
  solver's own spectrum (grid resonance, target Rabi Ω over the computed
  dipole element), never textbook constants.
- **Uniform B (3D):** minimal coupling split exactly — the diamagnetic
  `(B²/8)ρ⊥²` term folds into the potential table; the paramagnetic
  `exp(-i(B/2)L_axis dt)` factor is an exact three-shear (Paeth) rotation,
  each shear applied per-line via the Fourier shift theorem (`ses.rotation`,
  norm-conserving, no interpolation). Laser and B are mutually exclusive
  (the driven step applies the diamagnetic fold but not the rotation).
- **Localized / lattice flux (2D):** FFT split-operator cannot Trotterize
  `(p−A)²/2` for a localized flux, so the 2D scenes use
  `ses.lattice2d::PeierlsLattice2D` — a finite-difference bond propagator
  whose kinetic term is 4 disjoint bond groups of exact 2×2 rotations in a
  Strang palindrome (unitary to round-off). Flux enters EXACTLY as Peierls
  link phases: a string-gauge solenoid is purely topological (B = 0 on every
  reachable plaquette — the Aharonov–Bohm setup), a uniform field uses the
  Landau gauge `A_x = +By` anchored at y = 0. The imaginary-time twin keeps
  the link phases, so grounds IN a field (Fock–Darwin) are reachable.
- **Bloch tilt (1D):** `ses.bloch` uses a comoving gauge `A(t) = -Ft`,
  rebuilding the kinetic table each step at the midpoint drift (exact for
  linear A); bands come from the sin² lattice's central equation solved by
  Sturm bisection in ratio form.

## The atom machinery

- **Radial reduction:** a central potential reduces exactly to 1D per angular
  momentum. `ses.radial` solves the symmetric tridiagonal FD Hamiltonian per
  `l` (Sturm bisection + shifted inverse iteration); `ses.spectrum1d` reuses
  the same engine as the bound-state oracle for the solvable 1D wells.
- **`AtomModel`** parameterizes the tracked-manifold machinery over ANY
  central potential (hydrogen: the full m-resolved n≤6 shell = 91 states;
  the 3D trap: the N≤3 Fock ladder). States are synthesized on the GPU
  on demand as `(u/r)·Y_lm`; the synthesis captures per-state grid norms
  that normalize every projection and MCWF term.
- **Orbital-free projection:** no resident orbital atlas. One grid pass
  deposits `g_lm` components binned by radius (the deposit weights are
  contractually bit-identical between `ses.harmonics::fill_orbital` and the
  Slang deposit kernel); each amplitude `<n|ψ>` is then a 1D radial dot.
  The startup "atlas" is a transient montage — one orbital resident at a
  time, shown, audited, freed.
- **Factorized E1 channel table:** decay rates are
  `(energy gap)³ × (constexpr tesseral angular factor) × (1D radial dipole
  integral)²` — built instantly on the CPU; selection rules are hard zeros
  of the angular factor. Δm physics *emerges* from the same table.
- **Decay is QED, not TDSE:** competing-channel Poisson quantum jumps over
  accumulated dt; a jump collapses onto a synthesized eigenstate followed by
  a fixed post-collapse imaginary-time flush (cusp-junk removal, contract
  in `tests/eigenstate_flush_test.cpp`). Between jumps, optional MCWF
  no-jump damping. One shared display-acceleration factor keeps relative
  lifetimes physical. Emission rules are hydrogen-only by policy.

## Reuse boundary (purist reinvention)

The from-scratch mandate covers the **physics/numerics core** (the learning
target); infrastructure reaches for established libraries. The C++ standard
library is always fair game: `std::complex` (built with limited-range complex
arithmetic where supported), `std::vector`, `<cmath>`, `<random>` etc.

Hand-written (the learning lives here):
- vector/matrix/camera math (no GLM)
- FFT — 1D radix-2 → N-D on the CPU (the tested truth), plus hand-rolled
  compute-shader line-FFT kernels kept as a verified GPU alternative
- grid, complex field, finite-difference / spectral operators
- time propagation: split-operator Fourier (real + imaginary time), the
  Peierls bond propagator, the three-shear rotation, ladder operators,
  radial/tridiagonal eigen solvers
- potentials (harmonic, regularized bare Coulomb, lattices, molecules)
- visualization geometry/color math (marching cubes, transfer functions)
- all Vulkan rendering/compute logic: the raymarched volume renderer, the
  HDR post chain, the swapchain/present layer, every pipeline and barrier

Reused (plumbing, not the learning target):
- **SDL3** — window, input, Vulkan surface
- **Dear ImGui** (vendored submodule, `IMGUI_IMPL_VULKAN_USE_VOLK`)
- **GoogleTest**; **Boost.Program_options** (CLI parsing)
- **Vendored Vulkan infrastructure** — volk, VMA, VkFFT (production GPU FFT),
  glslang (VkFFT's runtime compiler); Slang (`slangc`) as the offline shader
  compiler

Explicitly *not* reused: GLM, Qt, FFTW, Eigen/BLAS/LAPACK, OpenMP.

## Numerical decisions

- **Atomic units** (ℏ = mₑ = e = 1) everywhere; eV/fs only at the display
  boundary.
- **CPU double is the truth; the GPU mirrors it.** Every shader transcribes
  unit-tested CPU math (the `ses.volume` "shader truth" pattern: raymarch
  math is pinned by CPU tests, the fragment shader transcribes it line by
  line; the phase colormap is baked to a LUT from the tested formula).
- **Determinism:** CPU parallelism is the project-owned `ses.parallel` pool —
  chunk boundaries depend only on `n`, reduction partials combine in fixed
  order, so threaded results are bitwise identical to serial for any worker
  count. (OpenMP is gone: MSVC miscompiles `#pragma omp` inside exported
  module functions.)
- **MSVC modules footguns (load-bearing):** function-local statics must live
  in NON-inline module functions (the pool accessor), and hot classes define
  members OUT of class in the interface (PeierlsLattice2D) — inline
  definitions instantiate per importing TU and broke both.
- **Grid contract:** uniform periodic grids (`xmax` aliases `xmin`), 3D flat
  layout x-fastest — FFT-contiguous lines and tight GPU 3D-texture uploads.
- **Never** `-ffast-math` / `/fp:fast`; FMA contraction stays off so the
  exact-value FP test oracles hold bitwise.
- **Regularize the bare Coulomb `-Z/r`** rather than softening it: the
  singular nucleus cell takes the analytic cell average `-Z·C/h`
  (`C ≈ 2.380`); every other cell keeps exact `-Z/r`; molecular centers snap
  to grid points so every center's cell is regularized. Radial solves feed
  bare `-Z/r` and reproduce the textbook spectrum. Softening instead would
  push `E(1s)` from -13.6 eV up to ~-9 eV. **No soft-Coulomb anywhere** (a
  standing project rule).
- **VRAM policy:** resident-state precision (fp32 vs fp16) is a pure-integer
  decision (`ses.vram_budget`) from the `VK_EXT_memory_budget` probe
  (`ses.vk.vram_probe`); an unmeasurable budget never silently degrades —
  it keeps fp32, and audit-critical states are exempt from fp16.

## Validation strategy

Validation is the spine, not a phase. Each numerical layer ships with an
analytic oracle as its red test:
- FFT: linearity, `IFFT(FFT(x)) = x`, Parseval, known transforms.
- Free propagation: 1D Gaussian dispersion `σ(t) = σ₀·√(1 + (t/(2σ₀²))²)`,
  center at `k₀`, norm 1.
- Bound dynamics: coherent-state oscillation; imaginary-time stationary
  states match `E` and shape; the solvable wells (Morse, PT) match closed
  forms; the lattice propagator matches gauge-invariance and plaquette-flux
  contracts.
- Every discretization carries a **convergence-order** test (error ~ `h^p`).
- **GPU kernels:** `sesolver_vkcheck` (windowless, ctest label `gpu`)
  compares every kernel and engine path against the CPU double core with
  fp32 tolerances, and fails on any validation-layer error. Every kernel
  lands together with its vkcheck comparison — the GPU analogue of
  red/green.
- **Scene behavior:** every scene regresses headlessly through the app's
  `--selftest-*` arcs (`--help` lists them). Arcs are condition-polled and
  scheduler-chained (manifold-ready probes, sim-time targets) rather than
  wall-clock-deadlined, so slower GPUs stretch runs instead of false-failing.
  `--dump-frame*` arcs verify the render path end to end.

## Build topology

`CMakeLists.txt` (root) → `core/` (always) → slangc/VkFFT CMake helpers →
`solver/` → `viz/` → `scenario/` (all windowing-free, always built) →
`tests/` (if `SES_BUILD_TESTS`; links core + scenario) → `bench/` (if
`SES_BUILD_BENCH`) → `app/` (if `SES_BUILD_APP` **and** SDL3 found). The app
is optional so the TDD loop never requires a GUI toolchain. Shaders are
authored in Slang and offline-baked to SPIR-V headers by `slangc` at build
time.
