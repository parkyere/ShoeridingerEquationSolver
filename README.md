# Schrödinger Equation Solver

A from-scratch, **reinvent-the-wheel** solver and 3D visualizer for the
**single-electron** time-dependent Schrödinger equation (TDSE), built for
learning. The probability cloud `|ψ(r,t)|²` evolves in real time on the GPU
and is volume-rendered as an **electron cloud** by a hand-written,
framework-free Vulkan renderer.

> Scope is deliberately bounded to a single electron. Many-electron and DFT are
> explicitly **out of scope** — solving the many-body wavefunction directly on a
> grid is exponential (`M^(3N)`) and dies at ~2 electrons. See
> [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Pillars (locked decisions)

| Decision | Choice |
|---|---|
| Language / build | C++20, CMake, pinned vcpkg submodule (static deps) |
| GUI shell | **Qt 6** — window, input, widgets, and ONE textured-triangle blit of the renderer's image; nothing else |
| GPU compute + rendering | **Framework-free Vulkan** (`ses_vk`: volk + VMA + VkFFT), shaders offline-baked to SPIR-V by glslangValidator |
| Physics core | CPU double-precision truth in `core/`, no GPU/GUI dependencies |
| Time propagator | **Split-operator (Fourier)** — hand-written FFT on the CPU core; VkFFT on the GPU, with hand-rolled line-FFT kernels kept as a verified alternative |
| Reinvention boundary | **Purist** — hand-roll math, FFT, physics, render logic; reuse only Qt (shell), GoogleTest, and vendored Vulkan infrastructure (volk / VMA / VkFFT / glslang) |
| Units | Atomic units (ℏ = mₑ = e = 1) |
| Testing | **Strict TDD** + Humble Object; a zero-Qt GPU oracle binary (`sesolver_vkcheck`) verifies every kernel |

## Layout

```
core/    Pure numerical/physics core. NO Qt, NO GPU. Fully unit-tested.
app/     Qt shell (window/input/UI) + ses_vk (framework-free Vulkan engine
         and renderer) + sesolver_vkcheck (zero-Qt GPU test oracle).
tests/   GoogleTest suite driving core/, test-first.
bench/   Manual micro-benchmark.
docs/    Architecture, TDD rules, physics roadmap.
tools/   Git hooks (TDD commit-discipline guard) + CMake helpers.
```

The hard seams: **`core` depends on nothing**; `ses_vk` depends on `core` +
Vulkan infrastructure only (proven by `sesolver_vkcheck`, which links zero Qt);
the Qt shell depends on both, and only raw Khronos handles cross the
shell ↔ `ses_vk` boundary. The physics is trivially testable, the GPU engine is
testable without a GUI, and Qt stays a thin, replaceable shell.

## Prerequisites

- CMake ≥ 3.21 and Ninja
- A C++20 compiler (MSVC 2022+, GCC ≥ 11, or Clang ≥ 14)
- A Vulkan 1.1+ GPU/driver — needed to RUN the app and `sesolver_vkcheck`;
  `core` + `tests` need none.
- No system Qt: all dependencies, static Qt included, come from the
  `external/vcpkg` submodule. The FIRST configure builds Qt from source
  (slow once, then binary-cached).
- Network access on first configure (vcpkg fetches sources; without the vcpkg
  toolchain, GoogleTest falls back to CMake `FetchContent`).

## Build & test

```sh
git submodule update --init --recursive      # after clone (or clone --recursive)
external/vcpkg/bootstrap-vcpkg.bat           # one-time (.sh on Linux)

# From an MSVC vcvarsall x64 environment on Windows:
cmake --preset msvc-release
cmake --build --preset msvc-release
ctest --preset msvc-release
```

Linux: the same flow with the `linux-release` preset (Qt's xcb platform plugin
needs the X11 dev stack: `libx11-dev libxkbcommon-dev libfontconfig1-dev
'^libxcb.*-dev'`). The routinely exercised configuration is Windows/MSVC.

Lightweight core-only loop (no vcpkg, no Qt, no GPU):

```sh
cmake -S . -B build -DSES_BUILD_APP=OFF
cmake --build build
ctest --test-dir build --output-on-failure
```

Dev-time Vulkan validation layers are an opt-in vcpkg manifest feature
(`VCPKG_MANIFEST_FEATURES=validation`); see [vcpkg.json](vcpkg.json) for the
activation environment variables.

### Visual Studio

Open the folder; VS picks up [CMakePresets.json](CMakePresets.json) — choose
**`msvc-release`** (avoid Debug for the suite: the 3D physics tests crawl
unoptimized). Select **`sesolver_app.exe`** as the startup item and press F5.
Tests run via the Test Explorer or `ctest --preset msvc-release`.

## Working agreement

This project follows **strict TDD**. No production code is written without a
failing (red) test first, and **test commits and production commits are never
mixed**. Read [docs/TDD_RULES.md](docs/TDD_RULES.md) before contributing, then
enable the guard hook once:

```sh
git config core.hooksPath tools/git-hooks
```

## What it does

At startup the app **solves the atom first**. The spherical potential reduces
exactly to 1D per angular momentum, so the hand-rolled radial engine (Sturm
bisection + inverse iteration) finds every bound level and its E1 lifetime —
the full spectrum table prints to the console. The 3D tracked manifold is the
full m-resolved bound shell the box can hold, synthesized directly on the GPU
as (u/r)·Y_lm, with every allowed decay channel's Einstein A reduced on the GPU
from OUR wavefunctions and the s-state energies audited against the grid
Hamiltonian on every launch. The Δl = ±1 rule is applied analytically; the
Δm selection physics **emerges numerically** from the computed dipoles
(forbidden channels come out suppressed by ~17 orders of magnitude, and
branching ratios like 3d_z² → 2p at 4:1:1 reproduce Clebsch-Gordan). Then the
wavepacket demo begins with spontaneous decay ARMED, as in nature. Energies are
shown in eV.

Controls:

- **1** real time / **2** relax to 1s / **3** relax to 2p_z / **4** relax to 2s
  (deflated imaginary time with live energy readout; relaxation AUTO-COMPLETES
  on convergence so the prepared state visibly decays) / **R** reset;
- **5** excite an n=3/4 state and watch the CASCADE: 3d → 2p (photon) → 1s
  (photon), or the triple chain 4f → 3d → 2p → 1s — the Δl = ±1 ladder;
- **M** soft Gaussian position measurement (a POVM whose width is chosen so the
  Heisenberg back-action localizes the electron without routinely ionizing the
  atom) / **E** projective energy measurement (collapse onto an eigenstate
  sampled by |⟨n|ψ⟩|², or an honest "outside the tracked manifold" outcome);
- **D** toggles decay — multi-channel quantum jumps competing as Poisson
  processes over the whole manifold, with ONE common, honestly-labeled display
  acceleration so relative lifetimes stay physical. Photon flash + counter +
  last-jump label;
- **L** resonant laser at ω = E(2p) − E(1s): Z-polarization Rabi-pumps
  1s → 2p_z (live populations), X-polarization pumps 2p_x instead — and its
  fluorescence clicks through the 2p_x decay channel. Laser + decay = repeated
  absorb/emit cycles;
- **static E-field (+z)** and **magnetic field (z/x/y)** sliders solve the
  field Hamiltonian PROPERLY: the E-field is a dipole term folded into the
  half-potential (Stark polarization / field ionization); the B-field is the
  paramagnetic (B/2)L Larmor rotation (exact unitary three-shear via the
  Fourier shift theorem) plus the diamagnetic (B²/8)ρ⊥² term, so a p_x cloud
  genuinely precesses into p_y. Crossed E-B works; a boundary absorbing mask
  stops ionized flux wrapping the periodic FFT box; a live readout gives the
  semiclassical Larmor radiated power of the oscillating dipole;
- **F** toggles probability-current flow particles (Bohmian tracers advected by
  v = j/ρ, sampled from |ψ|²);
- **Tab** switches to the marching-cubes isosurface view; drag orbits, wheel
  zooms, space pauses, **[ ]** tunes cloud density.

The cloud itself is rendered with an HDR pipeline: phase-tinted front-to-back
raymarch with interleaved-gradient-noise jitter and temporal accumulation,
occupancy-based empty-space skipping, an extinction self-shadow volume, and
dual-Kawase bloom under an ACES tonemap.

Verification: all shader math is transcribed from the unit-tested CPU double
core. `sesolver_vkcheck` (zero Qt) drives every GPU kernel and engine path
against CPU oracles and runs inside `ctest`; the demo arcs regress headlessly
via `--selftest-decay` / `--selftest-rabi` / `--selftest-cascade` /
`--selftest-manifold` / `--selftest-energy` / `--selftest-efield` /
`--selftest-magnetic`, and `--dump-frame` / `--dump-frame-near` verify the
render path end to end.
