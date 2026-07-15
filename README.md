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
| GUI shell | **SDL3** — window, input, and the Vulkan surface; **Dear ImGui** (vendored submodule) draws the control panel; the shell owns its device and swapchain outright |
| GPU compute + rendering | **Framework-free Vulkan** (`ses_vk`: volk + VMA + VkFFT); shaders authored in **Slang**, offline-baked to SPIR-V by `slangc` (fetched as a build-time tool, no runtime/vcpkg dependency) |
| Physics core | CPU double-precision truth in `core/`, no GPU/GUI dependencies |
| Time propagator | **Split-operator (Fourier)** — hand-written FFT on the CPU core; VkFFT on the GPU, with hand-rolled line-FFT kernels kept as a verified alternative |
| Reinvention boundary | **Purist** — hand-roll math, FFT, physics, render logic (incl. the swapchain); reuse only SDL3 (window/input), Dear ImGui (UI), GoogleTest, the Slang shader toolchain, and vendored Vulkan infrastructure (volk / VMA / VkFFT / glslang — glslang now only as VkFFT's runtime compiler) |
| Units | Atomic units (ℏ = mₑ = e = 1) |
| Testing | **Strict TDD** + Humble Object; a windowless GPU oracle binary (`sesolver_vkcheck`) verifies every kernel |

## Layout

```
core/    Pure numerical/physics core. NO GUI, NO GPU. Fully unit-tested.
app/     SDL3 shell (window/input/main loop/swapchain + ImGui panel) + ses_vk
         (framework-free Vulkan engine and renderer) + sesolver_vkcheck
         (windowless GPU test oracle).
tests/   GoogleTest suite driving core/, test-first.
bench/   Manual micro-benchmark.
docs/    Architecture and TDD rules.
tools/   Git hooks (TDD commit-discipline guard) + CMake helpers.
```

The hard seams: **`core` depends on nothing**; `ses_vk` depends on `core` +
Vulkan infrastructure only (proven by `sesolver_vkcheck`, which links no
windowing at all); the SDL3 shell depends on both, and only raw Khronos handles
cross the shell ↔ `ses_vk` boundary. The physics is trivially testable, the GPU
engine is testable without a GUI, and the shell stays thin and replaceable
(this seam already survived one full swap: Qt → SDL3).

## Prerequisites

- CMake ≥ 3.21 and Ninja
- A C++20 compiler (MSVC 2022+, GCC ≥ 11, or Clang ≥ 14)
- A Vulkan 1.1+ GPU/driver — needed to RUN the app and `sesolver_vkcheck`;
  `core` + `tests` need none.
- No system libraries: all dependencies come from the `external/vcpkg`
  submodule (SDL3, Vulkan infra, gtest — all small; the first configure takes
  minutes, then binary-cached). Dear ImGui is the `external/imgui` submodule,
  compiled into the app (the vcpkg port's vulkan feature would link the
  vulkan-1 import library, which is forbidden in the volk world).
- Network access on first configure (vcpkg fetches sources; without the vcpkg
  toolchain, GoogleTest falls back to CMake `FetchContent`).
- **Windows: clone to a reasonably short path** (e.g. `C:\src\...`). vcpkg's
  buildtrees nest deep enough to blow the 260-char `MAX_PATH` limit from a
  deeply nested clone ("Filename too long" during `vcpkg install`).

## Build & test

```sh
git submodule update --init --recursive      # after clone (or clone --recursive)
external/vcpkg/bootstrap-vcpkg.bat           # one-time (.sh on Linux)

# From an MSVC vcvarsall x64 environment on Windows:
cmake --preset msvc-release
cmake --build --preset msvc-release
ctest --preset msvc-release
```

Linux: the same flow with the `linux-release` preset (SDL3 wants the X11 /
Wayland dev stacks for its video backends). The routinely exercised
configuration is Windows/MSVC.

Lightweight core-only loop (no vcpkg, no GUI, no GPU):

```sh
cmake -S . -B build -DSES_BUILD_APP=OFF
cmake --build build
ctest --test-dir build --output-on-failure
```

Dev-time Vulkan validation layers are **installed by default** — every preset
sets `VCPKG_MANIFEST_FEATURES=validation` (in `*-base`), so the first configure
also builds+installs the `vulkan-validationlayers` (a ~14 min one-time cost, then
binary-cached). The `vulkan-validationlayers` port forces its own dynamic linkage,
so the layer DLL + manifest land in `<build>/vcpkg_installed/<triplet>/bin` even
on the otherwise-static triplet — installing the layer is inert; it does nothing
until you enable it at runtime:

```
set SES_VK_VALIDATION=1
set VK_ADD_LAYER_PATH=<build>\vcpkg_installed\x64-windows-static\bin
set VK_LOADER_LAYERS_ENABLE=*validation*
```

(To build without the layers, add `--no-default-features`-equivalent by clearing
`VCPKG_MANIFEST_FEATURES`. See [vcpkg.json](vcpkg.json).)

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
core. `sesolver_vkcheck` (windowless) drives every GPU kernel and engine path
against CPU oracles and runs inside `ctest`; the demo arcs regress headlessly
via `--selftest-decay` / `--selftest-rabi` / `--selftest-cascade` /
`--selftest-manifold` / `--selftest-energy` / `--selftest-efield` /
`--selftest-magnetic` / `--selftest-tunnel`, and `--dump-frame` /
`--dump-frame-near` verify the render path end to end.
