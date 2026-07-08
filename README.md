# Schrödinger Equation Solver

A from-scratch, **reinvent-the-wheel** solver and 3D visualizer for the
**single-electron** time-dependent Schrödinger equation (TDSE), built for
learning. The first goal is a hydrogen-scale electron treated as a **Gaussian
wavepacket** whose probability cloud `|ψ(r,t)|²` evolves in real time and is
rendered as an **electron cloud** with hand-written OpenGL.

> Scope is deliberately bounded to a single electron. Many-electron and DFT are
> explicitly **out of scope** — solving the many-body wavefunction directly on a
> grid is exponential (`M^(3N)`) and dies at ~2 electrons. See
> [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Pillars (locked decisions)

| Decision | Choice |
|---|---|
| Language / build | C++20, CMake |
| GUI / window / context | **Qt 6** (window, GL context, widgets only) |
| Rendering | **Hand-written OpenGL 4.3 core** (no GL helper libs) |
| First physics target | **Real-time Gaussian wavepacket dynamics** (TDSE) |
| Time propagator | **Split-operator (Fourier)** using a **hand-written FFT** |
| Reinvention boundary | **Purist** — hand-roll math, FFT, physics, render logic; reuse only Qt (shell) + GoogleTest |
| Units | Atomic units (ℏ = mₑ = e = 1) |
| Testing | **Strict TDD** + Humble Object (see below) |

## Layout

```
core/    Pure numerical/physics core. NO Qt, NO OpenGL. Fully unit-tested.
app/     Qt + hand-written OpenGL shell ("Humble Object"). Manually verified.
tests/   GoogleTest suite driving core/, test-first.
docs/    Architecture, TDD rules, physics roadmap.
tools/   Git hooks (TDD commit-discipline guard).
```

The hard seam: **`core` depends on nothing**; `app` depends on `core` + Qt;
`tests` depend on `core` + GoogleTest. This makes the physics trivially testable
and confines all untestable GL/Qt code to a thin shell.

## Prerequisites

- CMake ≥ 3.21
- A C++20 compiler (MSVC 2022, GCC ≥ 11, or Clang ≥ 14)
- Qt 6 (Core Gui Widgets OpenGL OpenGLWidgets) — **only needed for the GUI**;
  `core` + `tests` build without it.
- Network access on first configure (GoogleTest is fetched via CMake `FetchContent`).

## Build & test

```sh
# Configure (core + tests only; skip the GUI):
cmake -S . -B build -DSES_BUILD_APP=OFF

# Build and run the tests:
cmake --build build
ctest --test-dir build --output-on-failure
```

To build the GUI too, install Qt 6 and configure with `-DSES_BUILD_APP=ON`
(default). If Qt 6 is not found, the GUI target is skipped with a warning and
the core/test build still succeeds.

> Windows: pass your Qt install to CMake, e.g.
> `-DCMAKE_PREFIX_PATH="C:/Qt/6.x.x/msvc2022_64"`.

### Visual Studio

Open the folder; VS picks up [CMakePresets.json](CMakePresets.json) --
choose the **`MSVC RelWithDebInfo (Qt msvc2022_64)`** preset (avoid plain
Debug: the 3D physics tests crawl at `-Od`). Then select
**`sesolver_app.exe`** in the startup-item dropdown next to the Run button
and press F5. `windeployqt` stages the Qt runtime next to the exe after
every build, so it also launches from Explorer. Tests run via the Test
Explorer or `ctest --preset msvc`.

### VSCode on Linux (remote or local)

The repo ships `.vscode/` (CMake Tools driven) + Linux presets, so **F5**
builds and runs `sesolver_app` on Linux (needs a GL 4.3 context -- any recent
NVIDIA/Mesa driver; the app opens a window, so use a real display or
X/Wayland forwarding).

1. Accept the recommended extensions (VSCode prompts): **CMake Tools** +
   **C/C++**. Have a C++20 toolchain (`clang` or `g++`), CMake >= 3.21, Ninja,
   and Qt 6 (`apt install qt6-base-dev libqt6opengl6-dev`, or a Qt install on
   `CMAKE_PREFIX_PATH`).
2. In the CMake Tools status bar pick the **`linux-native`** configure preset
   -- it turns on `SES_NATIVE` (`-march=native`, host-CPU tuned) and
   auto-detects the newest installed `clang++`/`g++`. Then pick **`sesolver_app`**
   as the launch target.
3. Press **F5**: the `cmake-build` task builds the active preset, then gdb
   launches the app.

Presets: `linux-native` (max app perf) · `linux` (portable AVX2 -- prefer it
for the **test suite**: `-march=native` on an AVX-512 host can perturb a few
bitwise oracles) · `linux-clang` · `linux-gcc`. Run tests with
`ctest --preset linux`; verify GPU kernels with `sesolver_gpucheck` (needs a
GL context). To pin an exact compiler instead of auto-detect, set
`CMAKE_CXX_COMPILER` (e.g. `-DCMAKE_CXX_COMPILER=clang++-19`) or `$CXX`.

## Working agreement

This project follows **strict TDD**. No production code is written without a
failing (red) test first, and **test commits and production commits are never
mixed**. Read [docs/TDD_RULES.md](docs/TDD_RULES.md) before contributing, then
enable the guard hook once:

```sh
git config core.hooksPath tools/git-hooks
```

## Status

Phases 0-7 plus the GPU engine and the transitions, static-field, magnetic
(Larmor), and radiation arcs delivered (216 tests green): the hand-rolled
Complex/FFT/split-operator/imaginary-time stack is validated against analytic
oracles through 3D, and `sesolver_app` renders the TDSE as a TRUE
VOLUME-RENDERED electron cloud at 256^3 -- propagation, relaxation, orbital
synthesis, and rendering all GPU-resident (OpenGL 4.3 compute; every kernel
verified against the unit-tested CPU double core by `sesolver_gpucheck`). The
full atom-and-light demo works from first principles computed by the solver
itself:

At startup the app SOLVES the atom first. The spherical potential reduces
exactly to 1D per angular momentum, so the hand-rolled radial engine
(Sturm bisection + inverse iteration) finds ALL 55 bound levels to n = 10
with their E1 lifetimes -- the full table prints to the console (soft-Coulomb
a = 0.5: 2p 3.1 ns; 2s 337 ns through the degeneracy-lifted 2s->2p channel;
10s 18 us). The 3D tracked manifold is everything the +-80 Bohr box (256^3,
h = 0.625) and the GPU physically hold: the full m-resolved n <= 6 shell --
91 states, real Y_lm to l = 5 (h orbitals), synthesized directly on the GPU
as (u/r) Y_lm (no CPU field -- the atlas builds in seconds), with the decay
channels and dipole matrix elements reduced on the GPU and E_radial audited
vs <H>_grid on every launch. The m-selection physics emerges numerically
(3d_z2 -> 2p branches 4:1:1, Clebsch-Gordan). Then the wavepacket demo
begins with spontaneous decay ARMED, as in nature. Energies are shown in eV.

- **1** real time / **2** relax to 1s / **3** relax to 2p_z / **4** relax
  to 2s (deflated imaginary time, live energy readout; relaxation
  AUTO-COMPLETES on convergence so the prepared state visibly decays);
- **5** excite an n=3/4 state and watch the CASCADE: 3d -> 2p (photon) ->
  1s (photon), or the triple chain 4f -> 3d -> 2p -> 1s -- the
  Delta-l = +-1 ladder forced by the selection rules;
- **M** soft Gaussian position measurement / **E** projective energy
  measurement (collapse onto an eigenstate sampled by |<n|psi>|^2, or an
  honest "outside the tracked manifold" outcome from the bound set's
  incompleteness);
- **D** turns decay OFF (and back on) -- the switch for studying pure
  unitary evolution. Decay is multi-channel quantum jumps over the whole
  manifold: every downward pair gets its Einstein A from OUR
  wavefunctions -- A(2s->1s) ~ 1e-42 (exact nodal planes), Delta-l rules
  emerge numerically, and one common, honestly-labeled display
  acceleration keeps relative lifetimes physical. Photon flash + counter
  + last-jump label;
- **L** resonant laser at w = E(2p) - E(1s): Z-polarization Rabi-pumps
  1s -> 2p_z (live P(1s)/P(2pz) readout), X-polarization pumps 2p_x
  instead so P(2pz) stays flat -- and its fluorescence clicks through the
  2p_x decay channel. Laser + decay = repeated absorb/emit cycles;
- **static E-field (+z)** and **magnetic field (z/x/y)** sliders solve the
  field Hamiltonian PROPERLY: the E-field is a dipole term folded into the
  half-potential (Stark polarization / field ionization); the B-field is the
  paramagnetic (B/2)L Larmor rotation (exact unitary three-shear via the
  Fourier shift theorem) plus the diamagnetic (B^2/8)rho_perp^2, so a p_x
  cloud genuinely precesses into p_y. Crossed E-B works; a boundary absorbing
  mask stops ionized flux wrapping the periodic FFT box; a live readout gives
  the semiclassical Larmor radiated power of the oscillating dipole; an XYZ
  gizmo (z labeled) shows the axes.

Tab switches to the marching-cubes isosurface view; drag orbits, wheel
zooms, space pauses, [ ] tunes cloud density. All shader math is
unit-tested in core and transcribed into GLSL; the demo arcs regress
headlessly via `--selftest-decay` / `--selftest-rabi` / `--selftest-cascade`
/ `--selftest-manifold` / `--selftest-energy` / `--selftest-efield` /
`--selftest-magnetic`. See
[docs/ROADMAP.md](docs/ROADMAP.md) and [docs/GPU_PLAN.md](docs/GPU_PLAN.md).

> Toolchain note: build with the Qt-bundled MinGW kit
> (`-DCMAKE_PREFIX_PATH=C:/Qt/6.8.1/mingw_64`, compilers from
> `C:/Qt/Tools/mingw1310_64`); the repo is toolchain-agnostic but this is the
> tested configuration on Windows.
