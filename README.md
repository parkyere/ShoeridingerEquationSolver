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

## Working agreement

This project follows **strict TDD**. No production code is written without a
failing (red) test first, and **test commits and production commits are never
mixed**. Read [docs/TDD_RULES.md](docs/TDD_RULES.md) before contributing, then
enable the guard hook once:

```sh
git config core.hooksPath tools/git-hooks
```

## Status

Phases 0-7 plus the GPU engine and the transitions arc delivered (168 tests
green): the hand-rolled Complex/FFT/split-operator/imaginary-time stack is
validated against analytic oracles through 3D, and `sesolver_app` renders
the TDSE as a TRUE VOLUME-RENDERED electron cloud at 128^3 -- propagation,
relaxation, and rendering all GPU-resident (OpenGL 4.3 compute; every
kernel verified against the unit-tested CPU double core by
`sesolver_gpucheck`). The full atom-and-light demo works from first
principles computed by the solver itself:

At startup the app SOLVES the atom first: the n<=2 eigenstate atlas
(1s, 2p_x, 2p_y, 2p_z, 2s) builds chunked across frames -- watch each
state converge, progress in the title (~20 s one-time) -- and then the
wavepacket demo begins with spontaneous decay ARMED, as in nature.

- **1** real time / **2** relax to 1s / **3** relax to 2p_z / **4** relax
  to 2s (deflated imaginary time; the ITP energy readout converges live);
- **M** soft Gaussian position measurement (collapse and re-evolution);
- **D** turns decay OFF (and back on) -- the switch for studying pure
  unitary evolution. Decay itself is multi-channel quantum jumps over the
  whole manifold: every downward pair gets its Einstein A from OUR
  wavefunctions -- the 2p triplet decays with tau ~ 4.7 ns (same order as
  real hydrogen 2p) while A(2s->1s) ~ 1e-17 makes 2s METASTABLE, the
  selection rule emerging from the matrix element. Photon flash + counter
  + last-jump label; one common, honestly-labeled display acceleration
  keeps relative lifetimes physical;
- **L** resonant laser at w = E(2p) - E(1s): Z-polarization Rabi-pumps
  1s -> 2p_z (live P(1s)/P(2pz) readout), X-polarization pumps 2p_x
  instead so P(2pz) stays flat -- and its fluorescence clicks through the
  2p_x decay channel. Laser + decay = repeated absorb/emit cycles.

Tab switches to the marching-cubes isosurface view; drag orbits, wheel
zooms, space pauses, [ ] tunes cloud density. All shader math is
unit-tested in core and transcribed into GLSL; the demo arcs regress
headlessly via `--selftest-decay` / `--selftest-rabi` /
`--selftest-manifold`. See
[docs/ROADMAP.md](docs/ROADMAP.md) and [docs/GPU_PLAN.md](docs/GPU_PLAN.md).

> Toolchain note: build with the Qt-bundled MinGW kit
> (`-DCMAKE_PREFIX_PATH=C:/Qt/6.8.1/mingw_64`, compilers from
> `C:/Qt/Tools/mingw1310_64`); the repo is toolchain-agnostic but this is the
> tested configuration on Windows.
