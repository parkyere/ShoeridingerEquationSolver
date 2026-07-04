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

## Working agreement

This project follows **strict TDD**. No production code is written without a
failing (red) test first, and **test commits and production commits are never
mixed**. Read [docs/TDD_RULES.md](docs/TDD_RULES.md) before contributing, then
enable the guard hook once:

```sh
git config core.hooksPath tools/git-hooks
```

## Status

Phases 0-7 delivered (128 tests green): the hand-rolled Complex/FFT/
split-operator/imaginary-time stack is validated against analytic oracles
through 3D, and `sesolver_app` renders the TDSE in real time -- a Gaussian
electron wavepacket swinging past a soft-Coulomb nucleus, shown as a TRUE
VOLUME-RENDERED cloud (GPU ray marching, opacity ~ |psi|^2, hue = arg(psi)
via the cyclic colormap) at ~57 fps with the norm conserved to 1e-9 live.
Tab switches to the marching-cubes isosurface view; drag orbits, wheel
zooms, space pauses, [ ] tunes cloud density. All shader math (ray/box,
Beer-Lambert opacity, front-to-back compositing, phase LUT) is unit-tested
in core and transcribed into GLSL. See [docs/ROADMAP.md](docs/ROADMAP.md).

> Toolchain note: build with the Qt-bundled MinGW kit
> (`-DCMAKE_PREFIX_PATH=C:/Qt/6.8.1/mingw_64`, compilers from
> `C:/Qt/Tools/mingw1310_64`); the repo is toolchain-agnostic but this is the
> tested configuration on Windows.
