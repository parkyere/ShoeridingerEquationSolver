# Roadmap

The build order climbs the TDD ladder so that **every step has an analytic
oracle** to write the red test against. We go 1D first (fast, exactly solvable),
then 3D; physics first, rendering last.

Legend: each phase lists the *first* red tests that gate it.

> **Status:** Phases 0-7 are all complete; the ladder below is the historical
> record of the original build order (Phase 7's OpenGL renderer has since been
> rewritten twice and is now framework-free Vulkan — see
> [ARCHITECTURE.md](ARCHITECTURE.md)). The project has since grown several
> arcs well beyond this original ladder -- see **Beyond the ladder** at the end.

## Phase 0 — Scaffolding ✅
Build system, layered structure, GoogleTest harness, TDD guard hook, docs.

## Phase 1 — Core math (pure, trivial to test)
Hand-rolled, header-driven where sensible.
- `Complex<T>` arithmetic — add/mul/conj/abs; red: known products, `|i|=1`.
- `Vec3`, `Mat4`, `Quaternion` — red: identity, associativity, rotation of a
  known vector, `R(θ)` composition. (Used later for the camera.)

## Phase 2 — Grid & Field
- `Grid` (1D→3D): spacing `h`, extent, index⇄coordinate mapping, point count.
  Red: round-trip index↔coord, neighbor strides, boundary indices.
- `Field<Complex>`: complex scalar field over a grid; norm `Σ|ψ|²·hᵈ`,
  inner product, normalization. Red: a normalized Gaussian integrates to 1.

## Phase 3 — Hand-written FFT (the centerpiece)
- 1D radix-2 Cooley–Tukey forward/inverse.
  Red: `IFFT(FFT(x)) == x`; FFT of a real DC signal → single spike; FFT of
  `cos(2πk n/N)` → spikes at ±k; **Parseval** energy identity; linearity.
- N-D FFT by successive 1D transforms along each axis. Red: separable Gaussian
  ↔ Gaussian; 2D/3D round-trip.

## Phase 4 — Split-operator propagator (free particle)
- `exp(-iVΔt/2)·exp(-iTΔt)·exp(-iVΔt/2)`; kinetic step in k-space via Phase-3 FFT.
  Red (the headline oracle): a **1D free Gaussian wavepacket** disperses as
  `σ(t)=σ₀√(1+(t/(2σ₀²))²)`, its center translates at group velocity `k₀`, and
  `‖ψ‖` stays 1 to tolerance. Convergence: error shrinks as `Δt²`.

## Phase 5 — Potentials & bound dynamics
- `Potential`: harmonic well, softened Coulomb `-Z/√(r²+a²)`, box.
  Red: harmonic **coherent state** oscillates rigidly at `ω` without spreading;
  energy `⟨H⟩` conserved under real-time propagation.
- (Optional here) imaginary-time propagation → ground state; red: harmonic
  ground-state energy `½ω`, hydrogen 1s `-0.5` Hartree.

## Phase 6 — 3D
- Lift grid/field/FFT/propagator to 3D (mostly generic-dimension reuse).
  Red: 3D free Gaussian dispersion; 3D harmonic coherent state.

## Phase 7 — Visualization (hand-written OpenGL, Humble Object shell)
Pure, testable geometry/color math in `core/`; thin GL drawing in `app/`.
- `core/` geometry: marching-cubes vertex/normal generation, transfer-function
  mapping, camera/MVP matrices. Red: marching cubes on an analytic sphere field
  yields the right surface area / vertex count within tolerance; transfer
  function maps known density→color; camera matrix maps a known point correctly.
- `app/`: Qt `QOpenGLWidget`, hand-written shaders, upload `|ψ(r,t)|²` as a 3D
  texture, render the cloud (volume ray-march or density splatting), animate the
  real-time propagation, phase shown via a **cyclic** colormap. Verified by eye.

---

### Notes
- Atomic units throughout (ℏ=mₑ=e=1).
- **The startup flow has since inverted:** the app now first SOLVES and
  synthesizes the stationary **eigenstate atlas** on the GPU
  (`ψ = (u/r)Yₗₘ`, no imaginary-time ladder), then runs the moving-wavepacket
  `|ψ(r,t)|²` demo with spontaneous decay armed. Imaginary time survives as the
  Key-2 "relax to 1s" cooling demo.
- Keep `core` free of Qt/GL at every phase; if a thing is hard to test, it
  probably belongs in `core` as data, not in `app` as logic.

---

### Beyond the ladder (delivered since)

The project grew well past Phase 7. Each later arc is TDD'd against analytic /
CPU oracles and (for GPU kernels) verified in `sesolver_vkcheck`:

- **Transitions:** radial engine (all bound levels to n = 10 + E1 lifetimes),
  Einstein-A multi-channel quantum-jump decay, resonant laser / Rabi, position
  **and** energy-basis measurement, radiative cascades.
- **GPU compute** (design history in [GPU_PLAN.md](GPU_PLAN.md)): all real-
  and imaginary-time evolution, orbital synthesis, projection, and dipole /
  mean-force reductions run on GPU compute — first OpenGL 4.3, then Qt
  QRhi/Vulkan, now the framework-free raw-Vulkan `ses_vk` core
  (volk + VMA + VkFFT, offline SPIR-V), with every kernel verified by the
  zero-Qt `sesolver_vkcheck` inside ctest.
- **Manifold:** the tracked m-resolved bound shell raised to the largest n the
  box holds, with real Yₗₘ synthesized to the matching l; populations come
  from an orbital-free angular-decomposition projection (one grid pass, no
  resident atlas — design record in
  [ANGULAR_PROJECTION_SCOPING.md](ANGULAR_PROJECTION_SCOPING.md)).
- **Static fields, solved as a proper Hamiltonian:** E-field (Stark, a dipole
  term in the half-potential) and magnetic field **z/x/y** (paramagnetic Larmor
  via an exact three-shear rotation, `core/rotation.hpp`; diamagnetic folded
  into the potential, `core/magnetic.hpp`), crossed E-B; a boundary absorbing
  mask for the periodic FFT box.
- **Radiation:** semiclassical Larmor power from the oscillating dipole
  (Ehrenfest ⟨∇V⟩, `core/emission.hpp`).
- **Rendering:** raymarched electron cloud on an HDR pipeline — phase-tinted
  front-to-back march with IGN jitter and temporal accumulation, occupancy
  empty-space skipping, an extinction self-shadow volume, probability-current
  flow particles, and dual-Kawase bloom under an ACES tonemap.
