# GPU compute plan — HISTORICAL RECORD (OpenGL 4.3 era)

> **This document is a historical planning record, not the current state.**
> It is the original OpenGL-compute plan and its delivered milestones, kept
> for the design rationale. The GPU stack has since been migrated twice:
> OpenGL → Qt QRhi/Vulkan → the current **framework-free raw-Vulkan core
> `ses_vk`** (volk + VMA + VkFFT, shaders offline-baked to SPIR-V by
> glslangValidator; Qt reduced to window/input/one blit). The verification
> harness followed the same path: `sesolver_gpucheck` (GL) and
> `sesolver_qrhicheck` (QRhi) were retired after **`sesolver_vkcheck`**
> reproduced every recorded value digit-identically; it is now the sole GPU
> oracle. The doctrine below survived every migration unchanged: fp32 GPU
> for display speed, the CPU double core as THE tested truth, and every
> kernel landing together with its harness comparison. See
> [ARCHITECTURE.md](ARCHITECTURE.md) for the current architecture.

Decision (user, after AVX2 work): the GPU backend is **OpenGL compute** --
already our context version, no new dependency, keeps the Windows/Linux
pillar, and writing the FFT as a compute shader is squarely inside the
"reinvent the wheel" learning goal. CUDA was rejected (NVIDIA-only, needs the
MSVC host compiler), DirectX rejected (Windows-only).

## Why

64^3 is real-time on the CPU (OpenMP+AVX2). The GPU arc is the enabler for
**higher resolution** and removes the per-frame psi texture upload entirely
(the field becomes GPU-resident).

## Precision strategy (the honest constraint)

Consumer GPUs run fp64 at 1/16-1/32 of fp32 rate, so GPU propagation is
**fp32**. Norm/energy fidelity drops from ~1e-12 to ~1e-6 -- fine for
display, not for physics claims. Therefore:

- The CPU double path stays THE tested truth (all analytic oracles remain).
- The GPU path is a display-speed transcription, verified against the CPU
  path by a dedicated harness (below), with fp32 tolerances (~1e-5 relative).
- The window keeps showing live norm; fp32 drift beyond ~1e-5 would be
  visible there immediately.

## Verification (TDD boundary for shaders)

Compute shaders cannot be gtest-unit-tested from the pure core. Instead:

- `sesolver_gpucheck` (app-side executable): creates an offscreen 4.3 core
  context, runs each GPU kernel on deterministic inputs, and compares against
  the CPU core (which IS unit-tested) element-by-element with fp32
  tolerances. Non-zero exit on any mismatch.
- Registered with ctest (label `gpu`, SKIP if no 4.3 context) so the
  comparison runs with the suite on GPU-capable machines.
- Every kernel lands together with its gpucheck comparison -- the GPU
  analogue of red/green.

## Milestones

- [x] G1: harness + SSBO plumbing + pointwise complex phase multiply kernel
  (the e^{-iVdt/2} / e^{-ik^2dt/2} application). Verified vs the CPU double core.
- [x] G2+G3: axis-generic workgroup shared-memory radix-2 line FFT (one line
  per workgroup, base = (l%A)*B + (l/A)*C enumerates any axis) -> full 3D
  FFT; inverse via the conj/scale kernel. Verified vs CPU double fft3 on
  16x8x4 (distinct dims: axis mapping) and 64^3; the GPU round-trip restores
  the original.
- [x] G4: full split-operator step on GPU; psi lives in an SSBO; phase
  tables from SplitOperator3D's tested accessors; the bridge compute pass
  writes the RG32F 3D texture (no CPU round-trip). Verified: the GPU steps
  match the CPU propagator; the bridge is bitwise.
- [x] G5: shell steps on the GPU; measure/surface stay on the CPU
  double session behind the single cpu_is_truth_ sync invariant (relaxation
  later moved to the GPU too -- G7). Hardened by adversarial review (stale-bridge,
  buffer-update barriers, execute-side time accounting, peak tracking).
- [x] G6: norm/peak tree-reduction kernel (2 KB readback replaces the 16 MB
  title drain) + scale kernel for fp32 drift renormalization -- the norm now
  stays pinned at 1 indefinitely. Verified: the reduction matches the CPU
  double sum, the scale kernel is exact.
- [x] G7: imaginary-time relaxation on the GPU. Real weights pack as
  vec2(w, 0) into the verified complex-multiply kernel; per-step
  renormalization reuses the G6 reduction, whose pre-renorm norm gives the
  ITP energy estimator E ~= -ln||psi||^2/(2 dtau) for free. Verified: the GPU
  steps match the CPU relaxer, and the estimator hits the harmonic 3w/2. The
  whole stationary-state demo arc now runs on the GPU.

GPU residency has since grown past propagation: the startup **orbital
synthesis** (ψ = (u/r)Yₗₘ, no CPU field -- atlas build ~90s -> seconds), the
**dipole / mean-force ⟨∇V⟩ reductions** (decay channels + semiclassical
radiated power), and the **static E/B-field Hamiltonian** (dipole half-kick,
three-shear Larmor rotation, diamagnetic potential) all run on GPU compute too
-- each with its `sesolver_gpucheck` comparison. The CPU double session remains
the tested truth for position measurement, surface meshing, and as gpucheck's
reference.
