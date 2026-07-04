# GPU compute plan (OpenGL 4.3 compute shaders)

Decision (user, after AVX2 work): the GPU backend is **OpenGL compute** --
already our context version, no new dependency, keeps the Windows/Linux
pillar, and writing the FFT as a compute shader is squarely inside the
"reinvent the wheel" learning goal. CUDA was rejected (NVIDIA-only, needs the
MSVC host compiler), DirectX rejected (Windows-only).

## Why

64^3 is real-time on the CPU (OpenMP+AVX2: advance ~13.5 ms). The GPU arc is
the enabler for **128^3+** (estimated CPU ~110 ms/step) and removes the
per-frame psi texture upload entirely (the field becomes GPU-resident).

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
  (the e^{-iVdt/2} / e^{-ik^2dt/2} application). Verified: 1.6e-7.
- [x] G2+G3: axis-generic workgroup shared-memory radix-2 line FFT (one line
  per workgroup, base = (l%A)*B + (l/A)*C enumerates any axis) -> full 3D
  FFT; inverse via the conj/scale kernel. Verified vs CPU double fft3 on
  16x8x4 (distinct dims: axis mapping) and 64^3; GPU round-trip restores the
  original to ~1e-6.
- [x] G4: full split-operator step on GPU; psi lives in an SSBO; phase
  tables from SplitOperator3D's tested accessors; the bridge compute pass
  writes the RG32F 3D texture (no CPU round-trip). Verified: 20 GPU steps vs
  20 CPU steps at 1.2e-6; bridge bitwise.
- [x] G5: shell steps on the GPU at 128^3 (~50-55 fps; CPU would be ~9);
  relax/measure/surface stay on the CPU double session behind the single
  cpu_is_truth_ sync invariant. Hardened by adversarial review (stale-bridge,
  buffer-update barriers, execute-side time accounting, peak tracking).
- [x] G6: norm/peak tree-reduction kernel (2 KB readback replaces the 16 MB
  title drain) + scale kernel for fp32 drift renormalization -- the norm now
  stays pinned at 1 +- ~2e-5 indefinitely. Verified: reduction rel err 3e-9
  vs CPU double, scale exact.
- [ ] G7 (candidate): imaginary-time relaxation on the GPU (weight tables +
  per-step renormalization via the G6 kernels) so relax mode is also
  real-time at 128^3.
