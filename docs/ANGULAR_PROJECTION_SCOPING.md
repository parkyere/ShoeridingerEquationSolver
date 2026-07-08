<!-- POINT-IN-TIME SCOPING RECORD (2026-07-08). A design snapshot, NOT an
evergreen doc: line numbers and figures WILL go stale as the code changes,
and this is superseded once/if the projection is implemented. Kept as the
design rationale for the orbital-free angular-decomposition projection. -->

All anchors confirmed. Two corrections to fold in: tests live in the top-level `tests/` dir (not `core/tests/`), and `scratchpad/` is empty — the referenced PoC is genuinely absent. Here is the brief.

---

# Engineering Design Brief — Orbital-Free Angular-Decomposition Projection

**Status:** Scoping / design only. Not a build order. Decision to implement is deferred.
**Scope boundary:** the population **projection** `<n|psi>` only. Collapse-on-jump stays one on-demand synthesis; the dipole/Einstein-A channel table stays build-time; the 512³ live-working-set memory diet is a separate follow-on (noted, not designed here).

---

## 1. Goal & payoff

The resident eigenstate atlas — 91 complex-fp32 256³ SSBOs (`state_buf_[]`, `main.cpp:2103`), ~134 MB each, ~12.2 GB VRAM — exists only so that the per-title population loops can call `inner_with_psi(state_buf_[s])` for each state (`main.cpp:594-604`, `687-711`, `720-726`). This atlas is the single term whose VRAM scales with the state count, and it is what makes 512³ physically impossible (91 × 1.07 GB ≈ **97.7 GB**). This brief replaces it with an **orbital-free projector**: one grid pass over `psi` that deposits every `Y_lm(cell)·psi(cell)·dV` (with the 1/r Jacobian folded into a linear cloud-in-cell radial weight) into a small complex accumulator `g_lm[36][n_radial]` (~1.47 MB total), after which each state's amplitude is a cheap 1-D radial dot `<n|psi> = Σ_j u_nl[j]·g_lm[lm(n)][j]`. The grid work is **state-count-independent** (36 angular components regardless of #states); only the trivial 1-D dots scale with #states. Payoff: the atlas disappears (−12.2 GB at 256³; the impossible 97.7 GB at 512³ never exists), which simultaneously unlocks a **bigger basis** (more states = more 1-D dots, zero new 3-D buffers) and **512³** (the only blocker was the atlas).

---

## 2. The validated math (proven — build on it, do not re-derive)

Soft-Coulomb is central, so every eigenstate factorizes exactly as `|n> = (u_nl(r)/r)·Y_lm(Ω)`. The direct grid inner product is therefore, cell by cell,

```
<n|psi> = Σ_cells (u_interp(r_cell)/r_cell) · Y_lm(cell) · psi(cell) · dV
```

Reorganizing the sum — depositing the *cell-side* factors `Y_lm·psi·dV` into radial bins first, then contracting with `u` — gives the identity

```
<n|psi> = Σ_j u_nl[j] · g_lm[lm(l,m)][j],   g_lm[c][j] = Σ_cells W_j(r_cell)·Y_lm(cell)·psi(cell)·dV
```

where `W_j(r)` is the **same** linear tent-over-r used by `synthesize_orbital` (`harmonics.hpp:141-173`, interp branch ~150-166) and `kSynthSrc` (`gpu_engine.hpp:418`, radial branch ~509-519), so `Σ_j u[j]·W_j(r) ≡ u_interp(r)/r` by construction. The deposit shape, mirrored bit-for-bit:

- **r < h** (origin): `W_0 = 1/h`, all others 0. `u(0)=0` is pinned, so the inner segment is the *constant* `u[0]/h`, **not** `tent_0(r)/r`.
- **h ≤ r < rmax**: `t = r/h − 1`, `i0 = floor(t)`, `frac = t − i0`; `W_{i0} = (1−frac)/r`, `W_{i0+1} = frac/r` **only if** `i0+1 < n`. At the outer node `i0+1 == n` the upper term is **dropped** (`u(rmax)=0`), matching `synthesize_orbital`'s outermost `(1-frac)*u[i0]` branch.
- **r ≥ rmax**: no deposit (box-corner continuum → correctly part of the deficit).

The **radial** linearity is load-bearing (a nearest-bin deposit lands ~1e-4 off); the **angular** factor is exact analytically — `real_spherical_harmonic(l,m,x,y,z)` (`harmonics.hpp:34`) evaluated per Cartesian cell, no spherical resampling, no sin θ Jacobian. **Bin count must equal the radial nodes** (`RadialGrid{r_box, 5119}`, `main.cpp:1336`): with tent nodes coincident with the `u_nl` table, the reorganized sum equals the direct grid inner product to ~1e-13 (the PoC identity) **and** equals what the old atlas path computes — giving `gpucheck` a clean oracle. Coarsening buys nothing (accumulator is 1.47 MB either way) and injects an interpolation mismatch. Per-state norm is the analytic 1-D integral `N_n = Σ_j u_nl[j]²·h` (= 1 for eigen-`u`), **not** a grid pass.

> ⚠️ **The "machine-precision PoC" has no checked-in artifact.** `scratchpad/jacobian_poc.cpp` is **absent** from the working tree (confirmed: `scratchpad/` is empty) and from git history. The math is trusted, but the CPU oracle it describes must be **written and unit-tested first** — it is a prerequisite, not a formality (see §4, §6, §8-Phase 0).

---

## 3. Recommended GPU algorithm + the determinism decision (the crux)

### Algorithm — "static geometric counting-sort → per-bin deterministic segmented reduction"

The scatter is a **static sparse mat-vec**: `g_lm[j] = Σ_cells W_{lm,j}(cell)·psi(cell)` where `W` (radial tent/r, dV, 36 `Y_lm`) is pure geometry, fixed for the life of the grid; only `psi` changes per frame. So sort the in-sphere cells **once** by their radial index and the per-frame scatter becomes an output-stationary **gather** that needs zero atomics.

1. **Build-time (once, deterministic counting sort, O(n)):** for every cell with `r < rmax`, compute `i0` using the **identical fp32 expression sequence the shader uses** (see the determinism landmine below). Counting-sort the cell flat-indices into `sorted_cell[]` (uint32, one entry per in-sphere cell) + a CSR offset array `bin_off[n_bins+1]`. Cells with `r < h` go to a dedicated origin segment; `r ≥ rmax` cells never enter the sort. Store **only** `sorted_cell` + `bin_off`; recompute `r`, `frac`, `dV/r`, and the 36 `Y_lm` **in-shader** every frame from the cell index, so the arithmetic is bitwise-identical to `kSynthSrc`. Footprint: `sorted_cell` over ~8.78M in-sphere cells × uint32 ≈ **35 MB @256³** (~280 MB @512³); `bin_off[5120]` ≈ 20 KB. Read-only, geometry-only, built via `make_static_buffer` (`gpu_engine.hpp:1276`).
2. **Per-frame GPU (one dispatch, state-count-independent):** one workgroup per radial node `j` (256 threads). Node `j` collects two tent contributions — cells in segment `j` (weight `1−frac`) and cells in segment `j−1` (their `i0+1 == j`, weight `frac`). Each thread grid-strides its slice, recomputes `coords→r→frac→w=(tent)/r·dV` and the 36 real `Y_lm` (reuse `kSynthSrc`'s `real_Ylm`, factored into a shared GLSL string for bitwise identity), FMAs `w·Ylm·psi` into complex accumulators, then does the fixed-order `kNormPeak` tree reduction (`gpu_engine.hpp:121`, run pattern ~680) and writes 36 complex results to `g_lm[36][j]`. **Tiling by l is forced, not optional:** 256 threads × 36 complex = 73 KB shared > the 48 KB limit, so the 36 `(l,m)` are tiled by `l` (≤11 complex live) with a psi re-gather per tile — this locks in a ~6× psi gather.
3. **Finish (CPU double — the pattern already in use):** read back `g_lm` once (1.47 MB), then per state compute `<n|psi> = Σ_j u_nl[j]·g_lm[lm(n)][j]` as a 1-D double dot (90 × 5119 complex FMAs ≈ 0.5 Mflop). Deterministic double populations feed the decay draw.

### Determinism decision — stated plainly

**Is bitwise run-to-run reproducibility physically required here? No.** These amplitudes feed a stochastic Poisson channel-rate draw (`sample_energy_eigenstate`, `pick_decay_channel`; `main.cpp:601`, `687-701`). A 1e-6 — even 1e-4 — wobble in `pop[s]` cannot change the physics; the outcome is a random draw. This is **unlike** the pinned unitary time-evolution core (AVX2, `-ffp-contract=off`), where bitwise reproducibility is load-bearing because errors compound over thousands of steps.

**Do we build a deterministic kernel anyway? Yes — as project hygiene, at zero cost.** The recommended sorted-gather is deterministic *for free* (static sort, fixed workgroup=bin mapping, fixed thread→slot from the static array, fixed-order tree reduction, double CPU finish, **no atomics anywhere**). Keeping it deterministic keeps `sesolver_gpucheck` a **clean, non-flaky bitwise-close comparison** against the CPU double core (the project already tracks an "x-pol is a flake" annoyance — do not add another). `gpucheck` then sees only the pure fp32-vs-double rounding gap.

**Rejected: float `atomicAdd` histogram** (the "#1 danger" a risk probe raised — it applies only to *this* rejected approach) — non-deterministic add order (non-reproducible + flaky check), fp32 precision loss summing ~10³–10⁴ sign-varying terms per hot bin into one accumulator, and it isn't even core-GL-4.3-portable. **Rejected: shared-memory privatization** — 36×5119 complex = 1.47 MB won't fit in 48 KB shared. **Fallback only (note, don't design):** 64-bit fixed-point integer atomics are the *other* deterministic option (integer add is associative) but need an int64-atomic extension outside core GL 4.3, contend heavily on hot shells, and depart from the codebase's no-atomics pattern.

---

## 4. TDD / Humble-Object seam

### The pure core projector (no Qt, no GL)

New header `core/include/core/projection.hpp`, namespace `ses`, unit-tested in `tests/projection_test.cpp` (note: tests live in the **top-level `tests/`** dir, registered in `tests/CMakeLists.txt` — *not* `core/tests/`, which does not exist). The GPU kernel is the "humble object"; all logic and precision live in the testable core.

```cpp
// core/include/core/projection.hpp  (namespace ses) — pure, no Qt/GL.
struct ProjectorState { int level; int l; int m; };      // generative seed, no 3D orbital
inline int lm_index(int l, int m) { return l*l + (l+m); } // l∈[0,5], m∈[-l,l] → 36 comps
inline int lm_count(int l_max)    { return (l_max+1)*(l_max+1); }

struct RadialAngularProjection {
    std::vector<Complex<double>> amp;                        // <n|psi>, unit-normalized
    std::vector<double> norm2;                               // N_n = Σ_j u[j]²·h (=1 for eigen-u)
    std::vector<std::vector<Complex<double>>> g_lm;          // [36][n_radial], built ONCE, shared
    // amp[n] = (Σ_j u[j]·g_lm[lm(n)][j]) / sqrt(N_n);  raw = amp[n]*sqrt(N_n)  (for the 1e-12 oracle)
};

RadialAngularProjection project_radial_angular(
    const Field3D& psi, const RadialGrid& rgrid,
    const std::vector<std::vector<double>>& u_by_level,
    const std::vector<ProjectorState>& states, int l_max = 5);
```

**Required refactor** — single source of truth for the radial interp so the identity test can build the *un-normalized* oracle field:

```cpp
// core/include/core/harmonics.hpp
void fill_orbital(Field3D& psi, const Grid3D& g, const RadialGrid& rg,
                  const std::vector<double>& u, int l, int m);   // NO normalize
// synthesize_orbital (harmonics.hpp:141) becomes: fill_orbital(...); normalize(psi); return.
```

### Red-test list (most load-bearing first)

1. **`exact_reorg_identity`** *(tol 1e-12 — THE bit-match).* Pick a fixed-seeded complex `psi` and a state `n=(level,l,m)`. Oracle = `inner_product(orbital_raw, psi)` where `orbital_raw` = `fill_orbital(...)` (un-normalized `(u_interp/r)Y_lm`). Assert `|amp[n]·sqrt(norm2[n]) − oracle| < 1e-12·(1+|oracle|)`. Proves deposit+dot is the direct grid inner product merely reorganized. Run for l=0 (worst near-origin) and l=5.
2. **`linear_beats_naive`** *(ratio, not absolute).* Compute `amp[n]` three ways vs direct: (i) linear/CIC deposit, (ii) nearest-bin (`round(t)`, `W=1/r`), (iii) direct. Assert `err_linear < 1e-9` **and** `err_naive > 100·err_linear` (naive lands ~1e-4). Guards that radial linearity is load-bearing.
3. **`angular_orthogonality`** *(tol ~1e-2, tightening with n).* `psi = fill_orbital` for a pure `Y_{1,0}` profile; project onto the whole manifold. Assert the `(l=1,m=+1)` ("2p_x") component ≈ 0 while `(l=1,m=0)` ("2p_z") is recovered. This is Riemann-summed `<Y10|Y11>=0` — **approximate** (Cartesian angular quadrature), NOT machine precision; contrast in a comment with the exact radial part. Also assert `g_lm[lm(1,1)]·u ≈ 0` at the g_lm level.
4. **`self_projection_unit`** *(tol ~1e-3, grid-norm consistency).* `psi = synthesize_orbital(n0)` (grid-normalized). Assert `amp[n0] ≈ 1` (real, ± the ~1e-3 analytic-vs-grid norm gap) and every other-(l,m) state ≈ 0. Documents the norm difference as a known bounded quantity, not a bug.
5. **`completeness_deficit`** *(tol ~1e-3).* Complex weights `c_n` over a subset, `Σ|c_n|²=1`, `psi = Σ c_n|n>` (normalize once). Assert `Σ_n|amp[n]|² ≈ 1` and `deficit = 1 − Σ|amp|² ≥ −1e-9` (never significantly negative — Bessel). Then add a component **outside** the tracked basis (l>5 or high-n or plane wave), re-normalize, assert `deficit > 0` measurably (real continuum/truncation leakage).
6. **`origin_and_rmax_shape`** *(tol 1e-12).* (a) `psi` supported only in `r<h` cells contributes to bin 0 **only**, weight exactly `u[0]/h` in the reconstructed `u_over_r`; assert `g_lm[lm(0,0)][j]=0` for j>0 and an l=0 amp equals hand-computed `Σ (u[0]/h)·Y00·psi·dV`. (b) `psi` supported only in `r≥rmax` gives `amp[n]=0` exactly. Pins the two boundary branches that most easily drift from `synthesize_orbital`.
7. **`deterministic_repro`** *(tol 0, bitwise).* Run twice, and single-threaded vs OMP; assert `amp` and `g_lm` bitwise identical (`memcmp`). Enforces fixed-order-scatter / no-atomics so this CPU oracle stays a stable reference for gpucheck.
8. **`state_count_independence`** *(structural, tol 1e-12).* Call with a 3-state list and a 91-state list over the same psi; assert shared `g_lm` and overlapping `amp` are identical — proves the grid pass is done once and is #states-independent (the core scaling claim).

### gpucheck plan — `check_project()` in `app/src/gpucheck_main.cpp`

Model on `check_synth` (`:529`) and `check_dipole` (`:479`); use the harness `compare()` (`:92`) and its `trunc()` fp32-truncation trick (`:501`). Small CI grids (32³/64³ — reduction structure is grid-size-independent). Reuse `check_synth`'s case list `{0,0,0},{1,0,-1},{2,0,1},{3,0,-2},{4,0,3},{5,0,5},{5,0,0}`. Three tiers:

- **Tier A — algorithm identity (CPU-only, ~1e-12):** CPU scatter+dot == `inner_product(un-normalized (u/r)Y_lm field, psi)`. Run l=0 and l=5. Proves the projector adds no error beyond the inner product it replaces, at any grid coarseness. **The tight gate.**
- **Tier B — GPU vs CPU scatter (~1e-4 rel; ~1e-6 with the `trunc()` fp32-truncated psi+u):** GPU `g_lm`-dot vs the core CPU scatter, both analytic-normed. **The tight GPU gate.** Use an **absolute** floor (not relative) on near-zero / nodal-plane bins, or compare only the u-weighted amplitude, so the check doesn't flake on hard-cancelling bins.
- **Tier C — end-to-end vs the consumer being replaced (~1e-4, loose sanity only):** GPU projector `<n|psi>` vs `engine_.inner_with_psi(synthesize_state(...))`. **Built-in ~1e-3 floor** — `inner_with_psi` uses a *grid*-normalized orbital, the projector the analytic 1-D norm. Compare like-for-like or report the delta separately; never mistake it for a projector bug.

Invariants asserted every case: `Σ_n|<n|psi>|² ≤ 1+tol` (Bessel; overshoot = normalization/double-count bug); pure in-manifold psi → `Σ ≈ 1` minus radial truncation; `psi ∝ Y_{6,m}` → `Σ_n P_n ≈ 0` (l≤5 orthogonal to l=6). **Mandatory:** run the GPU projector twice, assert bitwise-identical output.

---

## 5. Integration change-list

| Site | Change |
|---|---|
| **`gpu_engine.hpp` — NEW kernel + methods** | Add `kProjectDepositSrc` (the §3 sorted-gather deposit, reusing `kSynthSrc` `real_Ylm`+interp verbatim). Add `project_psi(gl) → host g_lm[36][5119]`; `amplitude(u_nl,l,m) = Σ_j u[j]·g_lm[j]` (CPU-double dot); `synthesize_into_psi(gl,u,l,m,…)` (synthesize+normalize straight into `psi_buf_` for collapse). Build the static `sorted_cell`/`bin_off` via `make_static_buffer` (`:1276`). |
| **`main.cpp:687-711` — decay populations** | Replace the 90× `inner_with_psi(state_buf_[s])` loop (`689-691`) with **one** `project_psi()` pass, then `pop[s] = norm_sq(amplitude(radial_u_[level_s], l_s, m_s))` keyed by `kStateSpec`. On a fired jump, replace `copy_into_psi(state_buf_[ch.to])` (`706`) with `synthesize_into_psi(ch.to)` on demand. |
| **`main.cpp:720-726` — laser readout** | Reuse the **same** per-title `g_lm` pass (compute once per `gpu_title_due_`, share with the decay block); set `pop_excited_/pop_ground_` from `amplitude()` for `kP2Z`/`kS1` instead of two `inner_with_psi(state_buf_[…])`. Drop the `state_buf_[kP2Z]!=0` guard (use channels/manifold-ready). |
| **`main.cpp:594-604` — energy measurement (Key E)** | Replace the 91× loop (`595-596`) with one `project_psi` pass + 91 CPU dots; keep `sample_energy_eigenstate` and the `1−Σ P_n` continuum deficit; replace collapse `copy_into_psi(state_buf_[n])` (`604`) with `synthesize_into_psi(n)`. |
| **`main.cpp:1385-1450` — startup atlas montage** | Stop persisting into `state_buf_[s]`. Synthesize each orbital into a **transient** buffer, show it (`copy_into_psi`+bridge), keep the `E_radial` vs `<H>_grid` h-audit readback, then **free** it. Visual montage retained; no buffer survives. |
| **`main.cpp:1488-1536` — channel/dipole table build** | Build `channels_`/`dipole_z_` via **transient pair synthesis**: for each E1-allowed downward pair, ensure both endpoints in a tiny pool (synthesize if absent), run `dipole_between` (`gpu_engine.hpp:905`), evict — peak residency ~2-3 orbital buffers (<~400 MiB), never 91. Adds ~0.5 s startup. |
| **`main.cpp` deflation (keys 3/4)** | **Keep** `relax_deflated_step` + its 1-4 φ buffers, but hold them as **on-demand transients**, freed when relax completes/plateaus or the user leaves the demo. The projection `psi −= <φ\|psi>φ` needs φ as a full 3-D field, so `g_lm` cannot substitute. Max transient 4×128 MiB, only while active. |
| **`main.cpp:1117` `excite_n3`; `1247-1263` debug hooks** | Collapse → `synthesize_into_psi` of the cycled state; `debug_prepare_state`/`probe_population` → synth-on-demand target (or the projector). |
| **`main.cpp:2103` members** | `state_buf_` is no longer a resident atlas — make it a **sparse lazy handle table** with explicit release (only the deflation set or a just-synthesized collapse target is ever non-zero), or a small pool. Add a host-side `g_lm` accumulator member. |
| **`gpucheck_main.cpp` + core** | Add `ses::project_radial_angular` (core, unit-tested per §4) and `check_project()` per the §4 gpucheck plan. |

**Retained:** `radial_u_[]` (`main.cpp:2121`, ~0.9 MB host + the fp32 `radial_u_buf_` on GPU — the generative seed for both synthesis and the 1-D dot); `kStateSpec`/level specs (`main.cpp:356`); energies/channel table (`channels_`, now built from transient synthesis, still resident — a few hundred tiny structs); `synthesize_state`+`kSynthSrc` (now used for montage, channel-build, collapse, deflation) and its `real_Ylm`+interp (reused verbatim by the deposit kernel); `inner_with_psi`+`kInnerProductSrc` (still used by deflation and selftests — only the 90×/91× population usage is removed); all base engine buffers.

**Dropped:** the permanent 91-buffer atlas `state_buf_[]` (~12.2 GB) as a persistent structure; the 90×/91× per-title `inner_with_psi(state_buf_[s])` population loops (→ one `project_psi` pass + cheap CPU dots, O(1) in state count); the atlas's implicit role of keeping every orbital materialized for `dipole_between`.

**Collapse — still required, still an on-demand synthesis.** On a jump (`:706`) or energy-measurement collapse (`:604`), `psi_buf_` must be overwritten with the normalized target eigenstate. With the atlas gone this becomes one `synthesize_into_psi` (dispatch `(u/r)Y_lm` → norm-reduce → scale) instead of `copy_into_psi(state_buf_[target])`. Event-driven only — one synth per emitted photon or per Key-E, sub-ms to a few ms, negligible against the per-frame FFT stepping. Semantics preserved: a fresh real-only normalized eigenstate, phase reset.

**VRAM before/after (256³):**
- **Before** ≈ base working set ~1.15-1.25 GiB (psi/half/kin/relax×2/mask = 6×128 MiB + grad_v 256 MiB + psi_tex 128 MiB) **+ atlas 11.375 GiB** (91×128 MiB) ≈ **12.6 GiB**.
- **After** ≈ base ~1.25 GiB + `g_lm` ~1.47 MB + static sort table ~35 MB + ≤4 held deflation buffers (≤512 MiB, only during keys 3/4) + ≤2-3 transient synth buffers during startup/collapse ≈ **1.3-1.8 GiB**.
- **Net drop ≈ 11 GiB (~8×)** at 256³, and the removed term is O(1) in state count.

---

## 6. Risks + what must be PROTOTYPED before committing

The adversarial review verdict was **needs-prototype**: architecture sound, determinism claim holds for the recommended design, but two numerical unknowns and one process gap must be settled with a prototype (small GPU harness against the CPU double oracle) before committing.

**MUST prototype / resolve first:**

1. **Build-vs-shader `i0` consistency (correctness + determinism landmine).** The workgroup for bin `j` computes `frac = t − j` *trusting* the build sorted the cell into segment `j`. If the build computes `i0` in CPU double while the shader recomputes `r,t` in fp32 (`gpu_engine.hpp:507-515`), a boundary cell whose double-`i0` and fp32-`i0` differ by 1 gets `frac` slightly negative or >1 (an O(1)-wrong tent weight) **and** the GPU disagrees with the CPU oracle on the bin. With `h_radial ≈ 0.0156` and fp32 r-error ~5e-6 near r~80, ~2600 cells straddle a boundary at 256³. **Required fix:** compute the counting-sort key with the **identical fp32 arithmetic sequence** the shader uses (float coords from `box_min + idx·cell_h`, `float t = r/h_radial − 1`, `int(t)`). Prototype to confirm residual magnitude.

2. **fp32 per-bin accumulation on cancelling / continuum cases (the real accuracy weak point).** One workgroup per bin produces **one fp32 partial per bin**, so the whole cross-cell sum for a bin (up to ~10k sign-varying, hard-cancelling terms on outer / `Y_lm`-nodal shells) happens in fp32 — **structurally less precise** than `inner_with_psi`, which finishes its cross-workgroup sum in *double* (`gpu_engine.hpp:816-824`). Bound states are saved because `u[j] ~ e^{−r/n}` exponentially suppresses the noisy outer bins (u ~1e-6 at r=80 for n≤6) → low-n cases should hit ~1e-6. But the **completeness / Bessel / plane-wave / high-n** cases weight the outer bins fully and can push past 1e-4. **Must be measured against the double oracle across the full l=0..5 case list *and* the continuum cases before committing.** Fallback ready if it fails: **multiple fp32 partials per bin summed in double** (mirroring `inner_with_psi`), or compensated summation — either changes the kernel, so decide before building.

3. **Recreate the missing CPU oracle first.** `scratchpad/jacobian_poc.cpp` does not exist (confirmed empty `scratchpad/`, absent from history). Write `ses::project_radial_angular` (or a thin `scatter_glm` + `project_amplitude`) as a unit-tested core function — it is the Tier-A identity oracle and the gpucheck prerequisite. **Do this before any GPU work.**

4. **Normalization convention.** Analytic 1-D norm shifts populations ~1e-3 vs the grid-normed old path → after collapse `pop[n]` is `1 ± 1e-3`, not exactly 1. Make **Tier A (1e-12) and Tier B (~1e-6) the tight gates, Tier C only loose sanity.** Either precompute 91 per-state grid-norm scalars at build to stay value-consistent with the retired `inner_with_psi` path, or audit and loosen any selftest pinning `pop==1` / a Rabi peak tightly (don't add a second flake alongside x-pol).

5. **Performance — measure, don't assume.** Two risks: (a) the radius-**sorted** gather is **uncoalesced** (psi reads jump across the 16.8M-cell buffer in radius order — a possible 4-10× effective-bandwidth penalty vs a coalesced contiguous read); (b) 36 `Y_lm`/cell × the **forced** 6 l-tiles (l=5 polynomials ~50+ flops each) + ~6× psi re-gather. Prototype actual dispatch time vs the 90-reduction baseline before claiming any speedup. **Honest framing:** populations run only on the ~6 Hz title cadence (`gpu_title_due_`, `main.cpp:563/687`), **not** per frame — they are *not* the frame bottleneck (FFT stepping is). The scatter replaces 90 sequential dispatch+readback+sync round-trips with one sync and ~1/15 the traffic and frees ~11 GiB — a genuine win, but interactive responsiveness barely moves. **The decisive, uncontested wins are VRAM and state-count independence (the 512³ enabler), not wall-clock.**

**Lower-risk / accepted (no prototype needed):** near-origin poor sampling degrades standalone `g_lm(r)` but its weight in the amplitude is O(r^{l+2})-suppressed and is *identical* to the atlas path being replaced — the projector adds zero new near-origin error (Tier A holding at coarse grids is the proof). `l_max=5` truncation is consistent by construction — the deficit *is* the physical "outside tracked manifold" probability (`main.cpp:588-612`), not a bug. fp32/fp16 interplay is moot (no atlas; the orbital enters analytically at deposit time — precision-neutral-to-better, VRAM-strictly-better). Boundary handling has no trap: origin is a single cell (only `cell(128,128,128)` has `r<h_radial`, feeds l=0 with weight `1/h`); `r≥rmax` correctly excluded; collapse still one on-demand synth; deflation still needs full 3-D φ.

---

## 7. 512³ boundary — what this unlocks vs what stays a separate task

**Unlocked by this projection scope:** deleting the atlas removes the *only* term that scales with #states, so (a) a **bigger basis** costs only more 1-D dots (no new 3-D buffers), and (b) **512³ becomes feasible at all** — the atlas would be a physically-impossible 97.7 GB; the projector's `g_lm` is ~1.47 MB (or ~3 MB at a finer radial solve) and its static sort table ~280 MB @512³.

**NOT in this scope — a separate 512³ "memory diet" follow-on.** After the atlas is gone, the 512³ live working set (per-cell buffers scale 8× from 256³; RG32F = 1.07 GB, vec4 = 2.15 GB) is:
- **Always-resident during unitary evolution:** `psi_buf_` + `half_buf_` + `kin_buf_` + `mask_buf_` (`main.cpp:492/2172`) + `psi_tex_` volume texture (`main.cpp:1987/2178`) = **5.37 GB**.
- **Phase-specific/transient:** `grad_v_buf_` (vec4, 2.15 GB, mean_force/energy only — recomputable via the closed-form soft-Coulomb `grad V = Z·r_vec/(r²+a²)^{3/2}`); `relax_half_`/`relax_kin_` (2.15 GB, relaxation only — freeable); one on-demand collapse orbital (1.07 GB, one at a time).
- **Peak all-co-resident ≈ 9.66 GB** — already fits a 32 GB card *once the atlas is gone*. The diet (recompute `grad_v` in `mean_force` via `set_potential_gradient`'s analytic gradient, `gpu_engine.hpp:832`, −2.15 GB; free the relax pair when not relaxing, −2.15 GB) drops steady-state to ~5.4 GB with a ~7.5 GB transient peak for headroom.

**Two distinct axes — don't conflate.** 512³ at the current ±80 Bohr box (`Grid1D{-80,80,256}`, `main.cpp:135`) merely halves the spacing (`h`: 0.625 → 0.3125) — sharper soft-Coulomb cusp (`a` is pinned to `h`), **no new bound states** (box still holds only n≤6). A **bigger basis** (n>6, Rydberg) is **box-limited, not resolution-limited**: `<n|psi>` truncates any orbital support beyond the psi box, so growing `n_max` needs a larger `xmax` (a bigger box), which is independent of grid density. Both are separate follow-ons; this scope only removes the atlas and keeps one on-demand `synthesize_state` for collapse plus the build-time dipole table.

---

## 8. Phased, TDD-ordered implementation outline

Each phase is a testable unit; ship-gate each on green before the next. Phases 0-2 are pure CPU (no GPU risk); Phase 3 is the prototype that resolves §6's must-answer questions **before** any integration.

- **Phase 0 — CPU oracle (RED first).** Write `tests/projection_test.cpp` with `exact_reorg_identity` failing, then implement `fill_orbital` (refactor `synthesize_orbital` at `harmonics.hpp:141` to `fill_orbital(...); normalize; return`) and `ses::project_radial_angular` in the new `core/include/core/projection.hpp`. Register both in `tests/CMakeLists.txt`. **Gate: Tier-A identity green to 1e-12** for l=0 and l=5. *This is the prerequisite the whole plan depends on (§6.3).*
- **Phase 1 — core red-test suite.** Land tests 2-8 from §4 (`linear_beats_naive`, `angular_orthogonality`, `self_projection_unit`, `completeness_deficit`, `origin_and_rmax_shape`, `deterministic_repro`, `state_count_independence`). **Gate: all green;** determinism is bitwise. Now the CPU projector is a trusted oracle.
- **Phase 2 — static geometry (CPU, deterministic counting sort).** Implement the build-time `sorted_cell`/`bin_off` construction using the **fp32-identical `i0` expression** (§6.1). Unit-test: every cell lands in exactly one segment; `bin_off` is a valid CSR; origin/`r≥rmax` segmentation matches `kSynthSrc`; re-running the sort is bitwise-stable. **Gate: sort ≡ shader's `i0` for all cells** (the landmine test).
- **Phase 3 — GPU prototype (the go/no-go).** Stand up `kProjectDepositSrc` (sorted-gather, l-tiled, `kNormPeak` reduction) + `project_psi` in `gpu_engine.hpp`, checked *only* by a throwaway harness against the Phase-0 oracle. **Measure, don't assume:** (a) fp32 per-bin accuracy across the full l=0..5 case list **and** the plane-wave / high-n continuum cases vs the double oracle (§6.2 — if it misses 1e-6 on the hard cases, switch to multiple-partials-in-double *here*, before integration); (b) build-vs-shader `i0` residual (§6.1); (c) dispatch time vs the 90-reduction baseline, including the uncoalesced-gather penalty (§6.5). **Gate: bitwise-stable across two runs, Tier-B accuracy met, perf acceptable.** *Commit/no-commit decision lands here.*
- **Phase 4 — gpucheck gate.** Promote the throwaway harness into `check_project()` in `gpucheck_main.cpp` (Tiers A/B/C per §4, absolute floor on nodal bins, mandatory two-run determinism assert). Wire into the runner (`gpucheck_main.cpp:620-625`). **Gate: `check_project` green alongside `check_synth`/`check_dipole`.**
- **Phase 5 — integration, consumers first.** Add `amplitude()` and swap the three population paths (`main.cpp:594-604`, `687-711`, `720-726`) to one shared `project_psi` pass + CPU dots. Keep `state_buf_` alive in parallel initially so results can be diffed against the old path live. **Gate: populations match the old `inner_with_psi` path within the documented ~1e-3 norm gap.**
- **Phase 6 — collapse & synth-on-demand.** Add `synthesize_into_psi`; replace every `copy_into_psi(state_buf_[…])` collapse (`main.cpp:604`, `706`, `1117`, `1251`) with on-demand synthesis. **Gate: collapse selftests pass (loosen any `pop==1` pin per §6.4).**
- **Phase 7 — atlas removal & transient-pool conversion.** Convert the montage (`1385-1450`), channel/dipole build (`1488-1536`), and deflation (keys 3/4) to bounded transient synthesis; make `state_buf_` a sparse lazy handle table. **Gate: no permanent resident orbital survives a frame; VRAM readback confirms ~1.3-1.8 GiB (§5); manifold/channel selftests pass; `vram_budget_test` updated.**
- **Phase 8 (separate task, noted only) — 512³ memory diet + box growth.** Recompute `grad_v`, free the relax pair, then the box-`xmax` growth for n>6 — per §7, out of this scope.

---

**Key files (absolute):** `D:\HydrogenAtom\core\include\core\projection.hpp` (NEW), `D:\HydrogenAtom\core\include\core\harmonics.hpp` (`real_spherical_harmonic:34`, `synthesize_orbital:141` → `fill_orbital`), `D:\HydrogenAtom\core\include\core\radial.hpp` (`RadialGrid:34`, `h():38`, `normalize_u:117`), `D:\HydrogenAtom\app\src\gpu_engine.hpp` (`kNormPeakSrc:121`, `kInnerProductSrc:175`, `kSynthSrc:418`, `kNormPeakGroups:667`, `inner_with_psi:806`, `set_potential_gradient:832`, `dipole_between:905`, `synthesize_state:991`, `copy_into_psi:1039`, `make_static_buffer:1276`), `D:\HydrogenAtom\app\src\main.cpp` (`Grid1D{-80,80,256}:135`, `kNumStates:335`, `kStateSpec:356`, mask `486-492`, energy `594-604`, decay `687-711`, laser `720-726`, `RadialGrid{r_box,5119}:1336`, `radial_u_ fill:1347`, montage `1385-1450`, channel build `1488-1536`, `state_buf_:2103`, `radial_u_:2121`, `psi_tex_:1987/2178`), `D:\HydrogenAtom\app\src\gpucheck_main.cpp` (`compare():92`, `check_deflation:298`, `check_dipole:479`+`trunc:501`, `check_synth:529`, runner `620-625`), `D:\HydrogenAtom\tests\` (+ `tests\projection_test.cpp` NEW, `tests\CMakeLists.txt`). **Missing prerequisite:** `scratchpad\jacobian_poc.cpp` does not exist — recreate as the unit-tested core oracle in Phase 0.
