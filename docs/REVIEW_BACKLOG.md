# Code-Review Backlog

A whole-project critical review (2026-07-14: architecture/SOLID, antipatterns,
resource leaks, concurrency, test edge cases, comment hygiene, correctness)
produced 26 confirmed findings. The high/medium ones are **fixed**; this file
tracks the remainder so a later pass can pick them up. Every fix must clear the
gates in [`TDD_RULES.md`](TDD_RULES.md): `ctest` (all pass) + `sesolver_vkcheck`
(all PASS) + the `--selftest-*` arcs. Prefer RED-first for any testable logic.

## Already fixed (for context — do not redo)

- **State-synthesis leak** — `release_state()` recycles slots + sets via
  `free_full_states_` (was leaking 4 descriptor sets + a slot per synth/release).
- **Laser + B-field** — now mutually exclusive (`toggle_laser`/`set_bfield_b`);
  was an inconsistent Hamiltonian (paramagnetic rotation dropped) + a false
  "psi evolved" title.
- **Harmonic coherent state** — `sigma = 1/sqrt(2 w)` (was `1/sqrt(w)`, sqrt(2)x
  too wide → breathing instead of rigid oscillation).
- **Qt comment rot** — 24 source sites swept to present-tense contracts.
- **HydrogenDirector → BaseDirector reparent** — retires the sibling
  duplication + god-object findings (removes the duplicated member block and the
  plumbing methods; hydrogen keeps only its specialized overrides).
- **GPU marching-cubes oracle** — cyclic-hue colour metric + valid sort key
  (a discontinuous-wheel abs-RGB compare false-failed on the RTX 5090). *Still
  needs a Linux/5090 `sesolver_vkcheck` re-run to confirm — could not reproduce
  on the RTX 4060.*

## Remaining — architecture (1 medium, 1 low)

- **[medium] Shell down-casts to concrete directors + ~28 `if (hydrogen_) …`
  forwarders** (`app/src/main.cpp`, the Shell control/probe section) — OCP/LSP
  leak. Fire-and-forget buttons (toggle_laser, measure_*, relax_*, set_efield…)
  should route through the existing `handle_key`/`press` seam; give
  `ScenarioDirector` a `virtual draw_panel(shell)`. NOTE: ~13 of the ~28 are
  value-RETURNING selftest probes (`channel_a`, `state_energy`, `mean_z`,
  `probe_population`, `tunnel_transmitted_max`…) that `press()` cannot fold —
  those need a small typed capability sub-interface, not `handle_key`.
- **[low] Extract a `MeasurementEngine`** from `HydrogenDirector`
  (`run_partial_measure` / `rebuild_psi_from` / `project_manifold_out`, ~180
  lines) — the one cleanly cohesive unit left after the reparent. Optional; the
  shared `cpu_is_truth_`/display-bridge coupling makes fuller decomposition
  low-value.

## Remaining — latent correctness (all low; reachable only in adversarial / degenerate regimes the app does not currently produce)

- **`normalize()` has no zero-norm guard** (`core/field.hpp`, ~line 88) →
  `inv = +Inf` → NaN across the field. Called by `collapse_wavepacket`, relax,
  `relax_deflated`. Asymmetric with `nojump_damped_amplitudes` which DOES guard.
  Fix: `if (n2 <= 0) return;` + add the two degenerate test cases (a
  `relax_deflated` seed parallel to the span of the lower states; a collapse
  center far from all probability).
- **`observables.hpp` reductions divide by accumulated norm with no `den==0`
  guard** (`mean_position`/`sigma_position`/`mean_momentum`/`mean_energy`, the
  num/den sites) → NaN on an empty / fully-absorbed field, propagates into the
  title readout and the Larmor path. Also `sigma_position` lacks a
  `max(0, var)` clamp → `sqrt` of a tiny negative FP value → NaN on single-cell
  states (test-only path today). Fix: guard `den`, clamp variance, add a
  single-cell edge test.
- **`marching_cubes_at_fraction` dereferences `*max_element` with no
  empty / non-positive-peak guard** (`core/marching_cubes.hpp`, ~line 119).
- **`mean_potential_gradient` assumes `norm == 1`** (`core/emission.hpp`,
  ~line 20) unlike every other observable — a caller trap on an unnormalized
  input. Document the precondition or normalize internally.
- **FFT power-of-two requirement is only an `assert`** (`core/fft.hpp`, ~line 50)
  → compiled out under NDEBUG, then silent garbage / infinite loop on a
  mis-sized axis (this bit a `48^3` test during the coverage work). Add a
  runtime check or a death test; grids are power-of-two by construction today.
- **`upload_field_tables` ignores the `set_potential` / `set_potential_gradient`
  bool returns** (`app/src/hydrogen_director.hpp`) — inconsistent with
  `init_compute`, which treats the same gradient-upload failure as fatal. On
  failure psi keeps evolving under a stale half-potential and the Ehrenfest
  gradient desyncs (the "fake-Larmor" hazard the function's own comment warns
  about); the `uploaded_*` memo is committed BEFORE the upload, so the guard
  also blocks retry. Fix: check both returns; on failure log + `gpu_ok_ = false`
  or revert the `uploaded_*` memo.
- **Coupled photon-flash magic literals** — `flash_ticks_ = 25` and the
  `/ 25.0f` fade divisor (`app/src/hydrogen_director.hpp`) must stay equal but
  are two bare literals. Introduce `kFlashTicks` (the codebase already uses
  `kMeasureSigma`/`kIsoFraction` named-constant convention).

## Remaining — latent resource / concurrency (all low)

- **`Engine::destroy()` leaves ~20 lazily-created descriptor-set handles
  non-null and the `*_ok_` flags set** (`app/src/vk_engine.hpp`) → a *second*
  `initialize()` on the same Engine object would bind freed sets (e.g.
  `set_absorber`'s `pd_full_set_ == NULL` guard sees a stale non-null handle and
  skips re-wiring). NOT reachable today (init runs once; vkcheck uses fresh
  Engines). Fix: null the lazy handles and reset the flags/grow-cache sizes in
  `destroy()`.
- **Marching-cubes `mc_vbuf_` / `mc_indirect_` are `VK_SHARING_MODE_EXCLUSIVE`**
  (`create_device_buffer`) but written on the compute family and read as
  vertex/indirect on the graphics family with no queue-family ownership
  transfer — UB per spec on a *dedicated* async-compute family (host-fence
  serialization prevents a true race; discrete GPUs tolerate it; the
  combined-queue fallback is fine). Fix: mark them `CONCURRENT` like the display
  volume, or add a QFOT.
- **psi-reading OneShots lack a self-contained leading barrier**
  (`norm_and_peak`, `project_psi`, `scale` in `app/src/vk_engine.hpp`) — correct
  ONLY by the call-site invariant that `wait_async()` precedes them; a future
  readout that reads psi mid-batch without `wait_async` would silently read
  stale psi, and validation layers do NOT catch memory hazards. Fix: add the
  leading `barrier_compute_to_compute` locally, or document the invariant at
  each entry.
- **Compute `Kernel` keeps its `VkShaderModule` for the object lifetime**
  (`app/src/vk_compute.hpp`) — the graphics pipelines free theirs right after
  pipeline creation. Retained memory (~29 small modules for the session), not a
  leak. Minor consistency fix: free `module_` at the end of `Kernel::create`.

## Verified CLEAN (do not re-investigate)

Async compute↔render is correctly fence-synchronized (one-in-flight anchor
holds); OpenMP is bitwise-deterministic (fixed serial per-z-slab combine); there
are ZERO `std::mutex`/`std::thread` anywhere; `g_validation_errors` is a correct
`std::atomic`; the test `static const Relaxed` caches are pure memoization, not
order-coupled.

## See also

- [`TDD_RULES.md`](TDD_RULES.md) — the verification gates every fix must clear.
- Physics-audit open item: no `⟨L_z⟩` / probability-current *diagnostic* (the
  ±m ring states ARE preparable now via the L_z partial measurement); and the
  m-sign's absolute handedness vs Larmor rotation under B is untested — a
  B-on ring-rotation check would pin it.
