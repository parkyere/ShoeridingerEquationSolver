# Code-Review Backlog

Open work only. This git-tracked file is the **cross-machine handoff** (office ↔
home ↔ P5000) -- per-machine Claude memory does NOT travel, so anything needed to
continue on another machine lives here, self-contained. Completed work is not
recorded here; it's in git history + the per-machine memory files.

Every fix must clear the gates in [`TDD_RULES.md`](TDD_RULES.md): `ctest` (all
pass) + `sesolver_vkcheck` (all PASS) + the `--selftest-*` arcs. Prefer RED-first
for any testable logic.

## Open items

- **[verify] GPU marching-cubes oracle on 5090/Linux.** The cyclic-hue colour
  metric + valid sort key (a discontinuous-wheel abs-RGB compare false-failed on
  the RTX 5090) is fixed but *unconfirmed on that hardware* -- could not reproduce
  on the RTX 4060. Needs a Linux/5090 `sesolver_vkcheck` re-run to close.

- **[low, deferred] Extract a `MeasurementEngine`** from `HydrogenDirector`
  (`run_partial_measure` / `rebuild_psi_from` / `project_manifold_out`, ~180
  lines). Deferred: the shared `cpu_is_truth_`/display-bridge/engine coupling
  makes the extraction low-cohesion -- it would need a fat back-reference into
  `HydrogenDirector`, so the churn on the (refactored, manually-verified) shell is
  not worth the negligible cohesion gain. Revisit only if that coupling is broken
  first.

- **[physics] No quantitative `⟨L_z⟩` / probability-current diagnostic, and the
  m-sign handedness is untested.** The ±m ring states are preparable (via the L_z
  partial measurement) and the flow streaklines *visualize* the current, but there
  is no `⟨L_z⟩` number and the m-sign's absolute handedness vs Larmor rotation
  under B is unverified. A B-on ring-rotation check would pin the handedness.

## See also

- [`TDD_RULES.md`](TDD_RULES.md) — the verification gates every fix must clear.
