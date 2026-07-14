# Code-Review Backlog

A whole-project critical review (2026-07-14: architecture/SOLID, antipatterns,
resource leaks, concurrency, test edge cases, comment hygiene, correctness)
produced 26 confirmed findings. As of 2026-07-14 all but two are **fixed**
(the two remainders below are a deliberately-deferred low-value extraction and
one minor consistency cleanup). Every fix must clear the
gates in [`TDD_RULES.md`](TDD_RULES.md): `ctest` (all pass) + `sesolver_vkcheck`
(all PASS) + the `--selftest-*` arcs. Prefer RED-first for any testable logic.

**Cross-session continuity (2026-07-15):** this doc also carries forward work
across machines (office <-> home) -- per-machine Claude memory does NOT travel,
so anything needed to continue a session lives here, self-contained (file paths,
exact fixes, verify commands). Latest work = the "Validation-clean sweep"
section; open/planned work = the "Forward work" section near the bottom.

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

### Batch A/B/C (2026-07-14, ee0cfa5..HEAD; ctest 264/264, vkcheck all PASS)

- **[medium] Shell god-object / OCP-LSP leak** — the ~28 down-cast forwarders
  and the concrete `HydrogenDirector*`/`TunnelingDirector*` members are GONE.
  `scenario.hpp` now declares `HydrogenApi` / `TunnelApi` capability
  interfaces + `ScenarioDirector::hydrogen()`/`tunnel()` (null unless the
  scene implements them); the panel takes `HydrogenApi&`, the arcs go through
  `shell->hy()`/`tn()`. (Deviates from the literal `draw_panel(shell)`: that
  would pull ImGui into the framework-neutral director layer.)
- **All 7 latent-correctness guards** — `normalize()` zero-norm; observables
  `obs_ratio`/`obs_sigma` (den==0 + variance clamp, bitwise no-op when den>0);
  `marching_cubes_at_fraction` empty/non-positive-peak; `fft()` power-of-two
  runtime throw (assert survived NDEBUG); `mean_potential_gradient` norm==1
  precondition DOCUMENTED (normalizing would desync the GPU mean_force oracle);
  `upload_field_tables` checks both bool returns + moves the memo after a
  successful upload; `kFlashTicks` unifies the coupled 25 / 25.0f literals.
  `tests/degenerate_guards_test.cpp` (RED-verified 5/6 against unfixed code).
- **3 of 4 latent concurrency items** — mc vbuf/indirect `CONCURRENT`;
  `norm_and_peak`/`project_psi`/`scale` leading `barrier_compute_to_compute`
  (self-contained vs an unwaited async batch); `Engine::destroy()`
  `reset_lazy_state()` nulls the lazy handles + `*_ok_` flags + cache sizes
  (vkcheck `check_engine_reinit`: init->destroy->init->step parity).

### Validation-clean sweep (2026-07-15, commits 8924c20 + 46a6aaa)

Context: validation layers were re-enabled 2026-07-15 (vcpkg-prebuilt VVL 1.4.350
at `external/vcpkg/installed/x64-windows-static/bin`) and the offscreen scene
pass went dynamic-rendering (65b216f). Re-enabled validation then surfaced
pre-existing app defects the compute path (`sesolver_vkcheck`) never had. ALL
FIXED -- the whole app is now validation-clean (offscreen render + windowed
present + ImGui):

- **[error] `pCode-08740` x2** -- `slice.frag` + `volume.frag` (the only two frag
  shaders using `discard`) declare the `DemoteToHelperInvocation` capability:
  SPIR-V 1.6 (the vulkan1.4 shader target) lowers `discard` to
  `OpDemoteToHelperInvocation`. Fix: enable `shaderDemoteToHelperInvocation` in
  the `create_device` probe-and-enable chain (`app/src/vk_device.hpp`,
  `feat_demote_to_helper`); `vkcheck check_device_features` now asserts
  `demoteToHelper=1`, so it is the standing oracle.
- **[error] `descriptorType-00337` x1** -- the HDR scene color target `color_`
  (`R16G16B16A16_SFLOAT`, `COLOR_ATTACHMENT|STORAGE`) is ALSO sampled: bloom-down
  reads it linearly (`down0_color_set_` binding 0) and the shell blit samples the
  finished image. Fix: add `VK_IMAGE_USAGE_SAMPLED_BIT` in
  `app/src/vk_render.hpp` `create_target()`.
- **[warning] `AllocateDescriptorSets-WrongType` x2** -- NOT in the earlier
  3-error catalogue: `--dump-frame` renders OFFSCREEN and never inits
  ImGui/present, so these only appear in a WINDOWED run. Vendored ImGui is
  **1.92.8**, whose Vulkan backend (changed 2025-06) allocates `SAMPLED_IMAGE` +
  `SAMPLER` descriptors, not the pre-1.92 `COMBINED_IMAGE_SAMPLER` the app
  hand-sized `imgui_pool_` for. NVIDIA tolerated it; a strict driver could
  `OUT_OF_POOL_MEMORY`. Fix: drop the hand-built pool; set
  `InitInfo.DescriptorPoolSize=16` and leave `DescriptorPool` null so the backend
  owns a correctly typed pool and frees it in `ImGui_ImplVulkan_Shutdown`
  (`app/src/main.cpp`).

**Verify (reusable, both paths must be 0 errors + 0 warnings).** Env:
`SES_VK_VALIDATION=1`, `VK_ADD_LAYER_PATH=<vcpkg installed bin>`,
`VK_LOADER_LAYERS_ENABLE=*validation*`. The layer prints Error AND Warning to
STDOUT -- grep both streams.
- offscreen: `sesolver_app --dump-frame` (headless).
- windowed: launch `sesolver_app`, let it run ~10 s, then HARD-kill it
  (`Stop-Process -Force`; a hard kill runs no app cleanup, so ZERO
  teardown-validation noise -- init/first-frame errors, where shader-module +
  descriptor bugs fire, are already captured).
- gates: `vkcheck` 0 errors + `demoteToHelper=1`; `ctest` 264/264. All green.

## Remaining — architecture (1 low, deliberately deferred)

- **[low] Extract a `MeasurementEngine`** from `HydrogenDirector`
  (`run_partial_measure` / `rebuild_psi_from` / `project_manifold_out`, ~180
  lines). DEFERRED (2026-07-14): per this item's own note the shared
  `cpu_is_truth_`/display-bridge/engine coupling makes the extraction
  low-cohesion — it would need a fat back-reference into HydrogenDirector, so
  the churn on the (just-refactored, manually-verified) shell is not worth the
  negligible cohesion gain. Revisit only if that coupling is broken first.

## Remaining — latent correctness

ALL FIXED (Batch B, 2026-07-14) -- see the "Already fixed" section above.

## Remaining — latent resource / concurrency (1 low)

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

## Test-robustness observation (new, 2026-07-14)

- **`--selftest-cascade` is timing-marginal** — its 90 s wall window at
  time-scale 4x packs only ~2-3 display lifetimes of the 3d state at the
  CURRENT au/s (the 1-step-per-render policy lowered throughput), so it has a
  real Poisson false-fail probability (observed: one `photons = 0` run, then
  `photons = 2` on the immediate re-run). Not a correctness bug; the window /
  time-scale constant wants raising to restore the intended ~11-lifetime
  margin. (decay/rabi/manifold, which share the same photon-count path, are
  comfortably above threshold.)

## Forward work -- render modernization + perf (continuity)

Standing planned work from the Vulkan 1.4 modernization + Pascal-perf arc (NOT
code-review findings). **New-request status (2026-07-15):** the only new request
this session was the validation-clean sweep above -- DONE. Nothing else was newly
requested-but-undone; the items below are pre-existing planned work, recorded
here so the next session can pick up without this machine's Claude memory.

- **Present-path dynamic rendering (LAST render-modernization item).** The
  offscreen scene pass is already dynamic (65b216f), but `app/src/vk_present.hpp`
  still builds a `VkRenderPass` for the swapchain clear+blit pass (attachment
  ref ~line 252; `presenter_.render_pass()` is handed to ImGui at
  `main.cpp` `info.PipelineInfoMain.RenderPass`). Convert to
  `vkCmdBeginRendering`/`VkRenderingInfo`, and switch ImGui to
  `InitInfo.UseDynamicRendering = true` + `PipelineRenderingCreateInfo` (drop the
  render pass). Medium-risk RENDER change, but now SAFE: the windowed path has a
  live validation oracle again -- verify with the windowed validation run above.
  `dynamicRendering` is already enabled in the feature chain.
- **Perf track (fp32-EXACT; the real-time STEP is already at its fp32 ceiling --
  three step-perf hypotheses were measured flat/worse and reverted, so these are
  OTHER-workload levers).** Gated on P0:
  - **P0** -- instrument the relax reductions in `profile_step` (prereq for
    P3/P4; the app already has GPU timestamp queries from Item 0b / 5c97d3f).
  - **P3** -- `subgroupAdd` reduce scoped to `project_deposit` ONLY (MED-LOW;
    breaks its bit-exact golden but stays deterministic + in-tolerance; harness
    with `requiredSubgroupSize=32`, width-agnostic per the forward-compat rule).
  - **P4** -- relax GPU-finish fusion (2 submits -> 1; a deterministic
    fixed-order reduction tree, NOT `subgroupAdd`; keep the double-precision
    energy diagnostic async).
  - **BANNED (do not attempt):** `VK_EXT_shader_atomic_float`, `float_controls2`
    relaxed modes, `subgroupAdd` on the bandwidth-bound STEP reductions, any
    `VK_NV_*` (forward-unsafe), larger workgroups. The FFT pair (~70% of the
    step) needs fp16, which is HELD by the precision constraint -- NO fp16 in the
    propagation without an explicit user opt-in.

## See also

- [`TDD_RULES.md`](TDD_RULES.md) — the verification gates every fix must clear.
- Physics-audit open item: no `⟨L_z⟩` / probability-current *diagnostic* (the
  ±m ring states ARE preparable now via the L_z partial measurement); and the
  m-sign's absolute handedness vs Larmor rotation under B is untested — a
  B-on ring-rotation check would pin it.
