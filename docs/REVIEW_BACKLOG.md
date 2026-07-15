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
exact fixes, verify commands). Latest work = the "GLSL -> Slang migration"
section; open/planned work = the "Forward work" section near the bottom.

## GLSL -> Slang migration (2026-07-15; ctest 270/270, vkcheck all PASS)

All 50 ses_vk shaders were migrated GLSL -> **Slang**. The C++ engine/vkcheck
never changed: `tools/cmake/bin2h.cmake` takes `-DNAME`, so the blob symbols
(`k_<x>_spv`) and headers are language-agnostic.

- **Toolchain** ([`app/CMakeLists.txt`](../app/CMakeLists.txt)) — `slangc`
  v2026.13.1 is fetched as a prebuilt build-time tool via `FetchContent`
  (windows-x86_64 / linux-x86_64), located with `find_program(... HINTS
  ${ses_slang_SOURCE_DIR}/bin)`. `ses_bake_shader` baked either source during
  the migration (dual-path); now slangc-only. `glslangValidator` finder + the
  legacy `.comp/.vert/.frag` sources + `glslang[tools]` were all removed;
  glslang the *library* stays solely as VkFFT's runtime compiler.
- **Layout ABI** (the silent-corruption risk) — every buffer is pinned with a
  per-buffer generic: `ConstantBuffer<T, Std140DataLayout>` for UBOs,
  `RWStructuredBuffer<T, Std430DataLayout>` / `StructuredBuffer<...>` for SSBOs.
  Verified byte-exact by the oracles (e.g. scale UBO `{uint n; float scale}`).
- **Translation dictionary** — `vecN`->`floatN`; UBO fields via `params.`;
  `gl_GlobalInvocationID`->`SV_DispatchThreadID`, `gl_LocalInvocationIndex`->
  `SV_GroupIndex`, `gl_WorkGroupID`->`SV_GroupID`; `gl_NumWorkGroups`->
  `WorkgroupCount()`; `shared`->`groupshared`, `barrier()`->
  `GroupMemoryBarrierWithGroupSync()`; `findMSB`->`firstbithigh`,
  `bitfieldReverse`->`reversebits`, `atomicAdd`->`InterlockedAdd`, `mix`->`lerp`,
  `atan(y,x)`->`atan2`, `mod`->`fmod`, `fract`->`frac`, `inversesqrt`->`rsqrt`.
- **Three gotchas that needed real fixes** (not mechanical):
  1. **fp16 pack** — `f32tof16`/`f16tof32` declare the SPIR-V Float16/Int16
     capabilities (device lacks `shaderFloat16`); use `import glsl` +
     `packHalf2x16`/`unpackHalf2x16` (GLSL.std.450, no capability).
  2. **Subgroups** (`project_deposit`) — `gl_NumSubgroups`/`gl_SubgroupID` have
     no HLSL analogue; `subgroupAdd`/`subgroupElect` -> `WaveActiveSum`/
     `WaveIsFirstLane`, and per-wave slotting is derived from
     `WaveGetLaneCount()` + `SV_GroupIndex` (unique contiguous slots). Oracle:
     `deterministic = 1`, max rel 3.874e-08 (unchanged).
  3. **DrawParameters** — `SV_VertexID`/`SV_InstanceID` lower to
     `VertexIndex - BaseVertex`, pulling the `shaderDrawParameters` capability
     the device does not enable. Use `import glsl` + `gl_VertexIndex` /
     `gl_InstanceIndex` (raw SPIR-V `VertexIndex`/`InstanceIndex`, base = 0).
- **Samplers** — GLSL `sampler2D/3D/1D` -> Slang combined type
  `Sampler2D/3D/1D<float4>` (one COMBINED_IMAGE_SAMPLER descriptor;
  `[[vk::combinedImageSampler]]` is NOT supported in this Slang). `texelFetch`->
  `.Load(int4(c,0))`, `textureSize`->`.GetDimensions(...)`, compute `texture()`->
  `.SampleLevel(uvw,0)`, fragment `texture()`->`.Sample(uvw)`. Storage images:
  `image3D`+format -> `[[vk::image_format("rg16f")]] RWTexture3D<float4>`,
  `imageLoad/Store` -> `img[coord]`.
- **Render I/O** — `[[vk::location(N)]]` structs (SV_Position / SV_Target),
  `mul(ubo.mvp, float4(p,1))` with `-matrix-layout-column-major` to match the
  std140 mat4 upload, `[[vk::push_constant]] ConstantBuffer<Push>`.
- **Build discipline note** — ninja's Windows `/showIncludes` depfile does not
  reliably rebuild a blob consumer when only its baked `_spv.h` changes; touch
  `main.cpp` / `vkcheck_main.cpp` after a shader edit to force re-embed.
- **Verified** — ctest 270/270; `sesolver_vkcheck` all PASS (validation ON,
  reinit parity 0.000e+00, native VkFFT 4.9x); `--dump-frame --flow`,
  `--dump-frame-surface`, `--dump-frame-slice` all validation-clean and
  visually correct (surface mode is proton-only in BOTH GLSL and Slang -- an
  A/B confirmed pre-existing arc characteristic, not a migration regression).

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

## Remaining — latent resource / concurrency

ALL FIXED. The compute `Kernel` now frees its `VkShaderModule` at the end of
`Kernel::create` (`app/src/vk_compute.hpp`) — the module is consumed by pipeline
creation, matching the graphics pipelines; `destroy()` still guards it for the
partial-construction path. (vkcheck all-PASS confirms every kernel still builds.)

## Verified CLEAN (do not re-investigate)

Async compute↔render is correctly fence-synchronized (one-in-flight anchor
holds); OpenMP is bitwise-deterministic (fixed serial per-z-slab combine); there
are ZERO `std::mutex`/`std::thread` anywhere; `g_validation_errors` is a correct
`std::atomic`; the test `static const Relaxed` caches are pure memoization, not
order-coupled.

## Test-robustness observation — FIXED (2026-07-15)

- **`--selftest-cascade` timing-marginality** is fixed. The wall-only window was
  GPU-throughput-dependent (a wall x time_scale budget silently lost its margin
  when step throughput dropped). The arc now EARLY-EXITS the moment 2 photons
  land (the count only grows) with `time_scale` maxed at 16 and a 120 s window
  as the worst-case FAIL bound only. Robust across GPUs AND faster: success now
  exits in ~19 s instead of the fixed 90 s. (`app/src/selftest_arcs.hpp`.)

## Forward work -- render modernization + perf -- DONE (2026-07-15, office)

The Vulkan 1.4 modernization + perf arc is now complete. All verified on the
RTX 4060 (ctest 264/264 + vkcheck all-PASS + windowed validation 0-errors).

- **Present-path dynamic rendering (LAST render-modernization item) -- DONE.**
  `app/src/vk_present.hpp` dropped its `VkRenderPass` + framebuffers: the record
  path now transitions the acquired swapchain image UNDEFINED->COLOR_ATTACHMENT
  (image_layout_barrier), `vkCmdBeginRendering` with a `VkRenderingAttachmentInfo`
  (CLEAR/STORE), then ->PRESENT_SRC; the blit pipeline declares the swapchain
  format via `VkPipelineRenderingCreateInfo`. ImGui switched to
  `InitInfo.UseDynamicRendering=true` + `PipelineInfoMain.PipelineRenderingCreateInfo`
  (`presenter_.color_format()`; ImGui 1.92 deep-copies the format array). Verified
  validation-clean windowed (present + ImGui) AND offscreen (`--dump-frame`).
- **Perf track (fp32-EXACT) -- DONE, with the honest measurement.** P0 first
  (measure): the relax NORM reduction is only ~5% of the relax step (stepBody
  ~11 ms vs normReduce ~0.57 ms at 256^3); the FFT-bound step body dominates.
  - **P0** -- `Engine::profile_relax()` (timestamped stepBody vs normReduce),
    reported by `vkcheck` `timestamp profile 256^3 relax`. KEPT (the tool).
  - **P3** -- `project_deposit.comp` tree-reduce -> `subgroupAdd` + fixed-order
    per-subgroup combine. Correct + deterministic + in-tolerance (projection
    oracle `max rel = 3.87e-08, deterministic = 1`; requires subgroup
    arithmetic, width-agnostic via gl_NumSubgroups/gl_SubgroupID). KEPT.
  - **P4** -- fused relax renorm (norm_finalize.comp + scale_buf.comp: GPU
    computes inv and rescales in the SAME submit; double-precision energy stays
    async). Correct (relax oracle 8.1e-08, deterministic, reinit-clean). A clean
    same-process A/B measured **+0.2-0.9% (consistently positive but marginal)** --
    NOT the flat/worse pattern that gets reverted, so KEPT; the win grows on
    faster GPUs where per-step submit overhead is a larger fraction. Optional
    (blobs absent -> 2-submit fallback preserved).
  - **BANNED (unchanged):** `VK_EXT_shader_atomic_float`, `float_controls2`
    relaxed modes, `subgroupAdd` on the bandwidth-bound STEP reductions, any
    `VK_NV_*`, larger workgroups; NO fp16 in propagation without explicit opt-in.

## Validation layers -- ON BY DEFAULT in every preset (2026-07-15)

Every CMake preset now sets `VCPKG_MANIFEST_FEATURES=validation` (in `msvc-base`
and `linux-base`), so `vulkan-validationlayers` is always built+installed -- no
separate preset, no per-machine env var (that was why it "worked at home" but not
the office: the home enable was per-machine and never travelled). **The
`vulkan-validationlayers` port forces its OWN dynamic linkage** (`set(VCPKG_LIBRARY_LINKAGE
dynamic)` at portfile line 1) + `VCPKG_POLICY_DLLS_WITHOUT_LIBS`, so the layer
DLL + manifest land in `vcpkg_installed/<triplet>/bin` even on the static triplet
-- NO custom triplet needed (an overlay triplet force-rebuilds ALL deps, since
vcpkg hashes the triplet file into every package ABI -- dry-run confirmed; do NOT
do it). Installing the layer is inert; activate at runtime:
```
SES_VK_VALIDATION=1
VK_ADD_LAYER_PATH=<build>/vcpkg_installed/x64-windows-static/bin
VK_LOADER_LAYERS_ENABLE=*validation*
```
Confirmed end-to-end via the default `msvc-release` tree: `vkcheck` `[validation
ON]` + all-PASS. Windowed verify = launch `sesolver_app`, ~9 s, `Stop-Process
-Force`, grep stderr for `VUID`/error. First VVL build ~14 min from source, then
binary-cached. (The `--no-default-features` escape hatch: clear the var to skip.)

## See also

- [`TDD_RULES.md`](TDD_RULES.md) — the verification gates every fix must clear.
- Physics-audit open item: no `⟨L_z⟩` / probability-current *diagnostic* (the
  ±m ring states ARE preparable now via the L_z partial measurement); and the
  m-sign's absolute handedness vs Larmor rotation under B is untested — a
  B-on ring-rotation check would pin it.
