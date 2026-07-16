module;
#include <volk.h>
#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#include <vk_mem_alloc.h>
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <source_location>
#include <vector>
#ifdef SES_HAVE_VKFFT
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4702)  // unreachable code inside vkFFT.h at /O2
#endif
#include <VkFFT/vkFFT.h>
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
#endif
#include <complex>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
export module ses.vk.engine;
export import ses.vk.compute;
import ses.mc.tables;
export import ses.grid;
import ses.spectral;
export import ses.vec;
export import ses.drive;
export import ses.decay;


// ses_vk::Engine: the split-operator Strang step and imaginary-time
// relaxation, on raw Vulkan via the ses.vk.compute layer.
// SPIR-V blobs are dependency-injected (EngineKernels), so the engine has
// no resource system; the DeviceContext is passed in, so the same engine
// runs on a self-created device (headless: checks, clusters) or on handles
// adopted from the GUI shell.
//
// Numerical contract: Slang kernels baked offline to SPIR-V, the
// dispatch chain halfV . IFFT . kin . FFT . halfV (the inverse FFT = conj .
// FFT . conj/N), std140 parameter blocks, host-double reduction finishes.
// Synchronization is fully explicit: a compute-to-compute memory barrier
// before every dispatch that aliases psi (all of them), transfer barriers
// around uploads/readbacks, and a fence wait per submission.
// ses.vk GMF set, textually pre-claimed: volk.h supplies the VK_* macros
// (macros never cross module boundaries); vk_mem_alloc.h (same config as the
// module GMFs) keeps direct vma* calls compiling; both inoculate the TU
// against GMF/textual redefinitions.
// volk (this GMF's own include) already defined VK_NO_PROTOTYPES and declared the
// canonical vk* names as global function pointers, so header-only VkFFT
// compiles and links against volk's dispatch unmodified -- no include-order
// hack, no external command-buffer blocks: the engine owns the device and
// records VkFFTAppend straight into its own primary command buffers.


export namespace ses_vk {

// The SPIR-V blobs the engine's core propagation needs. The caller owns the
// storage (embedded C arrays in the harness; a file loader later).
struct EngineKernels {
    const unsigned char* mul = nullptr;    // phase_multiply.comp
    const unsigned char* half_mul = nullptr;  // half_mul.comp (in-shader V phase)
    std::size_t half_mul_size = 0;
    const unsigned char* kin_mul = nullptr;   // kin_mul.comp (1D k^2 tables)
    std::size_t kin_mul_size = 0;
    const unsigned char* damp = nullptr;      // damp_mul.comp (real absorber)
    std::size_t damp_size = 0;
    const unsigned char* pd = nullptr;  // phase_damp_mul.comp (fused V+mask;
    std::size_t pd_size = 0;            // optional: absent => separate passes)
    const unsigned char* mcwf = nullptr;  // mcwf_axpy.comp (fused multi-state
    std::size_t mcwf_size = 0;            // axpy; optional => per-state chain)
    const unsigned char* mc_density = nullptr;  // GPU marching cubes (all four
    std::size_t mc_density_size = 0;            // present or the path is off)
    const unsigned char* mc_classify = nullptr;
    std::size_t mc_classify_size = 0;
    const unsigned char* mc_scan = nullptr;
    std::size_t mc_scan_size = 0;
    const unsigned char* mc_emit = nullptr;
    std::size_t mc_emit_size = 0;
    std::size_t mul_size = 0;
    const unsigned char* conj = nullptr;   // conj_scale.comp
    std::size_t conj_size = 0;
    const unsigned char* fft = nullptr;    // fft_line<n>.comp for the grid n
    std::size_t fft_size = 0;
    const unsigned char* norm = nullptr;   // norm_peak.comp
    std::size_t norm_size = 0;
    const unsigned char* scale = nullptr;  // scale.comp
    std::size_t scale_size = 0;
    const unsigned char* norm_finalize = nullptr;  // norm_finalize.comp
    std::size_t norm_finalize_size = 0;             // (optional -> 2-submit relax)
    const unsigned char* scale_buf = nullptr;       // scale_buf.comp
    std::size_t scale_buf_size = 0;
    const unsigned char* kick = nullptr;   // dipole_kick.comp
    std::size_t kick_size = 0;
    const unsigned char* shear = nullptr;  // shear.comp
    std::size_t shear_size = 0;
    const unsigned char* inner = nullptr;  // inner_product.comp
    std::size_t inner_size = 0;
    const unsigned char* axpy = nullptr;   // axpy.comp
    std::size_t axpy_size = 0;
    const unsigned char* copy = nullptr;   // copy_state.comp
    std::size_t copy_size = 0;
    const unsigned char* synth = nullptr;  // synth.comp
    std::size_t synth_size = 0;
    const unsigned char* force = nullptr;  // mean_force.comp
    std::size_t force_size = 0;
    const unsigned char* dipole = nullptr;   // dipole.comp
    std::size_t dipole_size = 0;
    const unsigned char* project = nullptr;  // project_deposit.comp
    std::size_t project_size = 0;
    const unsigned char* bridge_store = nullptr;  // bridge_store.comp
    std::size_t bridge_store_size = 0;
    const unsigned char* bridge_load = nullptr;   // bridge_load.comp (checks)
    std::size_t bridge_load_size = 0;
    const unsigned char* flow_velocity = nullptr;  // flow_velocity.comp (fp32
    std::size_t flow_velocity_size = 0;            // Bohmian v; optional)
    const unsigned char* pack = nullptr;    // pack_half.comp (fp16 atlas)
    std::size_t pack_size = 0;
    const unsigned char* unpack = nullptr;  // unpack_half.comp
    std::size_t unpack_size = 0;
};

struct HalfMulParams {
    std::uint32_t n;
    float coef;  // -dt/2
    float pad0;
    float pad1;
};
// std140 mirror of mcwf_axpy.comp's Params (16-byte array strides).
struct alignas(16) McwfParams {
    std::uint32_t n;
    std::int32_t nx;
    std::int32_t ny;
    std::int32_t n_states;
    float box_min[4];
    float cell_h[4];
    std::int32_t n_radial;
    float h_radial;
    float rmax;
    float pad0;
    std::int32_t lm[8][4];  // [s] = {l, m, 0, 0}
    float coef[8][4];       // [s] = {cre, cim, inv_norm, 0}
};
struct KinMulParams {
    std::uint32_t n;
    std::int32_t nx;
    std::int32_t ny;
    float coef;  // -dt/2
};

class Engine {
public:
    // Free-energy estimate + normalized peak density from the per-step
    // renormalization.
    struct RelaxStats {
        double energy = 0.0;
        double peak = 0.0;
    };
    // GPU-reduced norm (sum |psi|^2 dV) and raw peak density.
    struct NormPeak {
        double sum = 0.0;
        double peak = 0.0;
    };

    Engine() = default;
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;
    ~Engine() { destroy(); }

    // half_v / kinetic are SplitOperator3D's phase tables; psi0 the initial
    // field. Cubic grids only (one baked fft_line<n>).
    // The propagator's phases are computed IN-SHADER from the scalar
    // potential (R32) and three 1D k^2 tables -- no complex phase tables.
    bool initialize(DeviceContext& ctx, const ses::Grid3D& grid,
                    const EngineKernels& blobs,
                    const std::vector<double>& potential, double dt,
                    const std::vector<std::complex<double>>& psi0) {
        dt_step_ = dt;
        ctx_ = &ctx;
        grid_ = grid;
        n_ = grid.x.n;
        cells_ = static_cast<std::size_t>(grid.size());
        cell_volume_ = grid.cell_volume();
        if (grid.y.n != n_ || grid.z.n != n_) {
            std::fprintf(stderr, "ses_vk::Engine: only cubic grids supported\n");
            return false;
        }
        mul_groups_ = static_cast<std::uint32_t>((cells_ + 255) / 256);
        field_bytes_ = 2 * cells_ * sizeof(float);

        // Dynamic-offset UBO slot stride: the device's minimum offset
        // alignment, grown to hold one KickParams block.
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(ctx.phys_dev, &props);
        kick_stride_ = static_cast<std::uint32_t>(
            props.limits.minUniformBufferOffsetAlignment);
        if (kick_stride_ == 0) {
            kick_stride_ = 256;
        }
        while (kick_stride_ < sizeof(KickParams)) {
            kick_stride_ *= 2;
        }

        if (!mul_.create(ctx, blobs.mul, blobs.mul_size,
                         {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                          {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                          {2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER}}) ||
            !half_mul_.create(ctx, blobs.half_mul, blobs.half_mul_size,
                              {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                               {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                               {2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER}}) ||
            !kin_mul_.create(ctx, blobs.kin_mul, blobs.kin_mul_size,
                             {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                              {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                              {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                              {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                              {4, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER}}) ||
            !damp_.create(ctx, blobs.damp, blobs.damp_size,
                          {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                           {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                           {2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER}}) ||
            !conj_.create(ctx, blobs.conj, blobs.conj_size,
                          {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                           {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER}}) ||
            !fft_.create(ctx, blobs.fft, blobs.fft_size,
                         {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                          {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER}}) ||
            !norm_.create(ctx, blobs.norm, blobs.norm_size,
                          {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                           {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER},
                           {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}}) ||
            !scale_.create(ctx, blobs.scale, blobs.scale_size,
                           {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                            {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER}}) ||
            !kick_.create(ctx, blobs.kick, blobs.kick_size,
                          {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                           {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC}}) ||
            !shear_.create(ctx, blobs.shear, blobs.shear_size,
                           {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                            {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER}}) ||
            !inner_.create(ctx, blobs.inner, blobs.inner_size,
                           {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                            {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER},
                            {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                            {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}}) ||
            !axpy_.create(ctx, blobs.axpy, blobs.axpy_size,
                          {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                           {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER},
                           {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}}) ||
            !copy_.create(ctx, blobs.copy, blobs.copy_size,
                          {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                           {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER},
                           {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}}) ||
            !synth_.create(ctx, blobs.synth, blobs.synth_size,
                           {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                            {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER},
                            {5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}}) ||
            !force_.create(ctx, blobs.force, blobs.force_size,
                           {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                            {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER},
                            {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                            {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}}) ||
            !dipole_.create(ctx, blobs.dipole, blobs.dipole_size,
                            {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                             {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER},
                             {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                             {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}}) ||
            !project_.create(ctx, blobs.project, blobs.project_size,
                             {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                              {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER},
                              {6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                              {7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                              {8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}}) ||
            !bridge_store_.create(ctx, blobs.bridge_store,
                                  blobs.bridge_store_size,
                                  {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                                   {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE},
                                   {2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER}}) ||
            !bridge_load_.create(ctx, blobs.bridge_load, blobs.bridge_load_size,
                                 {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                                  {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE},
                                  {2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER}}) ||
            !pack_.create(ctx, blobs.pack, blobs.pack_size,
                          {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                           {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER},
                           {6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}}) ||
            !unpack_.create(ctx, blobs.unpack, blobs.unpack_size,
                            {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                             {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER},
                             {6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}})) {
            return false;
        }

        if (!ctx.create_device_buffer(field_bytes_, &psi_) ||
            !ctx.create_device_buffer(cells_ * sizeof(float), &v_buf_) ||
            !ctx.create_device_buffer(static_cast<VkDeviceSize>(n_) *
                                          sizeof(float),
                                      &kx2_buf_) ||
            !ctx.create_device_buffer(static_cast<VkDeviceSize>(n_) *
                                          sizeof(float),
                                      &ky2_buf_) ||
            !ctx.create_device_buffer(static_cast<VkDeviceSize>(n_) *
                                          sizeof(float),
                                      &kz2_buf_) ||
            !ctx.create_device_buffer(2 * kGroups * sizeof(float), &partials_) ||
            !ctx.create_device_buffer(4 * sizeof(float), &renorm_) ||  // inv
            !ctx.create_host_buffer(field_bytes_,
                                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                    &staging_)) {
            return false;
        }
        staging_bytes_ = field_bytes_;

        // Optional fused V+mask kernel: absent blob => step() falls back to
        // separate half-kick + damp passes.
        if (blobs.pd != nullptr) {
            if (!phase_damp_.create(ctx, blobs.pd, blobs.pd_size,
                                    {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                                     {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                                     {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                                     {3, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER}})) {
                return false;
            }
            pd_kernel_ok_ = true;
        }
        // Optional fused multi-state axpy (the MCWF damping fast path).
        if (blobs.mcwf != nullptr) {
            if (!mcwf_.create(ctx, blobs.mcwf, blobs.mcwf_size,
                              {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                               {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER},
                               {5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}})) {
                return false;
            }
            mcwf_kernel_ok_ = true;
        }
        // Optional fused relax renorm: both blobs present => the GPU finishes
        // the norm (partials -> inv) and rescales in the same submit, so
        // relax_step() needs no host round-trip. Absent => 2-submit path.
        if (blobs.norm_finalize != nullptr && blobs.scale_buf != nullptr) {
            if (!finalize_.create(ctx, blobs.norm_finalize,
                                  blobs.norm_finalize_size,
                                  {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                                   {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER},
                                   {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}}) ||
                !scale_buf_.create(ctx, blobs.scale_buf, blobs.scale_buf_size,
                                   {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                                    {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER},
                                    {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}})) {
                return false;
            }
            fused_relax_ok_ = true;
        }
        // Optional fp32 Bohmian-velocity kernel for the streakline flow. Absent
        // blob => no velocity volume (the renderer's flow feature needs it).
        if (blobs.flow_velocity != nullptr) {
            if (!flow_vel_k_.create(ctx, blobs.flow_velocity,
                                    blobs.flow_velocity_size,
                                    {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                                     {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE},
                                     {2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER}})) {
                return false;
            }
            flow_vel_ok_ = true;
        }
        // Optional GPU marching cubes (Surface-mode isosurface extraction);
        // buffers are transient (mc_prepare/release_mc), kernels bake here.
        if (blobs.mc_density != nullptr && blobs.mc_classify != nullptr &&
            blobs.mc_scan != nullptr && blobs.mc_emit != nullptr) {
            const auto storage = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            const auto uniform = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            if (!mc_density_.create(ctx, blobs.mc_density,
                                    blobs.mc_density_size,
                                    {{0, storage}, {1, storage}, {2, uniform}}) ||
                !mc_classify_.create(
                    ctx, blobs.mc_classify, blobs.mc_classify_size,
                    {{0, storage}, {1, storage}, {2, storage}, {3, uniform}}) ||
                !mc_scan_.create(
                    ctx, blobs.mc_scan, blobs.mc_scan_size,
                    {{0, storage}, {1, storage}, {2, storage}, {3, uniform}}) ||
                !mc_emit_.create(ctx, blobs.mc_emit, blobs.mc_emit_size,
                                 {{0, storage},
                                  {1, storage},
                                  {2, storage},
                                  {3, storage},
                                  {4, storage},
                                  {5, uniform}})) {
                return false;
            }
            mc_kernel_ok_ = true;
        }

        // std140 parameter blocks, written once into host-mapped UBOs (every
        // submission fence-waits, so no in-flight aliasing; the scale UBO is
        // the only one rewritten, between submissions).
        const MulParams muln{static_cast<std::uint32_t>(cells_), 0, 0, 0};
        const HalfMulParams halfp{static_cast<std::uint32_t>(cells_),
                                  static_cast<float>(-0.5 * dt), 0.0f, 0.0f};
        // Full kick -dt: batch-interior half-kick pairs merged (see step()).
        const HalfMulParams fullp{static_cast<std::uint32_t>(cells_),
                                  static_cast<float>(-dt), 0.0f, 0.0f};
        const KinMulParams kinp{static_cast<std::uint32_t>(cells_), grid.x.n,
                                grid.y.n, static_cast<float>(-0.5 * dt)};
        const ConjParams conj1{static_cast<std::uint32_t>(cells_), 1.0f, 0.0f,
                               0.0f};
        const ConjParams conjN{static_cast<std::uint32_t>(cells_),
                               1.0f / static_cast<float>(cells_), 0.0f, 0.0f};
        const std::int32_t nn = n_ * n_;
        const FftParams fftp[3] = {
            {nn, n_, 0, 1, nn, 0, 0, 0},   // x-lines (contiguous)
            {n_, 1, nn, n_, nn, 0, 0, 0},  // y-lines
            {nn, 1, 0, nn, nn, 0, 0, 0},   // z-lines
        };
        // conjA: the per-axis inverse-FFT scale (1/n, the single transformed
        // axis) used by the shear path.
        const ConjParams conjA{static_cast<std::uint32_t>(cells_),
                               1.0f / static_cast<float>(n_), 0.0f, 0.0f};
        // finalize params: reuse ConjParams as {ngroups, cell_volume} so the
        // GPU can turn sum(partials)*dV into inv = 1/sqrt(norm_sq).
        const ConjParams renormp{kGroups,
                                 static_cast<float>(grid.cell_volume()), 0.0f,
                                 0.0f};
        const ShearParams shear_zero{};
        if (!write_ubo(&halfp_ubo_, &halfp, sizeof(halfp)) ||
            !write_ubo(&fullp_ubo_, &fullp, sizeof(fullp)) ||
            !write_ubo(&kinp_ubo_, &kinp, sizeof(kinp)) ||
            !write_ubo(&muln_ubo_, &muln, sizeof(muln)) ||
            !write_ubo(&conj1_ubo_, &conj1, sizeof(conj1)) ||
            !write_ubo(&conjN_ubo_, &conjN, sizeof(conjN)) ||
            !write_ubo(&fft_ubo_[0], &fftp[0], sizeof(FftParams)) ||
            !write_ubo(&fft_ubo_[1], &fftp[1], sizeof(FftParams)) ||
            !write_ubo(&fft_ubo_[2], &fftp[2], sizeof(FftParams)) ||
            !write_ubo(&scale_ubo_, &conj1, sizeof(conj1)) ||
            !write_ubo(&conjA_ubo_, &conjA, sizeof(conjA)) ||
            !write_ubo(&renorm_ubo_, &renormp, sizeof(renormp)) ||
            !write_ubo(&shear_ubo_[0], &shear_zero, sizeof(shear_zero)) ||
            !write_ubo(&shear_ubo_[1], &shear_zero, sizeof(shear_zero))) {
            return false;
        }
        if (!ctx.create_host_buffer(2 * kick_stride_,
                                    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                    &kick_ubo_)) {
            return false;
        }
        kick_slots_ = 2;

        // conj (0,0) coefficients: the axpy UBO is rewritten per projection;
        // the synth UBO per synthesis.
        const AxpyParams axpy0{static_cast<std::uint32_t>(cells_), 0, 0.0f,
                               0.0f};
        const SynthParams synth0{};
        if (!write_ubo(&axpy_ubo_, &axpy0, sizeof(axpy0)) ||
            !write_ubo(&synth_ubo_, &synth0, sizeof(synth0))) {
            return false;
        }

        // Descriptor pool shape: base sets + any-target sets + relax sets +
        // per-resident-state sets; the arena chains more pools of the same
        // shape when one runs dry.
        if (!arena_.create(ctx_ ? *ctx_ : ctx, 96, 192, 96, 2, 4)) {
            return false;
        }
        halfv_set_ = arena_.allocate(*ctx_, half_mul_.set_layout());
        if (halfv_set_ != VK_NULL_HANDLE) {
            arena_.write_buffer(*ctx_, halfv_set_, 0,
                                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, psi_.buf);
            arena_.write_buffer(*ctx_, halfv_set_, 1,
                                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, v_buf_.buf);
            arena_.write_buffer(*ctx_, halfv_set_, 2,
                                VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                halfp_ubo_.buf, sizeof(HalfMulParams));
        }
        fullv_set_ = arena_.allocate(*ctx_, half_mul_.set_layout());
        if (fullv_set_ != VK_NULL_HANDLE) {
            arena_.write_buffer(*ctx_, fullv_set_, 0,
                                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, psi_.buf);
            arena_.write_buffer(*ctx_, fullv_set_, 1,
                                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, v_buf_.buf);
            arena_.write_buffer(*ctx_, fullv_set_, 2,
                                VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                fullp_ubo_.buf, sizeof(HalfMulParams));
        }
        kin3_set_ = arena_.allocate(*ctx_, kin_mul_.set_layout());
        if (kin3_set_ != VK_NULL_HANDLE) {
            arena_.write_buffer(*ctx_, kin3_set_, 0,
                                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, psi_.buf);
            arena_.write_buffer(*ctx_, kin3_set_, 1,
                                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kx2_buf_.buf);
            arena_.write_buffer(*ctx_, kin3_set_, 2,
                                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, ky2_buf_.buf);
            arena_.write_buffer(*ctx_, kin3_set_, 3,
                                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kz2_buf_.buf);
            arena_.write_buffer(*ctx_, kin3_set_, 4,
                                VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                kinp_ubo_.buf, sizeof(KinMulParams));
        }
        conj1_set_ = make_unary_set(conj_, conj1_ubo_, sizeof(ConjParams));
        conjN_set_ = make_unary_set(conj_, conjN_ubo_, sizeof(ConjParams));
        for (int a = 0; a < 3; ++a) {
            fft_set_[a] = make_unary_set(fft_, fft_ubo_[a], sizeof(FftParams));
        }
        scale_set_ = make_unary_set(scale_, scale_ubo_, sizeof(ConjParams));
        conjA_set_ = make_unary_set(conj_, conjA_ubo_, sizeof(ConjParams));
        shear_set_[0] = make_unary_set(shear_, shear_ubo_[0], sizeof(ShearParams));
        shear_set_[1] = make_unary_set(shear_, shear_ubo_[1], sizeof(ShearParams));
        kick_set_ = arena_.allocate(*ctx_, kick_.set_layout());
        if (kick_set_ != VK_NULL_HANDLE) {
            arena_.write_buffer(*ctx_, kick_set_, 0,
                                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, psi_.buf);
            arena_.write_buffer(*ctx_, kick_set_, 1,
                                VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                                kick_ubo_.buf, sizeof(KickParams));
        }
        norm_set_ = arena_.allocate(*ctx_, norm_.set_layout());
        if (norm_set_ != VK_NULL_HANDLE) {
            arena_.write_buffer(*ctx_, norm_set_, 0,
                                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, psi_.buf);
            arena_.write_buffer(*ctx_, norm_set_, 1,
                                VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, muln_ubo_.buf,
                                sizeof(MulParams));
            arena_.write_buffer(*ctx_, norm_set_, 2,
                                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                partials_.buf);
        }
        if (fused_relax_ok_) {  // partials -> inv -> psi *= inv, one submit
            const auto storage = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            const auto uniform = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            finalize_set_ = arena_.allocate(*ctx_, finalize_.set_layout());
            if (finalize_set_ != VK_NULL_HANDLE) {
                arena_.write_buffer(*ctx_, finalize_set_, 0, storage,
                                    partials_.buf);
                arena_.write_buffer(*ctx_, finalize_set_, 1, uniform,
                                    renorm_ubo_.buf, sizeof(ConjParams));
                arena_.write_buffer(*ctx_, finalize_set_, 2, storage,
                                    renorm_.buf);
            }
            scale_buf_set_ = arena_.allocate(*ctx_, scale_buf_.set_layout());
            if (scale_buf_set_ != VK_NULL_HANDLE) {
                arena_.write_buffer(*ctx_, scale_buf_set_, 0, storage, psi_.buf);
                arena_.write_buffer(*ctx_, scale_buf_set_, 1, uniform,
                                    scale_ubo_.buf, sizeof(ConjParams));
                arena_.write_buffer(*ctx_, scale_buf_set_, 2, storage,
                                    renorm_.buf);
            }
            fused_relax_ok_ =
                finalize_set_ != VK_NULL_HANDLE && scale_buf_set_ != VK_NULL_HANDLE;
        }
        synth_any_set_ = arena_.allocate(*ctx_, synth_.set_layout());
        norm_any_set_ = arena_.allocate(*ctx_, norm_.set_layout());
        scale_any_set_ = arena_.allocate(*ctx_, scale_.set_layout());
        if (halfv_set_ == VK_NULL_HANDLE || fullv_set_ == VK_NULL_HANDLE ||
            kin3_set_ == VK_NULL_HANDLE ||
            conj1_set_ == VK_NULL_HANDLE || conjN_set_ == VK_NULL_HANDLE ||
            fft_set_[0] == VK_NULL_HANDLE || fft_set_[1] == VK_NULL_HANDLE ||
            fft_set_[2] == VK_NULL_HANDLE || scale_set_ == VK_NULL_HANDLE ||
            norm_set_ == VK_NULL_HANDLE || conjA_set_ == VK_NULL_HANDLE ||
            shear_set_[0] == VK_NULL_HANDLE || shear_set_[1] == VK_NULL_HANDLE ||
            kick_set_ == VK_NULL_HANDLE || synth_any_set_ == VK_NULL_HANDLE ||
            norm_any_set_ == VK_NULL_HANDLE ||
            scale_any_set_ == VK_NULL_HANDLE) {
            return false;
        }

        if (!upload_potential(potential) || !upload_k2_tables(grid) ||
            !upload_field(psi_, psi0)) {
            return false;
        }
        // Plan the VkFFT 3D transform directly on psi_'s VkBuffer (plan
        // creation compiles shaders via glslang). Failure leaves the engine
        // on the hand-rolled line FFT.
        ensure_vkfft();
        return true;
    }

    // Force the hand-rolled line-FFT path (A/B coverage); no-op without a plan.
    void set_use_vkfft(bool on) { use_vkfft_ = on; }
    bool vkfft_active() const {
#ifdef SES_HAVE_VKFFT
        return use_vkfft_ && vkfft_ready_;
#else
        return false;
#endif
    }

    // psi <- (halfV . IFFT . kin . FFT . halfV)^nsteps psi. One submission;
    // a compute-to-compute barrier precedes every psi-aliasing dispatch.
    // absorb=true damps psi by the absorbing mask after EVERY step (so the
    // absorption rate is independent of batch length); bridge=true records
    // the psi -> volume store into the SAME submission (no extra fences).
    void step(int nsteps, bool absorb = false, bool bridge = false) {
        OneShot shot;
        if (!shot.begin_compute(*ctx_)) {
            return;
        }
        const bool bridged = record_step_batch(shot.cb(), nsteps, absorb,
                                               bridge);
        shot.submit_and_wait(*ctx_);
        shot.destroy(*ctx_);
        if (bridged) {
            flip_volume();  // fence observed: the write is the display now
        }
    }

    // step()'s content into an arbitrary command buffer. Strang batch with
    // interior FULL kicks: e^{-iVdt/2} (kin e^{-iVdt})^{n-1} kin e^{-iVdt/2}
    // -- the operator product is IDENTICAL to per-step half-kick pairs, one
    // elementwise pass fewer per step. With absorb the mask rides the
    // trailing kick of every step (fused kernel; diagonal factors commute
    // exactly, per-step damping rate unchanged); fallback without the fused
    // kernel: separate passes. Returns whether the bridge tail recorded.
    bool record_step_batch(VkCommandBuffer cb, int nsteps, bool absorb,
                           bool bridge) {
        Recorder r{cb, true};
        const bool fuse = absorb && pd_half_set_ != VK_NULL_HANDLE;
        if (nsteps > 0) {
            r.dispatch(half_mul_, halfv_set_, mul_groups_);
            for (int s = 0; s < nsteps; ++s) {
                record_kin_body(r);
                const bool last = s + 1 == nsteps;
                if (fuse) {
                    r.dispatch(phase_damp_, last ? pd_half_set_ : pd_full_set_,
                               mul_groups_);
                } else {
                    r.dispatch(half_mul_, last ? halfv_set_ : fullv_set_,
                               mul_groups_);
                    record_absorb(r, absorb);
                }
            }
        }
        return record_bridge_tail(cb, bridge);
    }

    // Submit a step batch WITHOUT waiting: it runs on the compute queue
    // while the graphics queue renders the previous display volume
    // (ping-pong: the batch writes the OTHER image). Any engine OneShot
    // submitted meanwhile serializes AFTER the batch on the same queue, so
    // readouts stay correct; wait_async() reclaims the cb and flips the
    // display at a host-observed completion point. Falls back to the
    // blocking step() if the dedicated cb cannot be created.
    void step_async(int nsteps, bool absorb = false, bool bridge = false) {
        wait_async();
        if (ctx_->device_lost) {
            async_bridged_ = false;
            return;  // device lost: skip (the director drops to the CPU path)
        }
        if (!ensure_async()) {
            step(nsteps, absorb, bridge);
            return;
        }
        vkResetCommandPool(ctx_->device, async_pool_, 0);
        VkCommandBufferBeginInfo cbbi{};
        cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(async_cb_, &cbbi);
        async_bridged_ = record_step_batch(async_cb_, nsteps, absorb, bridge);
        vkEndCommandBuffer(async_cb_);
        vkResetFences(ctx_->device, 1, &async_fence_);
        VkSubmitInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &async_cb_;
        const VkResult asr = vkQueueSubmit(ctx_->compute_queue, 1, &si,
                                           async_fence_);
        if (asr != VK_SUCCESS) {
            std::fprintf(stderr, "ses_vk::Engine: async step-batch submit %s\n",
                         vk_result_str(asr));
            ctx_->device_lost = true;
            async_bridged_ = false;
            return;
        }
        async_pending_ = true;
    }

    // Wait for an in-flight async batch (no-op when none). Must run before
    // the async cb is re-recorded and before the display flip.
    void wait_async() {
        if (!async_pending_) {
            return;
        }
        const VkResult awr = vkWaitForFences(ctx_->device, 1, &async_fence_,
                                             VK_TRUE, 10ull * 1000 * 1000 * 1000);
        if (awr != VK_SUCCESS) {
            std::fprintf(stderr,
                         "ses_vk::Engine: async step-batch fence wait %s\n",
                         vk_result_str(awr));
            ctx_->device_lost = true;  // async batch hung/faulted: device is gone
        }
        async_pending_ = false;
        if (async_bridged_) {
            flip_volume();
            async_bridged_ = false;
        }
    }

    // Per-stage GPU timing of ONE representative propagation step on the VkFFT
    // hot path -- kick, forward FFT, kinetic multiply, inverse FFT -- via a
    // timestamp query pool. Only meaningful when vkfft_active(); returns
    // {valid=false} otherwise. Records its OWN one-shot command buffer, so the
    // normal step path is untouched.
    struct StepProfile {
        double kick_ms = 0.0;
        double fwd_fft_ms = 0.0;
        double kin_mul_ms = 0.0;
        double inv_fft_ms = 0.0;
        double total_ms = 0.0;
        bool valid = false;
    };
    StepProfile profile_step() {
        StepProfile p{};
#ifdef SES_HAVE_VKFFT
        if (!vkfft_active() || !ensure_profile_pool()) {
            return p;
        }
        // Host-side reset (hostQueryReset, enabled in create_device) -- no
        // in-flight queries here, so it cannot race the device.
        vkResetQueryPool(ctx_->device, profile_pool_, 0, kProfileStamps);
        OneShot shot;
        if (!shot.begin_compute(*ctx_)) {
            return p;
        }
        VkCommandBuffer cb = shot.cb();
        Recorder r{cb, true};
        // Stage boundaries: the RAW barriers between dispatches serialize the
        // stages, so BOTTOM_OF_PIPE stamps bracket each one cleanly.
        vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                             profile_pool_, 0);
        if (pd_full_set_ != VK_NULL_HANDLE) {
            r.dispatch(phase_damp_, pd_full_set_, mul_groups_);
        } else {
            r.dispatch(half_mul_, fullv_set_, mul_groups_);
        }
        vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
                             profile_pool_, 1);
        record_vkfft(r, -1);  // forward
        vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
                             profile_pool_, 2);
        r.dispatch(kin_mul_, kin3_set_, mul_groups_);
        vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
                             profile_pool_, 3);
        record_vkfft(r, 1);  // inverse (1/N by the plan)
        vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
                             profile_pool_, 4);
        shot.submit_and_wait(*ctx_);
        shot.destroy(*ctx_);

        std::uint64_t ts[kProfileStamps] = {};
        if (vkGetQueryPoolResults(
                ctx_->device, profile_pool_, 0, kProfileStamps, sizeof(ts), ts,
                sizeof(std::uint64_t),
                VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) !=
            VK_SUCCESS) {
            return p;
        }
        const double period = static_cast<double>(ctx_->timestamp_period);
        const std::uint32_t vb = ctx_->timestamp_valid_bits;
        const std::uint64_t mask =
            vb >= 64 ? ~0ull : ((std::uint64_t{1} << vb) - 1ull);
        auto ms = [&](std::uint64_t a, std::uint64_t b) {
            return static_cast<double>((b - a) & mask) * period * 1e-6;
        };
        p.kick_ms = ms(ts[0], ts[1]);
        p.fwd_fft_ms = ms(ts[1], ts[2]);
        p.kin_mul_ms = ms(ts[2], ts[3]);
        p.inv_fft_ms = ms(ts[3], ts[4]);
        p.total_ms = ms(ts[0], ts[4]);
        p.valid = true;
#endif  // SES_HAVE_VKFFT
        return p;
    }

    // Per-stage GPU timing of ONE imaginary-time relax iteration -- the Strang
    // step body vs the norm+peak reduction that follows it every step. Needs
    // relax tables; records its own one-shot buffer, so relax_step() is
    // untouched.
    struct RelaxProfile {
        double step_body_ms = 0.0;
        double norm_reduce_ms = 0.0;
        double total_ms = 0.0;
        bool valid = false;
    };
    RelaxProfile profile_relax() {
        RelaxProfile p{};
        if (!relax_tables_ready() || !ensure_profile_pool()) {
            return p;
        }
        vkResetQueryPool(ctx_->device, profile_pool_, 0, kProfileStamps);
        OneShot shot;
        if (!shot.begin_compute(*ctx_)) {
            return p;
        }
        VkCommandBuffer cb = shot.cb();
        Recorder r{cb, true};
        vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                             profile_pool_, 0);
        run_step_body(r, mul_, relax_half_set_, mul_, relax_kin_set_);
        vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
                             profile_pool_, 1);
        barrier_compute_to_compute(cb);  // step body writes psi; reduction reads
        norm_.bind(cb, norm_set_);
        vkCmdDispatch(cb, kGroups, 1, 1);
        vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
                             profile_pool_, 2);
        shot.submit_and_wait(*ctx_);
        shot.destroy(*ctx_);

        std::uint64_t ts[3] = {};
        if (vkGetQueryPoolResults(ctx_->device, profile_pool_, 0, 3, sizeof(ts),
                                  ts, sizeof(std::uint64_t),
                                  VK_QUERY_RESULT_64_BIT |
                                      VK_QUERY_RESULT_WAIT_BIT) != VK_SUCCESS) {
            return p;
        }
        const double period = static_cast<double>(ctx_->timestamp_period);
        const std::uint32_t vb = ctx_->timestamp_valid_bits;
        const std::uint64_t mask =
            vb >= 64 ? ~0ull : ((std::uint64_t{1} << vb) - 1ull);
        auto ms = [&](std::uint64_t a, std::uint64_t b) {
            return static_cast<double>((b - a) & mask) * period * 1e-6;
        };
        p.step_body_ms = ms(ts[0], ts[1]);
        p.norm_reduce_ms = ms(ts[1], ts[2]);
        p.total_ms = ms(ts[0], ts[2]);
        p.valid = true;
        return p;
    }

    // Driven Strang steps: kick(t) . step . kick(t+dt), theta = amplitude
    // cos(omega t) dt/2. Per-kick thetas differ within the batch, so the kick
    // parameters live in dynamic-offset slots of ONE host-mapped UBO and the
    // whole batch records as a single submission.
    void driven_step(const ses::DipoleDrive& d, double t0, double dt,
                     int nsteps, bool absorb = false, bool bridge = false) {
        const int kicks = 2 * nsteps;
        if (!ensure_kick_capacity(kicks)) {
            return;
        }
        char* slots = static_cast<char*>(kick_ubo_.mapped);
        for (int s = 0; s < nsteps; ++s) {
            const double t = t0 + s * dt;
            KickParams kp = make_kick_params(
                d.axis, d.amplitude * std::cos(d.omega * t) * 0.5 * dt);
            std::memcpy(slots + static_cast<std::size_t>(2 * s) * kick_stride_,
                        &kp, sizeof(kp));
            kp.theta = static_cast<float>(
                d.amplitude * std::cos(d.omega * (t + dt)) * 0.5 * dt);
            std::memcpy(
                slots + static_cast<std::size_t>(2 * s + 1) * kick_stride_, &kp,
                sizeof(kp));
        }
        vmaFlushAllocation(ctx_->allocator, kick_ubo_.alloc, 0, VK_WHOLE_SIZE);

        OneShot shot;
        if (!shot.begin_compute(*ctx_)) {
            return;
        }
        Recorder r{shot.cb(), true};
        for (int s = 0; s < nsteps; ++s) {
            r.dispatch_dyn(kick_, kick_set_, mul_groups_,
                           static_cast<std::uint32_t>(2 * s) * kick_stride_);
            run_step_body(r, half_mul_, halfv_set_, kin_mul_, kin3_set_);
            r.dispatch_dyn(kick_, kick_set_, mul_groups_,
                           static_cast<std::uint32_t>(2 * s + 1) * kick_stride_);
            record_absorb(r, absorb);
        }
        const bool bridged = record_bridge_tail(shot.cb(), bridge);
        shot.submit_and_wait(*ctx_);
        shot.destroy(*ctx_);
        if (bridged) {
            flip_volume();  // fence observed: the write is the display now
        }
    }

    // Swap the scalar potential (e.g. the field-augmented one before a
    // magnetic/Stark run); the half-kick phase is computed in-shader.
    bool set_potential(const std::vector<double>& v) {
        return upload_potential(v);
    }

    // The absorbing-boundary mask as a REAL R32 buffer (engine-owned);
    // batches damp with it when their absorb flag is set.
    bool set_absorber(const std::vector<double>& mask) {
        if (damp_buf_.buf == VK_NULL_HANDLE &&
            !ctx_->create_device_buffer(cells_ * sizeof(float), &damp_buf_)) {
            return false;
        }
        std::vector<float> mf(cells_);
        for (std::size_t i = 0; i < cells_; ++i) {
            mf[i] = static_cast<float>(mask[i]);
        }
        if (!upload_raw(damp_buf_, mf.data(), cells_ * sizeof(float))) {
            return false;
        }
        if (damp_set_ == VK_NULL_HANDLE) {
            damp_set_ = arena_.allocate(*ctx_, damp_.set_layout());
            if (damp_set_ == VK_NULL_HANDLE) {
                return false;
            }
            arena_.write_buffer(*ctx_, damp_set_, 0,
                                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, psi_.buf);
            arena_.write_buffer(*ctx_, damp_set_, 1,
                                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                damp_buf_.buf);
            arena_.write_buffer(*ctx_, damp_set_, 2,
                                VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                muln_ubo_.buf, sizeof(MulParams));
        }
        // Fused V+mask sets (step()'s fast path): full kick for interior
        // steps, half kick for the batch tail.
        if (pd_kernel_ok_ && pd_full_set_ == VK_NULL_HANDLE) {
            pd_full_set_ = arena_.allocate(*ctx_, phase_damp_.set_layout());
            pd_half_set_ = arena_.allocate(*ctx_, phase_damp_.set_layout());
            if (pd_full_set_ == VK_NULL_HANDLE ||
                pd_half_set_ == VK_NULL_HANDLE) {
                pd_full_set_ = VK_NULL_HANDLE;
                pd_half_set_ = VK_NULL_HANDLE;  // fallback path stays valid
                return true;
            }
            const auto wire = [this](VkDescriptorSet set, const Buffer& ubo) {
                arena_.write_buffer(*ctx_, set, 0,
                                    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                    psi_.buf);
                arena_.write_buffer(*ctx_, set, 1,
                                    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                    v_buf_.buf);
                arena_.write_buffer(*ctx_, set, 2,
                                    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                    damp_buf_.buf);
                arena_.write_buffer(*ctx_, set, 3,
                                    VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, ubo.buf,
                                    sizeof(HalfMulParams));
            };
            wire(pd_full_set_, fullp_ubo_);
            wire(pd_half_set_, halfp_ubo_);
        }
        return true;
    }

    // Exact three-shear rotation of psi about coordinate `axis` by theta --
    // the GPU transcription of ses.rotation rotate_axis. One submission.
    void rotate_axis_shear(int axis, double theta) {
        const int b = (axis + 1) % 3;  // in-plane axes (b x c = axis)
        const int c = (axis + 2) % 3;
        stage_rotation_ubos(b, c, theta);
        OneShot shot;
        if (!shot.begin_compute(*ctx_)) {
            return;
        }
        Recorder r{shot.cb(), true};
        record_rotation(r, b, c);
        shot.submit_and_wait(*ctx_);
        shot.destroy(*ctx_);
    }
    void rotate_z_shear(double theta) { rotate_axis_shear(2, theta); }

    // Magnetic Strang step: R(a) . real-step . R(a), a = (B/2)(dt/2), about
    // the field axis. half_ must hold the diamagnetic-augmented table
    // (set_half_potential). The half-angle is the same for every rotation in
    // the batch, so the two shear parameter sets are staged once and the
    // whole batch records as one submission.
    void magnetic_step(int axis, double half_angle, int nsteps,
                       bool absorb = false, bool bridge = false) {
        const int b = (axis + 1) % 3;
        const int c = (axis + 2) % 3;
        stage_rotation_ubos(b, c, half_angle);
        OneShot shot;
        if (!shot.begin_compute(*ctx_)) {
            return;
        }
        Recorder r{shot.cb(), true};
        for (int s = 0; s < nsteps; ++s) {
            record_rotation(r, b, c);
            run_step_body(r, half_mul_, halfv_set_, kin_mul_, kin3_set_);
            record_rotation(r, b, c);
            record_absorb(r, absorb);
        }
        const bool bridged = record_bridge_tail(shot.cb(), bridge);
        shot.submit_and_wait(*ctx_);
        shot.destroy(*ctx_);
        if (bridged) {
            flip_volume();  // fence observed: the write is the display now
        }
    }

    // Imaginary-time weight tables (packed vec2(w,0)) + dtau/dV for the
    // renormalization. Re-entrant: the tables are TRANSIENT --
    // directors upload them on entering relaxation and release_relax_tables
    // on leaving; the descriptor sets are allocated once and re-pointed.
    bool set_relax_tables(const std::vector<double>& half_w,
                          const std::vector<double>& kin_w, double dtau,
                          double cell_volume) {
        dtau_ = dtau;
        cell_volume_ = cell_volume;
        std::vector<float> hf(2 * cells_, 0.0f);
        std::vector<float> kf(2 * cells_, 0.0f);
        for (std::size_t i = 0; i < cells_; ++i) {
            hf[2 * i] = static_cast<float>(half_w[i]);
            kf[2 * i] = static_cast<float>(kin_w[i]);
        }
        if (relax_half_.buf == VK_NULL_HANDLE &&
            (!ctx_->create_device_buffer(field_bytes_, &relax_half_) ||
             !ctx_->create_device_buffer(field_bytes_, &relax_kin_))) {
            return false;
        }
        if (!upload_raw(relax_half_, hf.data(), field_bytes_) ||
            !upload_raw(relax_kin_, kf.data(), field_bytes_)) {
            return false;
        }
        if (relax_half_set_ == VK_NULL_HANDLE) {
            relax_half_set_ = make_mul_set(relax_half_.buf);
            relax_kin_set_ = make_mul_set(relax_kin_.buf);
        } else {
            // Re-point the once-allocated sets at the fresh buffers (legal:
            // every submission fence-waits).
            arena_.write_buffer(*ctx_, relax_half_set_, 1,
                                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                relax_half_.buf);
            arena_.write_buffer(*ctx_, relax_kin_set_, 1,
                                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                relax_kin_.buf);
        }
        return relax_half_set_ != VK_NULL_HANDLE &&
               relax_kin_set_ != VK_NULL_HANDLE;
    }

    // Free the transient imaginary-time tables (the sets stay, re-pointed by
    // the next set_relax_tables).
    void release_relax_tables() {
        if (relax_half_.buf == VK_NULL_HANDLE) {
            return;
        }
        vkDeviceWaitIdle(ctx_->device);
        ctx_->destroy_buffer(&relax_half_);
        ctx_->destroy_buffer(&relax_kin_);
    }
    bool relax_tables_ready() const {
        return relax_half_.buf != VK_NULL_HANDLE;
    }

    // e^{-H dtau} Strang steps with per-step renormalization. Each step: one
    // submission for the imaginary body + norm reduction + partials readback,
    // a host-double finish on THAT readback, then the 1/sqrt(norm) scale
    // submission. The pre-renorm norm decays as e^{-2 E dtau} -> free energy.
    RelaxStats relax_step(int nsteps) {
        RelaxStats stats;
        if (!relax_tables_ready()) {
            return stats;
        }
        for (int s = 0; s < nsteps; ++s) {
            OneShot shot;
            if (!shot.begin_compute(*ctx_)) {
                return stats;
            }
            Recorder r{shot.cb(), true};
            run_step_body(r, mul_, relax_half_set_, mul_, relax_kin_set_);
            barrier_compute_to_compute(shot.cb());
            norm_.bind(shot.cb(), norm_set_);
            vkCmdDispatch(shot.cb(), kGroups, 1, 1);
            if (fused_relax_ok_) {
                // The GPU finishes the renorm (partials -> inv) and rescales in
                // this same submit; the host reads the partials afterward only
                // for the double-precision energy/peak stat.
                barrier_compute_to_compute(shot.cb());
                finalize_.bind(shot.cb(), finalize_set_);
                vkCmdDispatch(shot.cb(), 1, 1, 1);
                barrier_compute_to_compute(shot.cb());
                scale_buf_.bind(shot.cb(), scale_buf_set_);
                vkCmdDispatch(shot.cb(), mul_groups_, 1, 1);
            }
            record_partials_readback(shot.cb());
            shot.submit_and_wait(*ctx_);
            shot.destroy(*ctx_);
            stats = fused_relax_ok_ ? finish_renorm_stats_only()
                                    : finish_renorm_from_staging();
        }
        return stats;
    }

    // Reset psi from a host field.
    bool upload_state(const std::vector<std::complex<double>>& psi) {
        return upload_field(psi_, psi);
    }

    // ---- resident states (one int handle space) ---------------------------

    // Upload a CPU state into its own resident fp32 buffer; returns a handle
    // usable with every per-state op, or -1 on failure.
    int create_state_buffer(const std::vector<std::complex<double>>& state) {
        const int handle = create_state_buffer_uninit();
        if (handle < 0) {
            return -1;
        }
        State* st = state_at(handle);
        if (!upload_field(st->buf, state)) {
            ctx_->destroy_buffer(&st->buf);
            return -1;
        }
        return handle;
    }

    // Free a resident state's buffer. The slot and its (expensive) descriptor
    // sets are RETAINED and the index recycled via free_full_states_ so a
    // synth/release churn doesn't leak them. Contract: the handle is dead
    // after this -- a reused slot may alias a later state (use-after-release
    // is a caller bug).
    void release_state(int handle) {
        State* st = state_at(handle);
        if (st == nullptr) {
            return;
        }
        vkDeviceWaitIdle(ctx_->device);
        const bool full = !st->is_half && st->inner_set != VK_NULL_HANDLE;
        ctx_->destroy_buffer(&st->buf);  // nulls st->buf -> state_at() == dead
        if (full) {
            free_full_states_.push_back(handle);
        }
    }

    // <state|psi> = sum conj(state)*psi * dV (fp16 states decode to scratch).
    std::complex<double> inner_with_psi(int handle) {
        State* st = state_at(handle);
        if (st == nullptr) {
            return {};
        }
        VkDescriptorSet set = st->inner_set;
        if (st->is_half) {
            const VkBuffer sbuf = decode(st, 0);
            if (sbuf == VK_NULL_HANDLE) {
                return {};
            }
            const auto storage = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            arena_.write_buffer(*ctx_, inner_any_set_, 0, storage, psi_.buf);
            arena_.write_buffer(*ctx_, inner_any_set_, 1,
                                VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                muln_ubo_.buf, sizeof(MulParams));
            arena_.write_buffer(*ctx_, inner_any_set_, 2, storage,
                                partials_.buf);
            arena_.write_buffer(*ctx_, inner_any_set_, 3, storage, sbuf);
            set = inner_any_set_;
        }
        OneShot shot;
        if (!shot.begin_compute(*ctx_)) {
            return {};
        }
        inner_.bind(shot.cb(), set);
        vkCmdDispatch(shot.cb(), kGroups, 1, 1);
        record_partials_readback(shot.cb());
        const bool ok = shot.submit_and_wait(*ctx_);
        shot.destroy(*ctx_);
        if (!ok) {
            return {};
        }
        vmaInvalidateAllocation(ctx_->allocator, staging_.alloc, 0,
                                VK_WHOLE_SIZE);
        const float* p = static_cast<const float*>(staging_.mapped);
        double re = 0.0;
        double im = 0.0;
        for (std::uint32_t g = 0; g < kGroups; ++g) {
            re += p[2 * g];
            im += p[2 * g + 1];
        }
        return std::complex<double>{re * cell_volume_, im * cell_volume_};
    }

    // Deflated imaginary-time relax: imaginary Strang body, Gram-Schmidt
    // project-out of every `lower` state (psi -= <phi|psi> phi), renorm.
    RelaxStats relax_deflated_step(const std::vector<int>& lower, int nsteps) {
        RelaxStats stats;
        if (!relax_tables_ready()) {
            return stats;
        }
        for (int s = 0; s < nsteps; ++s) {
            OneShot shot;
            if (!shot.begin_compute(*ctx_)) {
                return stats;
            }
            Recorder r{shot.cb(), true};
            run_step_body(r, mul_, relax_half_set_, mul_, relax_kin_set_);
            shot.submit_and_wait(*ctx_);
            shot.destroy(*ctx_);
            for (int h : lower) {
                const std::complex<double> c = inner_with_psi(h);
                subtract_projection(h, c.real(), c.imag());
            }
            stats = renormalize_and_estimate();
        }
        return stats;
    }

    // psi += (cre + i cim) * state: superposition seeding. fp32 states only.
    void add_state_into_psi(int handle, double cre, double cim) {
        subtract_projection(handle, -cre, -cim);
    }

    // Persistent scratch state (uninitialized fp32): callers that
    // re-synthesize repeatedly (the MCWF damping path) reuse ONE buffer
    // instead of per-call create + release (release_state device-idles).
    int create_scratch_state() { return create_state_buffer_uninit(); }

    static constexpr int kMcwfSlots = 8;
    struct McwfTerm {
        const std::vector<double>* u;  // radial table (n_radial entries)
        int l;
        int m;
        double cre;
        double cim;
        double inv_norm;  // 1 / sqrt(cached grid norm2 of the raw synth)
    };

    // psi += sum_s (cre + i cim)_s * (normalized eigenstate s), evaluated
    // fused in ONE dispatch. Same math as the per-state synth/normalize/axpy
    // chain; fp32 rounding differs only by in-register order. inv_norm comes
    // from the caller's cache (the grid norm of a fixed radial table is a
    // constant). False = kernel absent / slot overflow / setup failure: caller
    // falls back to the per-state chain.
    bool mcwf_axpy(const std::vector<McwfTerm>& terms, double h_radial,
                   double rmax, int n_radial) {
        if (!mcwf_kernel_ok_ || terms.empty() ||
            static_cast<int>(terms.size()) > kMcwfSlots || n_radial <= 0) {
            return false;
        }
        const std::size_t slot = static_cast<std::size_t>(n_radial);
        const VkDeviceSize rbytes =
            static_cast<VkDeviceSize>(kMcwfSlots) * slot * sizeof(float);
        if (mcwf_radial_.buf == VK_NULL_HANDLE ||
            mcwf_radial_bytes_ != rbytes) {
            vkDeviceWaitIdle(ctx_->device);
            ctx_->destroy_buffer(&mcwf_radial_);
            if (!ctx_->create_device_buffer(rbytes, &mcwf_radial_)) {
                return false;
            }
            mcwf_radial_bytes_ = rbytes;
            if (mcwf_set_ != VK_NULL_HANDLE) {
                arena_.write_buffer(*ctx_, mcwf_set_, 5,
                                    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                    mcwf_radial_.buf);
            }
        }
        McwfParams mp{};
        mp.n = static_cast<std::uint32_t>(cells_);
        mp.nx = grid_.x.n;
        mp.ny = grid_.y.n;
        mp.n_states = static_cast<std::int32_t>(terms.size());
        mp.box_min[0] = static_cast<float>(grid_.x.xmin);
        mp.box_min[1] = static_cast<float>(grid_.y.xmin);
        mp.box_min[2] = static_cast<float>(grid_.z.xmin);
        mp.cell_h[0] = static_cast<float>(grid_.x.spacing());
        mp.cell_h[1] = static_cast<float>(grid_.y.spacing());
        mp.cell_h[2] = static_cast<float>(grid_.z.spacing());
        mp.n_radial = n_radial;
        mp.h_radial = static_cast<float>(h_radial);
        mp.rmax = static_cast<float>(rmax);
        std::vector<float> slots(static_cast<std::size_t>(kMcwfSlots) * slot,
                                 0.0f);
        for (std::size_t s = 0; s < terms.size(); ++s) {
            const McwfTerm& t = terms[s];
            if (t.u == nullptr || t.inv_norm <= 0.0) {
                return false;
            }
            const std::size_t nsrc = std::min(t.u->size(), slot);
            for (std::size_t j = 0; j < nsrc; ++j) {
                slots[s * slot + j] = static_cast<float>((*t.u)[j]);
            }
            mp.lm[s][0] = t.l;
            mp.lm[s][1] = t.m;
            mp.coef[s][0] = static_cast<float>(t.cre);
            mp.coef[s][1] = static_cast<float>(t.cim);
            mp.coef[s][2] = static_cast<float>(t.inv_norm);
        }
        if (mcwf_ubo_.buf == VK_NULL_HANDLE) {
            if (!write_ubo(&mcwf_ubo_, &mp, sizeof(mp))) {
                return false;
            }
        } else {
            std::memcpy(mcwf_ubo_.mapped, &mp, sizeof(mp));
            vmaFlushAllocation(ctx_->allocator, mcwf_ubo_.alloc, 0,
                               VK_WHOLE_SIZE);
        }
        if (mcwf_set_ == VK_NULL_HANDLE) {
            mcwf_set_ = arena_.allocate(*ctx_, mcwf_.set_layout());
            if (mcwf_set_ == VK_NULL_HANDLE) {
                return false;
            }
            arena_.write_buffer(*ctx_, mcwf_set_, 0,
                                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, psi_.buf);
            arena_.write_buffer(*ctx_, mcwf_set_, 1,
                                VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                mcwf_ubo_.buf, sizeof(McwfParams));
            arena_.write_buffer(*ctx_, mcwf_set_, 5,
                                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                mcwf_radial_.buf);
        }
        if (!ensure_staging(rbytes)) {
            return false;
        }
        std::memcpy(staging_.mapped, slots.data(),
                    static_cast<std::size_t>(rbytes));
        vmaFlushAllocation(ctx_->allocator, staging_.alloc, 0, VK_WHOLE_SIZE);
        OneShot shot;
        if (!shot.begin_compute(*ctx_)) {
            return false;
        }
        const VkBufferCopy up{0, 0, rbytes};
        vkCmdCopyBuffer(shot.cb(), staging_.buf, mcwf_radial_.buf, 1, &up);
        barrier_transfer_to_compute(shot.cb());
        barrier_compute_to_compute(shot.cb());  // psi vs earlier submissions
        mcwf_.bind(shot.cb(), mcwf_set_);
        vkCmdDispatch(shot.cb(), mul_groups_, 1, 1);
        const bool ok = shot.submit_and_wait(*ctx_);
        shot.destroy(*ctx_);
        return ok;
    }

    // ---- GPU marching cubes (Surface mode) --------------------------------
    // Buffers are TRANSIENT like the relax tables: directors mc_prepare() on
    // Surface entry, release_mc() on exit. Determinism: every emit slot is
    // an integer function of the field (block scan), never of dispatch
    // order -- output is block-major cell-ordered; the vkcheck oracle
    // canonicalizes both sides by triangle centroid.

    bool mc_ready() const { return mc_buffers_ok_; }
    VkBuffer mc_vertex_buffer() const { return mc_vbuf_.buf; }
    VkBuffer mc_indirect_buffer() const { return mc_indirect_.buf; }

    bool mc_prepare(int max_tris) {
        if (!mc_kernel_ok_ || max_tris <= 0) {
            return false;
        }
        if (mc_buffers_ok_ && mc_max_tris_ == max_tris) {
            return true;
        }
        release_mc();
        // One workgroup per 8^3-cell block, dispatched 1D.
        const int nbx = (grid_.x.n - 1 + 7) / 8;
        const int nby = (grid_.y.n - 1 + 7) / 8;
        const int nbz = (grid_.z.n - 1 + 7) / 8;
        mc_nblk_[0] = nbx;
        mc_nblk_[1] = nby;
        mc_nblk_[2] = nbz;
        mc_nblocks_ = static_cast<std::uint32_t>(nbx * nby * nbz);
        if (mc_nblocks_ == 0 || mc_nblocks_ > 65535) {
            std::fprintf(stderr, "ses_vk: mc block count %u out of range\n",
                         mc_nblocks_);
            return false;
        }
        // Tables SSBO, uploaded from ses.mc.tables (single source of
        // truth with the CPU oracle): [0,256) edge, [256,4352) tri 256x16,
        // [4352,4608) per-case triangle counts.
        std::vector<std::int32_t> tab(4608, 0);
        for (int c = 0; c < 256; ++c) {
            tab[static_cast<std::size_t>(c)] = ses::mc::kEdgeTable[c];
            int count = 0;
            for (int t = 0; t < 16; ++t) {
                tab[static_cast<std::size_t>(256 + c * 16 + t)] =
                    ses::mc::kTriTable[c][t];
                if (ses::mc::kTriTable[c][t] != -1 && t % 3 == 0) {
                    ++count;
                }
            }
            tab[static_cast<std::size_t>(4352 + c)] = count;
        }
        const VkDeviceSize den_bytes = cells_ * sizeof(float);
        const VkDeviceSize blk_bytes = mc_nblocks_ * sizeof(std::uint32_t);
        const VkDeviceSize vb_bytes =
            static_cast<VkDeviceSize>(max_tris) * 27u * sizeof(float);
        if (!ctx_->create_device_buffer(den_bytes, &mc_den_) ||
            !ctx_->create_device_buffer(tab.size() * sizeof(std::int32_t),
                                        &mc_tables_) ||
            !ctx_->create_device_buffer(blk_bytes, &mc_block_sums_) ||
            !ctx_->create_device_buffer(blk_bytes, &mc_block_offsets_) ||
            // vbuf + indirect are written on the compute queue (mc_emit) and
            // read on the graphics queue (vertex / draw-indirect): CONCURRENT
            // so no queue-family ownership transfer is needed.
            !ctx_->create_device_buffer(vb_bytes, &mc_vbuf_,
                                        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                                        /*share_across_queues=*/true) ||
            !ctx_->create_device_buffer(6 * sizeof(std::uint32_t),
                                        &mc_indirect_,
                                        VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
                                        /*share_across_queues=*/true)) {
            release_mc();
            return false;
        }
        if (!upload_raw(mc_tables_, tab.data(),
                        tab.size() * sizeof(std::int32_t))) {
            release_mc();
            return false;
        }
        McParams mp{};
        mp.npts[0] = grid_.x.n;
        mp.npts[1] = grid_.y.n;
        mp.npts[2] = grid_.z.n;
        mp.nblk[0] = nbx;
        mp.nblk[1] = nby;
        mp.nblk[2] = nbz;
        mp.box_min[0] = static_cast<float>(grid_.x.xmin);
        mp.box_min[1] = static_cast<float>(grid_.y.xmin);
        mp.box_min[2] = static_cast<float>(grid_.z.xmin);
        mp.cell_h[0] = static_cast<float>(grid_.x.spacing());
        mp.cell_h[1] = static_cast<float>(grid_.y.spacing());
        mp.cell_h[2] = static_cast<float>(grid_.z.spacing());
        mp.iso = 0.0f;  // rewritten by every mc_extract
        mp.max_tris = max_tris;
        const McScanParams sp{static_cast<std::int32_t>(mc_nblocks_),
                              max_tris, 0, 0};
        if ((mc_ubo_.buf == VK_NULL_HANDLE &&
             !write_ubo(&mc_ubo_, &mp, sizeof(mp))) ||
            (mc_scan_ubo_.buf == VK_NULL_HANDLE &&
             !write_ubo(&mc_scan_ubo_, &sp, sizeof(sp)))) {
            release_mc();
            return false;
        }
        std::memcpy(mc_ubo_.mapped, &mp, sizeof(mp));
        std::memcpy(mc_scan_ubo_.mapped, &sp, sizeof(sp));
        vmaFlushAllocation(ctx_->allocator, mc_ubo_.alloc, 0, VK_WHOLE_SIZE);
        vmaFlushAllocation(ctx_->allocator, mc_scan_ubo_.alloc, 0,
                           VK_WHOLE_SIZE);
        // Sets are allocated once and re-pointed on re-prepare (arena sets
        // are never freed individually -- the relax-table pattern).
        const auto storage = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        const auto uniform = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        if (mc_den_set_ == VK_NULL_HANDLE) {
            mc_den_set_ = arena_.allocate(*ctx_, mc_density_.set_layout());
            mc_classify_set_ =
                arena_.allocate(*ctx_, mc_classify_.set_layout());
            mc_scan_set_ = arena_.allocate(*ctx_, mc_scan_.set_layout());
            mc_emit_set_ = arena_.allocate(*ctx_, mc_emit_.set_layout());
            if (mc_den_set_ == VK_NULL_HANDLE ||
                mc_classify_set_ == VK_NULL_HANDLE ||
                mc_scan_set_ == VK_NULL_HANDLE ||
                mc_emit_set_ == VK_NULL_HANDLE) {
                release_mc();
                return false;
            }
        }
        arena_.write_buffer(*ctx_, mc_den_set_, 0, storage, psi_.buf);
        arena_.write_buffer(*ctx_, mc_den_set_, 1, storage, mc_den_.buf);
        arena_.write_buffer(*ctx_, mc_den_set_, 2, uniform, muln_ubo_.buf,
                            sizeof(MulParams));
        arena_.write_buffer(*ctx_, mc_classify_set_, 0, storage, mc_den_.buf);
        arena_.write_buffer(*ctx_, mc_classify_set_, 1, storage,
                            mc_tables_.buf);
        arena_.write_buffer(*ctx_, mc_classify_set_, 2, storage,
                            mc_block_sums_.buf);
        arena_.write_buffer(*ctx_, mc_classify_set_, 3, uniform, mc_ubo_.buf,
                            sizeof(McParams));
        arena_.write_buffer(*ctx_, mc_scan_set_, 0, storage,
                            mc_block_sums_.buf);
        arena_.write_buffer(*ctx_, mc_scan_set_, 1, storage,
                            mc_block_offsets_.buf);
        arena_.write_buffer(*ctx_, mc_scan_set_, 2, storage,
                            mc_indirect_.buf);
        arena_.write_buffer(*ctx_, mc_scan_set_, 3, uniform,
                            mc_scan_ubo_.buf, sizeof(McScanParams));
        arena_.write_buffer(*ctx_, mc_emit_set_, 0, storage, mc_den_.buf);
        arena_.write_buffer(*ctx_, mc_emit_set_, 1, storage, mc_tables_.buf);
        arena_.write_buffer(*ctx_, mc_emit_set_, 2, storage,
                            mc_block_offsets_.buf);
        arena_.write_buffer(*ctx_, mc_emit_set_, 3, storage, mc_vbuf_.buf);
        arena_.write_buffer(*ctx_, mc_emit_set_, 4, storage, psi_.buf);
        arena_.write_buffer(*ctx_, mc_emit_set_, 5, uniform, mc_ubo_.buf,
                            sizeof(McParams));
        mc_max_tris_ = max_tris;
        mc_buffers_ok_ = true;
        return true;
    }

    void release_mc() {
        if (mc_den_.buf == VK_NULL_HANDLE && mc_vbuf_.buf == VK_NULL_HANDLE) {
            return;
        }
        vkDeviceWaitIdle(ctx_->device);  // the render may still read mc_vbuf_
        ctx_->destroy_buffer(&mc_den_);
        ctx_->destroy_buffer(&mc_tables_);
        ctx_->destroy_buffer(&mc_block_sums_);
        ctx_->destroy_buffer(&mc_block_offsets_);
        ctx_->destroy_buffer(&mc_vbuf_);
        ctx_->destroy_buffer(&mc_indirect_);
        mc_buffers_ok_ = false;
        mc_max_tris_ = 0;
    }

    // Extract the |psi|^2 isosurface at `iso` into the vertex buffer; one
    // fenced submission (waits any async physics batch first -- reads psi).
    // Returns the clamped triangle count, < 0 on failure.
    int mc_extract(double iso) {
        if (!mc_buffers_ok_) {
            return -1;
        }
        wait_async();
        // Per-call iso: rewrite the UBO between fenced submissions (the
        // scale-UBO precedent).
        McParams* mp = static_cast<McParams*>(mc_ubo_.mapped);
        mp->iso = static_cast<float>(iso);
        vmaFlushAllocation(ctx_->allocator, mc_ubo_.alloc, 0, VK_WHOLE_SIZE);
        OneShot shot;
        if (!shot.begin_compute(*ctx_)) {
            return -1;
        }
        Recorder r{shot.cb(), true};
        barrier_compute_to_compute(shot.cb());  // psi vs earlier submissions
        r.first = false;
        r.dispatch(mc_density_, mc_den_set_, mul_groups_);
        r.dispatch(mc_classify_, mc_classify_set_, mc_nblocks_);
        r.dispatch(mc_scan_, mc_scan_set_, 1);
        r.dispatch(mc_emit_, mc_emit_set_, mc_nblocks_);
        // Make the mc writes available. mc_vbuf_ (vertex fetch) and
        // mc_indirect_ (draw-indirect args) are read on the GRAPHICS queue in a
        // LATER submission (cross-queue), and mc_indirect_ by the transfer
        // readback just below. A COMPUTE-queue barrier CANNOT name graphics
        // stages (VERTEX_ATTRIBUTE_INPUT is invalid here -- validation flagged
        // it), so make the writes available broadly (ALL_COMMANDS + MEMORY_READ)
        // and let the submit_and_wait fence + CONCURRENT buffer sharing carry
        // the cross-queue visibility.
        VkBufferMemoryBarrier2 bmb[2]{};
        const VkBuffer bufs[2] = {mc_vbuf_.buf, mc_indirect_.buf};
        for (int i = 0; i < 2; ++i) {
            bmb[i].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
            bmb[i].srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            bmb[i].srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
            bmb[i].dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
            bmb[i].dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT;
            bmb[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            bmb[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            bmb[i].buffer = bufs[i];
            bmb[i].size = VK_WHOLE_SIZE;
        }
        VkDependencyInfo dep{};
        dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.bufferMemoryBarrierCount = 2;
        dep.pBufferMemoryBarriers = bmb;
        vkCmdPipelineBarrier2(shot.cb(), &dep);
        record_buffer_readback(shot.cb(), mc_indirect_,
                               6 * sizeof(std::uint32_t));
        const bool ok = shot.submit_and_wait(*ctx_);
        shot.destroy(*ctx_);
        if (!ok) {
            return -1;
        }
        vmaInvalidateAllocation(ctx_->allocator, staging_.alloc, 0,
                                VK_WHOLE_SIZE);
        const std::uint32_t* draw =
            static_cast<const std::uint32_t*>(staging_.mapped);
        if (draw[4] > draw[5] && !mc_overflow_warned_) {
            mc_overflow_warned_ = true;
            std::fprintf(stderr,
                         "ses_vk: mc surface clamped (%u > %u tris)\n",
                         draw[4], draw[5]);
        }
        return static_cast<int>(draw[5]);
    }

    // vkcheck readback: the first tri_count triangles as interleaved floats
    // (pos3 normal3 color3 per vertex).
    bool mc_read_vertices(std::vector<float>& out, int tri_count) {
        if (!mc_buffers_ok_ || tri_count < 0) {
            return false;
        }
        const VkDeviceSize bytes =
            static_cast<VkDeviceSize>(tri_count) * 27u * sizeof(float);
        if (bytes == 0) {
            out.clear();
            return true;
        }
        if (!ensure_staging(bytes)) {
            return false;
        }
        OneShot shot;
        if (!shot.begin_compute(*ctx_)) {
            return false;
        }
        record_buffer_readback(shot.cb(), mc_vbuf_, bytes);
        const bool ok = shot.submit_and_wait(*ctx_);
        shot.destroy(*ctx_);
        if (!ok) {
            return false;
        }
        vmaInvalidateAllocation(ctx_->allocator, staging_.alloc, 0,
                                VK_WHOLE_SIZE);
        const float* p = static_cast<const float*>(staging_.mapped);
        out.assign(p, p + static_cast<std::size_t>(tri_count) * 27u);
        return true;
    }

    // Re-synthesize into an existing fp32 state buffer (see scratch note).
    bool synthesize_state_over(int handle, const std::vector<double>& u,
                               int l, int m, double h_radial, double rmax,
                               int n_radial, double* out_norm2 = nullptr) {
        State* st = state_at(handle);
        if (st == nullptr || st->is_half) {
            return false;
        }
        return synthesize_into_buffer(st->buf, u, l, m, h_radial, rmax,
                                      n_radial, nullptr, out_norm2);
    }

    // psi <- src (bitwise; the quantum-jump collapse path). fp32 states only.
    void copy_into_psi(int handle) {
        State* st = state_at(handle);
        if (st == nullptr || st->is_half) {
            return;
        }
        OneShot shot;
        if (!shot.begin_compute(*ctx_)) {
            return;
        }
        copy_.bind(shot.cb(), st->copy_set);
        vkCmdDispatch(shot.cb(), mul_groups_, 1, 1);
        shot.submit_and_wait(*ctx_);
        shot.destroy(*ctx_);
    }

private:
    // Forward declaration must match the definition's access (clang enforces
    // [class.access.spec]; the definition lives in the private section).
    struct Recorder;

public:
    // psi <- psi * mask: the absorbing-boundary damp, recorded once PER STEP
    // so the absorption rate cannot depend on how steps are batched.
    void record_absorb(Recorder& r, bool absorb) {
        if (absorb && damp_set_ != VK_NULL_HANDLE) {
            r.dispatch(damp_, damp_set_, mul_groups_);
        }
    }

    // Record the real-time batch tail: the psi -> volume bridge into an
    // ALREADY-recording command buffer. Returns whether it recorded (the
    // caller flips the ping-pong at its host-observed completion point).
    bool record_bridge_tail(VkCommandBuffer cb, bool bridge) {
        // store_set_ (not ensure_volume): lazy creation submits its own
        // OneShot, which must never run while THIS cb is recording (the
        // shared OneShot pool would be reset under it). The first bridge
        // always goes through write_psi_to_volume, which creates the volume
        // outside any recording.
        if (bridge && store_set_[vol_write_] != VK_NULL_HANDLE) {
            barrier_compute_to_compute(cb);
            transition_volume(cb, vol_write_, VK_IMAGE_LAYOUT_GENERAL);
            bridge_store_.bind(cb, store_set_[vol_write_]);
            vkCmdDispatch(cb, mul_groups_, 1, 1);
            transition_volume(cb, vol_write_,
                              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            record_flow_velocity(cb);  // fp32 v from the same psi snapshot
            return true;
        }
        return false;
    }

    void apply_mask(int handle) {
        State* st = state_at(handle);
        if (st == nullptr || st->is_half) {
            return;
        }
        OneShot shot;
        if (!shot.begin_compute(*ctx_)) {
            return;
        }
        mul_.bind(shot.cb(), st->mul_set);
        vkCmdDispatch(shot.cb(), mul_groups_, 1, 1);
        shot.submit_and_wait(*ctx_);
        shot.destroy(*ctx_);
    }

    // GPU-reduced norm (sum |psi|^2 dV) and raw peak density.
    NormPeak norm_and_peak() {
        NormPeak out;
        OneShot shot;
        if (!shot.begin_compute(*ctx_)) {
            return out;
        }
        // Self-contained: depend on any prior compute (a possibly-unwaited
        // async step batch) writing psi, so a caller that forgot wait_async
        // still reads current psi, not stale.
        barrier_compute_to_compute(shot.cb());
        norm_.bind(shot.cb(), norm_set_);
        vkCmdDispatch(shot.cb(), kGroups, 1, 1);
        record_partials_readback(shot.cb());
        const bool ok = shot.submit_and_wait(*ctx_);
        shot.destroy(*ctx_);
        if (!ok) {
            return out;
        }
        vmaInvalidateAllocation(ctx_->allocator, staging_.alloc, 0,
                                VK_WHOLE_SIZE);
        const float* p = static_cast<const float*>(staging_.mapped);
        for (std::uint32_t g = 0; g < kGroups; ++g) {
            out.sum += p[2 * g];
            out.peak = std::max(out.peak, static_cast<double>(p[2 * g + 1]));
        }
        out.sum *= cell_volume_;
        return out;
    }

    // ---- SSBO -> 3D volume texture bridge (the renderer feed) -----------
    // The RG16F display volume (re, im): the last COMPLETED bridge write.
    // NULL until the first write (the renderer falls back).
    VkImage volume_image() {
        return vol_display_ >= 0 ? volume_[vol_display_].img : VK_NULL_HANDLE;
    }
    VkImageView volume_view() {
        return vol_display_ >= 0 ? volume_[vol_display_].view : VK_NULL_HANDLE;
    }
    // The low-res fp32-computed Bohmian velocity field (last completed bridge).
    VkImageView flow_velocity_view() {
        return (flow_vel_ok_ && vol_display_ >= 0) ? flow_vel_[vol_display_].view
                                                   : VK_NULL_HANDLE;
    }

    // Copy psi into the write volume (imageStore, one texel per cell). The
    // engine OWNS the layout round-trip: the store runs in GENERAL, then the
    // image is handed to SHADER_READ_ONLY_OPTIMAL -- the layout the
    // renderer's sampled-image descriptors declare for it. Nobody else may
    // transition these images.
    bool write_psi_to_volume() {
        if (!ensure_volume()) {
            return false;
        }
        // Immediate flip below: drain any in-flight async batch first, or a
        // caller inside the async-pending window (Key-R reseed) double-flips
        // and the renderer samples the stale pre-reset cloud for a frame.
        wait_async();
        OneShot shot;
        if (!shot.begin_compute(*ctx_)) {
            return false;
        }
        barrier_compute_to_compute(shot.cb());  // psi vs prior compute
        transition_volume(shot.cb(), vol_write_, VK_IMAGE_LAYOUT_GENERAL);
        bridge_store_.bind(shot.cb(), store_set_[vol_write_]);
        vkCmdDispatch(shot.cb(), mul_groups_, 1, 1);
        transition_volume(shot.cb(), vol_write_,
                          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        record_flow_velocity(shot.cb());
        shot.submit_and_wait(*ctx_);
        shot.destroy(*ctx_);
        flip_volume();  // fence observed: the write is the display now
        return true;
    }

    // psi -> volume -> scratch SSBO -> host; the RG16F texel quantizes, so
    // the check compares against CPU-side fp16 round-to-nearest (check only).
    bool bridge_roundtrip(std::vector<float>& out) {
        if (!write_psi_to_volume()) {
            return false;
        }
        if (scratch_bridge_.buf == VK_NULL_HANDLE) {
            if (!ctx_->create_device_buffer(field_bytes_, &scratch_bridge_)) {
                return false;
            }
            for (int i = 0; i < 2; ++i) {
                load_set_[i] = arena_.allocate(*ctx_, bridge_load_.set_layout());
                if (load_set_[i] == VK_NULL_HANDLE) {
                    return false;
                }
                arena_.write_buffer(*ctx_, load_set_[i], 0,
                                    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                    scratch_bridge_.buf);
                arena_.write_image(*ctx_, load_set_[i], 1, volume_[i].view);
                arena_.write_buffer(*ctx_, load_set_[i], 2,
                                    VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                    bridge_ubo_.buf, sizeof(BridgeParams));
            }
        }
        OneShot shot;
        if (!shot.begin_compute(*ctx_)) {
            return false;
        }
        barrier_compute_to_compute(shot.cb());  // image written by the store
        transition_volume(shot.cb(), vol_display_,
                          VK_IMAGE_LAYOUT_GENERAL);  // imageLoad
        bridge_load_.bind(shot.cb(), load_set_[vol_display_]);
        vkCmdDispatch(shot.cb(), mul_groups_, 1, 1);
        record_buffer_readback(shot.cb(), scratch_bridge_, field_bytes_);
        transition_volume(shot.cb(), vol_display_,
                          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        const bool ok = shot.submit_and_wait(*ctx_);
        shot.destroy(*ctx_);
        if (!ok) {
            return false;
        }
        vmaInvalidateAllocation(ctx_->allocator, staging_.alloc, 0,
                                VK_WHOLE_SIZE);
        const float* p = static_cast<const float*>(staging_.mapped);
        out.assign(p, p + 2 * cells_);
        return true;
    }

    // ---- semiclassical radiation (Ehrenfest mean force) -----------------
    // Upload the scalar potential (R32) for mean_force: the kernel takes the
    // periodic central differences IN-SHADER, so no gradient field is stored
    // (computed at read time from cache-hot neighbor taps).
    bool set_potential_gradient(const std::vector<double>& v) {
        if (force_v_buf_.buf == VK_NULL_HANDLE) {
            struct alignas(16) ForceParams {
                std::uint32_t n, nx, ny, nz;
                float inv_2h[4];
            };
            const ForceParams fp{
                static_cast<std::uint32_t>(cells_),
                static_cast<std::uint32_t>(grid_.x.n),
                static_cast<std::uint32_t>(grid_.y.n),
                static_cast<std::uint32_t>(grid_.z.n),
                {static_cast<float>(1.0 / (2.0 * grid_.x.spacing())),
                 static_cast<float>(1.0 / (2.0 * grid_.y.spacing())),
                 static_cast<float>(1.0 / (2.0 * grid_.z.spacing())), 0.0f}};
            if (!ctx_->create_device_buffer(cells_ * sizeof(float),
                                            &force_v_buf_) ||
                !ctx_->create_device_buffer(4 * kGroups * sizeof(float),
                                            &force_partials_) ||
                !write_ubo(&force_ubo_, &fp, sizeof(fp))) {
                return false;
            }
            force_set_ = arena_.allocate(*ctx_, force_.set_layout());
            if (force_set_ == VK_NULL_HANDLE) {
                return false;
            }
            const auto storage = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            arena_.write_buffer(*ctx_, force_set_, 0, storage, psi_.buf);
            arena_.write_buffer(*ctx_, force_set_, 1,
                                VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                force_ubo_.buf, sizeof(ForceParams));
            arena_.write_buffer(*ctx_, force_set_, 2, storage,
                                force_partials_.buf);
            arena_.write_buffer(*ctx_, force_set_, 4, storage,
                                force_v_buf_.buf);
        }
        std::vector<float> vf(cells_);
        for (std::size_t i = 0; i < cells_; ++i) {
            vf[i] = static_cast<float>(v[i]);
        }
        return upload_raw(force_v_buf_, vf.data(), cells_ * sizeof(float));
    }

    // <grad V> = sum |psi|^2 grad V * dV -- the Ehrenfest dipole
    // acceleration. Zero if no gradient was uploaded.
    ses::Vec3d mean_force() {
        if (force_set_ == VK_NULL_HANDLE) {
            return ses::Vec3d{};
        }
        OneShot shot;
        if (!shot.begin_compute(*ctx_)) {
            return ses::Vec3d{};
        }
        force_.bind(shot.cb(), force_set_);
        vkCmdDispatch(shot.cb(), kGroups, 1, 1);
        record_buffer_readback(shot.cb(), force_partials_,
                               4 * kGroups * sizeof(float));
        const bool ok = shot.submit_and_wait(*ctx_);
        shot.destroy(*ctx_);
        if (!ok) {
            return ses::Vec3d{};
        }
        vmaInvalidateAllocation(ctx_->allocator, staging_.alloc, 0,
                                VK_WHOLE_SIZE);
        const float* p = static_cast<const float*>(staging_.mapped);
        double gx = 0.0;
        double gy = 0.0;
        double gz = 0.0;
        for (std::uint32_t g = 0; g < kGroups; ++g) {
            gx += p[4 * g + 0];
            gy += p[4 * g + 1];
            gz += p[4 * g + 2];
        }
        return ses::Vec3d{gx * cell_volume_, gy * cell_volume_,
                          gz * cell_volume_};
    }

    // ---- orbital synthesis ----------------------------------------------
    // psi <- normalized (u(|r|)/|r|) Y_lm, synthesized on the GPU from the
    // radial table u_nl(r). h_radial = rmax/(n_radial+1).
    bool synthesize_into_psi(const std::vector<double>& u, int l, int m,
                             double h_radial, double rmax, int n_radial) {
        return synthesize_into_buffer(psi_, u, l, m, h_radial, rmax, n_radial);
    }

    // Synthesize a normalized fp32 eigenstate into its OWN resident state
    // buffer (the atlas path; psi is untouched) and return a handle.
    // *out_peak gets the normalized peak |psi|^2; *out_norm2 the
    // PRE-normalization grid norm (the projection population normalizer).
    int synthesize_state(const std::vector<double>& u, int l, int m,
                         double h_radial, double rmax, int n_radial,
                         double* out_peak = nullptr,
                         double* out_norm2 = nullptr) {
        const int handle = create_state_buffer_uninit();
        if (handle < 0) {
            return -1;
        }
        State* st = state_at(handle);
        if (!synthesize_into_buffer(st->buf, u, l, m, h_radial, rmax,
                                    n_radial, out_peak, out_norm2)) {
            ctx_->destroy_buffer(&st->buf);
            return -1;
        }
        return handle;
    }

    // Synthesize + normalize in fp32 (the tested path), then pack to fp16
    // storage (cells uints, half the footprint) and return an fp16 handle.
    // Consumers unpack fp16 to scratch on demand (decode-on-use).
    int synthesize_state_half(const std::vector<double>& u, int l, int m,
                              double h_radial, double rmax, int n_radial,
                              double* out_peak = nullptr,
                              double* out_norm2 = nullptr) {
        if (!ensure_fp16()) {
            return -1;
        }
        Buffer tmp{};
        if (!ctx_->create_device_buffer(field_bytes_, &tmp)) {
            return -1;
        }
        if (!synthesize_into_buffer(tmp, u, l, m, h_radial, rmax, n_radial,
                                    out_peak, out_norm2)) {
            ctx_->destroy_buffer(&tmp);
            return -1;
        }
        State st;
        st.is_half = true;
        if (!ctx_->create_device_buffer(cells_ * sizeof(std::uint32_t),
                                        &st.buf)) {
            ctx_->destroy_buffer(&tmp);
            return -1;
        }
        // pack tmp(fp32, binding 0) -> st.buf(fp16, binding 6).
        const auto storage = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        arena_.write_buffer(*ctx_, pack_set_, 0, storage, tmp.buf);
        arena_.write_buffer(*ctx_, pack_set_, 6, storage, st.buf.buf);
        OneShot shot;
        if (!shot.begin_compute(*ctx_)) {
            ctx_->destroy_buffer(&st.buf);
            ctx_->destroy_buffer(&tmp);
            return -1;
        }
        barrier_compute_to_compute(shot.cb());
        pack_.bind(shot.cb(), pack_set_);
        vkCmdDispatch(shot.cb(), mul_groups_, 1, 1);
        shot.submit_and_wait(*ctx_);
        shot.destroy(*ctx_);
        ctx_->destroy_buffer(&tmp);
        states_.push_back(st);
        return static_cast<int>(states_.size()) - 1;
    }

    // <to| r |from> = sum conj(to)*(x,y,z)*from * dV from two resident
    // states (fp32, or fp16 decoded to scratch), component-wise complex.
    ses::DipoleMatrixElement dipole_between(int to_h, int from_h) {
        State* to = state_at(to_h);
        State* from = state_at(from_h);
        if (to == nullptr || from == nullptr || !ensure_dipole()) {
            return {};
        }
        VkBuffer to_buf = decode(to, 0);
        VkBuffer from_buf = decode(from, 1);
        if (to_buf == VK_NULL_HANDLE || from_buf == VK_NULL_HANDLE) {
            return {};
        }
        DipoleParams dp{};
        dp.n = static_cast<std::uint32_t>(cells_);
        dp.nx = grid_.x.n;
        dp.ny = grid_.y.n;
        dp.box_min[0] = static_cast<float>(grid_.x.xmin);
        dp.box_min[1] = static_cast<float>(grid_.y.xmin);
        dp.box_min[2] = static_cast<float>(grid_.z.xmin);
        dp.cell_h[0] = static_cast<float>(grid_.x.spacing());
        dp.cell_h[1] = static_cast<float>(grid_.y.spacing());
        dp.cell_h[2] = static_cast<float>(grid_.z.spacing());
        std::memcpy(dipole_ubo_.mapped, &dp, sizeof(dp));
        vmaFlushAllocation(ctx_->allocator, dipole_ubo_.alloc, 0,
                           VK_WHOLE_SIZE);
        // The shared set is re-pointed per call (all submissions fence-wait,
        // so the set is never in flight when rewritten).
        const auto storage = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        arena_.write_buffer(*ctx_, dipole_set_, 0, storage, to_buf);
        arena_.write_buffer(*ctx_, dipole_set_, 3, storage, from_buf);

        OneShot shot;
        if (!shot.begin_compute(*ctx_)) {
            return {};
        }
        dipole_.bind(shot.cb(), dipole_set_);
        vkCmdDispatch(shot.cb(), kGroups, 1, 1);
        record_buffer_readback(shot.cb(), dipole_partials_,
                               6 * kGroups * sizeof(float));
        const bool ok = shot.submit_and_wait(*ctx_);
        shot.destroy(*ctx_);
        if (!ok) {
            return {};
        }
        vmaInvalidateAllocation(ctx_->allocator, staging_.alloc, 0,
                                VK_WHOLE_SIZE);
        const float* p = static_cast<const float*>(staging_.mapped);
        double d6[6] = {0, 0, 0, 0, 0, 0};
        for (std::uint32_t g = 0; g < kGroups; ++g) {
            for (int c = 0; c < 6; ++c) {
                d6[c] += p[6 * g + c];
            }
        }
        return ses::DipoleMatrixElement{
            std::complex<double>{d6[0] * cell_volume_, d6[1] * cell_volume_},
            std::complex<double>{d6[2] * cell_volume_, d6[3] * cell_volume_},
            std::complex<double>{d6[4] * cell_volume_, d6[5] * cell_volume_}};
    }

    // ---- orbital-free angular projection --------------------------------
    // Upload the static counting-sort geometry (ses::build_radial_bin_index)
    // and allocate g_lm[ncomp*nr]. Call once after the radial grid is fixed.
    bool set_projection_index(const std::vector<std::uint32_t>& sorted_cell,
                              const std::vector<std::uint32_t>& bin_off,
                              int n_radial, double h_radial, int l_max) {
        proj_nr_ = n_radial;
        proj_ncomp_ = (l_max + 1) * (l_max + 1);
        proj_h_radial_ = h_radial;
        const VkDeviceSize glm_bytes =
            2ull * proj_ncomp_ * proj_nr_ * sizeof(float);
        const ProjectParams pp0{};
        if (!ctx_->create_device_buffer(sorted_cell.size() * 4,
                                        &proj_sorted_buf_) ||
            !ctx_->create_device_buffer(bin_off.size() * 4, &proj_binoff_buf_) ||
            !ctx_->create_device_buffer(glm_bytes, &glm_buf_) ||
            !write_ubo(&proj_ubo_, &pp0, sizeof(pp0))) {
            return false;
        }
        if (!upload_raw(proj_sorted_buf_, sorted_cell.data(),
                        sorted_cell.size() * 4) ||
            !upload_raw(proj_binoff_buf_, bin_off.data(), bin_off.size() * 4)) {
            return false;
        }
        proj_set_ = arena_.allocate(*ctx_, project_.set_layout());
        if (proj_set_ == VK_NULL_HANDLE) {
            return false;
        }
        const auto storage = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        arena_.write_buffer(*ctx_, proj_set_, 0, storage, psi_.buf);
        arena_.write_buffer(*ctx_, proj_set_, 1,
                            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, proj_ubo_.buf,
                            sizeof(ProjectParams));
        arena_.write_buffer(*ctx_, proj_set_, 6, storage, proj_sorted_buf_.buf);
        arena_.write_buffer(*ctx_, proj_set_, 7, storage, proj_binoff_buf_.buf);
        arena_.write_buffer(*ctx_, proj_set_, 8, storage, glm_buf_.buf);
        return true;
    }

    // Deposit psi -> g_lm (ONE grid pass, independent of state count), read
    // back to glm_host_ as double. Then call project_amplitude per state.
    void project_psi() {
        if (proj_set_ == VK_NULL_HANDLE) {
            return;  // set_projection_index failed/absent
        }
        ProjectParams pp{};
        pp.nx = grid_.x.n;
        pp.ny = grid_.y.n;
        pp.nr = proj_nr_;
        pp.h_radial = static_cast<float>(proj_h_radial_);
        pp.box_min[0] = static_cast<float>(grid_.x.xmin);
        pp.box_min[1] = static_cast<float>(grid_.y.xmin);
        pp.box_min[2] = static_cast<float>(grid_.z.xmin);
        pp.cell_h[0] = static_cast<float>(grid_.x.spacing());
        pp.cell_h[1] = static_cast<float>(grid_.y.spacing());
        pp.cell_h[2] = static_cast<float>(grid_.z.spacing());
        pp.dv = static_cast<float>(cell_volume_);
        std::memcpy(proj_ubo_.mapped, &pp, sizeof(pp));
        vmaFlushAllocation(ctx_->allocator, proj_ubo_.alloc, 0, VK_WHOLE_SIZE);

        const VkDeviceSize glm_bytes =
            2ull * proj_ncomp_ * proj_nr_ * sizeof(float);
        if (!ensure_staging(glm_bytes)) {
            return;
        }
        OneShot shot;
        if (!shot.begin_compute(*ctx_)) {
            return;
        }
        barrier_compute_to_compute(shot.cb());  // see norm_and_peak
        project_.bind(shot.cb(), proj_set_);
        vkCmdDispatch(shot.cb(), static_cast<std::uint32_t>(proj_nr_), 1, 1);
        record_buffer_readback(shot.cb(), glm_buf_, glm_bytes);
        const bool ok = shot.submit_and_wait(*ctx_);
        shot.destroy(*ctx_);
        if (!ok) {
            return;
        }
        vmaInvalidateAllocation(ctx_->allocator, staging_.alloc, 0,
                                VK_WHOLE_SIZE);
        const float* raw = static_cast<const float*>(staging_.mapped);
        // Shape once, reuse thereafter (every element is overwritten below);
        // avoids re-allocating the ncomp vectors per projection.
        if (glm_host_.size() != static_cast<std::size_t>(proj_ncomp_)) {
            glm_host_.assign(static_cast<std::size_t>(proj_ncomp_),
                             std::vector<std::complex<double>>(
                                 static_cast<std::size_t>(proj_nr_)));
        }
        for (int c = 0; c < proj_ncomp_; ++c) {
            for (int j = 0; j < proj_nr_; ++j) {
                const std::size_t o =
                    2 * (static_cast<std::size_t>(c) * proj_nr_ +
                         static_cast<std::size_t>(j));
                glm_host_[static_cast<std::size_t>(c)]
                         [static_cast<std::size_t>(j)] =
                             std::complex<double>{raw[o], raw[o + 1]};
            }
        }
    }

    // <n|psi> raw amplitude = sum_j u_nl[j] g_lm[lm(l,m)][j] (double CPU
    // finish). Needs a prior project_psi().
    std::complex<double> project_amplitude(const std::vector<double>& u, int l,
                                           int m) const {
        const std::size_t comp = static_cast<std::size_t>(l * l + (l + m));
        if (comp >= glm_host_.size()) {
            return {};
        }
        const std::vector<std::complex<double>>& gc = glm_host_[comp];
        std::complex<double> raw{};
        const int n = std::min(static_cast<int>(u.size()), proj_nr_);
        for (int j = 0; j < n; ++j) {
            raw += u[static_cast<std::size_t>(j)] * gc[static_cast<std::size_t>(j)];
        }
        return raw;
    }

    // psi <- s * psi (fp32 drift renormalization).
    void scale(float s) {
        const ConjParams sp{static_cast<std::uint32_t>(cells_), s, 0.0f, 0.0f};
        std::memcpy(scale_ubo_.mapped, &sp, sizeof(sp));
        vmaFlushAllocation(ctx_->allocator, scale_ubo_.alloc, 0, VK_WHOLE_SIZE);
        OneShot shot;
        if (!shot.begin_compute(*ctx_)) {
            return;
        }
        barrier_compute_to_compute(shot.cb());  // see norm_and_peak
        scale_.bind(shot.cb(), scale_set_);
        vkCmdDispatch(shot.cb(), mul_groups_, 1, 1);
        shot.submit_and_wait(*ctx_);
        shot.destroy(*ctx_);
    }

    // Interleaved RG floats, 2 per cell.
    bool readback(std::vector<float>& out) {
        OneShot shot;
        if (!shot.begin_compute(*ctx_)) {
            return false;
        }
        barrier_compute_to_transfer(shot.cb());
        const VkBufferCopy down{0, 0, field_bytes_};
        vkCmdCopyBuffer(shot.cb(), psi_.buf, staging_.buf, 1, &down);
        barrier_transfer_to_host(shot.cb());
        const bool ok = shot.submit_and_wait(*ctx_);
        shot.destroy(*ctx_);
        if (!ok) {
            return false;
        }
        vmaInvalidateAllocation(ctx_->allocator, staging_.alloc, 0,
                                VK_WHOLE_SIZE);
        const float* p = static_cast<const float*>(staging_.mapped);
        out.assign(p, p + 2 * cells_);
        return true;
    }

    // True once a submit/fence has failed (VK_ERROR_DEVICE_LOST or a 10 s hang):
    // the GPU path is dead; the director should fall back to the CPU.
    bool device_lost() const { return ctx_->device_lost; }

    void destroy() {
        if (ctx_ == nullptr) {
            return;
        }
        if (ctx_->device != VK_NULL_HANDLE) {
            wait_async();
            vkDeviceWaitIdle(ctx_->device);
        }
        if (async_fence_ != VK_NULL_HANDLE) {
            vkDestroyFence(ctx_->device, async_fence_, nullptr);
            async_fence_ = VK_NULL_HANDLE;
        }
        if (async_pool_ != VK_NULL_HANDLE) {
            vkDestroyCommandPool(ctx_->device, async_pool_, nullptr);
            async_pool_ = VK_NULL_HANDLE;
            async_cb_ = VK_NULL_HANDLE;
        }
        if (profile_pool_ != VK_NULL_HANDLE) {
            vkDestroyQueryPool(ctx_->device, profile_pool_, nullptr);
            profile_pool_ = VK_NULL_HANDLE;
        }
        release_vkfft();
        for (State& st : states_) {
            ctx_->destroy_buffer(&st.buf);
        }
        states_.clear();
        free_full_states_.clear();
        arena_.destroy(*ctx_);
        unpack_.destroy(*ctx_);
        pack_.destroy(*ctx_);
        bridge_load_.destroy(*ctx_);
        bridge_store_.destroy(*ctx_);
        flow_vel_k_.destroy(*ctx_);
        project_.destroy(*ctx_);
        dipole_.destroy(*ctx_);
        force_.destroy(*ctx_);
        synth_.destroy(*ctx_);
        copy_.destroy(*ctx_);
        axpy_.destroy(*ctx_);
        inner_.destroy(*ctx_);
        shear_.destroy(*ctx_);
        kick_.destroy(*ctx_);
        scale_.destroy(*ctx_);
        scale_buf_.destroy(*ctx_);
        finalize_.destroy(*ctx_);
        norm_.destroy(*ctx_);
        fft_.destroy(*ctx_);
        conj_.destroy(*ctx_);
        mul_.destroy(*ctx_);
        half_mul_.destroy(*ctx_);
        kin_mul_.destroy(*ctx_);
        damp_.destroy(*ctx_);
        phase_damp_.destroy(*ctx_);
        mcwf_.destroy(*ctx_);
        release_mc();
        mc_density_.destroy(*ctx_);
        mc_classify_.destroy(*ctx_);
        mc_scan_.destroy(*ctx_);
        mc_emit_.destroy(*ctx_);
        ctx_->destroy_buffer(&mc_ubo_);
        ctx_->destroy_buffer(&mc_scan_ubo_);
        ctx_->destroy_buffer(&mcwf_radial_);
        ctx_->destroy_buffer(&mcwf_ubo_);
        ctx_->destroy_buffer(&relax_kin_);
        ctx_->destroy_buffer(&relax_half_);
        ctx_->destroy_buffer(&decode_scratch_[1]);
        ctx_->destroy_buffer(&decode_scratch_[0]);
        ctx_->destroy_buffer(&scratch_bridge_);
        ctx_->destroy_buffer(&bridge_ubo_);
        ctx_->destroy_image(&volume_[1]);
        ctx_->destroy_image(&volume_[0]);
        ctx_->destroy_buffer(&flow_vel_ubo_);
        ctx_->destroy_image(&flow_vel_[1]);
        ctx_->destroy_image(&flow_vel_[0]);
        ctx_->destroy_buffer(&proj_ubo_);
        ctx_->destroy_buffer(&glm_buf_);
        ctx_->destroy_buffer(&proj_binoff_buf_);
        ctx_->destroy_buffer(&proj_sorted_buf_);
        ctx_->destroy_buffer(&dipole_partials_);
        ctx_->destroy_buffer(&dipole_ubo_);
        ctx_->destroy_buffer(&force_partials_);
        ctx_->destroy_buffer(&force_v_buf_);
        ctx_->destroy_buffer(&force_ubo_);
        ctx_->destroy_buffer(&radial_buf_);
        ctx_->destroy_buffer(&synth_ubo_);
        ctx_->destroy_buffer(&axpy_ubo_);
        ctx_->destroy_buffer(&kick_ubo_);
        ctx_->destroy_buffer(&shear_ubo_[1]);
        ctx_->destroy_buffer(&shear_ubo_[0]);
        ctx_->destroy_buffer(&conjA_ubo_);
        ctx_->destroy_buffer(&scale_ubo_);
        ctx_->destroy_buffer(&renorm_);
        ctx_->destroy_buffer(&renorm_ubo_);
        for (int a = 0; a < 3; ++a) {
            ctx_->destroy_buffer(&fft_ubo_[a]);
        }
        ctx_->destroy_buffer(&conjN_ubo_);
        ctx_->destroy_buffer(&conj1_ubo_);
        ctx_->destroy_buffer(&muln_ubo_);
        ctx_->destroy_buffer(&staging_);
        ctx_->destroy_buffer(&partials_);
        ctx_->destroy_buffer(&v_buf_);
        ctx_->destroy_buffer(&kx2_buf_);
        ctx_->destroy_buffer(&ky2_buf_);
        ctx_->destroy_buffer(&kz2_buf_);
        ctx_->destroy_buffer(&halfp_ubo_);
        ctx_->destroy_buffer(&fullp_ubo_);
        ctx_->destroy_buffer(&kinp_ubo_);
        ctx_->destroy_buffer(&damp_buf_);
        ctx_->destroy_buffer(&psi_);
        reset_lazy_state();  // a second initialize() re-creates from scratch
        ctx_ = nullptr;
    }

private:
    static constexpr std::uint32_t kGroups = 256;

    // Null the lazily-created descriptor-set handles + the *_ok_ / cache-size
    // memos after destroy(): the arena that owned the sets is gone, so a
    // second initialize() on this object must re-allocate and re-wire them
    // (the `if (set != NULL) return` lazy guards would otherwise bind freed
    // sets). Eager sets are re-assigned by initialize() regardless; nulling
    // them too is harmless and keeps this exhaustive.
    void reset_lazy_state() {
        for (VkDescriptorSet* s :
             {&halfv_set_, &fullv_set_, &kin3_set_, &damp_set_, &pd_full_set_,
              &pd_half_set_, &mcwf_set_, &mc_den_set_, &mc_classify_set_,
              &mc_scan_set_, &mc_emit_set_, &conj1_set_, &conjN_set_,
              &norm_set_, &scale_set_, &finalize_set_, &scale_buf_set_,
              &conjA_set_, &kick_set_,
              &relax_half_set_, &relax_kin_set_, &force_set_, &dipole_set_,
              &proj_set_, &pack_set_, &unpack_set_, &inner_any_set_,
              &synth_any_set_, &norm_any_set_, &scale_any_set_, &load_set_[0],
              &load_set_[1], &store_set_[0], &store_set_[1],
              &flow_vel_set_[0], &flow_vel_set_[1], &fft_set_[0],
              &fft_set_[1], &fft_set_[2], &shear_set_[0], &shear_set_[1]}) {
            *s = VK_NULL_HANDLE;
        }
        pd_kernel_ok_ = false;
        fused_relax_ok_ = false;
        flow_vel_ok_ = false;
        mcwf_kernel_ok_ = false;
        mc_kernel_ok_ = false;
        mc_buffers_ok_ = false;
        mc_overflow_warned_ = false;
        async_pending_ = false;
        async_bridged_ = false;
        mcwf_radial_bytes_ = 0;
        radial_bytes_ = 0;
        staging_bytes_ = 0;
        mc_max_tris_ = 0;
        mc_nblocks_ = 0;
        mc_nblk_[0] = mc_nblk_[1] = mc_nblk_[2] = 0;
        vol_write_ = 0;
        vol_display_ = -1;
        volume_layout_[0] = VK_IMAGE_LAYOUT_UNDEFINED;
        volume_layout_[1] = VK_IMAGE_LAYOUT_UNDEFINED;
        flow_vel_layout_[0] = VK_IMAGE_LAYOUT_UNDEFINED;
        flow_vel_layout_[1] = VK_IMAGE_LAYOUT_UNDEFINED;
        flow_vel_m_ = 0;
    }

    struct alignas(16) MulParams {
        std::uint32_t n, p0, p1, p2;
    };
    struct alignas(16) ConjParams {
        std::uint32_t n;
        float scale;
        float p0, p1;
    };
    struct alignas(16) FftParams {
        std::int32_t mod_a, mul_b, mul_c, stride, n_lines, p0, p1, p2;
    };
    // std140 {uint n; vec2 c}: c aligns to offset 8 (uint + 4 bytes pad).
    struct alignas(16) AxpyParams {
        std::uint32_t n, pad;
        float cx, cy;
    };
    struct alignas(16) KickParams {
        std::uint32_t n;
        std::int32_t nx, ny;
        float theta;
        float box_min[4];
        float cell_h[4];
        float axis[4];
    };
    // std140: n@0, nx@4, ny@8, pad@12, box_min@16, cell_h@32 (dipole.comp).
    struct alignas(16) DipoleParams {
        std::uint32_t n;
        std::int32_t nx, ny, pad0;
        float box_min[4];
        float cell_h[4];
    };
    // std140: nx@0, ny@4, nr@8, h_radial@12, box_min@16, cell_h@32, dv@48.
    struct alignas(16) ProjectParams {
        std::int32_t nx, ny, nr;
        float h_radial;
        float box_min[4];
        float cell_h[4];
        float dv, pad0, pad1, pad2;
    };
    // std140: vec4-padded box_min/cell_h at 16/32, matches synth.comp order.
    struct alignas(16) SynthParams {
        std::uint32_t n;
        std::int32_t nx, ny, l;
        float box_min[4];
        float cell_h[4];
        std::int32_t m, n_radial;
        float h_radial, rmax;
    };
    struct alignas(16) BridgeParams {
        std::int32_t nx, ny, nz, pad;
    };
    struct alignas(16) FlowVelParams {  // flow_velocity.comp std140
        std::int32_t nx, ny, nz, m;
        float hx, hy, hz, vmax;
    };
    // std140 all-scalar block (tight 4-byte packing), matches shear.comp;
    // the trailing pad makes the 16-byte alignment explicit (dodges C4324).
    struct alignas(16) ShearParams {
        std::uint32_t n;
        std::int32_t nx, ny, nz;
        std::int32_t freq_axis, coord_axis, nf;
        float kscale, cmin, ch, coeff;
        float pad0;
    };

    // Emits the compute-to-compute hazard edge before every dispatch except
    // the first of a command buffer (prior submissions are fence-complete).
    struct Recorder {
        VkCommandBuffer cb;
        bool first;
        void dispatch(const Kernel& k, VkDescriptorSet set,
                      std::uint32_t groups) {
            if (!first) {
                barrier_compute_to_compute(cb);
            }
            first = false;
            k.bind(cb, set);
            vkCmdDispatch(cb, groups, 1, 1);
        }
        void dispatch_dyn(const Kernel& k, VkDescriptorSet set,
                          std::uint32_t groups, std::uint32_t offset) {
            if (!first) {
                barrier_compute_to_compute(cb);
            }
            first = false;
            k.bind(cb, set, offset);
            vkCmdDispatch(cb, groups, 1, 1);
        }
    };

    static std::vector<float> to_rg32f(
        const std::vector<std::complex<double>>& src) {
        std::vector<float> out(2 * src.size());
        for (std::size_t i = 0; i < src.size(); ++i) {
            out[2 * i] = static_cast<float>(src[i].real());
            out[2 * i + 1] = static_cast<float>(src[i].imag());
        }
        return out;
    }

    bool write_ubo(Buffer* ubo, const void* data, std::size_t size) {
        if (!ctx_->create_host_buffer(size, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                      ubo)) {
            return false;
        }
        std::memcpy(ubo->mapped, data, size);
        vmaFlushAllocation(ctx_->allocator, ubo->alloc, 0, VK_WHOLE_SIZE);
        return true;
    }

    VkDescriptorSet make_mul_set(VkBuffer table) {
        VkDescriptorSet set = arena_.allocate(*ctx_, mul_.set_layout());
        if (set == VK_NULL_HANDLE) {
            return set;
        }
        arena_.write_buffer(*ctx_, set, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                            psi_.buf);
        arena_.write_buffer(*ctx_, set, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                            table);
        arena_.write_buffer(*ctx_, set, 2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                            muln_ubo_.buf, sizeof(MulParams));
        return set;
    }

    VkDescriptorSet make_unary_set(const Kernel& k, const Buffer& ubo,
                                   std::size_t ubo_size) {
        VkDescriptorSet set = arena_.allocate(*ctx_, k.set_layout());
        if (set == VK_NULL_HANDLE) {
            return set;
        }
        arena_.write_buffer(*ctx_, set, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                            psi_.buf);
        arena_.write_buffer(*ctx_, set, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                            ubo.buf, ubo_size);
        return set;
    }

    bool upload_field(Buffer& dst,
                      const std::vector<std::complex<double>>& src) {
        const std::vector<float> f = to_rg32f(src);
        return upload_raw(dst, f.data(), f.size() * sizeof(float));
    }

    bool upload_raw(Buffer& dst, const void* data, VkDeviceSize bytes) {
        if (!ensure_staging(bytes)) {
            return false;
        }
        std::memcpy(staging_.mapped, data, bytes);
        vmaFlushAllocation(ctx_->allocator, staging_.alloc, 0, VK_WHOLE_SIZE);
        OneShot shot;
        if (!shot.begin_compute(*ctx_)) {
            return false;
        }
        const VkBufferCopy up{0, 0, bytes};
        vkCmdCopyBuffer(shot.cb(), staging_.buf, dst.buf, 1, &up);
        barrier_transfer_to_compute(shot.cb());
        const bool ok = shot.submit_and_wait(*ctx_);
        shot.destroy(*ctx_);
        return ok;
    }

    // A resident state: its buffer (fp32, or fp16-packed at half footprint)
    // + the four per-op descriptor sets (fp32 states only; fp16 consumers
    // decode to scratch and go through the shared any-target sets).
    struct State {
        bool is_half = false;
        Buffer buf{};
        VkDescriptorSet inner_set = VK_NULL_HANDLE;
        VkDescriptorSet axpy_set = VK_NULL_HANDLE;
        VkDescriptorSet copy_set = VK_NULL_HANDLE;
        VkDescriptorSet mul_set = VK_NULL_HANDLE;
    };

    State* state_at(int handle) {
        if (handle < 0 || handle >= static_cast<int>(states_.size())) {
            return nullptr;
        }
        State* st = &states_[static_cast<std::size_t>(handle)];
        return (st->buf.buf == VK_NULL_HANDLE) ? nullptr : st;
    }

    // Record: compute -> transfer edge, src -> staging copy, host edge.
    void record_buffer_readback(VkCommandBuffer cb, const Buffer& src,
                                VkDeviceSize bytes) {
        barrier_compute_to_transfer(cb);
        const VkBufferCopy down{0, 0, bytes};
        vkCmdCopyBuffer(cb, src.buf, staging_.buf, 1, &down);
        barrier_transfer_to_host(cb);
    }
    void record_partials_readback(VkCommandBuffer cb) {
        record_buffer_readback(cb, partials_, 2 * kGroups * sizeof(float));
    }

    // (Re)point a full state's four descriptor sets: only the st.buf bindings
    // vary; psi_/ubos/partials_ are stable but re-written idempotently so the
    // same call serves a fresh alloc and a free-list reuse.
    void wire_full_state_sets(State& st) {
        const auto storage = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        const auto uniform = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        arena_.write_buffer(*ctx_, st.inner_set, 0, storage, psi_.buf);
        arena_.write_buffer(*ctx_, st.inner_set, 1, uniform, muln_ubo_.buf,
                            sizeof(MulParams));
        arena_.write_buffer(*ctx_, st.inner_set, 2, storage, partials_.buf);
        arena_.write_buffer(*ctx_, st.inner_set, 3, storage, st.buf.buf);
        arena_.write_buffer(*ctx_, st.axpy_set, 0, storage, psi_.buf);
        arena_.write_buffer(*ctx_, st.axpy_set, 1, uniform, axpy_ubo_.buf,
                            sizeof(AxpyParams));
        arena_.write_buffer(*ctx_, st.axpy_set, 3, storage, st.buf.buf);
        arena_.write_buffer(*ctx_, st.copy_set, 0, storage, psi_.buf);
        arena_.write_buffer(*ctx_, st.copy_set, 1, uniform, muln_ubo_.buf,
                            sizeof(MulParams));
        arena_.write_buffer(*ctx_, st.copy_set, 3, storage, st.buf.buf);
        arena_.write_buffer(*ctx_, st.mul_set, 0, storage, psi_.buf);
        arena_.write_buffer(*ctx_, st.mul_set, 1, storage, st.buf.buf);
        arena_.write_buffer(*ctx_, st.mul_set, 2, uniform, muln_ubo_.buf,
                            sizeof(MulParams));
    }

    // Allocate a full state's buffer + descriptor sets without uploading
    // content. A released full-state slot (free_full_states_) is REUSED --
    // its four sets are re-pointed at the new buffer -- so a synth/release
    // churn (Key-R reseed, partial measurement) does not leak sets or slots.
    int create_state_buffer_uninit() {
        if (!free_full_states_.empty()) {
            const int idx = free_full_states_.back();
            free_full_states_.pop_back();
            State& st = states_[static_cast<std::size_t>(idx)];
            if (!ctx_->create_device_buffer(field_bytes_, &st.buf)) {
                free_full_states_.push_back(idx);  // still free; try again later
                return -1;
            }
            st.is_half = false;
            wire_full_state_sets(st);
            return idx;
        }
        State st;
        if (!ctx_->create_device_buffer(field_bytes_, &st.buf)) {
            return -1;
        }
        st.inner_set = arena_.allocate(*ctx_, inner_.set_layout());
        st.axpy_set = arena_.allocate(*ctx_, axpy_.set_layout());
        st.copy_set = arena_.allocate(*ctx_, copy_.set_layout());
        st.mul_set = arena_.allocate(*ctx_, mul_.set_layout());
        if (st.inner_set == VK_NULL_HANDLE || st.axpy_set == VK_NULL_HANDLE ||
            st.copy_set == VK_NULL_HANDLE || st.mul_set == VK_NULL_HANDLE) {
            ctx_->destroy_buffer(&st.buf);
            return -1;
        }
        wire_full_state_sets(st);
        states_.push_back(st);
        return static_cast<int>(states_.size()) - 1;
    }

    // Synthesize (u/r)Ylm into `out` via the shared any-target sets (re-
    // pointed per call; every submission fence-waits so never in flight),
    // then grid-normalize it. Reports pre-norm grid norm + normalized peak.
    bool synthesize_into_buffer(Buffer& out, const std::vector<double>& u,
                                int l, int m, double h_radial, double rmax,
                                int n_radial, double* out_peak = nullptr,
                                double* out_norm2 = nullptr) {
        std::vector<float> uf(u.size());
        for (std::size_t i = 0; i < u.size(); ++i) {
            uf[i] = static_cast<float>(u[i]);
        }
        const VkDeviceSize rbytes = uf.size() * sizeof(float);
        if (radial_buf_.buf == VK_NULL_HANDLE || radial_bytes_ != rbytes) {
            vkDeviceWaitIdle(ctx_->device);
            ctx_->destroy_buffer(&radial_buf_);
            if (!ctx_->create_device_buffer(rbytes, &radial_buf_)) {
                return false;
            }
            radial_bytes_ = rbytes;
        }
        if (!upload_raw(radial_buf_, uf.data(), rbytes)) {
            return false;
        }

        SynthParams sp{};
        sp.n = static_cast<std::uint32_t>(cells_);
        sp.nx = grid_.x.n;
        sp.ny = grid_.y.n;
        sp.l = l;
        sp.m = m;
        sp.n_radial = n_radial;
        sp.h_radial = static_cast<float>(h_radial);
        sp.rmax = static_cast<float>(rmax);
        sp.box_min[0] = static_cast<float>(grid_.x.xmin);
        sp.box_min[1] = static_cast<float>(grid_.y.xmin);
        sp.box_min[2] = static_cast<float>(grid_.z.xmin);
        sp.cell_h[0] = static_cast<float>(grid_.x.spacing());
        sp.cell_h[1] = static_cast<float>(grid_.y.spacing());
        sp.cell_h[2] = static_cast<float>(grid_.z.spacing());
        std::memcpy(synth_ubo_.mapped, &sp, sizeof(sp));
        vmaFlushAllocation(ctx_->allocator, synth_ubo_.alloc, 0, VK_WHOLE_SIZE);

        const auto storage = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        arena_.write_buffer(*ctx_, synth_any_set_, 0, storage, out.buf);
        arena_.write_buffer(*ctx_, synth_any_set_, 1,
                            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, synth_ubo_.buf,
                            sizeof(SynthParams));
        arena_.write_buffer(*ctx_, synth_any_set_, 5, storage, radial_buf_.buf);

        OneShot shot;
        if (!shot.begin_compute(*ctx_)) {
            return false;
        }
        synth_.bind(shot.cb(), synth_any_set_);
        vkCmdDispatch(shot.cb(), mul_groups_, 1, 1);
        shot.submit_and_wait(*ctx_);
        shot.destroy(*ctx_);

        const NormPeak np = normalize_buffer(out);
        if (out_norm2 != nullptr) {
            *out_norm2 = np.sum;
        }
        if (out_peak != nullptr) {
            *out_peak = (np.sum > 0.0) ? np.peak / np.sum : 0.0;
        }
        return true;
    }

    // Grid normalization of `buf` via the re-pointed any-target norm/scale
    // sets. Returns {pre-normalization grid norm (x dV), raw peak density}.
    NormPeak normalize_buffer(Buffer& buf) {
        NormPeak np;
        const auto storage = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        arena_.write_buffer(*ctx_, norm_any_set_, 0, storage, buf.buf);
        arena_.write_buffer(*ctx_, norm_any_set_, 1,
                            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, muln_ubo_.buf,
                            sizeof(MulParams));
        arena_.write_buffer(*ctx_, norm_any_set_, 2, storage, partials_.buf);
        OneShot shot;
        if (!shot.begin_compute(*ctx_)) {
            return np;
        }
        norm_.bind(shot.cb(), norm_any_set_);
        vkCmdDispatch(shot.cb(), kGroups, 1, 1);
        record_partials_readback(shot.cb());
        const bool ok = shot.submit_and_wait(*ctx_);
        shot.destroy(*ctx_);
        if (!ok) {
            return np;
        }
        vmaInvalidateAllocation(ctx_->allocator, staging_.alloc, 0,
                                VK_WHOLE_SIZE);
        const float* p = static_cast<const float*>(staging_.mapped);
        for (std::uint32_t g = 0; g < kGroups; ++g) {
            np.sum += p[2 * g];
            np.peak = std::max(np.peak, static_cast<double>(p[2 * g + 1]));
        }
        np.sum *= cell_volume_;

        const ConjParams sp{static_cast<std::uint32_t>(cells_),
                            static_cast<float>((np.sum > 0.0)
                                                   ? 1.0 / std::sqrt(np.sum)
                                                   : 0.0),
                            0.0f, 0.0f};
        std::memcpy(scale_ubo_.mapped, &sp, sizeof(sp));
        vmaFlushAllocation(ctx_->allocator, scale_ubo_.alloc, 0, VK_WHOLE_SIZE);
        arena_.write_buffer(*ctx_, scale_any_set_, 0, storage, buf.buf);
        arena_.write_buffer(*ctx_, scale_any_set_, 1,
                            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, scale_ubo_.buf,
                            sizeof(ConjParams));
        OneShot s2;
        if (!s2.begin_compute(*ctx_)) {
            return np;
        }
        scale_.bind(s2.cb(), scale_any_set_);
        vkCmdDispatch(s2.cb(), mul_groups_, 1, 1);
        s2.submit_and_wait(*ctx_);
        s2.destroy(*ctx_);
        return np;
    }

    // Lazily create the RG16F volume + the store side of the bridge: one
    // submission transitions UNDEFINED -> GENERAL for the compute stores
    // (write_psi_to_volume owns the GENERAL <-> SHADER_READ_ONLY round-trip
    // from then on). fp16 is DISPLAY-ONLY: physics stays in the fp32 psi
    // SSBO; the texture units convert fp16 texels to fp32 on sample, so
    // half the raymarch/occupancy/shadow traffic costs no shader math.
    bool ensure_volume() {
        if (store_set_[0] != VK_NULL_HANDLE) {
            return true;
        }
        const BridgeParams bp{grid_.x.n, grid_.y.n, grid_.z.n, 0};
        if (!write_ubo(&bridge_ubo_, &bp, sizeof(bp))) {
            return false;
        }
        for (int i = 0; i < 2; ++i) {
            if (!ctx_->create_storage_image_3d(
                    static_cast<std::uint32_t>(n_),
                    static_cast<std::uint32_t>(n_),
                    static_cast<std::uint32_t>(n_), VK_FORMAT_R16G16_SFLOAT,
                    &volume_[i], /*share_across_queues=*/true)) {
                return false;
            }
            store_set_[i] = arena_.allocate(*ctx_, bridge_store_.set_layout());
            if (store_set_[i] == VK_NULL_HANDLE) {
                store_set_[0] = VK_NULL_HANDLE;
                return false;
            }
            arena_.write_buffer(*ctx_, store_set_[i], 0,
                                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, psi_.buf);
            arena_.write_image(*ctx_, store_set_[i], 1, volume_[i].view);
            arena_.write_buffer(*ctx_, store_set_[i], 2,
                                VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                bridge_ubo_.buf, sizeof(BridgeParams));
        }
        // Companion low-res velocity volume (double-buffered with the display).
        if (flow_vel_ok_) {
            flow_vel_m_ = std::min(128, n_);
            const FlowVelParams fvp{grid_.x.n,
                                    grid_.y.n,
                                    grid_.z.n,
                                    flow_vel_m_,
                                    static_cast<float>(grid_.x.spacing()),
                                    static_cast<float>(grid_.y.spacing()),
                                    static_cast<float>(grid_.z.spacing()),
                                    3.0f};
            if (!write_ubo(&flow_vel_ubo_, &fvp, sizeof(fvp))) {
                return false;
            }
            for (int i = 0; i < 2; ++i) {
                if (!ctx_->create_storage_image_3d(
                        static_cast<std::uint32_t>(flow_vel_m_),
                        static_cast<std::uint32_t>(flow_vel_m_),
                        static_cast<std::uint32_t>(flow_vel_m_),
                        VK_FORMAT_R16G16B16A16_SFLOAT, &flow_vel_[i],
                        /*share_across_queues=*/true)) {
                    return false;
                }
                flow_vel_set_[i] =
                    arena_.allocate(*ctx_, flow_vel_k_.set_layout());
                if (flow_vel_set_[i] == VK_NULL_HANDLE) {
                    return false;
                }
                arena_.write_buffer(*ctx_, flow_vel_set_[i], 0,
                                    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, psi_.buf);
                arena_.write_image(*ctx_, flow_vel_set_[i], 1,
                                   flow_vel_[i].view);
                arena_.write_buffer(*ctx_, flow_vel_set_[i], 2,
                                    VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                    flow_vel_ubo_.buf, sizeof(FlowVelParams));
            }
        }
        OneShot shot;
        if (!shot.begin_compute(*ctx_)) {
            return false;
        }
        for (int i = 0; i < 2; ++i) {
            image_layout_barrier(shot.cb(), volume_[i].img,
                                 VK_IMAGE_LAYOUT_UNDEFINED,
                                 VK_IMAGE_LAYOUT_GENERAL,
                                 VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_ACCESS_SHADER_READ_BIT |
                                     VK_ACCESS_SHADER_WRITE_BIT);
            if (flow_vel_ok_) {
                image_layout_barrier(
                    shot.cb(), flow_vel_[i].img, VK_IMAGE_LAYOUT_UNDEFINED,
                    VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                    0, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
                flow_vel_layout_[i] = VK_IMAGE_LAYOUT_GENERAL;
            }
        }
        shot.submit_and_wait(*ctx_);
        shot.destroy(*ctx_);
        volume_layout_[0] = VK_IMAGE_LAYOUT_GENERAL;
        volume_layout_[1] = VK_IMAGE_LAYOUT_GENERAL;
        return true;
    }

    // Transition volume image `idx` if not already in `to`. COMPUTE stages
    // only: the engine may run on a compute-only queue, and the graphics
    // queue's fragment reads are ordered by the host fence that observes
    // the write before the render is submitted (ping-pong guarantees an
    // in-flight batch writes the OTHER image).
    void transition_volume(VkCommandBuffer cb, int idx, VkImageLayout to) {
        if (volume_layout_[idx] == to) {
            return;
        }
        const VkPipelineStageFlags stages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        const VkAccessFlags access =
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        image_layout_barrier(cb, volume_[idx].img, volume_layout_[idx], to,
                             stages, access, stages, access);
        volume_layout_[idx] = to;
    }
    void transition_flow_vel(VkCommandBuffer cb, int idx, VkImageLayout to) {
        if (flow_vel_layout_[idx] == to) {
            return;
        }
        const VkPipelineStageFlags stages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        const VkAccessFlags access =
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        image_layout_barrier(cb, flow_vel_[idx].img, flow_vel_layout_[idx], to,
                             stages, access, stages, access);
        flow_vel_layout_[idx] = to;
    }
    // Record the fp32 velocity field into flow_vel_[vol_write_] alongside the
    // display bridge (same psi snapshot). GENERAL for the store, then
    // SHADER_READ for the renderer's flow sampler.
    void record_flow_velocity(VkCommandBuffer cb) {
        if (!flow_vel_ok_ || flow_vel_set_[vol_write_] == VK_NULL_HANDLE) {
            return;
        }
        transition_flow_vel(cb, vol_write_, VK_IMAGE_LAYOUT_GENERAL);
        flow_vel_k_.bind(cb, flow_vel_set_[vol_write_]);
        const std::uint32_t g =
            (static_cast<std::uint32_t>(flow_vel_m_) + 3u) / 4u;
        vkCmdDispatch(cb, g, g, g);
        transition_flow_vel(cb, vol_write_,
                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }

    // Lazily create the async batch's pool/cb/fence (compute family).
    bool ensure_async() {
        if (async_cb_ != VK_NULL_HANDLE) {
            return true;
        }
        VkCommandPoolCreateInfo cpci{};
        cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cpci.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        cpci.queueFamilyIndex = ctx_->compute_family;
        if (vkCreateCommandPool(ctx_->device, &cpci, nullptr, &async_pool_) !=
            VK_SUCCESS) {
            return false;
        }
        VkCommandBufferAllocateInfo cbai{};
        cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbai.commandPool = async_pool_;
        cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1;
        if (vkAllocateCommandBuffers(ctx_->device, &cbai, &async_cb_) !=
            VK_SUCCESS) {
            return false;
        }
        VkFenceCreateInfo fci{};
        fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        return vkCreateFence(ctx_->device, &fci, nullptr, &async_fence_) ==
               VK_SUCCESS;
    }

    // Called at a HOST-OBSERVED completion point of a bridge write: the
    // written image becomes the display image, the other becomes writable.
    void flip_volume() {
        vol_display_ = vol_write_;
        vol_write_ = 1 - vol_write_;
    }

    // Lazily build the fp16 codec resources: two decode scratch buffers (a
    // 2-fp16-operand dipole needs both) + the shared re-pointed sets.
    bool ensure_fp16() {
        if (pack_set_ != VK_NULL_HANDLE) {
            return true;
        }
        if (!ctx_->create_device_buffer(field_bytes_, &decode_scratch_[0]) ||
            !ctx_->create_device_buffer(field_bytes_, &decode_scratch_[1])) {
            return false;
        }
        pack_set_ = arena_.allocate(*ctx_, pack_.set_layout());
        unpack_set_ = arena_.allocate(*ctx_, unpack_.set_layout());
        inner_any_set_ = arena_.allocate(*ctx_, inner_.set_layout());
        if (pack_set_ == VK_NULL_HANDLE || unpack_set_ == VK_NULL_HANDLE ||
            inner_any_set_ == VK_NULL_HANDLE) {
            pack_set_ = VK_NULL_HANDLE;
            return false;
        }
        const auto uniform = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        arena_.write_buffer(*ctx_, pack_set_, 1, uniform, muln_ubo_.buf,
                            sizeof(MulParams));
        arena_.write_buffer(*ctx_, unpack_set_, 1, uniform, muln_ubo_.buf,
                            sizeof(MulParams));
        return true;
    }

    // A readable fp32 buffer for `st`: its own buffer if fp32, else the fp16
    // content unpacked into decode_scratch_[slot] (0 or 1).
    VkBuffer decode(State* st, int slot) {
        if (!st->is_half) {
            return st->buf.buf;
        }
        if (!ensure_fp16()) {
            return VK_NULL_HANDLE;
        }
        const auto storage = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        arena_.write_buffer(*ctx_, unpack_set_, 0, storage,
                            decode_scratch_[slot].buf);
        arena_.write_buffer(*ctx_, unpack_set_, 6, storage, st->buf.buf);
        OneShot shot;
        if (!shot.begin_compute(*ctx_)) {
            return VK_NULL_HANDLE;
        }
        barrier_compute_to_compute(shot.cb());
        unpack_.bind(shot.cb(), unpack_set_);
        vkCmdDispatch(shot.cb(), mul_groups_, 1, 1);
        shot.submit_and_wait(*ctx_);
        shot.destroy(*ctx_);
        return decode_scratch_[slot].buf;
    }

    // Lazily build the dipole reduction resources (shared re-pointed set).
    bool ensure_dipole() {
        if (dipole_set_ != VK_NULL_HANDLE) {
            return true;
        }
        const DipoleParams dp0{};
        if (!ctx_->create_device_buffer(6 * kGroups * sizeof(float),
                                        &dipole_partials_) ||
            !write_ubo(&dipole_ubo_, &dp0, sizeof(dp0))) {
            return false;
        }
        dipole_set_ = arena_.allocate(*ctx_, dipole_.set_layout());
        if (dipole_set_ == VK_NULL_HANDLE) {
            return false;
        }
        arena_.write_buffer(*ctx_, dipole_set_, 1,
                            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, dipole_ubo_.buf,
                            sizeof(DipoleParams));
        arena_.write_buffer(*ctx_, dipole_set_, 2,
                            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                            dipole_partials_.buf);
        return true;
    }

    // Grow-only staging capacity (large glm/radial transfers).
    bool ensure_staging(VkDeviceSize bytes) {
        if (bytes <= staging_bytes_) {
            return true;
        }
        vkDeviceWaitIdle(ctx_->device);
        ctx_->destroy_buffer(&staging_);
        if (!ctx_->create_host_buffer(bytes,
                                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                          VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                      &staging_)) {
            return false;
        }
        staging_bytes_ = bytes;
        return true;
    }

    // psi -= (cre + i cim) * state (Gram-Schmidt subtract). fp32 states only.
    void subtract_projection(int handle, double cre, double cim) {
        State* st = state_at(handle);
        if (st == nullptr || st->is_half) {
            return;
        }
        const AxpyParams ap{static_cast<std::uint32_t>(cells_), 0,
                            static_cast<float>(cre), static_cast<float>(cim)};
        std::memcpy(axpy_ubo_.mapped, &ap, sizeof(ap));
        vmaFlushAllocation(ctx_->allocator, axpy_ubo_.alloc, 0, VK_WHOLE_SIZE);
        OneShot shot;
        if (!shot.begin_compute(*ctx_)) {
            return;
        }
        axpy_.bind(shot.cb(), st->axpy_set);
        vkCmdDispatch(shot.cb(), mul_groups_, 1, 1);
        shot.submit_and_wait(*ctx_);
        shot.destroy(*ctx_);
    }

    // Norm reduction + readback -> host finish -> 1/sqrt(norm) scale.
    RelaxStats renormalize_and_estimate() {
        OneShot shot;
        if (!shot.begin_compute(*ctx_)) {
            return {};
        }
        norm_.bind(shot.cb(), norm_set_);
        vkCmdDispatch(shot.cb(), kGroups, 1, 1);
        record_partials_readback(shot.cb());
        const bool ok = shot.submit_and_wait(*ctx_);
        shot.destroy(*ctx_);
        if (!ok) {
            return {};
        }
        return finish_renorm_from_staging();
    }

    // Host finish over partials ALREADY copied into staging_ (relax_step
    // records the reduction inside the step submission), then the
    // 1/sqrt(norm) scale.
    RelaxStats finish_renorm_from_staging() {
        RelaxStats stats;
        vmaInvalidateAllocation(ctx_->allocator, staging_.alloc, 0,
                                VK_WHOLE_SIZE);
        const float* p = static_cast<const float*>(staging_.mapped);
        double sum = 0.0;
        double peak = 0.0;
        for (std::uint32_t g = 0; g < kGroups; ++g) {
            sum += p[2 * g];
            peak = std::max(peak, static_cast<double>(p[2 * g + 1]));
        }
        const double norm_sq = sum * cell_volume_;
        const double inv = (norm_sq > 0.0) ? 1.0 / std::sqrt(norm_sq) : 0.0;
        stats.energy = (norm_sq > 0.0 && dtau_ > 0.0)
                           ? -std::log(norm_sq) / (2.0 * dtau_)
                           : 0.0;
        stats.peak = (norm_sq > 0.0) ? peak / norm_sq : 0.0;
        scale(static_cast<float>(inv));
        return stats;
    }

    // Fused-path stat finish: the GPU already computed inv and rescaled
    // psi, so this only distills the double-precision energy/peak diagnostic
    // from the (pre-scale) partials in staging -- no host scale submit.
    RelaxStats finish_renorm_stats_only() {
        RelaxStats stats;
        vmaInvalidateAllocation(ctx_->allocator, staging_.alloc, 0,
                                VK_WHOLE_SIZE);
        const float* p = static_cast<const float*>(staging_.mapped);
        double sum = 0.0;
        double peak = 0.0;
        for (std::uint32_t g = 0; g < kGroups; ++g) {
            sum += p[2 * g];
            peak = std::max(peak, static_cast<double>(p[2 * g + 1]));
        }
        const double norm_sq = sum * cell_volume_;
        stats.energy = (norm_sq > 0.0 && dtau_ > 0.0)
                           ? -std::log(norm_sq) / (2.0 * dtau_)
                           : 0.0;
        stats.peak = (norm_sq > 0.0) ? peak / norm_sq : 0.0;
        return stats;
    }

    KickParams make_kick_params(const ses::Vec3d& axis, double theta) const {
        KickParams kp{};
        kp.n = static_cast<std::uint32_t>(cells_);
        kp.nx = grid_.x.n;
        kp.ny = grid_.y.n;
        kp.theta = static_cast<float>(theta);
        kp.box_min[0] = static_cast<float>(grid_.x.xmin);
        kp.box_min[1] = static_cast<float>(grid_.y.xmin);
        kp.box_min[2] = static_cast<float>(grid_.z.xmin);
        kp.cell_h[0] = static_cast<float>(grid_.x.spacing());
        kp.cell_h[1] = static_cast<float>(grid_.y.spacing());
        kp.cell_h[2] = static_cast<float>(grid_.z.spacing());
        kp.axis[0] = static_cast<float>(axis.x);
        kp.axis[1] = static_cast<float>(axis.y);
        kp.axis[2] = static_cast<float>(axis.z);
        return kp;
    }

    // Grow the dynamic-offset kick UBO to `kicks` slots and re-point the
    // descriptor set at the new buffer (rewriting a fence-idle set is legal).
    bool ensure_kick_capacity(int kicks) {
        const VkDeviceSize need =
            static_cast<VkDeviceSize>(kick_stride_) * kicks;
        if (kick_ubo_.buf != VK_NULL_HANDLE && kick_slots_ >= kicks) {
            return true;
        }
        ctx_->destroy_buffer(&kick_ubo_);
        if (!ctx_->create_host_buffer(
                need, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, &kick_ubo_)) {
            return false;
        }
        kick_slots_ = kicks;
        arena_.write_buffer(*ctx_, kick_set_, 1,
                            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                            kick_ubo_.buf, sizeof(KickParams));
        return true;
    }

    const ses::Grid1D& axis_grid(int a) const {
        return a == 0 ? grid_.x : (a == 1 ? grid_.y : grid_.z);
    }

    // Shearing lines along freq_axis (frequency space along freq_axis),
    // shift each line by coeff * (its coord_axis coordinate).
    ShearParams make_shear_params(int freq_axis, int coord_axis,
                                  double coeff) const {
        const ses::Grid1D& fa = axis_grid(freq_axis);
        const ses::Grid1D& ca = axis_grid(coord_axis);
        const double two_pi = 6.283185307179586;
        ShearParams sp{};
        sp.n = static_cast<std::uint32_t>(cells_);
        sp.nx = grid_.x.n;
        sp.ny = grid_.y.n;
        sp.nz = grid_.z.n;
        sp.freq_axis = freq_axis;
        sp.coord_axis = coord_axis;
        sp.nf = fa.n;
        sp.kscale = static_cast<float>(two_pi / (fa.xmax - fa.xmin));
        sp.cmin = static_cast<float>(ca.xmin);
        sp.ch = static_cast<float>(ca.spacing());
        sp.coeff = static_cast<float>(coeff);
        return sp;
    }

    // Write the two shear parameter sets one (or many) three-shear
    // rotation(s) about the axis perpendicular to (b, c) need: set0 =
    // (b, c, -tan(theta/2)) used twice, set1 = (c, b, sin(theta)).
    void stage_rotation_ubos(int b, int c, double theta) {
        const double t = std::tan(0.5 * theta);
        const double sn = std::sin(theta);
        const ShearParams s0 = make_shear_params(b, c, -t);
        const ShearParams s1 = make_shear_params(c, b, sn);
        std::memcpy(shear_ubo_[0].mapped, &s0, sizeof(s0));
        std::memcpy(shear_ubo_[1].mapped, &s1, sizeof(s1));
        vmaFlushAllocation(ctx_->allocator, shear_ubo_[0].alloc, 0,
                           VK_WHOLE_SIZE);
        vmaFlushAllocation(ctx_->allocator, shear_ubo_[1].alloc, 0,
                           VK_WHOLE_SIZE);
    }

    // One staged shear: FFT along freq_axis, phase-shift (shear_set_[which]),
    // then the inverse FFT along that axis (conj -> FFT -> conj/n).
    void record_shear(Recorder& r, int which, int freq_axis) {
        const std::uint32_t nn = static_cast<std::uint32_t>(n_) *
                                 static_cast<std::uint32_t>(n_);
        r.dispatch(fft_, fft_set_[freq_axis], nn);
        r.dispatch(shear_, shear_set_[which], mul_groups_);
        r.dispatch(conj_, conj1_set_, mul_groups_);
        r.dispatch(fft_, fft_set_[freq_axis], nn);
        r.dispatch(conj_, conjA_set_, mul_groups_);
    }

    // One full three-shear rotation (set0 on b, set1 on c, set0 on b).
    void record_rotation(Recorder& r, int b, int c) {
        record_shear(r, 0, b);
        record_shear(r, 1, c);
        record_shear(r, 0, b);
    }

    // FFT . kin . IFFT -- the kinetic body of one Strang step; the potential
    // kicks around it are recorded by the caller (step()'s fused batching).
    void record_kin_body(Recorder& r) {
#ifdef SES_HAVE_VKFFT
        if (vkfft_active()) {
            record_vkfft(r, -1);  // forward
            r.dispatch(kin_mul_, kin3_set_, mul_groups_);
            record_vkfft(r, 1);   // inverse, normalized 1/N by the plan
            return;
        }
#endif
        fft3(r);
        r.dispatch(kin_mul_, kin3_set_, mul_groups_);
        r.dispatch(conj_, conj1_set_, mul_groups_);
        fft3(r);
        r.dispatch(conj_, conjN_set_, mul_groups_);
    }

    // halfV . IFFT . kin . FFT . halfV. With VkFFT active the two whole-3D
    // transforms are single VkFFTAppend blocks (coalesced transposes; the
    // inverse carries the 1/N normalize), replacing six line-FFT dispatches
    // and both conj-scale dispatches. Else the hand-rolled chain (the
    // inverse FFT = conj . FFT . conj/N), barriers explicit either way.
    void run_step_body(Recorder& r, const Kernel& half_k,
                       VkDescriptorSet half_set, const Kernel& kin_k,
                       VkDescriptorSet kin_set) {
#ifdef SES_HAVE_VKFFT
        if (vkfft_active()) {
            r.dispatch(half_k, half_set, mul_groups_);
            record_vkfft(r, -1);  // forward
            r.dispatch(kin_k, kin_set, mul_groups_);
            record_vkfft(r, 1);   // inverse, normalized 1/N by the plan
            r.dispatch(half_k, half_set, mul_groups_);
            return;
        }
#endif
        r.dispatch(half_k, half_set, mul_groups_);
        fft3(r);
        r.dispatch(kin_k, kin_set, mul_groups_);
        r.dispatch(conj_, conj1_set_, mul_groups_);
        fft3(r);
        r.dispatch(conj_, conjN_set_, mul_groups_);
        r.dispatch(half_k, half_set, mul_groups_);
    }

    // Upload helpers for the in-shader phase inputs.
    bool upload_potential(const std::vector<double>& v) {
        std::vector<float> vf(cells_);
        for (std::size_t i = 0; i < cells_; ++i) {
            vf[i] = static_cast<float>(v[i]);
        }
        return upload_raw(v_buf_, vf.data(), cells_ * sizeof(float));
    }
    bool upload_k2_tables(const ses::Grid3D& grid) {
        const std::vector<double> kx = ses::wavenumbers(grid.x);
        const std::vector<double> ky = ses::wavenumbers(grid.y);
        const std::vector<double> kz = ses::wavenumbers(grid.z);
        std::vector<float> t(static_cast<std::size_t>(n_));
        for (int i = 0; i < n_; ++i) {
            t[static_cast<std::size_t>(i)] = static_cast<float>(kx[i] * kx[i]);
        }
        if (!upload_raw(kx2_buf_, t.data(), t.size() * sizeof(float))) {
            return false;
        }
        for (int i = 0; i < n_; ++i) {
            t[static_cast<std::size_t>(i)] = static_cast<float>(ky[i] * ky[i]);
        }
        if (!upload_raw(ky2_buf_, t.data(), t.size() * sizeof(float))) {
            return false;
        }
        for (int i = 0; i < n_; ++i) {
            t[static_cast<std::size_t>(i)] = static_cast<float>(kz[i] * kz[i]);
        }
        return upload_raw(kz2_buf_, t.data(), t.size() * sizeof(float));
    }

#ifdef SES_HAVE_VKFFT
    // Plan a 3D C2C fp32 transform IN PLACE on psi_'s VkBuffer. The
    // configuration stores POINTERS; the pointees are members (or live in
    // the outliving DeviceContext) so VkFFTAppend can dereference per call.
    bool ensure_vkfft() {
        if (vkfft_ready_ || vkfft_failed_) {
            return vkfft_ready_;
        }
        vkfft_failed_ = true;
        VkCommandPoolCreateInfo pool_info{};
        pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pool_info.queueFamilyIndex = ctx_->compute_family;
        VkFenceCreateInfo fence_info{};
        fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        if (vkCreateCommandPool(ctx_->device, &pool_info, nullptr,
                                &vkfft_pool_) != VK_SUCCESS ||
            vkCreateFence(ctx_->device, &fence_info, nullptr, &vkfft_fence_) !=
                VK_SUCCESS) {
            release_vkfft();
            return false;
        }
        vkfft_psi_buf_ = psi_.buf;
        vkfft_buf_size_ = static_cast<std::uint64_t>(field_bytes_);
        VkFFTConfiguration conf{};
        conf.FFTdim = 3;
        conf.size[0] = static_cast<std::uint64_t>(grid_.x.n);
        conf.size[1] = static_cast<std::uint64_t>(grid_.y.n);
        conf.size[2] = static_cast<std::uint64_t>(grid_.z.n);
        conf.physicalDevice = &ctx_->phys_dev;
        conf.device = &ctx_->device;
        conf.queue = &ctx_->compute_queue;  // the engine's queue (see ctx)
        conf.commandPool = &vkfft_pool_;
        conf.fence = &vkfft_fence_;
        conf.buffer = &vkfft_psi_buf_;
        conf.bufferSize = &vkfft_buf_size_;
        conf.normalize = 1;  // inverse divides by N (replaces the conj/N pass)
        const VkFFTResult res = initializeVkFFT(&vkfft_app_, conf);
        if (res != VKFFT_SUCCESS) {
            std::fprintf(stderr,
                         "ses_vk::Engine: initializeVkFFT = %d -- staying on "
                         "the hand-rolled FFT\n",
                         static_cast<int>(res));
            release_vkfft();
            return false;
        }
        std::fprintf(stderr, "ses_vk::Engine: VkFFT 3D plan active (%dx%dx%d)\n",
                     grid_.x.n, grid_.y.n, grid_.z.n);
        vkfft_ready_ = true;
        vkfft_failed_ = false;
        return true;
    }

    void release_vkfft() {
        if (vkfft_ready_) {
            deleteVkFFT(&vkfft_app_);
            vkfft_ready_ = false;
        }
        if (vkfft_fence_ != VK_NULL_HANDLE) {
            vkDestroyFence(ctx_->device, vkfft_fence_, nullptr);
            vkfft_fence_ = VK_NULL_HANDLE;
        }
        if (vkfft_pool_ != VK_NULL_HANDLE) {
            vkDestroyCommandPool(ctx_->device, vkfft_pool_, nullptr);
            vkfft_pool_ = VK_NULL_HANDLE;
        }
    }

    // One whole-3D transform straight into the current primary command
    // buffer, hazard-fenced on both sides (VkFFT records plain dispatches).
    // direction: -1 forward, 1 inverse (1/N). After the block the recorder
    // is told a barrier was just emitted, so the next dispatch skips its own.
    void record_vkfft(Recorder& r, int direction) {
        if (!r.first) {
            barrier_compute_to_compute(r.cb);
        }
        VkCommandBuffer cb = r.cb;
        VkFFTLaunchParams lp{};
        lp.buffer = &vkfft_psi_buf_;
        lp.commandBuffer = &cb;
        const VkFFTResult res = VkFFTAppend(&vkfft_app_, direction, &lp);
        if (res != VKFFT_SUCCESS) {
            std::fprintf(stderr, "ses_vk::Engine: VkFFTAppend = %d\n",
                         static_cast<int>(res));
        }
        barrier_compute_to_compute(r.cb);
        r.first = true;  // the trailing barrier covers the next dispatch
    }
#else
    bool ensure_vkfft() { return false; }
    void release_vkfft() {}
#endif  // SES_HAVE_VKFFT

    void fft3(Recorder& r) {
        const std::uint32_t nn = static_cast<std::uint32_t>(n_) *
                                 static_cast<std::uint32_t>(n_);
        for (int a = 0; a < 3; ++a) {
            r.dispatch(fft_, fft_set_[a], nn);
        }
    }

    // Lazy timestamp query pool for profile_step() (4 stages -> 5 stamps).
    // Created only on first profile, so the normal step path never touches it.
    // Fails closed when the compute queue lacks HW timestamps.
    static constexpr std::uint32_t kProfileStamps = 5;
    bool ensure_profile_pool() {
        if (profile_pool_ != VK_NULL_HANDLE) {
            return true;
        }
        if (ctx_->timestamp_valid_bits == 0) {
            return false;
        }
        VkQueryPoolCreateInfo qpci{};
        qpci.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        qpci.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qpci.queryCount = kProfileStamps;
        return vkCreateQueryPool(ctx_->device, &qpci, nullptr, &profile_pool_) ==
               VK_SUCCESS;
    }
    VkQueryPool profile_pool_ = VK_NULL_HANDLE;

    DeviceContext* ctx_ = nullptr;
    ses::Grid3D grid_{};
    int n_ = 0;
    std::size_t cells_ = 0;
    std::uint32_t mul_groups_ = 0;
    VkDeviceSize field_bytes_ = 0;
    double cell_volume_ = 1.0;
    double dtau_ = 0.0;
    std::uint32_t kick_stride_ = 256;
    int kick_slots_ = 0;

    Kernel mul_;
    Kernel half_mul_;
    Kernel kin_mul_;
    Kernel damp_;
    Kernel phase_damp_;  // fused V+mask (optional; pd_kernel_ok_)
    Kernel mcwf_;        // fused multi-state axpy (optional; mcwf_kernel_ok_)
    Kernel mc_density_;  // GPU marching cubes (optional; mc_kernel_ok_)
    Kernel mc_classify_;
    Kernel mc_scan_;
    Kernel mc_emit_;
    Kernel conj_;
    Kernel fft_;
    Kernel norm_;
    Kernel scale_;
    Kernel finalize_;   // GPU norm finish (partials -> inv), fused relax
    Kernel scale_buf_;  // psi *= renorm[0], factor read from the SSBO
    Kernel kick_;
    Kernel shear_;
    Kernel inner_;
    Kernel axpy_;
    Kernel copy_;
    Kernel synth_;
    Kernel force_;
    Kernel dipole_;
    Kernel project_;
    Kernel bridge_store_;
    Kernel bridge_load_;
    Kernel pack_;
    Kernel unpack_;
    DescriptorArena arena_;
    // Dedicated async-batch objects (compute family): step_async records
    // here so the shared OneShot cb is never in flight unwaited.
    VkCommandPool async_pool_ = VK_NULL_HANDLE;
    VkCommandBuffer async_cb_ = VK_NULL_HANDLE;
    VkFence async_fence_ = VK_NULL_HANDLE;
    bool async_pending_ = false;
    bool async_bridged_ = false;

    // Ping-pong display volumes: bridges write volume_[vol_write_]; the
    // renderer samples volume_[vol_display_] (the last HOST-OBSERVED
    // completed write, so an in-flight async batch never races the render).
    DeviceContext::Image volume_[2]{};
    VkImageLayout volume_layout_[2] = {VK_IMAGE_LAYOUT_UNDEFINED,
                                       VK_IMAGE_LAYOUT_UNDEFINED};
    int vol_write_ = 0;
    int vol_display_ = -1;  // -1: nothing written yet (renderer falls back)
    // Flow velocity field (fp32 Bohmian v -> low-res rgba16f), double-buffered
    // with the display volume (same psi snapshot, same vol_write_/vol_display_).
    Kernel flow_vel_k_;
    DeviceContext::Image flow_vel_[2]{};
    VkImageLayout flow_vel_layout_[2] = {VK_IMAGE_LAYOUT_UNDEFINED,
                                         VK_IMAGE_LAYOUT_UNDEFINED};
    VkDescriptorSet flow_vel_set_[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    Buffer flow_vel_ubo_{};
    int flow_vel_m_ = 0;  // velocity-volume side (0 = not created)
    Buffer decode_scratch_[2]{};
    std::vector<State> states_;
    std::vector<int> free_full_states_;  // released full-state slots, reusable
    VkDeviceSize staging_bytes_ = 0;
    VkDeviceSize radial_bytes_ = 0;
    int proj_nr_ = 0;
    int proj_ncomp_ = 0;
    double proj_h_radial_ = 0.0;
    std::vector<std::vector<std::complex<double>>> glm_host_;

    Buffer psi_{};
    Buffer v_buf_{};    // scalar potential, R32
    Buffer kx2_buf_{};  // per-axis k^2, R32 x n
    Buffer ky2_buf_{};
    Buffer kz2_buf_{};
    Buffer halfp_ubo_{};
    Buffer fullp_ubo_{};  // coef = -dt (interior full kick)
    Buffer kinp_ubo_{};
    Buffer damp_buf_{};  // real absorber mask, R32
    double dt_step_ = 0.0;
    Buffer partials_{};
    Buffer staging_{};
    Buffer muln_ubo_{};
    Buffer conj1_ubo_{};
    Buffer conjN_ubo_{};
    Buffer fft_ubo_[3]{};
    Buffer scale_ubo_{};
    Buffer renorm_{};      // SSBO holding the GPU-computed inv (1 float)
    Buffer renorm_ubo_{};  // finalize params {ngroups, cell_volume}
    Buffer conjA_ubo_{};
    Buffer shear_ubo_[2]{};
    Buffer kick_ubo_{};
    Buffer axpy_ubo_{};
    Buffer synth_ubo_{};
    Buffer radial_buf_{};
    Buffer force_v_buf_{};  // scalar V for the in-shader mean-force gradient
    Buffer force_ubo_{};
    Buffer force_partials_{};
    Buffer dipole_ubo_{};
    Buffer dipole_partials_{};
    Buffer proj_sorted_buf_{};
    Buffer proj_binoff_buf_{};
    Buffer glm_buf_{};
    Buffer proj_ubo_{};
    Buffer bridge_ubo_{};
    Buffer scratch_bridge_{};
    Buffer relax_half_{};
    Buffer relax_kin_{};

    VkDescriptorSet halfv_set_ = VK_NULL_HANDLE;
    VkDescriptorSet fullv_set_ = VK_NULL_HANDLE;
    VkDescriptorSet kin3_set_ = VK_NULL_HANDLE;
    VkDescriptorSet damp_set_ = VK_NULL_HANDLE;
    VkDescriptorSet pd_full_set_ = VK_NULL_HANDLE;  // fused V+mask, coef -dt
    VkDescriptorSet pd_half_set_ = VK_NULL_HANDLE;  // fused V+mask, coef -dt/2
    bool pd_kernel_ok_ = false;
    bool fused_relax_ok_ = false;  // finalize_ + scale_buf_ wired
    bool flow_vel_ok_ = false;     // flow_velocity kernel wired
    Buffer mcwf_radial_{};  // kMcwfSlots x n_radial floats (slotted tables)
    VkDeviceSize mcwf_radial_bytes_ = 0;
    Buffer mcwf_ubo_{};
    VkDescriptorSet mcwf_set_ = VK_NULL_HANDLE;
    bool mcwf_kernel_ok_ = false;

    // GPU marching cubes: std140 param mirrors + transient buffers.
    struct McParams {
        std::int32_t npts[4];
        std::int32_t nblk[4];
        float box_min[4];
        float cell_h[4];
        float iso;
        std::int32_t max_tris;
        float pad0;
        float pad1;
    };
    struct McScanParams {
        std::int32_t nblocks;
        std::int32_t max_tris;
        std::int32_t pad0;
        std::int32_t pad1;
    };
    Buffer mc_den_{};        // |psi|^2 scalar field (R32, cells_)
    Buffer mc_tables_{};     // edge/tri/count tables (ses.mc.tables verbatim)
    Buffer mc_block_sums_{};
    Buffer mc_block_offsets_{};
    Buffer mc_vbuf_{};       // interleaved pos3 normal3 color3 (+VERTEX)
    Buffer mc_indirect_{};   // VkDrawIndirectCommand + raw/clamped totals
    Buffer mc_ubo_{};
    Buffer mc_scan_ubo_{};
    VkDescriptorSet mc_den_set_ = VK_NULL_HANDLE;
    VkDescriptorSet mc_classify_set_ = VK_NULL_HANDLE;
    VkDescriptorSet mc_scan_set_ = VK_NULL_HANDLE;
    VkDescriptorSet mc_emit_set_ = VK_NULL_HANDLE;
    bool mc_kernel_ok_ = false;
    bool mc_buffers_ok_ = false;
    bool mc_overflow_warned_ = false;
    int mc_max_tris_ = 0;
    int mc_nblk_[3] = {0, 0, 0};
    std::uint32_t mc_nblocks_ = 0;
    VkDescriptorSet conj1_set_ = VK_NULL_HANDLE;
    VkDescriptorSet conjN_set_ = VK_NULL_HANDLE;
    VkDescriptorSet fft_set_[3] = {VK_NULL_HANDLE, VK_NULL_HANDLE,
                                   VK_NULL_HANDLE};
    VkDescriptorSet norm_set_ = VK_NULL_HANDLE;
    VkDescriptorSet scale_set_ = VK_NULL_HANDLE;
    VkDescriptorSet finalize_set_ = VK_NULL_HANDLE;
    VkDescriptorSet scale_buf_set_ = VK_NULL_HANDLE;
    VkDescriptorSet conjA_set_ = VK_NULL_HANDLE;
    VkDescriptorSet shear_set_[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkDescriptorSet kick_set_ = VK_NULL_HANDLE;
    VkDescriptorSet relax_half_set_ = VK_NULL_HANDLE;
    VkDescriptorSet relax_kin_set_ = VK_NULL_HANDLE;
    VkDescriptorSet force_set_ = VK_NULL_HANDLE;
    VkDescriptorSet dipole_set_ = VK_NULL_HANDLE;
    VkDescriptorSet proj_set_ = VK_NULL_HANDLE;
    VkDescriptorSet store_set_[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkDescriptorSet load_set_[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkDescriptorSet pack_set_ = VK_NULL_HANDLE;
    VkDescriptorSet unpack_set_ = VK_NULL_HANDLE;
    VkDescriptorSet inner_any_set_ = VK_NULL_HANDLE;
    VkDescriptorSet synth_any_set_ = VK_NULL_HANDLE;
    VkDescriptorSet norm_any_set_ = VK_NULL_HANDLE;
    VkDescriptorSet scale_any_set_ = VK_NULL_HANDLE;

    bool use_vkfft_ = true;
#ifdef SES_HAVE_VKFFT
    // VkFFT plan on psi_'s VkBuffer; pointees of the configuration.
    bool vkfft_ready_ = false;
    bool vkfft_failed_ = false;
    VkFFTApplication vkfft_app_{};
    VkBuffer vkfft_psi_buf_ = VK_NULL_HANDLE;
    std::uint64_t vkfft_buf_size_ = 0;
    VkCommandPool vkfft_pool_ = VK_NULL_HANDLE;
    VkFence vkfft_fence_ = VK_NULL_HANDLE;
#endif
};

}  // namespace ses_vk
