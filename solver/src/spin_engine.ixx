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
#include <spin_site_gate_spv.h>
#include <spin_bond_gate_spv.h>
#include <spin_site_bloch_spv.h>
#include <spin_mf_snapshot_spv.h>
#include <spin_mf_sweep_spv.h>
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <vector>
export module ses.vk.spin_engine;
export import ses.vk.device;
export import ses.vk.compute;
export import ses.spinexact;
// volk/VMA textually first: VK_*/VMA macros never cross module boundaries.


// GPU exact 2^N Heisenberg; fp32 bit-faithful to CPU oracle
// (ses.spinexact gates; vkcheck check_spin_step).
// Gates alias the state SSBO in place -> compute-to-compute barrier between dispatches is mandatory.
// step()/upload() fold a per-site Bloch reduction so only 48 floats read back
// per frame (state stays GPU-resident); download_state() pulls the full 2^16
// only on demand (measurement). CONTRACT: vkcheck check_spin_bloch.


export namespace ses_vk {

class SpinEngine {
public:
    SpinEngine() = default;
    SpinEngine(const SpinEngine&) = delete;
    SpinEngine& operator=(const SpinEngine&) = delete;
    ~SpinEngine() { destroy(); }

    // false => caller stays on CPU.
    bool initialize(DeviceContext& ctx) {
        ctx_ = &ctx;
        dim_ = ses::kExactDim;
        half_groups_ =
            static_cast<std::uint32_t>((dim_ / 2 + 255) / 256);
        quarter_groups_ =
            static_cast<std::uint32_t>((dim_ / 4 + 255) / 256);
        nbonds_ = ses::exact_bonds(bonds_);

        if (!site_k_.create(ctx, k_spin_site_gate_spv,
                            k_spin_site_gate_spv_size,
                            {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                             {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER}}) ||
            !bond_k_.create(ctx, k_spin_bond_gate_spv,
                            k_spin_bond_gate_spv_size,
                            {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                             {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER}})) {
            return false;
        }
        const VkDeviceSize bytes = dim_ * 2 * sizeof(float);
        if (!ctx.create_device_buffer(bytes, &state_) ||
            !ctx.create_host_buffer(bytes,
                                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                    &staging_)) {
            return false;
        }
        const int n_sites = ses::kExactSites;
        // 2*sites (half + full angle field) + bonds; +1 more set for the reduce.
        const int n_sets = 2 * n_sites + nbonds_;
        if (!arena_.create(ctx, static_cast<std::uint32_t>(n_sets + 1),
                           static_cast<std::uint32_t>(n_sets + 2),
                           static_cast<std::uint32_t>(n_sets + 1))) {
            return false;
        }
        site_ubo_.resize(static_cast<std::size_t>(n_sites));
        site_ubo_full_.resize(static_cast<std::size_t>(n_sites));
        bond_ubo_.resize(static_cast<std::size_t>(nbonds_));
        site_set_.assign(static_cast<std::size_t>(n_sites), VK_NULL_HANDLE);
        site_set_full_.assign(static_cast<std::size_t>(n_sites), VK_NULL_HANDLE);
        bond_set_.assign(static_cast<std::size_t>(nbonds_), VK_NULL_HANDLE);
        for (int i = 0; i < n_sites; ++i) {
            const std::size_t si = static_cast<std::size_t>(i);
            for (int full = 0; full < 2; ++full) {
                Buffer& ubo = full ? site_ubo_full_[si] : site_ubo_[si];
                VkDescriptorSet& set = full ? site_set_full_[si] : site_set_[si];
                if (!ctx.create_host_buffer(sizeof(SiteParams),
                                            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                            &ubo))
                    return false;
                set = arena_.allocate(ctx, site_k_.set_layout());
                if (set == VK_NULL_HANDLE) return false;
                arena_.write_buffer(ctx, set, 0,
                                    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                    state_.buf);
                arena_.write_buffer(ctx, set, 1,
                                    VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, ubo.buf,
                                    sizeof(SiteParams));
            }
        }
        for (int i = 0; i < nbonds_; ++i) {
            if (!ctx.create_host_buffer(sizeof(BondParams),
                                        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                        &bond_ubo_[static_cast<std::size_t>(i)]))
                return false;
            const std::size_t bi = static_cast<std::size_t>(i);
            bond_set_[bi] = arena_.allocate(ctx, bond_k_.set_layout());
            if (bond_set_[bi] == VK_NULL_HANDLE) return false;
            arena_.write_buffer(ctx, bond_set_[bi], 0,
                                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, state_.buf);
            arena_.write_buffer(ctx, bond_set_[bi], 1,
                                VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                bond_ubo_[bi].buf, sizeof(BondParams));
        }
        // Per-site Bloch reduction: state SSBO -> 48-float device buffer.
        if (!reduce_k_.create(ctx, k_spin_site_bloch_spv,
                              k_spin_site_bloch_spv_size,
                              {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                               {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                               {2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER}})) {
            return false;
        }
        const VkDeviceSize bloch_bytes = kBlochFloats * sizeof(float);
        if (!ctx.create_device_buffer(bloch_bytes, &bloch_dev_) ||
            !ctx.create_host_buffer(bloch_bytes,
                                    VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                    &bloch_host_) ||
            !ctx.create_host_buffer(sizeof(BlochParams),
                                    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                    &bloch_ubo_)) {
            return false;
        }
        BlochParams bp{};
        bp.half_n = static_cast<std::uint32_t>(dim_ / 2);
        write_ubo(bloch_ubo_, &bp, sizeof(bp));
        reduce_set_ = arena_.allocate(ctx, reduce_k_.set_layout());
        if (reduce_set_ == VK_NULL_HANDLE) return false;
        arena_.write_buffer(ctx, reduce_set_, 0,
                            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, state_.buf);
        arena_.write_buffer(ctx, reduce_set_, 1,
                            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, bloch_dev_.buf,
                            bloch_bytes);
        arena_.write_buffer(ctx, reduce_set_, 2,
                            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, bloch_ubo_.buf,
                            sizeof(BlochParams));
        ready_ = true;
        return true;
    }

    void destroy() {
        if (ctx_ == nullptr) {
            return;
        }
        for (Buffer& b : site_ubo_) ctx_->destroy_buffer(&b);
        for (Buffer& b : site_ubo_full_) ctx_->destroy_buffer(&b);
        for (Buffer& b : bond_ubo_) ctx_->destroy_buffer(&b);
        site_ubo_.clear();
        site_ubo_full_.clear();
        bond_ubo_.clear();
        arena_.destroy(*ctx_);
        site_k_.destroy(*ctx_);
        bond_k_.destroy(*ctx_);
        reduce_k_.destroy(*ctx_);
        ctx_->destroy_buffer(&state_);
        ctx_->destroy_buffer(&staging_);
        ctx_->destroy_buffer(&bloch_dev_);
        ctx_->destroy_buffer(&bloch_host_);
        ctx_->destroy_buffer(&bloch_ubo_);
        ctx_ = nullptr;
        ready_ = false;
    }

    bool ready() const { return ready_; }
    std::size_t dim() const { return dim_; }

    // Host UBO writes only, no GPU dispatch.
    void set_params(double bx, double by, double bz, double j, double dt) {
        const double bmag = std::sqrt(bx * bx + by * by + bz * bz);
        has_field_ = bmag > 0.0;
        if (has_field_) {
            const double nx = bx / bmag, ny = by / bmag, nz = bz / bmag;
            // half = per-step-end sweep; full = merged step-boundary (2x angle).
            const ses::SiteGate g =
                ses::site_gate_matrix(nx, ny, nz, bmag * 0.5 * dt);
            const ses::SiteGate gf =
                ses::site_gate_matrix(nx, ny, nz, bmag * dt);
            for (int i = 0; i < ses::kExactSites; ++i) {
                for (int full = 0; full < 2; ++full) {
                    const ses::SiteGate& u = full ? gf : g;
                    SiteParams sp{};
                    sp.half_n = static_cast<std::uint32_t>(dim_ / 2);
                    sp.site = static_cast<std::uint32_t>(i);
                    fill_c(sp.row0, u.a00);
                    fill_c(sp.row0 + 2, u.a01);
                    fill_c(sp.row1, u.a10);
                    fill_c(sp.row1 + 2, u.a11);
                    write_ubo(full ? site_ubo_full_[static_cast<std::size_t>(i)]
                                   : site_ubo_[static_cast<std::size_t>(i)],
                              &sp, sizeof(sp));
                }
            }
        }
        const ses::BondGate bg = ses::bond_gate_params(0.5 * j * dt);
        for (int i = 0; i < nbonds_; ++i) {
            BondParams bp{};
            bp.quarter_n = static_cast<std::uint32_t>(dim_ / 4);
            bp.site_i = static_cast<std::uint32_t>(bonds_[i][0]);
            bp.site_j = static_cast<std::uint32_t>(bonds_[i][1]);
            fill_c(bp.gate, bg.phase);
            fill_c(bp.gate + 2, bg.diag);
            fill_c(bp.off4, bg.off);
            write_ubo(bond_ubo_[static_cast<std::size_t>(i)], &bp,
                      sizeof(bp));
        }
    }

    void upload(const std::vector<std::complex<double>>& c) {
        float* dst = static_cast<float*>(staging_.mapped);
        for (std::size_t m = 0; m < dim_; ++m) {
            dst[2 * m] = static_cast<float>(c[m].real());
            dst[2 * m + 1] = static_cast<float>(c[m].imag());
        }
        vmaFlushAllocation(ctx_->allocator, staging_.alloc, 0, VK_WHOLE_SIZE);
        OneShot shot;
        if (!shot.begin_compute(*ctx_)) {
            return;
        }
        VkCommandBuffer cb = shot.cb();
        const VkDeviceSize bytes = dim_ * 2 * sizeof(float);
        const VkBufferCopy r{0, 0, bytes};
        vkCmdCopyBuffer(cb, staging_.buf, state_.buf, 1, &r);
        barrier_transfer_to_compute(cb);
        record_reduce_and_copy(cb);  // bloch() reflects the uploaded state
        shot.submit_and_wait(*ctx_);
        vmaInvalidateAllocation(ctx_->allocator, bloch_host_.alloc, 0,
                                VK_WHOLE_SIZE);
    }

    // n Strang steps then a per-site Bloch reduction, all in one submit; only
    // 48 floats read back (state stays GPU-resident).
    void step(int n) {
        OneShot shot;
        if (!shot.begin_compute(*ctx_)) {
            return;
        }
        VkCommandBuffer cb = shot.cb();
        bool first = true;
        // Strang palindrome over n steps, field half-sweeps MERGED at internal
        // boundaries: (1/2)F [ B_fwd B_rev (F | (1/2)F) ]^n, using
        // exp(θH)exp(θH)=exp(2θH) so 2 half-sweeps collapse to 1 full-sweep.
        if (has_field_) {
            for (int s = 0; s < ses::kExactSites; ++s) {
                dispatch_site(cb, s, first, site_set_);  // leading half-sweep
                first = false;
            }
        }
        for (int k = 0; k < n; ++k) {
            for (int b = 0; b < nbonds_; ++b) {
                dispatch_bond(cb, b, first);
                first = false;
            }
            for (int b = nbonds_ - 1; b >= 0; --b) {
                dispatch_bond(cb, b, first);
                first = false;
            }
            if (has_field_) {
                // full-angle sweep between steps; half-angle only at the last.
                const std::vector<VkDescriptorSet>& sets =
                    (k + 1 < n) ? site_set_full_ : site_set_;
                for (int s = 0; s < ses::kExactSites; ++s) {
                    dispatch_site(cb, s, first, sets);
                    first = false;
                }
            }
        }
        barrier_compute_to_compute(cb);  // last gate -> reduce reads state_
        record_reduce_and_copy(cb);
        shot.submit_and_wait(*ctx_);
        vmaInvalidateAllocation(ctx_->allocator, bloch_host_.alloc, 0,
                                VK_WHOLE_SIZE);
    }

    // Full 2^16 state -> host staging; measurement path only (not per frame).
    void download_state() {
        OneShot shot;
        if (!shot.begin_compute(*ctx_)) {
            return;
        }
        VkCommandBuffer cb = shot.cb();
        barrier_compute_to_transfer(cb);  // prior step's writes -> transfer read
        const VkDeviceSize bytes = dim_ * 2 * sizeof(float);
        const VkBufferCopy r{0, 0, bytes};
        vkCmdCopyBuffer(cb, state_.buf, staging_.buf, 1, &r);
        barrier_transfer_to_host(cb);
        shot.submit_and_wait(*ctx_);
        vmaInvalidateAllocation(ctx_->allocator, staging_.alloc, 0,
                                VK_WHOLE_SIZE);
    }

    // The full fp32 state (interleaved re/im), valid after download_state()/upload.
    const float* state() const {
        return static_cast<const float*>(staging_.mapped);
    }

    // 48 floats: [3*site+0..2] = (<sx>,<sy>,<sz>); valid after step()/upload.
    const float* bloch() const {
        return static_cast<const float*>(bloch_host_.mapped);
    }

private:
    struct alignas(16) SiteParams {
        std::uint32_t half_n, site, pad0, pad1;
        float row0[4];
        float row1[4];
    };
    struct alignas(16) BondParams {
        std::uint32_t quarter_n, site_i, site_j, pad;
        float gate[4];
        float off4[4];
    };
    struct alignas(16) BlochParams {
        std::uint32_t half_n, pad0, pad1, pad2;
    };
    static constexpr int kBlochFloats = 3 * ses::kExactSites;

    static void fill_c(float* dst, const std::complex<double>& z) {
        dst[0] = static_cast<float>(z.real());
        dst[1] = static_cast<float>(z.imag());
    }
    void write_ubo(Buffer& b, const void* src, std::size_t sz) {
        std::memcpy(b.mapped, src, sz);
        vmaFlushAllocation(ctx_->allocator, b.alloc, 0, VK_WHOLE_SIZE);
    }
    void dispatch_site(VkCommandBuffer cb, int s, bool first,
                       const std::vector<VkDescriptorSet>& sets) {
        if (!first) {
            barrier_compute_to_compute(cb);
        }
        site_k_.bind(cb, sets[static_cast<std::size_t>(s)]);
        vkCmdDispatch(cb, half_groups_, 1, 1);
    }
    void dispatch_bond(VkCommandBuffer cb, int b, bool first) {
        if (!first) {
            barrier_compute_to_compute(cb);
        }
        bond_k_.bind(cb, bond_set_[static_cast<std::size_t>(b)]);
        vkCmdDispatch(cb, quarter_groups_, 1, 1);
    }
    // One workgroup per site reduces state_ -> bloch_dev_, then copy 48 floats
    // to host. Caller must barrier the state writes visible to this read.
    void record_reduce_and_copy(VkCommandBuffer cb) {
        reduce_k_.bind(cb, reduce_set_);
        vkCmdDispatch(cb, static_cast<std::uint32_t>(ses::kExactSites), 1, 1);
        barrier_compute_to_transfer(cb);
        const VkDeviceSize bytes = kBlochFloats * sizeof(float);
        const VkBufferCopy r{0, 0, bytes};
        vkCmdCopyBuffer(cb, bloch_dev_.buf, bloch_host_.buf, 1, &r);
        barrier_transfer_to_host(cb);
    }

    DeviceContext* ctx_ = nullptr;
    std::size_t dim_ = 0;
    std::uint32_t half_groups_ = 0;
    std::uint32_t quarter_groups_ = 0;
    int bonds_[2 * ses::kExactSites][2]{};
    int nbonds_ = 0;
    Buffer state_{};
    Buffer staging_{};
    Kernel site_k_;
    Kernel bond_k_;
    Kernel reduce_k_;
    DescriptorArena arena_;
    std::vector<Buffer> site_ubo_;       // half-angle field (step-end sweeps)
    std::vector<Buffer> site_ubo_full_;  // full-angle field (merged step-boundary)
    std::vector<Buffer> bond_ubo_;
    std::vector<VkDescriptorSet> site_set_;
    std::vector<VkDescriptorSet> site_set_full_;
    std::vector<VkDescriptorSet> bond_set_;
    Buffer bloch_dev_{};
    Buffer bloch_host_{};
    Buffer bloch_ubo_{};
    VkDescriptorSet reduce_set_ = VK_NULL_HANDLE;
    bool has_field_ = false;
    bool ready_ = false;
};

// GPU mean-field Heisenberg: 16 unit spinors, checkerboard Strang (snapshot +
// parity sweeps 0/1/0). fp32 mirror of ses::spinlattice_step; only 48 Bloch
// floats read back per tick. CONTRACT: vkcheck check_spin_mf.
class SpinMeanFieldEngine {
public:
    SpinMeanFieldEngine() = default;
    SpinMeanFieldEngine(const SpinMeanFieldEngine&) = delete;
    SpinMeanFieldEngine& operator=(const SpinMeanFieldEngine&) = delete;
    ~SpinMeanFieldEngine() { destroy(); }

    static constexpr int kSites = 16;
    static constexpr int kBlochFloats = 3 * kSites;

    bool initialize(DeviceContext& ctx) {
        ctx_ = &ctx;
        if (!snap_k_.create(ctx, k_spin_mf_snapshot_spv,
                            k_spin_mf_snapshot_spv_size,
                            {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                             {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}}) ||
            !sweep_k_.create(ctx, k_spin_mf_sweep_spv, k_spin_mf_sweep_spv_size,
                             {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                              {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                              {2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER}})) {
            return false;
        }
        const VkDeviceSize sp_bytes = kSites * 4 * sizeof(float);
        const VkDeviceSize bl_bytes = kBlochFloats * sizeof(float);
        if (!ctx.create_device_buffer(sp_bytes, &spinors_) ||
            !ctx.create_host_buffer(sp_bytes,
                                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                    &sp_staging_) ||
            !ctx.create_device_buffer(bl_bytes, &bloch_dev_) ||
            !ctx.create_host_buffer(bl_bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                    &bloch_host_) ||
            !ctx.create_host_buffer(sizeof(MfParams),
                                    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, &ubo0_) ||
            !ctx.create_host_buffer(sizeof(MfParams),
                                    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, &ubo1_)) {
            return false;
        }
        if (!arena_.create(ctx, 3, 6, 2)) {
            return false;
        }
        snap_set_ = arena_.allocate(ctx, snap_k_.set_layout());
        sweep0_set_ = arena_.allocate(ctx, sweep_k_.set_layout());
        sweep1_set_ = arena_.allocate(ctx, sweep_k_.set_layout());
        if (snap_set_ == VK_NULL_HANDLE || sweep0_set_ == VK_NULL_HANDLE ||
            sweep1_set_ == VK_NULL_HANDLE) {
            return false;
        }
        arena_.write_buffer(ctx, snap_set_, 0,
                            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, spinors_.buf);
        arena_.write_buffer(ctx, snap_set_, 1,
                            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, bloch_dev_.buf);
        for (VkDescriptorSet set : {sweep0_set_, sweep1_set_}) {
            arena_.write_buffer(ctx, set, 0,
                                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, spinors_.buf);
            arena_.write_buffer(ctx, set, 1,
                                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                bloch_dev_.buf);
        }
        arena_.write_buffer(ctx, sweep0_set_, 2,
                            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, ubo0_.buf,
                            sizeof(MfParams));
        arena_.write_buffer(ctx, sweep1_set_, 2,
                            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, ubo1_.buf,
                            sizeof(MfParams));
        ready_ = true;
        return true;
    }

    void destroy() {
        if (ctx_ == nullptr) {
            return;
        }
        arena_.destroy(*ctx_);
        snap_k_.destroy(*ctx_);
        sweep_k_.destroy(*ctx_);
        ctx_->destroy_buffer(&spinors_);
        ctx_->destroy_buffer(&sp_staging_);
        ctx_->destroy_buffer(&bloch_dev_);
        ctx_->destroy_buffer(&bloch_host_);
        ctx_->destroy_buffer(&ubo0_);
        ctx_->destroy_buffer(&ubo1_);
        ctx_ = nullptr;
        ready_ = false;
    }

    bool ready() const { return ready_; }

    void set_params(double bx, double by, double bz, double j, double alpha,
                    double dt) {
        MfParams p{};
        p.bx = static_cast<float>(bx);
        p.by = static_cast<float>(by);
        p.bz = static_cast<float>(bz);
        p.j = static_cast<float>(j);
        p.alpha = static_cast<float>(alpha);
        p.nx = 4;
        p.ny = 4;
        p.h = static_cast<float>(0.5 * dt);
        p.parity = 0;
        write_ubo(ubo0_, &p, sizeof(p));
        p.h = static_cast<float>(dt);
        p.parity = 1;
        write_ubo(ubo1_, &p, sizeof(p));
    }

    void upload(const std::vector<std::complex<double>>& up,
                const std::vector<std::complex<double>>& dn) {
        float* d = static_cast<float*>(sp_staging_.mapped);
        for (int i = 0; i < kSites; ++i) {
            const std::size_t si = static_cast<std::size_t>(i);
            d[4 * i + 0] = static_cast<float>(up[si].real());
            d[4 * i + 1] = static_cast<float>(up[si].imag());
            d[4 * i + 2] = static_cast<float>(dn[si].real());
            d[4 * i + 3] = static_cast<float>(dn[si].imag());
        }
        vmaFlushAllocation(ctx_->allocator, sp_staging_.alloc, 0,
                           VK_WHOLE_SIZE);
        OneShot shot;
        if (!shot.begin_compute(*ctx_)) {
            return;
        }
        VkCommandBuffer cb = shot.cb();
        const VkBufferCopy r{0, 0, kSites * 4 * sizeof(float)};
        vkCmdCopyBuffer(cb, sp_staging_.buf, spinors_.buf, 1, &r);
        barrier_transfer_to_compute(cb);
        snap_k_.bind(cb, snap_set_);
        vkCmdDispatch(cb, 1, 1, 1);
        copy_bloch_to_host(cb);
        shot.submit_and_wait(*ctx_);
        vmaInvalidateAllocation(ctx_->allocator, bloch_host_.alloc, 0,
                                VK_WHOLE_SIZE);
    }

    // n steps; each = snapshot + parity sweeps 0,1,0. Reads back 48 Bloch floats.
    void step(int n) {
        OneShot shot;
        if (!shot.begin_compute(*ctx_)) {
            return;
        }
        VkCommandBuffer cb = shot.cb();
        for (int k = 0; k < n; ++k) {
            if (k > 0) {
                barrier_compute_to_compute(cb);
            }
            snap_k_.bind(cb, snap_set_);
            vkCmdDispatch(cb, 1, 1, 1);
            barrier_compute_to_compute(cb);
            sweep_k_.bind(cb, sweep0_set_);
            vkCmdDispatch(cb, 1, 1, 1);
            barrier_compute_to_compute(cb);
            sweep_k_.bind(cb, sweep1_set_);
            vkCmdDispatch(cb, 1, 1, 1);
            barrier_compute_to_compute(cb);
            sweep_k_.bind(cb, sweep0_set_);
            vkCmdDispatch(cb, 1, 1, 1);
        }
        copy_bloch_to_host(cb);
        shot.submit_and_wait(*ctx_);
        vmaInvalidateAllocation(ctx_->allocator, bloch_host_.alloc, 0,
                                VK_WHOLE_SIZE);
    }

    // 48 floats: [3*site+0..2] = <sigma_site>; valid after step()/upload.
    const float* bloch() const {
        return static_cast<const float*>(bloch_host_.mapped);
    }

    // 16 spinors (interleaved up.re,up.im,dn.re,dn.im); valid after download().
    void download() {
        OneShot shot;
        if (!shot.begin_compute(*ctx_)) {
            return;
        }
        VkCommandBuffer cb = shot.cb();
        barrier_compute_to_transfer(cb);
        const VkBufferCopy r{0, 0, kSites * 4 * sizeof(float)};
        vkCmdCopyBuffer(cb, spinors_.buf, sp_staging_.buf, 1, &r);
        barrier_transfer_to_host(cb);
        shot.submit_and_wait(*ctx_);
        vmaInvalidateAllocation(ctx_->allocator, sp_staging_.alloc, 0,
                                VK_WHOLE_SIZE);
    }
    const float* spinors_host() const {
        return static_cast<const float*>(sp_staging_.mapped);
    }

private:
    struct alignas(16) MfParams {
        float bx, by, bz, j;
        float alpha, h, pad0, pad1;
        std::uint32_t parity, nx, ny, pad2;
    };
    void write_ubo(Buffer& b, const void* src, std::size_t sz) {
        std::memcpy(b.mapped, src, sz);
        vmaFlushAllocation(ctx_->allocator, b.alloc, 0, VK_WHOLE_SIZE);
    }
    void copy_bloch_to_host(VkCommandBuffer cb) {
        barrier_compute_to_transfer(cb);
        const VkBufferCopy r{0, 0, kBlochFloats * sizeof(float)};
        vkCmdCopyBuffer(cb, bloch_dev_.buf, bloch_host_.buf, 1, &r);
        barrier_transfer_to_host(cb);
    }

    DeviceContext* ctx_ = nullptr;
    Kernel snap_k_;
    Kernel sweep_k_;
    DescriptorArena arena_;
    Buffer spinors_{};
    Buffer sp_staging_{};
    Buffer bloch_dev_{};
    Buffer bloch_host_{};
    Buffer ubo0_{};
    Buffer ubo1_{};
    VkDescriptorSet snap_set_ = VK_NULL_HANDLE;
    VkDescriptorSet sweep0_set_ = VK_NULL_HANDLE;
    VkDescriptorSet sweep1_set_ = VK_NULL_HANDLE;
    bool ready_ = false;
};

}  // namespace ses_vk
