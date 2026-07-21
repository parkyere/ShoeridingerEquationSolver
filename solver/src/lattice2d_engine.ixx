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
#include <phase_multiply_spv.h>
#include <lat2d_sweep_x_spv.h>
#include <lat2d_sweep_y_spv.h>
#include <norm_peak_spv.h>
#include <norm_finalize_spv.h>
#include <scale_buf_spv.h>
#include <damp_mul_spv.h>
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <vector>
export module ses.vk.lattice2d_engine;
export import ses.vk.device;
export import ses.vk.compute;
import ses.grid;
// volk/VMA textually first: VK_*/VMA macros never cross module boundaries.


// GPU port of ses::PeierlsLattice2D: the Strang-split exact-2x2 Peierls bond
// sweep, one compute dispatch per (phase / sweep_x / sweep_y) with a
// compute-to-compute barrier between every dispatch (all alias state_ in place).
// relax() adds cosh/sinh imaginary-time sweeps + a fused norm reduce->rescale
// (norm_peak/norm_finalize/scale_buf). fp32, bit-faithful to the CPU oracle.
// CONTRACT: vkcheck check_lattice2d_step / check_lattice2d_relax.


export namespace ses_vk {

class Lattice2DEngine {
public:
    Lattice2DEngine() = default;
    Lattice2DEngine(const Lattice2DEngine&) = delete;
    Lattice2DEngine& operator=(const Lattice2DEngine&) = delete;
    ~Lattice2DEngine() { destroy(); }

    // false => caller stays on CPU.
    bool initialize(DeviceContext& ctx) {
        ctx_ = &ctx;
        const std::initializer_list<BindingDesc> ssb_ssb_ubo = {
            {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
            {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
            {2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER}};
        const std::initializer_list<BindingDesc> ssb_ubo_ssb = {
            {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
            {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER},
            {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}};
        if (!phase_k_.create(ctx, k_phase_multiply_spv,
                             k_phase_multiply_spv_size, ssb_ssb_ubo) ||
            !sweepx_k_.create(ctx, k_lat2d_sweep_x_spv, k_lat2d_sweep_x_spv_size,
                              ssb_ssb_ubo) ||
            !sweepy_k_.create(ctx, k_lat2d_sweep_y_spv, k_lat2d_sweep_y_spv_size,
                              ssb_ssb_ubo) ||
            !normpeak_k_.create(ctx, k_norm_peak_spv, k_norm_peak_spv_size,
                                ssb_ubo_ssb) ||
            !normfinal_k_.create(ctx, k_norm_finalize_spv,
                                 k_norm_finalize_spv_size, ssb_ubo_ssb) ||
            !scalebuf_k_.create(ctx, k_scale_buf_spv, k_scale_buf_spv_size,
                                ssb_ubo_ssb) ||
            !damp_k_.create(ctx, k_damp_mul_spv, k_damp_mul_spv_size,
                            ssb_ssb_ubo)) {
            return false;
        }
        // 15 sets (RT: phase+2sx+2sy; IT: phase+2sx+2sy; 3 norm); headroom.
        if (!arena_.create(ctx, 24, 64, 32)) {
            return false;
        }
        ready_ = true;
        return true;
    }

    void destroy() {
        if (ctx_ == nullptr) {
            return;
        }
        arena_.destroy(*ctx_);
        phase_k_.destroy(*ctx_);
        sweepx_k_.destroy(*ctx_);
        sweepy_k_.destroy(*ctx_);
        normpeak_k_.destroy(*ctx_);
        normfinal_k_.destroy(*ctx_);
        scalebuf_k_.destroy(*ctx_);
        damp_k_.destroy(*ctx_);
        free_lattice();
        ctx_ = nullptr;
        ready_ = false;
    }

    bool ready() const { return ready_ && n_cells_ > 0; }

    // (Re)build all lattice-sized buffers + the Strang coefficients for a grid,
    // on-site potential, and dt. Links default to 1 (set_uniform_field adds a
    // field). Safe to call repeatedly (dt / potential changes).
    void set_lattice(const ses::Grid3D& g, const std::vector<double>& potential,
                     double dt) {
        g_ = &g;
        nx_ = g.x.n;
        ny_ = g.y.n;
        const int cells = nx_ * ny_;
        const double tx = 0.5 / (g.x.spacing() * g.x.spacing());
        const double ty = 0.5 / (g.y.spacing() * g.y.spacing());
        if (cells != n_cells_) {
            free_lattice();
            n_cells_ = cells;
            if (!alloc_lattice(cells)) {
                n_cells_ = 0;
                return;
            }
            allocate_sets();
            const std::vector<std::complex<double>> ones(
                static_cast<std::size_t>(cells), std::complex<double>{1.0, 0.0});
            upload_complex(link_x_, ones);
            upload_complex(link_y_, ones);
        }
        // On-site tables: real-time e^{-i e0 dt/2}, imag-time e^{-e0 dt/2}.
        std::vector<std::complex<double>> hv_rt(static_cast<std::size_t>(cells));
        std::vector<std::complex<double>> hv_it(static_cast<std::size_t>(cells));
        for (int idx = 0; idx < cells; ++idx) {
            const double e0 = potential[static_cast<std::size_t>(idx)] +
                              2.0 * tx + 2.0 * ty;
            hv_rt[static_cast<std::size_t>(idx)] = std::complex<double>{
                std::cos(-0.5 * e0 * dt), std::sin(-0.5 * e0 * dt)};
            hv_it[static_cast<std::size_t>(idx)] =
                std::complex<double>{std::exp(-0.5 * e0 * dt), 0.0};
        }
        upload_complex(hv_rt_, hv_rt);
        upload_complex(hv_it_, hv_it);

        const NParams np{static_cast<std::uint32_t>(cells), 0, 0, 0};
        write_ubo(phase_ubo_, &np, sizeof(np));
        write_ubo(np_ubo_, &np, sizeof(np));
        write_ubo(sc_ubo_, &np, sizeof(np));
        const NFParams nf{kGroups, static_cast<float>(g.cell_volume()), 0.0f,
                          0.0f};
        write_ubo(nf_ubo_, &nf, sizeof(nf));
        // Real-time bonds: c = cos(t dt/2), mix = i sin(t dt/2); y-odd = full dt.
        write_sweep(sx_ubo_[0], 0, std::cos(tx * 0.5 * dt),
                    {0.0, std::sin(tx * 0.5 * dt)});
        write_sweep(sx_ubo_[1], 1, std::cos(tx * 0.5 * dt),
                    {0.0, std::sin(tx * 0.5 * dt)});
        write_sweep(sy_ubo_[0], 0, std::cos(ty * 0.5 * dt),
                    {0.0, std::sin(ty * 0.5 * dt)});
        write_sweep(sy_ubo_[1], 1, std::cos(ty * dt), {0.0, std::sin(ty * dt)});
        // Imag-time bonds: c = cosh, mix = sinh (real).
        write_sweep(sx_it_ubo_[0], 0, std::cosh(tx * 0.5 * dt),
                    {std::sinh(tx * 0.5 * dt), 0.0});
        write_sweep(sx_it_ubo_[1], 1, std::cosh(tx * 0.5 * dt),
                    {std::sinh(tx * 0.5 * dt), 0.0});
        write_sweep(sy_it_ubo_[0], 0, std::cosh(ty * 0.5 * dt),
                    {std::sinh(ty * 0.5 * dt), 0.0});
        write_sweep(sy_it_ubo_[1], 1, std::cosh(ty * dt),
                    {std::sinh(ty * dt), 0.0});
    }

    // Uniform field B along z (Landau gauge A_x = B y): x-links of row j carry
    // e^{-i B hx y_j}, anchored at y = 0. Mirrors PeierlsLattice2D. link_y = 1.
    void set_uniform_field(double b) {
        if (n_cells_ == 0) {
            return;
        }
        const double bh = b * g_->x.spacing();
        std::vector<std::complex<double>> lx(static_cast<std::size_t>(n_cells_));
        for (int j = 0; j < ny_; ++j) {
            const double th = -bh * g_->y.coord(j);
            const std::complex<double> u{std::cos(th), std::sin(th)};
            for (int i = 0; i < nx_; ++i) {
                lx[static_cast<std::size_t>(j * nx_ + i)] = u;
            }
        }
        upload_complex(link_x_, lx);
    }

    // Localized flux (double-slit solenoid) via the string gauge: x-links on one
    // column above/below (is,js) carry e^{-+i phi}. Mirrors PeierlsLattice2D.
    void set_solenoid(double phi, double xs, double ys, bool cut_up = true) {
        if (n_cells_ == 0) {
            return;
        }
        std::vector<std::complex<double>> lx(static_cast<std::size_t>(n_cells_),
                                             std::complex<double>{1.0, 0.0});
        int is = -1;
        for (int i = 0; i + 1 < nx_; ++i) {
            if (g_->x.coord(i) <= xs && xs < g_->x.coord(i + 1)) {
                is = i;
            }
        }
        int js = -1;
        for (int j = 0; j + 1 < ny_; ++j) {
            if (g_->y.coord(j) <= ys && ys < g_->y.coord(j + 1)) {
                js = j;
            }
        }
        if (is >= 0 && js >= 0) {
            const std::complex<double> up{std::cos(phi), -std::sin(phi)};
            const std::complex<double> dn{std::cos(phi), std::sin(phi)};
            if (cut_up) {
                for (int j = js + 1; j < ny_; ++j) {
                    lx[static_cast<std::size_t>(j * nx_ + is)] = up;
                }
            } else {
                for (int j = 0; j <= js; ++j) {
                    lx[static_cast<std::size_t>(j * nx_ + is)] = dn;
                }
            }
        }
        upload_complex(link_x_, lx);
    }

    // Real absorbing mask (interior 1, taper to 0 at walls) applied after each
    // real-time step. Empty / not-called => no absorber. Not used during relax.
    void set_absorber(const std::vector<double>& mask) {
        if (n_cells_ == 0 || static_cast<int>(mask.size()) != n_cells_) {
            return;
        }
        upload_real(mask_, mask);
        has_absorber_ = true;
    }
    void clear_absorber() { has_absorber_ = false; }

    void upload(const std::vector<std::complex<double>>& psi) {
        upload_complex(state_, psi);
    }

    // state_ -> host; caller reads state() as interleaved re/im fp32.
    void download() {
        OneShot shot;
        if (!shot.begin_compute(*ctx_)) {
            return;
        }
        VkCommandBuffer cb = shot.cb();
        barrier_compute_to_transfer(cb);
        const VkDeviceSize bytes =
            static_cast<VkDeviceSize>(n_cells_) * 2 * sizeof(float);
        const VkBufferCopy r{0, 0, bytes};
        vkCmdCopyBuffer(cb, state_.buf, staging_.buf, 1, &r);
        barrier_transfer_to_host(cb);
        shot.submit_and_wait(*ctx_);
        vmaInvalidateAllocation(ctx_->allocator, staging_.alloc, 0,
                                VK_WHOLE_SIZE);
    }

    const float* state() const {
        return static_cast<const float*>(staging_.mapped);
    }

    // n real-time Strang steps in one submit; leaves state_ GPU-resident.
    void step(int n) {
        OneShot shot;
        if (!shot.begin_compute(*ctx_)) {
            return;
        }
        VkCommandBuffer cb = shot.cb();
        for (int s = 0; s < n; ++s) {
            palindrome(cb, phase_set_, sx_set_, sy_set_);
            if (has_absorber_) {
                damp_k_.bind(cb, damp_set_);
                vkCmdDispatch(cb, cell_groups(), 1, 1);
                barrier_compute_to_compute(cb);
            }
        }
        shot.submit_and_wait(*ctx_);
    }

    // n imaginary-time steps, each renormalized to <psi|psi> = 1 (cell measure).
    void relax(int n) {
        OneShot shot;
        if (!shot.begin_compute(*ctx_)) {
            return;
        }
        VkCommandBuffer cb = shot.cb();
        const std::uint32_t groups = cell_groups();
        for (int s = 0; s < n; ++s) {
            palindrome(cb, phase_it_set_, sx_it_set_, sy_it_set_);
            normpeak_k_.bind(cb, normpeak_set_);
            vkCmdDispatch(cb, kGroups, 1, 1);
            barrier_compute_to_compute(cb);
            normfinal_k_.bind(cb, normfinal_set_);
            vkCmdDispatch(cb, 1, 1, 1);
            barrier_compute_to_compute(cb);
            scalebuf_k_.bind(cb, scale_set_);
            vkCmdDispatch(cb, groups, 1, 1);
            barrier_compute_to_compute(cb);
        }
        shot.submit_and_wait(*ctx_);
    }

private:
    static constexpr std::uint32_t kGroups = 256;

    struct NParams {
        std::uint32_t n, pad0, pad1, pad2;
    };
    struct NFParams {
        std::uint32_t ngroups;
        float cell_volume, pad0, pad1;
    };
    struct SweepParams {
        std::uint32_t nx, ny, parity;
        float c;
        float mix_re, mix_im, pad0, pad1;
    };

    std::uint32_t cell_groups() const {
        return static_cast<std::uint32_t>((n_cells_ + 255) / 256);
    }

    // One Strang palindrome: (1/2)V . Bx . By(full) . Bx . (1/2)V, with a
    // compute-to-compute barrier between every dispatch (state aliased in place).
    void palindrome(VkCommandBuffer cb, VkDescriptorSet phase_set,
                    const VkDescriptorSet* sx, const VkDescriptorSet* sy) {
        const std::uint32_t groups = cell_groups();
        auto go = [&](const Kernel& k, VkDescriptorSet set) {
            k.bind(cb, set);
            vkCmdDispatch(cb, groups, 1, 1);
            barrier_compute_to_compute(cb);
        };
        go(phase_k_, phase_set);
        go(sweepx_k_, sx[0]);
        go(sweepx_k_, sx[1]);
        go(sweepy_k_, sy[0]);
        go(sweepy_k_, sy[1]);
        go(sweepy_k_, sy[0]);
        go(sweepx_k_, sx[1]);
        go(sweepx_k_, sx[0]);
        go(phase_k_, phase_set);
    }

    void write_ubo(Buffer& b, const void* src, std::size_t bytes) {
        std::memcpy(b.mapped, src, bytes);
        vmaFlushAllocation(ctx_->allocator, b.alloc, 0, VK_WHOLE_SIZE);
    }

    void write_sweep(Buffer& b, std::uint32_t parity, double c,
                     std::complex<double> mix) {
        SweepParams sp{static_cast<std::uint32_t>(nx_),
                       static_cast<std::uint32_t>(ny_),
                       parity,
                       static_cast<float>(c),
                       static_cast<float>(mix.real()),
                       static_cast<float>(mix.imag()),
                       0.0f,
                       0.0f};
        write_ubo(b, &sp, sizeof(sp));
    }

    void upload_complex(Buffer& dst,
                        const std::vector<std::complex<double>>& src) {
        float* p = static_cast<float*>(staging_.mapped);
        for (std::size_t i = 0; i < src.size(); ++i) {
            p[2 * i] = static_cast<float>(src[i].real());
            p[2 * i + 1] = static_cast<float>(src[i].imag());
        }
        vmaFlushAllocation(ctx_->allocator, staging_.alloc, 0, VK_WHOLE_SIZE);
        OneShot shot;
        if (!shot.begin_compute(*ctx_)) {
            return;
        }
        VkCommandBuffer cb = shot.cb();
        const VkDeviceSize bytes =
            static_cast<VkDeviceSize>(src.size()) * 2 * sizeof(float);
        const VkBufferCopy r{0, 0, bytes};
        vkCmdCopyBuffer(cb, staging_.buf, dst.buf, 1, &r);
        barrier_transfer_to_compute(cb);
        shot.submit_and_wait(*ctx_);
    }

    void upload_real(Buffer& dst, const std::vector<double>& src) {
        float* p = static_cast<float*>(staging_.mapped);
        for (std::size_t i = 0; i < src.size(); ++i) {
            p[i] = static_cast<float>(src[i]);
        }
        vmaFlushAllocation(ctx_->allocator, staging_.alloc, 0, VK_WHOLE_SIZE);
        OneShot shot;
        if (!shot.begin_compute(*ctx_)) {
            return;
        }
        VkCommandBuffer cb = shot.cb();
        const VkBufferCopy r{0, 0,
                             static_cast<VkDeviceSize>(src.size()) *
                                 sizeof(float)};
        vkCmdCopyBuffer(cb, staging_.buf, dst.buf, 1, &r);
        barrier_transfer_to_compute(cb);
        shot.submit_and_wait(*ctx_);
    }

    bool alloc_lattice(int cells) {
        const VkDeviceSize bytes =
            static_cast<VkDeviceSize>(cells) * 2 * sizeof(float);
        const VkBufferUsageFlags ubo = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
        return ctx_->create_device_buffer(bytes, &state_) &&
               ctx_->create_device_buffer(bytes, &link_x_) &&
               ctx_->create_device_buffer(bytes, &link_y_) &&
               ctx_->create_device_buffer(bytes, &hv_rt_) &&
               ctx_->create_device_buffer(bytes, &hv_it_) &&
               ctx_->create_device_buffer(2 * kGroups * sizeof(float),
                                          &partials_) &&
               ctx_->create_device_buffer(sizeof(float), &renorm_) &&
               ctx_->create_device_buffer(
                   static_cast<VkDeviceSize>(cells) * sizeof(float), &mask_) &&
               ctx_->create_host_buffer(bytes,
                                        VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                        &staging_) &&
               ctx_->create_host_buffer(sizeof(NParams), ubo, &phase_ubo_) &&
               ctx_->create_host_buffer(sizeof(NParams), ubo, &np_ubo_) &&
               ctx_->create_host_buffer(sizeof(NFParams), ubo, &nf_ubo_) &&
               ctx_->create_host_buffer(sizeof(NParams), ubo, &sc_ubo_) &&
               ctx_->create_host_buffer(sizeof(SweepParams), ubo, &sx_ubo_[0]) &&
               ctx_->create_host_buffer(sizeof(SweepParams), ubo, &sx_ubo_[1]) &&
               ctx_->create_host_buffer(sizeof(SweepParams), ubo, &sy_ubo_[0]) &&
               ctx_->create_host_buffer(sizeof(SweepParams), ubo, &sy_ubo_[1]) &&
               ctx_->create_host_buffer(sizeof(SweepParams), ubo,
                                        &sx_it_ubo_[0]) &&
               ctx_->create_host_buffer(sizeof(SweepParams), ubo,
                                        &sx_it_ubo_[1]) &&
               ctx_->create_host_buffer(sizeof(SweepParams), ubo,
                                        &sy_it_ubo_[0]) &&
               ctx_->create_host_buffer(sizeof(SweepParams), ubo,
                                        &sy_it_ubo_[1]);
    }

    void allocate_sets() {
        phase_set_ = arena_.allocate(*ctx_, phase_k_.set_layout());
        write3(phase_set_, hv_rt_.buf, phase_ubo_.buf, sizeof(NParams));
        phase_it_set_ = arena_.allocate(*ctx_, phase_k_.set_layout());
        write3(phase_it_set_, hv_it_.buf, phase_ubo_.buf, sizeof(NParams));
        for (int p = 0; p < 2; ++p) {
            const std::size_t pi = static_cast<std::size_t>(p);
            sx_set_[p] = arena_.allocate(*ctx_, sweepx_k_.set_layout());
            write3(sx_set_[p], link_x_.buf, sx_ubo_[pi].buf, sizeof(SweepParams));
            sy_set_[p] = arena_.allocate(*ctx_, sweepy_k_.set_layout());
            write3(sy_set_[p], link_y_.buf, sy_ubo_[pi].buf, sizeof(SweepParams));
            sx_it_set_[p] = arena_.allocate(*ctx_, sweepx_k_.set_layout());
            write3(sx_it_set_[p], link_x_.buf, sx_it_ubo_[pi].buf,
                   sizeof(SweepParams));
            sy_it_set_[p] = arena_.allocate(*ctx_, sweepy_k_.set_layout());
            write3(sy_it_set_[p], link_y_.buf, sy_it_ubo_[pi].buf,
                   sizeof(SweepParams));
        }
        // norm chain layouts: {0:storage, 1:ubo, 2:storage}.
        normpeak_set_ = arena_.allocate(*ctx_, normpeak_k_.set_layout());
        write_reduce(normpeak_set_, state_.buf, np_ubo_.buf, sizeof(NParams),
                     partials_.buf);
        normfinal_set_ = arena_.allocate(*ctx_, normfinal_k_.set_layout());
        write_reduce(normfinal_set_, partials_.buf, nf_ubo_.buf, sizeof(NFParams),
                     renorm_.buf);
        scale_set_ = arena_.allocate(*ctx_, scalebuf_k_.set_layout());
        write_reduce(scale_set_, state_.buf, sc_ubo_.buf, sizeof(NParams),
                     renorm_.buf);
        // damp layout matches the sweeps: {0:state, 1:mask, 2:ubo{n}}.
        damp_set_ = arena_.allocate(*ctx_, damp_k_.set_layout());
        write3(damp_set_, mask_.buf, np_ubo_.buf, sizeof(NParams));
    }

    // 3-binding sweep/phase layout: 0 = state (RW), 1 = aux SSBO, 2 = params UBO.
    void write3(VkDescriptorSet set, VkBuffer aux, VkBuffer ubo,
                VkDeviceSize ubo_bytes) {
        arena_.write_buffer(*ctx_, set, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                            state_.buf);
        arena_.write_buffer(*ctx_, set, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                            aux);
        arena_.write_buffer(*ctx_, set, 2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                            ubo, ubo_bytes);
    }

    // norm-chain layout: 0 = data SSBO, 1 = params UBO, 2 = aux SSBO.
    void write_reduce(VkDescriptorSet set, VkBuffer data, VkBuffer ubo,
                      VkDeviceSize ubo_bytes, VkBuffer aux) {
        arena_.write_buffer(*ctx_, set, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                            data);
        arena_.write_buffer(*ctx_, set, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                            ubo, ubo_bytes);
        arena_.write_buffer(*ctx_, set, 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                            aux);
    }

    void free_lattice() {
        if (n_cells_ == 0) {
            return;
        }
        for (Buffer* b : {&state_, &link_x_, &link_y_, &hv_rt_, &hv_it_,
                          &partials_, &renorm_, &mask_, &staging_, &phase_ubo_,
                          &np_ubo_, &nf_ubo_, &sc_ubo_, &sx_ubo_[0], &sx_ubo_[1],
                          &sy_ubo_[0], &sy_ubo_[1], &sx_it_ubo_[0],
                          &sx_it_ubo_[1], &sy_it_ubo_[0], &sy_it_ubo_[1]}) {
            ctx_->destroy_buffer(b);
        }
        has_absorber_ = false;
        n_cells_ = 0;
    }

    DeviceContext* ctx_ = nullptr;
    const ses::Grid3D* g_ = nullptr;
    bool ready_ = false;
    int nx_ = 0;
    int ny_ = 0;
    int n_cells_ = 0;

    Kernel phase_k_;
    Kernel sweepx_k_;
    Kernel sweepy_k_;
    Kernel normpeak_k_;
    Kernel normfinal_k_;
    Kernel scalebuf_k_;
    Kernel damp_k_;
    DescriptorArena arena_;
    bool has_absorber_ = false;

    Buffer state_;
    Buffer link_x_;
    Buffer link_y_;
    Buffer hv_rt_;
    Buffer hv_it_;
    Buffer partials_;
    Buffer renorm_;
    Buffer mask_;
    Buffer staging_;
    Buffer phase_ubo_;
    Buffer np_ubo_;
    Buffer nf_ubo_;
    Buffer sc_ubo_;
    Buffer sx_ubo_[2];
    Buffer sy_ubo_[2];
    Buffer sx_it_ubo_[2];
    Buffer sy_it_ubo_[2];

    VkDescriptorSet phase_set_ = VK_NULL_HANDLE;
    VkDescriptorSet phase_it_set_ = VK_NULL_HANDLE;
    VkDescriptorSet sx_set_[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkDescriptorSet sy_set_[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkDescriptorSet sx_it_set_[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkDescriptorSet sy_it_set_[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkDescriptorSet normpeak_set_ = VK_NULL_HANDLE;
    VkDescriptorSet normfinal_set_ = VK_NULL_HANDLE;
    VkDescriptorSet scale_set_ = VK_NULL_HANDLE;
    VkDescriptorSet damp_set_ = VK_NULL_HANDLE;
};

}  // namespace ses_vk
