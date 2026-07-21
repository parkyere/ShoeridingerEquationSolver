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
// fp32, bit-faithful to the CPU oracle. CONTRACT: vkcheck check_lattice2d_step.


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
        if (!phase_k_.create(ctx, k_phase_multiply_spv, k_phase_multiply_spv_size,
                             {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                              {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                              {2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER}}) ||
            !sweepx_k_.create(ctx, k_lat2d_sweep_x_spv, k_lat2d_sweep_x_spv_size,
                              {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                               {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                               {2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER}}) ||
            !sweepy_k_.create(ctx, k_lat2d_sweep_y_spv, k_lat2d_sweep_y_spv_size,
                              {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                               {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                               {2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER}})) {
            return false;
        }
        // 5 sets (phase + 2 sweep_x + 2 sweep_y); headroom for relax later.
        if (!arena_.create(ctx, 16, 48, 24)) {
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
        free_lattice();
        ctx_ = nullptr;
        ready_ = false;
    }

    bool ready() const { return ready_ && n_cells_ > 0; }

    // (Re)build all lattice-sized buffers + the Strang coefficients for a grid,
    // on-site potential, and dt. link phases default to 1 (set_uniform_field to
    // add a field). Safe to call repeatedly (dt / potential changes).
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
            const VkDeviceSize bytes =
                static_cast<VkDeviceSize>(cells) * 2 * sizeof(float);
            const bool ok =
                ctx_->create_device_buffer(bytes, &state_) &&
                ctx_->create_device_buffer(bytes, &link_x_) &&
                ctx_->create_device_buffer(bytes, &link_y_) &&
                ctx_->create_device_buffer(bytes, &hv_rt_) &&
                ctx_->create_host_buffer(bytes,
                                         VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                             VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                         &staging_) &&
                ctx_->create_host_buffer(sizeof(PhaseParams),
                                         VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                         &phase_ubo_) &&
                ctx_->create_host_buffer(sizeof(SweepParams),
                                         VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                         &sx_ubo_[0]) &&
                ctx_->create_host_buffer(sizeof(SweepParams),
                                         VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                         &sx_ubo_[1]) &&
                ctx_->create_host_buffer(sizeof(SweepParams),
                                         VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                         &sy_ubo_[0]) &&
                ctx_->create_host_buffer(sizeof(SweepParams),
                                         VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                         &sy_ubo_[1]);
            if (!ok) {
                n_cells_ = 0;
                return;
            }
            allocate_sets();
            std::vector<std::complex<double>> ones(
                static_cast<std::size_t>(cells), std::complex<double>{1.0, 0.0});
            upload_complex(link_x_, ones);
            upload_complex(link_y_, ones);
        }
        // On-site real-time phase table e^{-i (V + 2tx + 2ty) dt/2}.
        std::vector<std::complex<double>> hv(static_cast<std::size_t>(cells));
        for (int idx = 0; idx < cells; ++idx) {
            const double e0 = potential[static_cast<std::size_t>(idx)] +
                              2.0 * tx + 2.0 * ty;
            hv[static_cast<std::size_t>(idx)] =
                std::complex<double>{std::cos(-0.5 * e0 * dt),
                                     std::sin(-0.5 * e0 * dt)};
        }
        upload_complex(hv_rt_, hv);

        PhaseParams pp{static_cast<std::uint32_t>(cells), 0, 0, 0};
        write_ubo(phase_ubo_, &pp, sizeof(pp));
        // Real-time bond coefficients: c = cos(t dt/2), mix = i sin(t dt/2);
        // y-odd bond is the full-dt palindrome centre.
        write_sweep(sx_ubo_[0], 0, std::cos(tx * 0.5 * dt),
                    {0.0, std::sin(tx * 0.5 * dt)});
        write_sweep(sx_ubo_[1], 1, std::cos(tx * 0.5 * dt),
                    {0.0, std::sin(tx * 0.5 * dt)});
        write_sweep(sy_ubo_[0], 0, std::cos(ty * 0.5 * dt),
                    {0.0, std::sin(ty * 0.5 * dt)});
        write_sweep(sy_ubo_[1], 1, std::cos(ty * dt),
                    {0.0, std::sin(ty * dt)});
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
        const std::uint32_t groups =
            static_cast<std::uint32_t>((n_cells_ + 255) / 256);
        auto go = [&](const Kernel& k, VkDescriptorSet set) {
            k.bind(cb, set);
            vkCmdDispatch(cb, groups, 1, 1);
            barrier_compute_to_compute(cb);
        };
        for (int s = 0; s < n; ++s) {
            go(phase_k_, phase_set_);
            go(sweepx_k_, sx_set_[0]);
            go(sweepx_k_, sx_set_[1]);
            go(sweepy_k_, sy_set_[0]);
            go(sweepy_k_, sy_set_[1]);
            go(sweepy_k_, sy_set_[0]);
            go(sweepx_k_, sx_set_[1]);
            go(sweepx_k_, sx_set_[0]);
            go(phase_k_, phase_set_);
        }
        shot.submit_and_wait(*ctx_);
    }

private:
    struct PhaseParams {
        std::uint32_t n, pad0, pad1, pad2;
    };
    struct SweepParams {
        std::uint32_t nx, ny, parity;
        float c;
        float mix_re, mix_im, pad0, pad1;
    };

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

    void allocate_sets() {
        phase_set_ = arena_.allocate(*ctx_, phase_k_.set_layout());
        write3(phase_set_, hv_rt_.buf, phase_ubo_.buf, sizeof(PhaseParams));
        for (int p = 0; p < 2; ++p) {
            sx_set_[p] = arena_.allocate(*ctx_, sweepx_k_.set_layout());
            write3(sx_set_[p], link_x_.buf, sx_ubo_[p].buf, sizeof(SweepParams));
            sy_set_[p] = arena_.allocate(*ctx_, sweepy_k_.set_layout());
            write3(sy_set_[p], link_y_.buf, sy_ubo_[p].buf, sizeof(SweepParams));
        }
    }

    // Shared 3-binding layout: 0 = state (RW), 1 = aux SSBO, 2 = params UBO.
    void write3(VkDescriptorSet set, VkBuffer aux, VkBuffer ubo,
                VkDeviceSize ubo_bytes) {
        arena_.write_buffer(*ctx_, set, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                            state_.buf);
        arena_.write_buffer(*ctx_, set, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                            aux);
        arena_.write_buffer(*ctx_, set, 2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                            ubo, ubo_bytes);
    }

    void free_lattice() {
        if (n_cells_ == 0) {
            return;
        }
        ctx_->destroy_buffer(&state_);
        ctx_->destroy_buffer(&link_x_);
        ctx_->destroy_buffer(&link_y_);
        ctx_->destroy_buffer(&hv_rt_);
        ctx_->destroy_buffer(&staging_);
        ctx_->destroy_buffer(&phase_ubo_);
        ctx_->destroy_buffer(&sx_ubo_[0]);
        ctx_->destroy_buffer(&sx_ubo_[1]);
        ctx_->destroy_buffer(&sy_ubo_[0]);
        ctx_->destroy_buffer(&sy_ubo_[1]);
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
    DescriptorArena arena_;

    Buffer state_;
    Buffer link_x_;
    Buffer link_y_;
    Buffer hv_rt_;
    Buffer staging_;
    Buffer phase_ubo_;
    Buffer sx_ubo_[2];
    Buffer sy_ubo_[2];

    VkDescriptorSet phase_set_ = VK_NULL_HANDLE;
    VkDescriptorSet sx_set_[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkDescriptorSet sy_set_[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
};

}  // namespace ses_vk
