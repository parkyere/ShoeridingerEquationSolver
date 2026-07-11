// Humble Object shell -- the Qt + hand-written OpenGL boundary.
//
// NO domain logic lives here (docs/ARCHITECTURE.md). The TDSE runs in core's
// WavepacketSimulation; matrices, colormaps, marching cubes, and ALL the
// volume-rendering math (ray_box, Beer-Lambert alpha, front-to-back
// compositing, phase LUT) live in core and are unit-tested -- the GLSL below
// is a line-by-line transcription of those verified formulas.
//
// v5 deliverable: THE STATIONARY-STATE DEMO ARC on top of the volume-
// rendered cloud. Key 2 switches stepping to imaginary time: you WATCH the
// packet cool into the 1s ground state (energy readout converging). Key 1
// returns to real time: the density freezes -- only the hue cycles at
// e^{-i E0 t} -- the definition of a stationary state, on screen. Key R
// releases a fresh wavepacket.
// Controls: drag = orbit, wheel = zoom, space = pause, Tab = cloud/surface,
// 1 = real time, 2 = relax (imaginary time), 3 = relax to 2p, 4 = relax to
// 2s, 5 = excite an n=3 state (cascade demo), R = reset, M = measure,
// D = decay off/on, L = laser (off -> Z -> X -> off),
// [ ] = thinner/denser cloud.
//
// T3 (laser): a resonant dipole drive at w = E(2p) - E(1s) pumps 1s -> 2p_z
// and Rabi-flops the populations (title readout); with decay ON the photon
// counter clicks through repeated absorb/emit cycles -- fluorescence. The
// X-polarized flavor pumps the ORTHOGONAL 2p_x, so the monitored P(2pz)
// stays flat: the selection rule, live on screen.
//
// T5 (multi-channel decay): EVERY tracked orbital gets its lifetime. The
// whole n<=2 manifold (1s, 2p_x, 2p_y, 2p_z, 2s) is cached; every downward
// pair gets its Einstein A from our own wavefunctions and the channels
// compete as Poisson processes (core-tested pick_decay_channel). Selection
// rules emerge: A(2s->1s) ~ 0, so 2s (key 4) just sits there -- metastable
// -- while any 2p decays in a ~4.5 ns-scale lifetime (display-accelerated
// by ONE common factor, so RELATIVE lifetimes stay honest).
//
// T6 (decay by default): spontaneous emission is not opt-in in nature, so
// it is not opt-in here. At startup the app SOLVES the atom first -- the
// eigenstate atlas builds chunked across frames (watch each state converge,
// progress in the title) -- then the wavepacket demo starts with decay
// armed. D turns decay OFF for studying pure unitary evolution.
//
// T7 (lifetimes to n = 10): the potential is spherical, so the radial
// engine (core/radial.hpp) solves EVERY bound level to n = 10 in 1D --
// energies and E1 lifetimes for all 55 levels, printed at startup. The 3D
// tracked manifold is what the +-80 Bohr box can hold: n <= 6 (91 states,
// 6s/6p box-critical), each synthesized ON THE GPU as (u/r) Y_lm straight
// into its resident buffer (gpu_engine kSynthSrc, mirroring the unit-tested
// core/harmonics.hpp) -- no imaginary-time ladder, no CPU field. Key 5 excites
// an n = 3 state to watch the CASCADE (e.g.
// 3d -> 2p -> 1s, two photons). Relaxation demos auto-complete: when the
// ITP energy plateaus, the app returns to real time so lifetimes act.

// ses_vk first: volk (inside) defines VK_NO_PROTOTYPES and must own the
// vulkan.h inclusion before any Qt header pulls its own Vulkan integration.
#include "vk_blobs.hpp"

#include <core/camera.hpp>
#include <core/colormap.hpp>
#include <core/decay.hpp>
#include <core/emission.hpp>
#include <core/radial.hpp>
#include <core/field.hpp>
#include <core/grid.hpp>
#include <core/imaginary_time.hpp>
#include <core/magnetic.hpp>
#include <core/marching_cubes.hpp>
#include <core/observables.hpp>
#include <core/potential.hpp>
#include <core/projection.hpp>
#include <core/sampling.hpp>
#include <core/simulation.hpp>
#include <core/sphere.hpp>
#include <core/vec.hpp>
#include <core/volume.hpp>
#include <core/vram_budget.hpp>

#include "atom_model.hpp"
#include "selftest_arcs.hpp"
#include "manifold_spec.hpp"

#include <cstring>  // std::strcmp for the VRAM extension probe

#include <QApplication>
#include <QFile>
#include <QImage>
#include <QKeyEvent>
#include <QMainWindow>
#include <QMatrix4x4>
#include <QMessageBox>
#include <QMouseEvent>
#include <QRhiWidget>
#include <QString>
#include <QTimer>
#include <QLabel>
#include <QSlider>
#include <QToolBar>
#include <QVulkanFunctions>
#include <QVulkanInstance>
#include <QWheelEvent>

#include <rhi/qrhi_platform.h>  // QRhiVulkanNativeHandles (VRAM budget probe)

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <random>
#include <utility>
#include <vector>

namespace {

[[noreturn]] void fatal_rhi_error(const char* stage, const QString& detail) {
    qCritical("%s: %s", stage, qPrintable(detail));
    QMessageBox::critical(nullptr, QStringLiteral("Graphics error"),
                          QStringLiteral("%1\n\n%2").arg(QLatin1String(stage), detail));
    std::exit(EXIT_FAILURE);
}

// The atom's potential is the BARE Coulomb -Z/r (true hydrogen), REGULARIZED on
// the grid rather than softened. The 3D grid has a point on the nucleus where
// -1/r would be -infinity, so that ONE cell takes the analytic cell average
// (ses::regularized_coulomb_potential); every other cell is exact -1/r. The
// radial solves feed bare -1/r directly -- their grid r_i = (i+1)h never hits 0.
// Away from the nucleus cell the two are the SAME operator, so synthesized
// orbitals stay eigenstates of the propagated Hamiltonian; only the coarse
// nucleus cell differs, leaving a small s-state cusp gap in the startup h-audit
// (E_radial reads textbook; <H>_grid is ~1.5 eV shallower for 1s at h = 0.625,
// shrinking ~1/n^3 so 4s/5s/6s land within ~0.02 eV). This replaces the old
// soft-Coulomb -Z/sqrt(r^2 + a^2), which rounded the whole well and pushed E(1s)
// up to -9 eV; bare Coulomb restores the textbook -13.6 eV Rydberg spectrum and
// its exact l-degeneracy (2s = 2p).

ses::WavepacketSimulation make_simulation() {
    // 128^3: real-time stepping runs on the GPU engine (docs/GPU_PLAN.md G5);
    // the CPU session stays the double-precision truth for relax / measure /
    // surface meshing, synced on demand.
    //
    // Box +-80 Bohr at 256^3 (h = 0.625): holds the n <= 6 shell, box-critical
    // at n = 6 (the diffuse 6s is ~92% enclosed, its tail kissing the
    // u(R_box) = 0 wall; the structured 6d/6f/6g/6h are 97-99.9% held). Fixed
    // at 256^3 because the split-operator FFT demands a power-of-two size (512
    // would be 8x the work), so reaching n = 6 spends grid spacing: h grows
    // 0.5 -> 0.625. The startup atlas cross-check E_radial vs <H>_grid audits h
    // on every launch. The full m-resolved n <= 6 manifold is 91 state buffers
    // (~12 GB VRAM -- also box-critical). No host mirror: each orbital's double
    // field is transient (synthesized, uploaded fp32, freed), and the atlas
    // dipole integrals reduce on the GPU straight from the resident buffers, so
    // host RAM stays flat through the build instead of holding 91 copies.
    const ses::Grid1D axis{-80.0, 80.0, 256};
    const ses::Grid3D grid{axis, axis, axis};
    return ses::WavepacketSimulation{ses::WavepacketSimulation::Config{
        grid,
        ses::regularized_coulomb_potential(grid, 1.0, ses::Vec3d{}),
        ses::Vec3d{3.0, 0.0, 0.0},  // r0: beside the nucleus
        ses::Vec3d{1.5, 1.5, 1.5},  // sigma
        ses::Vec3d{0.0, 0.4, 0.0},  // k0: tangential kick
        0.04,                       // dt
    }};
}

constexpr int kStepsPerTick = 1;
// Backlog cap: a stalled paint cannot spiral (time is credited at execution,
// dropped ticks drop cleanly). Throughput note (measured on the P5000): the
// laser path saturates the GPU at ~26 steps/s (~38 ms/step at 256^3 -- the
// hand-rolled FFT's uncoalesced y/z passes), so raising this cap does NOT
// raise the sim rate (verified 8 vs 32: 1.02 vs 1.04 au/s); it only lengthens
// the per-paint block. 8 keeps paints ~300 ms. The real lever is the M4
// VkFFT swap (coalesced transposes).
constexpr int kMaxPendingGpuSteps = 8;
constexpr int kRelaxStepsPerTick = 1;
constexpr double kRelaxDtau = 0.05;
constexpr int kTickMs = 16;
constexpr double kIsoFraction = 0.25;
constexpr double kMeasureSigma = 0.625;  // position measurement resolution (Bohr):
                                         // the sharpest a h = 0.625 grid resolves
                                         // without aliasing (smaller needs finer h)
// Display decay rate: the TRUE Einstein-A lifetime is ~1e8 a.u. (unwatchable);
// this gives tau_display ~ 8 a.u. (~3 s of wall time). The title reports the
// true lifetime and the acceleration factor honestly.
constexpr double kDecayGammaDisplay = 0.125;
constexpr double kHaToEv = 27.211386;  // 1 Hartree in eV (physicist-facing display)
constexpr double kAbsorbWidth = 10.0;  // Bohr: boundary absorber layer thickness
                                       // (interior +-70 Bohr stays untouched --
                                       // clears the n<=6 states; real-time only)
// T3 laser: E0 is derived from a TARGET Rabi frequency over the computed
// dipole matrix element (Omega = E0 |<2p|z|1s>|). Omega = 0.04 keeps the drive
// well under the ~0.35 Ha 1s->2p gap (bare Coulomb on the grid; RWA-ish
// two-level flopping) while a full flop (2 pi / Omega ~ 157 au) takes seconds at
// the laser tick rate. The carrier is tuned to the GRID resonance -- the cooled
// 1s <H>, ~0.35 above 2p, not the textbook 0.375 label -- see toggle_laser.
constexpr double kRabiTargetOmega = 0.04;
constexpr int kLaserStepsPerTick = 6;  // the pump demo runs hotter than 1x

// ---- isosurface (mesh) shaders ----


enum class ViewMode { Cloud, Surface };
enum class Stepping { RealTime, Relaxing, RelaxingExcited };
enum class LaserPol { Off, Z, X };

constexpr int kAtlasMontageFrames = 3;  // frames each synthesized orbital shows
constexpr int kAtlasPairsPerFrame = 4;  // dipole pairs evaluated per paint

class Viewport : public QRhiWidget {
public:
    explicit Viewport(QWidget* parent = nullptr)
        : QRhiWidget(parent), sim_(make_simulation()) {
        setApi(QRhiWidget::Api::Vulkan);  // before the widget is first backed
        setFocusPolicy(Qt::StrongFocus);
        remesh();
        stage_volume();
        connect(&timer_, &QTimer::timeout, this, &Viewport::tick);
        timer_.start(kTickMs);
    }

protected:
    // GPU stepping covers the Cloud view in BOTH real and imaginary time
    // (G7); measure and surface meshing run on the CPU double session,
    // synced through the single cpu_is_truth_ invariant.
    bool use_gpu_path() const { return gpu_ok_ && mode_ == ViewMode::Cloud; }

    // The compute half of a frame runs BEFORE the widget's own QRhi frame is
    // recorded: the engine drives its own offscreen frames (beginOffscreenFrame
    // is illegal while the widget frame is active). This override is the QRhi
    // analog of "GPU stepping happens in paintGL where the context is current"
    // -- once per paint, even while paused (Key E works paused).
    void paintEvent(QPaintEvent* e) override {
        if (rhi_ready_) {
            run_gpu_frame();
            render_scene_offscreen();  // the whole scene, in ses_vk
        }
        QRhiWidget::paintEvent(e);
    }

    // The widget's QRhi (and thus the adopted VkDevice) is about to go away:
    // the core must tear down EVERYTHING it created on that device first.
    // Simulation state on the GPU dies with it -- the fatal-on-rhi-change
    // guard in initialize() keeps the policy honest (no silent migration).
    void releaseResources() override {
        blit_pipe_.reset();
        blit_srb_.reset();
        scene_wrap_.reset();
        scene_wrapped_img_ = VK_NULL_HANDLE;
        blit_sampler_.reset();
        vk_renderer_.destroy();
        vk_renderer_ready_ = false;
        engine_.destroy();
        vk_ctx_.destroy();
        gpu_ok_ = false;
    }

    // Build the RENDER resources (pipelines, samplers, UBOs, static vertex
    // buffers, the phase LUT). Called by QRhiWidget once the backing QRhi
    // exists, and again on every resize -- the guard makes re-entry a no-op
    // (nothing here depends on the surface size; the projection is per-frame).
    // COMPUTE setup (the engine) is deferred to init_compute(): the widget
    // frame is ACTIVE during initialize(), and the engine drives its own
    // offscreen frames, which are illegal while a frame is being recorded.
    void initialize(QRhiCommandBuffer* cb) override {
        if (rhi_ready_) {
            if (rhi() != rhi_) {
                fatal_rhi_error("QRhi changed",
                                QStringLiteral("The widget was moved to another window; "
                                               "GPU state cannot be migrated."));
            }
            return;
        }
        rhi_ = rhi();

        // Plan B: the scene renders in ses_vk. Qt's whole render layer is
        // one sampler + one fullscreen-blit pipeline (built lazily in
        // render() once the scene image exists and can seed the SRB).
        blit_sampler_.reset(rhi_->newSampler(
            QRhiSampler::Nearest, QRhiSampler::Nearest, QRhiSampler::None,
            QRhiSampler::ClampToEdge, QRhiSampler::ClampToEdge,
            QRhiSampler::ClampToEdge));
        if (!blit_sampler_->create()) {
            fatal_rhi_error("render resources",
                            QStringLiteral("blit sampler create failed"));
        }
        Q_UNUSED(cb);

        mesh_dirty_ = true;
        volume_dirty_ = true;
        rhi_ready_ = true;
    }

    // COMPUTE setup, deferred to the first run_gpu_frame() (outside any widget
    // frame): the GPU propagation engine (fp32 transcription of the tested CPU
    // tables; verified by sesolver_vkcheck), atlas precision, atom solve,
    // projection index, absorber mask. Falls back to CPU stepping on failure.
    void init_compute() {
        // Adopt the QRhiWidget's Vulkan device into the framework-free core:
        // Khronos handles cross the boundary, nothing else. Qt stays the
        // device provider + presentation layer; the physics runs in ses_vk.
        const auto* h =
            static_cast<const QRhiVulkanNativeHandles*>(rhi_->nativeHandles());
        const bool adopted =
            h != nullptr && h->inst != nullptr &&
            vk_ctx_.adopt(h->inst->vkInstance(), h->physDev, h->dev,
                          h->gfxQueueFamilyIdx, h->gfxQueue) ==
                ses_vk::Boot::ok;
        // The scene renderer needs only the DEVICE, not the engine: the CPU
        // fallback path (engine init failure) still renders.
        vk_renderer_ready_ =
            adopted && vk_renderer_.initialize(vk_ctx_, sim_.grid(),
                                               ses_shell::app_render_blobs());
        if (!vk_renderer_ready_) {
            fatal_rhi_error("render resources",
                            QStringLiteral("ses_vk renderer initialization "
                                           "failed (see stderr)"));
        }
        gpu_ok_ =
            adopted &&
            engine_.initialize(vk_ctx_, sim_.grid(),
                               ses_shell::app_engine_blobs(sim_.grid().x.n),
                               sim_.propagator().half_potential_phase(),
                               sim_.propagator().kinetic_phase(),
                               sim_.psi().data());
        if (gpu_ok_) {
            // Pick the atlas storage precision from free VRAM. The fp32 n<=6
            // manifold (91 x 256^3 complex-fp32 ~= 12 GB) oversubscribes a small
            // card, and WDDM then pages it into system RAM -- host RAM balloons
            // and the frame rate collapses. fp16 halves it (~6 GB). Headroom
            // covers the live working set + textures + the unpack scratch +
            // driver; an unmeasurable budget keeps fp32 (the big-VRAM default).
            constexpr std::int64_t kVramHeadroomBytes = 2LL * 1024 * 1024 * 1024;
            const std::int64_t bytes_per_state =
                static_cast<std::int64_t>(sim_.grid().size()) * 2 *
                static_cast<std::int64_t>(sizeof(float));
            bool atlas_fits = true;
            atom_.set_precision(ses::choose_state_precision(
                query_free_vram_bytes(), kNumStates, bytes_per_state,
                kVramHeadroomBytes, &atlas_fits));
            std::fprintf(stderr, "vram: atlas precision = %s%s\n",
                         atom_.precision() == ses::GpuPrecision::Fp16 ? "fp16 (half)"
                                                                     : "fp32",
                         atlas_fits ? ""
                                    : "  [WARNING: even fp16 is tight -- consider a "
                                      "smaller box or manifold]");
            // Imaginary-time weights from the tested CPU relaxer (G7).
            // These QRhi ports are FALLIBLE (buffer/pipeline create can fail
            // under VRAM pressure, unlike the void GL uploads); a failed relax
            // table would silently run REAL-TIME phase tables in imaginary
            // time, so failure demotes to the CPU fallback path wholesale.
            const ses::ImaginaryTimePropagator3D relaxer{sim_.grid(), sim_.potential(),
                                                         kRelaxDtau};
            if (!engine_.set_relax_tables(relaxer.half_potential_weight(),
                                          relaxer.kinetic_weight(), kRelaxDtau,
                                          sim_.grid().cell_volume()) ||
                !engine_.set_potential_gradient(sim_.potential())) {
                std::fprintf(stderr, "engine: relax/gradient setup failed -- "
                                     "falling back to CPU stepping\n");
                gpu_ok_ = false;
                decay_on_ = false;
                atlas_done_ = true;
                return;
            }
            // T6/T7: solve the atom up front. The radial engine gets every
            // bound level to n = 10 (the full lifetime table, printed
            // below); the 3D tracked manifold (n <= 6, what the box holds)
            // is then synthesized chunked across frames so decay is armed
            // BY DEFAULT and every demo entry point is instant afterwards.
            atom_.solve_radial_atom(sim_.grid().x.xmax);
            // Orbital-free projection index (Phase 5): the static counting-sort
            // geometry, uploaded once. Populations then come from ONE project_psi
            // deposit pass (state-count independent) instead of a per-state
            // inner_with_psi over the resident atlas.
            {
                const ses::RadialBinIndex bin_idx =
                    ses::build_radial_bin_index(sim_.grid(), atom_.radial_grid());
                proj_ready_ = engine_.set_projection_index(
                    bin_idx.sorted_cell, bin_idx.bin_off, atom_.radial_grid().n,
                    atom_.radial_grid().h(), 5);
                if (!proj_ready_) {
                    std::fprintf(stderr, "engine: projection index setup failed -- "
                                         "populations/decay/laser disabled\n");
                    decay_on_ = false;  // trials need projected populations
                }
            }
            synth_queue_.clear();
            for (int idx = 0; idx < kNumStates; ++idx) {
                synth_queue_.push_back(idx);
            }
            // Boundary absorber: build the mask (interior = 1) and upload it as
            // a (mask, 0) complex buffer so the tested elementwise multiply can
            // damp outgoing flux each real-time step (no wrap-around).
            {
                const std::vector<double> mask =
                    ses::absorbing_mask(sim_.grid(), kAbsorbWidth);
                ses::Field3D mf{sim_.grid()};
                for (std::size_t i = 0; i < mf.data().size(); ++i) {
                    mf.data()[i] = ses::Complex<double>{mask[i], 0.0};
                }
                mask_buf_ = engine_.create_state_buffer(mf.data());
            }
            // A field slider moved before this point stored its value but
            // could not upload the augmented half-potential (gpu_ok_ was
            // false): re-apply so the table matches the UI. Self-healing.
            if (efield_e0_ > 0.0 || bfield_b_ > 0.0) {
                upload_field_tables();
            }
        } else {
            decay_on_ = false;  // jump trials are GPU-only
            atlas_done_ = true;
        }
    }

    // The COMPUTE half of the old paintGL: engine stepping, atlas build,
    // measurement service, decay/laser trials. Runs once per paint, BEFORE the
    // widget frame (engine offscreen frames are illegal mid-frame).
    void run_gpu_frame() {
        if (!compute_init_done_) {
            init_compute();
            compute_init_done_ = true;
        }
        // T6/T7: the startup atlas build advances regardless of the view
        // mode -- a Tab to Surface during the startup window must not wedge
        // solving() forever (the build owns the psi buffer either way; the
        // Surface view simply does not display it while it runs).
        if (gpu_ok_ && !atlas_done_) {
            run_atlas_chunk();
            pending_gpu_steps_ = 0;
            if (gpu_title_due_) {
                gpu_title_due_ = false;
                refresh_title();
            }
            return;
        }
        if (use_gpu_path()) {
            if (cpu_is_truth_) {
                // The CPU state is authoritative here: refresh the brightness
                // normalizer from it (covers post-M collapse, post-R reset).
                double pk = 0.0;
                for (const ses::Complex<double>& z : sim_.psi().data()) {
                    pk = std::max(pk, ses::norm_sq(z));
                }
                if (pk > 0.0) {
                    peak_ = pk;
                }
                engine_.upload_state(sim_.psi().data());
                cpu_is_truth_ = false;
                volume_dirty_ = false;  // texture comes from the bridge now
                // Bridge immediately: with an empty step queue (paused R/M,
                // first frame) the block below would never refresh psi_tex_
                // and the screen would keep the stale (or undefined) cloud.
                write_display_texture();
            }
            // Projective ENERGY measurement (Key E): sample an eigenstate n
            // from P_n = |<phi_n|psi>|^2 over the tracked manifold and collapse
            // psi onto it. The incomplete-manifold deficit 1 - sum(P_n) is the
            // continuum outcome (n = -1): leave psi and say so. Reuses the same
            // GPU inner-product / collapse primitives as the decay jump; works
            // even while paused (it writes psi_tex_ itself).
            if (pending_energy_measure_) {
                pending_energy_measure_ = false;
                engine_.project_psi();
                std::vector<double> pop(static_cast<std::size_t>(kNumStates));
                for (int s = 0; s < kNumStates; ++s) {
                    pop[static_cast<std::size_t>(s)] = atom_.project_population(engine_, s);
                }
                std::uniform_real_distribution<double> uniform(0.0, 1.0);
                const int n = ses::sample_energy_eigenstate(pop, uniform(rng_));
                last_measured_index_ = n;  // >=0 eigenstate, -1 outside manifold
                if (n >= 0) {
                    atom_.collapse_onto(engine_, n);
                    write_display_texture();
                    last_measure_ =
                        QStringLiteral("%1 (E %2 eV)")
                            .arg(QLatin1String(kStateSpec[static_cast<std::size_t>(n)].name))
                            .arg(atom_.state_energy(n) * kHaToEv, 0,
                                 'f', 3);
                } else {
                    last_measure_ = QStringLiteral("outside tracked manifold");
                }
                refresh_title();
            }
            if (pending_gpu_steps_ > 0) {
                if (stepping_ == Stepping::RealTime) {
                    if (gpu_title_due_) {
                        // GPU-reduced norm+peak (2 KB readback), taken BEFORE
                        // enqueueing new steps so the implicit sync waits
                        // only on long-finished work.
                        const ses_vk::Engine::NormPeak np = engine_.norm_and_peak();
                        norm_display_ = np.sum;
                        if (np.peak > 0.0) {
                            peak_ = np.peak;  // brightness tracks the cloud
                        }
                        // fp32 drift renormalization: the split-operator is
                        // unitary in exact arithmetic; pinning the norm back
                        // to 1 removes pure numerical decay.
                        if (np.sum > 0.0 && std::abs(np.sum - 1.0) > 1e-4) {
                            engine_.scale(static_cast<float>(1.0 / std::sqrt(np.sum)));
                        }
                        // Radiation: the semiclassical radiated power from the
                        // oscillating dipole, P = (2/3)alpha^3|<grad V>|^2, via
                        // the GPU mean-force reduction (a 4 KB readback). ~0 for
                        // a stationary eigenstate, nonzero for a superposition.
                        radiated_power_ = ses::larmor_power(engine_.mean_force());
                    }
                    if (laser_pol_ != LaserPol::Off) {
                        // T3: resonant dipole drive. t0 is the same clock
                        // that credits gpu_time_, so the carrier phase
                        // cos(w t) stays continuous across batches/pauses.
                        const ses::DipoleDrive d{laser_axis(), laser_e0_, laser_omega_};
                        engine_.driven_step(d, sim_.time() + gpu_time_, sim_.dt(),
                                            pending_gpu_steps_);
                    } else if (bfield_b_ > 0.0) {
                        // Magnetic field along bfield_axis_: the PROPER minimal-
                        // coupling solve. psi evolves under H = H0 + (E z) +
                        // (B/2)L_axis + (B^2/8)rho_perp^2 -- the static E and the
                        // diamagnetic term are already folded into the
                        // half-potential (upload_field_tables), the paramagnetic
                        // L_axis is the exact three-shear rotation. Because it is
                        // the combined solve, crossed E(z)-B(x/y) is genuine.
                        engine_.magnetic_step(bfield_axis_,
                                              0.5 * bfield_b_ * (0.5 * sim_.dt()),
                                              pending_gpu_steps_);
                    } else {
                        // The static E-field (if any) is folded into the
                        // half-potential -- exact, and equal to the old omega=0
                        // dipole kick -- so a plain step polarizes / field-
                        // ionizes correctly.
                        engine_.step(pending_gpu_steps_);
                    }
                    // Time is credited where steps EXECUTE, so a stalled or
                    // occluded paint cannot desync the clock from the state.
                    gpu_time_ += pending_gpu_steps_ * sim_.dt();

                    // Boundary absorber (real-time only): damp outgoing/ionized
                    // flux at the walls so it leaves instead of wrapping around
                    // the periodic FFT box. Interior mask = 1, so the bound atom
                    // is untouched; imaginary-time relaxation never runs this.
                    if (mask_buf_ >= 0) {
                        engine_.apply_mask(mask_buf_);
                    }

                    // Orbital-free populations (Phase 5): ONE deposit pass on the
                    // post-step psi, shared by the decay and laser readouts this
                    // title tick (state-count independent) -- replaces the
                    // per-state inner_with_psi over the resident atlas.
                    if (gpu_title_due_ && proj_ready_ &&
                        ((decay_on_ && !atom_.channels().empty()) ||
                         laser_pol_ != LaserPol::Off)) {
                        engine_.project_psi();
                    }

                    // T4/T5/T7: competing-channels Poisson trials over the
                    // whole tracked manifold. The exponential is memoryless,
                    // so trials run on the TITLE cadence with the sim time
                    // accumulated since the last trial (identical statistics,
                    // 13 GPU reductions every ~10 ticks instead of per
                    // frame). Selection arithmetic is the core-tested
                    // pick_decay_channel; a jump collapses onto the fired
                    // channel's destination ("photon out").
                    if (decay_on_ && !atom_.channels().empty()) {
                        decay_accum_dt_ += pending_gpu_steps_ * sim_.dt();
                        if (gpu_title_due_) {
                            std::array<double, kNumStates> pop{};
                            for (int s = 1; s < kNumStates; ++s) {
                                pop[s] = atom_.project_population(engine_, s);
                            }
                            std::vector<double> rates(atom_.channels().size());
                            for (std::size_t c = 0; c < atom_.channels().size(); ++c) {
                                rates[c] = atom_.channels()[c].gamma_display *
                                           pop[atom_.channels()[c].from];
                            }
                            std::uniform_real_distribution<double> uniform(0.0, 1.0);
                            const ses::ChannelPick pick = ses::pick_decay_channel(
                                rates, decay_accum_dt_, uniform(rng_), uniform(rng_));
                            decay_accum_dt_ = 0.0;
                            if (pick.channel >= 0) {
                                const ShellChannel& ch =
                                    atom_.channels()[static_cast<std::size_t>(pick.channel)];
                                atom_.collapse_onto(engine_, ch.to);
                                flash_ticks_ = 25;
                                ++photon_count_;
                                last_jump_ = QStringLiteral("%1->%2").arg(
                                    QLatin1String(kStateSpec[ch.from].name),
                                    QLatin1String(kStateSpec[ch.to].name));
                                refresh_title();
                            }
                        }
                    }

                    // T3: live populations for the title readout (and the
                    // Rabi peak the selftest asserts on), on the title
                    // cadence -- two 2 KB reductions every ~10 ticks.
                    if (laser_pol_ != LaserPol::Off && gpu_title_due_ &&
                        manifold_ready()) {
                        pop_excited_ = atom_.project_population(engine_, kP2Z);
                        pop_ground_ = atom_.project_population(engine_, kS1);
                        rabi_peak_ = std::max(rabi_peak_, pop_excited_);
                    }
                } else {
                    // GPU imaginary time (G7/T1): renormalized every step;
                    // the ITP estimator gives the convergence readout free.
                    // The excited flavor deflates the cached ground state.
                    const ses_vk::Engine::RelaxStats stats =
                        (stepping_ == Stepping::RelaxingExcited &&
                         !relax_deflate_.empty())
                            ? engine_.relax_deflated_step(relax_deflate_,
                                                          pending_gpu_steps_)
                            : engine_.relax_step(pending_gpu_steps_);
                    relax_energy_display_ = stats.energy;
                    if (stats.peak > 0.0) {
                        peak_ = stats.peak;
                    }
                    norm_display_ = 1.0;  // pinned by per-step renormalization

                    // T7: relaxation auto-completes. When the ITP energy
                    // readout plateaus the state has converged; return to
                    // real time so the lifetimes ACT (a prepared 2p should
                    // decay, not sit in imaginary time forever).
                    if (gpu_title_due_) {
                        if (std::abs(stats.energy - relax_prev_energy_) < 5e-5) {
                            ++relax_plateau_;
                        } else {
                            relax_plateau_ = 0;
                        }
                        relax_prev_energy_ = stats.energy;
                        if (relax_plateau_ >= 12) {  // ~2 s of stable readout
                            relax_plateau_ = 0;
                            stepping_ = Stepping::RealTime;
                            free_deflation_buffers();  // converged -> free the phi
                        }
                    }
                }
                pending_gpu_steps_ = 0;
                write_display_texture();
                volume_dirty_ = false;
                if (gpu_title_due_) {
                    gpu_title_due_ = false;
                    refresh_title();
                }
            }
        }
    }

    // The DRAW half of a frame: the shell computes the per-frame inputs
    // (camera, view mode, brightness, staged uploads) and SceneRenderer
    // records the resource updates + the one render pass.
    // Render the whole scene in ses_vk (called at the END of run_gpu_frame,
    // outside any QRhi frame): resize the offscreen target to the widget's
    // pixel size, assemble the per-frame inputs, submit.
    void render_scene_offscreen() {
        if (!vk_renderer_ready_) {
            return;
        }
        const qreal dpr = devicePixelRatio();
        const std::uint32_t w =
            static_cast<std::uint32_t>(std::max<qreal>(1, width() * dpr));
        const std::uint32_t h =
            static_cast<std::uint32_t>(std::max<qreal>(1, height() * dpr));
        if (!vk_renderer_.resize(w, h)) {
            return;
        }

        ses_vk::SceneRenderer::FrameInput in;
        in.cloud = (mode_ == ViewMode::Cloud);
        in.azimuth = azimuth_;
        in.elevation = elevation_;
        in.distance = distance_;
        in.peak = peak_;
        in.absorbance = absorbance_;
        // Photon flash: a brief warm background right after a quantum jump.
        if (flash_ticks_ > 0) {
            in.flash = static_cast<float>(flash_ticks_) / 25.0f;
            --flash_ticks_;
        }
        // Temporal accumulation: keep averaging only while NOTHING changed
        // -- camera, display params, view mode, flash, and the psi volume
        // itself (any bridge write this frame resets the history).
        in.frame_index = static_cast<float>(frame_index_++ % 4096);
        const bool scene_static =
            !volume_written_ && azimuth_ == acc_prev_.azimuth &&
            elevation_ == acc_prev_.elevation &&
            distance_ == acc_prev_.distance && peak_ == acc_prev_.peak &&
            absorbance_ == acc_prev_.absorbance && in.flash == 0.0f &&
            acc_prev_.flash == 0.0f && in.cloud == acc_prev_.cloud;
        in.accumulate = scene_static;
        acc_prev_ = {azimuth_, elevation_, distance_, peak_, absorbance_,
                     in.flash, in.cloud};
        volume_written_ = false;
        // The psi display volume: the engine's bridge image on the GPU path;
        // null lets the renderer fall back to its CPU-staged texture.
        if (gpu_ok_) {
            in.psi_volume = engine_.volume_view();
        }
        if (in.cloud) {
            if (volume_dirty_) {
                // CPU staging only -- the bridge owns the volume on the GPU
                // path, and until compute init has been ATTEMPTED the 268 MB
                // fallback texture must not be allocated (it would be orphaned
                // one frame later and would deflate the VRAM-budget probe).
                if (compute_init_done_ && !gpu_ok_) {
                    in.volume_staging = &psi_staging_;
                }
                volume_dirty_ = false;
            }
        } else if (mesh_dirty_) {
            in.mesh = &mesh_;
            in.mesh_colors = &colors_;
            mesh_dirty_ = false;
        }
        vk_renderer_.render(in);
    }

    // Qt's entire remaining draw: one fullscreen triangle sampling the
    // ses_vk scene image (imported via createFrom; re-imported when the
    // offscreen target is recreated on resize).
    void render(QRhiCommandBuffer* cb) override {
        const VkImage scene_img = vk_renderer_.color_image();
        if (scene_img == VK_NULL_HANDLE) {
            // Nothing rendered yet: clear only.
            cb->beginPass(renderTarget(), QColor(10, 13, 23),
                          {1.0f, 0}, nullptr);
            cb->endPass();
            return;
        }
        if (scene_img != scene_wrapped_img_) {
            scene_wrap_.reset(rhi_->newTexture(
                QRhiTexture::RGBA8,
                QSize(static_cast<int>(vk_renderer_.width()),
                      static_cast<int>(vk_renderer_.height()))));
            QRhiTexture::NativeTexture src;
            src.object = quint64(reinterpret_cast<std::uintptr_t>(scene_img));
            src.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            if (!scene_wrap_->createFrom(src)) {
                std::fprintf(stderr, "blit: scene image import failed\n");
                scene_wrap_.reset();
                scene_wrapped_img_ = VK_NULL_HANDLE;
                return;
            }
            scene_wrapped_img_ = scene_img;
            blit_srb_.reset(rhi_->newShaderResourceBindings());
            blit_srb_->setBindings({QRhiShaderResourceBinding::sampledTexture(
                0, QRhiShaderResourceBinding::FragmentStage, scene_wrap_.data(),
                blit_sampler_.data())});
            if (!blit_srb_->create()) {
                std::fprintf(stderr, "blit: SRB create failed\n");
                blit_srb_.reset();
                return;
            }
            if (blit_pipe_.isNull()) {
                auto load_qsb = [](const char* path) {
                    QFile f{QLatin1String(path)};
                    if (!f.open(QIODevice::ReadOnly)) {
                        return QShader();
                    }
                    return QShader::fromSerialized(f.readAll());
                };
                blit_pipe_.reset(rhi_->newGraphicsPipeline());
                blit_pipe_->setShaderStages({
                    {QRhiShaderStage::Vertex,
                     load_qsb(":/shaders/blit.vert.qsb")},
                    {QRhiShaderStage::Fragment,
                     load_qsb(":/shaders/blit.frag.qsb")},
                });
                blit_pipe_->setVertexInputLayout({});
                blit_pipe_->setShaderResourceBindings(blit_srb_.data());
                blit_pipe_->setRenderPassDescriptor(
                    renderTarget()->renderPassDescriptor());
                blit_pipe_->setTopology(QRhiGraphicsPipeline::Triangles);
                blit_pipe_->setDepthTest(false);
                blit_pipe_->setDepthWrite(false);
                if (!blit_pipe_->create()) {
                    std::fprintf(stderr, "blit: pipeline create failed\n");
                    blit_pipe_.reset();
                    return;
                }
            }
        }
        if (blit_pipe_.isNull() || blit_srb_.isNull()) {
            return;
        }
        cb->beginPass(renderTarget(), QColor(0, 0, 0), {1.0f, 0}, nullptr);
        cb->setGraphicsPipeline(blit_pipe_.data());
        const QSize px = renderTarget()->pixelSize();
        cb->setViewport(QRhiViewport(0, 0, static_cast<float>(px.width()),
                                     static_cast<float>(px.height())));
        cb->setShaderResources(blit_srb_.data());
        cb->draw(3);
        cb->endPass();
    }

    void mousePressEvent(QMouseEvent* e) override { last_pos_ = e->position(); }

    void mouseMoveEvent(QMouseEvent* e) override {
        const QPointF delta = e->position() - last_pos_;
        last_pos_ = e->position();
        azimuth_ -= 0.01 * delta.x();
        elevation_ += 0.01 * delta.y();
        elevation_ = std::clamp(elevation_, -1.5, 1.5);
        update();
    }

    void wheelEvent(QWheelEvent* e) override {
        distance_ *= std::pow(0.999, e->angleDelta().y());
        distance_ = std::clamp(distance_, 4.0, 300.0);  // out past the +-80 box
                                                         // (dynamic zfar follows)
        update();
    }

public:
    // Control entry points, shared by the keyboard and the toolbar buttons.
    void toggle_pause() {
        paused_ = !paused_;
        after_control();
    }
    void set_real_time() {
        stepping_ = Stepping::RealTime;
        after_control();
    }
    void set_relaxing() {
        stepping_ = Stepping::Relaxing;
        relax_plateau_ = 0;
        relax_prev_energy_ = 0.0;
        if (!use_gpu_path()) {
            ensure_cpu_current();  // CPU relax (Surface view / no GPU)
        }
        after_control();
    }
    void reset_simulation() {
        if (solving()) {
            return;  // the startup atlas build owns the GPU state
        }
        sim_ = make_simulation();
        stepping_ = Stepping::RealTime;
        free_deflation_buffers();  // drop any owned deflation phi
        laser_pol_ = LaserPol::Off;  // reset returns to the vanilla packet demo
        bfield_b_ = 0.0;             // and to no magnetic field
        upload_field_tables();    // restore the base half-potential
        cpu_is_truth_ = true;  // GPU state discarded with the reset
        gpu_time_ = 0.0;
        pending_gpu_steps_ = 0;
        stage_active_view();
        after_control();
    }
    // Soft position measurement: sample from |psi|^2 (RNG lives here in the
    // shell; core takes the uniform draw) and let the sharpened packet
    // re-evolve.
    void measure_now() {
        if (solving()) {
            return;
        }
        ensure_cpu_current();
        std::uniform_real_distribution<double> uniform(0.0, 1.0);
        sim_.measure(uniform(rng_), kMeasureSigma);
        stepping_ = Stepping::RealTime;
        stage_active_view();
        after_control();
    }

    // Projective ENERGY measurement (the energy-basis analogue of M): request
    // a collapse onto an energy eigenstate sampled by |<phi_n|psi>|^2. The GPU
    // reductions + collapse-copy need a current context, so the actual work is
    // deferred to paintGL (see pending_energy_measure_). Needs the manifold.
    void measure_energy_now() {
        if (solving() || !use_gpu_path() || !manifold_ready()) {
            return;
        }
        pending_energy_measure_ = true;
        stepping_ = Stepping::RealTime;  // observe, then let H evolve it
        laser_pol_ = LaserPol::Off;
        update();
    }
    void toggle_view_mode() {
        if (solving()) {
            return;
        }
        mode_ = (mode_ == ViewMode::Cloud) ? ViewMode::Surface : ViewMode::Cloud;
        // Re-stage for the newly selected mode: its data may be stale (tick
        // only stages the active mode, and we may be paused).
        if (mode_ == ViewMode::Surface) {
            ensure_cpu_current();  // meshing reads the CPU field
            if (stepping_ == Stepping::RelaxingExcited) {
                stepping_ = Stepping::Relaxing;  // deflation is GPU-only (v1)
            }
            laser_pol_ = LaserPol::Off;  // the drive is GPU-only too
            decay_on_ = false;  // so are the jump trials: OFF beats a lying title
        }
        stage_active_view();
        after_control();
    }

    // Transitions arc T1: relax into the z-aligned first excited state.
    // The ground state is computed and cached on first use (a few seconds,
    // one-time); the z-odd seed keeps the whole flow in the odd-parity
    // sector, so it converges to the 2p_z-like state deterministically.
    void relax_to_excited() {
        start_excited_relax(make_axis_odd_seed(2), QStringLiteral("2p"), false);
    }

    // T5: relax into 2s -- the radial node appears live. With decay ON it
    // then just SITS there: A(2s -> 1s) ~ 0 makes it metastable, from our
    // own matrix elements. The 2p triplet is cached and deflated too (2s
    // sits ABOVE it), see start_excited_relax.
    void relax_to_2s() {
        start_excited_relax(
            ses::gaussian_wavepacket(sim_.grid(), ses::Vec3d{},
                                     ses::Vec3d{4.0, 4.0, 4.0}, ses::Vec3d{}),
            QStringLiteral("2s"), true);
    }

    // Transitions arc T4/T5: toggle spontaneous decay (quantum jumps) over
    // the whole tracked manifold. Every channel rate is einstein_a(gap,
    // |<f|r|i>|^2) from our own eigenstates; true lifetimes are ~1e8 a.u.,
    // so the display accelerates ALL channels by one common factor (title
    // reports it honestly and relative lifetimes stay physical).
    void toggle_decay() {
        if (!gpu_ok_ || solving()) {
            return;
        }
        if (!decay_on_) {
            if (mode_ != ViewMode::Cloud) {
                mode_ = ViewMode::Cloud;  // jump trials run on the GPU path only
            }
            if (!atom_.prepare_manifold_cache(engine_, kDecayGammaDisplay)) {
                return;
            }
            decay_accum_dt_ = 0.0;  // no hazard accrues while decay is off
        }
        decay_on_ = !decay_on_;
        after_control();
    }

    // Selftest access: the computed Einstein A for a channel (0 if absent)
    // and the cached eigenenergies.
    double channel_a(int from, int to) const {
        for (const ShellChannel& c : atom_.channels()) {
            if (c.from == from && c.to == to) {
                return c.a_true;
            }
        }
        return 0.0;
    }
    bool solving() const { return gpu_ok_ && !atlas_done_; }
    // Ready only once the FULL table is assembled (channels_ fills
    // incrementally during the pair phase -- do not race it).
    bool manifold_ready() const { return atlas_done_ && !atom_.channels().empty(); }

    // T7: instantly excite an n = 3 state (cycles through a small set) and
    // watch the CASCADE: e.g. 3d -> 2p (photon) -> 1s (photon).
    void excite_n3() {
        if (!gpu_ok_ || solving()) {
            return;
        }
        if (mode_ != ViewMode::Cloud) {
            mode_ = ViewMode::Cloud;
        }
        if (!atom_.prepare_manifold_cache(engine_, kDecayGammaDisplay)) {
            return;
        }
        static constexpr int kCycle[] = {k3DZ0, k4F0, k3S, k4S};
        const int idx = kCycle[excite_cycle_++ % 4];
        atom_.collapse_onto(engine_, idx);
        cpu_is_truth_ = false;  // the GPU state is ahead now
        stepping_ = Stepping::RealTime;
        after_control();
    }
    double state_energy(int idx) const { return atom_.state_energy(idx); }

    // Transitions arc T3: cycle the laser off -> Z-pol -> X-pol -> off. The
    // carrier w comes from OUR spectrum (the cached ITP energies) and E0
    // from a target Rabi frequency over OUR dipole matrix element, so the
    // pump is first-principles end to end. Z pumps 1s -> 2p_z (watch P(2pz)
    // flop); X pumps the orthogonal 2p_x instead, so the monitored P(2pz)
    // stays flat -- the selection rule, live.
    void toggle_laser() {
        if (!gpu_ok_ || solving()) {
            return;  // the drive runs on the GPU path only
        }
        if (laser_pol_ == LaserPol::Off) {
            if (mode_ != ViewMode::Cloud) {
                mode_ = ViewMode::Cloud;
            }
            if (!manifold_ready()) {
                return;
            }
            // Drive the GRID resonance, not the textbook label: bare Coulomb on
            // the coarse grid leaves the 1s a cusp gap, and a RELAXED 1s sits
            // ~0.03 Ha below a synthesized one, so the resonance is the COOLED
            // 1s <H> (~ -0.478 Ha), not the radial -0.5. Use the live relaxation
            // energy once the atom has cooled to the deeply-bound 1s; fall back
            // to the synthesized 1s grid energy (from the h-audit) otherwise. 2p
            // has no cusp gap, so its radial energy is exact.
            const double e_1s = (relax_energy_display_ < -0.35)
                                    ? relax_energy_display_
                                    : (grid_energy_1s_ != 0.0 ? grid_energy_1s_
                                                              : atom_.state_energy(kS1));
            laser_omega_ = atom_.state_energy(kP2Z) - e_1s;
            laser_e0_ = atom_.dipole_z() > 0.0 ? kRabiTargetOmega / atom_.dipole_z() : 0.0;
            rabi_peak_ = 0.0;
            laser_pol_ = LaserPol::Z;
            stepping_ = Stepping::RealTime;  // the drive lives in real time
        } else if (laser_pol_ == LaserPol::Z) {
            laser_pol_ = LaserPol::X;
        } else {
            laser_pol_ = LaserPol::Off;
        }
        after_control();
    }

    // Static uniform E-field magnitude along +z (atomic units); 0 = off. Driven
    // by the toolbar slider. Acts in the GPU cloud/real-time path; the laser, if
    // on, takes precedence in the stepping branch.
    void set_efield_e0(double e0) {
        efield_e0_ = e0;
        upload_field_tables();  // fold E*z into the half-potential (with diamag if B on)
        if (e0 > 0.0 && !solving()) {
            stepping_ = Stepping::RealTime;  // let the field actually act
        }
        refresh_title();
        update();
    }

    // Magnetic field strength (au) along +z; 0 = off. PROPER minimal-coupling
    // solve: psi evolves under H = H0 + (B/2)L_z + (B^2/8)rho^2. Uploading the
    // diamagnetic-augmented half-potential here means the per-frame
    // magnetic_step only has to add the paramagnetic rotation.
    void set_bfield_b(double b) {
        bfield_b_ = b;
        upload_field_tables();
        if (b > 0.0 && !solving()) {
            stepping_ = Stepping::RealTime;
        }
        refresh_title();
        update();
    }

    // Cycle the field axis z -> x -> y -> z. The diamagnetic term is
    // perpendicular to the axis, so the half-potential table is rebuilt.
    void toggle_bfield_axis() {
        bfield_axis_ = (bfield_axis_ == 2) ? 0 : (bfield_axis_ == 0 ? 1 : 2);
        upload_field_tables();
        refresh_title();
        update();
    }
    int bfield_axis() const { return bfield_axis_; }

    // Rebuild the GPU half-potential for the current static fields:
    //   V  +  E z (Stark, diagonal)  +  (B^2/8) rho_perp^2 (diamagnetic).
    // Both static-E and diamagnetic are diagonal in position, so folding them
    // in is exact (the old omega=0 dipole-kick equals this). The per-frame
    // magnetic_step then only adds the paramagnetic rotation. Crossed E-B
    // (E along z, B along x/y) is now a genuine combined solve.
    void upload_field_tables() {
        if (!gpu_ok_) {
            return;
        }
        if (efield_e0_ > 0.0 || bfield_b_ > 0.0) {
            const ses::Grid3D& g = sim_.grid();
            std::vector<double> v = sim_.potential();
            if (efield_e0_ > 0.0) {
                for (int k = 0; k < g.z.n; ++k) {
                    const double ez = efield_e0_ * g.z.coord(k);
                    for (int j = 0; j < g.y.n; ++j) {
                        for (int i = 0; i < g.x.n; ++i) {
                            v[static_cast<std::size_t>(g.flat(i, j, k))] += ez;
                        }
                    }
                }
            }
            if (bfield_b_ > 0.0) {
                // Reuse the core diamagnetic (perpendicular to the field axis).
                const ses::MagneticPropagator3D mprop{g, v, sim_.dt(), bfield_b_,
                                                      bfield_axis_};
                v = mprop.effective_potential();
            }
            const ses::SplitOperator3D aug{g, v, sim_.dt()};
            engine_.set_half_potential(aug.half_potential_phase());
        } else {
            engine_.set_half_potential(sim_.propagator().half_potential_phase());
        }
    }

    long long photon_count() const { return photon_count_; }
    // Result of the most recent energy measurement: eigenstate index, -1 for
    // the outside-the-manifold outcome, -2 if none has run yet (selftest hook).
    int last_measured_index() const { return last_measured_index_; }
    // <z> of the current cloud (bridges the GPU state to the CPU session first);
    // a selftest hook for the Stark polarization along +z.
    double mean_z() {
        ensure_cpu_current();
        return ses::mean_position(sim_.psi()).z;
    }
    double peak_excited_population() const { return rabi_peak_; }

    // Verification hook (--dump-frame-near): place the camera at an explicit
    // distance -- inside the box (< ~80) exercises the volume proxy's
    // front-face culling, which only matters from an interior viewpoint.
    void debug_set_camera_distance(double d) {
        distance_ = std::clamp(d, 4.0, 300.0);
        update();
    }

    // Selftest hooks (magnetic Larmor precession): set psi to a cached
    // manifold eigenstate and probe another state's population. A field along
    // z rotates 2p_x -> 2p_y at omega_L = B/2, so P(2p_y) must rise -- proving
    // the field evolves psi itself, not just the display.
    void debug_prepare_state(int idx) {
        if (!manifold_ready() || idx < 0 || idx >= kNumStates) {
            return;
        }
        atom_.collapse_onto(engine_, idx);
        cpu_is_truth_ = false;
        stepping_ = Stepping::RealTime;
        after_control();
    }
    double probe_population(int idx) {
        if (!manifold_ready() || idx < 0 || idx >= kNumStates) {
            return 0.0;
        }
        engine_.project_psi();
        const double p = atom_.project_population(engine_, idx);
        return p;
    }

protected:
    ses::Vec3d laser_axis() const {
        return laser_pol_ == LaserPol::X ? ses::Vec3d{1.0, 0.0, 0.0}
                                         : ses::Vec3d{0.0, 0.0, 1.0};
    }

    void after_control() {
        refresh_title();
        update();
    }

    // The common launcher for excited-state relaxation demos (keys 3/4):
    // ensure the deflation targets are cached, then hand the seed to the
    // GPU deflated imaginary-time flow. The live flow must deflate EVERY
    // state below the target: 2s sits above the 2p triplet, and with only
    // 1s removed the fp32 parity leakage of the GPU FFT (~1e-7/step) grows
    // as e^{(E2s-E2p) tau} until the on-screen "2s" morphs into 2p within
    // minutes (adversarial-review finding). 2p itself is safe with {1s}:
    // its only competitors are the DEGENERATE other 2p's (gap ~1e-6).
    void start_excited_relax(const ses::Field3D& seed, const QString& label,
                             bool deflate_p_triplet) {
        if (!gpu_ok_ || solving()) {
            return;  // deflation runs on the GPU path only (v1)
        }
        if (mode_ != ViewMode::Cloud) {
            mode_ = ViewMode::Cloud;
        }
        if (!manifold_ready()) {
            return;
        }
        // Deflation set synthesized into OWNED transient fp32 buffers (no resident
        // atlas): freed at auto-complete / reset / the next relaxation.
        // relax_deflated_step reads them as fp32 phi every relax frame.
        free_deflation_buffers();
        relax_deflate_.push_back(atom_.synth_transient(engine_, kS1));
        if (deflate_p_triplet) {
            relax_deflate_.push_back(atom_.synth_transient(engine_, kP2X));
            relax_deflate_.push_back(atom_.synth_transient(engine_, kP2Y));
            relax_deflate_.push_back(atom_.synth_transient(engine_, kP2Z));
        }
        relax_deflate_owned_ = relax_deflate_;
        sim_.set_psi(seed);
        cpu_is_truth_ = true;
        relax_label_ = label;
        relax_plateau_ = 0;
        relax_prev_energy_ = 0.0;
        stepping_ = Stepping::RelaxingExcited;
        after_control();
    }

    ses::Field3D make_axis_odd_seed(int axis) const {  // 0 = x, 1 = y, 2 = z
        const ses::Grid3D& g = sim_.grid();
        ses::Field3D seed{g};
        for (int k = 0; k < g.z.n; ++k) {
            for (int j = 0; j < g.y.n; ++j) {
                for (int i = 0; i < g.x.n; ++i) {
                    const double c[3] = {g.x.coord(i), g.y.coord(j), g.z.coord(k)};
                    const double env = std::exp(
                        -(c[0] * c[0] + c[1] * c[1] + c[2] * c[2]) / (4.0 * 2.0 * 2.0));
                    seed(i, j, k) = ses::Complex<double>{c[axis] * env, 0.0};
                }
            }
        }
        ses::normalize(seed);
        return seed;
    }

    // Free VRAM in bytes via Vulkan's VK_EXT_memory_budget (heap budget minus
    // current usage, summed over device-local heaps), or ses::kVramUnknown when
    // the extension / entry points are absent. Physical-device property queries
    // only need the extension SUPPORTED, not enabled on the logical device.
    std::int64_t query_free_vram_bytes() {
        const QRhiVulkanNativeHandles* h =
            static_cast<const QRhiVulkanNativeHandles*>(rhi()->nativeHandles());
        if (h == nullptr || h->inst == nullptr || h->physDev == VK_NULL_HANDLE) {
            return ses::kVramUnknown;
        }
        QVulkanFunctions* f = h->inst->functions();
        uint32_t n_ext = 0;
        f->vkEnumerateDeviceExtensionProperties(h->physDev, nullptr, &n_ext, nullptr);
        std::vector<VkExtensionProperties> exts(n_ext);
        f->vkEnumerateDeviceExtensionProperties(h->physDev, nullptr, &n_ext, exts.data());
        bool budget = false;
        for (const VkExtensionProperties& e : exts) {
            if (std::strcmp(e.extensionName, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME) == 0) {
                budget = true;
                break;
            }
        }
        if (!budget) {
            return ses::kVramUnknown;
        }
        auto get_props2 = reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties2>(
            h->inst->getInstanceProcAddr("vkGetPhysicalDeviceMemoryProperties2"));
        if (get_props2 == nullptr) {
            get_props2 = reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties2>(
                h->inst->getInstanceProcAddr("vkGetPhysicalDeviceMemoryProperties2KHR"));
        }
        if (get_props2 == nullptr) {
            return ses::kVramUnknown;
        }
        VkPhysicalDeviceMemoryBudgetPropertiesEXT bud{};
        bud.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;
        VkPhysicalDeviceMemoryProperties2 props{};
        props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
        props.pNext = &bud;
        get_props2(h->physDev, &props);
        std::int64_t free_total = 0;
        bool any = false;
        for (uint32_t i = 0; i < props.memoryProperties.memoryHeapCount; ++i) {
            if ((props.memoryProperties.memoryHeaps[i].flags &
                 VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) == 0) {
                continue;
            }
            if (bud.heapBudget[i] > bud.heapUsage[i]) {
                free_total += static_cast<std::int64_t>(bud.heapBudget[i] - bud.heapUsage[i]);
            }
            any = true;
        }
        return any ? free_total : ses::kVramUnknown;
    }

    // Free the OWNED transient deflation buffers (synthesized at relax-start).
    void free_deflation_buffers() {
        for (int b : relax_deflate_owned_) {
            engine_.release_state(b);
        }
        relax_deflate_owned_.clear();
        relax_deflate_.clear();
    }

    // T7: advance the startup atlas build by one chunk (current context:
    // called from paintGL). Phase 1 synthesizes one orbital per visit and
    // SHOWS it (the montage); phase 2 evaluates dipole channel pairs; the
    // finale assembles the channel table and resumes the wavepacket.
    void run_atlas_chunk() {
        if (montage_hold_ > 0) {
            --montage_hold_;
            return;
        }
        if (!synth_queue_.empty()) {
            const int idx = synth_queue_.front();
            synth_queue_.erase(synth_queue_.begin());
            if (!atom_.radial_ready()) {
                atlas_done_ = true;  // no radial solve: give up gracefully
                return;
            }
            // Synthesize into a TRANSIENT fp32 buffer: the model captures its
            // grid norm (for populations), SHOW it (montage), audit, then FREE.
            // The orbital-free projection keeps no atlas, so the montage holds
            // ONE orbital at a time instead of accumulating all 91 (the old
            // startup VRAM ramp).
            double pk = 0.0;
            const int buf = atom_.synth_transient(engine_, idx, &pk);
            if (buf < 0) {
                atlas_done_ = true;  // GPU buffer alloc failed: give up gracefully
                return;
            }
            engine_.copy_into_psi(buf);  // show (fp32)
            // The h-audit: cross-check the 1D radial energy against the full 3D
            // spectral <H> for the resolution-critical 1s and the box-critical
            // 4s/5s/6s -- the ONLY states read back to the CPU (4 of 91).
            if (idx == kS1 || idx == k4S || idx == k5S || idx == k6S) {
                engine_.readback(readback_buf_);
                ses::Field3D f{sim_.grid()};
                for (std::size_t i = 0; i < f.data().size(); ++i) {
                    f.data()[i] = ses::Complex<double>{readback_buf_[2 * i],
                                                       readback_buf_[2 * i + 1]};
                }
                const double e_grid = ses::mean_energy(f, sim_.potential());
                if (idx == kS1) {
                    grid_energy_1s_ = e_grid;  // the laser's true (grid) resonance
                }
                std::fprintf(stderr,
                             "atlas: %-8s E_radial = %.6f Ha   <H>_grid = %.6f Ha\n",
                             kStateSpec[idx].name, atom_.state_energy(idx), e_grid);
            } else {
                std::fprintf(stderr, "atlas: %-8s E_radial = %.6f Ha\n",
                             kStateSpec[idx].name, atom_.state_energy(idx));
            }
            engine_.release_state(buf);  // TRANSIENT: freed after show + audit
            if (pk > 0.0) {
                peak_ = pk;
            }
            write_display_texture();
            volume_dirty_ = false;
            montage_hold_ = kAtlasMontageFrames;
            if (synth_queue_.empty()) {
                atom_.collect_channel_pairs();
            }
            return;
        }
        if (!atom_.pair_queue().empty()) {
            for (int c = 0; c < kAtlasPairsPerFrame && !atom_.pair_queue().empty(); ++c) {
                atom_.evaluate_channel_pair(engine_, atom_.pair_queue().back());
                atom_.pair_queue().pop_back();
            }
            if (atom_.pair_queue().empty()) {
                atom_.finalize_channel_table(kDecayGammaDisplay);
                // Free the channel-build 'from' cache; nothing else is resident
                // (montage + pairs were transient). Populations come from the
                // projection and collapse/deflation re-synthesize on demand -- NO
                // resident atlas -> ~1.2 GB runtime, and 512^3 is feasible.
                atom_.release_pair_cache(engine_);
                atlas_done_ = true;
                cpu_is_truth_ = true;  // resume the untouched wavepacket
                refresh_title();
            }
        }
    }

    void keyPressEvent(QKeyEvent* e) override {
        switch (e->key()) {
            case Qt::Key_Space:
                toggle_pause();
                break;
            case Qt::Key_1:
                set_real_time();
                break;
            case Qt::Key_2:
                set_relaxing();
                break;
            case Qt::Key_3:
                relax_to_excited();
                break;
            case Qt::Key_4:
                relax_to_2s();
                break;
            case Qt::Key_5:
                excite_n3();
                break;
            case Qt::Key_R:
                reset_simulation();
                break;
            case Qt::Key_M:
                measure_now();
                break;
            case Qt::Key_E:
                measure_energy_now();
                break;
            case Qt::Key_D:
                toggle_decay();
                break;
            case Qt::Key_L:
                toggle_laser();
                break;
            case Qt::Key_Tab:
                toggle_view_mode();
                break;
            case Qt::Key_BracketLeft:
                absorbance_ = std::max(0.1, absorbance_ / 1.3);
                after_control();
                break;
            case Qt::Key_BracketRight:
                absorbance_ = std::min(50.0, absorbance_ * 1.3);
                after_control();
                break;
            default:
                QRhiWidget::keyPressEvent(e);
                return;
        }
    }

private:
    void tick() {
        if (paused_) {
            return;
        }
        if (use_gpu_path()) {
            // Steps run in paintGL (context current there). Cap the backlog
            // so a stalled paint cannot spiral; time is credited at
            // execution, so dropped ticks drop cleanly. The laser demo steps
            // hotter so a Rabi flop fits in seconds of wall time.
            const int per_tick =
                (stepping_ == Stepping::RealTime && laser_pol_ != LaserPol::Off)
                    ? kLaserStepsPerTick
                    : kStepsPerTick;
            pending_gpu_steps_ =
                std::min(pending_gpu_steps_ + per_tick, kMaxPendingGpuSteps);
            if (++ticks_ % 10 == 0) {
                gpu_title_due_ = true;
            }
            update();
            return;
        }
        ensure_cpu_current();
        if (stepping_ == Stepping::RealTime) {
            sim_.advance(kStepsPerTick);
        } else {
            sim_.relax(kRelaxStepsPerTick, kRelaxDtau);
        }
        stage_active_view();
        if (++ticks_ % 10 == 0) {
            norm_display_ = ses::norm_sq(sim_.psi());
            refresh_title();
        }
        update();
    }

    // Pull the GPU-evolved state back into the CPU double session (fp32
    // precision at the handoff -- the display path's documented tradeoff).
    void ensure_cpu_current() {
        // Queued-but-unexecuted steps were never credited to gpu_time_;
        // discard them so they cannot fire later against a different state.
        pending_gpu_steps_ = 0;
        if (cpu_is_truth_ || !gpu_ok_) {
            return;
        }
        engine_.readback(readback_buf_);
        ses::Field3D f{sim_.grid()};
        for (std::size_t i = 0; i < f.data().size(); ++i) {
            f.data()[i] = ses::Complex<double>{readback_buf_[2 * i], readback_buf_[2 * i + 1]};
        }
        sim_.set_psi(f);
        cpu_is_truth_ = true;
    }

    void stage_active_view() {
        if (mode_ == ViewMode::Cloud) {
            if (use_gpu_path()) {
                return;  // paintGL uploads the state and bridges the texture
            }
            stage_volume();
            volume_dirty_ = true;
        } else {
            remesh();
            mesh_dirty_ = true;
        }
    }

    // ---- CPU staging (no GL context needed) ----

    void remesh() {
        mesh_ = ses::marching_cubes_at_fraction(sim_.density(), sim_.grid(), kIsoFraction);
        colors_ = ses::phase_colors(mesh_, sim_.psi());
    }

    // Pack complex psi into RG float pairs and track the density peak.
    void stage_volume() {
        const auto& field = sim_.psi().data();
        psi_staging_.resize(field.size() * 2);
        double peak = 0.0;
        for (std::size_t i = 0; i < field.size(); ++i) {
            psi_staging_[2 * i] = static_cast<float>(field[i].real());
            psi_staging_[2 * i + 1] = static_cast<float>(field[i].imag());
            peak = std::max(peak, ses::norm_sq(field[i]));
        }
        peak_ = peak;
    }

    void refresh_title() {
        // Convergence readout while relaxing: exact <H> on the CPU session,
        // or the free ITP estimator (-ln||psi||^2 / 2 dtau) on the GPU path.
        QString energy;
        if (stepping_ != Stepping::RealTime) {
            energy = cpu_is_truth_
                         ? QStringLiteral("E = %1 eV   ")
                               .arg(ses::mean_energy(sim_.psi(), sim_.potential()) *
                                        kHaToEv,
                                    0, 'f', 3)
                         : QStringLiteral("E ~ %1 eV   ")
                               .arg(relax_energy_display_ * kHaToEv, 0, 'f', 3);
        }
        window()->setWindowTitle(
            QStringLiteral("Electron near a hydrogen nucleus   t = %1   %2"
                           "norm = %3   [%4, %5, %6]  1=real 2=relax R=reset tab=view "
                           "[ ]=density M=pos E=energy")
                .arg(sim_.time() + gpu_time_, 0, 'f', 2)
                .arg(energy)
                .arg(norm_display_, 0, 'f', 6)
                .arg(mode_ == ViewMode::Cloud ? QStringLiteral("cloud")
                                              : QStringLiteral("surface"))
                .arg(stepping_ == Stepping::RealTime
                         ? QStringLiteral("real-time")
                         : (stepping_ == Stepping::Relaxing
                                ? QStringLiteral("relaxing->1s")
                                : QStringLiteral("relaxing->%1").arg(relax_label_)))
                .arg(use_gpu_path() ? QStringLiteral("gpu 256^3")
                                    : QStringLiteral("cpu 256^3")) +
            (stepping_ == Stepping::RealTime && !solving()
                 ? QStringLiteral("  emit P = %1 au").arg(radiated_power_, 0, 'e', 2)
                 : QString()) +
            (solving()
                 ? (synth_queue_.empty()
                        ? QStringLiteral("  solving atom: dipole channels (%1 left)")
                              .arg(static_cast<int>(atom_.pair_queue().size()))
                        : QStringLiteral("  solving atom: %1 (%2/%3)")
                              .arg(QLatin1String(kStateSpec[synth_queue_.front()].name))
                              .arg(kNumStates -
                                   static_cast<int>(synth_queue_.size()) + 1)
                              .arg(kNumStates))
                 : QString()) +
            (decay_on_ && !atom_.channels().empty()
                 ? QStringLiteral("  decay ON: tau(2p) %1 au, tau(2s) %2 au, x%3, "
                                  "photons %4%5")
                       .arg(atom_.lifetime_of(kP2Z), 0, 'e', 2)
                       .arg(atom_.lifetime_of(kS2), 0, 'e', 2)
                       .arg(atom_.accel_display(), 0, 'e', 1)
                       .arg(photon_count_)
                       .arg(last_jump_.isEmpty()
                                ? QString()
                                : QStringLiteral(", last %1").arg(last_jump_))
                 : QString()) +
            (laser_pol_ != LaserPol::Off
                 ? QStringLiteral("  laser %1: w %2, E0 %3, P(1s) %4, P(2pz) %5")
                       .arg(laser_pol_ == LaserPol::Z ? QStringLiteral("Z")
                                                      : QStringLiteral("X"))
                       .arg(laser_omega_, 0, 'f', 4)
                       .arg(laser_e0_, 0, 'f', 4)
                       .arg(pop_ground_, 0, 'f', 3)
                       .arg(pop_excited_, 0, 'f', 3)
                 : QString()) +
            (efield_e0_ > 0.0 && laser_pol_ == LaserPol::Off
                 ? QStringLiteral("  E-field +z: %1 au (%2 V/m)")
                       .arg(efield_e0_, 0, 'f', 4)
                       .arg(efield_e0_ * 5.14220674e11, 0, 'e', 2)
                 : QString()) +
            (bfield_b_ > 0.0
                 ? QStringLiteral("  B-field %1: %2 au, omega_L %3 au (psi evolved)")
                       .arg(bfield_axis_ == 2 ? QStringLiteral("z")
                                              : (bfield_axis_ == 0 ? QStringLiteral("x")
                                                                   : QStringLiteral("y")))
                       .arg(bfield_b_, 0, 'f', 4)
                       .arg(0.5 * bfield_b_, 0, 'f', 4)
                 : QString()) +
            (last_measure_.isEmpty()
                 ? QString()
                 : QStringLiteral("  measured %1").arg(last_measure_)));
    }

    // Bridge psi to the display texture. The magnetic field now evolves psi
    // itself (MagneticPropagator on the GPU: diamagnetic in the potential +
    // exact three-shear paramagnetic rotation), so the display is just the
    // real wavefunction -- no display-only rotation trick.
    void write_display_texture() {
        engine_.write_psi_to_volume();
        volume_written_ = true;  // resets the temporal accumulation
    }

    ses::WavepacketSimulation sim_;
    ViewMode mode_ = ViewMode::Cloud;
    Stepping stepping_ = Stepping::RealTime;

    // GPU stepping state (docs/GPU_PLAN.md G5). cpu_is_truth_ is the single
    // sync invariant: true -> sim_.psi() is current, false -> the engine's
    // psi buffer is ahead and must be read back before any CPU-side operation.
    ses_vk::DeviceContext vk_ctx_;  // adopts the QRhiWidget's device
    ses_vk::Engine engine_;
    ses_vk::SceneRenderer vk_renderer_;  // the whole scene, framework-free
    bool vk_renderer_ready_ = false;
    // Temporal-accumulation bookkeeping (render polish).
    struct AccumPrev {
        double azimuth = 1e9, elevation = 0, distance = 0, peak = 0,
               absorbance = 0;
        float flash = 0.0f;
        bool cloud = true;
    } acc_prev_;
    long long frame_index_ = 0;
    bool volume_written_ = false;  // bridge wrote psi this frame
    // Qt's entire render surface: the imported scene image + one blit pass.
    QScopedPointer<QRhiTexture> scene_wrap_;  // createFrom import (non-owning)
    VkImage scene_wrapped_img_ = VK_NULL_HANDLE;
    QScopedPointer<QRhiSampler> blit_sampler_;
    QScopedPointer<QRhiShaderResourceBindings> blit_srb_;
    QScopedPointer<QRhiGraphicsPipeline> blit_pipe_;
    QRhi* rhi_ = nullptr;             // the widget's QRhi, cached in initialize()
    bool rhi_ready_ = false;          // render resources built
    bool compute_init_done_ = false;  // engine + atom solve done (run_gpu_frame)
    bool gpu_ok_ = false;
    bool cpu_is_truth_ = true;
    int pending_gpu_steps_ = 0;
    bool pending_energy_measure_ = false;  // Key E: serviced in run_gpu_frame
    bool gpu_title_due_ = false;
    double gpu_time_ = 0.0;
    double norm_display_ = 1.0;
    double relax_energy_display_ = 0.0;
    double radiated_power_ = 0.0;  // semiclassical Larmor power (au)
    std::vector<float> readback_buf_;

    // The tracked atom (Stage 5b): radial solve, eigenstate synthesis
    // bookkeeping, and the E1 decay channel table live in AtomModel;
    // engine-backed calls pass engine_ explicitly.
    ses_shell::AtomModel atom_;
    bool proj_ready_ = false;  // static projection index uploaded

    // Transitions arc T1/T4/T5: jump bookkeeping.
    QString last_jump_;
    QString last_measure_;  // last energy-measurement readout (Key E)
    int last_measured_index_ = -2;  // last energy-measurement outcome (selftest)
    QString relax_label_ = QStringLiteral("2p");
    std::vector<int> relax_deflate_;        // live RelaxingExcited deflation set
    std::vector<int> relax_deflate_owned_;  // owned transient states to release
    double relax_prev_energy_ = 0.0;     // T7 auto-complete plateau tracking
    int relax_plateau_ = 0;

    // T7: startup atlas build (radial solve + synthesis, chunked in paint).
    std::vector<int> synth_queue_;
    int montage_hold_ = 0;
    bool atlas_done_ = false;
    double decay_accum_dt_ = 0.0;  // sim time since the last decay trial
    int excite_cycle_ = 0;         // key-5 n=3 cycle position
    // Decay is the DEFAULT, as in nature; D is the off-switch for studying
    // pure unitary evolution. Armed once the startup atlas build finishes.
    bool decay_on_ = true;
    int flash_ticks_ = 0;
    long long photon_count_ = 0;

    // Transitions arc T3: laser (resonant dipole drive) bookkeeping.
    LaserPol laser_pol_ = LaserPol::Off;
    double laser_omega_ = 0.0;
    double laser_e0_ = 0.0;
    double efield_e0_ = 0.0;  // static +z electric field magnitude (au); 0 = off
    double bfield_b_ = 0.0;      // magnetic field strength (au); 0 = off
    int bfield_axis_ = 2;        // field direction: 2=z, 0=x, 1=y
    double grid_energy_1s_ = 0.0;  // <H>_grid of the 1s (from the h-audit); the
                                   // laser drives THIS (grid) resonance, not the
                                   // radial label, since bare Coulomb leaves the
                                   // 1s a cusp gap on the coarse grid
    double pop_ground_ = 0.0;
    double pop_excited_ = 0.0;
    double rabi_peak_ = 0.0;  // max P(2pz) since the laser came on

    ses::Mesh mesh_;
    std::vector<ses::Rgb> colors_;
    std::vector<float> psi_staging_;
    double peak_ = 0.0;
    double absorbance_ = 0.68;  // was 1.5; lightened ~3 '[' steps (/1.3^3) --
                                // the default cloud was too opaque

    QTimer timer_;
    bool paused_ = false;
    bool mesh_dirty_ = false;
    bool volume_dirty_ = false;
    long long ticks_ = 0;

    int mask_buf_ = -1;  // boundary absorber (mask, 0) complex state handle
    std::mt19937 rng_{std::random_device{}()};

    double azimuth_ = 0.6;
    double elevation_ = 0.4;
    double distance_ = 150.0;  // default frames ~+-62 Bohr (45 deg fovy): the
                               // n<=6 manifold body, incl. the ~60 Bohr 6s/6p
                               // (wheel in toward 4 for a close-up of small ones)
    QPointF last_pos_;
};

}  // namespace

int main(int argc, char** argv) {
    // GUI-subsystem stderr through a redirect is a fully buffered pipe; a
    // crash then eats every diagnostic. Unbuffered keeps them honest.
    std::setvbuf(stderr, nullptr, _IONBF, 0);
    // QRhi on the Vulkan backend (Viewport::setApi). No MSAA: the volume ray
    // marcher is a full-screen fragment pass where 4x multisampling only
    // multiplies its cost (it smooths nothing but the cube edges) -- at 256^3
    // that budget belongs to the physics. QRhiWidget's sampleCount default (1)
    // already matches.
    QApplication app(argc, argv);

    QMainWindow window;
    window.setWindowTitle(QStringLiteral("Electron wavepacket near a hydrogen nucleus"));
    auto* viewport = new Viewport();
    window.setCentralWidget(viewport);

    // Discoverable controls: toolbar buttons mirroring the hotkeys (the
    // buttons keep Qt::NoFocus by default, so the keys stay live).
    QToolBar* controls = window.addToolBar(QStringLiteral("Controls"));
    controls->setMovable(false);
    controls->addAction(QStringLiteral("Measure (M)"), viewport,
                        [viewport] { viewport->measure_now(); });
    controls->addAction(QStringLiteral("Measure E (E)"), viewport,
                        [viewport] { viewport->measure_energy_now(); });
    controls->addSeparator();
    controls->addAction(QStringLiteral("Real time (1)"), viewport,
                        [viewport] { viewport->set_real_time(); });
    controls->addAction(QStringLiteral("Relax to ground state (2)"), viewport,
                        [viewport] { viewport->set_relaxing(); });
    controls->addAction(QStringLiteral("Relax to 2p (3)"), viewport,
                        [viewport] { viewport->relax_to_excited(); });
    controls->addAction(QStringLiteral("Relax to 2s (4)"), viewport,
                        [viewport] { viewport->relax_to_2s(); });
    controls->addAction(QStringLiteral("Excite n=3/4 (5)"), viewport,
                        [viewport] { viewport->excite_n3(); });
    controls->addAction(QStringLiteral("Decay (D)"), viewport,
                        [viewport] { viewport->toggle_decay(); });
    controls->addAction(QStringLiteral("Laser (L)"), viewport,
                        [viewport] { viewport->toggle_laser(); });
    // Draggable static E-field (+z) magnitude: 0 = off, full-scale = 0.1 au
    // (~5.1e10 V/m). The field is the dipole drive at omega = 0 (Stark). The 1s
    // ground state is STIFF -- it barely polarizes (~0.2 Bohr) below ~0.03 au,
    // then field-ionizes (cloud streams off +z). A live label shows the value.
    controls->addWidget(new QLabel(QStringLiteral(" E-field +z ")));
    {
        constexpr double kMaxEfield = 0.1;  // au at full slider
        auto* efield_val = new QLabel(QStringLiteral("off      "));
        efield_val->setMinimumWidth(96);
        auto* efield_slider = new QSlider(Qt::Horizontal);
        efield_slider->setRange(0, 100);
        efield_slider->setFixedWidth(140);
        efield_slider->setFocusPolicy(Qt::NoFocus);  // keep the hotkeys live
        efield_slider->setToolTip(QStringLiteral(
            "Static uniform E-field along +z (Stark). 0 = off; full = 0.1 au "
            "(~5.1e10 V/m).\nThe 1s ground state barely moves below ~0.03 au, "
            "then field-ionizes."));
        QObject::connect(
            efield_slider, &QSlider::valueChanged, viewport,
            [viewport, efield_val](int val) {
                const double e0 = val / 100.0 * kMaxEfield;
                viewport->set_efield_e0(e0);
                efield_val->setText(
                    e0 > 0.0 ? QStringLiteral("%1 au / %2 V/m")
                                   .arg(e0, 0, 'f', 3)
                                   .arg(e0 * 5.14220674e11, 0, 'e', 1)
                             : QStringLiteral("off"));
            });
        controls->addWidget(efield_slider);
        controls->addWidget(efield_val);
    }
    // Magnetic field: axis cycle (z -> x -> y) + strength slider. psi evolves
    // under the proper minimal-coupling Hamiltonian (paramagnetic precession
    // at omega = B/2 about the axis + diamagnetic contraction). States not
    // symmetric about the axis visibly precess.
    {
        constexpr double kMaxB = 0.2;  // au at full slider
        auto axis_text = [](int a) {
            return a == 2 ? QStringLiteral(" B z ")
                          : (a == 0 ? QStringLiteral(" B x ") : QStringLiteral(" B y "));
        };
        QLabel* b_axis_label = new QLabel(axis_text(2));
        controls->addWidget(b_axis_label);
        controls->addAction(QStringLiteral("axis"), viewport,
                            [viewport, b_axis_label, axis_text] {
                                viewport->toggle_bfield_axis();
                                b_axis_label->setText(axis_text(viewport->bfield_axis()));
                            });
        auto* b_val = new QLabel(QStringLiteral("off"));
        b_val->setMinimumWidth(60);
        auto* b_slider = new QSlider(Qt::Horizontal);
        b_slider->setRange(0, 100);
        b_slider->setFixedWidth(140);
        b_slider->setFocusPolicy(Qt::NoFocus);
        b_slider->setToolTip(QStringLiteral(
            "Magnetic field along the chosen axis (z or x). The cloud precesses "
            "(Larmor) at omega = B/2. Prepare a p_x / d_xy state to see it rotate; "
            "s and p_z do not precess about z."));
        QObject::connect(b_slider, &QSlider::valueChanged, viewport,
                         [viewport, b_val](int val) {
                             const double b = val / 100.0 * kMaxB;
                             viewport->set_bfield_b(b);
                             b_val->setText(b > 0.0 ? QStringLiteral("%1 au").arg(b, 0, 'f', 3)
                                                    : QStringLiteral("off"));
                         });
        controls->addWidget(b_slider);
        controls->addWidget(b_val);
    }
    controls->addSeparator();
    controls->addAction(QStringLiteral("Reset packet (R)"), viewport,
                        [viewport] { viewport->reset_simulation(); });
    controls->addAction(QStringLiteral("Cloud/Surface (Tab)"), viewport,
                        [viewport] { viewport->toggle_view_mode(); });
    controls->addAction(QStringLiteral("Pause (Space)"), viewport,
                        [viewport] { viewport->toggle_pause(); });

    window.resize(1024, 768);
    window.show();
    // StrongFocus only ACCEPTS focus; grab it so the 1/2/R/M/space keys work
    // immediately without a click.
    viewport->setFocus();

    // Verification + selftest arcs (--dump-frame*, --selftest-*): registered
    // from selftest_arcs.hpp so main() stays a shell.
    ses_shell::register_verification_arcs(app, viewport);

    return app.exec();
}
