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
#include <core/sampling.hpp>
#include <core/simulation.hpp>
#include <core/sphere.hpp>
#include <core/vec.hpp>
#include <core/volume.hpp>
#include <core/vram_budget.hpp>

#include "gpu_engine.hpp"
#include "render_shaders.hpp"
#include "manifold_spec.hpp"

#include <cstring>  // std::strcmp for the VRAM extension probe

#include <QApplication>
#include <QKeyEvent>
#include <QMainWindow>
#include <QMessageBox>
#include <QMouseEvent>
#include <QOpenGLFunctions_4_3_Core>
#include <QOpenGLWidget>
#include <QString>
#include <QSurfaceFormat>
#include <QTimer>
#include <QLabel>
#include <QSlider>
#include <QToolBar>
#include <QWheelEvent>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <random>
#include <utility>
#include <vector>

namespace {

[[noreturn]] void fatal_gl_error(const char* stage, const QString& detail) {
    qCritical("%s: %s", stage, qPrintable(detail));
    QMessageBox::critical(nullptr, QStringLiteral("OpenGL error"),
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
constexpr int kMaxPendingGpuSteps = 8;  // backlog cap: a stalled paint cannot spiral
constexpr int kRelaxStepsPerTick = 1;
constexpr double kRelaxDtau = 0.05;
constexpr int kTickMs = 16;
constexpr double kIsoFraction = 0.25;
constexpr int kPhaseLutSize = 256;
constexpr double kMeasureSigma = 0.625;  // position measurement resolution (Bohr):
                                         // the sharpest a h = 0.625 grid resolves
                                         // without aliasing (smaller needs finer h)
// Display decay rate: the TRUE Einstein-A lifetime is ~1e8 a.u. (unwatchable);
// this gives tau_display ~ 8 a.u. (~3 s of wall time). The title reports the
// true lifetime and the acceleration factor honestly.
constexpr double kDecayGammaDisplay = 0.125;
constexpr double kProtonMarkerRadius = 0.35;  // symbolic (a real proton is ~1e-5 Bohr)
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

class Viewport : public QOpenGLWidget, protected QOpenGLFunctions_4_3_Core {
public:
    explicit Viewport(QWidget* parent = nullptr)
        : QOpenGLWidget(parent), sim_(make_simulation()) {
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

    void initializeGL() override {
        if (!initializeOpenGLFunctions()) {
            const QSurfaceFormat got = context()->format();
            fatal_gl_error("OpenGL 4.3 core profile is required",
                           QStringLiteral("The driver provided only %1.%2. Update the GPU "
                                          "driver or run on hardware with OpenGL 4.3 support.")
                               .arg(got.majorVersion())
                               .arg(got.minorVersion()));
        }
        glClearColor(0.04f, 0.05f, 0.09f, 1.0f);

        // -- mesh pipeline --
        mesh_program_ = link_program(kMeshVertexShader, kMeshFragmentShader);
        mesh_mvp_loc_ = glGetUniformLocation(mesh_program_, "mvp");
        mesh_eye_loc_ = glGetUniformLocation(mesh_program_, "eye");
        glGenVertexArrays(1, &mesh_vao_);
        glBindVertexArray(mesh_vao_);
        glGenBuffers(1, &mesh_vbo_);
        glBindBuffer(GL_ARRAY_BUFFER, mesh_vbo_);
        constexpr GLsizei kStride = 9 * sizeof(float);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, kStride, reinterpret_cast<void*>(0));
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, kStride,
                              reinterpret_cast<void*>(3 * sizeof(float)));
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, kStride,
                              reinterpret_cast<void*>(6 * sizeof(float)));
        glEnableVertexAttribArray(2);
        glBindVertexArray(0);

        upload_proton_marker();
        upload_axes_gizmo();

        // -- GPU propagation engine (fp32 transcription of the tested CPU
        //    tables; verified by sesolver_gpucheck). Falls back to CPU
        //    stepping when compute programs fail to build. --
        gpu_ok_ = engine_.initialize(*this, sim_.grid(),
                                     sim_.propagator().half_potential_phase(),
                                     sim_.propagator().kinetic_phase(), sim_.psi());
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
            state_precision_ = ses::choose_state_precision(
                query_free_vram_bytes(), kNumStates, bytes_per_state,
                kVramHeadroomBytes, &atlas_fits);
            std::fprintf(stderr, "vram: atlas precision = %s%s\n",
                         state_precision_ == ses::GpuPrecision::Fp16 ? "fp16 (half)"
                                                                     : "fp32",
                         atlas_fits ? ""
                                    : "  [WARNING: even fp16 is tight -- consider a "
                                      "smaller box or manifold]");
            // Imaginary-time weights from the tested CPU relaxer (G7).
            const ses::ImaginaryTimePropagator3D relaxer{sim_.grid(), sim_.potential(),
                                                         kRelaxDtau};
            engine_.set_relax_tables(*this, relaxer.half_potential_weight(),
                                     relaxer.kinetic_weight(), kRelaxDtau);
            // Radiation: the atomic potential gradient (for the live
            // semiclassical radiated power <grad V> reduction).
            engine_.set_potential_gradient(*this, sim_.potential());
            // T6/T7: solve the atom up front. The radial engine gets every
            // bound level to n = 10 (the full lifetime table, printed
            // below); the 3D tracked manifold (n <= 6, what the box holds)
            // is then synthesized chunked across frames so decay is armed
            // BY DEFAULT and every demo entry point is instant afterwards.
            solve_radial_atom();
            // Orbital-free projection index (Phase 5): the static counting-sort
            // geometry, uploaded once. Populations then come from ONE project_psi
            // deposit pass (state-count independent) instead of a per-state
            // inner_with_psi over the resident atlas.
            {
                const ses::RadialBinIndex bin_idx =
                    ses::build_radial_bin_index(sim_.grid(), radial_grid_);
                engine_.set_projection_index(*this, bin_idx.sorted_cell, bin_idx.bin_off,
                                             radial_grid_.n, radial_grid_.h(), 5);
                proj_ready_ = true;
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
                mask_buf_ = engine_.create_state_buffer(*this, mf);
            }
        } else {
            decay_on_ = false;  // jump trials are GPU-only
            atlas_done_ = true;
        }

        // -- volume pipeline --
        volume_program_ = link_program(kVolumeVertexShader, kVolumeFragmentShader);
        vol_mvp_loc_ = glGetUniformLocation(volume_program_, "mvp");
        vol_eye_loc_ = glGetUniformLocation(volume_program_, "eye");
        vol_boxmin_loc_ = glGetUniformLocation(volume_program_, "box_min");
        vol_boxmax_loc_ = glGetUniformLocation(volume_program_, "box_max");
        vol_invpeak_loc_ = glGetUniformLocation(volume_program_, "inv_peak");
        vol_absorb_loc_ = glGetUniformLocation(volume_program_, "absorbance");
        vol_pcenter_loc_ = glGetUniformLocation(volume_program_, "proton_center");
        vol_pradius_loc_ = glGetUniformLocation(volume_program_, "proton_radius");
        vol_pcolor_loc_ = glGetUniformLocation(volume_program_, "proton_color");
        glUseProgram(volume_program_);
        glUniform1i(glGetUniformLocation(volume_program_, "psi_tex"), 0);
        glUniform1i(glGetUniformLocation(volume_program_, "phase_tex"), 1);

        upload_box_cube();
        create_textures();

        mesh_dirty_ = true;
        volume_dirty_ = true;
    }

    void paintGL() override {
        // Photon flash: a brief warm background right after a quantum jump.
        if (flash_ticks_ > 0) {
            const float w = static_cast<float>(flash_ticks_) / 25.0f;
            glClearColor(0.04f + 0.55f * w, 0.05f + 0.42f * w, 0.09f + 0.18f * w, 1.0f);
            --flash_ticks_;
        } else {
            glClearColor(0.04f, 0.05f, 0.09f, 1.0f);
        }
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        const double aspect = static_cast<double>(width()) / std::max(1, height());
        // The far plane must always enclose the whole box BEHIND the target, or
        // the volume's back-face proxy geometry (we cull front faces) gets
        // clipped and leaves dark triangular holes when zoomed out. Track the
        // camera distance plus the box's center-to-corner reach -- the farthest
        // any box point can sit from the eye at any orientation.
        const ses::Grid3D& gbox = sim_.grid();
        const double bx = std::max(std::abs(gbox.x.xmin), std::abs(gbox.x.xmax));
        const double by = std::max(std::abs(gbox.y.xmin), std::abs(gbox.y.xmax));
        const double bz = std::max(std::abs(gbox.z.xmin), std::abs(gbox.z.xmax));
        const double box_reach = std::sqrt(bx * bx + by * by + bz * bz);
        const double zfar = distance_ + box_reach + 1.0;
        const ses::Mat4 proj =
            ses::perspective(45.0 * 3.14159265358979323846 / 180.0, aspect, 0.1, zfar);
        const ses::Vec3d eye = ses::orbit_eye(azimuth_, elevation_, distance_, ses::Vec3d{});
        const ses::Mat4 view = ses::look_at(eye, ses::Vec3d{}, ses::Vec3d{0.0, 1.0, 0.0});
        const ses::Mat4 mvp = proj * view;
        float mvp_f[16];
        for (int i = 0; i < 16; ++i) {
            mvp_f[i] = static_cast<float>(mvp.m[i]);
        }

        // GPU stepping happens here, where the context is guaranteed current.
        if (use_gpu_path()) {
            if (!atlas_done_) {
                // T6/T7: startup atlas build owns the psi buffer this
                // frame; each synthesized orbital IS the picture (a
                // flick-through of the atom's states). Normal stepping
                // resumes when the whole channel table is assembled.
                run_atlas_chunk();
                pending_gpu_steps_ = 0;
                if (gpu_title_due_) {
                    gpu_title_due_ = false;
                    refresh_title();
                }
            } else {
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
                engine_.upload_state(*this, sim_.psi());
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
                engine_.project_psi(*this);
                std::vector<double> pop(static_cast<std::size_t>(kNumStates));
                for (int s = 0; s < kNumStates; ++s) {
                    pop[static_cast<std::size_t>(s)] = project_population(s);
                }
                std::uniform_real_distribution<double> uniform(0.0, 1.0);
                const int n = ses::sample_energy_eigenstate(pop, uniform(rng_));
                last_measured_index_ = n;  // >=0 eigenstate, -1 outside manifold
                if (n >= 0) {
                    collapse_onto(n);
                    write_display_texture();
                    last_measure_ =
                        QStringLiteral("%1 (E %2 eV)")
                            .arg(QLatin1String(kStateSpec[static_cast<std::size_t>(n)].name))
                            .arg(state_energy_[static_cast<std::size_t>(n)] * kHaToEv, 0,
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
                        const ses_gpu::NormPeak np = engine_.norm_and_peak(*this);
                        norm_display_ = np.sum;
                        if (np.peak > 0.0) {
                            peak_ = np.peak;  // brightness tracks the cloud
                        }
                        // fp32 drift renormalization: the split-operator is
                        // unitary in exact arithmetic; pinning the norm back
                        // to 1 removes pure numerical decay.
                        if (np.sum > 0.0 && std::abs(np.sum - 1.0) > 1e-4) {
                            engine_.scale(*this,
                                          static_cast<float>(1.0 / std::sqrt(np.sum)));
                        }
                        // Radiation: the semiclassical radiated power from the
                        // oscillating dipole, P = (2/3)alpha^3|<grad V>|^2, via
                        // the GPU mean-force reduction (a 4 KB readback). ~0 for
                        // a stationary eigenstate, nonzero for a superposition.
                        radiated_power_ = ses::larmor_power(engine_.mean_force(*this));
                    }
                    if (laser_pol_ != LaserPol::Off) {
                        // T3: resonant dipole drive. t0 is the same clock
                        // that credits gpu_time_, so the carrier phase
                        // cos(w t) stays continuous across batches/pauses.
                        const ses::DipoleDrive d{laser_axis(), laser_e0_, laser_omega_};
                        engine_.driven_step(*this, d, sim_.time() + gpu_time_,
                                            sim_.dt(), pending_gpu_steps_);
                    } else if (bfield_b_ > 0.0) {
                        // Magnetic field along bfield_axis_: the PROPER minimal-
                        // coupling solve. psi evolves under H = H0 + (E z) +
                        // (B/2)L_axis + (B^2/8)rho_perp^2 -- the static E and the
                        // diamagnetic term are already folded into the
                        // half-potential (upload_field_tables), the paramagnetic
                        // L_axis is the exact three-shear rotation. Because it is
                        // the combined solve, crossed E(z)-B(x/y) is genuine.
                        engine_.magnetic_step(*this, bfield_axis_,
                                              0.5 * bfield_b_ * (0.5 * sim_.dt()),
                                              pending_gpu_steps_);
                    } else {
                        // The static E-field (if any) is folded into the
                        // half-potential -- exact, and equal to the old omega=0
                        // dipole kick -- so a plain step polarizes / field-
                        // ionizes correctly.
                        engine_.step(*this, pending_gpu_steps_);
                    }
                    // Time is credited where steps EXECUTE, so a stalled or
                    // occluded paint cannot desync the clock from the state.
                    gpu_time_ += pending_gpu_steps_ * sim_.dt();

                    // Boundary absorber (real-time only): damp outgoing/ionized
                    // flux at the walls so it leaves instead of wrapping around
                    // the periodic FFT box. Interior mask = 1, so the bound atom
                    // is untouched; imaginary-time relaxation never runs this.
                    if (mask_buf_ != 0) {
                        engine_.apply_mask(*this, mask_buf_);
                    }

                    // Orbital-free populations (Phase 5): ONE deposit pass on the
                    // post-step psi, shared by the decay and laser readouts this
                    // title tick (state-count independent) -- replaces the
                    // per-state inner_with_psi over the resident atlas.
                    if (gpu_title_due_ && proj_ready_ &&
                        ((decay_on_ && !channels_.empty()) ||
                         laser_pol_ != LaserPol::Off)) {
                        engine_.project_psi(*this);
                    }

                    // T4/T5/T7: competing-channels Poisson trials over the
                    // whole tracked manifold. The exponential is memoryless,
                    // so trials run on the TITLE cadence with the sim time
                    // accumulated since the last trial (identical statistics,
                    // 13 GPU reductions every ~10 ticks instead of per
                    // frame). Selection arithmetic is the core-tested
                    // pick_decay_channel; a jump collapses onto the fired
                    // channel's destination ("photon out").
                    if (decay_on_ && !channels_.empty()) {
                        decay_accum_dt_ += pending_gpu_steps_ * sim_.dt();
                        if (gpu_title_due_) {
                            std::array<double, kNumStates> pop{};
                            for (int s = 1; s < kNumStates; ++s) {
                                pop[s] = project_population(s);
                            }
                            std::vector<double> rates(channels_.size());
                            for (std::size_t c = 0; c < channels_.size(); ++c) {
                                rates[c] = channels_[c].gamma_display *
                                           pop[channels_[c].from];
                            }
                            std::uniform_real_distribution<double> uniform(0.0, 1.0);
                            const ses::ChannelPick pick = ses::pick_decay_channel(
                                rates, decay_accum_dt_, uniform(rng_), uniform(rng_));
                            decay_accum_dt_ = 0.0;
                            if (pick.channel >= 0) {
                                const ShellChannel& ch =
                                    channels_[static_cast<std::size_t>(pick.channel)];
                                collapse_onto(ch.to);
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
                        pop_excited_ = project_population(kP2Z);
                        pop_ground_ = project_population(kS1);
                        rabi_peak_ = std::max(rabi_peak_, pop_excited_);
                    }
                } else {
                    // GPU imaginary time (G7/T1): renormalized every step;
                    // the ITP estimator gives the convergence readout free.
                    // The excited flavor deflates the cached ground state.
                    const ses_gpu::GpuEngine::RelaxStats stats =
                        (stepping_ == Stepping::RelaxingExcited &&
                         !relax_deflate_.empty())
                            ? engine_.relax_deflated_step(*this, relax_deflate_,
                                                          pending_gpu_steps_)
                            : engine_.relax_step(*this, pending_gpu_steps_);
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
            }  // end of the non-building (normal stepping) branch
        }

        if (mode_ == ViewMode::Cloud) {
            if (volume_dirty_) {
                upload_volume();
                volume_dirty_ = false;
            }
            glDisable(GL_DEPTH_TEST);
            glEnable(GL_BLEND);
            glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);  // premultiplied over clear color
            glEnable(GL_CULL_FACE);
            glCullFace(GL_FRONT);  // raster the BACK faces: works from inside too

            glUseProgram(volume_program_);
            glUniformMatrix4fv(vol_mvp_loc_, 1, GL_FALSE, mvp_f);
            glUniform3f(vol_eye_loc_, static_cast<float>(eye.x), static_cast<float>(eye.y),
                        static_cast<float>(eye.z));
            const ses::Grid3D& g = sim_.grid();
            glUniform3f(vol_boxmin_loc_, static_cast<float>(g.x.xmin),
                        static_cast<float>(g.y.xmin), static_cast<float>(g.z.xmin));
            glUniform3f(vol_boxmax_loc_, static_cast<float>(g.x.xmax),
                        static_cast<float>(g.y.xmax), static_cast<float>(g.z.xmax));
            glUniform1f(vol_invpeak_loc_, static_cast<float>(peak_ > 0.0 ? 1.0 / peak_ : 0.0));
            glUniform1f(vol_absorb_loc_, static_cast<float>(absorbance_));
            // The nucleus lives INSIDE the ray marcher now (analytic sphere):
            // fogged by cloud in front, occluding cloud behind.
            glUniform3f(vol_pcenter_loc_, 0.0f, 0.0f, 0.0f);
            glUniform1f(vol_pradius_loc_, static_cast<float>(kProtonMarkerRadius));
            glUniform3f(vol_pcolor_loc_, 1.0f, 0.45f, 0.2f);

            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_3D, psi_tex_);
            glActiveTexture(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_1D, phase_tex_);

            glBindVertexArray(cube_vao_);
            glDrawArrays(GL_TRIANGLES, 0, 36);
            glBindVertexArray(0);

            glDisable(GL_CULL_FACE);
            glDisable(GL_BLEND);
        } else {
            if (mesh_dirty_) {
                upload_mesh();
                mesh_dirty_ = false;
            }
            // Surface mode keeps the mesh proton (depth-tested against the
            // opaque isosurface, so occlusion is already correct there).
            draw_proton(mvp_f, eye);
            glEnable(GL_DEPTH_TEST);
            glUseProgram(mesh_program_);
            glUniformMatrix4fv(mesh_mvp_loc_, 1, GL_FALSE, mvp_f);
            glUniform3f(mesh_eye_loc_, static_cast<float>(eye.x), static_cast<float>(eye.y),
                        static_cast<float>(eye.z));
            glBindVertexArray(mesh_vao_);
            glDrawArrays(GL_TRIANGLES, 0, vertex_count_);
            glBindVertexArray(0);
        }

        // Orientation gizmo overlays both views (its own corner viewport).
        draw_axes_gizmo();
    }

    // XYZ orientation gizmo in the bottom-left corner: the same orbit
    // orientation as the scene but at a FIXED distance (constant screen size),
    // in a square viewport whose depth is cleared so it neither occludes nor is
    // occluded by the scene. Lets the user read field / axis directions.
    void draw_axes_gizmo() {
        if (gizmo_vertex_count_ == 0) {
            return;
        }
        GLint vp[4];
        glGetIntegerv(GL_VIEWPORT, vp);  // full framebuffer viewport (device px)
        const GLint side = std::clamp(std::min(vp[2], vp[3]) / 6, 96, 180);
        const GLint margin = side / 8;

        const double d_g = 3.2;  // frames the length-1 arrows in the corner box
        const ses::Vec3d eye = ses::orbit_eye(azimuth_, elevation_, d_g, ses::Vec3d{});
        const ses::Mat4 proj =
            ses::perspective(45.0 * 3.14159265358979323846 / 180.0, 1.0, 0.1, 50.0);
        const ses::Mat4 view =
            ses::look_at(eye, ses::Vec3d{}, ses::Vec3d{0.0, 1.0, 0.0});
        const ses::Mat4 mvp = proj * view;
        float mvp_f[16];
        for (int i = 0; i < 16; ++i) {
            mvp_f[i] = static_cast<float>(mvp.m[i]);
        }

        glViewport(vp[0] + margin, vp[1] + margin, side, side);
        glEnable(GL_SCISSOR_TEST);
        glScissor(vp[0] + margin, vp[1] + margin, side, side);
        glDisable(GL_BLEND);
        glEnable(GL_DEPTH_TEST);
        glDepthMask(GL_TRUE);
        glClear(GL_DEPTH_BUFFER_BIT);  // own depth region; no scene occlusion

        glUseProgram(mesh_program_);
        glUniformMatrix4fv(mesh_mvp_loc_, 1, GL_FALSE, mvp_f);
        glUniform3f(mesh_eye_loc_, static_cast<float>(eye.x), static_cast<float>(eye.y),
                    static_cast<float>(eye.z));
        glBindVertexArray(gizmo_vao_);
        glDrawArrays(GL_TRIANGLES, 0, gizmo_vertex_count_);
        glBindVertexArray(0);

        draw_z_label(view, eye);  // label only Z; X, Y follow by the RH rule

        glDisable(GL_SCISSOR_TEST);
        glViewport(vp[0], vp[1], vp[2], vp[3]);  // restore the full viewport
    }

    // A small billboarded "z" glyph just past the +Z arrow tip, always facing
    // the gizmo camera. Built each frame (18 verts) into z_label_vbo_; drawn
    // with the mesh program / gizmo MVP already bound by draw_axes_gizmo.
    void draw_z_label(const ses::Mat4& view, const ses::Vec3d& eye) {
        const ses::Vec3d right{view.m[0], view.m[4], view.m[8]};
        const ses::Vec3d up{view.m[1], view.m[5], view.m[9]};
        const ses::Vec3d center{0.0, 0.0, 1.18};  // just past the length-1 tip
        const double s = 0.24;
        const ses::Vec3d nrm = normalized(eye - center);  // faces the camera

        // "z" as three strokes in [-0.4,0.4] x [-0.5,0.5]: top bar, diagonal
        // (top-right -> bottom-left), bottom bar. Each a rectangle (2 tris).
        std::vector<std::array<double, 2>> pts;
        auto quad = [&](std::array<double, 2> a, std::array<double, 2> b,
                        std::array<double, 2> c, std::array<double, 2> d) {
            pts.push_back(a);
            pts.push_back(b);
            pts.push_back(c);
            pts.push_back(a);
            pts.push_back(c);
            pts.push_back(d);
        };
        quad({-0.4, 0.5}, {0.4, 0.5}, {0.4, 0.34}, {-0.4, 0.34});      // top bar
        quad({-0.4, -0.34}, {0.4, -0.34}, {0.4, -0.5}, {-0.4, -0.5});  // bottom
        const double ax = 0.4, ay = 0.40, bx = -0.4, by = -0.40, ht = 0.11;
        const double dx = bx - ax, dy = by - ay;
        const double dl = std::sqrt(dx * dx + dy * dy);
        const double px = -dy / dl * ht, py = dx / dl * ht;  // perpendicular
        quad({ax + px, ay + py}, {ax - px, ay - py}, {bx - px, by - py},
             {bx + px, by + py});  // diagonal bar

        std::vector<float> data;
        data.reserve(pts.size() * 9);
        for (const std::array<double, 2>& p : pts) {
            const ses::Vec3d w = center + (p[0] * s) * right + (p[1] * s) * up;
            data.push_back(static_cast<float>(w.x));
            data.push_back(static_cast<float>(w.y));
            data.push_back(static_cast<float>(w.z));
            data.push_back(static_cast<float>(nrm.x));
            data.push_back(static_cast<float>(nrm.y));
            data.push_back(static_cast<float>(nrm.z));
            data.push_back(0.75f);
            data.push_back(0.85f);
            data.push_back(1.0f);
        }
        glBindVertexArray(z_label_vao_);
        glBindBuffer(GL_ARRAY_BUFFER, z_label_vbo_);
        glBufferData(GL_ARRAY_BUFFER,
                     static_cast<GLsizeiptr>(data.size() * sizeof(float)),
                     data.data(), GL_DYNAMIC_DRAW);
        glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(pts.size()));
        glBindVertexArray(0);
    }

    void draw_proton(const float* mvp_f, const ses::Vec3d& eye) {
        glEnable(GL_DEPTH_TEST);
        glUseProgram(mesh_program_);
        glUniformMatrix4fv(mesh_mvp_loc_, 1, GL_FALSE, mvp_f);
        glUniform3f(mesh_eye_loc_, static_cast<float>(eye.x), static_cast<float>(eye.y),
                    static_cast<float>(eye.z));
        glBindVertexArray(proton_vao_);
        glDrawArrays(GL_TRIANGLES, 0, proton_vertex_count_);
        glBindVertexArray(0);
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
        makeCurrent();
        free_deflation_buffers();  // drop any owned deflation phi
        doneCurrent();
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
            if (!prepare_manifold_cache()) {
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
        for (const ShellChannel& c : channels_) {
            if (c.from == from && c.to == to) {
                return c.a_true;
            }
        }
        return 0.0;
    }
    bool solving() const { return gpu_ok_ && !atlas_done_; }
    // Ready only once the FULL table is assembled (channels_ fills
    // incrementally during the pair phase -- do not race it).
    bool manifold_ready() const { return atlas_done_ && !channels_.empty(); }

    // T7: instantly excite an n = 3 state (cycles through a small set) and
    // watch the CASCADE: e.g. 3d -> 2p (photon) -> 1s (photon).
    void excite_n3() {
        if (!gpu_ok_ || solving()) {
            return;
        }
        if (mode_ != ViewMode::Cloud) {
            mode_ = ViewMode::Cloud;
        }
        if (!prepare_manifold_cache()) {
            return;
        }
        static constexpr int kCycle[] = {k3DZ0, k4F0, k3S, k4S};
        const int idx = kCycle[excite_cycle_++ % 4];
        makeCurrent();
        collapse_onto(idx);
        doneCurrent();
        cpu_is_truth_ = false;  // the GPU state is ahead now
        stepping_ = Stepping::RealTime;
        after_control();
    }
    double state_energy(int idx) const { return state_energy_[static_cast<std::size_t>(idx)]; }

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
                                                              : state_energy_[kS1]);
            laser_omega_ = state_energy_[kP2Z] - e_1s;
            laser_e0_ = dipole_z_ > 0.0 ? kRabiTargetOmega / dipole_z_ : 0.0;
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
        makeCurrent();
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
            engine_.set_half_potential(*this, aug.half_potential_phase());
        } else {
            engine_.set_half_potential(*this, sim_.propagator().half_potential_phase());
        }
        doneCurrent();
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

    // Selftest hooks (magnetic Larmor precession): set psi to a cached
    // manifold eigenstate and probe another state's population. A field along
    // z rotates 2p_x -> 2p_y at omega_L = B/2, so P(2p_y) must rise -- proving
    // the field evolves psi itself, not just the display.
    void debug_prepare_state(int idx) {
        if (!manifold_ready() || idx < 0 || idx >= kNumStates) {
            return;
        }
        makeCurrent();
        collapse_onto(idx);
        doneCurrent();
        cpu_is_truth_ = false;
        stepping_ = Stepping::RealTime;
        after_control();
    }
    double probe_population(int idx) {
        if (!manifold_ready() || idx < 0 || idx >= kNumStates) {
            return 0.0;
        }
        makeCurrent();
        engine_.project_psi(*this);
        const double p = project_population(idx);
        doneCurrent();
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
        makeCurrent();
        free_deflation_buffers();
        relax_deflate_.push_back(synth_transient(kS1));
        if (deflate_p_triplet) {
            relax_deflate_.push_back(synth_transient(kP2X));
            relax_deflate_.push_back(synth_transient(kP2Y));
            relax_deflate_.push_back(synth_transient(kP2Z));
        }
        relax_deflate_owned_ = relax_deflate_;
        doneCurrent();
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

    // T7: solve the radial atom once (blocking, well under a second).
    // In-box levels back the tracked manifold (u(R_box) = 0 is exactly what
    // the periodic grid supports); the free-atom table to n = 10 is the
    // full lifetime atlas, printed for the record.
    void solve_radial_atom() {
        const double r_box = sim_.grid().x.xmax;  // 64 Bohr
        radial_grid_ = ses::RadialGrid{r_box, 5119};
        std::vector<double> v(static_cast<std::size_t>(radial_grid_.n));
        for (int i = 0; i < radial_grid_.n; ++i) {
            v[static_cast<std::size_t>(i)] = -1.0 / radial_grid_.r(i);  // r=(i+1)h>0
        }
        for (int lev = 0; lev < kNumLevels; ++lev) {
            const ses::RadialState st = ses::radial_eigenstate(
                radial_grid_,
                ses::radial_hamiltonian(radial_grid_, v, kLevelSpec[lev].l),
                kLevelSpec[lev].k);
            radial_u_[static_cast<std::size_t>(lev)] = st.u;
            radial_energy_[static_cast<std::size_t>(lev)] = st.energy;
        }
        radial_ready_ = true;

        // The full free-atom lifetime table to n = 10 (55 levels).
        const ses::RadialGrid free_grid{600.0, 14999};
        std::vector<double> vf(static_cast<std::size_t>(free_grid.n));
        for (int i = 0; i < free_grid.n; ++i) {
            vf[static_cast<std::size_t>(i)] = -1.0 / free_grid.r(i);
        }
        const std::vector<ses::LevelInfo> table =
            ses::bound_level_table(free_grid, vf, 10);
        std::fprintf(stderr,
                     "spectrum: free hydrogen atom (true -1/r), ALL %d bound levels to "
                     "n = 10 (E1 lifetimes from our radial engine)\n",
                     static_cast<int>(table.size()));
        const char* kSpdf = "spdfghijkl";
        for (const ses::LevelInfo& e : table) {
            std::fprintf(stderr,
                         "spectrum: %2d%c  E = %11.6f Ha   tau = %.3e au (%.3e ns)%s\n",
                         e.n, kSpdf[e.l], e.energy, e.lifetime,
                         e.lifetime * 2.4188843e-17 * 1e9,
                         e.lifetime == 0.0 ? "  [E1-stable]" : "");
        }
    }

    // Synthesize (and cache) tracked state `idx` from the radial solution:
    // psi = (u/r) Y_lm -- exact separation of variables, no ITP ladder.
    // Needs a current GL context for the buffer upload.
    // Synthesize state `idx` = (u/r) Y_lm STRAIGHT into a resident GPU buffer:
    // the radial u_nl(r) table goes to the GPU and a compute shader places the
    // orbital + normalizes it on-device -- no CPU field, no host copy. Caches
    // the energy; *out_peak (if given) receives the normalized peak density.
    // Needs a current GL context.
    // Free VRAM in bytes via a vendor GL extension (NVIDIA NVX / AMD ATI), or
    // ses::kVramUnknown when neither is present (Mesa, remote GL). Core GL has
    // no such query and these enums are absent from the Qt headers.
    std::int64_t query_free_vram_bytes() {
        constexpr GLenum kAvailNvx = 0x9049;    // CURRENT_AVAILABLE_VIDMEM_NVX (KiB)
        constexpr GLenum kTexFreeAti = 0x87FC;  // TEXTURE_FREE_MEMORY_ATI (KB; [0]=free)
        GLint n_ext = 0;
        glGetIntegerv(GL_NUM_EXTENSIONS, &n_ext);
        bool nvx = false;
        bool ati = false;
        for (GLint i = 0; i < n_ext; ++i) {
            const char* e = reinterpret_cast<const char*>(
                glGetStringi(GL_EXTENSIONS, static_cast<GLuint>(i)));
            if (e == nullptr) {
                continue;
            }
            if (std::strcmp(e, "GL_NVX_gpu_memory_info") == 0) {
                nvx = true;
            }
            if (std::strcmp(e, "GL_ATI_meminfo") == 0) {
                ati = true;
            }
        }
        while (glGetError() != GL_NO_ERROR) {
        }
        if (nvx) {
            GLint kib = 0;
            glGetIntegerv(kAvailNvx, &kib);
            if (glGetError() == GL_NO_ERROR && kib > 0) {
                return static_cast<std::int64_t>(kib) * 1024;
            }
        }
        if (ati) {
            GLint info[4] = {0, 0, 0, 0};
            glGetIntegerv(kTexFreeAti, info);
            if (glGetError() == GL_NO_ERROR && info[0] > 0) {
                return static_cast<std::int64_t>(info[0]) * 1024;
            }
        }
        return ses::kVramUnknown;
    }

    // Whether atlas state `idx` is stored fp16. Fp32 is kept for the h-audit
    // s-states (E_radial vs <H>_grid amplifies fp16 noise through the Laplacian)
    // and the deflation set (relax_deflated_step reads them as fp32 phi buffers).
    bool state_is_fp16(int idx) const {
        if (state_precision_ != ses::GpuPrecision::Fp16) {
            return false;
        }
        switch (idx) {
            case kS1:
            case kP2X:
            case kP2Y:
            case kP2Z:
            case k4S:
            case k5S:
            case k6S:
                return false;
            default:
                return true;
        }
    }

    GLuint gpu_synthesize(int idx, double* out_peak = nullptr) {
        const std::size_t s = static_cast<std::size_t>(idx);
        const StateSpec& sp = kStateSpec[s];
        state_energy_[s] = radial_energy_[static_cast<std::size_t>(sp.level)];
        if (state_is_fp16(idx)) {
            return engine_.synthesize_state_half(
                *this, radial_u_[static_cast<std::size_t>(sp.level)], sp.l, sp.m,
                radial_grid_.h(), radial_grid_.rmax, radial_grid_.n, out_peak,
                &state_norm2_[s]);
        }
        return engine_.synthesize_state(
            *this, radial_u_[static_cast<std::size_t>(sp.level)], sp.l, sp.m,
            radial_grid_.h(), radial_grid_.rmax, radial_grid_.n, out_peak,
            &state_norm2_[s]);
    }

    // Population |<n|psi>|^2 from the last engine_.project_psi() pass: the 1-D
    // radial dot g_lm . u_nl, grid-normalized to be value-identical to the
    // retired inner_with_psi(grid-normalized orbital) path.
    double project_population(int idx) const {
        const StateSpec& sp = kStateSpec[static_cast<std::size_t>(idx)];
        const double n2 = state_norm2_[static_cast<std::size_t>(idx)];
        if (n2 <= 0.0) {
            return 0.0;
        }
        return ses::norm_sq(engine_.project_amplitude(
                   radial_u_[static_cast<std::size_t>(sp.level)], sp.l, sp.m)) /
               n2;
    }

    // Collapse psi onto eigenstate `idx` by synthesizing it ON DEMAND (no
    // resident atlas): psi_buf_ is overwritten with the normalized orbital.
    void collapse_onto(int idx) {
        const StateSpec& sp = kStateSpec[static_cast<std::size_t>(idx)];
        engine_.synthesize_into_psi(*this, radial_u_[static_cast<std::size_t>(sp.level)],
                                    sp.l, sp.m, radial_grid_.h(), radial_grid_.rmax,
                                    radial_grid_.n);
    }

    // Synthesize eigenstate idx into a FRESH fp32 buffer (caller frees); captures
    // its grid norm (for populations) and energy. ALL build/deflation work uses
    // these transients -- no resident atlas -- so the startup montage holds ONE
    // orbital at a time (not all 91) and 512^3, where a resident atlas is
    // physically impossible, becomes feasible. (fp16 is kept dormant for a future
    // big-box preset; it only ever mattered for a RESIDENT atlas.)
    GLuint synth_transient(int idx, double* out_norm2 = nullptr, double* out_peak = nullptr) {
        const StateSpec& sp = kStateSpec[static_cast<std::size_t>(idx)];
        state_energy_[static_cast<std::size_t>(idx)] =
            radial_energy_[static_cast<std::size_t>(sp.level)];
        return engine_.synthesize_state(*this, radial_u_[static_cast<std::size_t>(sp.level)],
                                        sp.l, sp.m, radial_grid_.h(), radial_grid_.rmax,
                                        radial_grid_.n, out_peak, out_norm2);
    }

    // Free the OWNED transient deflation buffers (synthesized at relax-start).
    // Caller must hold a current GL context.
    void free_deflation_buffers() {
        for (GLuint b : relax_deflate_owned_) {
            if (b != 0) {
                GLuint x = b;
                glDeleteBuffers(1, &x);
            }
        }
        relax_deflate_owned_.clear();
        relax_deflate_.clear();
    }

    // Ensure state `idx` has a resident GPU buffer, synthesized on the GPU.
    bool ensure_state(int idx) {
        const std::size_t s = static_cast<std::size_t>(idx);
        if (state_buf_[s] != 0) {
            return true;
        }
        if (!radial_ready_) {
            return false;
        }
        state_buf_[s] = gpu_synthesize(idx);
        return state_buf_[s] != 0;
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
            if (!radial_ready_) {
                atlas_done_ = true;  // no radial solve: give up gracefully
                return;
            }
            const std::size_t s = static_cast<std::size_t>(idx);
            // Synthesize into a TRANSIENT fp32 buffer: capture its grid norm (for
            // populations), SHOW it (montage), audit, then FREE. The orbital-free
            // projection keeps no atlas, so the montage holds ONE orbital at a
            // time instead of accumulating all 91 (the old startup VRAM ramp).
            double pk = 0.0;
            GLuint buf = synth_transient(idx, &state_norm2_[s], &pk);
            if (buf == 0) {
                atlas_done_ = true;  // GPU buffer alloc failed: give up gracefully
                return;
            }
            engine_.copy_into_psi(*this, buf);  // show (fp32)
            // The h-audit: cross-check the 1D radial energy against the full 3D
            // spectral <H> for the resolution-critical 1s and the box-critical
            // 4s/5s/6s -- the ONLY states read back to the CPU (4 of 91).
            if (idx == kS1 || idx == k4S || idx == k5S || idx == k6S) {
                engine_.readback(*this, readback_buf_);
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
                             kStateSpec[s].name, state_energy_[s], e_grid);
            } else {
                std::fprintf(stderr, "atlas: %-8s E_radial = %.6f Ha\n",
                             kStateSpec[s].name, state_energy_[s]);
            }
            glDeleteBuffers(1, &buf);  // TRANSIENT: freed after show + audit
            if (pk > 0.0) {
                peak_ = pk;
            }
            write_display_texture();
            volume_dirty_ = false;
            montage_hold_ = kAtlasMontageFrames;
            if (synth_queue_.empty()) {
                collect_channel_pairs();
            }
            return;
        }
        if (!pair_queue_.empty()) {
            for (int c = 0; c < kAtlasPairsPerFrame && !pair_queue_.empty(); ++c) {
                evaluate_channel_pair(pair_queue_.back());
                pair_queue_.pop_back();
            }
            if (pair_queue_.empty()) {
                finalize_channel_table();
                // Free the channel-build 'from' cache; nothing else is resident
                // (montage + pairs were transient). Populations come from the
                // projection and collapse/deflation re-synthesize on demand -- NO
                // resident atlas -> ~1.2 GB runtime, and 512^3 is feasible.
                if (pair_from_buf_ != 0) {
                    glDeleteBuffers(1, &pair_from_buf_);
                    pair_from_buf_ = 0;
                }
                pair_from_idx_ = -1;
                atlas_done_ = true;
                cpu_is_truth_ = true;  // resume the untouched wavepacket
                refresh_title();
            }
        }
    }

    // Downward pairs worth a dipole integral: gap > 1e-3 skips both the
    // degenerate m-splittings (zero by construction here) and sub-mHa
    // radio-frequency channels whose omega^3 rates are irrelevant; the
    // |dl| = 1 filter applies the E1 selection rule analytically (the
    // synthesis KNOWS each state's l -- emergence was demonstrated in T5,
    // and 30 states would otherwise cost ~450 forbidden integrals).
    void collect_channel_pairs() {
        pair_queue_.clear();
        for (int from = 0; from < kNumStates; ++from) {
            for (int to = 0; to < kNumStates; ++to) {
                const bool downward =
                    state_energy_[static_cast<std::size_t>(from)] -
                        state_energy_[static_cast<std::size_t>(to)] >
                    1e-3;
                const bool e1_allowed =
                    std::abs(kStateSpec[from].l - kStateSpec[to].l) == 1;
                if (downward && e1_allowed) {
                    pair_queue_.push_back({from, to});
                }
            }
        }
    }

    void evaluate_channel_pair(const std::pair<int, int>& p) {
        const std::size_t from = static_cast<std::size_t>(p.first);
        const std::size_t to = static_cast<std::size_t>(p.second);
        const double gap = state_energy_[from] - state_energy_[to];
        // Transient endpoints (no resident atlas): cache the 'from' orbital
        // across its consecutive channels (pairs are grouped by 'from'), and
        // synthesize each 'to' fresh -- peak residency is 2 orbitals, never 91.
        if (pair_from_idx_ != p.first) {
            if (pair_from_buf_ != 0) {
                glDeleteBuffers(1, &pair_from_buf_);
            }
            pair_from_buf_ = synth_transient(p.first);
            pair_from_idx_ = p.first;
        }
        GLuint to_buf = synth_transient(p.second);
        const ses::DipoleMatrixElement d =
            engine_.dipole_between(*this, to_buf, pair_from_buf_);
        glDeleteBuffers(1, &to_buf);
        channels_.push_back(ShellChannel{
            p.first, p.second, ses::einstein_a(gap, ses::dipole_strength_sq(d)), 0.0});
        if (p.first == kP2Z && p.second == kS1) {
            dipole_z_ = std::abs(d.z);  // T3 laser E0, ready for free
        }
    }

    bool finalize_channel_table() {
        double a_max = 0.0;
        for (const ShellChannel& c : channels_) {
            a_max = std::max(a_max, c.a_true);
        }
        if (a_max <= 0.0) {
            channels_.clear();
            return false;
        }
        accel_display_ = kDecayGammaDisplay / a_max;
        for (ShellChannel& c : channels_) {
            c.gamma_display = c.a_true * accel_display_;
        }
        std::fprintf(stderr, "manifold: display acceleration x%.3e\n", accel_display_);
        for (int s = 0; s < kNumStates; ++s) {
            std::fprintf(stderr, "manifold: E(%s) = %.6f Ha\n", kStateSpec[s].name,
                         state_energy_[static_cast<std::size_t>(s)]);
        }
        for (const ShellChannel& c : channels_) {
            std::fprintf(stderr, "manifold: %s -> %s  A = %.3e /au  tau = %.3e au%s\n",
                         kStateSpec[c.from].name, kStateSpec[c.to].name, c.a_true,
                         c.a_true > 0.0 ? 1.0 / c.a_true : 0.0,
                         c.a_true < 1e-3 * a_max ? "  [forbidden/suppressed]" : "");
        }
        return true;
    }

    // T7: the blocking fallbacks are now thin wrappers over synthesis --
    // after the startup atlas everything is already cached, so these cost
    // nothing; if called early they synthesize just what is needed. All
    // need a current GL context only for buffer creation (ensure_state),
    // hence the makeCurrent bracket.
    bool prepare_ground_cache() {
        makeCurrent();
        const bool ok = ensure_state(kS1);
        doneCurrent();
        return ok;
    }

    // The laser pair (1s + 2p_z); dipole_z_ (the T3 drive strength) comes
    // from the channel table or is computed here if the table is not up.
    bool prepare_excited_cache() {
        makeCurrent();
        const bool ok = ensure_state(kS1) && ensure_state(kP2Z);
        if (ok && dipole_z_ == 0.0) {
            const ses::DipoleMatrixElement d =
                engine_.dipole_between_p(*this, state_buf_[kP2Z], state_is_fp16(kP2Z),
                                         state_buf_[kS1], state_is_fp16(kS1));
            dipole_z_ = std::abs(d.z);
        }
        doneCurrent();
        return ok;
    }

    bool prepare_p_triplet() {
        if (!prepare_excited_cache()) {
            return false;
        }
        makeCurrent();
        const bool ok = ensure_state(kP2X) && ensure_state(kP2Y);
        doneCurrent();
        return ok;
    }

    bool prepare_manifold_cache() {
        if (!channels_.empty()) {
            return true;
        }
        // One context bracket over the whole build: ensure_state uploads the
        // buffers and evaluate_channel_pair reduces the dipoles on the GPU, so
        // both need the context current (the dipoles no longer touch the CPU).
        makeCurrent();
        bool ok = true;
        for (int idx = 0; idx < kNumStates && ok; ++idx) {
            ok = ensure_state(idx);
        }
        if (ok) {
            collect_channel_pairs();
            while (!pair_queue_.empty()) {
                evaluate_channel_pair(pair_queue_.back());
                pair_queue_.pop_back();
            }
        }
        doneCurrent();
        if (!ok) {
            return false;
        }
        return finalize_channel_table();
    }

    // Total decay lifetime of a tracked state (sum over its channels), in
    // TRUE atomic units; 0 means no open channel (stable/metastable).
    double lifetime_of(int state) const {
        double a_sum = 0.0;
        for (const ShellChannel& c : channels_) {
            if (c.from == state) {
                a_sum += c.a_true;
            }
        }
        return a_sum > 0.0 ? 1.0 / a_sum : 0.0;
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
                QOpenGLWidget::keyPressEvent(e);
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
        makeCurrent();
        engine_.readback(*this, readback_buf_);
        doneCurrent();
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
        const auto& data = sim_.psi().data();
        psi_staging_.resize(data.size() * 2);
        double peak = 0.0;
        for (std::size_t i = 0; i < data.size(); ++i) {
            psi_staging_[2 * i] = static_cast<float>(data[i].real());
            psi_staging_[2 * i + 1] = static_cast<float>(data[i].imag());
            peak = std::max(peak, ses::norm_sq(data[i]));
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
                              .arg(static_cast<int>(pair_queue_.size()))
                        : QStringLiteral("  solving atom: %1 (%2/%3)")
                              .arg(QLatin1String(kStateSpec[synth_queue_.front()].name))
                              .arg(kNumStates -
                                   static_cast<int>(synth_queue_.size()) + 1)
                              .arg(kNumStates))
                 : QString()) +
            (decay_on_ && !channels_.empty()
                 ? QStringLiteral("  decay ON: tau(2p) %1 au, tau(2s) %2 au, x%3, "
                                  "photons %4%5")
                       .arg(lifetime_of(kP2Z), 0, 'e', 2)
                       .arg(lifetime_of(kS2), 0, 'e', 2)
                       .arg(accel_display_, 0, 'e', 1)
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

    // ---- GL uploads (current context required: called from initializeGL/paintGL) ----

    // Bridge psi to the display texture. The magnetic field now evolves psi
    // itself (MagneticPropagator on the GPU: diamagnetic in the potential +
    // exact three-shear paramagnetic rotation), so the display is just the
    // real wavefunction -- no display-only rotation trick.
    void write_display_texture() {
        engine_.write_psi_texture(*this, psi_tex_);
    }

    // XYZ orientation gizmo: three arrows from the origin -- X red, Y green,
    // Z blue -- baked into the mesh vertex format (per-vertex color). Uploaded
    // once; drawn each frame by draw_axes_gizmo().
    void upload_axes_gizmo() {
        struct Axis {
            ses::Vec3d dir;
            float r, g, b;
        };
        const Axis axes[3] = {
            {ses::Vec3d{1.0, 0.0, 0.0}, 0.95f, 0.25f, 0.25f},  // X red
            {ses::Vec3d{0.0, 1.0, 0.0}, 0.30f, 0.85f, 0.30f},  // Y green
            {ses::Vec3d{0.0, 0.0, 1.0}, 0.35f, 0.55f, 1.00f},  // Z blue
        };
        std::vector<float> interleaved;
        for (const Axis& ax : axes) {
            const ses::Mesh arrow = ses::arrow_mesh(ax.dir, 1.0, 0.045, 0.13, 0.32, 16);
            for (std::size_t i = 0; i < arrow.vertices.size(); ++i) {
                const ses::Vec3d& p = arrow.vertices[i];
                const ses::Vec3d& n = arrow.normals[i];
                interleaved.push_back(static_cast<float>(p.x));
                interleaved.push_back(static_cast<float>(p.y));
                interleaved.push_back(static_cast<float>(p.z));
                interleaved.push_back(static_cast<float>(n.x));
                interleaved.push_back(static_cast<float>(n.y));
                interleaved.push_back(static_cast<float>(n.z));
                interleaved.push_back(ax.r);
                interleaved.push_back(ax.g);
                interleaved.push_back(ax.b);
            }
        }
        gizmo_vertex_count_ = static_cast<int>(interleaved.size() / 9);
        glGenVertexArrays(1, &gizmo_vao_);
        glBindVertexArray(gizmo_vao_);
        glGenBuffers(1, &gizmo_vbo_);
        glBindBuffer(GL_ARRAY_BUFFER, gizmo_vbo_);
        glBufferData(GL_ARRAY_BUFFER,
                     static_cast<GLsizeiptr>(interleaved.size() * sizeof(float)),
                     interleaved.data(), GL_STATIC_DRAW);
        constexpr GLsizei kStride = 9 * sizeof(float);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, kStride, reinterpret_cast<void*>(0));
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, kStride,
                              reinterpret_cast<void*>(3 * sizeof(float)));
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, kStride,
                              reinterpret_cast<void*>(6 * sizeof(float)));
        glEnableVertexAttribArray(2);
        glBindVertexArray(0);

        // Dynamic VBO for the billboarded "z" glyph (rebuilt each frame in
        // draw_z_label); same attribute layout as the arrows.
        glGenVertexArrays(1, &z_label_vao_);
        glBindVertexArray(z_label_vao_);
        glGenBuffers(1, &z_label_vbo_);
        glBindBuffer(GL_ARRAY_BUFFER, z_label_vbo_);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, kStride, reinterpret_cast<void*>(0));
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, kStride,
                              reinterpret_cast<void*>(3 * sizeof(float)));
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, kStride,
                              reinterpret_cast<void*>(6 * sizeof(float)));
        glEnableVertexAttribArray(2);
        glBindVertexArray(0);
    }

    // Static warm-colored sphere at the nucleus, in the mesh vertex format.
    void upload_proton_marker() {
        const ses::Mesh sphere = ses::sphere_mesh(ses::Vec3d{}, kProtonMarkerRadius, 16, 24);
        std::vector<float> interleaved;
        interleaved.reserve(sphere.vertices.size() * 9);
        for (std::size_t i = 0; i < sphere.vertices.size(); ++i) {
            const ses::Vec3d& p = sphere.vertices[i];
            const ses::Vec3d& n = sphere.normals[i];
            interleaved.push_back(static_cast<float>(p.x));
            interleaved.push_back(static_cast<float>(p.y));
            interleaved.push_back(static_cast<float>(p.z));
            interleaved.push_back(static_cast<float>(n.x));
            interleaved.push_back(static_cast<float>(n.y));
            interleaved.push_back(static_cast<float>(n.z));
            interleaved.push_back(1.0f);   // warm proton color
            interleaved.push_back(0.45f);
            interleaved.push_back(0.20f);
        }
        proton_vertex_count_ = static_cast<int>(sphere.vertices.size());
        glGenVertexArrays(1, &proton_vao_);
        glBindVertexArray(proton_vao_);
        glGenBuffers(1, &proton_vbo_);
        glBindBuffer(GL_ARRAY_BUFFER, proton_vbo_);
        glBufferData(GL_ARRAY_BUFFER,
                     static_cast<GLsizeiptr>(interleaved.size() * sizeof(float)),
                     interleaved.data(), GL_STATIC_DRAW);
        constexpr GLsizei kStride = 9 * sizeof(float);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, kStride, reinterpret_cast<void*>(0));
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, kStride,
                              reinterpret_cast<void*>(3 * sizeof(float)));
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, kStride,
                              reinterpret_cast<void*>(6 * sizeof(float)));
        glEnableVertexAttribArray(2);
        glBindVertexArray(0);
    }

    void upload_box_cube() {
        const ses::Grid3D& g = sim_.grid();
        const float lo[3] = {static_cast<float>(g.x.xmin), static_cast<float>(g.y.xmin),
                             static_cast<float>(g.z.xmin)};
        const float hi[3] = {static_cast<float>(g.x.xmax), static_cast<float>(g.y.xmax),
                             static_cast<float>(g.z.xmax)};
        // 12 triangles, CCW seen from OUTSIDE (verified per-face normals).
        const float v[] = {
            lo[0],lo[1],lo[2], lo[0],lo[1],hi[2], lo[0],hi[1],hi[2],
            lo[0],lo[1],lo[2], lo[0],hi[1],hi[2], lo[0],hi[1],lo[2],
            hi[0],lo[1],lo[2], hi[0],hi[1],lo[2], hi[0],hi[1],hi[2],
            hi[0],lo[1],lo[2], hi[0],hi[1],hi[2], hi[0],lo[1],hi[2],
            lo[0],lo[1],lo[2], hi[0],lo[1],lo[2], hi[0],lo[1],hi[2],
            lo[0],lo[1],lo[2], hi[0],lo[1],hi[2], lo[0],lo[1],hi[2],
            lo[0],hi[1],lo[2], lo[0],hi[1],hi[2], hi[0],hi[1],hi[2],
            lo[0],hi[1],lo[2], hi[0],hi[1],hi[2], hi[0],hi[1],lo[2],
            lo[0],lo[1],lo[2], lo[0],hi[1],lo[2], hi[0],hi[1],lo[2],
            lo[0],lo[1],lo[2], hi[0],hi[1],lo[2], hi[0],lo[1],lo[2],
            lo[0],lo[1],hi[2], hi[0],lo[1],hi[2], hi[0],hi[1],hi[2],
            lo[0],lo[1],hi[2], hi[0],hi[1],hi[2], lo[0],hi[1],hi[2],
        };
        glGenVertexArrays(1, &cube_vao_);
        glBindVertexArray(cube_vao_);
        glGenBuffers(1, &cube_vbo_);
        glBindBuffer(GL_ARRAY_BUFFER, cube_vbo_);
        glBufferData(GL_ARRAY_BUFFER, sizeof(v), v, GL_STATIC_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float),
                              reinterpret_cast<void*>(0));
        glEnableVertexAttribArray(0);
        glBindVertexArray(0);
    }

    void create_textures() {
        const ses::Grid3D& g = sim_.grid();

        glGenTextures(1, &psi_tex_);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_3D, psi_tex_);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
        glTexImage3D(GL_TEXTURE_3D, 0, GL_RG32F, g.x.n, g.y.n, g.z.n, 0, GL_RG, GL_FLOAT,
                     nullptr);

        // The TESTED cyclic phase colormap, baked to a repeating 1D texture.
        const std::vector<ses::Rgb> lut = ses::phase_lut(kPhaseLutSize);
        std::vector<float> lut_f;
        lut_f.reserve(lut.size() * 3);
        for (const ses::Rgb& c : lut) {
            lut_f.push_back(static_cast<float>(c.r));
            lut_f.push_back(static_cast<float>(c.g));
            lut_f.push_back(static_cast<float>(c.b));
        }
        glGenTextures(1, &phase_tex_);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_1D, phase_tex_);
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_REPEAT);  // cyclic!
        glTexImage1D(GL_TEXTURE_1D, 0, GL_RGB32F, kPhaseLutSize, 0, GL_RGB, GL_FLOAT,
                     lut_f.data());
    }

    void upload_volume() {
        const ses::Grid3D& g = sim_.grid();
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_3D, psi_tex_);
        glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, g.x.n, g.y.n, g.z.n, GL_RG, GL_FLOAT,
                        psi_staging_.data());
    }

    void upload_mesh() {
        std::vector<float> interleaved;
        interleaved.reserve(mesh_.vertices.size() * 9);
        for (std::size_t i = 0; i < mesh_.vertices.size(); ++i) {
            const ses::Vec3d& p = mesh_.vertices[i];
            const ses::Vec3d& n = mesh_.normals[i];
            const ses::Rgb& c = colors_[i];
            interleaved.push_back(static_cast<float>(p.x));
            interleaved.push_back(static_cast<float>(p.y));
            interleaved.push_back(static_cast<float>(p.z));
            interleaved.push_back(static_cast<float>(n.x));
            interleaved.push_back(static_cast<float>(n.y));
            interleaved.push_back(static_cast<float>(n.z));
            interleaved.push_back(static_cast<float>(c.r));
            interleaved.push_back(static_cast<float>(c.g));
            interleaved.push_back(static_cast<float>(c.b));
        }
        vertex_count_ = static_cast<int>(mesh_.vertices.size());
        glBindBuffer(GL_ARRAY_BUFFER, mesh_vbo_);
        glBufferData(GL_ARRAY_BUFFER,
                     static_cast<GLsizeiptr>(interleaved.size() * sizeof(float)),
                     interleaved.data(), GL_DYNAMIC_DRAW);
    }

    GLuint compile_shader(GLenum type, const char* src) {
        const GLuint shader = glCreateShader(type);
        glShaderSource(shader, 1, &src, nullptr);
        glCompileShader(shader);
        GLint ok = GL_FALSE;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
        if (ok != GL_TRUE) {
            char log[2048];
            glGetShaderInfoLog(shader, sizeof(log), nullptr, log);
            fatal_gl_error("shader compile failed", QString::fromLatin1(log));
        }
        return shader;
    }

    GLuint link_program(const char* vs_src, const char* fs_src) {
        const GLuint vs = compile_shader(GL_VERTEX_SHADER, vs_src);
        const GLuint fs = compile_shader(GL_FRAGMENT_SHADER, fs_src);
        const GLuint prog = glCreateProgram();
        glAttachShader(prog, vs);
        glAttachShader(prog, fs);
        glLinkProgram(prog);
        GLint ok = GL_FALSE;
        glGetProgramiv(prog, GL_LINK_STATUS, &ok);
        if (ok != GL_TRUE) {
            char log[2048];
            glGetProgramInfoLog(prog, sizeof(log), nullptr, log);
            fatal_gl_error("program link failed", QString::fromLatin1(log));
        }
        glDeleteShader(vs);
        glDeleteShader(fs);
        return prog;
    }

    ses::WavepacketSimulation sim_;
    ViewMode mode_ = ViewMode::Cloud;
    Stepping stepping_ = Stepping::RealTime;

    // GPU stepping state (docs/GPU_PLAN.md G5). cpu_is_truth_ is the single
    // sync invariant: true -> sim_.psi() is current, false -> the engine's
    // SSBO is ahead and must be read back before any CPU-side operation.
    ses_gpu::GpuEngine engine_;
    bool gpu_ok_ = false;
    bool cpu_is_truth_ = true;
    int pending_gpu_steps_ = 0;
    bool pending_energy_measure_ = false;  // Key E: serviced in paintGL
    bool gpu_title_due_ = false;
    double gpu_time_ = 0.0;
    double norm_display_ = 1.0;
    double relax_energy_display_ = 0.0;
    double radiated_power_ = 0.0;  // semiclassical Larmor power (au)
    std::vector<float> readback_buf_;

    // Transitions arc T1/T4/T5: the cached eigenstate manifold, the decay
    // channel table (built on first decay toggle), and jump bookkeeping.
    std::array<GLuint, kNumStates> state_buf_{};
    // Atlas storage precision, decided at startup from free VRAM: the fp32
    // n<=6 manifold is ~12 GB and oversubscribes a small card. Fp16 halves it.
    ses::GpuPrecision state_precision_ = ses::GpuPrecision::Fp32;
    // Orbital-free projection (Phase 5): grid norm of each raw (u/r)Ylm orbital
    // (captured at synthesis) normalizes projected populations to match the
    // retired inner_with_psi path; proj_ready_ once the static index is uploaded.
    std::array<double, kNumStates> state_norm2_{};
    bool proj_ready_ = false;
    std::array<double, kNumStates> state_energy_{};
    std::vector<ShellChannel> channels_;
    double accel_display_ = 0.0;  // common display acceleration factor
    QString last_jump_;
    QString last_measure_;  // last energy-measurement readout (Key E)
    int last_measured_index_ = -2;  // last energy-measurement outcome (selftest)
    QString relax_label_ = QStringLiteral("2p");
    std::vector<GLuint> relax_deflate_;        // live RelaxingExcited deflation set
    std::vector<GLuint> relax_deflate_owned_;  // owned transient buffers to free
    int pair_from_idx_ = -1;                   // channel-build 'from' cache (transient)
    GLuint pair_from_buf_ = 0;
    double relax_prev_energy_ = 0.0;     // T7 auto-complete plateau tracking
    int relax_plateau_ = 0;

    // T7: startup atlas build (radial solve + synthesis, chunked in paint).
    std::vector<int> synth_queue_;
    std::vector<std::pair<int, int>> pair_queue_;
    int montage_hold_ = 0;
    bool atlas_done_ = false;
    ses::RadialGrid radial_grid_{};
    std::array<std::vector<double>, kNumLevels> radial_u_;
    std::array<double, kNumLevels> radial_energy_{};
    bool radial_ready_ = false;
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
    double dipole_z_ = 0.0;   // |<2p_z| z |1s>| from the cached states
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

    GLuint mesh_program_ = 0;
    GLuint mesh_vao_ = 0;
    GLuint mesh_vbo_ = 0;
    GLint mesh_mvp_loc_ = -1;
    GLint mesh_eye_loc_ = -1;
    int vertex_count_ = 0;

    GLuint proton_vao_ = 0;
    GLuint proton_vbo_ = 0;
    int proton_vertex_count_ = 0;
    GLuint gizmo_vao_ = 0;
    GLuint gizmo_vbo_ = 0;
    int gizmo_vertex_count_ = 0;
    GLuint z_label_vao_ = 0;
    GLuint z_label_vbo_ = 0;  // billboarded "z" glyph, rebuilt each frame
    GLuint mask_buf_ = 0;     // boundary absorber (mask, 0) complex buffer
    std::mt19937 rng_{std::random_device{}()};

    GLuint volume_program_ = 0;
    GLuint cube_vao_ = 0;
    GLuint cube_vbo_ = 0;
    GLuint psi_tex_ = 0;
    GLuint phase_tex_ = 0;
    GLint vol_mvp_loc_ = -1;
    GLint vol_eye_loc_ = -1;
    GLint vol_boxmin_loc_ = -1;
    GLint vol_boxmax_loc_ = -1;
    GLint vol_invpeak_loc_ = -1;
    GLint vol_absorb_loc_ = -1;
    GLint vol_pcenter_loc_ = -1;
    GLint vol_pradius_loc_ = -1;
    GLint vol_pcolor_loc_ = -1;

    double azimuth_ = 0.6;
    double elevation_ = 0.4;
    double distance_ = 150.0;  // default frames ~+-62 Bohr (45 deg fovy): the
                               // n<=6 manifold body, incl. the ~60 Bohr 6s/6p
                               // (wheel in toward 4 for a close-up of small ones)
    QPointF last_pos_;
};

// Selftest helper: poll until the startup atlas build has produced the
// channel table, then run the arc (slower GPUs just stretch the wait).
template <typename F>
void run_when_manifold_ready(Viewport* viewport, F fn) {
    if (viewport->manifold_ready()) {
        fn();
        return;
    }
    QTimer::singleShot(500, viewport,
                       [viewport, fn] { run_when_manifold_ready(viewport, fn); });
}

}  // namespace

int main(int argc, char** argv) {
    QSurfaceFormat format;
    format.setVersion(4, 3);
    format.setProfile(QSurfaceFormat::CoreProfile);
    format.setDepthBufferSize(24);
    // No MSAA: the volume ray marcher is a full-screen fragment pass where
    // 4x multisampling only multiplies its cost (it smooths nothing but
    // the cube edges) -- at 256^3 that budget belongs to the physics.
    QSurfaceFormat::setDefaultFormat(format);

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

    // Headless-ish regression of the decay demo arc: prepare 2p, return to
    // real time, enable decay, and require at least one quantum jump.
    // (After the first jump the atom sits in 1s with P_e ~ 0, so exactly one
    // photon is the physically expected outcome without a re-pump laser.)
    // Selftest arcs wait for the startup atlas build (run_when_manifold_ready)
    // and are then CHAINED, so a slower GPU stretches the run instead of
    // false-failing a wall-clock verdict. Decay is ON by default (T6);
    // photon verdicts count from a baseline captured at the arc's start.
    if (app.arguments().contains(QStringLiteral("--selftest-decay"))) {
        run_when_manifold_ready(viewport, [viewport, &app] {
            viewport->relax_to_excited();  // caches ready: no block
            QTimer::singleShot(13500, viewport, [viewport, &app] {
                const long long baseline = viewport->photon_count();
                viewport->set_real_time();  // decay is already armed
                QTimer::singleShot(30000, viewport, [viewport, &app, baseline] {
                    const long long fresh = viewport->photon_count() - baseline;
                    std::fprintf(stderr, "selftest-decay: photons = %lld  [%s]\n",
                                 fresh, fresh >= 1 ? "PASS" : "FAIL");
                    app.exit(fresh >= 1 ? 0 : 1);
                });
            });
        });
    }

    // Headless-ish regression of the energy-measurement feature: relax to 1s
    // (decay OFF so the prepared state stays put), then a projective energy
    // measurement must collapse onto -- and report -- the 1s eigenstate.
    if (app.arguments().contains(QStringLiteral("--selftest-energy"))) {
        run_when_manifold_ready(viewport, [viewport, &app] {
            viewport->toggle_decay();  // OFF: keep the relaxed state stationary
            viewport->set_relaxing();  // cool to 1s
            QTimer::singleShot(12000, viewport, [viewport, &app] {
                viewport->measure_energy_now();
                QTimer::singleShot(1500, viewport, [viewport, &app] {
                    const int idx = viewport->last_measured_index();
                    const bool pass = idx == kS1;
                    std::fprintf(
                        stderr, "selftest-energy: measured %s  [%s]\n",
                        idx >= 0 ? kStateSpec[static_cast<std::size_t>(idx)].name
                                 : "outside-manifold",
                        pass ? "PASS" : "FAIL");
                    app.exit(pass ? 0 : 1);
                });
            });
        });
    }

    // Headless-ish regression of the static E-field: relax to 1s (symmetric,
    // <z> ~ 0), switch on a sub-ionization +z field, and require the cloud to
    // polarize -- <z> shifts measurably off center (Stark). Proves the field
    // actually acts on the cloud.
    if (app.arguments().contains(QStringLiteral("--selftest-efield"))) {
        run_when_manifold_ready(viewport, [viewport, &app] {
            viewport->toggle_decay();  // OFF: keep the state put
            viewport->set_relaxing();  // cool to 1s
            QTimer::singleShot(12000, viewport, [viewport, &app] {
                const double z0 = viewport->mean_z();
                viewport->set_real_time();
                viewport->set_efield_e0(0.02);  // sub-ionization: clean polarization
                QTimer::singleShot(15000, viewport, [viewport, &app, z0] {
                    const double z1 = viewport->mean_z();
                    const bool pass = std::abs(z1 - z0) > 0.03;
                    std::fprintf(stderr,
                                 "selftest-efield: <z> %.4f -> %.4f Bohr  [%s]\n",
                                 z0, z1, pass ? "PASS" : "FAIL");
                    app.exit(pass ? 0 : 1);
                });
            });
        });
    }

    // Headless-ish regression of the T3 pump demo: relax to 1s, laser ON
    // (Z-pol), require a Rabi peak P(2pz) >= 0.5; then decay ON as well and
    // require >= 2 photons -- repeated absorb/emit cycles. A ground-start
    // run WITHOUT the pump emits zero photons, so 2 is unambiguous.
    if (app.arguments().contains(QStringLiteral("--selftest-rabi"))) {
        run_when_manifold_ready(viewport, [viewport, &app] {
            viewport->toggle_decay();  // OFF: study the clean coherent flop
            viewport->set_relaxing();  // cool to 1s
            QTimer::singleShot(11500, viewport, [viewport, &app] {
                viewport->set_real_time();
                viewport->toggle_laser();  // cached: instant
                // 256^3 runs ~3 au/s of sim time: the half-flop (pi/Omega
                // ~ 79 au) needs most of this window.
                QTimer::singleShot(60000, viewport, [viewport, &app] {
                    const double peak = viewport->peak_excited_population();
                    std::fprintf(stderr, "selftest-rabi: peak P(2pz) = %.3f  [%s]\n",
                                 peak, peak >= 0.5 ? "PASS" : "FAIL");
                    if (peak < 0.5) {
                        app.exit(1);
                        return;
                    }
                    const long long baseline = viewport->photon_count();
                    viewport->toggle_decay();  // back ON: fluorescence
                    QTimer::singleShot(180000, viewport,
                                       [viewport, &app, baseline] {
                        const long long fresh =
                            viewport->photon_count() - baseline;
                        std::fprintf(stderr,
                                     "selftest-rabi: photons = %lld  [%s]\n",
                                     fresh, fresh >= 2 ? "PASS" : "FAIL");
                        app.exit(fresh >= 2 ? 0 : 1);
                    });
                });
            });
        });
    }

    // T7 regression: the CASCADE. Excite 3d_z2 instantly (key-5 path) and
    // require at least two photons: 3d cannot reach 1s directly (dl = 2),
    // so two photons prove the chain 3d -> 2p -> 1s fired through the
    // multi-level channel table.
    if (app.arguments().contains(QStringLiteral("--selftest-cascade"))) {
        run_when_manifold_ready(viewport, [viewport, &app] {
            const long long baseline = viewport->photon_count();
            viewport->excite_n3();  // first in the cycle: 3d_z2
            QTimer::singleShot(90000, viewport, [viewport, &app, baseline] {
                const long long fresh = viewport->photon_count() - baseline;
                std::fprintf(stderr, "selftest-cascade: photons = %lld  [%s]\n",
                             fresh, fresh >= 2 ? "PASS" : "FAIL");
                app.exit(fresh >= 2 ? 0 : 1);
            });
        });
    }

    // Magnetic regression: a field along z rotates 2p_x -> 2p_y at
    // omega_L = B/2. Prepare 2p_x (decay off, for pure precession), turn B on,
    // and require P(2p_y) to rise past 0.3 -- proving psi ITSELF precesses (the
    // old display trick left psi pristine, so P(2p_y) would stay 0). Probed
    // periodically so the sin^2 oscillation phase cannot alias the verdict.
    if (app.arguments().contains(QStringLiteral("--selftest-magnetic"))) {
        run_when_manifold_ready(viewport, [viewport, &app] {
            viewport->toggle_decay();            // decay OFF: pure precession
            viewport->debug_prepare_state(kP2X);  // psi = 2p_x
            // Modest field: omega_L = B/2 precession with only a MILD
            // diamagnetic term, so 2p_x rotates substantially into 2p_y (a
            // large B would diamagnetically deform the state and cap the
            // overlap -- itself why the display trick was wrong). Real-time
            // magnetic stepping is slow (~0.4 au/s), so the window is long.
            viewport->set_bfield_b(0.08);         // B along z (default axis)
            auto max_py = std::make_shared<double>(0.0);
            auto* probe = new QTimer(viewport);
            QObject::connect(probe, &QTimer::timeout, viewport, [viewport, max_py] {
                *max_py = std::max(*max_py, viewport->probe_population(kP2Y));
            });
            probe->start(1500);
            QTimer::singleShot(90000, viewport, [&app, max_py, probe] {
                probe->stop();
                std::fprintf(stderr, "selftest-magnetic: max P(2p_y) = %.3f  [%s]\n",
                             *max_py, *max_py > 0.3 ? "PASS" : "FAIL");
                app.exit(*max_py > 0.3 ? 0 : 1);
            });
        });
    }

    // T5 regression: (a) deterministic physics of the computed channel
    // table -- the selection rule A(2s->1s) ~ 0 and the 2p degeneracy --
    // then (b) live wiring of the non-2p_z channels: an X-polarized pump
    // from 1s can only fluoresce through 2p_x, so new photons prove the
    // multi-channel trial fires beyond the old single 2p_z channel.
    if (app.arguments().contains(QStringLiteral("--selftest-manifold"))) {
        run_when_manifold_ready(viewport, [viewport, &app] {
            // Deterministic physics of the freshly built channel table.
            const double a_pz = viewport->channel_a(kP2Z, kS1);
            const double a_px = viewport->channel_a(kP2X, kS1);
            const double a_2s1s = viewport->channel_a(kS2, kS1);
            std::fprintf(stderr,
                         "selftest-manifold: A(2pz->1s)=%.3e A(2px->1s)=%.3e "
                         "A(2s->1s)=%.3e  E(1s)=%.4f E(2pz)=%.4f E(2s)=%.4f\n",
                         a_pz, a_px, a_2s1s, viewport->state_energy(kS1),
                         viewport->state_energy(kP2Z), viewport->state_energy(kS2));
            const bool selection = a_pz > 0.0 && a_2s1s < 1e-3 * a_pz;
            const bool degeneracy = a_pz > 0.0 && std::abs(a_px / a_pz - 1.0) < 0.05;
            const bool ordering =
                viewport->state_energy(kS1) < viewport->state_energy(kP2Z) &&
                viewport->state_energy(kS1) < viewport->state_energy(kS2);
            // T7: the n = 3 shell -- cascade paths open, Delta-l selection
            // rules hold (3s -> 1s and 3d -> 1s are E1-forbidden).
            const double a_3s2p = viewport->channel_a(k3S, kP2Z);
            const double a_3d2p = viewport->channel_a(k3DZ0, kP2Z);
            const double a_3s1s = viewport->channel_a(k3S, kS1);
            const double a_3d1s = viewport->channel_a(k3DZ0, kS1);
            const bool cascade = a_3s2p > 0.0 && a_3d2p > 0.0;
            const bool dl_rule =
                a_3d2p > 0.0 && a_3s1s < 1e-3 * a_3d2p && a_3d1s < 1e-3 * a_3d2p;
            std::fprintf(stderr,
                         "selftest-manifold: A(3s->2pz)=%.3e A(3dz2->2pz)=%.3e "
                         "A(3s->1s)=%.3e A(3dz2->1s)=%.3e\n",
                         a_3s2p, a_3d2p, a_3s1s, a_3d1s);
            std::fprintf(stderr,
                         "selftest-manifold: selection %s, degeneracy %s, ordering "
                         "%s, cascade %s, dl-rule %s\n",
                         selection ? "PASS" : "FAIL", degeneracy ? "PASS" : "FAIL",
                         ordering ? "PASS" : "FAIL", cascade ? "PASS" : "FAIL",
                         dl_rule ? "PASS" : "FAIL");
            if (!(selection && degeneracy && ordering && cascade && dl_rule)) {
                app.exit(1);
                return;
            }
            viewport->set_relaxing();  // cool to 1s for the X-pol pump
            QTimer::singleShot(12000, viewport, [viewport, &app] {
                viewport->set_real_time();
                viewport->toggle_laser();  // Z (cached: no block)
                viewport->toggle_laser();  // -> X
                const long long baseline = viewport->photon_count();
                QTimer::singleShot(60000, viewport, [viewport, &app, baseline] {
                    const long long fresh = viewport->photon_count() - baseline;
                    std::fprintf(stderr,
                                 "selftest-manifold: x-pol photons = %lld  [%s]\n",
                                 fresh, fresh >= 2 ? "PASS" : "FAIL");
                    app.exit(fresh >= 2 ? 0 : 1);
                });
            });
        });
    }

    return app.exec();
}
