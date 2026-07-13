#pragma once

// The hydrogen scenario (ScenarioDirector implementation): the CPU truth
// session, the ses_vk engine, the AtomModel, and the atom demo state machine
// (atlas build, decay, laser, fields, measurement, relaxation), plus the
// cpu_is_truth_ sync invariant. Qt-free.

#include "atom_model.hpp"
#include "manifold_spec.hpp"
#include "scenario.hpp"
#include "vk_engine.hpp"

#include <core/colormap.hpp>
#include <core/decay.hpp>
#include <core/emission.hpp>
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
#include <core/vec.hpp>
#include <core/vram_budget.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

namespace ses_shell {

enum class ViewMode { Cloud, Surface };
enum class Stepping { RealTime, Relaxing, RelaxingExcited };
enum class LaserPol { Off, Z, X };

constexpr int kStepsPerTick = 1;
// Backlog cap: a stalled paint cannot spiral (time is credited at execution,
// dropped ticks drop cleanly). The cap does NOT set the sim rate (measured:
// 8 vs 32 gave the same au/s) -- only the per-paint block length, i.e. the
// UI freeze. 4 halves the freeze of the old 8 at zero throughput cost.
constexpr int kMaxPendingGpuSteps = 4;
constexpr int kRelaxStepsPerTick = 1;
constexpr double kRelaxDtau = 0.05;
constexpr double kIsoFraction = 0.25;
constexpr double kMeasureSigma = 1.25;  // Bohr; sigma keeps the measurement
                                        // back-action 3/(8 sigma^2) = 0.24 Ha
                                        // under the 0.5 Ha binding (the old
                                        // 0.625 ionized on nearly every click)
// Display decay rate: tau_display ~ 8 au (~3 s wall); true lifetimes are
// ~1e8 au. The title reports the true lifetime and the acceleration factor.
constexpr double kDecayGammaDisplay = 0.125;
constexpr double kHaToEv = 27.211386;  // 1 Hartree in eV
constexpr double kAuToFs = 2.4188843e-2;  // 1 au of time in femtoseconds
constexpr double kAbsorbWidth = 10.0;  // Bohr; boundary absorber layer
                                       // (interior +-70 untouched; real-time only)
// Laser: E0 = kRabiTargetOmega / |<2p|z|1s>| (target Rabi frequency over the
// computed dipole element); the carrier is tuned to the GRID resonance, not
// the textbook 0.375 -- see toggle_laser.
constexpr double kRabiTargetOmega = 0.04;
constexpr int kLaserStepsPerTick = 6;  // the pump demo runs hotter than 1x

constexpr int kAtlasMontageFrames = 3;  // frames each synthesized orbital shows
constexpr int kAtlasPairsPerFrame = 4;  // dipole pairs evaluated per paint

// Potential: bare -Z/r with only the nucleus cell regularized (analytic cell
// average), so synthesized orbitals stay eigenstates of the propagated
// Hamiltonian; the s-state cusp gap shows up in the startup h-audit.
inline ses::WavepacketSimulation make_simulation() {
    // +-80 Bohr / 256^3 (power-of-two FFT): holds the n <= 6 shell; no
    // resident atlas. This Gaussian is only the pre-solve placeholder and
    // the no-GPU fallback: the GPU path replaces it with a random bound
    // n <= 5 superposition at the atlas finale (seed_bound_superposition) --
    // a free packet, unlike a bound superposition, leaks past the box.
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

class HydrogenDirector final : public ScenarioDirector {
public:
    HydrogenDirector() : sim_(make_simulation()) {
        remesh();
        stage_volume();
    }

    const ses::Grid3D& grid() const override { return sim_.grid(); }

    // ---- lifecycle ----

    // COMPUTE setup: engine init (fp32 tables verified by sesolver_vkcheck),
    // atlas precision from probed VRAM, atom solve, projection index,
    // absorber mask. Any failure falls back to CPU stepping.
    void init_compute(ses_vk::DeviceContext& ctx, bool device_ok,
                      std::int64_t free_vram_bytes) override {
        compute_attempted_ = true;
        gpu_ok_ = device_ok &&
                  engine_.initialize(ctx, sim_.grid(),
                                     ses_shell::app_engine_blobs(sim_.grid().x.n),
                                     sim_.potential(), sim_.dt(),
                                     sim_.psi().data());
        if (gpu_ok_) {
            // Atlas precision from free VRAM: a resident fp32 manifold
            // (~12 GB at 256^3) oversubscribes a small card and WDDM paging
            // collapses the frame rate; fp16 halves it. Unmeasurable -> fp32.
            constexpr std::int64_t kVramHeadroomBytes = 2LL * 1024 * 1024 * 1024;
            const std::int64_t bytes_per_state =
                static_cast<std::int64_t>(sim_.grid().size()) * 2 *
                static_cast<std::int64_t>(sizeof(float));
            bool atlas_fits = true;
            atom_.set_precision(ses::choose_state_precision(
                free_vram_bytes, kNumStates, bytes_per_state,
                kVramHeadroomBytes, &atlas_fits));
            std::fprintf(stderr, "vram: atlas precision = %s%s\n",
                         atom_.precision() == ses::GpuPrecision::Fp16 ? "fp16 (half)"
                                                                     : "fp32",
                         atlas_fits ? ""
                                    : "  [WARNING: even fp16 is tight -- consider a "
                                      "smaller box or manifold]");
            // Relax tables are TRANSIENT (uploaded on Key-2/3/4, freed on
            // completion); only the gradient upload stays fallible-fatal.
            if (!engine_.set_potential_gradient(sim_.potential())) {
                std::fprintf(stderr, "engine: gradient setup failed -- "
                                     "falling back to CPU stepping\n");
                gpu_ok_ = false;
                decay_on_ = false;
                atlas_done_ = true;
                return;
            }
            // Radial solve up front (all levels to n = 10); the 3D manifold
            // is then synthesized chunked across frames so decay is armed by
            // default.
            atom_.solve_radial_atom(sim_.grid().x.xmax);
            // Orbital-free projection index: static counting-sort geometry,
            // uploaded once; populations come from ONE project_psi deposit
            // pass instead of per-state inner products.
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
            // Boundary absorber: (mask, 0) complex buffer (interior = 1) so
            // the elementwise multiply damps outgoing flux each real-time step.
            absorber_on_ = engine_.set_absorber(
                ses::absorbing_mask(sim_.grid(), kAbsorbWidth));
            // A slider moved before gpu_ok_ stored its value but could not
            // upload the augmented half-potential: re-apply to match the UI.
            if (efield_e0_ > 0.0 || bfield_b_ > 0.0) {
                upload_field_tables();
            }
        } else {
            decay_on_ = false;  // jump trials are GPU-only
            atlas_done_ = true;
        }
    }

    // The widget's device is about to go away: tear down everything the
    // engine created on it. GPU simulation state dies with it.
    void release_gpu() override {
        engine_.destroy();
        gpu_ok_ = false;
    }

    // GPU stepping covers the Cloud view in BOTH real and imaginary time;
    // measure and surface meshing run on the CPU double session, synced
    // through the single cpu_is_truth_ invariant.
    bool use_gpu_path() const { return gpu_ok_ && mode_ == ViewMode::Cloud; }

    // ---- the compute half of a frame ----
    // Engine stepping, atlas build, measurement service, decay/laser trials.
    // Runs once per paint, BEFORE the widget frame (engine offscreen frames
    // are illegal mid-frame).
    void run_frame() override {
        // The atlas build advances regardless of view mode: a Tab to Surface
        // during startup must not wedge solving() forever.
        if (gpu_ok_ && !atlas_done_) {
            run_atlas_chunk();
            pending_gpu_steps_ = 0;
            if (gpu_title_due_) {
                gpu_title_due_ = false;
                title_dirty_ = true;
            }
            return;
        }
        if (use_gpu_path()) {
            if (cpu_is_truth_) {
                // CPU state authoritative: refresh the brightness normalizer
                // (post-M collapse, post-R reset), then upload.
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
                // first frame) the display would keep the stale cloud.
                write_display_texture();
            }
            // Projective ENERGY measurement (Key E): sample n from
            // P_n = |<phi_n|psi>|^2 and collapse; the deficit 1 - sum(P_n) is
            // the continuum outcome (n = -1, manifold projected OUT).
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
                    reset_ionized_tally();
                    write_display_texture();
                    last_measure_ = strf(
                        "%s (E %.3f eV)",
                        kStateSpec[static_cast<std::size_t>(n)].name,
                        atom_.state_energy(n) * kHaToEv);
                } else {
                    // A continuum verdict must not leave bound populations
                    // behind: psi <- (1 - P)|psi>, renormalized. Estimate the
                    // residual BEFORE mutating psi -- renormalizing an
                    // fp32-noise remnant would amplify garbage ~1000x into a
                    // "physical" cloud, so below the floor psi stays as is.
                    double bound = 0.0;
                    for (double p : pop) {
                        bound += p;
                    }
                    const double residual = engine_.norm_and_peak().sum - bound;
                    if (residual > 1e-4) {
                        for (int s = 0; s < kNumStates; ++s) {
                            if (pop[static_cast<std::size_t>(s)] < 1e-9) {
                                continue;
                            }
                            const ses::Complex<double> c =
                                atom_.project_state_amplitude(engine_, s);
                            const int buf = atom_.synth_transient(engine_, s);
                            if (buf >= 0) {
                                engine_.add_state_into_psi(buf, -c.real(),
                                                           -c.imag());
                                engine_.release_state(buf);
                            }
                        }
                        const ses_vk::Engine::NormPeak np =
                            engine_.norm_and_peak();
                        if (np.sum > 1e-12) {
                            engine_.scale(
                                static_cast<float>(1.0 / std::sqrt(np.sum)));
                        }
                        reset_ionized_tally();
                        write_display_texture();
                    }
                    last_measure_ = "outside tracked manifold";
                }
                title_dirty_ = true;
            }
            if (pending_gpu_steps_ > 0) {
                if (stepping_ == Stepping::RealTime) {
                    // Mask + bridge ride the step submission (batch tail).
                    run_real_time_batch();
                    volume_written_ = true;
                } else {
                    run_relax_batch();
                    write_display_texture();
                }
                pending_gpu_steps_ = 0;
                volume_dirty_ = false;
                if (gpu_title_due_) {
                    gpu_title_due_ = false;
                    title_dirty_ = true;
                }
            }
        }
    }

    // ---- the timer tick: accumulate work / CPU fallback stepping ----
    void tick() override {
        if (use_gpu_path()) {
            // Steps execute in run_frame (once per paint); cap the backlog.
            // The laser demo steps hotter so a flop fits in wall seconds.
            // time_scale_ multiplies the SUPPLY and the cap together (same
            // dt, more steps per rendered frame): visualized time runs Nx
            // while the GPU keeps up, then saturates into a lower fps -- a
            // bounded backlog, never a UI freeze.
            const int per_tick =
                ((stepping_ == Stepping::RealTime && laser_pol_ != LaserPol::Off)
                     ? kLaserStepsPerTick
                     : kStepsPerTick) *
                time_scale_;
            pending_gpu_steps_ =
                std::min(pending_gpu_steps_ + per_tick,
                         std::max(kMaxPendingGpuSteps, 2 * time_scale_));
            if (++ticks_ % 10 == 0) {
                gpu_title_due_ = true;
            }
            return;
        }
        ensure_cpu_current();
        // CPU fallback: deliberately NOT time-scaled -- steps run
        // synchronously inside the tick, so scaling would stall the UI.
        if (stepping_ == Stepping::RealTime) {
            sim_.advance(kStepsPerTick);
        } else {
            sim_.relax(kRelaxStepsPerTick, kRelaxDtau);
        }
        stage_active_view();
        if (++ticks_ % 10 == 0) {
            norm_display_ = ses::norm_sq(sim_.psi());
            title_dirty_ = true;
        }
    }

    // Visualized time scale (see scenario.hpp): steps per tick, not dt.
    void set_time_scale(int scale) override {
        time_scale_ = std::clamp(scale, 1, 16);
    }
    int time_scale() const override { return time_scale_; }

    // Simulated clock for the shell's au/s readout (same expression the
    // title and the jump log use).
    double sim_time() const override { return sim_.time() + gpu_time_; }
    double sim_dt() const override { return sim_.dt(); }

    // MCWF no-jump damping toggle (panel checkbox; see apply_mcwf_damping).
    void set_mcwf_damping(bool on) { mcwf_damping_ = on; }
    bool mcwf_damping() const { return mcwf_damping_; }

    // ---- controls (the shell's key/toolbar entry points) ----

    void set_real_time() override {
        if (stepping_ != Stepping::RealTime) {
            reset_ionized_tally();  // manual relax exit = fresh preparation
        }
        stepping_ = Stepping::RealTime;
        drop_relax_tables();
    }

    void set_relaxing() {
        if (use_gpu_path() && !ensure_relax_tables()) {
            return;  // no tables, no imaginary time
        }
        stepping_ = Stepping::Relaxing;
        relax_plateau_ = 0;
        relax_prev_energy_ = 0.0;
        if (!use_gpu_path()) {
            ensure_cpu_current();  // CPU relax (Surface view / no GPU)
            return;
        }
        // ITP converges to the lowest component PRESENT in the seed: a state
        // orthogonal to 1s (the n = 5 shell superposition) would stall
        // in-shell until fp32 noise leaks, and the plateau auto-complete
        // would fire first. Mix in 1% of an s-symmetric Gaussian so the
        // descent starts immediately. Only needed when the GPU state is
        // authoritative -- a CPU-truth state gets re-uploaded next frame.
        if (!cpu_is_truth_) {
            const ses::Field3D g = ses::gaussian_wavepacket(
                sim_.grid(), ses::Vec3d{}, ses::Vec3d{2.0, 2.0, 2.0},
                ses::Vec3d{});
            const int buf = engine_.create_state_buffer(g.data());
            if (buf >= 0) {
                engine_.add_state_into_psi(buf, 0.1, 0.0);
                engine_.release_state(buf);
                const ses_vk::Engine::NormPeak np = engine_.norm_and_peak();
                if (np.sum > 0.0) {
                    engine_.scale(static_cast<float>(1.0 / std::sqrt(np.sum)));
                }
            }
        }
    }

    void reset_simulation() override {
        if (solving()) {
            return;  // the startup atlas build owns the GPU state
        }
        sim_ = make_simulation();
        stepping_ = Stepping::RealTime;
        free_deflation_buffers();  // drop any owned deflation phi
        drop_relax_tables();
        laser_pol_ = LaserPol::Off;  // reset returns to the vanilla packet demo
        bfield_b_ = 0.0;             // and to no magnetic field
        upload_field_tables();    // restore the base half-potential
        reset_ionized_tally();
        cpu_is_truth_ = true;  // GPU state discarded with the reset
        gpu_time_ = 0.0;
        pending_gpu_steps_ = 0;
        if (gpu_ok_ && manifold_ready() && mode_ == ViewMode::Cloud) {
            seed_bound_superposition();  // a fresh random draw each reset
        }
        stage_active_view();
    }

    // Soft position measurement: sample from |psi|^2 (RNG lives here; core
    // takes the uniform draw) and let the sharpened packet re-evolve.
    void measure_now() override {
        if (solving()) {
            return;
        }
        ensure_cpu_current();
        std::uniform_real_distribution<double> uniform(0.0, 1.0);
        sim_.measure(uniform(rng_), kMeasureSigma);
        reset_ionized_tally();  // the electron was FOUND: nothing escaped
        stepping_ = Stepping::RealTime;
        stage_active_view();
    }

    // Projective ENERGY measurement (Key E): deferred to run_frame (the GPU
    // reductions + collapse run in the frame's compute half). Needs the
    // manifold.
    void measure_energy_now() {
        if (solving() || !use_gpu_path() || !manifold_ready()) {
            return;
        }
        pending_energy_measure_ = true;
        stepping_ = Stepping::RealTime;  // observe, then let H evolve it
        laser_pol_ = LaserPol::Off;
    }

    void toggle_view_mode() override {
        if (solving()) {
            return;
        }
        mode_ = (mode_ == ViewMode::Cloud) ? ViewMode::Surface : ViewMode::Cloud;
        // Re-stage for the newly selected mode: its data may be stale (tick
        // only stages the active mode, and we may be paused).
        if (mode_ == ViewMode::Surface) {
            ensure_cpu_current();  // meshing reads the CPU field
            if (stepping_ == Stepping::RelaxingExcited) {
                stepping_ = Stepping::Relaxing;  // deflation is GPU-only
            }
            laser_pol_ = LaserPol::Off;  // the drive is GPU-only too
            decay_on_ = false;  // so are the jump trials: OFF beats a lying title
        }
        stage_active_view();
    }

    // Relax into the z-aligned 2p: the z-odd seed keeps the flow in the
    // odd-parity sector, so it converges deterministically.
    void relax_to_excited() {
        start_excited_relax(make_axis_odd_seed(2), "2p", false);
    }

    // Relax into 2s. The 2p triplet is deflated too (2s sits ABOVE it) --
    // see start_excited_relax.
    void relax_to_2s() {
        start_excited_relax(
            ses::gaussian_wavepacket(sim_.grid(), ses::Vec3d{},
                                     ses::Vec3d{4.0, 4.0, 4.0}, ses::Vec3d{}),
            "2s", true);
    }

    // Toggle spontaneous decay (quantum jumps) over the tracked manifold.
    // All channels share one display acceleration factor (title reports it;
    // relative lifetimes stay physical).
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
    }

    // Instantly excite an n = 3/4 state (cycles) for the decay-cascade demo.
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
        reset_ionized_tally();
        cpu_is_truth_ = false;  // the GPU state is ahead now
        stepping_ = Stepping::RealTime;
    }

    // Cycle the laser off -> Z -> X -> off. Carrier and E0 both come from
    // our own spectrum / dipole element; X pumps 2p_x, so the monitored
    // P(2pz) stays flat (selection rule).
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
            // Drive the GRID resonance: the coarse-grid 1s carries a cusp gap
            // and a relaxed 1s sits ~0.03 Ha below a synthesized one, so use
            // the cooled <H> when available, else the h-audit grid energy.
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
    }

    // Static uniform E-field magnitude along +z (au); 0 = off. GPU
    // cloud/real-time path; the laser, if on, takes precedence.
    void set_efield_e0(double e0) {
        efield_e0_ = e0;
        upload_field_tables();  // fold E*z into the half-potential (with diamag if B on)
        if (e0 > 0.0 && !solving()) {
            stepping_ = Stepping::RealTime;  // let the field actually act
        }
    }

    // Magnetic field strength (au) along the current axis; 0 = off. Minimal
    // coupling: the diamagnetic term is folded into the half-potential here;
    // the per-frame magnetic_step adds only the paramagnetic rotation.
    void set_bfield_b(double b) {
        bfield_b_ = b;
        upload_field_tables();
        if (b > 0.0 && !solving()) {
            stepping_ = Stepping::RealTime;
        }
    }

    // Cycle the field axis z -> x -> y; the diamagnetic term is
    // axis-dependent, so the half-potential table is rebuilt.
    void toggle_bfield_axis() {
        bfield_axis_ = (bfield_axis_ == 2) ? 0 : (bfield_axis_ == 0 ? 1 : 2);
        upload_field_tables();
    }
    int bfield_axis() const { return bfield_axis_; }

    // ---- selftest / verification hooks ----

    // The computed Einstein A for a channel (0 if absent).
    double channel_a(int from, int to) const {
        for (const ShellChannel& c : atom_.channels()) {
            if (c.from == from && c.to == to) {
                return c.a_true;
            }
        }
        return 0.0;
    }
    // Scenario keys beyond the generic set (1/R/M/Tab live in the shell).
    bool handle_key(char key) override {
        switch (key) {
            case '2': set_relaxing(); return true;
            case '3': relax_to_excited(); return true;
            case '4': relax_to_2s(); return true;
            case '5': excite_n3(); return true;
            case 'D': toggle_decay(); return true;
            case 'E': measure_energy_now(); return true;
            case 'L': toggle_laser(); return true;
            default: return false;
        }
    }
    bool scene_ready() const override { return manifold_ready(); }

    bool solving() const override { return gpu_ok_ && !atlas_done_; }
    // Ready only once the FULL table is assembled (channels_ fills
    // incrementally during the pair phase -- do not race it).
    bool manifold_ready() const { return atlas_done_ && !atom_.channels().empty(); }
    double state_energy(int idx) const { return atom_.state_energy(idx); }
    long long photon_count() const { return photon_count_; }
    // Cumulative absorbed (ionized) fraction since the last collapse/prep.
    double ionized_fraction() const {
        return std::max(0.0, 1.0 - bound_survival_);
    }
    // Result of the most recent energy measurement: eigenstate index, -1 for
    // the outside-the-manifold outcome, -2 if none has run yet.
    int last_measured_index() const { return last_measured_index_; }
    // <z> of the current cloud (bridges the GPU state to the CPU session
    // first); the hook for the Stark polarization along +z.
    double mean_z() {
        ensure_cpu_current();
        return ses::mean_position(sim_.psi()).z;
    }
    double peak_excited_population() const { return rabi_peak_; }

    // Magnetic Larmor hooks: prepare an eigenstate, probe another state's
    // population -- proves the field evolves psi itself, not just the display.
    void debug_prepare_state(int idx) {
        if (!manifold_ready() || idx < 0 || idx >= kNumStates) {
            return;
        }
        atom_.collapse_onto(engine_, idx);
        reset_ionized_tally();
        cpu_is_truth_ = false;
        stepping_ = Stepping::RealTime;
    }
    double probe_population(int idx) {
        if (!manifold_ready() || idx < 0 || idx >= kNumStates) {
            return 0.0;
        }
        engine_.project_psi();
        return atom_.project_population(engine_, idx);
    }

    // ---- display-facing accessors (the shell's FrameInput assembly) ----

    bool cloud() const override { return mode_ == ViewMode::Cloud; }
    double peak() const override { return peak_; }
    bool compute_attempted() const override { return compute_attempted_; }
    bool gpu_ok() const override { return gpu_ok_; }
    // The engine's bridge image on the GPU path; null lets the renderer fall
    // back to its CPU-staged texture.
    VkImageView psi_volume_view() override { return gpu_ok_ ? engine_.volume_view() : VK_NULL_HANDLE; }
    // Photon flash: a brief warm background right after a quantum jump.
    float next_flash_intensity() override {
        if (flash_ticks_ <= 0) {
            return 0.0f;
        }
        const float v = static_cast<float>(flash_ticks_) / 25.0f;
        --flash_ticks_;
        return v;
    }
    bool take_volume_written() override {
        const bool w = volume_written_;
        volume_written_ = false;
        return w;
    }
    bool take_volume_dirty() override {
        const bool d = volume_dirty_;
        volume_dirty_ = false;
        return d;
    }
    bool take_mesh_dirty() override {
        const bool d = mesh_dirty_;
        mesh_dirty_ = false;
        return d;
    }
    void mark_display_dirty() override {  // shell resize / first frame
        mesh_dirty_ = true;
        volume_dirty_ = true;
    }
    bool take_title_dirty() override {
        const bool t = title_dirty_;
        title_dirty_ = false;
        return t;
    }
    const std::vector<float>& psi_staging() const override { return psi_staging_; }
    const ses::Mesh& mesh() const override { return mesh_; }
    const std::vector<ses::Rgb>& colors() const override { return colors_; }

    // The full window-title readout, composed Qt-free.
    std::string title_text() override {
        // While relaxing: exact <H> on the CPU session, or the free ITP
        // estimator on the GPU path.
        const double t_au = sim_.time() + gpu_time_;
        std::string s = "Electron near a hydrogen nucleus   t = " +
                        strf("%.2f au (%.3f fs)", t_au, t_au * kAuToFs) + "   ";
        if (stepping_ != Stepping::RealTime) {
            s += cpu_is_truth_
                     ? strf("E = %.3f eV   ",
                            ses::mean_energy(sim_.psi(), sim_.potential()) * kHaToEv)
                     : strf("E ~ %.3f eV   ", relax_energy_display_ * kHaToEv);
        }
        s += strf("norm = %.6f   [%s, %s, %s]  1=real 2=relax R=reset tab=view "
                  "[ ]=density M=pos E=energy",
                  norm_display_,
                  mode_ == ViewMode::Cloud ? "cloud" : "surface",
                  stepping_ == Stepping::RealTime
                      ? "real-time"
                      : (stepping_ == Stepping::Relaxing
                             ? "relaxing->1s"
                             : strf("relaxing->%s", relax_label_.c_str()).c_str()),
                  use_gpu_path() ? "gpu 256^3" : "cpu 256^3");
        if (stepping_ == Stepping::RealTime && !solving()) {
            s += strf("  emit P = %.2e au", radiated_power_);
        }
        if (solving()) {
            s += synth_queue_.empty()
                     ? strf("  solving atom: dipole channels (%d left)",
                            static_cast<int>(atom_.pair_queue().size()))
                     : strf("  solving atom: %s (%d/%d)",
                            kStateSpec[synth_queue_.front()].name,
                            kNumStates - static_cast<int>(synth_queue_.size()) + 1,
                            kNumStates);
        }
        if (decay_on_ && !atom_.channels().empty()) {
            s += strf("  decay ON: tau(2p) %.2e au, tau(2s) %.2e au, x%.1e, photons %lld",
                      atom_.lifetime_of(kP2Z), atom_.lifetime_of(kS2),
                      atom_.accel_display(), photon_count_);
            if (!last_jump_.empty()) {
                s += strf(", last %s", last_jump_.c_str());
            }
        }
        if (laser_pol_ != LaserPol::Off) {
            s += strf("  laser %s: w %.4f, E0 %.4f, P(1s) %.3f, P(2pz) %.3f",
                      laser_pol_ == LaserPol::Z ? "Z" : "X", laser_omega_,
                      laser_e0_, pop_ground_, pop_excited_);
        }
        if (efield_e0_ > 0.0 && laser_pol_ == LaserPol::Off) {
            s += strf("  E-field +z: %.4f au (%.2e V/m)", efield_e0_,
                      efield_e0_ * 5.14220674e11);
        }
        if (bfield_b_ > 0.0) {
            s += strf("  B-field %s: %.4f au, omega_L %.4f au (psi evolved)",
                      bfield_axis_ == 2 ? "z" : (bfield_axis_ == 0 ? "x" : "y"),
                      bfield_b_, 0.5 * bfield_b_);
        }
        if (absorber_on_ && 1.0 - bound_survival_ > 5e-4) {
            s += strf("  ionized %.1f%%", (1.0 - bound_survival_) * 100.0);
        }
        if (!last_measure_.empty()) {
            s += strf("  measured %s", last_measure_.c_str());
        }
        return s;
    }

private:
    static std::string strf(const char* fmt, ...) {
        char buf[192];
        va_list args;
        va_start(args, fmt);
        std::vsnprintf(buf, sizeof buf, fmt, args);
        va_end(args);
        return std::string{buf};
    }

    ses::Vec3d laser_axis() const {
        return laser_pol_ == LaserPol::X ? ses::Vec3d{1.0, 0.0, 0.0}
                                         : ses::Vec3d{0.0, 0.0, 1.0};
    }

    // A collapse or a fresh preparation resolves the absorbed-flux record:
    // the electron was found bound, so the tally restarts from norm 1.
    void reset_ionized_tally() {
        bound_survival_ = 1.0;
        norm_baseline_ = 1.0;
    }

    // MCWF no-jump damping: psi += (e^{-gamma_n dt/2} - 1) c_n |n> for each
    // occupied excited state, then renormalize -- the non-Hermitian H_eff
    // conditioned on "no photon detected". A pure eigenstate is unchanged
    // (the renorm cancels its damping); a superposition visibly breathes its
    // excited fraction out between jumps. Cost control: only states with
    // pop >= 1e-3, the strongest kMcwfMaxStates per tick (one transient
    // synthesis + axpy each) -- exact once the cloud concentrates onto a
    // few states, approximate-but-convergent for the 55-state seed.
    void apply_mcwf_damping(const std::array<double, kNumStates>& pop,
                            double dt) {
        constexpr int kMcwfMaxStates = 8;
        constexpr double kMcwfMinPop = 1e-3;
        std::array<double, kNumStates> gamma_out{};
        for (const ShellChannel& ch : atom_.channels()) {
            gamma_out[static_cast<std::size_t>(ch.from)] += ch.gamma_display;
        }
        std::array<int, kNumStates> order{};
        int n_cand = 0;
        for (int s = 1; s < kNumStates; ++s) {
            if (gamma_out[static_cast<std::size_t>(s)] > 0.0 &&
                pop[static_cast<std::size_t>(s)] >= kMcwfMinPop) {
                order[static_cast<std::size_t>(n_cand++)] = s;
            }
        }
        if (n_cand == 0) {
            return;
        }
        std::partial_sort(order.begin(),
                          order.begin() + std::min(n_cand, kMcwfMaxStates),
                          order.begin() + n_cand, [&pop](int a, int b) {
                              return pop[static_cast<std::size_t>(a)] >
                                     pop[static_cast<std::size_t>(b)];
                          });
        bool damped = false;
        double damp_loss = 0.0;  // norm removed by H_eff (NOT the absorber)
        for (int i = 0; i < std::min(n_cand, kMcwfMaxStates); ++i) {
            const int s = order[static_cast<std::size_t>(i)];
            const double f =
                std::exp(-0.5 * gamma_out[static_cast<std::size_t>(s)] * dt);
            const ses::Complex<double> c =
                atom_.project_state_amplitude(engine_, s);
            const ses::Complex<double> d = (f - 1.0) * c;
            const int buf = atom_.synth_transient(engine_, s);
            if (buf >= 0) {
                engine_.add_state_into_psi(buf, d.real(), d.imag());
                engine_.release_state(buf);
                damp_loss += pop[static_cast<std::size_t>(s)] * (1.0 - f * f);
                damped = true;
            }
        }
        if (damped) {
            const ses_vk::Engine::NormPeak np = engine_.norm_and_peak();
            // Ionization tally: back out the analytically known H_eff loss
            // so this renorm only launders the damping, not absorbed flux.
            if (absorber_on_ && norm_baseline_ > 0.0) {
                bound_survival_ *= (np.sum + damp_loss) / norm_baseline_;
            }
            if (np.sum > 0.0) {
                engine_.scale(static_cast<float>(1.0 / std::sqrt(np.sum)));
            }
            norm_baseline_ = 1.0;
        }
    }

    // One real-time step batch: propagate (driven / magnetic / plain),
    // absorb at the walls, then the title-cadence readouts and trials.
    void run_real_time_batch() {
        if (gpu_title_due_) {
            // GPU-reduced norm+peak (2 KB readback), taken BEFORE enqueueing
            // new steps so the implicit sync waits on long-finished work.
            const ses_vk::Engine::NormPeak np = engine_.norm_and_peak();
            norm_display_ = np.sum;
            if (np.peak > 0.0) {
                peak_ = np.peak;  // brightness tracks the cloud
            }
            // Absorbed (ionized) flux tally BEFORE the renorm below, which
            // would otherwise silently pump the lost norm back into the
            // cloud. Signed ratio: fp32 drift cancels as a zero-mean walk.
            if (absorber_on_ && np.sum > 0.0 && norm_baseline_ > 0.0) {
                bound_survival_ *= np.sum / norm_baseline_;
            }
            // fp32 drift renormalization: the split-operator is unitary in
            // exact arithmetic; pinning norm = 1 removes numerical decay.
            if (np.sum > 0.0 && std::abs(np.sum - 1.0) > 1e-4) {
                engine_.scale(static_cast<float>(1.0 / std::sqrt(np.sum)));
                norm_baseline_ = 1.0;
            } else if (np.sum > 0.0) {
                norm_baseline_ = np.sum;
            }
            // Semiclassical radiated power via the GPU mean-force reduction
            // (4 KB readback); ~0 for a stationary eigenstate.
            radiated_power_ = ses::larmor_power(engine_.mean_force());
        }
        if (laser_pol_ != LaserPol::Off) {
            // t0 is the same clock that credits gpu_time_, so the carrier
            // phase cos(w t) stays continuous across batches/pauses.
            const ses::DipoleDrive d{laser_axis(), laser_e0_, laser_omega_};
            engine_.driven_step(d, sim_.time() + gpu_time_, sim_.dt(),
                                pending_gpu_steps_, absorber_on_, true);
        } else if (bfield_b_ > 0.0) {
            // Minimal-coupling magnetic step: static E + diamagnetic are
            // already folded into the half-potential (upload_field_tables);
            // the paramagnetic L_axis is the exact three-shear rotation.
            engine_.magnetic_step(bfield_axis_,
                                  0.5 * bfield_b_ * (0.5 * sim_.dt()),
                                  pending_gpu_steps_, absorber_on_, true);
        } else {
            // The static E-field (if any) is folded into the half-potential,
            // so a plain step polarizes / field-ionizes correctly. The
            // absorbing mask damps after every step; the display bridge
            // records into the same submission.
            engine_.step(pending_gpu_steps_, absorber_on_, true);
        }
        // Time is credited where steps EXECUTE, so a stalled or
        // occluded paint cannot desync the clock from the state.
        gpu_time_ += pending_gpu_steps_ * sim_.dt();

        // Orbital-free populations: ONE deposit pass on the post-step psi,
        // shared by the decay and laser readouts this title tick.
        if (gpu_title_due_ && proj_ready_ &&
            ((decay_on_ && !atom_.channels().empty()) ||
             laser_pol_ != LaserPol::Off)) {
            engine_.project_psi();
        }

        // Competing-channel Poisson trials on the TITLE cadence: the
        // exponential is memoryless, so accumulated-dt trials give identical
        // statistics with far fewer GPU reductions. A jump collapses psi
        // onto the fired channel's destination.
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
                const double trial_dt = decay_accum_dt_;
                const ses::ChannelPick pick = ses::pick_decay_channel(
                    rates, trial_dt, uniform(rng_), uniform(rng_));
                decay_accum_dt_ = 0.0;
                if (pick.channel >= 0) {
                    const ShellChannel& ch =
                        atom_.channels()[static_cast<std::size_t>(pick.channel)];
                    atom_.collapse_onto(engine_, ch.to);
                    reset_ionized_tally();
                    flash_ticks_ = 25;
                    ++photon_count_;
                    last_jump_ = strf("%s->%s", kStateSpec[ch.from].name,
                                      kStateSpec[ch.to].name);
                    std::fprintf(stderr,
                                 "decay: jump %s (photon #%lld, t=%.1f au)\n",
                                 last_jump_.c_str(), photon_count_,
                                 sim_.time() + gpu_time_);
                    title_dirty_ = true;
                } else if (mcwf_damping_ && laser_pol_ == LaserPol::Off) {
                    // No photon detected this interval: MCWF no-jump
                    // evolution (H_eff) damps each excited amplitude by
                    // e^{-gamma_n dt/2}. Skipped after a jump (the projected
                    // amplitudes are stale post-collapse) and while the
                    // laser drives (display-accelerated gammas would swamp
                    // the coherent flop).
                    apply_mcwf_damping(pop, trial_dt);
                }
            }
        }

        // Title-cadence populations (and the Rabi peak the selftest asserts
        // on): two 2 KB reductions every ~10 ticks.
        if (laser_pol_ != LaserPol::Off && gpu_title_due_ &&
            manifold_ready()) {
            pop_excited_ = atom_.project_population(engine_, kP2Z);
            pop_ground_ = atom_.project_population(engine_, kS1);
            rabi_peak_ = std::max(rabi_peak_, pop_excited_);
        }
    }

    // One imaginary-time batch: renormalized every step; the ITP estimator
    // gives the convergence readout free. The excited flavor deflates the
    // states below the target.
    void run_relax_batch() {
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

        // Auto-complete: when the ITP energy plateaus, return to real time
        // so the lifetimes act.
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
                std::fprintf(stderr,
                             "relax: auto-complete -> real time (E=%.6f Ha, "
                             "t=%.1f au)\n",
                             stats.energy, sim_.time() + gpu_time_);
                reset_ionized_tally();  // fresh preparation
                free_deflation_buffers();  // converged -> free the phi
                drop_relax_tables();
            }
        }
    }

    // Launcher for the excited relaxation demos (keys 3/4). FOOTGUN: deflate
    // EVERY state below the target -- with only 1s removed, fp32 parity
    // leakage grows exponentially until the on-screen "2s" morphs into 2p.
    void start_excited_relax(const ses::Field3D& seed, const char* label,
                             bool deflate_p_triplet) {
        if (!gpu_ok_ || solving()) {
            return;  // deflation runs on the GPU path only
        }
        if (mode_ != ViewMode::Cloud) {
            mode_ = ViewMode::Cloud;
        }
        if (!manifold_ready() || !ensure_relax_tables()) {
            return;
        }
        // Deflation set: OWNED transient fp32 buffers (no resident atlas),
        // freed at auto-complete / reset / the next relaxation.
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

    // Rebuild the GPU half-potential: V + E z (Stark) + (B^2/8) rho_perp^2
    // (diamagnetic) -- both diagonal in position, so folding them in is
    // exact; the per-frame magnetic_step adds only the paramagnetic rotation.
    void upload_field_tables() {
        if (!gpu_ok_) {
            return;
        }
        // Memo: sliders re-fire on every drag frame and reset always calls
        // with 0/0 -- skip the two 67 MB uploads when nothing changed.
        if (efield_e0_ == uploaded_e0_ && bfield_b_ == uploaded_b_ &&
            bfield_axis_ == uploaded_axis_) {
            return;
        }
        uploaded_e0_ = efield_e0_;
        uploaded_b_ = bfield_b_;
        uploaded_axis_ = bfield_axis_;
        std::vector<double> v = sim_.potential();
        if (efield_e0_ > 0.0) {
            const ses::Grid3D& g = sim_.grid();
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
            const ses::MagneticPropagator3D mprop{sim_.grid(), v, sim_.dt(),
                                                  bfield_b_, bfield_axis_};
            v = mprop.effective_potential();
        }
        engine_.set_potential(v);
        // Ehrenfest mean force must differentiate the SAME V psi evolves
        // under: a stationary state of the FULL Hamiltonian reads
        // <grad V> = 0 (no fake Larmor power with fields on).
        engine_.set_potential_gradient(v);
    }

    // Free the OWNED transient deflation buffers (synthesized at relax-start).
    void free_deflation_buffers() {
        for (int b : relax_deflate_owned_) {
            engine_.release_state(b);
        }
        relax_deflate_owned_.clear();
        relax_deflate_.clear();
    }

    // Advance the startup atlas build one chunk per paint: synthesize + show
    // one orbital (montage), then evaluate dipole pairs, then assemble the
    // channel table and resume the wavepacket.
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
            // Synthesize into a TRANSIENT fp32 buffer: capture the grid norm,
            // SHOW it (montage), audit, then FREE -- one orbital resident at
            // a time.
            double pk = 0.0;
            const int buf = atom_.synth_transient(engine_, idx, &pk);
            if (buf < 0) {
                atlas_done_ = true;  // GPU buffer alloc failed: give up gracefully
                return;
            }
            engine_.copy_into_psi(buf);  // show (fp32)
            // h-audit: cross-check 1D radial energy vs 3D spectral <H> for
            // the resolution-critical 1s and box-critical 4s/5s/6s -- the
            // only states read back to the CPU.
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
                // Free the 'from' cache; nothing else is resident (no atlas
                // -> ~1.2 GB runtime, 512^3 feasible).
                atom_.release_pair_cache(engine_);
                atlas_done_ = true;
                seed_bound_superposition();  // the demo starts bound
                title_dirty_ = true;
            }
        }
    }

    // Seed psi with a random complex superposition of the WHOLE n <= 5
    // manifold ([0, k6S), 55 bound states): contained by construction --
    // zero continuum component, so nothing can leave the box (a free packet
    // could). Cross-shell beats make the density genuinely evolve (fastest:
    // 1s-2p at T = 2 pi / 0.375 ~ 17 au); renormalized to 1 on the GPU.
    void seed_bound_superposition() {
        if (!gpu_ok_ || !atom_.radial_ready()) {
            cpu_is_truth_ = true;  // fallback: resume the CPU packet
            return;
        }
        // Complex-Gaussian coefficients = uniform on the state sphere after
        // the final renormalization; c[0]'s phase is folded out (global).
        std::normal_distribution<double> gauss(0.0, 1.0);
        std::array<ses::Complex<double>, k6S> c{};
        for (auto& z : c) {
            z = ses::Complex<double>{gauss(rng_), gauss(rng_)};
        }
        const double mag0 = std::sqrt(ses::norm_sq(c[0]));
        if (mag0 > 0.0) {
            const ses::Complex<double> rot{c[0].real() / mag0,
                                           -c[0].imag() / mag0};
            for (auto& z : c) {
                z = z * rot;  // c[0] is now real >= 0
            }
        }
        int buf = atom_.synth_transient(engine_, 0);
        if (buf < 0) {
            cpu_is_truth_ = true;
            return;
        }
        engine_.copy_into_psi(buf);
        engine_.release_state(buf);
        engine_.scale(static_cast<float>(c[0].real()));
        for (int s = 1; s < k6S; ++s) {
            buf = atom_.synth_transient(engine_, s);
            if (buf < 0) {
                continue;  // member missing: superpose the rest
            }
            engine_.add_state_into_psi(buf, c[s].real(), c[s].imag());
            engine_.release_state(buf);
        }
        const ses_vk::Engine::NormPeak np = engine_.norm_and_peak();
        if (np.sum > 0.0) {
            engine_.scale(static_cast<float>(1.0 / std::sqrt(np.sum)));
            peak_ = np.peak / np.sum;
        }
        norm_display_ = 1.0;
        reset_ionized_tally();  // fresh preparation
        cpu_is_truth_ = false;  // the GPU state is the seed
        write_display_texture();
        volume_dirty_ = false;
    }

    // Transient relax tables: uploaded on demand (a table build + 268 MB
    // upload per relax entry), freed on completion -- they were the largest
    // always-resident allocation after the phase tables.
    bool ensure_relax_tables() {
        if (engine_.relax_tables_ready()) {
            return true;
        }
        const ses::ImaginaryTimePropagator3D relaxer{sim_.grid(), sim_.potential(),
                                                     kRelaxDtau};
        if (!engine_.set_relax_tables(relaxer.half_potential_weight(),
                                      relaxer.kinetic_weight(), kRelaxDtau,
                                      sim_.grid().cell_volume())) {
            std::fprintf(stderr, "engine: relax table upload failed\n");
            return false;
        }
        return true;
    }
    void drop_relax_tables() { engine_.release_relax_tables(); }

    // Pull the GPU-evolved state back into the CPU double session (fp32
    // precision at the handoff).
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
                return;  // run_frame uploads the state and bridges the texture
            }
            stage_volume();
            volume_dirty_ = true;
        } else {
            remesh();
            mesh_dirty_ = true;
        }
    }

    // ---- CPU staging ----

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

    // Bridge psi to the display texture (always the real wavefunction).
    void write_display_texture() {
        engine_.write_psi_to_volume();
        volume_written_ = true;  // resets the temporal accumulation
    }

    ses::WavepacketSimulation sim_;
    ses_vk::Engine engine_;
    ViewMode mode_ = ViewMode::Cloud;
    Stepping stepping_ = Stepping::RealTime;

    // cpu_is_truth_ is the single sync invariant: true -> sim_.psi() is
    // current, false -> the engine's psi buffer is ahead and must be read
    // back before any CPU-side operation.
    bool compute_attempted_ = false;
    bool gpu_ok_ = false;
    bool cpu_is_truth_ = true;
    int pending_gpu_steps_ = 0;
    int time_scale_ = 1;  // steps-per-tick multiplier (dt untouched)
    bool mcwf_damping_ = true;  // no-jump H_eff damping between jumps
    bool pending_energy_measure_ = false;  // Key E: serviced in run_frame
    bool gpu_title_due_ = false;
    bool title_dirty_ = false;
    double gpu_time_ = 0.0;
    double norm_display_ = 1.0;
    double relax_energy_display_ = 0.0;
    double radiated_power_ = 0.0;  // semiclassical Larmor power (au)
    std::vector<float> readback_buf_;

    // Tracked atom: radial solve, synthesis bookkeeping, E1 channel table;
    // engine-backed calls pass engine_ explicitly.
    ses_shell::AtomModel atom_;
    bool proj_ready_ = false;  // static projection index uploaded

    // Absorbed-flux (ionization) bookkeeping: 1 - bound_survival_ is the
    // cumulative escaped fraction; any collapse/preparation resolves it.
    double bound_survival_ = 1.0;
    double norm_baseline_ = 1.0;  // norm at the last tally/renorm point

    // Last field values whose tables reached the GPU (upload memo). The
    // engine starts with the bare potential + gradient (init), i.e. 0/0.
    double uploaded_e0_ = 0.0;
    double uploaded_b_ = 0.0;
    int uploaded_axis_ = 2;

    // Quantum-jump bookkeeping.
    std::string last_jump_;
    std::string last_measure_;  // last energy-measurement readout (Key E)
    int last_measured_index_ = -2;  // last energy-measurement outcome
    std::string relax_label_ = "2p";
    std::vector<int> relax_deflate_;        // live RelaxingExcited deflation set
    std::vector<int> relax_deflate_owned_;  // owned transient states to release
    double relax_prev_energy_ = 0.0;     // relax auto-complete plateau tracking
    int relax_plateau_ = 0;

    // Startup atlas build (radial solve + synthesis, chunked in paint).
    std::vector<int> synth_queue_;
    int montage_hold_ = 0;
    bool atlas_done_ = false;
    double decay_accum_dt_ = 0.0;  // sim time since the last decay trial
    int excite_cycle_ = 0;         // key-5 n=3 cycle position
    // Decay defaults ON (as in nature); D is the off-switch. Armed once the
    // startup atlas build finishes.
    bool decay_on_ = true;
    int flash_ticks_ = 0;
    long long photon_count_ = 0;

    // Laser (resonant dipole drive) bookkeeping.
    LaserPol laser_pol_ = LaserPol::Off;
    double laser_omega_ = 0.0;
    double laser_e0_ = 0.0;
    double efield_e0_ = 0.0;  // static +z electric field magnitude (au); 0 = off
    double bfield_b_ = 0.0;      // magnetic field strength (au); 0 = off
    int bfield_axis_ = 2;        // field direction: 2=z, 0=x, 1=y
    double grid_energy_1s_ = 0.0;  // <H>_grid of 1s from the h-audit; the
                                   // laser drives THIS resonance (cusp gap)
    double pop_ground_ = 0.0;
    double pop_excited_ = 0.0;
    double rabi_peak_ = 0.0;  // max P(2pz) since the laser came on

    // CPU staging for the renderer's fallback texture and the Surface mesh.
    ses::Mesh mesh_;
    std::vector<ses::Rgb> colors_;
    std::vector<float> psi_staging_;
    double peak_ = 0.0;
    bool mesh_dirty_ = false;
    bool volume_dirty_ = false;
    bool volume_written_ = false;  // bridge wrote psi this frame
    long long ticks_ = 0;

    bool absorber_on_ = false;  // real R32 absorber uploaded (set_absorber)
    std::mt19937 rng_{std::random_device{}()};
};

}  // namespace ses_shell
