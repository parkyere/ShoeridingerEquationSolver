#pragma once

// The Qt-free simulation director: everything the demo IS, minus the window.
// Owns the CPU truth session, the ses_vk engine, the AtomModel, and the whole
// demo state machine -- stepping modes, the startup atlas build, decay Poisson
// trials, the laser drive, static E/B fields, position/energy measurement,
// excited-state relaxation with deflation, and the cpu_is_truth_ sync
// invariant. The Qt shell translates input events into the control methods
// below, polls take_title_dirty()/title_text() for the readout, and assembles
// the renderer's FrameInput from the display-facing accessors. No Qt type
// crosses this boundary.

#include "atom_model.hpp"
#include "manifold_spec.hpp"
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
// dropped ticks drop cleanly). Throughput note (measured on the P5000 with
// the hand-rolled FFT): the laser path saturates the GPU at ~26 steps/s
// (~38 ms/step at 256^3), so raising this cap does NOT raise the sim rate
// (verified 8 vs 32: 1.02 vs 1.04 au/s); it only lengthens the per-paint
// block. 8 keeps paints ~300 ms.
constexpr int kMaxPendingGpuSteps = 8;
constexpr int kRelaxStepsPerTick = 1;
constexpr double kRelaxDtau = 0.05;
constexpr double kIsoFraction = 0.25;
constexpr double kMeasureSigma = 1.25;  // position measurement resolution (Bohr):
                                        // the Heisenberg back-action deposits
                                        // <T> = 3/(8 sigma^2) = 0.24 Ha, safely
                                        // under the 0.5 Ha binding -- a typical
                                        // measurement localizes the electron
                                        // WITHOUT ionizing the atom (at the old
                                        // one-cell 0.625 it injected 0.96 Ha and
                                        // nearly every click blew the atom apart)
// Display decay rate: the TRUE Einstein-A lifetime is ~1e8 a.u. (unwatchable);
// this gives tau_display ~ 8 a.u. (~3 s of wall time). The title reports the
// true lifetime and the acceleration factor honestly.
constexpr double kDecayGammaDisplay = 0.125;
constexpr double kHaToEv = 27.211386;  // 1 Hartree in eV (physicist-facing display)
constexpr double kAbsorbWidth = 10.0;  // Bohr: boundary absorber layer thickness
                                       // (interior +-70 Bohr stays untouched --
                                       // clears the n<=6 states; real-time only)
// Laser: E0 is derived from a TARGET Rabi frequency over the computed
// dipole matrix element (Omega = E0 |<2p|z|1s>|). Omega = 0.04 keeps the drive
// well under the ~0.35 Ha 1s->2p gap (bare Coulomb on the grid; RWA-ish
// two-level flopping) while a full flop (2 pi / Omega ~ 157 au) takes seconds at
// the laser tick rate. The carrier is tuned to the GRID resonance -- the cooled
// 1s <H>, ~0.35 above 2p, not the textbook 0.375 label -- see toggle_laser.
constexpr double kRabiTargetOmega = 0.04;
constexpr int kLaserStepsPerTick = 6;  // the pump demo runs hotter than 1x

constexpr int kAtlasMontageFrames = 3;  // frames each synthesized orbital shows
constexpr int kAtlasPairsPerFrame = 4;  // dipole pairs evaluated per paint

// The atom's potential is the BARE Coulomb -Z/r (true hydrogen), REGULARIZED on
// the grid rather than softened. The 3D grid has a point on the nucleus where
// -1/r would be -infinity, so that ONE cell takes the analytic cell average
// (ses::regularized_coulomb_potential); every other cell is exact -1/r. The
// radial solves feed bare -1/r directly -- their grid r_i = (i+1)h never hits 0.
// Away from the nucleus cell the two are the SAME operator, so synthesized
// orbitals stay eigenstates of the propagated Hamiltonian; only the coarse
// nucleus cell differs, leaving a small s-state cusp gap in the startup h-audit
// (E_radial reads textbook; <H>_grid is ~1.5 eV shallower for 1s at h = 0.625,
// shrinking ~1/n^3 so 4s/5s/6s land within ~0.02 eV).
inline ses::WavepacketSimulation make_simulation() {
    // Stepping runs on the GPU engine; the CPU session stays the
    // double-precision truth for measurement, surface meshing, and the
    // no-GPU fallback, synced on demand (the cpu_is_truth_ invariant).
    //
    // Box +-80 Bohr at 256^3 (h = 0.625): holds the n <= 6 shell, box-critical
    // at n = 6 (the diffuse 6s is ~92% enclosed, its tail kissing the
    // u(R_box) = 0 wall; the structured 6d/6f/6g/6h are 97-99.9% held). Fixed
    // at 256^3 because the split-operator FFT demands a power-of-two size (512
    // would be 8x the work), so reaching n = 6 spends grid spacing: h grows
    // 0.5 -> 0.625. The startup atlas cross-check E_radial vs <H>_grid audits h
    // on every launch. The full m-resolved n <= 6 manifold is tracked WITHOUT
    // a resident atlas: each orbital is synthesized on the GPU into a
    // transient buffer (shown, audited, freed) and the dipole integrals
    // reduce on the GPU, so neither VRAM nor host RAM ever holds the whole
    // manifold at once.
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

class SimDirector {
public:
    SimDirector() : sim_(make_simulation()) {
        remesh();
        stage_volume();
    }

    const ses::Grid3D& grid() const { return sim_.grid(); }

    // ---- lifecycle ----

    // COMPUTE setup: the GPU propagation engine (fp32 transcription of the
    // tested CPU tables; verified by sesolver_vkcheck), atlas precision from
    // the shell-probed free VRAM, atom solve, projection index, absorber
    // mask. device_ok short-circuits engine init when the shell failed to
    // adopt a device; any failure falls back to CPU stepping.
    void init_compute(ses_vk::DeviceContext& ctx, bool device_ok,
                      std::int64_t free_vram_bytes) {
        compute_attempted_ = true;
        gpu_ok_ = device_ok &&
                  engine_.initialize(ctx, sim_.grid(),
                                     ses_shell::app_engine_blobs(sim_.grid().x.n),
                                     sim_.propagator().half_potential_phase(),
                                     sim_.propagator().kinetic_phase(),
                                     sim_.psi().data());
        if (gpu_ok_) {
            // Pick the atlas storage precision from free VRAM. A resident fp32
            // n<=6 manifold (kNumStates complex-fp32 buffers, ~12 GB at 256^3)
            // oversubscribes a small card, and WDDM then pages it into system
            // RAM -- host RAM balloons and the frame rate collapses. fp16
            // halves it (~6 GB). Headroom covers the live working set +
            // textures + the unpack scratch + driver; an unmeasurable budget
            // keeps fp32 (the big-VRAM default).
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
            // Imaginary-time weights from the tested CPU relaxer. These
            // engine uploads are FALLIBLE (buffer/pipeline create can fail
            // under VRAM pressure); a failed relax table would silently run
            // REAL-TIME phase tables in imaginary time, so failure demotes
            // to the CPU fallback path wholesale.
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
            // Solve the atom up front. The radial engine gets every
            // bound level to n = 10 (the full lifetime table, printed
            // below); the 3D tracked manifold (n <= 6, what the box holds)
            // is then synthesized chunked across frames so decay is armed
            // BY DEFAULT and every demo entry point is instant afterwards.
            atom_.solve_radial_atom(sim_.grid().x.xmax);
            // Orbital-free projection index: the static counting-sort
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

    // The widget's device is about to go away: tear down everything the
    // engine created on it. GPU simulation state dies with it.
    void release_gpu() {
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
    void run_frame() {
        // The startup atlas build advances regardless of the view
        // mode -- a Tab to Surface during the startup window must not wedge
        // solving() forever (the build owns the psi buffer either way; the
        // Surface view simply does not display it while it runs).
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
                // first frame) the block below would never refresh the
                // display volume and the screen would keep the stale (or
                // undefined) cloud.
                write_display_texture();
            }
            // Projective ENERGY measurement (Key E): sample an eigenstate n
            // from P_n = |<phi_n|psi>|^2 over the tracked manifold and collapse
            // psi onto it. The incomplete-manifold deficit 1 - sum(P_n) is the
            // continuum outcome (n = -1): leave psi and say so. Reuses the same
            // GPU inner-product / collapse primitives as the decay jump; works
            // even while paused (it bridges the display volume itself).
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
                    last_measure_ = strf(
                        "%s (E %.3f eV)",
                        kStateSpec[static_cast<std::size_t>(n)].name,
                        atom_.state_energy(n) * kHaToEv);
                } else {
                    last_measure_ = "outside tracked manifold";
                }
                title_dirty_ = true;
            }
            if (pending_gpu_steps_ > 0) {
                if (stepping_ == Stepping::RealTime) {
                    run_real_time_batch();
                } else {
                    run_relax_batch();
                }
                pending_gpu_steps_ = 0;
                write_display_texture();
                volume_dirty_ = false;
                if (gpu_title_due_) {
                    gpu_title_due_ = false;
                    title_dirty_ = true;
                }
            }
        }
    }

    // ---- the timer tick: accumulate work / CPU fallback stepping ----
    void tick() {
        if (use_gpu_path()) {
            // Steps execute in run_frame (once per paint). Cap the
            // backlog so a stalled paint cannot spiral; time is credited at
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
            title_dirty_ = true;
        }
    }

    // ---- controls (the shell's key/toolbar entry points) ----

    void set_real_time() { stepping_ = Stepping::RealTime; }

    void set_relaxing() {
        stepping_ = Stepping::Relaxing;
        relax_plateau_ = 0;
        relax_prev_energy_ = 0.0;
        if (!use_gpu_path()) {
            ensure_cpu_current();  // CPU relax (Surface view / no GPU)
        }
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
    }

    // Projective ENERGY measurement (the energy-basis analogue of M): request
    // a collapse onto an energy eigenstate sampled by |<phi_n|psi>|^2. The GPU
    // reductions + collapse-copy run in the frame's compute half, so the work
    // is deferred to run_frame (see pending_energy_measure_). Needs the
    // manifold.
    void measure_energy_now() {
        if (solving() || !use_gpu_path() || !manifold_ready()) {
            return;
        }
        pending_energy_measure_ = true;
        stepping_ = Stepping::RealTime;  // observe, then let H evolve it
        laser_pol_ = LaserPol::Off;
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
                stepping_ = Stepping::Relaxing;  // deflation is GPU-only
            }
            laser_pol_ = LaserPol::Off;  // the drive is GPU-only too
            decay_on_ = false;  // so are the jump trials: OFF beats a lying title
        }
        stage_active_view();
    }

    // Relax into the z-aligned first excited state.
    // The z-odd seed keeps the whole flow in the odd-parity sector, so it
    // converges to the 2p_z-like state deterministically.
    void relax_to_excited() {
        start_excited_relax(make_axis_odd_seed(2), "2p", false);
    }

    // Relax into 2s -- the radial node appears live. With decay ON it
    // then just SITS there: A(2s -> 1s) ~ 0 makes it metastable, from our
    // own matrix elements. The 2p triplet is deflated too (2s sits ABOVE
    // it), see start_excited_relax.
    void relax_to_2s() {
        start_excited_relax(
            ses::gaussian_wavepacket(sim_.grid(), ses::Vec3d{},
                                     ses::Vec3d{4.0, 4.0, 4.0}, ses::Vec3d{}),
            "2s", true);
    }

    // Toggle spontaneous decay (quantum jumps) over the whole tracked
    // manifold. Every channel rate is einstein_a(gap, |<f|r|i>|^2) from our
    // own eigenstates; true lifetimes are ~1e8 a.u., so the display
    // accelerates ALL channels by one common factor (title reports it
    // honestly and relative lifetimes stay physical).
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

    // Instantly excite an n = 3 state (cycles through a small set) and
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
    }

    // Cycle the laser off -> Z-pol -> X-pol -> off. The
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
    }

    // Static uniform E-field magnitude along +z (atomic units); 0 = off.
    // Acts in the GPU cloud/real-time path; the laser, if on, takes
    // precedence in the stepping branch.
    void set_efield_e0(double e0) {
        efield_e0_ = e0;
        upload_field_tables();  // fold E*z into the half-potential (with diamag if B on)
        if (e0 > 0.0 && !solving()) {
            stepping_ = Stepping::RealTime;  // let the field actually act
        }
    }

    // Magnetic field strength (au) along the current axis; 0 = off. PROPER
    // minimal-coupling solve: psi evolves under H = H0 + (B/2)L_axis +
    // (B^2/8)rho_perp^2. Uploading the diamagnetic-augmented half-potential
    // here means the per-frame magnetic_step only has to add the paramagnetic
    // rotation.
    void set_bfield_b(double b) {
        bfield_b_ = b;
        upload_field_tables();
        if (b > 0.0 && !solving()) {
            stepping_ = Stepping::RealTime;
        }
    }

    // Cycle the field axis z -> x -> y -> z. The diamagnetic term is
    // perpendicular to the axis, so the half-potential table is rebuilt.
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
    bool solving() const { return gpu_ok_ && !atlas_done_; }
    // Ready only once the FULL table is assembled (channels_ fills
    // incrementally during the pair phase -- do not race it).
    bool manifold_ready() const { return atlas_done_ && !atom_.channels().empty(); }
    double state_energy(int idx) const { return atom_.state_energy(idx); }
    long long photon_count() const { return photon_count_; }
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

    // Magnetic Larmor hooks: set psi to a manifold eigenstate and probe
    // another state's population. A field along z rotates 2p_x -> 2p_y at
    // omega_L = B/2, so P(2p_y) must rise -- proving the field evolves psi
    // itself, not just the display.
    void debug_prepare_state(int idx) {
        if (!manifold_ready() || idx < 0 || idx >= kNumStates) {
            return;
        }
        atom_.collapse_onto(engine_, idx);
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

    bool cloud() const { return mode_ == ViewMode::Cloud; }
    double peak() const { return peak_; }
    bool compute_attempted() const { return compute_attempted_; }
    bool gpu_ok() const { return gpu_ok_; }
    // The engine's bridge image on the GPU path; null lets the renderer fall
    // back to its CPU-staged texture.
    VkImageView psi_volume_view() { return gpu_ok_ ? engine_.volume_view() : VK_NULL_HANDLE; }
    // Photon flash: a brief warm background right after a quantum jump.
    float next_flash_intensity() {
        if (flash_ticks_ <= 0) {
            return 0.0f;
        }
        const float v = static_cast<float>(flash_ticks_) / 25.0f;
        --flash_ticks_;
        return v;
    }
    bool take_volume_written() {
        const bool w = volume_written_;
        volume_written_ = false;
        return w;
    }
    bool take_volume_dirty() {
        const bool d = volume_dirty_;
        volume_dirty_ = false;
        return d;
    }
    bool take_mesh_dirty() {
        const bool d = mesh_dirty_;
        mesh_dirty_ = false;
        return d;
    }
    void mark_display_dirty() {  // shell resize / first frame
        mesh_dirty_ = true;
        volume_dirty_ = true;
    }
    bool take_title_dirty() {
        const bool t = title_dirty_;
        title_dirty_ = false;
        return t;
    }
    const std::vector<float>& psi_staging() const { return psi_staging_; }
    const ses::Mesh& mesh() const { return mesh_; }
    const std::vector<ses::Rgb>& colors() const { return colors_; }

    // The full window-title readout, composed Qt-free.
    std::string title_text() {
        // Convergence readout while relaxing: exact <H> on the CPU session,
        // or the free ITP estimator (-ln||psi||^2 / 2 dtau) on the GPU path.
        std::string s = "Electron near a hydrogen nucleus   t = " +
                        strf("%.2f", sim_.time() + gpu_time_) + "   ";
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

    // One real-time step batch: propagate (driven / magnetic / plain),
    // absorb at the walls, then the title-cadence readouts and trials.
    void run_real_time_batch() {
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
            // Resonant dipole drive: t0 is the same clock
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

        // Orbital-free populations: ONE deposit pass on the
        // post-step psi, shared by the decay and laser readouts this
        // title tick (state-count independent) -- replaces the
        // per-state inner_with_psi over the resident atlas.
        if (gpu_title_due_ && proj_ready_ &&
            ((decay_on_ && !atom_.channels().empty()) ||
             laser_pol_ != LaserPol::Off)) {
            engine_.project_psi();
        }

        // Competing-channels Poisson trials over the
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
                    last_jump_ = strf("%s->%s", kStateSpec[ch.from].name,
                                      kStateSpec[ch.to].name);
                    title_dirty_ = true;
                }
            }
        }

        // Live populations for the title readout (and the
        // Rabi peak the selftest asserts on), on the title
        // cadence -- two 2 KB reductions every ~10 ticks.
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

        // Relaxation auto-completes: when the ITP energy
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

    // The common launcher for excited-state relaxation demos (keys 3/4):
    // ensure the deflation targets are synthesized, then hand the seed to the
    // GPU deflated imaginary-time flow. The live flow must deflate EVERY
    // state below the target: 2s sits above the 2p triplet, and with only
    // 1s removed the fp32 parity leakage of the GPU FFT (~1e-7/step) grows
    // as e^{(E2s-E2p) tau} until the on-screen "2s" morphs into 2p within
    // minutes (adversarial-review finding). 2p itself is safe with {1s}:
    // its only competitors are the DEGENERATE other 2p's (gap ~1e-6).
    void start_excited_relax(const ses::Field3D& seed, const char* label,
                             bool deflate_p_triplet) {
        if (!gpu_ok_ || solving()) {
            return;  // deflation runs on the GPU path only
        }
        if (mode_ != ViewMode::Cloud) {
            mode_ = ViewMode::Cloud;
        }
        if (!manifold_ready()) {
            return;
        }
        // Deflation set synthesized into OWNED transient fp32 buffers (no
        // resident atlas): freed at auto-complete / reset / the next
        // relaxation. relax_deflated_step reads them as fp32 phi every relax
        // frame.
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

    // Rebuild the GPU half-potential for the current static fields:
    //   V  +  E z (Stark, diagonal)  +  (B^2/8) rho_perp^2 (diamagnetic).
    // Both static-E and diamagnetic are diagonal in position, so folding them
    // in is exact (the old omega=0 dipole-kick equals this). The per-frame
    // magnetic_step then only adds the paramagnetic rotation. Crossed E-B
    // (E along z, B along x/y) is a genuine combined solve.
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

    // Free the OWNED transient deflation buffers (synthesized at relax-start).
    void free_deflation_buffers() {
        for (int b : relax_deflate_owned_) {
            engine_.release_state(b);
        }
        relax_deflate_owned_.clear();
        relax_deflate_.clear();
    }

    // Advance the startup atlas build by one chunk (called from run_frame).
    // Phase 1 synthesizes one orbital per visit and SHOWS it (the montage);
    // phase 2 evaluates dipole channel pairs; the finale assembles the
    // channel table and resumes the wavepacket.
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
            // ONE orbital at a time instead of accumulating the whole
            // manifold.
            double pk = 0.0;
            const int buf = atom_.synth_transient(engine_, idx, &pk);
            if (buf < 0) {
                atlas_done_ = true;  // GPU buffer alloc failed: give up gracefully
                return;
            }
            engine_.copy_into_psi(buf);  // show (fp32)
            // The h-audit: cross-check the 1D radial energy against the full 3D
            // spectral <H> for the resolution-critical 1s and the box-critical
            // 4s/5s/6s -- the ONLY states read back to the CPU.
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
                title_dirty_ = true;
            }
        }
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

    // Bridge psi to the display texture. The magnetic field evolves psi
    // itself (diamagnetic in the potential + exact three-shear paramagnetic
    // rotation), so the display is just the real wavefunction -- no
    // display-only rotation trick.
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
    bool pending_energy_measure_ = false;  // Key E: serviced in run_frame
    bool gpu_title_due_ = false;
    bool title_dirty_ = false;
    double gpu_time_ = 0.0;
    double norm_display_ = 1.0;
    double relax_energy_display_ = 0.0;
    double radiated_power_ = 0.0;  // semiclassical Larmor power (au)
    std::vector<float> readback_buf_;

    // The tracked atom: radial solve, eigenstate synthesis bookkeeping, and
    // the E1 decay channel table live in AtomModel; engine-backed calls pass
    // engine_ explicitly.
    ses_shell::AtomModel atom_;
    bool proj_ready_ = false;  // static projection index uploaded

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
    // Decay is the DEFAULT, as in nature; D is the off-switch for studying
    // pure unitary evolution. Armed once the startup atlas build finishes.
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
    double grid_energy_1s_ = 0.0;  // <H>_grid of the 1s (from the h-audit); the
                                   // laser drives THIS (grid) resonance, not the
                                   // radial label, since bare Coulomb leaves the
                                   // 1s a cusp gap on the coarse grid
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

    int mask_buf_ = -1;  // boundary absorber (mask, 0) complex state handle
    std::mt19937 rng_{std::random_device{}()};
};

}  // namespace ses_shell
