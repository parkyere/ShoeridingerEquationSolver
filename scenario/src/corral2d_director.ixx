module;
#include <algorithm>
#include <cstddef>
#include <cmath>
#include <complex>
#include <memory>
#include <string>
#include <utility>
#include <vector>
export module ses.scenario.corral2d_director;
export import ses.scenario.lattice2d_director;
import ses.heightfield;
import ses.potential;
import ses.imaginary_time;
import ses.observables;
import ses.propagator;
import ses.parallel;


// Quantum corral (Crommie, Lutz & Eigler, IBM 1993): 48 Fe adatom ring
// on Cu(111), circular standing waves. B = 0, so the propagator is the
// SPECTRAL split-operator on the (512, 512, 1) grid -- exact continuum
// k^2/2 dispersion, the faithful model of the Cu(111) 2DEG (physics-first
// rule: the Peierls lattice is reserved for scenes that need gauge-exact
// flux). States relax INSIDE the leaky fence by spectral imaginary time
// with per-step Gram-Schmidt deflation, annealed coarse -> fine dtau --
// [2] re-relaxes the ground, [3] captures the next state, [F] scatters a
// packet off the fence live. The adatoms render as shaded iron balls.
// Display is the STM-style HEIGHT surface: z = |psi|^2 (peak-tracked),
// phase as vertex color -- not the volume slab.
// Restored IBM physics (contracts: tests/corral2d_test.cpp): the surface
// state runs the Cu(111) effective mass m* = 0.38; the adatoms are ~50%-
// absorbing black dots (per-step damp exp(-W dt) on the bump profile); the
// substrate is INFINITE -- an outer absorber swallows leaked flux (no
// periodic wrap-around) and the surviving wavefunction is renormalized
// each tick (conditional state, norm stays 1). Damping acts in REAL time
// only; state prep (ITP + disc projection) stays unitary.


export namespace ses_shell {

constexpr double kCr2dBox = 16.0;
constexpr int kCr2dN = 512;
constexpr int kCr2dNz = 4;
constexpr double kCr2dZHalf = 2.0;
constexpr double kCr2dSurfH = 6.0;   // peak |psi|^2 -> 6 Bohr of height
constexpr int kCr2dMeshStride = 2;   // 512^2 physics -> 256^2 display mesh
constexpr double kCr2dDt = 0.02;
// Annealed relax: the coarse dtau settles fast (spectral kinetic decay is
// exact, no stability limit), the fine dtau then polishes the Strang
// [T,V] Trotter bias away before capture -- at 512^2 the k-space
// bandwidth is large enough that a single big dtau biases E0 visibly.
constexpr double kCr2dCoarseDtau = 0.05;
constexpr double kCr2dFineDtau = 0.005;
constexpr int kCr2dAtoms = 48;
constexpr double kCr2dR = 10.0;
constexpr double kCr2dRMin = 6.0;
constexpr double kCr2dRMax = 12.0;
constexpr double kCr2dBumpA = 1.5;
constexpr double kCr2dBumpSigma = 0.6;
// Cu(111) surface-state effective mass (Crommie et al.), atomic units.
constexpr double kCr2dMass = 0.38;
// Infinite-substrate open boundary: outer cos^2 absorber width (Bohr).
constexpr double kCr2dEdgeW = 3.0;
// Black-dot adatoms: peak of the absorber W(r) riding the bump profile;
// exp(-W dt) per step eats a PART of a crossing packet (never all of it).
constexpr double kCr2dFenceW0 = 0.8;
constexpr int kCr2dMaxStates = 6;
// Relax convergence: energy drift per check below this for 3 checks.
constexpr double kCr2dConvTol = 1e-7;
// Coarse-stage plateau tolerance: crossing it flips to the fine stage.
constexpr double kCr2dCoarseTol = 1e-5;

class Corral2DDirector final : public Lattice2DDirectorBase,
                               public CorralApi {
public:
    Corral2DDirector()
        : Lattice2DDirectorBase(kCr2dBox, kCr2dN, kCr2dNz, kCr2dZHalf) {
        rebuild_potential();
        start_relax(0);
    }

    CorralApi* corral() override { return this; }

    // ---- CorralApi ----
    void set_radius(double r) override {
        radius_ = std::clamp(r, kCr2dRMin, kCr2dRMax);
        rebuild_potential();
        start_relax(0);
    }
    double radius() const override { return radius_; }
    double mass() const override { return kCr2dMass; }
    void relax_next() override {
        if (!relaxing_ &&
            static_cast<int>(captured_.size()) < kCr2dMaxStates) {
            start_relax(static_cast<int>(captured_.size()));
        }
    }
    int captured() const override {
        return static_cast<int>(captured_.size());
    }
    double energy(int k) const override {
        return k >= 0 && k < static_cast<int>(energies_.size())
                   ? energies_[static_cast<std::size_t>(k)]
                   : 0.0;
    }
    double confinement() const override {
        double inside = 0.0;
        double total = 0.0;
        for (int j = 0; j < kCr2dN; ++j) {
            for (int i = 0; i < kCr2dN; ++i) {
                const double w = std::norm(psi_(i, j, 0));
                total += w;
                if (std::hypot(phys_grid_.x.coord(i),
                               phys_grid_.y.coord(j)) < radius_ - 1.0) {
                    inside += w;
                }
            }
        }
        return total > 0.0 ? inside / total : 0.0;
    }
    bool relaxing() const override { return relaxing_; }
    void fire_packet() override {
        disp_peak_ = 0.0;  // re-snap: the packet scale differs from a ground
        relaxing_ = false;
        ses::parallel_for(kCr2dN, [&](int j) {
            const double y = phys_grid_.y.coord(j);
            for (int i = 0; i < kCr2dN; ++i) {
                const double x = phys_grid_.x.coord(i);
                psi_(i, j, 0) =
                    std::exp(-(x * x + y * y) / (4.0 * 1.5 * 1.5));
            }
        });
        ses::normalize(psi_);
        track_peak();
        mark_fired();
    }

    // CONTRACT: tests/corral2d_test.cpp FermiWaveIsQuasiStationary.
    void fermi_wave() override {
        // Node count is fence-limited, not display-limited: the E_F wave
        // must still SCATTER off the sigma = 0.6 bumps. E = (j0_n/R)^2 /
        // (2 m*) climbs fast (j0_10 -> 12.3 Ha vs the 1.5 fence: smooth
        // bumps are transparent there -- the real corral reflects at E_F
        // via Fe d-resonance phase shifts our toy cannot mimic), so the
        // demo uses the highest count the fence quasi-confines.
        const double j0_n = 10.1734681350627;  // 3rd zero of J0 (E < saddle)
        const double kf = j0_n / radius_;
        disp_peak_ = 0.0;  // re-snap: ~10 rings are much flatter than a dome
        relaxing_ = false;
        ses::parallel_for(kCr2dN, [&](int j) {
            const double y = phys_grid_.y.coord(j);
            for (int i = 0; i < kCr2dN; ++i) {
                const double x = phys_grid_.x.coord(i);
                const double r = std::sqrt(x * x + y * y);
                // J0(kf r) inside, smooth cos^2 rolloff across the fence.
                double w = 1.0;
                if (r > radius_) {
                    const double t = (r - radius_) / (2.0 * kCr2dBumpSigma);
                    w = t >= 1.0 ? 0.0 : std::cos(0.5 * 3.14159265358979323846 * t);
                    w *= w;
                }
                psi_(i, j, 0) = w * std::cyl_bessel_j(0.0, kf * r);
            }
        });
        ses::normalize(psi_);
        track_peak();
        mark_fired();
        title_dirty_ = true;
    }

    void reset_simulation() override {
        captured_.clear();
        energies_.clear();
        start_relax(0);
    }

    bool handle_key(char key) override {
        if (key == '5') {
            fermi_wave();
            return true;
        }
        if (key == '2') {
            reset_simulation();
            return true;
        }
        if (key == '3') {
            relax_next();
            return true;
        }
        if (key == 'F') {
            fire_packet();
            return true;
        }
        return false;
    }

    double sim_dt() const override { return kCr2dDt; }

    // ---- STM-style surface display (mesh path; cloud off) ----
    bool cloud() const override { return false; }
    const ses::Mesh& mesh() const override { return hf_.mesh; }
    const std::vector<ses::Rgb>& colors() const override {
        return hf_.colors;
    }
    bool take_mesh_dirty() override {
        return std::exchange(mesh_dirty_, false);
    }

    // Tilted boot view so the height relief reads (face-on would hide it).
    double default_camera_azimuth() const override { return 0.35; }
    double default_camera_elevation() const override { return 0.95; }
    double default_camera_distance() const override { return 60.0; }

    // The 48 iron adatoms as shaded balls (CPK-ish iron orange).
    int marker_count() const override { return kCr2dAtoms; }
    SceneMarker marker(int i) const override {
        const double pi = 3.14159265358979323846;
        const double th = 2.0 * pi * i / kCr2dAtoms;
        return {static_cast<float>(radius_ * std::cos(th)),
                static_cast<float>(radius_ * std::sin(th)),
                0.0f,
                0.45f,
                0.88f,
                0.42f,
                0.15f};
    }

    std::string title_text() override {
        std::string s = strf(
            "Quantum corral (48 adatoms, IBM 1993)  |  t = %.1f au  "
            "R = %.1f  states %d",
            sim_time_, radius_, captured());
        for (std::size_t k = 0; k < energies_.size(); ++k) {
            s += strf("  E%zu = %.4f", k, energies_[k]);
        }
        if (relaxing_) {
            s += "  [relaxing...]";
        }
        s += "  keys: 2 ground / 3 next state / 5 Fermi wave / F packet";
        return s;
    }

protected:
    // Heightfield surface instead of the volume slab: peak-tracked height
    // keeps the relief visible through relax; phase rides as vertex color.
    // Surface normalizer: SNAP to the first observed max (the base's
    // peak_ starts at an arbitrary 1.0 and only decays 2%/frame -- heavy
    // relax frames are far too few to reach the true ~1e-2 scale, which
    // squashed the dome to ~0.1 Bohr), then 0.98-decay smoothing.
    void rebuild_display() override {
        double cur = 0.0;
        for (int j = 0; j < kCr2dN; ++j) {
            for (int i = 0; i < kCr2dN; ++i) {
                cur = std::max(cur, std::norm(psi_(i, j, 0)));
            }
        }
        disp_peak_ = disp_peak_ <= 0.0 ? cur
                                       : std::max(cur, 0.98 * disp_peak_);
        hf_ = ses::heightfield_surface(psi_, kCr2dSurfH, disp_peak_,
                                       kCr2dMeshStride);
        mesh_dirty_ = true;
    }

    void do_steps(int n) override {
        if (relaxing_) {
            // Imaginary time runs at 4x the tick budget (same cost per
            // step as real time; convergence is what the user waits on).
            // Per-step Gram-Schmidt deflation rides inside relax_deflated;
            // the disc projection between chunks makes the PREPARATION the
            // closed-corral Dirichlet problem (the J0 object) -- without
            // it the leaky fence honestly drains the relax into the lower
            // outside quasi-continuum and no plateau ever forms. Real time
            // (F key) runs the true leaky Hamiltonian, unprojected.
            std::vector<const ses::Field3D*> lower;
            lower.reserve(captured_.size());
            for (const ses::Field3D& s : captured_) {
                lower.push_back(&s);
            }
            int left = 4 * n;
            const int chunk = fine_ ? 100 : 20;  // tau ~ 0.5 / 1.0 per cut
            while (left > 0) {
                const int c = std::min(left, chunk);
                (fine_ ? itp_fine_ : itp_coarse_)
                    ->relax_deflated(psi_, lower, c);
                project_disc();
                left -= c;
            }
            const double e = ses::mean_energy(psi_, v_, kCr2dMass);
            const double tol = fine_ ? kCr2dConvTol : kCr2dCoarseTol;
            if (std::abs(e - last_e_) < tol * std::max(1.0, e)) {
                if (!fine_) {
                    fine_ = true;  // coarse settled: polish at small dtau
                    conv_streak_ = 0;
                } else if (++conv_streak_ >= 3) {
                    capture(e);
                }
            } else {
                conv_streak_ = 0;
            }
            last_e_ = e;
            track_peak();
            return;
        }
        // Per-step damp (black dots + open boundary), then the conditional
        // renormalization: the lost norm is escaped/absorbed probability.
        for (int s2 = 0; s2 < n; ++s2) {
            prop_->step(psi_, 1);
            ses::parallel_for(static_cast<int>(damp_.size()), [&](int i) {
                psi_.data()[static_cast<std::size_t>(i)] *=
                    damp_[static_cast<std::size_t>(i)];
            });
        }
        ses::normalize(psi_);
        sim_time_ += n * kCr2dDt;
        track_peak();
    }

private:
    void rebuild_potential() {
        const double pi = 3.14159265358979323846;
        std::vector<double> v(
            static_cast<std::size_t>(phys_grid_.size()), 0.0);
        for (int a = 0; a < kCr2dAtoms; ++a) {
            const double th = 2.0 * pi * a / kCr2dAtoms;
            const double ax = radius_ * std::cos(th);
            const double ay = radius_ * std::sin(th);
            for (int j = 0; j < kCr2dN; ++j) {
                const double dy = phys_grid_.y.coord(j) - ay;
                if (std::abs(dy) > 4.0 * kCr2dBumpSigma) {
                    continue;
                }
                for (int i = 0; i < kCr2dN; ++i) {
                    const double dx = phys_grid_.x.coord(i) - ax;
                    if (std::abs(dx) > 4.0 * kCr2dBumpSigma) {
                        continue;
                    }
                    v[static_cast<std::size_t>(
                        phys_grid_.flat(i, j, 0))] +=
                        kCr2dBumpA *
                        std::exp(-(dx * dx + dy * dy) /
                                 (2.0 * kCr2dBumpSigma * kCr2dBumpSigma));
                }
            }
        }
        // Restored physics: per-step damp = outer open-boundary absorber
        // (infinite substrate -- leaked flux vanishes) x black-dot fence
        // absorption exp(-W dt), W = kCr2dFenceW0 * bump/A. Mirrors
        // tests/corral2d_test.cpp exactly.
        const std::vector<double> edge =
            ses::absorbing_mask(phys_grid_, kCr2dEdgeW);
        damp_.assign(v.size(), 1.0);
        for (std::size_t i = 0; i < v.size(); ++i) {
            damp_[i] = edge[i] * std::exp(-kCr2dFenceW0 *
                                          (v[i] / kCr2dBumpA) * kCr2dDt);
        }
        v_ = std::move(v);
        prop_ = std::make_unique<ses::SplitOperator3D>(phys_grid_, v_,
                                                      kCr2dDt, kCr2dMass);
        itp_coarse_ = std::make_unique<ses::ImaginaryTimePropagator3D>(
            phys_grid_, v_, kCr2dCoarseDtau, kCr2dMass);
        itp_fine_ = std::make_unique<ses::ImaginaryTimePropagator3D>(
            phys_grid_, v_, kCr2dFineDtau, kCr2dMass);
        captured_.clear();
        energies_.clear();
    }

    void start_relax(int k) {
        // Seed with enough asymmetry to overlap the next mode; the
        // deflation projects the captured states out either way.
        const double off = k == 0 ? 0.0 : 0.35 * radius_;
        ses::parallel_for(kCr2dN, [&](int j) {
            const double y = phys_grid_.y.coord(j);
            for (int i = 0; i < kCr2dN; ++i) {
                const double x = phys_grid_.x.coord(i);
                const double dx = x - off;
                psi_(i, j, 0) =
                    std::exp(-(dx * dx + y * y) / (4.0 * 4.0 * 4.0));
            }
        });
        ses::normalize(psi_);
        deflate();
        relaxing_ = true;
        fine_ = false;
        disp_peak_ = 0.0;  // re-snap the surface normalizer
        conv_streak_ = 0;
        last_e_ = 1e30;
        mark_fired();
    }

    // Preparation-only Dirichlet cut at the fence centerline: zero psi
    // outside r = R and renormalize (projected ITP = the closed-corral
    // eigenproblem). Never applied in real time.
    void project_disc() {
        ses::parallel_for(kCr2dN, [&](int j) {
            const double y = phys_grid_.y.coord(j);
            for (int i = 0; i < kCr2dN; ++i) {
                if (std::hypot(phys_grid_.x.coord(i), y) > radius_) {
                    psi_(i, j, 0) = 0.0;
                }
            }
        });
        ses::normalize(psi_);
    }

    // Gram-Schmidt against every captured state (the deflation).
    void deflate() {
        const double cell = phys_grid_.x.spacing() *
                            phys_grid_.y.spacing() *
                            phys_grid_.z.spacing();
        for (const ses::Field3D& s : captured_) {
            std::complex<double> ov{};
            for (int i = 0; i < phys_grid_.size(); ++i) {
                ov += std::conj(s.data()[static_cast<std::size_t>(i)]) *
                      psi_.data()[static_cast<std::size_t>(i)];
            }
            ov *= cell;
            for (int i = 0; i < phys_grid_.size(); ++i) {
                psi_.data()[static_cast<std::size_t>(i)] -=
                    ov * s.data()[static_cast<std::size_t>(i)];
            }
        }
        ses::normalize(psi_);
    }

    void capture(double e) {
        captured_.push_back(psi_);
        energies_.push_back(e);
        relaxing_ = false;
        title_dirty_ = true;
    }

    std::vector<double> v_;  // fence potential (mean_energy readout)
    std::vector<double> damp_;  // per-step black-dot x open-boundary factor
    std::unique_ptr<ses::SplitOperator3D> prop_;
    std::unique_ptr<ses::ImaginaryTimePropagator3D> itp_coarse_;
    std::unique_ptr<ses::ImaginaryTimePropagator3D> itp_fine_;
    ses::Heightfield hf_;
    bool mesh_dirty_ = false;
    std::vector<ses::Field3D> captured_;
    std::vector<double> energies_;
    double radius_ = kCr2dR;
    double last_e_ = 1e30;
    double disp_peak_ = 0.0;  // surface height normalizer (snap-first)
    int conv_streak_ = 0;
    bool relaxing_ = false;
    bool fine_ = false;  // anneal stage: coarse dtau -> fine polish
};

}  // namespace ses_shell
