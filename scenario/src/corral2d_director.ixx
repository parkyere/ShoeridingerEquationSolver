module;
#include <algorithm>
#include <cmath>
#include <complex>
#include <memory>
#include <string>
#include <utility>
#include <vector>
export module ses.scenario.corral2d_director;
export import ses.scenario.lattice2d_director;
import ses.parallel;


// The quantum corral -- Crommie, Lutz & Eigler (IBM, 1993): 48 iron
// adatoms placed in a ring on Cu(111), the surface electrons trapped
// inside as circular standing waves (the iconic STM ripple image). Here:
// 48 Gaussian bumps on a ring of radius R, states relaxed INSIDE the
// leaky fence by lattice imaginary time with Gram-Schmidt deflation --
// [2] re-relaxes the ground (the central dome + rings), [3] captures the
// next state (higher ripple modes), [F] scatters a packet off the fence
// live. The adatoms render as shaded iron balls (the marker machinery).


export namespace ses_shell {

constexpr double kCr2dBox = 16.0;
constexpr int kCr2dN = 256;
constexpr int kCr2dNz = 4;
constexpr double kCr2dZHalf = 2.0;
constexpr double kCr2dDt = 0.02;
constexpr int kCr2dAtoms = 48;
constexpr double kCr2dR = 10.0;
constexpr double kCr2dRMin = 6.0;
constexpr double kCr2dRMax = 12.0;
constexpr double kCr2dBumpA = 1.5;
constexpr double kCr2dBumpSigma = 0.6;
constexpr int kCr2dMaxStates = 6;
// Relax convergence: energy drift per check below this for 3 checks.
constexpr double kCr2dConvTol = 1e-7;

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

    void reset_simulation() override {
        captured_.clear();
        energies_.clear();
        start_relax(0);
    }

    bool handle_key(char key) override {
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
    double default_camera_distance() const override { return 95.0; }

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
        s += "  keys: 2 ground / 3 next state / F packet";
        return s;
    }

protected:
    void do_steps(int n) override {
        if (relaxing_) {
            // Imaginary time runs at 4x the tick budget (same cost per
            // step as real time; convergence is what the user waits on).
            prop_->relax(psi_, 4 * n);
            deflate();
            const double e = prop_->energy(psi_);
            if (std::abs(e - last_e_) < kCr2dConvTol * std::max(1.0, e)) {
                if (++conv_streak_ >= 3) {
                    capture(e);
                }
            } else {
                conv_streak_ = 0;
            }
            last_e_ = e;
            track_peak();
            return;
        }
        prop_->step(psi_, n);
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
        prop_ = std::make_unique<ses::PeierlsLattice2D>(phys_grid_, v,
                                                        kCr2dDt);
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
        conv_streak_ = 0;
        last_e_ = 1e30;
        mark_fired();
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

    std::unique_ptr<ses::PeierlsLattice2D> prop_;
    std::vector<ses::Field3D> captured_;
    std::vector<double> energies_;
    double radius_ = kCr2dR;
    double last_e_ = 1e30;
    int conv_streak_ = 0;
    bool relaxing_ = false;
};

}  // namespace ses_shell
