module;
#include <cstddef>
#include <algorithm>
#include <cmath>
#include <complex>
#include <memory>
#include <string>
#include <utility>
#include <vector>
export module ses.scenario.carpet1d_director;
export import ses.scenario.lattice2d_director;
import ses.heightfield;
import ses.propagator;
import ses.wavepacket;
import ses.parallel;


// Quantum carpet: a free packet on a PERIODIC RING (the FFT box itself --
// no walls, no wall artifacts) weaves the temporal Talbot carpet. With
// k_n = 2 pi n / L and E_n = k_n^2 / 2, every phase realigns at
// T_rev = L^2 / pi (EXACT on the spectral grid): the packet fully
// revives; at T/2 it reappears displaced by L/2 (the half-Talbot clone);
// rational fractions of T weave the fractional-revival lattice. The
// display stacks |psi(x)|^2 rows into an (x, t) height carpet, one full
// revival per carpet height -- after the first pass the wrap lands on
// the SAME pattern and the carpet stands still.
// CONTRACT: tests/carpet1d_test.cpp + --selftest-carpet.


export namespace ses_shell {

// Full revival time of the ring of circumference L.
inline double carpet_revival_time(double l) {
    return l * l / 3.14159265358979323846;
}

constexpr double kCp1dHalf = 20.0;
constexpr int kCp1dN = 1024;         // ring points AND carpet rows
constexpr int kCp1dNz = 4;
constexpr double kCp1dZHalf = 2.0;
constexpr double kCp1dDt = 0.05;     // V = 0: the split step is EXACT
constexpr double kCp1dX0 = -5.0;
constexpr double kCp1dSigma = 2.0;
constexpr double kCp1dK0 = 1.0;      // drift tilts the weave
constexpr int kCp1dStepsPerTick = 64;
constexpr double kCp1dSurfH = 6.0;
constexpr int kCp1dMeshStride = 2;   // 1024^2 carpet -> 512^2 mesh

class Carpet1DDirector final : public Lattice2DDirectorBase,
                               public CarpetApi {
public:
    Carpet1DDirector()
        : Lattice2DDirectorBase(kCp1dHalf, kCp1dN, kCp1dNz, kCp1dZHalf),
          grid1d_{-kCp1dHalf, kCp1dHalf, kCp1dN},
          psi1d_{grid1d_},
          init_{grid1d_},
          prop_{grid1d_,
                std::vector<double>(static_cast<std::size_t>(kCp1dN), 0.0),
                kCp1dDt} {
        // One carpet height = one revival: rows * row_steps * dt = T_rev.
        const double t_rev = carpet_revival_time(2.0 * kCp1dHalf);
        row_steps_ = std::max(
            1, static_cast<int>(t_rev / (kCp1dDt * kCp1dN) + 0.5));
        fire();
    }

    CarpetApi* carpet() override { return this; }

    // ---- CarpetApi ----
    void refire() override { fire(); }
    double revival_time() const override {
        return carpet_revival_time(2.0 * kCp1dHalf);
    }
    double revival_overlap() const override {
        std::complex<double> ov{};
        for (int i = 0; i < kCp1dN; ++i) {
            ov += std::conj(init_[i]) * psi1d_[i];
        }
        ov *= grid1d_.spacing();
        return std::norm(ov);
    }
    double mid_scramble_max() const override { return mid_max_; }
    double best_revival() const override { return best_; }

    void reset_simulation() override { fire(); }

    bool handle_key(char key) override {
        if (key == '2') {
            fire();
            return true;
        }
        return false;
    }

    double sim_dt() const override { return kCp1dDt; }
    int steps_per_tick() const override { return kCp1dStepsPerTick; }

    // ---- carpet display (mesh path; corral rule) ----
    bool cloud() const override { return false; }
    const ses::Mesh& mesh() const override { return hf_.mesh; }
    const std::vector<ses::Rgb>& colors() const override {
        return hf_.colors;
    }
    bool take_mesh_dirty() override {
        return std::exchange(mesh_dirty_, false);
    }
    double default_camera_azimuth() const override { return 0.35; }
    double default_camera_elevation() const override { return 0.95; }
    double default_camera_distance() const override { return 75.0; }

    std::string title_text() override {
        const double t_rev = revival_time();
        return strf(
            "Quantum carpet (free ring, temporal Talbot)  |  t = %.0f au  "
            "t/T_rev = %.2f  |<psi0|psi>|^2 = %.2f  T_rev = L^2/pi = %.0f "
            "au  keys: 2 refire",
            sim_time_, t_rev > 0.0 ? sim_time_ / t_rev : 0.0,
            revival_overlap(), t_rev);
    }

protected:
    void rebuild_display() override {
        double cur = 0.0;
        for (int j = 0; j < kCp1dN; ++j) {
            for (int i = 0; i < kCp1dN; ++i) {
                cur = std::max(cur, std::norm(psi_(i, j, 0)));
            }
        }
        disp_peak_ = disp_peak_ <= 0.0 ? cur
                                       : std::max(cur, 0.98 * disp_peak_);
        if (disp_peak_ > 0.0) {
            hf_ = ses::heightfield_surface(psi_, kCp1dSurfH, disp_peak_,
                                           kCp1dMeshStride);
            mesh_dirty_ = true;
        }
    }

    // Evolve the ring; every row_steps_ steps stamp |psi(x)|^2 (with its
    // phase) into the next carpet row -- the wrap after one revival lands
    // on the SAME weave, so the standing carpet is itself the proof.
    void do_steps(int n) override {
        for (int s = 0; s < n; ++s) {
            prop_.step(psi1d_, 1);
            if (++step_in_row_ >= row_steps_) {
                step_in_row_ = 0;
                for (int i = 0; i < kCp1dN; ++i) {
                    psi_(i, row_, 0) = psi1d_[i];
                }
                row_ = (row_ + 1) % kCp1dN;
                // Row-cadence revival tracking (the arc's oracle).
                const double t_rev = revival_time();
                const double frac = sim_time_ / t_rev;
                const double ov = revival_overlap();
                if (frac > 0.15 && frac < 0.6) {
                    mid_max_ = std::max(mid_max_, ov);
                } else if (frac >= 0.6) {
                    best_ = std::max(best_, ov);
                }
            }
        }
        sim_time_ += n * kCp1dDt;
        track_peak();
    }

private:
    void fire() {
        psi1d_ = ses::gaussian_wavepacket(grid1d_, kCp1dX0, kCp1dSigma,
                                          kCp1dK0);
        init_ = psi1d_;
        for (auto& c : psi_.data()) {
            c = 0.0;
        }
        row_ = 0;
        step_in_row_ = 0;
        disp_peak_ = 0.0;
        mid_max_ = 0.0;
        best_ = 0.0;
        mark_fired();
    }

    ses::Grid1D grid1d_;
    ses::Field1D psi1d_;
    ses::Field1D init_;
    ses::SplitOperator1D prop_;
    ses::Heightfield hf_;
    bool mesh_dirty_ = false;
    double disp_peak_ = 0.0;
    int row_ = 0;
    int step_in_row_ = 0;
    int row_steps_ = 1;
    double mid_max_ = 0.0;
    double best_ = 0.0;
};

}  // namespace ses_shell
