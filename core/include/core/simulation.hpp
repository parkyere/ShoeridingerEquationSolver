#pragma once

// WavepacketSimulation: the tested orchestration layer the app shell drives.
// Owns grid + potential + propagator + psi; advance() steps real time.
// Exactly equivalent to gaussian_wavepacket followed by SplitOperator3D
// steps (pinned by tests) -- no hidden physics lives here.

#include <core/field.hpp>
import ses.grid;
#include <core/imaginary_time.hpp>
#include <core/measurement.hpp>
#include <core/propagator.hpp>
import ses.vec;
#include <core/wavepacket.hpp>

#include <optional>
#include <utility>
#include <vector>

namespace ses {

class WavepacketSimulation {
public:
    struct Config {
        Grid3D grid;
        std::vector<double> potential;
        Vec3d r0;
        Vec3d sigma;
        Vec3d k0;
        double dt;
    };

    explicit WavepacketSimulation(Config cfg)
        : grid_(cfg.grid),
          dt_(cfg.dt),
          potential_(std::move(cfg.potential)),
          propagator_(cfg.grid, potential_, cfg.dt),
          psi_(gaussian_wavepacket(cfg.grid, cfg.r0, cfg.sigma, cfg.k0)) {}

    void advance(int nsteps) {
        propagator_.step(psi_, nsteps);
        steps_ += nsteps;
    }

    // Imaginary-time relaxation toward the ground state. Does NOT advance
    // real time (tau is not t). The propagator's decay tables are cached and
    // rebuilt only when dtau changes.
    void relax(int nsteps, double dtau) {
        if (!relaxer_ || relaxer_dtau_ != dtau) {
            relaxer_.emplace(grid_, potential_, dtau);
            relaxer_dtau_ = dtau;
        }
        relaxer_->relax(psi_, nsteps);
    }

    // Soft position measurement: sample a collapse cell from the sigma_m-
    // blurred POVM outcome density (detector-consistent Born rule) using the
    // injected uniform draw u, collapse psi onto a Gaussian of width sigma_m
    // there, and return the collapse center. Instantaneous.
    Vec3d measure(double u, double sigma_m) {
        const int idx = sample_povm_index(psi_, sigma_m, u);
        const int i = idx % grid_.x.n;
        const int j = (idx / grid_.x.n) % grid_.y.n;
        const int k = idx / (grid_.x.n * grid_.y.n);
        const Vec3d center{grid_.x.coord(i), grid_.y.coord(j), grid_.z.coord(k)};
        collapse_wavepacket(psi_, center, sigma_m);
        return center;
    }

    constexpr double time() const noexcept { return steps_ * dt_; }
    constexpr double dt() const noexcept { return dt_; }
    const Grid3D& grid() const noexcept { return grid_; }
    const Field3D& psi() const noexcept { return psi_; }
    const std::vector<double>& potential() const noexcept { return potential_; }
    const SplitOperator3D& propagator() const noexcept { return propagator_; }

    // Replace the state (e.g. hand a GPU-evolved field back to this CPU
    // session). Grid sizes must match.
    void set_psi(const Field3D& psi) {
        assert(psi.data().size() == psi_.data().size());
        psi_ = psi;
    }
    std::vector<double> density() const { return probability_density(psi_); }

private:
    Grid3D grid_;
    double dt_;
    std::vector<double> potential_;
    SplitOperator3D propagator_;
    Field3D psi_;
    long long steps_ = 0;
    std::optional<ImaginaryTimePropagator3D> relaxer_;
    double relaxer_dtau_ = 0.0;
};

}  // namespace ses
