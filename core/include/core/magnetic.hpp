#pragma once

// Magnetic split-operator propagator: uniform B along a coordinate axis
// (symmetric gauge, atomic units, minimal coupling),
//     H = 1/2 p^2 + V(r) + (B/2) L_axis + (B^2/8) rho_perp^2.
// Diamagnetic term folds into the internal SplitOperator3D's potential;
// paramagnetic exp(-i (B/2) L_axis tau) is the exact three-shear rotate_axis,
// Strang-split: R(a) . [halfV.kin.halfV] . R(a), a = (B/2)(dt/2). Unitary.

#include <core/field.hpp>
import ses.grid;
#include <core/propagator.hpp>
#include <core/rotation.hpp>

#include <cmath>
#include <cstddef>
#include <vector>

namespace ses {

class MagneticPropagator3D {
public:
    // axis: 0=x, 1=y, 2=z (the field direction).
    MagneticPropagator3D(const Grid3D& g, const std::vector<double>& v, double dt,
                         double bfield, int axis = 2)
        : dt_(dt), half_angle_(0.5 * bfield * (0.5 * dt)), axis_(axis),
          veff_(build_veff(g, v, bfield, axis)), core_(g, veff_, dt) {}

    constexpr double dt() const noexcept { return dt_; }

    // V + (B^2/8) rho_perp^2 (the diamagnetic-augmented potential the core uses).
    const std::vector<double>& effective_potential() const noexcept { return veff_; }

    // One magnetic step: R(a) . core-step . R(a), a = (B/2)(dt/2). Chained
    // half-rotations between successive core steps merge into the full
    // per-step Larmor angle (B/2) dt.
    void step(Field3D& psi, int nsteps = 1) const {
        for (int s = 0; s < nsteps; ++s) {
            if (half_angle_ != 0.0) {
                rotate_axis(psi, axis_, half_angle_);
            }
            core_.step(psi, 1);
            if (half_angle_ != 0.0) {
                rotate_axis(psi, axis_, half_angle_);
            }
        }
    }

private:
    // Diamagnetic term (B^2/8) rho_perp^2, rho_perp = distance from the field
    // axis (the two coordinates perpendicular to `axis`).
    static std::vector<double> build_veff(const Grid3D& g, const std::vector<double>& v,
                                          double bfield, int axis) {
        std::vector<double> veff = v;
        const double c = bfield * bfield / 8.0;
        if (c != 0.0) {
            for (int k = 0; k < g.z.n; ++k) {
                for (int j = 0; j < g.y.n; ++j) {
                    for (int i = 0; i < g.x.n; ++i) {
                        const double coord[3] = {g.x.coord(i), g.y.coord(j),
                                                 g.z.coord(k)};
                        double perp2 = 0.0;
                        for (int a = 0; a < 3; ++a) {
                            if (a != axis) {
                                perp2 += coord[a] * coord[a];
                            }
                        }
                        veff[static_cast<std::size_t>(g.flat(i, j, k))] += c * perp2;
                    }
                }
            }
        }
        return veff;
    }

    double dt_;
    double half_angle_;
    int axis_;
    std::vector<double> veff_;
    SplitOperator3D core_;
};

}  // namespace ses
