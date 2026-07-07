#pragma once

// The magnetic split-operator propagator (magnetic-field arc). For an
// electron in a uniform field B = B z^ (symmetric gauge A = (B/2)(-y,x,0),
// atomic units) minimal coupling gives
//     H = 1/2 p^2 + V(r) + (B/2) L_z + (B^2/8) rho^2,   rho^2 = x^2 + y^2.
// This solves it PROPERLY -- psi genuinely evolves, not a display rotation:
//   - the diamagnetic (B^2/8) rho^2 is diagonal in position, so it folds into
//     the potential of an internal SplitOperator3D;
//   - the paramagnetic (B/2) L_z is the generator of z-rotations, so its
//     exp(-i (B/2) L_z tau) is the exact three-shear rotate_z, Strang-split
//     around the core:  R(a) . [halfV . kin . halfV] . R(a),  a = (B/2)(dt/2).
// [T, L_z] = 0 and, for a z-symmetric V, [V', L_z] = 0, so the split is exact
// there; for a general V it is the usual O(dt^2) Strang. Unitary throughout
// (the core is unitary; rotate_z is unitary).

#include <core/field.hpp>
#include <core/grid.hpp>
#include <core/propagator.hpp>
#include <core/rotation.hpp>

#include <cmath>
#include <cstddef>
#include <vector>

namespace ses {

class MagneticPropagator3D {
public:
    MagneticPropagator3D(const Grid3D& g, const std::vector<double>& v, double dt,
                         double bfield)
        : dt_(dt), half_angle_(0.5 * bfield * (0.5 * dt)), veff_(build_veff(g, v, bfield)),
          core_(g, veff_, dt) {}

    double dt() const { return dt_; }

    // V + (B^2/8) rho^2 (the diamagnetic-augmented potential the core uses).
    const std::vector<double>& effective_potential() const { return veff_; }

    // One magnetic step: R(a) . core-step . R(a), a = (B/2)(dt/2). Chained
    // half-rotations between successive core steps merge into the full
    // per-step Larmor angle (B/2) dt.
    void step(Field3D& psi, int nsteps = 1) const {
        for (int s = 0; s < nsteps; ++s) {
            if (half_angle_ != 0.0) {
                rotate_z(psi, half_angle_);
            }
            core_.step(psi, 1);
            if (half_angle_ != 0.0) {
                rotate_z(psi, half_angle_);
            }
        }
    }

private:
    static std::vector<double> build_veff(const Grid3D& g, const std::vector<double>& v,
                                          double bfield) {
        std::vector<double> veff = v;
        const double c = bfield * bfield / 8.0;
        if (c != 0.0) {
            for (int k = 0; k < g.z.n; ++k) {
                for (int j = 0; j < g.y.n; ++j) {
                    for (int i = 0; i < g.x.n; ++i) {
                        const double rho2 = g.x.coord(i) * g.x.coord(i) +
                                            g.y.coord(j) * g.y.coord(j);
                        veff[static_cast<std::size_t>(g.flat(i, j, k))] += c * rho2;
                    }
                }
            }
        }
        return veff;
    }

    double dt_;
    double half_angle_;
    std::vector<double> veff_;
    SplitOperator3D core_;
};

}  // namespace ses
