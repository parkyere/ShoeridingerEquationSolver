#pragma once

// Time-dependent dipole drive: V_drive = amplitude * (axis . r) * cos(omega t).
// Enters the Strang step as half-kicks AROUND the untouched static tables,
//     psi <- kick(t0+dt) . halfV . IFFT . kinetic . FFT . halfV . kick(t0),
// which keeps global O(dt^2) accuracy.

#include <core/complex.hpp>
#include <core/fft.hpp>
#include <core/field.hpp>
import ses.grid;
#include <core/propagator.hpp>
import ses.vec;

#include <cmath>
#include <cstddef>
#include <vector>

namespace ses {

struct DipoleDrive {
    Vec3d axis;          // polarization direction (need not be unit; scales E0)
    double amplitude{};  // field strength E0 (atomic units)
    double omega{};      // angular frequency; 0 = static field
};

// psi *= exp(-i * theta * (axis . r)) with theta = amplitude cos(omega t) dt/2.
inline void apply_dipole_halfkick(Field3D& psi, const DipoleDrive& d, double t, double dt) noexcept {
    const double theta = d.amplitude * std::cos(d.omega * t) * 0.5 * dt;
    const Grid3D& g = psi.grid();
    for (int k = 0; k < g.z.n; ++k) {
        for (int j = 0; j < g.y.n; ++j) {
            for (int i = 0; i < g.x.n; ++i) {
                const double u = d.axis.x * g.x.coord(i) + d.axis.y * g.y.coord(j) +
                                 d.axis.z * g.z.coord(k);
                const double ang = -theta * u;
                psi(i, j, k) =
                    psi(i, j, k) * Complex<double>{std::cos(ang), std::sin(ang)};
            }
        }
    }
}

namespace drive_detail {

inline void multiply_table(Field3D& psi, const std::vector<Complex<double>>& table) noexcept {
    std::vector<Complex<double>>& a = psi.data();
    for (std::size_t i = 0; i < a.size(); ++i) {
        a[i] = a[i] * table[i];
    }
}

}  // namespace drive_detail

// Driven Strang steps built from the TESTED static tables of `prop` plus the
// dipole half-kicks. t0 is the physical time at the start of the first step.
inline void driven_step(Field3D& psi, const SplitOperator3D& prop, const DipoleDrive& d,
                        double t0, int nsteps) {
    const double dt = prop.dt();
    for (int s = 0; s < nsteps; ++s) {
        const double t = t0 + s * dt;
        apply_dipole_halfkick(psi, d, t, dt);
        drive_detail::multiply_table(psi, prop.half_potential_phase());
        fft(psi);
        drive_detail::multiply_table(psi, prop.kinetic_phase());
        ifft(psi);
        drive_detail::multiply_table(psi, prop.half_potential_phase());
        apply_dipole_halfkick(psi, d, t + dt, dt);
    }
}

}  // namespace ses
