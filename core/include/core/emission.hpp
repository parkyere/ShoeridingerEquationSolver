#pragma once

// Semiclassical (Larmor) emission from the oscillating dipole:
//     P = (2/3) alpha^3 |d_ddot|^2,  d_ddot = <grad V> (Ehrenfest, a.u.).
// Coherent-superposition emission only: exactly 0 for a pure eigenstate,
// whose spontaneous decay is the Einstein-A quantum jumps (core/decay.hpp).

#include <core/decay.hpp>  // kFineStructureConstant
#include <core/field.hpp>
#include <core/grid.hpp>
#include <core/vec.hpp>

#include <cstddef>
#include <vector>

namespace ses {

// <grad V> = integral |psi|^2 grad V dr, grad V by central differences on the
// periodic grid (exact for a harmonic well's linear force).
inline Vec3d mean_potential_gradient(const Field3D& psi, const std::vector<double>& v,
                                     const Grid3D& g) {
    const int nx = g.x.n;
    const int ny = g.y.n;
    const int nz = g.z.n;
    const double inv2hx = 1.0 / (2.0 * g.x.spacing());
    const double inv2hy = 1.0 / (2.0 * g.y.spacing());
    const double inv2hz = 1.0 / (2.0 * g.z.spacing());
    Vec3d acc{};
    for (int k = 0; k < nz; ++k) {
        const int kp = (k + 1) % nz;
        const int km = (k - 1 + nz) % nz;
        for (int j = 0; j < ny; ++j) {
            const int jp = (j + 1) % ny;
            const int jm = (j - 1 + ny) % ny;
            for (int i = 0; i < nx; ++i) {
                const int ip = (i + 1) % nx;
                const int im = (i - 1 + nx) % nx;
                const double rho = norm_sq(psi(i, j, k));
                const double gx =
                    (v[static_cast<std::size_t>(g.flat(ip, j, k))] -
                     v[static_cast<std::size_t>(g.flat(im, j, k))]) * inv2hx;
                const double gy =
                    (v[static_cast<std::size_t>(g.flat(i, jp, k))] -
                     v[static_cast<std::size_t>(g.flat(i, jm, k))]) * inv2hy;
                const double gz =
                    (v[static_cast<std::size_t>(g.flat(i, j, kp))] -
                     v[static_cast<std::size_t>(g.flat(i, j, km))]) * inv2hz;
                acc.x += rho * gx;
                acc.y += rho * gy;
                acc.z += rho * gz;
            }
        }
    }
    const double dv = g.cell_volume();
    return Vec3d{acc.x * dv, acc.y * dv, acc.z * dv};
}

// Larmor radiated power P = (2/3) alpha^3 |d_ddot|^2 (atomic units).
inline double larmor_power(const Vec3d& dipole_accel) {
    const double a3 = kFineStructureConstant * kFineStructureConstant *
                      kFineStructureConstant;
    return (2.0 / 3.0) * a3 * dot(dipole_accel, dipole_accel);
}

}  // namespace ses
