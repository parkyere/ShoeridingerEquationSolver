#pragma once

// Exact, unitary rotation of a 3D field about the z-axis (magnetic-field arc).
// The paramagnetic term (B/2) L_z of the magnetic Hamiltonian generates
// z-rotations, so evolving psi under it is a rigid rotation of the ACTUAL
// wavefunction. Done by the three-shear (Paeth) decomposition
//     R(theta) = X-shear(-tan(theta/2)) . Y-shear(sin theta) . X-shear(-tan(theta/2)),
// each shear a per-line SHIFT applied EXACTLY via the Fourier shift theorem
// (X(k) *= e^{-i k d}). This is information-preserving: exactly norm-
// conserving (each factor is a Fourier phase multiply) and free of the
// interpolation blur a resample would add. Valid for |theta| < pi (the shear
// tangent blows up at pi); the propagator drives it with small per-step
// angles.

#include <core/complex.hpp>
#include <core/fft.hpp>
#include <core/field.hpp>
#include <core/grid.hpp>
#include <core/spectral.hpp>

#include <cmath>
#include <vector>

namespace ses {

namespace rotation_detail {

// Shift every x-line by d = coeff * y (the line's y coordinate), exactly, via
// the Fourier shift theorem: line(x) -> line(x - d)  <=>  X(k) *= e^{-i k d}.
inline void x_shear(Field3D& f, double coeff) {
    const Grid3D& g = f.grid();
    const std::vector<double> kx = wavenumbers(g.x);
    std::vector<Complex<double>> line(static_cast<std::size_t>(g.x.n));
    for (int k = 0; k < g.z.n; ++k) {
        for (int j = 0; j < g.y.n; ++j) {
            const double d = coeff * g.y.coord(j);
            for (int i = 0; i < g.x.n; ++i) {
                line[static_cast<std::size_t>(i)] = f(i, j, k);
            }
            fft(line);
            for (int i = 0; i < g.x.n; ++i) {
                const double ph = -kx[static_cast<std::size_t>(i)] * d;
                line[static_cast<std::size_t>(i)] =
                    line[static_cast<std::size_t>(i)] *
                    Complex<double>{std::cos(ph), std::sin(ph)};
            }
            ifft(line);
            for (int i = 0; i < g.x.n; ++i) {
                f(i, j, k) = line[static_cast<std::size_t>(i)];
            }
        }
    }
}

// Shift every y-line by d = coeff * x (the line's x coordinate).
inline void y_shear(Field3D& f, double coeff) {
    const Grid3D& g = f.grid();
    const std::vector<double> ky = wavenumbers(g.y);
    std::vector<Complex<double>> line(static_cast<std::size_t>(g.y.n));
    for (int k = 0; k < g.z.n; ++k) {
        for (int i = 0; i < g.x.n; ++i) {
            const double d = coeff * g.x.coord(i);
            for (int j = 0; j < g.y.n; ++j) {
                line[static_cast<std::size_t>(j)] = f(i, j, k);
            }
            fft(line);
            for (int j = 0; j < g.y.n; ++j) {
                const double ph = -ky[static_cast<std::size_t>(j)] * d;
                line[static_cast<std::size_t>(j)] =
                    line[static_cast<std::size_t>(j)] *
                    Complex<double>{std::cos(ph), std::sin(ph)};
            }
            ifft(line);
            for (int j = 0; j < g.y.n; ++j) {
                f(i, j, k) = line[static_cast<std::size_t>(j)];
            }
        }
    }
}

}  // namespace rotation_detail

// Rotate the field about +z by theta (active, counterclockwise), exactly and
// unitarily. |theta| < pi.
inline void rotate_z(Field3D& f, double theta) {
    const double t = std::tan(0.5 * theta);
    const double s = std::sin(theta);
    rotation_detail::x_shear(f, -t);
    rotation_detail::y_shear(f, s);
    rotation_detail::x_shear(f, -t);
}

}  // namespace ses
