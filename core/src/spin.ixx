module;
#include <algorithm>
#include <cmath>
#include <complex>
export module ses.spin;


// A single electron spin, pinned in space: the 2-spinor under
// H = (1/2) B . sigma (atomic units, gamma folded in -- Larmor omega =
// |B|). Every step is the EXACT SU(2) rotation exp(-i |B| dt/2 B_hat .
// sigma) (Rodrigues on 2x2: cos - i sin n.sigma), so the norm is exact
// to round-off and there is no Trotter anywhere. A static E field adds
// only a global phase to a pinned spin (no electric dipole): the scene
// displays it honestly as flux, never as Bloch motion.
// CONTRACT: tests/spin_test.cpp (Larmor rate, Rabi flip, collapse, echo).


export namespace ses {

struct Spinor {
    std::complex<double> up{1.0, 0.0};
    std::complex<double> dn{0.0, 0.0};
};

// Exact rotation by `angle` about the UNIT axis n:
// exp(-i angle/2 n.sigma) = cos(angle/2) - i sin(angle/2) n.sigma.
inline void spin_rotate(Spinor& s, double nx, double ny, double nz,
                        double angle) {
    const double c = std::cos(0.5 * angle);
    const double sn = std::sin(0.5 * angle);
    const std::complex<double> i{0.0, 1.0};
    const std::complex<double> up =
        (c - i * sn * nz) * s.up - i * sn * (nx - i * ny) * s.dn;
    const std::complex<double> dn =
        -i * sn * (nx + i * ny) * s.up + (c + i * sn * nz) * s.dn;
    s.up = up;
    s.dn = dn;
}

// One dt under H = (1/2) B . sigma: rotation by |B| dt about B_hat.
inline void spin_step(Spinor& s, double bx, double by, double bz,
                      double dt) {
    const double b = std::sqrt(bx * bx + by * by + bz * bz);
    if (b <= 0.0) {
        return;
    }
    spin_rotate(s, bx / b, by / b, bz / b, b * dt);
}

// The Bloch vector <sigma>.
inline void bloch_vector(const Spinor& s, double* x, double* y,
                         double* z) {
    const std::complex<double> cr = std::conj(s.up) * s.dn;
    *x = 2.0 * cr.real();
    *y = 2.0 * cr.imag();
    *z = std::norm(s.up) - std::norm(s.dn);
}

// Born measurement along the UNIT axis n: collapse to |+n> (return +1)
// when u < p_plus, else |-n> (return -1). |+-n> built by rotating |up>
// onto +-n (axis = z x n).
inline int spin_measure(Spinor& s, double nx, double ny, double nz,
                        double u) {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    bloch_vector(s, &x, &y, &z);
    const double p_plus = 0.5 * (1.0 + x * nx + y * ny + z * nz);
    const int outcome = u < p_plus ? +1 : -1;
    const double tx = outcome * nx;
    const double ty = outcome * ny;
    const double tz = outcome * nz;
    Spinor eig;  // |up>, rotated onto the target axis
    const double th = std::acos(std::clamp(tz, -1.0, 1.0));
    const double axn = std::hypot(-ty, tx);  // z x t
    if (axn > 1e-12) {
        spin_rotate(eig, -ty / axn, tx / axn, 0.0, th);
    } else if (tz < 0.0) {
        spin_rotate(eig, 1.0, 0.0, 0.0, 3.14159265358979323846);
    }
    s = eig;
    return outcome;
}

}  // namespace ses
