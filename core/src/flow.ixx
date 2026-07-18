module;
#include <complex>
#include <algorithm>
export module ses.flow;
import ses.vec;


// Probability current and the Bohmian velocity field it induces -- the physics
// behind the streakline flow visualization (and the raw material for an <L_z>
// diagnostic). Atomic units (hbar = m_e = 1). Vec3d arrives via import
// ses.vec; std::complex rides in the GMF, so consumers naming it still
// #include <complex> themselves.


export namespace ses {

// Probability current j = Im(conj(psi) grad psi) = rho v (Madelung).
inline Vec3d probability_current(std::complex<double> psi, std::complex<double> dpsi_dx,
                                 std::complex<double> dpsi_dy,
                                 std::complex<double> dpsi_dz) noexcept {
    const auto jc = [&](std::complex<double> d) {
        return psi.real() * d.imag() - psi.imag() * d.real();
    };
    return Vec3d{jc(dpsi_dx), jc(dpsi_dy), jc(dpsi_dz)};
}

// Bohmian velocity v = j / rho (rho = |psi|^2), guarded at nodes where rho->0.
inline Vec3d bohmian_velocity(std::complex<double> psi, std::complex<double> dpsi_dx,
                              std::complex<double> dpsi_dy,
                              std::complex<double> dpsi_dz) noexcept {
    const double rho = psi.real() * psi.real() + psi.imag() * psi.imag();
    const Vec3d j = probability_current(psi, dpsi_dx, dpsi_dy, dpsi_dz);
    const double inv = rho > 1e-12 ? 1.0 / rho : 0.0;
    return Vec3d{j.x * inv, j.y * inv, j.z * inv};
}

// Streakline trail fade: tail (v = 0, oldest) transparent -> head
// (v = trail_len-1, newest) opaque. Linear ramp in [0, 1].
inline double trail_fade(int v, int trail_len) noexcept {
    if (trail_len <= 1) {
        return 1.0;
    }
    const double t =
        static_cast<double>(v) / static_cast<double>(trail_len - 1);
    return std::clamp(t, 0.0, 1.0);
}

}  // namespace ses
