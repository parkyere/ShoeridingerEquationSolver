module;
#include <complex>
#include <cmath>
#include <cstddef>
#include <vector>
export module ses.harmonics;
export import ses.radial;
export import ses.field;
export import ses.grid;
import ses.parallel;


// Real spherical harmonics (l <= 5) and 3D orbital synthesis:
// psi = (u_nl(r)/r) Y_lm from the radial engine's u tables. Cartesian
// polynomial forms keep the nodal planes exact on the grid.


export namespace ses {

// Y_lm normalization constants, hoisted out of real_spherical_harmonic:
// MSVC /fp:precise does not constant-fold sqrt(literal), so evaluating them
// inline would cost one libm call per grid point in fill_orbital.
namespace ynorm {
inline constexpr double kPi = 3.14159265358979323846;
inline const double sqrt_pi = std::sqrt(kPi);                    // l = 0
inline const double s3_4pi = std::sqrt(3.0 / (4.0 * kPi));       // l = 1
inline const double s15_pi = std::sqrt(15.0 / kPi);              // l = 2
inline const double s5_pi = std::sqrt(5.0 / kPi);
inline const double s35_2pi = std::sqrt(35.0 / (2.0 * kPi));     // l = 3
inline const double s105_pi = std::sqrt(105.0 / kPi);
inline const double s21_2pi = std::sqrt(21.0 / (2.0 * kPi));
inline const double s7_pi = std::sqrt(7.0 / kPi);
inline const double s35_pi = std::sqrt(35.0 / kPi);              // l = 4
inline const double s5_2pi = std::sqrt(5.0 / (2.0 * kPi));
inline const double s1_pi = std::sqrt(1.0 / kPi);
inline const double s1386_pi = std::sqrt(1386.0 / kPi);          // l = 5
inline const double s3465_pi = std::sqrt(3465.0 / kPi);
inline const double s770_pi = std::sqrt(770.0 / kPi);
inline const double s1155_pi = std::sqrt(1155.0 / kPi);
inline const double s165_pi = std::sqrt(165.0 / kPi);
inline const double s11_pi = std::sqrt(11.0 / kPi);
}  // namespace ynorm

// Real tesseral Y_lm at (x, y, z)/r, m = -l..l: m < 0 sin / m > 0 cos sector,
// canonical (x+iy)^|m| real/imag parts (l=1 anchor: -1 -> y, 0 -> z, +1 -> x).
inline double real_spherical_harmonic(int l, int m, double x, double y, double z) noexcept {
    const double r2 = x * x + y * y + z * z;
    if (l == 0) {
        return 1.0 / (2.0 * ynorm::sqrt_pi);
    }
    if (r2 == 0.0) {
        return 0.0;  // direction undefined; l > 0 harmonics vanish with r^l
    }
    if (l == 1) {
        const double c = ynorm::s3_4pi / std::sqrt(r2);
        switch (m) {
            case -1: return c * y;
            case 0: return c * z;
            default: return c * x;
        }
    }
    if (l == 2) {
        const double c = ynorm::s15_pi / r2;
        switch (m) {
            case -2: return 0.5 * c * x * y;
            case -1: return 0.5 * c * y * z;
            case 0: return 0.25 * ynorm::s5_pi * (3.0 * z * z - r2) / r2;
            case 1: return 0.5 * c * z * x;
            default: return 0.25 * c * (x * x - y * y);
        }
    }
    if (l == 3) {
        const double r3 = r2 * std::sqrt(r2);
        switch (m) {
            case -3: return 0.25 * ynorm::s35_2pi * y * (3.0 * x * x - y * y) / r3;
            case -2: return 0.5 * ynorm::s105_pi * x * y * z / r3;
            case -1: return 0.25 * ynorm::s21_2pi * y * (5.0 * z * z - r2) / r3;
            case 0: return 0.25 * ynorm::s7_pi * z * (5.0 * z * z - 3.0 * r2) / r3;
            case 1: return 0.25 * ynorm::s21_2pi * x * (5.0 * z * z - r2) / r3;
            case 2: return 0.25 * ynorm::s105_pi * z * (x * x - y * y) / r3;
            default: return 0.25 * ynorm::s35_2pi * x * (x * x - 3.0 * y * y) / r3;
        }
    }
    if (l == 4) {
        // l == 4; default case is m = +4.
        const double r4 = r2 * r2;
        switch (m) {
            case -4: return 0.75 * ynorm::s35_pi * x * y * (x * x - y * y) / r4;
            case -3: return 0.75 * ynorm::s35_2pi * y * z * (3.0 * x * x - y * y) / r4;
            case -2: return 0.75 * ynorm::s5_pi * x * y * (7.0 * z * z - r2) / r4;
            case -1: return 0.75 * ynorm::s5_2pi * y * z * (7.0 * z * z - 3.0 * r2) / r4;
            case 0:
                return (3.0 / 16.0) * ynorm::s1_pi *
                       (35.0 * z * z * z * z - 30.0 * z * z * r2 + 3.0 * r2 * r2) / r4;
            case 1: return 0.75 * ynorm::s5_2pi * x * z * (7.0 * z * z - 3.0 * r2) / r4;
            case 2: return 0.375 * ynorm::s5_pi * (x * x - y * y) * (7.0 * z * z - r2) / r4;
            case 3: return 0.75 * ynorm::s35_2pi * x * z * (x * x - 3.0 * y * y) / r4;
            default:
                return (3.0 / 16.0) * ynorm::s35_pi *
                       (x * x * x * x - 6.0 * x * x * y * y + y * y * y * y) / r4;
        }
    }
    // l == 5; default case is m = +5 (orthonormality pinned by tests).
    const double r5 = r2 * r2 * std::sqrt(r2);
    const double x2 = x * x;
    const double y2 = y * y;
    const double z2 = z * z;
    const double zpoly = 21.0 * z2 * z2 - 14.0 * z2 * r2 + r2 * r2;  // m = +-1
    switch (m) {
        case -5:
            return (1.0 / 32.0) * ynorm::s1386_pi *
                   y * (5.0 * x2 * x2 - 10.0 * x2 * y2 + y2 * y2) / r5;
        case -4:
            return (1.0 / 16.0) * ynorm::s3465_pi *
                   4.0 * x * y * (x2 - y2) * z / r5;
        case -3:
            return (1.0 / 32.0) * ynorm::s770_pi *
                   y * (3.0 * x2 - y2) * (9.0 * z2 - r2) / r5;
        case -2:
            return (1.0 / 8.0) * ynorm::s1155_pi *
                   2.0 * x * y * z * (3.0 * z2 - r2) / r5;
        case -1:
            return (1.0 / 16.0) * ynorm::s165_pi * y * zpoly / r5;
        case 0:
            return (1.0 / 16.0) * ynorm::s11_pi *
                   z * (63.0 * z2 * z2 - 70.0 * z2 * r2 + 15.0 * r2 * r2) / r5;
        case 1:
            return (1.0 / 16.0) * ynorm::s165_pi * x * zpoly / r5;
        case 2:
            return (1.0 / 8.0) * ynorm::s1155_pi *
                   (x2 - y2) * z * (3.0 * z2 - r2) / r5;
        case 3:
            return (1.0 / 32.0) * ynorm::s770_pi *
                   x * (x2 - 3.0 * y2) * (9.0 * z2 - r2) / r5;
        case 4:
            return (1.0 / 16.0) * ynorm::s3465_pi *
                   (x2 * x2 - 6.0 * x2 * y2 + y2 * y2) * z / r5;
        default:
            return (1.0 / 32.0) * ynorm::s1386_pi *
                   x * (x2 * x2 - 10.0 * x2 * y2 + 5.0 * y2 * y2) / r5;
    }
}

// psi = (u(r)/r) Y_lm with u linearly interpolated (u/r -> u[0]/h as r -> 0).
// fill_orbital writes the UN-normalized field; synthesize_orbital normalizes.
// CONTRACT: ses.projection's deposit shape (core/src/projection.ixx) mirrors this interpolation.
inline void fill_orbital(Field3D& psi, const Grid3D& g, const RadialGrid& rg,
                         const std::vector<double>& u, int l, int m) noexcept {
    const double h = rg.h();
    // Disjoint z-slabs via ses.parallel: bitwise-deterministic parallelism
    // (project rule). NOT OpenMP: MSVC silently miscompiles `#pragma omp`
    // inside an exported module function (zero field) -- ses.parallel exists
    // to replace it.
    parallel_for(g.z.n, [&](int k) {
        for (int j = 0; j < g.y.n; ++j) {
            for (int i = 0; i < g.x.n; ++i) {
                const double x = g.x.coord(i);
                const double y = g.y.coord(j);
                const double z = g.z.coord(k);
                const double r = std::sqrt(x * x + y * y + z * z);
                double u_over_r = 0.0;
                if (r < h) {
                    u_over_r = u[0] / h;
                } else if (r < rg.rmax) {
                    const double t = r / h - 1.0;  // r_i = (i+1) h
                    const std::size_t i0 = static_cast<std::size_t>(t);
                    const double frac = t - static_cast<double>(i0);
                    const double ui =
                        (i0 + 1 < u.size())
                            ? (1.0 - frac) * u[i0] + frac * u[i0 + 1]
                            : (1.0 - frac) * u[i0];  // outermost: u(rmax) = 0
                    u_over_r = ui / r;
                }
                const double value =
                    u_over_r * real_spherical_harmonic(l, m, x, y, z);
                psi(i, j, k) = std::complex<double>{value, 0.0};
            }
        }
    });
}

inline Field3D synthesize_orbital(const Grid3D& g, const RadialGrid& rg,
                                  const std::vector<double>& u, int l, int m) {
    Field3D psi{g};
    fill_orbital(psi, g, rg, u, l, m);
    normalize(psi);
    return psi;
}

// ---- E1 angular strengths in the tesseral basis --------------------------
// For states (u(r)/r) Y_lm the dipole integral factorizes exactly:
// <f|r_q|i> = R_radial * A_q, and |A_q|^2 = (theta recursion element)^2 *
// (phi overlap)^2 -- every factor an exact RATIONAL, so the table is
// constexpr. The channel build multiplies by the runtime radial integral;
// no 3D integral is ever needed.

namespace e1_detail {

// <Theta_{l+1,mu}|cos theta|Theta_{l,mu}>^2 between the shell pair (l, l+1).
constexpr double cos_sq(int l, int mu) noexcept {
    return static_cast<double>((l + 1) * (l + 1) - mu * mu) /
           static_cast<double>((2 * l + 1) * (2 * l + 3));
}
// <Theta_{l+1,mu+1}|sin theta|Theta_{l,mu}>^2: mu raised WITH l.
constexpr double sin_up_sq(int l, int mu) noexcept {
    return static_cast<double>((l + mu + 1) * (l + mu + 2)) /
           static_cast<double>((2 * l + 1) * (2 * l + 3));
}
// <Theta_{l-1,mu+1}|sin theta|Theta_{l,mu}>^2: mu raised, l lowered.
constexpr double sin_down_sq(int l, int mu) noexcept {
    const int num = (l - mu) * (l - mu - 1);
    return num <= 0 ? 0.0
                    : static_cast<double>(num) /
                          static_cast<double>((2 * l - 1) * (2 * l + 1));
}

// Squared phi overlap of tesseral phi factors under cos(phi) (axis 0 = x)
// or sin(phi) (axis 1 = y): the product-to-sum algebra leaves 1/2 when one
// side is the m = 0 constant and 1/4 otherwise; x preserves the cos/sin
// sector (m = 0 pairs with m' = +1), y flips it (m = 0 pairs with m' = -1).
constexpr double phi_xy_sq(int axis, int m_to, int m_from) noexcept {
    if (m_from == 0 || m_to == 0) {
        const int other = m_from == 0 ? m_to : m_from;
        if (other * other != 1) {
            return 0.0;
        }
        return (axis == 0 ? other > 0 : other < 0) ? 0.5 : 0.0;
    }
    const bool same_sector = (m_to > 0) == (m_from > 0);
    return (axis == 0 ? same_sector : !same_sector) ? 0.25 : 0.0;
}

}  // namespace e1_detail

// |<Y_{l_to,m_to}| r_q/r |Y_{l_from,m_from}>|^2 for real Y_lm; axis q:
// 0 = x, 1 = y, 2 = z. Exact E1 selection rules fall out as hard zeros:
// |dl| = 1, and z: m' == m, x/y: ||m'|-|m|| = 1 with the sector algebra
// (m' == -m is exactly forbidden). Symmetric in (to, from).
constexpr double tesseral_e1_axis_sq(int axis, int l_to, int m_to, int l_from,
                                     int m_from) noexcept {
    const int dl = l_to - l_from;
    if (dl != 1 && dl != -1) {
        return 0.0;
    }
    const int mu_to = m_to < 0 ? -m_to : m_to;
    const int mu_from = m_from < 0 ? -m_from : m_from;
    if (axis == 2) {
        return m_to == m_from
                   ? e1_detail::cos_sq(dl > 0 ? l_from : l_to, mu_from)
                   : 0.0;
    }
    const int dmu = mu_to - mu_from;
    if (dmu != 1 && dmu != -1) {
        return 0.0;
    }
    const double phi = e1_detail::phi_xy_sq(axis, m_to, m_from);
    if (phi == 0.0) {
        return 0.0;
    }
    // Theta element between (la, mua) and (lb, mua + 1), la = lower-mu side.
    const int la = dmu > 0 ? l_from : l_to;
    const int lb = dmu > 0 ? l_to : l_from;
    const int mua = dmu > 0 ? mu_from : mu_to;
    const double theta = lb == la + 1 ? e1_detail::sin_up_sq(la, mua)
                                      : e1_detail::sin_down_sq(la, mua);
    return theta * phi;
}

// Sum over the three polarizations: what Einstein A needs per channel.
constexpr double tesseral_e1_sq(int l_to, int m_to, int l_from,
                                int m_from) noexcept {
    return tesseral_e1_axis_sq(0, l_to, m_to, l_from, m_from) +
           tesseral_e1_axis_sq(1, l_to, m_to, l_from, m_from) +
           tesseral_e1_axis_sq(2, l_to, m_to, l_from, m_from);
}

// Compile-time physics anchors: every 2p -> 1s orientation carries exactly
// 1/3, and the tesseral m' == -m channel is a hard zero.
static_assert(tesseral_e1_axis_sq(2, 0, 0, 1, 0) == 1.0 / 3.0);
static_assert(tesseral_e1_axis_sq(0, 0, 0, 1, 1) == 1.0 / 3.0);
static_assert(tesseral_e1_axis_sq(1, 0, 0, 1, -1) == 1.0 / 3.0);
static_assert(tesseral_e1_sq(1, 1, 2, -1) == 0.0);

}  // namespace ses
