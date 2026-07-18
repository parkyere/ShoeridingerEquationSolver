module;
#include <complex>
#include <cmath>
#include <numbers>
export module ses.wavepacket;
export import ses.field;
export import ses.grid;
export import ses.vec;


// Gaussian wavepacket factory (atomic units): psi = (2 pi s^2)^(-1/4) exp(-(x-x0)^2/(4 s^2)) exp(i k0 x).
// Continuum amplitude is unit-norm; final discrete normalize absorbs grid sampling error.


export namespace ses {

inline Field1D gaussian_wavepacket(const Grid1D& g, double x0, double sigma, double k0) {
    Field1D psi{g};
    const double amp = std::pow(2.0 * std::numbers::pi * sigma * sigma, -0.25);
    for (int i = 0; i < psi.size(); ++i) {
        const double x = g.coord(i);
        const double envelope = amp * std::exp(-(x - x0) * (x - x0) / (4.0 * sigma * sigma));
        psi[i] = std::complex<double>{envelope * std::cos(k0 * x), envelope * std::sin(k0 * x)};
    }
    normalize(psi);
    return psi;
}

// 3D packet: product of three 1D packets with per-axis width and momentum.
inline Field3D gaussian_wavepacket(const Grid3D& g, Vec3d r0, Vec3d sigma, Vec3d k0) {
    Field3D psi{g};
    auto envelope = [](double u, double u0, double s) {
        return std::pow(2.0 * std::numbers::pi * s * s, -0.25) *
               std::exp(-(u - u0) * (u - u0) / (4.0 * s * s));
    };
    for (int k = 0; k < g.z.n; ++k) {
        for (int j = 0; j < g.y.n; ++j) {
            for (int i = 0; i < g.x.n; ++i) {
                const double x = g.x.coord(i);
                const double y = g.y.coord(j);
                const double z = g.z.coord(k);
                const double env = envelope(x, r0.x, sigma.x) * envelope(y, r0.y, sigma.y) *
                                   envelope(z, r0.z, sigma.z);
                const double phase = k0.x * x + k0.y * y + k0.z * z;
                psi(i, j, k) = std::complex<double>{env * std::cos(phase), env * std::sin(phase)};
            }
        }
    }
    normalize(psi);
    return psi;
}

}  // namespace ses
