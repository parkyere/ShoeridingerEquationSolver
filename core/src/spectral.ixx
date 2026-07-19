module;
#include <numbers>
#include <vector>
export module ses.spectral;
import ses.grid;


// FFT-bin -> physical-wavenumber mapping for a periodic Grid1D (fftfreq layout):
//     k_j = 2 pi j / L        j = 0 .. n/2 - 1
//     k_j = 2 pi (j - n) / L  j = n/2 .. n - 1


export namespace ses {

inline std::vector<double> wavenumbers(const Grid1D& g) {
    const double dk = 2.0 * std::numbers::pi / (g.xmax - g.xmin);
    std::vector<double> k(static_cast<std::size_t>(g.n));
    for (int j = 0; j < g.n; ++j) {
        // 2j < n (not j < n/2): keeps DC at 0 for n = 1, and matches the
        // fftfreq positive/negative split for every n (identical for even n).
        const int shifted = (2 * j < g.n) ? j : j - g.n;
        k[static_cast<std::size_t>(j)] = dk * shifted;
    }
    return k;
}

}  // namespace ses
