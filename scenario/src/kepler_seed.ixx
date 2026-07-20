module;
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
export module ses.scenario.kepler_seed;
export import ses.scenario.manifold_spec;
import ses.measurement;


// Rydberg (Kepler) wave-packet seed: Gaussian-in-n weights over the
// CIRCULAR states |n, l = n-1, m = +l>, expressed as coefficients on the
// tracked real-Y_lm manifold via the pinned cos/sin pairing
// |l, +l> = (|cos> + i |sin>)/sqrt(2). Superposed, adjacent shells beat at
// E_{n+1} - E_n ~ 1/n^3: the packet ORBITS the nucleus counterclockwise
// at the classical Kepler frequency (correspondence principle).
// CONTRACT: tests/kepler_test.cpp (pair table, purity, orbit rate).


export namespace ses_shell {

// Scene defaults: n_bar mid-manifold (r ~ n^2 ~ 20 a0 fits the +-80 box);
// sigma trades angular localization against the low-n fast-rate skew.
inline constexpr double kKeplerNBar = 4.5;
inline constexpr double kKeplerSigmaN = 1.0;

struct KeplerPair {
    int n;        // principal quantum number (l = m = n - 1)
    int idx_cos;  // kStateSpec entry with m = +l
    int idx_sin;  // its sin partner, m = -l
};

// The five circular-state pairs of the n <= 6 manifold (n = 2..6),
// scanned off kStateSpec (no hardcoded indices to rot).
inline std::array<KeplerPair, 5> kepler_pairs() {
    std::array<KeplerPair, 5> pairs{};
    std::size_t out = 0;
    for (int i = 0; i < kNumStates && out < pairs.size(); ++i) {
        const StateSpec& sc = kStateSpec[i];
        if (sc.l < 1 || sc.m != sc.l || state_n(i) != sc.l + 1) {
            continue;  // circular cos entry: l = n - 1, m = +l
        }
        int sin_idx = -1;
        for (int j = 0; j < kNumStates; ++j) {
            if (kStateSpec[j].level == sc.level && kStateSpec[j].m == -sc.l) {
                sin_idx = j;
                break;
            }
        }
        pairs[out++] = KeplerPair{sc.l + 1, i, sin_idx};
    }
    return pairs;
}

// Normalized manifold coefficients of the packet: weight
// w_n = exp(-(n - n_bar)^2 / (4 sigma^2)) on each circular |n, m = +l>,
// laid onto the real pair as cos = w/sqrt(2), sin = i w/sqrt(2)
// (ses.measurement pair_from_signed_m, sign +1).
inline std::array<std::complex<double>, kNumStates> kepler_coefficients(
    double n_bar, double sigma) {
    std::array<std::complex<double>, kNumStates> c{};
    double sum = 0.0;
    for (const KeplerPair& p : kepler_pairs()) {
        const double dn = p.n - n_bar;
        const double w = std::exp(-dn * dn / (4.0 * sigma * sigma));
        const ses::RealPair rp = ses::pair_from_signed_m(w, +1);
        c[static_cast<std::size_t>(p.idx_cos)] = rp.c_cos;
        c[static_cast<std::size_t>(p.idx_sin)] = rp.c_sin;
        sum += w * w;
    }
    if (sum > 0.0) {
        const double inv = 1.0 / std::sqrt(sum);
        for (auto& z : c) {
            z *= inv;
        }
    }
    return c;
}

}  // namespace ses_shell
