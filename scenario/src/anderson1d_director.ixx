module;
#include <cstddef>
#include <algorithm>
#include <cmath>
#include <complex>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>
export module ses.scenario.anderson1d_director;
export import ses.scenario.line1d_director;
import ses.wavepacket;
import ses.observables;


// Anderson localization in 1D: a packet with energy ABOVE every barrier
// (a classical particle sails through) is STOPPED by coherent multiple
// scattering off a random landscape -- in 1D every eigenstate is
// exponentially localized for any disorder. Sharp scatterers on purpose
// (sigma < lambda): smooth bumps are transparent at k (the corral fence
// lesson), sharp ones backscatter. Clean (W = 0) contrast: ballistic
// flight. CONTRACT: tests/anderson1d_test.cpp.


export namespace ses_shell {

constexpr double kAn1dSpacing = 2.0;    // scatterer spacing (Bohr)
constexpr double kAn1dBumpSigma = 0.35; // SHARP: k sigma ~ 0.5 at k = 1.5
constexpr double kAn1dK0 = 1.5;         // E = 1.125 Ha > every |bump|
constexpr double kAn1dW = 1.0;          // amplitude range [-W, W]

// Random landscape: Gaussian bumps at every lattice site, amplitudes
// uniform in [-w, w] from the SEEDED mt19937 (deterministic per seed).
inline std::vector<double> anderson_potential(const ses::Grid1D& g, double w,
                                              unsigned seed) {
    return std::vector<double>(static_cast<std::size_t>(g.n), 0.0);  // RED
}

}  // namespace ses_shell
