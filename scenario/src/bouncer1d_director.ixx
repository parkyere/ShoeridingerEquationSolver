module;
#include <cstddef>
#include <algorithm>
#include <cmath>
#include <complex>
#include <memory>
#include <string>
#include <utility>
#include <vector>
export module ses.scenario.bouncer1d_director;
export import ses.scenario.line1d_director;
import ses.imaginary_time;
import ses.observables;
import ses.wavepacket;


// Quantum bouncer: a particle on a hard floor under gravity, V = g z
// (z >= 0) with a steep linear wall below -- the GRANIT experiment's
// geometry (ultracold neutrons bounce on a mirror; the bound states are
// Airy functions, E_n = a_n (g^2/2)^{1/3} with -a_n the Airy zeros).
// [2] relaxes the Airy ground in-place (a 1D ITP is instant); [F] drops
// a packet from height and it bounces, dephases, and revives.
// CONTRACT: tests/bouncer1d_test.cpp + --selftest-bouncer.


export namespace ses_shell {

// Airy zeros a_1, a_2 (Ai(-a_n) = 0).
inline constexpr double kAiryZero1 = 2.33810741045977;
inline constexpr double kAiryZero2 = 4.08794944413097;

// E_n = a_n (g^2 / 2)^{1/3} for the ideal hard floor (m = hbar = 1).
inline double bouncer_energy(double /*g*/, double /*a_n*/) {
    return 0.0;  // RED stub
}

// V = g z above the floor, steep linear wall (slope `wall`) below --
// continuous at z = 0, no Gibbs step.
inline std::vector<double> bouncer_potential(const ses::Grid1D& g,
                                             double grav, double wall) {
    return std::vector<double>(static_cast<std::size_t>(g.n), 0.0);  // RED
}

}  // namespace ses_shell
