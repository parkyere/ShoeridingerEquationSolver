module;
#include <cstddef>
#include <algorithm>
#include <cmath>
#include <complex>
#include <memory>
#include <string>
#include <utility>
#include <vector>
export module ses.scenario.qpc2d_director;
export import ses.scenario.lattice2d_director;
import ses.heightfield;
import ses.propagator;
import ses.field;
import ses.parallel;


// Quantum point contact: one narrow constriction in a hard wall. A wide
// wave front at k0 transmits through QUANTIZED transverse modes -- a
// channel opens each time the gap grows by lambda/2 = pi/k0, so the
// transmitted flux climbs a STAIRCASE in the gap width (conductance
// quantization, the mesoscopic classic). Spectral split-operator on the
// 512^2 plane; edge CAPs (right-cap tally = transmission, the Anderson
// wire's metric). CONTRACT: tests/qpc2d_test.cpp + --selftest-qpc2d.


export namespace ses_shell {

// Pierced-wall potential: V0 inside the slab [wall_lo, wall_hi] except
// the gap |y| < w/2, with a smooth half-cosine lip of width `lip` on
// both gap edges (raw corners diffract Gibbs fringes).
inline std::vector<double> qpc_potential(const ses::Grid3D& g, double w,
                                         double wall_lo, double wall_hi,
                                         double v0, double lip) {
    return std::vector<double>(static_cast<std::size_t>(g.size()), 0.0);
    // RED stub
}

}  // namespace ses_shell
