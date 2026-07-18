module;
#include <cstddef>
#include <vector>
export module ses.spectrum1d;
export import ses.field;
export import ses.grid;
import ses.radial;


// Bound states of an ARBITRARY 1D potential on the scene grid: the 1D box
// [xmin, xmax] with Dirichlet ends IS a radial problem at l = 0, so this is
// a thin adapter over the radial engine's tridiagonal FD solver (Sturm
// bisection + shifted inverse iteration -- the same machinery that builds
// the hydrogen manifold). The mapping is exact: with m = g.n - 1 interior
// points the radial spacing L/(m+1) equals the scene spacing, and interior
// point j sits on scene point j+1; scene point 0 (= the periodic wrap
// point) carries the Dirichlet zero. For states confined well inside the
// box the Dirichlet-vs-periodic difference is exponentially negligible.
//
// This is the shared oracle of the solvable-well scenes: the double well's
// splitting doublet, the Morse ladder, the Poschl-Teller bound set --
// verified in tests against the HO ladder and the closed-form Morse
// spectrum. Eigenfunctions are real (stored in the complex Field1D for
// direct use as scene states), discretely normalized, with the radial
// engine's sign convention (positive near xmin).


export namespace ses {

struct Bound1D {
    double energy{};
    Field1D psi;

    explicit Bound1D(const Grid1D& g) : psi(g) {}
};

inline std::vector<Bound1D> bound_states_1d(const Grid1D& g,
                                            const std::vector<double>& v,
                                            int count) {
    const RadialGrid rg{g.xmax - g.xmin, g.n - 1};
    std::vector<double> vr(static_cast<std::size_t>(g.n - 1));
    for (int j = 0; j + 1 < g.n; ++j) {
        vr[static_cast<std::size_t>(j)] = v[static_cast<std::size_t>(j + 1)];
    }
    const RadialHamiltonian ham = radial_hamiltonian(rg, vr, 0);
    std::vector<Bound1D> out;
    out.reserve(static_cast<std::size_t>(count));
    for (int k = 0; k < count; ++k) {
        const RadialState s = radial_eigenstate(rg, ham, k);
        Bound1D b{g};
        b.energy = s.energy;
        b.psi[0] = 0.0;  // the Dirichlet end (also the periodic wrap point)
        for (int j = 0; j + 1 < g.n; ++j) {
            b.psi[j + 1] = s.u[static_cast<std::size_t>(j)];
        }
        out.push_back(std::move(b));
    }
    return out;
}

}  // namespace ses
