module;
#include <cassert>
#include <cmath>
#include <complex>
#include <cstddef>
#include <vector>
export module ses.lattice2d;
export import ses.field;
export import ses.grid;
import ses.parallel;


// 2D lattice (finite-difference) propagator with Peierls link phases: the
// honest engine for a LOCALIZED magnetic flux (the double-slit solenoid).
// The FFT split-operator cannot Trotterize (p - A)^2/2 for a flux line --
// in every gauge A couples both coordinates, so no factor is diagonal in
// any FFT basis, and approximate A.p splittings drift the norm. On the
// lattice the flux enters EXACTLY:
//
//   H = sum_bonds -t (e^{i theta_ab} |a><b| + h.c.) + diag(V + 2tx + 2ty)
//
// with t = 1/(2 h^2) per axis (the FD Laplacian; onsite 2t per axis kept
// so energies are honest). Kinetic bonds split into 4 disjoint groups
// (x-even, x-odd, y-even, y-odd); each group is a direct sum of 2-site
// bonds whose exponential is an EXACT 2x2 rotation:
//
//   exp(-i dt H_bond):  a' = c a + i s e^{i theta} b
//                       b' = i s e^{-i theta} a + c b
//   c = cos(t dt), s = sin(t dt)
//
// -- unitary to round-off by construction; the only approximation is the
// (symmetric Strang) ordering, O(dt^2). Dispersion is the DISCRETE band
// E(k) = (1 - cos k h)/h^2 per axis, group velocity sin(k h)/h.
//
// The solenoid (string gauge): the cut runs from the solenoid cell
// straight up (+y, or down) to the boundary; the x-links crossing it
// carry e^{-+i Phi}. The phase-product around every elementary plaquette
// is then 1 EXCEPT the solenoid's own plaquette, which carries Phi:
// B = 0 everywhere the electron goes, the flux is pure topology.
// Boundaries are OPEN (no wrap bonds): a hard box; scenes add absorbers.


export namespace ses {

class PeierlsLattice2D {
public:
    PeierlsLattice2D(const Grid3D& g, const std::vector<double>& potential,
                     double dt)
        : g_(&g), nx_(g.x.n), ny_(g.y.n), dt_(dt), v_(potential) {
        assert(g.z.n == 1);
        assert(static_cast<int>(potential.size()) == g.size());
        tx_ = 0.5 / (g.x.spacing() * g.x.spacing());
        ty_ = 0.5 / (g.y.spacing() * g.y.spacing());
        // Exact 2x2 bond rotations, half-dt for the palindromic ordering
        // (the y-odd group sits in the middle at full dt).
        cx_ = std::cos(tx_ * 0.5 * dt);
        sx_ = std::sin(tx_ * 0.5 * dt);
        cy_ = std::cos(ty_ * 0.5 * dt);
        sy_ = std::sin(ty_ * 0.5 * dt);
        cy2_ = std::cos(ty_ * dt);
        sy2_ = std::sin(ty_ * dt);
        // Imaginary-time (relax) twins: cosh/sinh mixing, real decay.
        rcx_ = std::cosh(tx_ * 0.5 * dt);
        rsx_ = std::sinh(tx_ * 0.5 * dt);
        rcy_ = std::cosh(ty_ * 0.5 * dt);
        rsy_ = std::sinh(ty_ * 0.5 * dt);
        rcy2_ = std::cosh(ty_ * dt);
        rsy2_ = std::sinh(ty_ * dt);
        half_v_.resize(potential.size());
        relax_half_v_.resize(potential.size());
        for (std::size_t i = 0; i < potential.size(); ++i) {
            const double e0 = potential[i] + 2.0 * tx_ + 2.0 * ty_;
            const double th = -0.5 * e0 * dt;
            half_v_[i] = std::complex<double>{std::cos(th), std::sin(th)};
            relax_half_v_[i] = std::exp(-0.5 * e0 * dt);
        }
        link_x_.assign(static_cast<std::size_t>(nx_ * ny_), 1.0);
        link_y_.assign(static_cast<std::size_t>(nx_ * ny_), 1.0);
    }

    double dt() const noexcept { return dt_; }

    // Thread flux phi through the plaquette containing (xs, ys). String
    // gauge: with cut_up the x-links of the straddling column ABOVE the
    // solenoid carry e^{-i phi}; with cut_down those below carry e^{+i
    // phi} -- same physics (a gauge transformation), tested as such.
    void set_solenoid(double phi, double xs, double ys, bool cut_up = true) {
        link_x_.assign(link_x_.size(), 1.0);
        int is = -1;
        for (int i = 0; i + 1 < nx_; ++i) {
            if (g_->x.coord(i) <= xs && xs < g_->x.coord(i + 1)) {
                is = i;
            }
        }
        int js = -1;
        for (int j = 0; j + 1 < ny_; ++j) {
            if (g_->y.coord(j) <= ys && ys < g_->y.coord(j + 1)) {
                js = j;
            }
        }
        if (is < 0 || js < 0) {
            return;  // solenoid outside the lattice: no flux anywhere
        }
        const std::complex<double> up{std::cos(phi), -std::sin(phi)};
        const std::complex<double> dn{std::cos(phi), std::sin(phi)};
        if (cut_up) {
            for (int j = js + 1; j < ny_; ++j) {
                link_x_[static_cast<std::size_t>(j * nx_ + is)] = up;
            }
        } else {
            for (int j = 0; j <= js; ++j) {
                link_x_[static_cast<std::size_t>(j * nx_ + is)] = dn;
            }
        }
    }

    // UNIFORM field B along z (Landau levels / cyclotron motion), Landau
    // gauge ANCHORED at y = 0 (the box center): the x-links of row j
    // carry e^{-i B hx y_j}. By the plane-wave band E ~ (k + theta/h)^2/2
    // this is the Peierls form of A_x = +B y, A_y = 0: every plaquette
    // holds the same flux B hx hy, mechanical = canonical momentum on the
    // y = 0 row, and v rotates COUNTERCLOCKWISE at omega_c = B (anchoring
    // at ymin instead would hand the packet a spurious v_x = -B*ymin).
    // Replaces any solenoid.
    void set_uniform_field(double b) {
        const double bh = b * g_->x.spacing();
        for (int j = 0; j < ny_; ++j) {
            const double th = -bh * g_->y.coord(j);
            const std::complex<double> u{std::cos(th), std::sin(th)};
            for (int i = 0; i < nx_; ++i) {
                link_x_[static_cast<std::size_t>(j * nx_ + i)] = u;
            }
        }
    }

    // Directed link variables (U on edge (i,j)->(i+1,j) resp. (i,j)->
    // (i,j+1)) for the plaquette-topology contract.
    std::complex<double> link_x(int i, int j) const {
        return link_x_[static_cast<std::size_t>(j * nx_ + i)];
    }
    std::complex<double> link_y(int i, int j) const {
        return link_y_[static_cast<std::size_t>(j * nx_ + i)];
    }

    void step(Field3D& psi, int nsteps = 1) const;

    // Imaginary-time relaxation toward the ground state: the SAME bond
    // splitting with cosh/sinh mixing (e^{-tau H_bond} since the bond
    // Hamiltonian squares to t^2 I) and a real onsite decay, renormalized
    // every step. The LINK PHASES ride along, so this can relax the
    // ground of a dot IN a magnetic field (the Fock-Darwin ground) --
    // out of reach for any B = 0 imaginary-time machinery.
    void relax(Field3D& psi, int nsteps = 1) const;

    // <H> = hops + onsite (V + 2tx + 2ty), normalized -- the live energy
    // readout (and the relax convergence check).
    double energy(const Field3D& psi) const;

private:
    void phase(std::vector<std::complex<double>>& a,
               const std::vector<std::complex<double>>& table) const;

    // One x-bond group (bonds (i, i+1) with i of the given parity), all
    // rows: rows are independent, bonds within a row disjoint by parity.
    // mix = i*sin for real time, sinh (real) for imaginary time.
    void sweep_x(std::vector<std::complex<double>>& a, int parity, double c,
                 std::complex<double> mix) const;

    // One y-bond group (bonds (j, j+1) with j of the given parity):
    // bond-rows are disjoint by parity, x runs contiguous inside.
    void sweep_y(std::vector<std::complex<double>>& a, int parity, double c,
                 std::complex<double> mix) const;

    const Grid3D* g_;
    int nx_;
    int ny_;
    double dt_;
    double tx_ = 0.0;
    double ty_ = 0.0;
    double cx_, sx_;    // x bonds, dt/2
    double cy_, sy_;    // y-even bonds, dt/2
    double cy2_, sy2_;  // y-odd bonds, full dt (palindrome center)
    double rcx_, rsx_;  // imaginary-time twins (cosh/sinh)
    double rcy_, rsy_;
    double rcy2_, rsy2_;
    std::vector<double> v_;
    std::vector<std::complex<double>> half_v_;
    std::vector<std::complex<double>> relax_half_v_;
    std::vector<std::complex<double>> link_x_;
    std::vector<std::complex<double>> link_y_;
};


// Out-of-class definitions ON PURPOSE: members defined outside the class
// in a module interface are NOT implicitly inline, so they are compiled
// exactly once, in THIS module's TU. Defined in-class they were compiled
// per importer, and one importing scene TU's instantiation of the
// parallel_for bodies crashed a pool worker at runtime (MSVC modules
// codegen roulette; kin of the OpenMP-in-module-interface miscompile
// that begat ses.parallel). One canonical copy, no roulette.

void PeierlsLattice2D::phase(
    std::vector<std::complex<double>>& a,
    const std::vector<std::complex<double>>& table) const {
    parallel_for(static_cast<int>(a.size()), [&](int i) {
        a[static_cast<std::size_t>(i)] *= table[static_cast<std::size_t>(i)];
    });
}

void PeierlsLattice2D::sweep_x(std::vector<std::complex<double>>& a,
                               int parity, double c,
                               std::complex<double> mix) const {
    parallel_for(ny_, [&](int j) {
        const std::size_t row = static_cast<std::size_t>(j * nx_);
        for (int i = parity; i + 1 < nx_; i += 2) {
            const std::complex<double> u =
                link_x_[row + static_cast<std::size_t>(i)];
            const std::complex<double> pa =
                a[row + static_cast<std::size_t>(i)];
            const std::complex<double> pb =
                a[row + static_cast<std::size_t>(i + 1)];
            a[row + static_cast<std::size_t>(i)] = c * pa + mix * u * pb;
            a[row + static_cast<std::size_t>(i + 1)] =
                mix * std::conj(u) * pa + c * pb;
        }
    });
}

void PeierlsLattice2D::sweep_y(std::vector<std::complex<double>>& a,
                               int parity, double c,
                               std::complex<double> mix) const {
    const int rows = (ny_ - 1 - parity + 1) / 2;  // j = parity, +2, ...
    parallel_for(rows, [&](int r) {
        const int j = parity + 2 * r;
        if (j + 1 >= ny_) {
            return;
        }
        const std::size_t lo = static_cast<std::size_t>(j * nx_);
        const std::size_t hi = static_cast<std::size_t>((j + 1) * nx_);
        for (int i = 0; i < nx_; ++i) {
            const std::complex<double> u =
                link_y_[lo + static_cast<std::size_t>(i)];
            const std::complex<double> pa =
                a[lo + static_cast<std::size_t>(i)];
            const std::complex<double> pb =
                a[hi + static_cast<std::size_t>(i)];
            a[lo + static_cast<std::size_t>(i)] = c * pa + mix * u * pb;
            a[hi + static_cast<std::size_t>(i)] =
                mix * std::conj(u) * pa + c * pb;
        }
    });
}

void PeierlsLattice2D::step(Field3D& psi, int nsteps) const {
    assert(psi.data().size() == half_v_.size());
    std::vector<std::complex<double>>& a = psi.data();
    for (int s = 0; s < nsteps; ++s) {
        phase(a, half_v_);
        sweep_x(a, 0, cx_, {0.0, sx_});
        sweep_x(a, 1, cx_, {0.0, sx_});
        sweep_y(a, 0, cy_, {0.0, sy_});
        sweep_y(a, 1, cy2_, {0.0, sy2_});
        sweep_y(a, 0, cy_, {0.0, sy_});
        sweep_x(a, 1, cx_, {0.0, sx_});
        sweep_x(a, 0, cx_, {0.0, sx_});
        phase(a, half_v_);
    }
}

void PeierlsLattice2D::relax(Field3D& psi, int nsteps) const {
    assert(psi.data().size() == half_v_.size());
    std::vector<std::complex<double>>& a = psi.data();
    for (int s = 0; s < nsteps; ++s) {
        phase(a, relax_half_v_);
        sweep_x(a, 0, rcx_, {rsx_, 0.0});
        sweep_x(a, 1, rcx_, {rsx_, 0.0});
        sweep_y(a, 0, rcy_, {rsy_, 0.0});
        sweep_y(a, 1, rcy2_, {rsy2_, 0.0});
        sweep_y(a, 0, rcy_, {rsy_, 0.0});
        sweep_x(a, 1, rcx_, {rsx_, 0.0});
        sweep_x(a, 0, rcx_, {rsx_, 0.0});
        phase(a, relax_half_v_);
        normalize(psi);
    }
}

double PeierlsLattice2D::energy(const Field3D& psi) const {
    double e = 0.0;
    double den = 0.0;
    for (int j = 0; j < ny_; ++j) {
        for (int i = 0; i < nx_; ++i) {
            const std::complex<double> z = psi(i, j, 0);
            const double w = std::norm(z);
            den += w;
            e += (v_[static_cast<std::size_t>(g_->flat(i, j, 0))] +
                  2.0 * tx_ + 2.0 * ty_) *
                 w;
            if (i + 1 < nx_) {
                e += -tx_ * 2.0 *
                     (std::conj(z) * link_x(i, j) * psi(i + 1, j, 0)).real();
            }
            if (j + 1 < ny_) {
                e += -ty_ * 2.0 *
                     (std::conj(z) * link_y(i, j) * psi(i, j + 1, 0)).real();
            }
        }
    }
    return e / den;
}

// Landau-level ladder operator, in the SAME Landau gauge as
// set_uniform_field (A = (-B y, 0), anchored at y = 0):
//     a(-)    = (pi_x - i pi_y) / sqrt(2 B)   (lowers; annihilates the LLL)
//     a(-dag) = (pi_x + i pi_y) / sqrt(2 B)   (raises <H> by omega_c = B)
// with pi = -i grad - A discretized by central differences (periodic wrap;
// the gauge is not wrap-consistent, so states must stay off the boundary --
// magnetic length 1/sqrt(B) against the box, as the scene guarantees).
// Result is UNNORMALIZED (a|n> = sqrt(n)|n-1>); callers renormalize.
// CONTRACT: tests/lattice2d_test.cpp LandauLadderClimbsOneCyclotronQuantum.
inline Field3D landau_ladder(const Field3D& psi, double b, bool up) {
    const Grid3D& g = psi.grid();
    const double h = g.x.spacing();
    const double inv2h = 1.0 / (2.0 * h);
    const double s = 1.0 / std::sqrt(2.0 * b);
    Field3D out{g};
    parallel_for(g.y.n, [&](int j) {
        const int ny = g.y.n;
        const int nx = g.x.n;
        const double y = g.y.coord(j);
        const int jp = (j + 1) % ny;
        const int jm = (j - 1 + ny) % ny;
        const std::complex<double> kI{0.0, 1.0};
        for (int i = 0; i < nx; ++i) {
            const int ip = (i + 1) % nx;
            const int im = (i - 1 + nx) % nx;
            const std::complex<double> ddx =
                (psi(ip, j, 0) - psi(im, j, 0)) * inv2h;
            const std::complex<double> ddy =
                (psi(i, jp, 0) - psi(i, jm, 0)) * inv2h;
            // pi_x = -i d_x - B y, pi_y = -i d_y: the lattice link
            // theta = -B h y corresponds to A_x = +B y (empirically pinned
            // by the contract -- the +B y sign made the pair the intra-level
            // guiding-center operators, <H> unchanged).
            const std::complex<double> pix = -kI * ddx - b * y * psi(i, j, 0);
            const std::complex<double> piy = -kI * ddy;
            out(i, j, 0) = s * (up ? (pix + kI * piy) : (pix - kI * piy));
        }
    });
    return out;
}

// 2D isotropic-HO CIRCULAR ladder at B = 0: a_R = (a_x - i a_y)/sqrt(2)
// (right-circular quantum: a_R-dag adds exactly omega to <H> and +1 to
// <L_z>). Central differences; UNNORMALIZED like landau_ladder.
// CONTRACT: tests/ho2d_test.cpp.
inline Field3D ho2d_ladder(const Field3D& psi, double omega, bool up) {
    const Grid3D& g = psi.grid();
    const double inv2h = 1.0 / (2.0 * g.x.spacing());
    const double cx = std::sqrt(omega / 2.0);   // sqrt(w/2) x term
    const double cd = 1.0 / std::sqrt(2.0 * omega);  // d/dx term
    const double s = 1.0 / std::sqrt(2.0);      // circular mix
    Field3D out{g};
    parallel_for(g.y.n, [&](int j) {
        const int ny = g.y.n;
        const int nx = g.x.n;
        const double y = g.y.coord(j);
        const int jp = (j + 1) % ny;
        const int jm = (j - 1 + ny) % ny;
        const std::complex<double> kI{0.0, 1.0};
        for (int i = 0; i < nx; ++i) {
            const int ip = (i + 1) % nx;
            const int im = (i - 1 + nx) % nx;
            const double x = g.x.coord(i);
            const std::complex<double> ddx =
                (psi(ip, j, 0) - psi(im, j, 0)) * inv2h;
            const std::complex<double> ddy =
                (psi(i, jp, 0) - psi(i, jm, 0)) * inv2h;
            // a_x = sqrt(w/2) x + d_x / sqrt(2w) (and the dagger flips
            // the derivative sign); a_R = (a_x - i a_y)/sqrt(2).
            out(i, j, 0) =
                up ? s * (cx * (x + kI * y) * psi(i, j, 0) -
                          cd * (ddx + kI * ddy))
                   : s * (cx * (x - kI * y) * psi(i, j, 0) +
                          cd * (ddx - kI * ddy));
        }
    });
    return out;
}

}  // namespace ses

