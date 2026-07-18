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
        : g_(&g), nx_(g.x.n), ny_(g.y.n), dt_(dt) {
        assert(g.z.n == 1);
        assert(static_cast<int>(potential.size()) == g.size());
        const double tx = 0.5 / (g.x.spacing() * g.x.spacing());
        const double ty = 0.5 / (g.y.spacing() * g.y.spacing());
        // Exact 2x2 bond rotations, half-dt for the palindromic ordering
        // (the y-odd group sits in the middle at full dt).
        cx_ = std::cos(tx * 0.5 * dt);
        sx_ = std::sin(tx * 0.5 * dt);
        cy_ = std::cos(ty * 0.5 * dt);
        sy_ = std::sin(ty * 0.5 * dt);
        cy2_ = std::cos(ty * dt);
        sy2_ = std::sin(ty * dt);
        half_v_.resize(potential.size());
        for (std::size_t i = 0; i < potential.size(); ++i) {
            const double th =
                -0.5 * (potential[i] + 2.0 * tx + 2.0 * ty) * dt;
            half_v_[i] = std::complex<double>{std::cos(th), std::sin(th)};
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

    void step(Field3D& psi, int nsteps = 1) const {
        assert(psi.data().size() == half_v_.size());
        std::vector<std::complex<double>>& a = psi.data();
        for (int s = 0; s < nsteps; ++s) {
            phase(a);
            sweep_x(a, 0, cx_, sx_);
            sweep_x(a, 1, cx_, sx_);
            sweep_y(a, 0, cy_, sy_);
            sweep_y(a, 1, cy2_, sy2_);
            sweep_y(a, 0, cy_, sy_);
            sweep_x(a, 1, cx_, sx_);
            sweep_x(a, 0, cx_, sx_);
            phase(a);
        }
    }

private:
    void phase(std::vector<std::complex<double>>& a) const {
        parallel_for(static_cast<int>(a.size()), [&](int i) {
            a[static_cast<std::size_t>(i)] *=
                half_v_[static_cast<std::size_t>(i)];
        });
    }

    // One x-bond group (bonds (i, i+1) with i of the given parity), all
    // rows: rows are independent, bonds within a row disjoint by parity.
    void sweep_x(std::vector<std::complex<double>>& a, int parity, double c,
                 double s) const {
        parallel_for(ny_, [&](int j) {
            const std::size_t row = static_cast<std::size_t>(j * nx_);
            for (int i = parity; i + 1 < nx_; i += 2) {
                const std::complex<double> u =
                    link_x_[row + static_cast<std::size_t>(i)];
                const std::complex<double> pa =
                    a[row + static_cast<std::size_t>(i)];
                const std::complex<double> pb =
                    a[row + static_cast<std::size_t>(i + 1)];
                const std::complex<double> is{0.0, s};
                a[row + static_cast<std::size_t>(i)] = c * pa + is * u * pb;
                a[row + static_cast<std::size_t>(i + 1)] =
                    is * std::conj(u) * pa + c * pb;
            }
        });
    }

    // One y-bond group (bonds (j, j+1) with j of the given parity):
    // bond-rows are disjoint by parity, x runs contiguous inside.
    void sweep_y(std::vector<std::complex<double>>& a, int parity, double c,
                 double s) const {
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
                const std::complex<double> is{0.0, s};
                a[lo + static_cast<std::size_t>(i)] = c * pa + is * u * pb;
                a[hi + static_cast<std::size_t>(i)] =
                    is * std::conj(u) * pa + c * pb;
            }
        });
    }

    const Grid3D* g_;
    int nx_;
    int ny_;
    double dt_;
    double cx_, sx_;    // x bonds, dt/2
    double cy_, sy_;    // y-even bonds, dt/2
    double cy2_, sy2_;  // y-odd bonds, full dt (palindrome center)
    std::vector<std::complex<double>> half_v_;
    std::vector<std::complex<double>> link_x_;
    std::vector<std::complex<double>> link_y_;
};

}  // namespace ses
