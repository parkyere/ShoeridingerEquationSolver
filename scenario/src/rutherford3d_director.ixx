module;
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
export module ses.scenario.rutherford3d_director;
export import ses.scenario.base_director;
import ses.potential;


// Rutherford scattering: a Gaussian packet fired head-on (+x) at a REPULSIVE
// Coulomb center -- the 3D Coulomb generator (ses.potential) with a FLIPPED
// sign (charge -2Z -> V = +2Z/r). The packet decelerates, turns around near
// the classical closest approach r_min = 2Z/E, and backscatters off the
// barrier that revealed the nucleus. Sliders set the incident kinetic energy
// E and the target charge Z; defaults are Rutherford's: gold Z=79 vs the
// helium/alpha projectile (charge 2). A boundary absorber swallows the
// scattered flux.
//
// SCALING (the physics-correct way): the real alpha (mass ~7300, ~5 MeV, fm
// wavelength) is effectively CLASSICAL and unresolvable on any affordable
// grid, so this is a UNIT-MASS analog. The scattering is governed by the
// Sommerfeld parameter eta = Z_eff / v (Z_eff = 2Z) -- it depends on the
// charge and the VELOCITY, NOT on mass and energy separately. So the repulsion
// Z_eff stays PHYSICAL (158 = Z_Au*Z_He); what compensates the mass scaling is
// the ENERGY, chosen so the velocity matches the alpha's. A 5 MeV alpha has
// v = sqrt(2E/m) ~ 7.1 a.u.; the default E = 25 Ha at m=1 gives v = sqrt(50) =
// 7.07 a.u. -> eta = 158/7.07 = 22.3, matching the real gold-alpha experiment.
// (Scaling Z instead would BREAK eta.) Only the de Broglie wavelength differs
// (this packet is more wave-like) -- honest for a quantum simulator.
// The regularized center cell is a finite hard core; the packet turns at
// r_min = Z_eff/E >> a cell, so it never samples the core spike.


export namespace ses_shell {

constexpr double kRu3dBox = 40.0;       // Bohr half-extent (256^3, h ~ 0.31)
constexpr int kRu3dPoints = 256;
constexpr double kRu3dZDefault = 79.0;  // gold nucleus
constexpr double kRu3dZProj = 2.0;      // helium/alpha projectile charge
constexpr double kRu3dEDefault = 25.0;  // incident KE (Ha); r_min = 2Z/E ~ 6.3
constexpr double kRu3dEMin = 5.0;
constexpr double kRu3dEMax = 80.0;
constexpr double kRu3dZMin = 5.0;
constexpr double kRu3dZMax = 100.0;
constexpr double kRu3dLaunchX = -30.0;
constexpr double kRu3dSigma = 4.0;
constexpr double kRu3dDt = 0.01;        // small: V*dt/2 moderate at the turn

class Rutherford3DDirector final : public BaseDirector, public RutherfordApi {
public:
    Rutherford3DDirector() : BaseDirector(make(kRu3dEDefault, kRu3dZDefault)) {}

    RutherfordApi* rutherford() override { return this; }

    // ---- RutherfordApi ----
    void set_energy(double e) override {
        const double v = std::clamp(e, kRu3dEMin, kRu3dEMax);
        if (v != energy_) {
            energy_ = v;
            reset_simulation();  // only the launch momentum changes
        }
    }
    double energy() const override { return energy_; }
    void set_z(double z) override {
        const double v = std::clamp(z, kRu3dZMin, kRu3dZMax);
        if (v != z_) {
            z_ = v;
            reset_simulation();  // the repulsive potential changes
        }
    }
    double z() const override { return z_; }
    void refire() override { reset_simulation(); }
    double turning_point() const override {
        return kRu3dZProj * z_ / energy_;  // classical r_min = 2Z/E
    }
    double closest_approach() const override { return r_min_seen_; }
    double backscattered_fraction() const override { return back_; }

    // Relaunch (R key / slider): remake sim (new packet + potential) AND push
    // the new potential to the engine (the base reset re-uploads psi only).
    void reset_simulation() override {
        BaseDirector::reset_simulation();
        if (gpu_ok_) {
            engine_.wait_async();
            engine_.set_potential(sim_.potential());
            engine_.set_potential_gradient(sim_.potential());
        }
    }

protected:
    ses::WavepacketSimulation remake_simulation() const override {
        return make(energy_, z_);
    }
    const char* scene_name() const override {
        return "Rutherford scattering (repulsive Coulomb)";
    }
    double absorber_width() const override { return 10.0; }
    bool relax_allowed() const override { return false; }  // no bound target

    // The gold nucleus as a CPK-gold ball at the origin.
    int marker_count() const override { return 1; }
    SceneMarker marker(int /*i*/) const override {
        return {0.0f, 0.0f, 0.0f, 0.6f, 0.95f, 0.78f, 0.20f};
    }

    double default_camera_azimuth() const override { return 0.22; }
    double default_camera_elevation() const override { return 0.22; }
    double default_camera_distance() const override { return 110.0; }

    std::string title_suffix() override {
        // eta = Z_eff / v = 2Z / sqrt(2E): the classicality parameter,
        // matched to a real ~5 MeV alpha (eta ~ 22) at the defaults.
        const double v = std::sqrt(2.0 * energy_);
        const double eta = kRu3dZProj * z_ / v;
        return strf("  Au(Z=%.0f) <- He++ (alpha; v=%.2f a.u., eta=%.1f ~ real "
                    "5 MeV)  E = %.1f Ha  r_min = %.1f bohr  closest <r> = %.1f "
                    " backscatter %.2f",
                    z_, v, eta, energy_, turning_point(),
                    r_min_seen_ < 1e8 ? r_min_seen_ : 0.0, back_);
    }

    // Track the closest mean approach and the backscattered fraction on the
    // reduced probe cadence (a full readback is ~10 ms; every 3rd title tick
    // is plenty for ~100 au of scattering dynamics).
    void after_step_batch() override {
        if (!gpu_title_due_ || ++probe_phase_ % 3 != 0) {
            return;
        }
        engine_.wait_async();
        if (!engine_.readback(readback_buf_)) {
            return;
        }
        const ses::Grid3D& g = sim_.grid();
        const int nx = g.x.n;
        const int ny = g.y.n;
        const std::size_t cells = readback_buf_.size() / 2;
        double mr = 0.0;
        double den = 0.0;
        double back = 0.0;
        for (std::size_t idx = 0; idx < cells; ++idx) {
            const double re = readback_buf_[2 * idx];
            const double im = readback_buf_[2 * idx + 1];
            const double d = re * re + im * im;
            if (d <= 0.0) {
                continue;
            }
            const int i = static_cast<int>(idx % nx);
            const int j = static_cast<int>((idx / nx) % ny);
            const int k = static_cast<int>(idx / (nx * ny));
            const double x = g.x.coord(i);
            const double y = g.y.coord(j);
            const double z = g.z.coord(k);
            mr += std::sqrt(x * x + y * y + z * z) * d;
            den += d;
            // Backscattered: returned UPSTREAM of the launch point (x < -30).
            if (x < kRu3dLaunchX) {
                back += d;
            }
        }
        if (den > 0.0) {
            const double mean_r = mr / den;
            r_min_seen_ = std::min(r_min_seen_, mean_r);
            back_ = std::max(back_, back / den);
            title_dirty_ = true;
        }
    }

    void after_reset() override {
        r_min_seen_ = 1e9;
        back_ = 0.0;
        probe_phase_ = 0;
    }

private:
    static ses::WavepacketSimulation make(double e, double z) {
        const ses::Grid1D axis{-kRu3dBox, kRu3dBox, kRu3dPoints};
        const ses::Grid3D grid{axis, axis, axis};
        // Repulsive Coulomb: the attractive generator with charge -2Z gives
        // V = +2Z/r (regularized finite core). p0 = sqrt(2 m E), m = 1.
        const double p0 = std::sqrt(2.0 * e);
        return ses::WavepacketSimulation{ses::WavepacketSimulation::Config{
            grid,
            ses::regularized_coulomb_potential(grid, -kRu3dZProj * z,
                                               ses::Vec3d{}),
            ses::Vec3d{kRu3dLaunchX, 0.0, 0.0},
            ses::Vec3d{kRu3dSigma, kRu3dSigma, kRu3dSigma},
            ses::Vec3d{p0, 0.0, 0.0},
            kRu3dDt,
        }};
    }

    double energy_ = kRu3dEDefault;
    double z_ = kRu3dZDefault;
    double r_min_seen_ = 1e9;
    double back_ = 0.0;
    int probe_phase_ = 0;
};

}  // namespace ses_shell
