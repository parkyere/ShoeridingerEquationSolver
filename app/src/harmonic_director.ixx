module;
#include <cmath>
#include <string>
export module ses.app.harmonic_director;
export import ses.app.base_director;


// The 3D isotropic harmonic trap scenario: a coherent state (the ground
// Gaussian displaced by x0) oscillates rigidly at omega -- the textbook
// oracle the core tests pin. Key 2 relaxes to the ground state
// (E -> 3 omega / 2); R re-displaces a fresh coherent state.


export namespace ses_shell {

constexpr double kTrapOmega = 0.25;      // au; period 2 pi / w ~ 25 au
constexpr double kTrapBox = 20.0;        // Bohr half-extent (h = 0.15625)
constexpr double kCoherentOffset = 8.0;  // Bohr; classical turning point

class HarmonicDirector final : public BaseDirector {
public:
    HarmonicDirector() : BaseDirector(make()) {}

protected:
    ses::WavepacketSimulation remake_simulation() const override { return make(); }
    const char* scene_name() const override { return "Harmonic trap"; }
    double default_camera_distance() const override { return 45.0; }
    std::string title_suffix() override {
        return strf("  w = %.2f au (T = %.1f au, E0 = %.3f Ha)", kTrapOmega,
                    6.28318530717959 / kTrapOmega, 1.5 * kTrapOmega);
    }

private:
    // Coherent state: sigma = 1/sqrt(2 w) is the ground |psi|^2 width (var
    // 1/(2w)), so the displaced Gaussian oscillates rigidly without breathing
    // -- the same width harmonic_dynamics_test pins (kSigmaGs).
    static ses::WavepacketSimulation make() {
        const ses::Grid1D axis{-kTrapBox, kTrapBox, 256};
        const ses::Grid3D grid{axis, axis, axis};
        const double sigma = 1.0 / std::sqrt(2.0 * kTrapOmega);
        return ses::WavepacketSimulation{ses::WavepacketSimulation::Config{
            grid,
            ses::harmonic_potential(grid, kTrapOmega, ses::Vec3d{}),
            ses::Vec3d{kCoherentOffset, 0.0, 0.0},
            ses::Vec3d{sigma, sigma, sigma},
            ses::Vec3d{},  // released at rest: x(t) = x0 cos(w t)
            0.04,          // dt
        }};
    }
};

}  // namespace ses_shell
