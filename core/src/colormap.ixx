module;
#include <algorithm>
#include <cmath>
#include <numbers>
export module ses.colormap;


// Colormaps for wavefunction visualization (values in [0, 1]).
//
//  - phase_color: cyclic HSV hue wheel for arg(psi) -- periodic at +-pi by
//    construction, so the phase never shows an artificial seam.
//  - magnitude_color: sequential dark-navy -> white ramp with monotone
//    brightness for densities.


export namespace ses {

struct Rgb {
    double r{};
    double g{};
    double b{};
};

// theta in [-pi, pi] (any real accepted; periodic). Full-saturation,
// full-value HSV hue wheel.
inline Rgb phase_color(double theta) noexcept {
    // Map theta to hue in [0, 6).
    double h = (theta + std::numbers::pi) / (2.0 * std::numbers::pi) * 6.0;
    h = h - 6.0 * std::floor(h / 6.0);
    const double x = 1.0 - std::abs(std::fmod(h, 2.0) - 1.0);
    switch (static_cast<int>(h)) {
        case 0: return {1.0, x, 0.0};
        case 1: return {x, 1.0, 0.0};
        case 2: return {0.0, 1.0, x};
        case 3: return {0.0, x, 1.0};
        case 4: return {x, 0.0, 1.0};
        default: return {1.0, 0.0, x};
    }
}

// t in [0, 1], clamped. Dark navy -> white, each channel monotone.
inline Rgb magnitude_color(double t) noexcept {
    t = std::clamp(t, 0.0, 1.0);
    return {std::pow(t, 1.6), std::pow(t, 0.9), 0.25 + 0.75 * t};
}

}  // namespace ses
