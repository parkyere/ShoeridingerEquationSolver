module;
#include <algorithm>
#include <cmath>
#include <numbers>
export module ses.colormap;


// Colormaps for wavefunction visualization (values in [0, 1]).
//
//  - phase_color: ISOLUMINANT cyclic wheel for arg(psi) -- periodic at +-pi
//    by construction, and every phase carries the SAME Rec.709 linear
//    luminance, so brightness encodes |psi|^2 alone (an HSV wheel's 13x
//    yellow-vs-blue luminance swing read as amplitude beating).
//  - magnitude_color: sequential dark-navy -> white ramp with monotone
//    brightness for densities.


export namespace ses {

struct Rgb {
    double r{};
    double g{};
    double b{};
};

// theta in [-pi, pi] (any real accepted; periodic). The color moves on a
// circle in the two ZERO-luminance chroma directions u = (1,0,0) - wR*(1,1,1)
// and v = (0,0,1) - wB*(1,1,1) (the R-Y / B-Y axes): w . u = w . v = 0
// exactly because the Rec.709 weights sum to 1, so w . rgb == kY0 for every
// theta. Amplitudes 0.55/0.50 are the largest round values keeping all
// channels in [0, 1] (worst case B: 0.5 + hypot(0.2126*0.55, 0.9278*0.50)
// = 0.978). CONTRACT: the GPU marching-cubes emit kernel
// (solver/shaders/mc_emit.comp.slang) mirrors this formula verbatim --
// vkcheck pins the two within fp32 rounding.
inline Rgb phase_color(double theta) noexcept {
    constexpr double kY0 = 0.5;
    constexpr double kWr = 0.2126;
    constexpr double kWb = 0.0722;
    const double cu = 0.55 * std::cos(theta);
    const double sv = 0.50 * std::sin(theta);
    return {kY0 + cu * (1.0 - kWr) - sv * kWb,
            kY0 - cu * kWr - sv * kWb,
            kY0 - cu * kWr + sv * (1.0 - kWb)};
}

// t in [0, 1], clamped. Dark navy -> white, each channel monotone.
inline Rgb magnitude_color(double t) noexcept {
    t = std::clamp(t, 0.0, 1.0);
    return {std::pow(t, 1.6), std::pow(t, 0.9), 0.25 + 0.75 * t};
}

}  // namespace ses
