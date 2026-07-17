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

// theta in [-pi, pi] (any real accepted; periodic). The color moves in the
// plane of the two ZERO-luminance chroma directions u = (1,0,0) - wR*(1,1,1)
// and v = (0,0,1) - wB*(1,1,1) (the R-Y / B-Y axes): w . u = w . v = 0
// exactly because the Rec.709 weights sum to 1, so w . rgb == kY0 for EVERY
// chroma scale -- which frees the scale to be per-hue GAMUT-MAXIMAL: each
// direction reaches 95% of its own channel boundary (Y0 = 0.5 centers the
// gamut, both bounds 0.5 away) instead of a fixed circle capped by the
// single worst hue (that version read washed-out). Saturated green stays
// dim by NATURE: green is 72% of luminance, so a vivid bright green cannot
// exist at constant Y. CONTRACT: the GPU marching-cubes emit kernel
// (solver/shaders/mc_emit.comp.slang) mirrors this formula verbatim --
// vkcheck pins the two within fp32 rounding.
inline Rgb phase_color(double theta) noexcept {
    constexpr double kY0 = 0.5;
    constexpr double kWr = 0.2126;
    constexpr double kWb = 0.0722;
    const double c = std::cos(theta);
    const double s = std::sin(theta);
    const double dr = c * (1.0 - kWr) - s * kWb;
    const double dg = -c * kWr - s * kWb;
    const double db = -c * kWr + s * (1.0 - kWb);
    const double m = std::max({std::abs(dr), std::abs(dg), std::abs(db)});
    const double sc = 0.475 / m;  // 0.95 * (gamut half-width at Y0 = 0.5)
    return {kY0 + sc * dr, kY0 + sc * dg, kY0 + sc * db};
}

// t in [0, 1], clamped. Dark navy -> white, each channel monotone.
inline Rgb magnitude_color(double t) noexcept {
    t = std::clamp(t, 0.0, 1.0);
    return {std::pow(t, 1.6), std::pow(t, 0.9), 0.25 + 0.75 * t};
}

}  // namespace ses
