#pragma once

// ses::Complex is std::complex. The reinvention boundary excludes the C++
// STANDARD LIBRARY -- only third-party libraries (FFTW, GLM, ...) are
// hand-rolled (user decision; see docs/ARCHITECTURE.md).
//
// The build adds -fcx-limited-range on GCC/Clang so complex multiply/divide
// compile to the naive formulas (no C99 Annex G NaN/inf fixup calls): the
// exact arithmetic the test oracles are pinned against, and free of
// __muldc3 branches in the FFT hot loop.

#include <complex>

namespace ses {

template <typename T>
using Complex = std::complex<T>;

// |z|^2 without the sqrt -- the probability-density operation. A named
// wrapper over std::norm for readability at physics call sites. Call
// QUALIFIED from outside namespace ses: ADL associates std::complex with
// std, not ses.
template <typename T>
constexpr T norm_sq(const Complex<T>& z) {
    return std::norm(z);
}

}  // namespace ses
