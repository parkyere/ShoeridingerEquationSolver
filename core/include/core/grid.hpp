#pragma once

// 1D uniform PERIODIC grid.
//
// Convention (required by the split-operator Fourier propagator):
//     h = (xmax - xmin) / n,    x_i = xmin + i*h,    i = 0 .. n-1.
// xmax is NOT a grid point; it aliases to xmin under periodicity.

namespace ses {

struct Grid1D {
    double xmin{};
    double xmax{};
    int n{};

    constexpr int size() const { return n; }
    constexpr double spacing() const { return (xmax - xmin) / n; }
    constexpr double coord(int i) const { return xmin + i * spacing(); }
};

}  // namespace ses
