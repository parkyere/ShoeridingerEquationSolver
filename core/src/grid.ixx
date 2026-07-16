export module ses.grid;


// 1D uniform PERIODIC grid.
//
// Convention (required by the split-operator Fourier propagator):
//     h = (xmax - xmin) / n,    x_i = xmin + i*h,    i = 0 .. n-1.
// xmax is NOT a grid point; it aliases to xmin under periodicity.


export namespace ses {

struct Grid1D {
    double xmin{};
    double xmax{};
    int n{};

    constexpr int size() const noexcept { return n; }
    constexpr double spacing() const noexcept { return (xmax - xmin) / n; }
    constexpr double coord(int i) const noexcept { return xmin + i * spacing(); }
};

// 3D periodic grid: three independent Grid1D axes.
// Flat layout is X FASTEST -- flat(i,j,k) = i + nx*(j + ny*k) -- so x-lines
// are contiguous (3D FFT) and the layout matches the tightly packed
// row-major order GPU 3D-texture uploads expect.
struct Grid3D {
    Grid1D x{};
    Grid1D y{};
    Grid1D z{};

    constexpr int size() const noexcept { return x.n * y.n * z.n; }
    constexpr int flat(int i, int j, int k) const noexcept { return i + x.n * (j + y.n * k); }
    constexpr double cell_volume() const noexcept { return x.spacing() * y.spacing() * z.spacing(); }
};

}  // namespace ses
