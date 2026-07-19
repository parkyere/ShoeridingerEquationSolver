module;
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <vector>
export module ses.heightfield;
export import ses.marching_cubes;  // Mesh (triangle soup: vertices+normals)
export import ses.colormap;       // Rgb + the shared phase wheel
export import ses.field;
export import ses.grid;
export import ses.vec;
import ses.parallel;


// 2D scene surface display: |psi|^2 as HEIGHT (z = z_scale * |psi|^2 / norm),
// phase as per-vertex color (the shared ses::phase_color wheel), emitted as
// the triangle soup the mesh render path consumes. stride decimates the
// source lattice so a high-res physics grid can feed a lighter display mesh.
// norm <= 0 yields a flat zero-height plane (no self-normalization here --
// the scene owns its peak tracking).


export namespace ses {

struct Heightfield {
    Mesh mesh;
    std::vector<Rgb> colors;
};

inline Heightfield heightfield_surface(const Field3D& psi, double z_scale,
                                       double norm, int stride = 1) {
    const Grid3D& g = psi.grid();
    const int st = std::max(1, stride);
    const int sx = 1 + (g.x.n - 1) / st;
    const int sy = 1 + (g.y.n - 1) / st;
    const std::size_t samples =
        static_cast<std::size_t>(sx) * static_cast<std::size_t>(sy);
    const double gain = norm > 0.0 ? z_scale / norm : 0.0;

    // Sampled height/phase per decimated vertex (disjoint rows: parallel).
    std::vector<double> hgt(samples);
    std::vector<double> phs(samples);
    parallel_for(sy, [&](int jj) {
        const int j = jj * st;
        for (int ii = 0; ii < sx; ++ii) {
            const std::complex<double> z = psi(ii * st, j, 0);
            const std::size_t o = static_cast<std::size_t>(jj) *
                                      static_cast<std::size_t>(sx) +
                                  static_cast<std::size_t>(ii);
            hgt[o] = std::norm(z) * gain;
            phs[o] = std::arg(z);
        }
    });

    // Per-sample normals from height differences on the DECIMATED spacing
    // (central inside, one-sided at the edges); flat field -> exact +z.
    const double hx = st * g.x.spacing();
    const double hy = st * g.y.spacing();
    std::vector<Vec3d> nrm(samples);
    parallel_for(sy, [&](int jj) {
        for (int ii = 0; ii < sx; ++ii) {
            auto at = [&](int a, int b) {
                a = std::clamp(a, 0, sx - 1);
                b = std::clamp(b, 0, sy - 1);
                return hgt[static_cast<std::size_t>(b) *
                               static_cast<std::size_t>(sx) +
                           static_cast<std::size_t>(a)];
            };
            const double sxw = ii > 0 && ii < sx - 1 ? 2.0 : 1.0;
            const double syw = jj > 0 && jj < sy - 1 ? 2.0 : 1.0;
            const double dhx = (at(ii + 1, jj) - at(ii - 1, jj)) / (sxw * hx);
            const double dhy = (at(ii, jj + 1) - at(ii, jj - 1)) / (syw * hy);
            nrm[static_cast<std::size_t>(jj) * static_cast<std::size_t>(sx) +
                static_cast<std::size_t>(ii)] =
                normalized(Vec3d{-dhx, -dhy, 1.0});
        }
    });

    // Triangle soup: per cell (a b / c d) the triangles (a,b,d) and (a,d,c).
    const int cx = sx - 1;
    const int cy = sy - 1;
    Heightfield hf;
    const std::size_t nvert = static_cast<std::size_t>(cx) *
                              static_cast<std::size_t>(cy) * 6;
    hf.mesh.vertices.resize(nvert);
    hf.mesh.normals.resize(nvert);
    hf.colors.resize(nvert);
    parallel_for(cy, [&](int jj) {
        for (int ii = 0; ii < cx; ++ii) {
            const std::size_t base =
                (static_cast<std::size_t>(jj) * static_cast<std::size_t>(cx) +
                 static_cast<std::size_t>(ii)) *
                6;
            const int corner_i[4] = {ii, ii + 1, ii, ii + 1};
            const int corner_j[4] = {jj, jj, jj + 1, jj + 1};
            // Soup order a b d, a d c (indices into the corner tables).
            const int pick[6] = {0, 1, 3, 0, 3, 2};
            for (int v = 0; v < 6; ++v) {
                const int ci = corner_i[pick[v]];
                const int cj = corner_j[pick[v]];
                const std::size_t s =
                    static_cast<std::size_t>(cj) *
                        static_cast<std::size_t>(sx) +
                    static_cast<std::size_t>(ci);
                hf.mesh.vertices[base + v] =
                    Vec3d{g.x.coord(ci * st), g.y.coord(cj * st), hgt[s]};
                hf.mesh.normals[base + v] = nrm[s];
                hf.colors[base + v] = phase_color(phs[s]);
            }
        }
    });
    return hf;
}

}  // namespace ses
