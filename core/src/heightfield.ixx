module;
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <vector>
export module ses.heightfield;
export import ses.marching_cubes;
export import ses.colormap;
export import ses.field;
export import ses.grid;
export import ses.vec;
import ses.parallel;


// norm <= 0 -> flat plane (no self-norm; scene owns peak).


export namespace ses {

struct Heightfield {
    Mesh mesh;
    std::vector<Rgb> colors;
};

// amp_threshold > 0 crops the mesh to the |psi|^2 >= amp_threshold*norm
// bounding box (+1-cell skirt), skipping the negligible remainder -- for a
// compact packet in a large box this is a large triangle-count cut. 0 = whole
// grid. Cropped coords stay in the source lattice frame (rendered in place).
inline Heightfield heightfield_surface(const Field3D& psi, double z_scale,
                                       double norm, int stride = 1,
                                       double amp_threshold = 0.0) {
    const Grid3D& g = psi.grid();
    const int st = std::max(1, stride);
    const int sx = 1 + (g.x.n - 1) / st;
    const int sy = 1 + (g.y.n - 1) / st;
    const double gain = norm > 0.0 ? z_scale / norm : 0.0;

    int wi0 = 0;
    int wi1 = sx - 1;
    int wj0 = 0;
    int wj1 = sy - 1;
    if (amp_threshold > 0.0 && norm > 0.0) {
        const double cut = amp_threshold * norm;
        int lo_i = sx;
        int hi_i = -1;
        int lo_j = sy;
        int hi_j = -1;
        for (int jj = 0; jj < sy; ++jj) {
            for (int ii = 0; ii < sx; ++ii) {
                if (std::norm(psi(ii * st, jj * st, 0)) >= cut) {
                    lo_i = std::min(lo_i, ii);
                    hi_i = std::max(hi_i, ii);
                    lo_j = std::min(lo_j, jj);
                    hi_j = std::max(hi_j, jj);
                }
            }
        }
        if (hi_i < 0) {
            return {};  // all negligible: nothing to draw
        }
        wi0 = std::max(0, lo_i - 1);
        wi1 = std::min(sx - 1, hi_i + 1);
        wj0 = std::max(0, lo_j - 1);
        wj1 = std::min(sy - 1, hi_j + 1);
    }
    const int wx = wi1 - wi0 + 1;
    const int wy = wj1 - wj0 + 1;
    const std::size_t samples =
        static_cast<std::size_t>(wx) * static_cast<std::size_t>(wy);

    // Window-local (li,lj) -> sample (wi0+li, wj0+lj); disjoint rows.
    std::vector<double> hgt(samples);
    std::vector<double> phs(samples);
    parallel_for(wy, [&](int lj) {
        const int j = (wj0 + lj) * st;
        for (int li = 0; li < wx; ++li) {
            const std::complex<double> z = psi((wi0 + li) * st, j, 0);
            const std::size_t o = static_cast<std::size_t>(lj) *
                                      static_cast<std::size_t>(wx) +
                                  static_cast<std::size_t>(li);
            hgt[o] = std::norm(z) * gain;
            phs[o] = std::arg(z);
        }
    });

    // Height finite-diff normals; one-sided at window edges.
    const double hx = st * g.x.spacing();
    const double hy = st * g.y.spacing();
    std::vector<Vec3d> nrm(samples);
    parallel_for(wy, [&](int lj) {
        for (int li = 0; li < wx; ++li) {
            auto at = [&](int a, int b) {
                a = std::clamp(a, 0, wx - 1);
                b = std::clamp(b, 0, wy - 1);
                return hgt[static_cast<std::size_t>(b) *
                               static_cast<std::size_t>(wx) +
                           static_cast<std::size_t>(a)];
            };
            const double sxw = li > 0 && li < wx - 1 ? 2.0 : 1.0;
            const double syw = lj > 0 && lj < wy - 1 ? 2.0 : 1.0;
            const double dhx = (at(li + 1, lj) - at(li - 1, lj)) / (sxw * hx);
            const double dhy = (at(li, lj + 1) - at(li, lj - 1)) / (syw * hy);
            nrm[static_cast<std::size_t>(lj) * static_cast<std::size_t>(wx) +
                static_cast<std::size_t>(li)] =
                normalized(Vec3d{-dhx, -dhy, 1.0});
        }
    });

    // Two tris per cell: a,b,d and a,d,c (corners a b / c d).
    const int cx = wx - 1;
    const int cy = wy - 1;
    Heightfield hf;
    if (cx <= 0 || cy <= 0) {
        return hf;  // window too thin for a cell
    }
    const std::size_t nvert = static_cast<std::size_t>(cx) *
                              static_cast<std::size_t>(cy) * 6;
    hf.mesh.vertices.resize(nvert);
    hf.mesh.normals.resize(nvert);
    hf.colors.resize(nvert);
    parallel_for(cy, [&](int lj) {
        for (int li = 0; li < cx; ++li) {
            const std::size_t base =
                (static_cast<std::size_t>(lj) * static_cast<std::size_t>(cx) +
                 static_cast<std::size_t>(li)) *
                6;
            const int corner_i[4] = {li, li + 1, li, li + 1};
            const int corner_j[4] = {lj, lj, lj + 1, lj + 1};
            const int pick[6] = {0, 1, 3, 0, 3, 2};
            for (int v = 0; v < 6; ++v) {
                const int ci = corner_i[pick[v]];
                const int cj = corner_j[pick[v]];
                const std::size_t s =
                    static_cast<std::size_t>(cj) *
                        static_cast<std::size_t>(wx) +
                    static_cast<std::size_t>(ci);
                hf.mesh.vertices[base + v] = Vec3d{g.x.coord((wi0 + ci) * st),
                                                   g.y.coord((wj0 + cj) * st),
                                                   hgt[s]};
                hf.mesh.normals[base + v] = nrm[s];
                hf.colors[base + v] = phase_color(phs[s]);
            }
        }
    });
    return hf;
}

}  // namespace ses
