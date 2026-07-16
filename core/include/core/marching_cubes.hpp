#pragma once

// Marching cubes: extract the isosurface of a real scalar field sampled on a
// Grid3D. The surface encloses the region where field > isovalue (the inside
// of a density cloud); normals are unit length and point OUTWARD, i.e. down
// the gradient. Output is an unindexed triangle soup (3 vertices + 3 normals
// per triangle).

import ses.mc.tables;

import ses.grid;
import ses.vec;

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace ses {

struct Mesh {
    std::vector<Vec3d> vertices;
    std::vector<Vec3d> normals;
};

namespace mc_detail {

// Central-difference gradient of the field at a grid point, one-sided at the
// domain boundary (indices clamped).
inline Vec3d gradient(const std::vector<double>& f, const Grid3D& g, int i, int j, int k) noexcept {
    auto at = [&](int ii, int jj, int kk) {
        ii = ii < 0 ? 0 : (ii >= g.x.n ? g.x.n - 1 : ii);
        jj = jj < 0 ? 0 : (jj >= g.y.n ? g.y.n - 1 : jj);
        kk = kk < 0 ? 0 : (kk >= g.z.n ? g.z.n - 1 : kk);
        return f[static_cast<std::size_t>(g.flat(ii, jj, kk))];
    };
    return Vec3d{(at(i + 1, j, k) - at(i - 1, j, k)) / (2.0 * g.x.spacing()),
                 (at(i, j + 1, k) - at(i, j - 1, k)) / (2.0 * g.y.spacing()),
                 (at(i, j, k + 1) - at(i, j, k - 1)) / (2.0 * g.z.spacing())};
}

}  // namespace mc_detail

inline Mesh marching_cubes(const std::vector<double>& field, const Grid3D& g, double isovalue) {
    Mesh mesh;

    Vec3d corner_pos[8];
    double corner_val[8];
    Vec3d corner_grad[8];
    Vec3d edge_vertex[12];
    Vec3d edge_normal[12];

    for (int k = 0; k + 1 < g.z.n; ++k) {
        for (int j = 0; j + 1 < g.y.n; ++j) {
            for (int i = 0; i + 1 < g.x.n; ++i) {
                int cube_index = 0;
                for (int c = 0; c < 8; ++c) {
                    const int ci = i + mc::kCornerOffset[c][0];
                    const int cj = j + mc::kCornerOffset[c][1];
                    const int ck = k + mc::kCornerOffset[c][2];
                    corner_pos[c] = Vec3d{g.x.coord(ci), g.y.coord(cj), g.z.coord(ck)};
                    corner_val[c] = field[static_cast<std::size_t>(g.flat(ci, cj, ck))];
                    if (corner_val[c] < isovalue) {
                        cube_index |= 1 << c;
                    }
                }

                const int edges = mc::kEdgeTable[cube_index];
                if (edges == 0) {
                    continue;
                }

                for (int c = 0; c < 8; ++c) {
                    corner_grad[c] = mc_detail::gradient(field, g, i + mc::kCornerOffset[c][0],
                                                         j + mc::kCornerOffset[c][1],
                                                         k + mc::kCornerOffset[c][2]);
                }

                for (int e = 0; e < 12; ++e) {
                    if ((edges & (1 << e)) == 0) {
                        continue;
                    }
                    const int a = mc::kEdgeCorner[e][0];
                    const int b = mc::kEdgeCorner[e][1];
                    const double t = (isovalue - corner_val[a]) / (corner_val[b] - corner_val[a]);
                    const Vec3d& pa = corner_pos[a];
                    const Vec3d& pb = corner_pos[b];
                    edge_vertex[e] = Vec3d{pa.x + t * (pb.x - pa.x), pa.y + t * (pb.y - pa.y),
                                           pa.z + t * (pb.z - pa.z)};
                    const Vec3d& ga = corner_grad[a];
                    const Vec3d& gb = corner_grad[b];
                    // Outward = down-gradient (inside has the higher value).
                    Vec3d n{-(ga.x + t * (gb.x - ga.x)), -(ga.y + t * (gb.y - ga.y)),
                            -(ga.z + t * (gb.z - ga.z))};
                    const double len = std::sqrt(n.x * n.x + n.y * n.y + n.z * n.z);
                    if (len > 0.0) {
                        n = Vec3d{n.x / len, n.y / len, n.z / len};
                    }
                    edge_normal[e] = n;
                }

                for (int t = 0; mc::kTriTable[cube_index][t] != -1; t += 3) {
                    for (int v = 0; v < 3; ++v) {
                        const int e = mc::kTriTable[cube_index][t + v];
                        mesh.vertices.push_back(edge_vertex[e]);
                        mesh.normals.push_back(edge_normal[e]);
                    }
                }
            }
        }
    }

    return mesh;
}

// Isovalue as a fraction of the current field peak -- keeps an animated
// (dispersing) cloud visible as its maximum density decays.
inline Mesh marching_cubes_at_fraction(const std::vector<double>& field, const Grid3D& g,
                                       double fraction) {
    if (field.empty()) {
        return Mesh{};  // *max_element(end()) is UB
    }
    const double peak = *std::max_element(field.begin(), field.end());
    if (!(peak > 0.0)) {
        return Mesh{};  // no positive density -> no isosurface at any fraction
    }
    return marching_cubes(field, g, fraction * peak);
}

}  // namespace ses
