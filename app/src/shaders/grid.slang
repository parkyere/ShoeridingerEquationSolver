// grid.slang -- flat-index <-> (i,j,k) helpers for the periodic FFT grid, and
// the cell -> world map. Two unflatten variants preserve each caller's exact
// arithmetic (so the baked ops and the oracles are unchanged):
//   unflatten_pow2   -- nx,ny powers of two: mask + shift (no runtime div/mod;
//                       ~20+ cyc saved per cell on Pascal).
//   unflatten_moddiv -- general % / (the kernels that were written that way).
// Import with `import grid;`.
module grid;

// idx -> (i,j,k), x-fastest, for POWER-OF-TWO nx,ny (mask/shift).
public int3 unflatten_pow2(uint idx, int nx, int ny) {
    int lx = int(firstbithigh(uint(nx)));
    int ly = int(firstbithigh(uint(ny)));
    return int3(int(idx & (uint(nx) - 1u)),
                int((idx >> lx) & (uint(ny) - 1u)),
                int(idx >> (lx + ly)));
}

// idx -> (i,j,k), x-fastest, general (% and /).
public int3 unflatten_moddiv(uint idx, int nx, int ny) {
    int fi = int(idx);
    return int3(fi % nx, (fi / nx) % ny, fi / (nx * ny));
}

// Grid point (i,j,k) -> world position box_min + (i,j,k)*cell_h.
public float3 cell_pos(int3 ijk, float3 box_min, float3 cell_h) {
    return box_min + float3(ijk) * cell_h;
}
