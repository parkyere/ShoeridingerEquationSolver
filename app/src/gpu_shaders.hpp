#pragma once

// GLSL compute-shader sources for the GPU engine -- the SINGLE SOURCE OF
// TRUTH: sesolver_gpucheck compiles these same strings to verify each kernel.
// Extracted verbatim from gpu_engine.hpp (no logic change).

namespace ses_gpu {

// ---- kernel sources -------------------------------------------------------

// In-place pointwise complex multiply: psi <- psi * phase.
inline const char* kPhaseMultiplySrc = R"(#version 430 core
layout(local_size_x = 256) in;
layout(std430, binding = 0) buffer PsiBuf { vec2 psi[]; };
layout(std430, binding = 1) readonly buffer PhaseBuf { vec2 phase[]; };
uniform uint n;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= n) {
        return;
    }
    vec2 a = psi[i];
    vec2 b = phase[i];
    psi[i] = vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}
)";

// psi <- scale * conj(psi): with the forward FFT this yields the inverse.
inline const char* kConjScaleSrc = R"(#version 430 core
layout(local_size_x = 256) in;
layout(std430, binding = 0) buffer DataBuf { vec2 data[]; };
uniform uint n;
uniform float scale;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= n) {
        return;
    }
    data[i] = scale * vec2(data[i].x, -data[i].y);
}
)";

// Axis-generic shared-memory radix-2 line FFT: one workgroup per line,
// bit-reversal load, log2(N) barrier-separated butterfly stages. Line
// enumeration base = (l % A)*B + (l / A)*C serves any axis.
inline const char* kLineFftTemplate = R"(#version 430 core
layout(local_size_x = @NHALF@) in;
layout(std430, binding = 0) buffer DataBuf { vec2 data[]; };
uniform int mod_a;
uniform int mul_b;
uniform int mul_c;
uniform int stride;
uniform int n_lines;

shared vec2 line_sm[@N@];

const float kTwoPi = 6.28318530717958647692;

uint bit_reverse(uint v) { return bitfieldReverse(v) >> (32u - @LOG2N@u); }

void main() {
    int l = int(gl_WorkGroupID.x);
    if (l >= n_lines) {
        return;
    }
    int base = (l % mod_a) * mul_b + (l / mod_a) * mul_c;
    uint t = gl_LocalInvocationID.x;
    uint i0 = t;
    uint i1 = t + @NHALF@u;

    line_sm[i0] = data[base + int(bit_reverse(i0)) * stride];
    line_sm[i1] = data[base + int(bit_reverse(i1)) * stride];
    barrier();

    for (uint len = 2u; len <= @N@u; len <<= 1u) {
        uint half_len = len >> 1u;
        uint j = t % half_len;
        uint pos = (t / half_len) * len + j;
        float ang = -kTwoPi * float(j) / float(len);
        vec2 w = vec2(cos(ang), sin(ang));
        vec2 u = line_sm[pos];
        vec2 q = line_sm[pos + half_len];
        vec2 v = vec2(q.x * w.x - q.y * w.y, q.x * w.y + q.y * w.x);
        line_sm[pos] = u + v;
        line_sm[pos + half_len] = u - v;
        barrier();
    }

    data[base + int(i0) * stride] = line_sm[i0];
    data[base + int(i1) * stride] = line_sm[i1];
}
)";

// Sum and max of |psi_i|^2 in one pass: grid-stride accumulation, then a
// shared-memory tree reduction; one vec2(sum, max) per workgroup lands in a
// small partials SSBO (binding 2). Replaces full-buffer readbacks for the
// norm display, brightness peak, and fp32 renormalization.
inline const char* kNormPeakSrc = R"(#version 430 core
layout(local_size_x = 256) in;
layout(std430, binding = 0) readonly buffer PsiBuf { vec2 psi[]; };
layout(std430, binding = 2) writeonly buffer PartialsBuf { vec2 partials[]; };
uniform uint n;

shared float s_sum[256];
shared float s_max[256];

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint stride = gl_NumWorkGroups.x * 256u;
    float acc = 0.0;
    float mx = 0.0;
    for (uint i = gl_GlobalInvocationID.x; i < n; i += stride) {
        vec2 v = psi[i];
        float d = dot(v, v);
        acc += d;
        mx = max(mx, d);
    }
    s_sum[tid] = acc;
    s_max[tid] = mx;
    barrier();
    for (uint half_len = 128u; half_len > 0u; half_len >>= 1u) {
        if (tid < half_len) {
            s_sum[tid] += s_sum[tid + half_len];
            s_max[tid] = max(s_max[tid], s_max[tid + half_len]);
        }
        barrier();
    }
    if (tid == 0u) {
        partials[gl_WorkGroupID.x] = vec2(s_sum[0], s_max[0]);
    }
}
)";

// psi <- scale * psi (fp32 drift renormalization).
inline const char* kScaleSrc = R"(#version 430 core
layout(local_size_x = 256) in;
layout(std430, binding = 0) buffer DataBuf { vec2 data[]; };
uniform uint n;
uniform float scale;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= n) {
        return;
    }
    data[i] = scale * data[i];
}
)";

// Partial reduction of the complex inner product <phi|psi> = sum conj(phi)*psi:
// phi at binding 3 (conjugated side), psi at binding 0, vec2 partial sums to
// binding 2 (same partials buffer as the norm/peak kernel).
inline const char* kInnerProductSrc = R"(#version 430 core
layout(local_size_x = 256) in;
layout(std430, binding = 0) readonly buffer PsiBuf { vec2 psi[]; };
layout(std430, binding = 3) readonly buffer PhiBuf { vec2 phi[]; };
layout(std430, binding = 2) writeonly buffer PartialsBuf { vec2 partials[]; };
uniform uint n;

shared vec2 s_sum[256];

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint stride = gl_NumWorkGroups.x * 256u;
    vec2 acc = vec2(0.0);
    for (uint i = gl_GlobalInvocationID.x; i < n; i += stride) {
        vec2 a = phi[i];
        vec2 b = psi[i];
        // conj(a) * b
        acc += vec2(a.x * b.x + a.y * b.y, a.x * b.y - a.y * b.x);
    }
    s_sum[tid] = acc;
    barrier();
    for (uint half_len = 128u; half_len > 0u; half_len >>= 1u) {
        if (tid < half_len) {
            s_sum[tid] += s_sum[tid + half_len];
        }
        barrier();
    }
    if (tid == 0u) {
        partials[gl_WorkGroupID.x] = s_sum[0];
    }
}
)";

// psi <- psi - c * phi (complex c): the deflation projection update.
inline const char* kAxpySrc = R"(#version 430 core
layout(local_size_x = 256) in;
layout(std430, binding = 0) buffer PsiBuf { vec2 psi[]; };
layout(std430, binding = 3) readonly buffer PhiBuf { vec2 phi[]; };
uniform uint n;
uniform vec2 c;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= n) {
        return;
    }
    vec2 q = phi[i];
    psi[i] -= vec2(c.x * q.x - c.y * q.y, c.x * q.y + c.y * q.x);
}
)";

// Dipole half-kick (transitions arc T3): psi *= exp(-i * theta * (axis . r))
// -- the tested core/drive.hpp apply_dipole_halfkick with the scalar
// theta = E0 cos(omega t) dt/2 evaluated by the CALLER in double. The
// coordinate comes from the flat index (x fastest, coord = xmin + i*h),
// matching Grid1D::coord.
inline const char* kDipoleKickSrc = R"(#version 430 core
layout(local_size_x = 256) in;
layout(std430, binding = 0) buffer PsiBuf { vec2 psi[]; };
uniform uint n;
uniform int nx;
uniform int ny;
uniform vec3 box_min;
uniform vec3 cell_h;
uniform vec3 axis;
uniform float theta;
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= n) {
        return;
    }
    int i = int(idx) % nx;
    int j = (int(idx) / nx) % ny;
    int k = int(idx) / (nx * ny);
    vec3 r = box_min + vec3(float(i), float(j), float(k)) * cell_h;
    float ang = -theta * dot(axis, r);
    vec2 w = vec2(cos(ang), sin(ang));
    vec2 a = psi[idx];
    psi[idx] = vec2(a.x * w.x - a.y * w.y, a.x * w.y + a.y * w.x);
}
)";

// SSBO psi -> RG32F 3D image (the volume renderer's texture), no CPU trip.
inline const char* kBridgeSrc = R"(#version 430 core
layout(local_size_x = 256) in;
layout(std430, binding = 0) readonly buffer PsiBuf { vec2 psi[]; };
layout(rg32f, binding = 0) uniform writeonly image3D img;
uniform int nx;
uniform int ny;
uniform int nz;
void main() {
    int idx = int(gl_GlobalInvocationID.x);
    if (idx >= nx * ny * nz) {
        return;
    }
    int i = idx % nx;
    int j = (idx / nx) % ny;
    int k = idx / (nx * ny);
    imageStore(img, ivec3(i, j, k), vec4(psi[idx], 0.0, 0.0));
}
)";

// One shear of the three-shear Fourier rotation (magnetic core): after a 1D
// FFT along the freq axis, shift each line by d = coeff * (perp coord) via
// the shift theorem psi(k) *= e^{-i k d}. This is the GPU transcription of
// core/rotation.hpp's per-line phase ramp -- exact and unitary (no resample
// blur), the paramagnetic (B/2)L_z evolution of the ACTUAL psi.
inline const char* kShearSrc = R"(#version 430 core
layout(local_size_x = 256) in;
layout(std430, binding = 0) buffer PsiBuf { vec2 psi[]; };
uniform uint n;
uniform int nx;
uniform int ny;
uniform int nz;
uniform int freq_axis;   // 0=x,1=y,2=z: the FFT (freq) axis
uniform int coord_axis;  // the shift-coordinate axis (perpendicular)
uniform int nf;          // length of the freq axis
uniform float kscale;    // 2*pi / L_freq  (signed wavenumber = kscale*signed_f)
uniform float cmin;      // shift-coordinate axis min
uniform float ch;        // shift-coordinate axis spacing
uniform float coeff;     // shear coefficient
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= n) {
        return;
    }
    int e[3] = int[3](int(idx) % nx, (int(idx) / nx) % ny, int(idx) / (nx * ny));
    int f = e[freq_axis];
    int p = e[coord_axis];
    int sf = (f < nf / 2) ? f : f - nf;  // signed fftfreq
    float kf = kscale * float(sf);
    float d = coeff * (cmin + float(p) * ch);
    float ph = -kf * d;
    vec2 w = vec2(cos(ph), sin(ph));
    vec2 a = psi[idx];
    psi[idx] = vec2(a.x * w.x - a.y * w.y, a.x * w.y + a.y * w.x);
}
)";

// Mean force <grad V> = sum |psi|^2 grad V (radiation arc): the Ehrenfest
// dipole acceleration, reduced on the GPU against a precomputed grad-V field
// (binding 4, packed vec4 (gx,gy,gz,0)). vec4 partials (binding 2) tree-
// reduced per workgroup; the caller sums the 256 partials and scales by dV.
inline const char* kMeanForceSrc = R"(#version 430 core
layout(local_size_x = 256) in;
layout(std430, binding = 0) readonly buffer PsiBuf { vec2 psi[]; };
layout(std430, binding = 4) readonly buffer GradBuf { vec4 grad_v[]; };
layout(std430, binding = 2) writeonly buffer PartialsBuf { vec4 partials[]; };
uniform uint n;

shared vec4 s_acc[256];

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint stride = gl_NumWorkGroups.x * 256u;
    vec4 acc = vec4(0.0);
    for (uint i = gl_GlobalInvocationID.x; i < n; i += stride) {
        vec2 p = psi[i];
        acc += dot(p, p) * grad_v[i];  // |psi|^2 * (gx,gy,gz,0)
    }
    s_acc[tid] = acc;
    barrier();
    for (uint half_len = 128u; half_len > 0u; half_len >>= 1u) {
        if (tid < half_len) {
            s_acc[tid] += s_acc[tid + half_len];
        }
        barrier();
    }
    if (tid == 0u) {
        partials[gl_WorkGroupID.x] = s_acc[0];
    }
}
)";

// Dipole matrix element <to| r |from> = sum conj(to) * (x,y,z) * from * dV,
// reduced on the GPU straight from two resident state buffers (no CPU copy).
// conj(to)*from is the inner-product complex product; the coordinate comes
// from the flat index (x fastest), matching Grid1D::coord. Three complex
// components -> 6 floats per work group in the partials buffer; the caller
// sums the groups and scales by dV. Verified against core dipole_matrix_element
// by sesolver_gpucheck.
inline const char* kDipoleSrc = R"(#version 430 core
layout(local_size_x = 256) in;
layout(std430, binding = 0) readonly buffer ToBuf { vec2 fto[]; };
layout(std430, binding = 3) readonly buffer FromBuf { vec2 ffrom[]; };
layout(std430, binding = 2) writeonly buffer PartialsBuf { float partials[]; };
uniform uint n;
uniform int nx;
uniform int ny;
uniform vec3 box_min;
uniform vec3 cell_h;

shared vec2 sx[256];
shared vec2 sy[256];
shared vec2 sz[256];

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint stride = gl_NumWorkGroups.x * 256u;
    vec2 ax = vec2(0.0);
    vec2 ay = vec2(0.0);
    vec2 az = vec2(0.0);
    for (uint idx = gl_GlobalInvocationID.x; idx < n; idx += stride) {
        vec2 a = fto[idx];
        vec2 b = ffrom[idx];
        vec2 c = vec2(a.x * b.x + a.y * b.y, a.x * b.y - a.y * b.x);  // conj(to)*from
        int i = int(idx) % nx;
        int j = (int(idx) / nx) % ny;
        int k = int(idx) / (nx * ny);
        vec3 r = box_min + vec3(float(i), float(j), float(k)) * cell_h;
        ax += c * r.x;
        ay += c * r.y;
        az += c * r.z;
    }
    sx[tid] = ax;
    sy[tid] = ay;
    sz[tid] = az;
    barrier();
    for (uint half_len = 128u; half_len > 0u; half_len >>= 1u) {
        if (tid < half_len) {
            sx[tid] += sx[tid + half_len];
            sy[tid] += sy[tid + half_len];
            sz[tid] += sz[tid + half_len];
        }
        barrier();
    }
    if (tid == 0u) {
        uint g = gl_WorkGroupID.x;
        partials[6u * g + 0u] = sx[0].x;
        partials[6u * g + 1u] = sx[0].y;
        partials[6u * g + 2u] = sy[0].x;
        partials[6u * g + 3u] = sy[0].y;
        partials[6u * g + 4u] = sz[0].x;
        partials[6u * g + 5u] = sz[0].y;
    }
}
)";

// Orbital synthesis psi = (u(r)/r) Y_lm on the GPU: the atlas builds every
// eigenstate straight into its resident buffer, no CPU field. real_Ylm mirrors
// core/harmonics.hpp (l <= 5), and the radial interpolation mirrors
// synthesize_orbital's (u(0)=0 pins the inner segment). Verified against the
// unit-tested core by sesolver_gpucheck. Not normalized here -- the caller runs
// the norm reduction + scale afterwards (as the core's normalize() does).
inline const char* kSynthSrc = R"(#version 430 core
layout(local_size_x = 256) in;
layout(std430, binding = 0) writeonly buffer PsiBuf { vec2 psi[]; };
layout(std430, binding = 5) readonly buffer RadialBuf { float u[]; };
uniform uint n;
uniform int nx;
uniform int ny;
uniform vec3 box_min;
uniform vec3 cell_h;
uniform int l;
uniform int m;
uniform float h_radial;   // radial spacing rmax/(n_radial+1)
uniform float rmax;
uniform int n_radial;

const float PI = 3.14159265358979323846;

float real_Ylm(float x, float y, float z) {
    float r2 = x * x + y * y + z * z;
    if (l == 0) return 1.0 / (2.0 * sqrt(PI));
    if (r2 == 0.0) return 0.0;
    if (l == 1) {
        float c = sqrt(3.0 / (4.0 * PI)) / sqrt(r2);
        if (m == -1) return c * y;
        if (m == 0) return c * z;
        return c * x;
    }
    if (l == 2) {
        float c = sqrt(15.0 / PI) / r2;
        if (m == -2) return 0.5 * c * x * y;
        if (m == -1) return 0.5 * c * y * z;
        if (m == 0) return 0.25 * sqrt(5.0 / PI) * (3.0 * z * z - r2) / r2;
        if (m == 1) return 0.5 * c * z * x;
        return 0.25 * c * (x * x - y * y);
    }
    if (l == 3) {
        float r3 = r2 * sqrt(r2);
        if (m == -3) return 0.25 * sqrt(35.0 / (2.0 * PI)) * y * (3.0 * x * x - y * y) / r3;
        if (m == -2) return 0.5 * sqrt(105.0 / PI) * x * y * z / r3;
        if (m == -1) return 0.25 * sqrt(21.0 / (2.0 * PI)) * y * (5.0 * z * z - r2) / r3;
        if (m == 0) return 0.25 * sqrt(7.0 / PI) * z * (5.0 * z * z - 3.0 * r2) / r3;
        if (m == 1) return 0.25 * sqrt(21.0 / (2.0 * PI)) * x * (5.0 * z * z - r2) / r3;
        if (m == 2) return 0.25 * sqrt(105.0 / PI) * z * (x * x - y * y) / r3;
        return 0.25 * sqrt(35.0 / (2.0 * PI)) * x * (x * x - 3.0 * y * y) / r3;
    }
    if (l == 4) {
        float r4 = r2 * r2;
        if (m == -4) return 0.75 * sqrt(35.0 / PI) * x * y * (x * x - y * y) / r4;
        if (m == -3) return 0.75 * sqrt(35.0 / (2.0 * PI)) * y * z * (3.0 * x * x - y * y) / r4;
        if (m == -2) return 0.75 * sqrt(5.0 / PI) * x * y * (7.0 * z * z - r2) / r4;
        if (m == -1) return 0.75 * sqrt(5.0 / (2.0 * PI)) * y * z * (7.0 * z * z - 3.0 * r2) / r4;
        if (m == 0) return (3.0 / 16.0) * sqrt(1.0 / PI) *
                           (35.0 * z * z * z * z - 30.0 * z * z * r2 + 3.0 * r2 * r2) / r4;
        if (m == 1) return 0.75 * sqrt(5.0 / (2.0 * PI)) * x * z * (7.0 * z * z - 3.0 * r2) / r4;
        if (m == 2) return 0.375 * sqrt(5.0 / PI) * (x * x - y * y) * (7.0 * z * z - r2) / r4;
        if (m == 3) return 0.75 * sqrt(35.0 / (2.0 * PI)) * x * z * (x * x - 3.0 * y * y) / r4;
        return (3.0 / 16.0) * sqrt(35.0 / PI) *
               (x * x * x * x - 6.0 * x * x * y * y + y * y * y * y) / r4;
    }
    // l == 5
    float r5 = r2 * r2 * sqrt(r2);
    float x2 = x * x;
    float y2 = y * y;
    float z2 = z * z;
    float zpoly = 21.0 * z2 * z2 - 14.0 * z2 * r2 + r2 * r2;
    if (m == -5) return (1.0 / 32.0) * sqrt(1386.0 / PI) *
                        y * (5.0 * x2 * x2 - 10.0 * x2 * y2 + y2 * y2) / r5;
    if (m == -4) return (1.0 / 16.0) * sqrt(3465.0 / PI) * 4.0 * x * y * (x2 - y2) * z / r5;
    if (m == -3) return (1.0 / 32.0) * sqrt(770.0 / PI) * y * (3.0 * x2 - y2) * (9.0 * z2 - r2) / r5;
    if (m == -2) return (1.0 / 8.0) * sqrt(1155.0 / PI) * 2.0 * x * y * z * (3.0 * z2 - r2) / r5;
    if (m == -1) return (1.0 / 16.0) * sqrt(165.0 / PI) * y * zpoly / r5;
    if (m == 0) return (1.0 / 16.0) * sqrt(11.0 / PI) *
                       z * (63.0 * z2 * z2 - 70.0 * z2 * r2 + 15.0 * r2 * r2) / r5;
    if (m == 1) return (1.0 / 16.0) * sqrt(165.0 / PI) * x * zpoly / r5;
    if (m == 2) return (1.0 / 8.0) * sqrt(1155.0 / PI) * (x2 - y2) * z * (3.0 * z2 - r2) / r5;
    if (m == 3) return (1.0 / 32.0) * sqrt(770.0 / PI) * x * (x2 - 3.0 * y2) * (9.0 * z2 - r2) / r5;
    if (m == 4) return (1.0 / 16.0) * sqrt(3465.0 / PI) *
                       (x2 * x2 - 6.0 * x2 * y2 + y2 * y2) * z / r5;
    return (1.0 / 32.0) * sqrt(1386.0 / PI) * x * (x2 * x2 - 10.0 * x2 * y2 + 5.0 * y2 * y2) / r5;
}

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= n) {
        return;
    }
    int i = int(idx) % nx;
    int j = (int(idx) / nx) % ny;
    int k = int(idx) / (nx * ny);
    vec3 r3 = box_min + vec3(float(i), float(j), float(k)) * cell_h;
    float r = length(r3);
    float u_over_r = 0.0;
    if (r < h_radial) {
        u_over_r = u[0] / h_radial;             // u(0)=0 pinned: l=0 limit u_0/h
    } else if (r < rmax) {
        float t = r / h_radial - 1.0;           // r_i = (i+1) h
        int i0 = int(t);
        float frac = t - float(i0);
        float ui = (i0 + 1 < n_radial) ? (1.0 - frac) * u[i0] + frac * u[i0 + 1]
                                       : (1.0 - frac) * u[i0];  // outer: u(rmax)=0
        u_over_r = ui / r;
    }
    psi[idx] = vec2(u_over_r * real_Ylm(r3.x, r3.y, r3.z), 0.0);
}
)";

// Pack a complex-fp32 field (vec2 @ binding 0) into fp16 (half) storage
// (uint @ binding 6): dst[i] = packHalf2x16(src[i]). Halves the resident
// eigenstate-atlas footprint (256^3: 134 MB -> 67 MB per state) so it fits a
// small-VRAM GPU. The fp32 consumers (inner product, dipole) are unchanged --
// an fp16 state is unpacked back to a scratch fp32 buffer on demand.
inline const char* kPackHalfSrc = R"(#version 430 core
layout(local_size_x = 256) in;
layout(std430, binding = 0) readonly buffer SrcBuf { vec2 src[]; };
layout(std430, binding = 6) writeonly buffer DstBuf { uint dst[]; };
uniform uint n;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= n) {
        return;
    }
    dst[i] = packHalf2x16(src[i]);
}
)";

// Unpack fp16 (half) storage (uint @ binding 6) back to a complex-fp32 field
// (vec2 @ binding 0): dst[i] = unpackHalf2x16(src[i]).
inline const char* kUnpackHalfSrc = R"(#version 430 core
layout(local_size_x = 256) in;
layout(std430, binding = 6) readonly buffer SrcBuf { uint src[]; };
layout(std430, binding = 0) writeonly buffer DstBuf { vec2 dst[]; };
uniform uint n;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= n) {
        return;
    }
    dst[i] = unpackHalf2x16(src[i]);
}
)";

// Orbital-free projection deposit (docs/ANGULAR_PROJECTION_SCOPING.md, Phase 3).
// ONE workgroup per radial bin gathers the cells that deposit to it -- from the
// static counting-sort (core/projection.hpp build_radial_bin_index): its own
// segment (primary, weight (1-frac)/r or 1/h at the origin) and the previous
// segment (secondary, weight frac/r; their i0+1 == this bin). Each thread
// grid-strides its slice, recomputes coords/r/frac and all 36 real Y_lm (the
// SAME formulas as kSynthSrc), and accumulates Y_lm * psi * dV * w into 36
// complex partials in shared; a fixed-order tree reduction (kNormPeak pattern,
// NO atomics -> deterministic) writes g_lm[c*nr + bin]. State-count independent.
inline const char* kProjectDepositSrc = R"(#version 430 core
layout(local_size_x = 128) in;
layout(std430, binding = 0) readonly buffer PsiBuf { vec2 psi[]; };
layout(std430, binding = 6) readonly buffer SortedBuf { uint sorted_cell[]; };
layout(std430, binding = 7) readonly buffer OffBuf { uint bin_off[]; };
layout(std430, binding = 8) writeonly buffer GlmBuf { vec2 g_lm[]; };
uniform int nx;
uniform int ny;
uniform vec3 box_min;
uniform vec3 cell_h;
uniform float h_radial;
uniform float dv;
uniform int nr;

const float PI = 3.14159265358979323846;

// Parameterized real spherical harmonic (l,m): identical formulas to kSynthSrc
// (there l,m are uniforms; here args, so one dispatch can fill all 36).
float real_Ylm(int l, int m, float x, float y, float z) {
    float r2 = x * x + y * y + z * z;
    if (l == 0) return 1.0 / (2.0 * sqrt(PI));
    if (r2 == 0.0) return 0.0;
    if (l == 1) {
        float c = sqrt(3.0 / (4.0 * PI)) / sqrt(r2);
        if (m == -1) return c * y;
        if (m == 0) return c * z;
        return c * x;
    }
    if (l == 2) {
        float c = sqrt(15.0 / PI) / r2;
        if (m == -2) return 0.5 * c * x * y;
        if (m == -1) return 0.5 * c * y * z;
        if (m == 0) return 0.25 * sqrt(5.0 / PI) * (3.0 * z * z - r2) / r2;
        if (m == 1) return 0.5 * c * z * x;
        return 0.25 * c * (x * x - y * y);
    }
    if (l == 3) {
        float r3 = r2 * sqrt(r2);
        if (m == -3) return 0.25 * sqrt(35.0 / (2.0 * PI)) * y * (3.0 * x * x - y * y) / r3;
        if (m == -2) return 0.5 * sqrt(105.0 / PI) * x * y * z / r3;
        if (m == -1) return 0.25 * sqrt(21.0 / (2.0 * PI)) * y * (5.0 * z * z - r2) / r3;
        if (m == 0) return 0.25 * sqrt(7.0 / PI) * z * (5.0 * z * z - 3.0 * r2) / r3;
        if (m == 1) return 0.25 * sqrt(21.0 / (2.0 * PI)) * x * (5.0 * z * z - r2) / r3;
        if (m == 2) return 0.25 * sqrt(105.0 / PI) * z * (x * x - y * y) / r3;
        return 0.25 * sqrt(35.0 / (2.0 * PI)) * x * (x * x - 3.0 * y * y) / r3;
    }
    if (l == 4) {
        float r4 = r2 * r2;
        if (m == -4) return 0.75 * sqrt(35.0 / PI) * x * y * (x * x - y * y) / r4;
        if (m == -3) return 0.75 * sqrt(35.0 / (2.0 * PI)) * y * z * (3.0 * x * x - y * y) / r4;
        if (m == -2) return 0.75 * sqrt(5.0 / PI) * x * y * (7.0 * z * z - r2) / r4;
        if (m == -1) return 0.75 * sqrt(5.0 / (2.0 * PI)) * y * z * (7.0 * z * z - 3.0 * r2) / r4;
        if (m == 0) return (3.0 / 16.0) * sqrt(1.0 / PI) *
                           (35.0 * z * z * z * z - 30.0 * z * z * r2 + 3.0 * r2 * r2) / r4;
        if (m == 1) return 0.75 * sqrt(5.0 / (2.0 * PI)) * x * z * (7.0 * z * z - 3.0 * r2) / r4;
        if (m == 2) return 0.375 * sqrt(5.0 / PI) * (x * x - y * y) * (7.0 * z * z - r2) / r4;
        if (m == 3) return 0.75 * sqrt(35.0 / (2.0 * PI)) * x * z * (x * x - 3.0 * y * y) / r4;
        return (3.0 / 16.0) * sqrt(35.0 / PI) *
               (x * x * x * x - 6.0 * x * x * y * y + y * y * y * y) / r4;
    }
    float r5 = r2 * r2 * sqrt(r2);
    float x2 = x * x;
    float y2 = y * y;
    float z2 = z * z;
    float zpoly = 21.0 * z2 * z2 - 14.0 * z2 * r2 + r2 * r2;
    if (m == -5) return (1.0 / 32.0) * sqrt(1386.0 / PI) *
                        y * (5.0 * x2 * x2 - 10.0 * x2 * y2 + y2 * y2) / r5;
    if (m == -4) return (1.0 / 16.0) * sqrt(3465.0 / PI) * 4.0 * x * y * (x2 - y2) * z / r5;
    if (m == -3) return (1.0 / 32.0) * sqrt(770.0 / PI) * y * (3.0 * x2 - y2) * (9.0 * z2 - r2) / r5;
    if (m == -2) return (1.0 / 8.0) * sqrt(1155.0 / PI) * 2.0 * x * y * z * (3.0 * z2 - r2) / r5;
    if (m == -1) return (1.0 / 16.0) * sqrt(165.0 / PI) * y * zpoly / r5;
    if (m == 0) return (1.0 / 16.0) * sqrt(11.0 / PI) *
                       z * (63.0 * z2 * z2 - 70.0 * z2 * r2 + 15.0 * r2 * r2) / r5;
    if (m == 1) return (1.0 / 16.0) * sqrt(165.0 / PI) * x * zpoly / r5;
    if (m == 2) return (1.0 / 8.0) * sqrt(1155.0 / PI) * (x2 - y2) * z * (3.0 * z2 - r2) / r5;
    if (m == 3) return (1.0 / 32.0) * sqrt(770.0 / PI) * x * (x2 - 3.0 * y2) * (9.0 * z2 - r2) / r5;
    if (m == 4) return (1.0 / 16.0) * sqrt(3465.0 / PI) *
                       (x2 * x2 - 6.0 * x2 * y2 + y2 * y2) * z / r5;
    return (1.0 / 32.0) * sqrt(1386.0 / PI) * x * (x2 * x2 - 10.0 * x2 * y2 + 5.0 * y2 * y2) / r5;
}

shared vec2 acc[128 * 36];

void main() {
    int bin = int(gl_WorkGroupID.x);
    if (bin >= nr) {
        return;
    }
    uint tid = gl_LocalInvocationID.x;
    for (int c = 0; c < 36; ++c) {
        acc[tid * 36u + uint(c)] = vec2(0.0);
    }
    uint p_beg = bin_off[bin];
    uint p_end = bin_off[bin + 1];
    uint s_beg = (bin > 0) ? bin_off[bin - 1] : 0u;
    uint nP = p_end - p_beg;
    uint nS = (bin > 0) ? (bin_off[bin] - s_beg) : 0u;
    uint total = nP + nS;
    for (uint e = tid; e < total; e += 128u) {
        bool primary = (e < nP);
        uint cell = primary ? sorted_cell[p_beg + e] : sorted_cell[s_beg + (e - nP)];
        int i = int(cell) % nx;
        int j = (int(cell) / nx) % ny;
        int k = int(cell) / (nx * ny);
        float x = box_min.x + float(i) * cell_h.x;
        float y = box_min.y + float(j) * cell_h.y;
        float z = box_min.z + float(k) * cell_h.z;
        float r = sqrt(x * x + y * y + z * z);
        float w;
        if (primary) {
            if (r < h_radial) {
                w = 1.0 / h_radial;  // origin: constant u[0]/h (bin 0 only)
            } else {
                float t = r / h_radial - 1.0;
                float frac = t - float(int(t));
                w = (1.0 - frac) / r;
            }
        } else {
            if (r < h_radial) {
                continue;  // origin has no secondary (i0+1) contribution
            }
            float t = r / h_radial - 1.0;
            float frac = t - float(int(t));
            w = frac / r;
        }
        vec2 pv = psi[cell] * (w * dv);
        for (int l = 0; l <= 5; ++l) {
            for (int m = -l; m <= l; ++m) {
                float Y = real_Ylm(l, m, x, y, z);
                acc[tid * 36u + uint(l * l + (l + m))] += Y * pv;
            }
        }
    }
    barrier();
    for (uint half_len = 64u; half_len > 0u; half_len >>= 1u) {
        if (tid < half_len) {
            for (int c = 0; c < 36; ++c) {
                acc[tid * 36u + uint(c)] += acc[(tid + half_len) * 36u + uint(c)];
            }
        }
        barrier();
    }
    if (tid == 0u) {
        for (int c = 0; c < 36; ++c) {
            g_lm[uint(c) * uint(nr) + uint(bin)] = acc[uint(c)];
        }
    }
}
)";

}  // namespace ses_gpu
