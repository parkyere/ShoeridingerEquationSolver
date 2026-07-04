#pragma once

// GPU-resident split-operator engine (docs/GPU_PLAN.md, G4).
//
// fp32 display-speed transcription of the tested CPU double core. The kernel
// sources here are the SINGLE SOURCE OF TRUTH: sesolver_gpucheck consumes
// these same strings and verifies every kernel -- and GpuEngine::step
// end-to-end -- against the unit-tested CPU implementation.
//
// Phase tables come from SplitOperator3D's tested accessors; nothing is
// re-derived on the GPU side except the FFT twiddles (in-shader, fp32).

#include <core/complex.hpp>
#include <core/field.hpp>
#include <core/grid.hpp>

#include <QOpenGLFunctions_4_3_Core>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <string>
#include <vector>

namespace ses_gpu {

using Gl = QOpenGLFunctions_4_3_Core;

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

// ---- helpers ---------------------------------------------------------------

inline std::string instantiate(std::string tmpl, const char* token, int value) {
    const std::string t{token};
    const std::string v = std::to_string(value);
    for (std::size_t p = tmpl.find(t); p != std::string::npos; p = tmpl.find(t, p)) {
        tmpl.replace(p, t.size(), v);
        p += v.size();
    }
    return tmpl;
}

inline std::string line_fft_source(int n) {
    int log2n = 0;
    while ((1 << log2n) < n) {
        ++log2n;
    }
    std::string src{kLineFftTemplate};
    src = instantiate(std::move(src), "@NHALF@", n / 2);
    src = instantiate(std::move(src), "@N@", n);
    src = instantiate(std::move(src), "@LOG2N@", log2n);
    return src;
}

// Compile+link a compute program; returns 0 and logs on failure.
inline GLuint build_program(Gl& gl, const char* src, const char* what) {
    const GLuint shader = gl.glCreateShader(GL_COMPUTE_SHADER);
    gl.glShaderSource(shader, 1, &src, nullptr);
    gl.glCompileShader(shader);
    GLint ok = GL_FALSE;
    gl.glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (ok != GL_TRUE) {
        char log[2048];
        gl.glGetShaderInfoLog(shader, sizeof(log), nullptr, log);
        std::fprintf(stderr, "%s: compute shader compile failed:\n%s\n", what, log);
        gl.glDeleteShader(shader);
        return 0;
    }
    const GLuint prog = gl.glCreateProgram();
    gl.glAttachShader(prog, shader);
    gl.glLinkProgram(prog);
    gl.glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    gl.glDeleteShader(shader);
    if (ok != GL_TRUE) {
        char log[2048];
        gl.glGetProgramInfoLog(prog, sizeof(log), nullptr, log);
        std::fprintf(stderr, "%s: compute program link failed:\n%s\n", what, log);
        gl.glDeleteProgram(prog);
        return 0;
    }
    return prog;
}

struct AxisPass {
    int n{};
    int n_lines{};
    int mod_a{};
    int mul_b{};
    int mul_c{};
    int stride{};
};

inline void axis_passes(const ses::Grid3D& g, AxisPass out[3]) {
    const int nx = g.x.n;
    const int ny = g.y.n;
    const int nz = g.z.n;
    out[0] = {nx, ny * nz, ny * nz, nx, 0, 1};       // x-lines (contiguous)
    out[1] = {ny, nx * nz, nx, 1, nx * ny, nx};      // y-lines
    out[2] = {nz, nx * ny, nx * ny, 1, 0, nx * ny};  // z-lines
}

struct FftPrograms {
    GLuint axis[3] = {0, 0, 0};
};

inline FftPrograms build_fft_programs(Gl& gl, const ses::Grid3D& g) {
    AxisPass p[3];
    axis_passes(g, p);
    FftPrograms progs;
    for (int a = 0; a < 3; ++a) {
        progs.axis[a] = build_program(gl, line_fft_source(p[a].n).c_str(), "line fft");
    }
    return progs;
}

// Forward 3D FFT on whatever SSBO is bound at binding 0.
inline void run_fft3(Gl& gl, const ses::Grid3D& g, const FftPrograms& progs) {
    AxisPass p[3];
    axis_passes(g, p);
    for (int a = 0; a < 3; ++a) {
        gl.glUseProgram(progs.axis[a]);
        gl.glUniform1i(gl.glGetUniformLocation(progs.axis[a], "mod_a"), p[a].mod_a);
        gl.glUniform1i(gl.glGetUniformLocation(progs.axis[a], "mul_b"), p[a].mul_b);
        gl.glUniform1i(gl.glGetUniformLocation(progs.axis[a], "mul_c"), p[a].mul_c);
        gl.glUniform1i(gl.glGetUniformLocation(progs.axis[a], "stride"), p[a].stride);
        gl.glUniform1i(gl.glGetUniformLocation(progs.axis[a], "n_lines"), p[a].n_lines);
        gl.glDispatchCompute(static_cast<GLuint>(p[a].n_lines), 1, 1);
        gl.glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    }
}

inline void run_conj_scale(Gl& gl, GLuint prog, std::size_t n, float scale) {
    gl.glUseProgram(prog);
    gl.glUniform1ui(gl.glGetUniformLocation(prog, "n"), static_cast<GLuint>(n));
    gl.glUniform1f(gl.glGetUniformLocation(prog, "scale"), scale);
    gl.glDispatchCompute(static_cast<GLuint>((n + 255) / 256), 1, 1);
    gl.glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

inline std::vector<float> to_rg32f(const std::vector<ses::Complex<double>>& src) {
    std::vector<float> out(2 * src.size());
    for (std::size_t i = 0; i < src.size(); ++i) {
        out[2 * i] = static_cast<float>(src[i].real());
        out[2 * i + 1] = static_cast<float>(src[i].imag());
    }
    return out;
}

// Real weights packed as vec2(w, 0): the verified complex-multiply kernel
// then computes exactly w * psi -- no separate real-multiply kernel needed.
inline std::vector<float> pack_real_as_rg(const std::vector<double>& w) {
    std::vector<float> out(2 * w.size(), 0.0f);
    for (std::size_t i = 0; i < w.size(); ++i) {
        out[2 * i] = static_cast<float>(w[i]);
    }
    return out;
}

constexpr int kNormPeakGroups = 256;  // partials buffer: 256 vec2 = 2 KB

struct NormPeak {
    double sum{};   // sum of |psi_i|^2 (multiply by dV for the norm)
    double peak{};  // max of |psi_i|^2
};

// Reduce |psi|^2 over the SSBO bound at binding 0 into the partials buffer
// (binding 2), then finish the 256 partial pairs on the CPU in double.
inline NormPeak run_norm_peak(Gl& gl, GLuint prog, GLuint partials_buf, std::size_t n) {
    gl.glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, partials_buf);
    gl.glUseProgram(prog);
    gl.glUniform1ui(gl.glGetUniformLocation(prog, "n"), static_cast<GLuint>(n));
    gl.glDispatchCompute(kNormPeakGroups, 1, 1);
    gl.glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

    float partials[2 * kNormPeakGroups];
    gl.glBindBuffer(GL_SHADER_STORAGE_BUFFER, partials_buf);
    gl.glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(partials), partials);

    NormPeak out;
    for (int g = 0; g < kNormPeakGroups; ++g) {
        out.sum += partials[2 * g];
        out.peak = std::max(out.peak, static_cast<double>(partials[2 * g + 1]));
    }
    return out;
}

inline void run_scale(Gl& gl, GLuint prog, std::size_t n, float scale) {
    gl.glUseProgram(prog);
    gl.glUniform1ui(gl.glGetUniformLocation(prog, "n"), static_cast<GLuint>(n));
    gl.glUniform1f(gl.glGetUniformLocation(prog, "scale"), scale);
    gl.glDispatchCompute(static_cast<GLuint>((n + 255) / 256), 1, 1);
    gl.glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

// ---- the engine ------------------------------------------------------------

class GpuEngine {
public:
    // All GL calls require a current 4.3 context. Returns false if any
    // program fails to build (caller falls back to the CPU path).
    bool initialize(Gl& gl, const ses::Grid3D& grid,
                    const std::vector<ses::Complex<double>>& half_v,
                    const std::vector<ses::Complex<double>>& kinetic,
                    const ses::Field3D& psi0) {
        grid_ = grid;
        cells_ = static_cast<std::size_t>(grid.size());

        mul_prog_ = build_program(gl, kPhaseMultiplySrc, "phase multiply");
        conj_prog_ = build_program(gl, kConjScaleSrc, "conj scale");
        bridge_prog_ = build_program(gl, kBridgeSrc, "texture bridge");
        norm_prog_ = build_program(gl, kNormPeakSrc, "norm/peak reduce");
        scale_prog_ = build_program(gl, kScaleSrc, "scale");
        inner_prog_ = build_program(gl, kInnerProductSrc, "inner product");
        axpy_prog_ = build_program(gl, kAxpySrc, "axpy");
        fft_progs_ = build_fft_programs(gl, grid);
        if (mul_prog_ == 0 || conj_prog_ == 0 || bridge_prog_ == 0 || norm_prog_ == 0 ||
            scale_prog_ == 0 || inner_prog_ == 0 || axpy_prog_ == 0 ||
            fft_progs_.axis[0] == 0 || fft_progs_.axis[1] == 0 ||
            fft_progs_.axis[2] == 0) {
            return false;
        }

        psi_buf_ = make_buffer(gl, to_rg32f(psi0.data()));
        half_buf_ = make_buffer(gl, to_rg32f(half_v));
        kin_buf_ = make_buffer(gl, to_rg32f(kinetic));
        partials_buf_ = make_buffer(gl, std::vector<float>(2 * kNormPeakGroups, 0.0f));
        return true;
    }

    // Imaginary-time weight tables (from ImaginaryTimePropagator3D's tested
    // accessors), packed as vec2(w, 0) for the complex-multiply kernel.
    void set_relax_tables(Gl& gl, const std::vector<double>& half_w,
                          const std::vector<double>& kin_w, double dtau) {
        relax_dtau_ = dtau;
        relax_half_buf_ = make_buffer(gl, pack_real_as_rg(half_w));
        relax_kin_buf_ = make_buffer(gl, pack_real_as_rg(kin_w));
    }

    struct RelaxStats {
        double energy{};  // ITP estimator: E ~= -ln||psi||^2 / (2 dtau)
        double peak{};    // max |psi_i|^2 after renormalization
    };

    // e^{-H dtau} Strang steps with per-step renormalization (the flow is
    // not unitary). The pre-renormalization norm decays as e^{-2 E dtau}
    // near an eigenstate, so the energy estimate is free.
    RelaxStats relax_step(Gl& gl, int nsteps) {
        RelaxStats stats;
        gl.glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, psi_buf_);
        for (int s = 0; s < nsteps; ++s) {
            strang_imaginary(gl);
            stats = renormalize_and_estimate(gl);
        }
        return stats;
    }

    // Deflated variant (transitions arc T1-GPU): after each Strang step the
    // given lower eigenstates (as state buffers) are projected out, so the
    // flow converges to the next EXCITED state; the free energy estimator
    // then reads that state's energy.
    RelaxStats relax_deflated_step(Gl& gl, const std::vector<GLuint>& lower, int nsteps) {
        RelaxStats stats;
        gl.glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, psi_buf_);
        for (int s = 0; s < nsteps; ++s) {
            strang_imaginary(gl);
            for (const GLuint phi : lower) {
                const NormPeak ip = inner_with_psi(gl, phi);  // (re, im) in sum/peak
                subtract_projection(gl, phi, ip.sum, ip.peak);
            }
            stats = renormalize_and_estimate(gl);
        }
        return stats;
    }

    // <phi|psi> via the GPU reduction; returned as NormPeak{re, im} (the
    // partials pair is reused as a complex accumulator). Includes dV.
    NormPeak inner_with_psi(Gl& gl, GLuint phi_buf) {
        gl.glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, psi_buf_);
        gl.glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, partials_buf_);
        gl.glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, phi_buf);
        gl.glUseProgram(inner_prog_);
        gl.glUniform1ui(gl.glGetUniformLocation(inner_prog_, "n"),
                        static_cast<GLuint>(cells_));
        gl.glDispatchCompute(kNormPeakGroups, 1, 1);
        gl.glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

        float partials[2 * kNormPeakGroups];
        gl.glBindBuffer(GL_SHADER_STORAGE_BUFFER, partials_buf_);
        gl.glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(partials), partials);
        double re = 0.0;
        double im = 0.0;
        for (int g = 0; g < kNormPeakGroups; ++g) {
            re += partials[2 * g];
            im += partials[2 * g + 1];
        }
        const double dv = grid_.cell_volume();
        return NormPeak{re * dv, im * dv};
    }

    // psi <- psi - c * phi (complex c = (cre, cim)): projects phi out when
    // c is the overlap <phi|psi>.
    void subtract_projection(Gl& gl, GLuint phi_buf, double cre, double cim) {
        gl.glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, psi_buf_);
        gl.glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, phi_buf);
        gl.glUseProgram(axpy_prog_);
        gl.glUniform1ui(gl.glGetUniformLocation(axpy_prog_, "n"),
                        static_cast<GLuint>(cells_));
        gl.glUniform2f(gl.glGetUniformLocation(axpy_prog_, "c"), static_cast<float>(cre),
                       static_cast<float>(cim));
        gl.glDispatchCompute(static_cast<GLuint>((cells_ + 255) / 256), 1, 1);
        gl.glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    }

    // Upload an auxiliary state (e.g. a cached eigenstate) into its own SSBO.
    GLuint create_state_buffer(Gl& gl, const ses::Field3D& state) {
        return make_buffer(gl, to_rg32f(state.data()));
    }

    // GPU-reduced norm (sum |psi|^2 * dV) and peak density -- a 2 KB readback
    // instead of the full field.
    NormPeak norm_and_peak(Gl& gl) {
        gl.glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, psi_buf_);
        NormPeak np = run_norm_peak(gl, norm_prog_, partials_buf_, cells_);
        np.sum *= grid_.cell_volume();
        return np;
    }

    // psi <- s * psi (fp32 drift renormalization).
    void scale(Gl& gl, float s) {
        gl.glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, psi_buf_);
        run_scale(gl, scale_prog_, cells_, s);
    }

    void upload_state(Gl& gl, const ses::Field3D& psi) {
        const std::vector<float> staged = to_rg32f(psi.data());
        // Order this API write against any in-flight shader writes from
        // earlier frames (GL 4.3: SSBO shader writes are incoherent, and the
        // per-dispatch SHADER_STORAGE barrier only covers shader-to-shader).
        gl.glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
        gl.glBindBuffer(GL_SHADER_STORAGE_BUFFER, psi_buf_);
        gl.glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                           static_cast<GLsizeiptr>(staged.size() * sizeof(float)),
                           staged.data());
    }

    // Interleaved RG floats, 2 per cell.
    void readback(Gl& gl, std::vector<float>& out) const {
        out.resize(2 * cells_);
        // Make incoherent compute writes visible to the buffer-update API
        // before glGetBufferSubData (spec-required; drivers often forgive).
        gl.glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
        gl.glBindBuffer(GL_SHADER_STORAGE_BUFFER, psi_buf_);
        gl.glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                              static_cast<GLsizeiptr>(out.size() * sizeof(float)),
                              out.data());
    }

    // Strang step, fully GPU-resident:
    //   psi <- halfV . IFFT . kinetic . FFT . halfV psi
    // with IFFT = conj -> FFT -> conj/N (the same identity as the CPU core).
    void step(Gl& gl, int nsteps) {
        gl.glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, psi_buf_);
        for (int s = 0; s < nsteps; ++s) {
            multiply(gl, half_buf_);
            run_fft3(gl, grid_, fft_progs_);
            multiply(gl, kin_buf_);
            run_conj_scale(gl, conj_prog_, cells_, 1.0f);
            run_fft3(gl, grid_, fft_progs_);
            run_conj_scale(gl, conj_prog_, cells_, 1.0f / static_cast<float>(cells_));
            multiply(gl, half_buf_);
        }
    }

    // Write psi into the volume renderer's RG32F 3D texture, GPU-to-GPU.
    void write_psi_texture(Gl& gl, GLuint texture) {
        gl.glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, psi_buf_);
        gl.glBindImageTexture(0, texture, 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_RG32F);
        gl.glUseProgram(bridge_prog_);
        gl.glUniform1i(gl.glGetUniformLocation(bridge_prog_, "nx"), grid_.x.n);
        gl.glUniform1i(gl.glGetUniformLocation(bridge_prog_, "ny"), grid_.y.n);
        gl.glUniform1i(gl.glGetUniformLocation(bridge_prog_, "nz"), grid_.z.n);
        gl.glDispatchCompute(static_cast<GLuint>((cells_ + 255) / 256), 1, 1);
        gl.glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT | GL_TEXTURE_UPDATE_BARRIER_BIT);
    }

private:
    // One e^{-H dtau} Strang pass over psi (no normalization).
    void strang_imaginary(Gl& gl) {
        multiply(gl, relax_half_buf_);
        run_fft3(gl, grid_, fft_progs_);
        multiply(gl, relax_kin_buf_);
        run_conj_scale(gl, conj_prog_, cells_, 1.0f);
        run_fft3(gl, grid_, fft_progs_);
        run_conj_scale(gl, conj_prog_, cells_, 1.0f / static_cast<float>(cells_));
        multiply(gl, relax_half_buf_);
    }

    RelaxStats renormalize_and_estimate(Gl& gl) {
        const NormPeak np = run_norm_peak(gl, norm_prog_, partials_buf_, cells_);
        const double norm_sq = np.sum * grid_.cell_volume();
        RelaxStats stats;
        stats.energy = -std::log(norm_sq) / (2.0 * relax_dtau_);
        stats.peak = np.peak / norm_sq;
        run_scale(gl, scale_prog_, cells_, static_cast<float>(1.0 / std::sqrt(norm_sq)));
        return stats;
    }

    static GLuint make_buffer(Gl& gl, const std::vector<float>& data) {
        GLuint buf = 0;
        gl.glGenBuffers(1, &buf);
        gl.glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf);
        gl.glBufferData(GL_SHADER_STORAGE_BUFFER,
                        static_cast<GLsizeiptr>(data.size() * sizeof(float)), data.data(),
                        GL_DYNAMIC_COPY);
        return buf;
    }

    void multiply(Gl& gl, GLuint phase_buf) {
        gl.glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, phase_buf);
        gl.glUseProgram(mul_prog_);
        gl.glUniform1ui(gl.glGetUniformLocation(mul_prog_, "n"),
                        static_cast<GLuint>(cells_));
        gl.glDispatchCompute(static_cast<GLuint>((cells_ + 255) / 256), 1, 1);
        gl.glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    }

    ses::Grid3D grid_{};
    std::size_t cells_ = 0;
    GLuint psi_buf_ = 0;
    GLuint half_buf_ = 0;
    GLuint kin_buf_ = 0;
    GLuint relax_half_buf_ = 0;
    GLuint relax_kin_buf_ = 0;
    double relax_dtau_ = 0.0;
    GLuint partials_buf_ = 0;
    GLuint mul_prog_ = 0;
    GLuint conj_prog_ = 0;
    GLuint bridge_prog_ = 0;
    GLuint norm_prog_ = 0;
    GLuint scale_prog_ = 0;
    GLuint inner_prog_ = 0;
    GLuint axpy_prog_ = 0;
    FftPrograms fft_progs_;
};

}  // namespace ses_gpu
