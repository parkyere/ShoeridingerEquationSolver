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
#include <core/decay.hpp>
#include <core/drive.hpp>
#include <core/field.hpp>
#include <core/grid.hpp>
#include <core/projection.hpp>

#include <QOpenGLFunctions_4_3_Core>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>
#include "gpu_shaders.hpp"

namespace ses_gpu {

using Gl = QOpenGLFunctions_4_3_Core;

// Named SSBO binding slots (replace the magic ints in glBindBufferBase).
// Slot 6 is reused across programs: fp16-packed state, and the projection
// sorted-cell index (never bound together).
namespace bind {
constexpr GLuint kPsi = 0;       // primary in/out (psi or a target buffer)
constexpr GLuint kPhase = 1;     // phase / mask multiply table
constexpr GLuint kPartials = 2;  // reduction partials
constexpr GLuint kPhi = 3;       // second operand (inner/axpy/dipole-from)
constexpr GLuint kGradV = 4;     // grad V field (mean_force)
constexpr GLuint kRadial = 5;    // radial u_nl table (synthesis)
constexpr GLuint kAux = 6;       // fp16-packed state / projection sorted cells
constexpr GLuint kBinOff = 7;    // projection CSR bin offsets
constexpr GLuint kGlm = 8;       // projection g_lm accumulator
}  // namespace bind


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
        // The per-axis stride/count uniforms are CONSTANT for a fixed grid, so
        // set them ONCE here (they persist in program state) instead of before
        // every dispatch in the hot FFT loop (~60 redundant GL calls/step).
        gl.glUseProgram(progs.axis[a]);
        gl.glUniform1i(gl.glGetUniformLocation(progs.axis[a], "mod_a"), p[a].mod_a);
        gl.glUniform1i(gl.glGetUniformLocation(progs.axis[a], "mul_b"), p[a].mul_b);
        gl.glUniform1i(gl.glGetUniformLocation(progs.axis[a], "mul_c"), p[a].mul_c);
        gl.glUniform1i(gl.glGetUniformLocation(progs.axis[a], "stride"), p[a].stride);
        gl.glUniform1i(gl.glGetUniformLocation(progs.axis[a], "n_lines"), p[a].n_lines);
    }
    return progs;
}

// Forward 3D FFT on whatever SSBO is bound at binding 0.
inline void run_fft3(Gl& gl, const ses::Grid3D& g, const FftPrograms& progs) {
    AxisPass p[3];
    axis_passes(g, p);
    for (int a = 0; a < 3; ++a) {
        gl.glUseProgram(progs.axis[a]);  // uniforms fixed at build (see above)
        gl.glDispatchCompute(static_cast<GLuint>(p[a].n_lines), 1, 1);
        gl.glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    }
}

// Forward 1D FFT along a single axis (0=x,1=y,2=z) of the SSBO at binding 0.
inline void run_fft_axis(Gl& gl, const ses::Grid3D& g, const FftPrograms& progs,
                         int a) {
    AxisPass p[3];
    axis_passes(g, p);
    gl.glUseProgram(progs.axis[a]);  // uniforms fixed at build (build_fft_programs)
    gl.glDispatchCompute(static_cast<GLuint>(p[a].n_lines), 1, 1);
    gl.glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
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

// A resident atlas state buffer plus its storage precision. The consumers
// (inner_with_psi / dipole_between) take this so one signature handles both
// precisions: an fp16 operand is decoded to a scratch fp32 buffer, then the
// SAME tested fp32 kernel runs; fp32 takes the fast path. Bundling the flag
// with the buffer removes the parallel _p/_half method family.
struct StateHandle {
    GLuint buf = 0;
    bool fp16 = false;
};

// Reduce |psi|^2 over the SSBO bound at binding 0 into the partials buffer
// (binding 2), then finish the 256 partial pairs on the CPU in double.
inline NormPeak run_norm_peak(Gl& gl, GLuint prog, GLuint partials_buf, std::size_t n) {
    gl.glBindBufferBase(GL_SHADER_STORAGE_BUFFER, bind::kPartials, partials_buf);
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

// SSBO creation helpers (shared by the engine and the concern modules). The
// usage hint matters: GL_DYNAMIC_COPY for buffers re-uploaded often, GL_STATIC_COPY
// for write-once data (index tables, eigenstates) so the driver keeps no host shadow.
inline GLuint make_buffer(Gl& gl, const std::vector<float>& data) {
    GLuint buf = 0;
    gl.glGenBuffers(1, &buf);
    gl.glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf);
    gl.glBufferData(GL_SHADER_STORAGE_BUFFER,
                    static_cast<GLsizeiptr>(data.size() * sizeof(float)), data.data(),
                    GL_DYNAMIC_COPY);
    return buf;
}

inline GLuint make_u32_buffer(Gl& gl, const std::vector<std::uint32_t>& data) {
    GLuint buf = 0;
    gl.glGenBuffers(1, &buf);
    gl.glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf);
    gl.glBufferData(GL_SHADER_STORAGE_BUFFER,
                    static_cast<GLsizeiptr>(data.size() * sizeof(std::uint32_t)),
                    data.data(), GL_STATIC_COPY);
    return buf;
}

inline GLuint make_static_buffer(Gl& gl, const std::vector<float>& data) {
    GLuint buf = 0;
    gl.glGenBuffers(1, &buf);
    gl.glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf);
    gl.glBufferData(GL_SHADER_STORAGE_BUFFER,
                    static_cast<GLsizeiptr>(data.size() * sizeof(float)), data.data(),
                    GL_STATIC_COPY);
    return buf;
}

struct GpuState {
    ses::Grid3D grid_{};
    std::size_t cells_ = 0;
    GLuint psi_buf_ = 0;
    GLuint partials_buf_ = 0;
    GLuint mul_prog_ = 0;
    GLuint conj_prog_ = 0;
    GLuint bridge_prog_ = 0;
    GLuint norm_prog_ = 0;
    GLuint scale_prog_ = 0;
    FftPrograms fft_progs_;
};

// Concern modules (self-contained, verified by sesolver_gpucheck). Included
// here -- AFTER the shared utilities above (bind, build_program, make_*) -- so
// they can use them; they are not standalone headers.
#include "gpu_projection.hpp"
#include "gpu_orbital.hpp"
#include "gpu_observables.hpp"

// ---- the engine ------------------------------------------------------------

class GpuEngine {
public:
    // All GL calls require a current 4.3 context. Returns false if any
    // program fails to build (caller falls back to the CPU path).
    bool initialize(Gl& gl, const ses::Grid3D& grid,
                    const std::vector<ses::Complex<double>>& half_v,
                    const std::vector<ses::Complex<double>>& kinetic,
                    const ses::Field3D& psi0) {
        st_.grid_ = grid;
        st_.cells_ = static_cast<std::size_t>(grid.size());

        st_.mul_prog_ = build_program(gl, kPhaseMultiplySrc, "phase multiply");
        st_.conj_prog_ = build_program(gl, kConjScaleSrc, "conj scale");
        st_.bridge_prog_ = build_program(gl, kBridgeSrc, "texture bridge");
        st_.norm_prog_ = build_program(gl, kNormPeakSrc, "norm/peak reduce");
        st_.scale_prog_ = build_program(gl, kScaleSrc, "scale");
        axpy_prog_ = build_program(gl, kAxpySrc, "axpy");
        kick_prog_ = build_program(gl, kDipoleKickSrc, "dipole kick");
        shear_prog_ = build_program(gl, kShearSrc, "shear");
        obs_.build(gl);
        orbital_.build(gl);
        projector_.build(gl);
        st_.fft_progs_ = build_fft_programs(gl, grid);
        if (st_.mul_prog_ == 0 || st_.conj_prog_ == 0 || st_.bridge_prog_ == 0 || st_.norm_prog_ == 0 ||
            st_.scale_prog_ == 0 || axpy_prog_ == 0 || kick_prog_ == 0 || shear_prog_ == 0 ||
            obs_.inner_prog_ == 0 || obs_.force_prog_ == 0 || obs_.dipole_prog_ == 0 ||
            orbital_.synth_prog_ == 0 || orbital_.pack_prog_ == 0 ||
            orbital_.unpack_prog_ == 0 || projector_.prog_ == 0 ||
            st_.fft_progs_.axis[0] == 0 || st_.fft_progs_.axis[1] == 0 ||
            st_.fft_progs_.axis[2] == 0) {
            return false;
        }

        st_.psi_buf_ = make_buffer(gl, to_rg32f(psi0.data()));
        half_buf_ = make_buffer(gl, to_rg32f(half_v));
        kin_buf_ = make_buffer(gl, to_rg32f(kinetic));
        st_.partials_buf_ = make_buffer(gl, std::vector<float>(2 * kNormPeakGroups, 0.0f));
        obs_.alloc_partials(gl);
        return true;
    }

    // Replace the half-potential phase table in place. Used to fold in the
    // diamagnetic (B^2/8) rho^2 term for the magnetic step (and to restore the
    // base atom when the field is off); the table size is unchanged.
    void set_half_potential(Gl& gl, const std::vector<ses::Complex<double>>& half_v) {
        const std::vector<float> staged = to_rg32f(half_v);
        gl.glBindBuffer(GL_SHADER_STORAGE_BUFFER, half_buf_);
        gl.glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                           static_cast<GLsizeiptr>(staged.size() * sizeof(float)),
                           staged.data());
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
        gl.glBindBufferBase(GL_SHADER_STORAGE_BUFFER, bind::kPsi, st_.psi_buf_);
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
        gl.glBindBufferBase(GL_SHADER_STORAGE_BUFFER, bind::kPsi, st_.psi_buf_);
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

    // <phi|psi> -- forwarded to the Observables concern module (binds the live
    // psi as the first operand). Returned as NormPeak{re, im}, includes dV.
    NormPeak inner_with_psi(Gl& gl, GLuint phi_buf) {
        return obs_.inner(gl, st_, st_.psi_buf_, phi_buf);
    }

    // Upload the potential-gradient field grad V so mean_force can reduce
    // against it -- forwarded to the Observables concern module.
    void set_potential_gradient(Gl& gl, const std::vector<double>& v) {
        obs_.set_potential_gradient(gl, st_, v);
    }

    // <grad V> = sum |psi|^2 grad V * dV -- the Ehrenfest dipole acceleration
    // for the semiclassical radiated power. Forwarded to Observables.
    ses::Vec3d mean_force(Gl& gl) { return obs_.mean_force(gl, st_, st_.psi_buf_); }

    // <to|r|from> from two resident state buffers -- forwarded to Observables.
    ses::DipoleMatrixElement dipole_between(Gl& gl, GLuint to_buf, GLuint from_buf) {
        return obs_.dipole_between(gl, st_, to_buf, from_buf);
    }

    // psi <- psi - c * phi (complex c = (cre, cim)): projects phi out when
    // c is the overlap <phi|psi>.
    void subtract_projection(Gl& gl, GLuint phi_buf, double cre, double cim) {
        gl.glBindBufferBase(GL_SHADER_STORAGE_BUFFER, bind::kPsi, st_.psi_buf_);
        gl.glBindBufferBase(GL_SHADER_STORAGE_BUFFER, bind::kPhi, phi_buf);
        gl.glUseProgram(axpy_prog_);
        gl.glUniform1ui(gl.glGetUniformLocation(axpy_prog_, "n"),
                        static_cast<GLuint>(st_.cells_));
        gl.glUniform2f(gl.glGetUniformLocation(axpy_prog_, "c"), static_cast<float>(cre),
                       static_cast<float>(cim));
        gl.glDispatchCompute(static_cast<GLuint>((st_.cells_ + 255) / 256), 1, 1);
        gl.glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    }

    // Upload an auxiliary state (e.g. a cached eigenstate) into its own SSBO.
    // These are WRITE-ONCE -- a synthesized eigenstate never changes after
    // upload -- so they use GL_STATIC_COPY, not the engine's GL_DYNAMIC_COPY.
    // The dynamic hint makes drivers keep a system-memory shadow for fast
    // re-uploads; with 91 resident 134 MB buffers that shadow ~doubles the host
    // footprint (~12 GB VRAM -> ~24 GB committed). Static keeps them GPU-side.
    GLuint create_state_buffer(Gl& gl, const ses::Field3D& state) {
        return make_static_buffer(gl, to_rg32f(state.data()));
    }

    // ---- fp16 (half) atlas storage -------------------------------------
    // A small-VRAM fallback: store an eigenstate as packed fp16 (uint per
    // complex cell, HALF the fp32 footprint) so the whole n<=6 manifold fits an
    // 8 GB card. The tested fp32 consumers (inner product / dipole) never
    // change -- an fp16 state is unpacked to a scratch fp32 buffer on demand
    // (decode-on-use). Accuracy: ~1e-3 relative, safe for populations/dipoles
    // (the h-audited s-states stay fp32; see the shell).

    // Allocate a write-once fp16 state buffer (cells uints = half the fp32 size).
    GLuint make_half_state_buffer(Gl& gl) {
        return orbital_.make_half_state_buffer(gl, st_.cells_);
    }

    // dst_half <- packHalf2x16(src_fp32).
    void pack_to_half(Gl& gl, GLuint src_fp32, GLuint dst_half) {
        orbital_.pack_to_half(gl, src_fp32, dst_half, st_.cells_);
    }

    // dst_fp32 <- unpackHalf2x16(src_half).
    void unpack_from_half(Gl& gl, GLuint src_half, GLuint dst_fp32) {
        orbital_.unpack_from_half(gl, src_half, dst_fp32, st_.cells_);
    }

    // psi <- unpackHalf2x16(src_half): collapse onto an fp16-stored eigenstate.
    void unpack_into_psi(Gl& gl, GLuint src_half) {
        gl.glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
        unpack_from_half(gl, src_half, st_.psi_buf_);
    }

    // Synthesize psi = (u/r) Y_lm (normalized) directly INTO st_.psi_buf_ -- the
    // quantum-jump / measurement collapse onto an eigenstate WITHOUT a resident
    // atlas (a single transient orbital, built + copied + freed on demand).
    void synthesize_into_psi(Gl& gl, const std::vector<double>& u, int l, int m,
                             double h_radial, double rmax, int n_radial) {
        const GLuint tmp = synthesize_state(gl, u, l, m, h_radial, rmax, n_radial);
        copy_into_psi(gl, tmp);
        gl.glDeleteBuffers(1, &tmp);
    }

    // Synthesize psi = (u/r) Y_lm straight into a resident fp16 buffer: build +
    // normalize in fp32 (the tested path), then pack to half and free the fp32
    // temp. HALF the resident footprint. *out_peak (if given) is the fp32
    // normalized peak (pre-pack), matching synthesize_state.
    GLuint synthesize_state_half(Gl& gl, const std::vector<double>& u, int l, int m,
                                 double h_radial, double rmax, int n_radial,
                                 double* out_peak = nullptr, double* out_norm2 = nullptr) {
        return orbital_.synthesize_state_half(gl, st_, u, l, m, h_radial, rmax, n_radial,
                                              out_peak, out_norm2);
    }

    // ---- precision-aware consumers (decode-on-use) ---------------------
    // A StateHandle carries the buffer AND its precision, so these single
    // overloads cover both: an fp16 operand is unpacked to a scratch fp32
    // buffer (decode-on-use), then the SAME tested fp32 kernel runs.

    GLuint decode(Gl& gl, StateHandle s, GLuint& scratch) {
        if (!s.fp16) {
            return s.buf;
        }
        unpack_from_half(gl, s.buf, ensure_scratch(gl, scratch));
        return scratch;
    }

    NormPeak inner_with_psi(Gl& gl, StateHandle s) {
        return inner_with_psi(gl, decode(gl, s, scratch_a_));
    }

    // <to|r|from> with either operand possibly fp16 (the channel table pairs an
    // fp16 p-state with an fp32 s-state, so mixed precision must work).
    ses::DipoleMatrixElement dipole_between(Gl& gl, StateHandle to, StateHandle from) {
        const GLuint tb = decode(gl, to, scratch_a_);
        const GLuint fb = decode(gl, from, scratch_b_);
        return dipole_between(gl, tb, fb);
    }

    // ---- orbital-free projection -- forwarded to the Projector concern module.
    void set_projection_index(Gl& gl, const std::vector<std::uint32_t>& sorted_cell,
                              const std::vector<std::uint32_t>& bin_off, int n_radial,
                              double h_radial, int l_max) {
        projector_.set_index(gl, sorted_cell, bin_off, n_radial, h_radial, l_max);
    }
    void project_psi(Gl& gl) { projector_.project(gl, st_.psi_buf_, st_.grid_); }
    ses::Complex<double> project_amplitude(const std::vector<double>& u, int l, int m) const {
        return projector_.amplitude(u, l, m);
    }

    // Upload the radial u_nl(r) table (fp32) that kSynthSrc interpolates.
    void upload_radial(Gl& gl, const std::vector<double>& u) { orbital_.upload_radial(gl, u); }

    // Synthesize psi = (u/r) Y_lm INTO A NEW normalized state buffer on the GPU
    // -- forwarded to the OrbitalSynth concern module (gpu_orbital.hpp). Never
    // touches st_.psi_buf_, so it is safe to call mid-simulation.
    GLuint synthesize_state(Gl& gl, const std::vector<double>& u, int l, int m,
                            double h_radial, double rmax, int n_radial,
                            double* out_peak = nullptr, double* out_norm2 = nullptr) {
        return orbital_.synthesize_state(gl, st_, u, l, m, h_radial, rmax, n_radial,
                                         out_peak, out_norm2);
    }

    // psi <- contents of another state buffer (e.g. the quantum-jump
    // collapse onto a cached eigenstate).
    void copy_into_psi(Gl& gl, GLuint src_buf) {
        gl.glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
        gl.glBindBuffer(GL_COPY_READ_BUFFER, src_buf);
        gl.glBindBuffer(GL_COPY_WRITE_BUFFER, st_.psi_buf_);
        gl.glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0,
                               static_cast<GLsizeiptr>(2 * st_.cells_ * sizeof(float)));
    }

    // GPU-reduced norm (sum |psi|^2 * dV) and peak density -- a 2 KB readback
    // instead of the full field.
    NormPeak norm_and_peak(Gl& gl) {
        gl.glBindBufferBase(GL_SHADER_STORAGE_BUFFER, bind::kPsi, st_.psi_buf_);
        NormPeak np = run_norm_peak(gl, st_.norm_prog_, st_.partials_buf_, st_.cells_);
        np.sum *= st_.grid_.cell_volume();
        return np;
    }

    // psi <- s * psi (fp32 drift renormalization).
    void scale(Gl& gl, float s) {
        gl.glBindBufferBase(GL_SHADER_STORAGE_BUFFER, bind::kPsi, st_.psi_buf_);
        run_scale(gl, st_.scale_prog_, st_.cells_, s);
    }

    void upload_state(Gl& gl, const ses::Field3D& psi) {
        const std::vector<float> staged = to_rg32f(psi.data());
        // Order this API write against any in-flight shader writes from
        // earlier frames (GL 4.3: SSBO shader writes are incoherent, and the
        // per-dispatch SHADER_STORAGE barrier only covers shader-to-shader).
        gl.glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
        gl.glBindBuffer(GL_SHADER_STORAGE_BUFFER, st_.psi_buf_);
        gl.glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                           static_cast<GLsizeiptr>(staged.size() * sizeof(float)),
                           staged.data());
    }

    // Interleaved RG floats, 2 per cell.
    void readback(Gl& gl, std::vector<float>& out) const {
        out.resize(2 * st_.cells_);
        // Make incoherent compute writes visible to the buffer-update API
        // before glGetBufferSubData (spec-required; drivers often forgive).
        gl.glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
        gl.glBindBuffer(GL_SHADER_STORAGE_BUFFER, st_.psi_buf_);
        gl.glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                              static_cast<GLsizeiptr>(out.size() * sizeof(float)),
                              out.data());
    }

    // Strang step, fully GPU-resident:
    //   psi <- halfV . IFFT . kinetic . FFT . halfV psi
    // with IFFT = conj -> FFT -> conj/N (the same identity as the CPU core).
    // psi <- psi * mask, with the mask stored as a (real, 0) complex buffer so
    // the tested elementwise multiply applies a real per-cell damping. Binds
    // psi first, so it is safe to call standalone (the boundary absorber).
    void apply_mask(Gl& gl, GLuint mask_buf) {
        gl.glBindBufferBase(GL_SHADER_STORAGE_BUFFER, bind::kPsi, st_.psi_buf_);
        multiply(gl, mask_buf);
    }

    void step(Gl& gl, int nsteps) {
        gl.glBindBufferBase(GL_SHADER_STORAGE_BUFFER, bind::kPsi, st_.psi_buf_);
        for (int s = 0; s < nsteps; ++s) {
            multiply(gl, half_buf_);
            run_fft3(gl, st_.grid_, st_.fft_progs_);
            multiply(gl, kin_buf_);
            run_conj_scale(gl, st_.conj_prog_, st_.cells_, 1.0f);
            run_fft3(gl, st_.grid_, st_.fft_progs_);
            run_conj_scale(gl, st_.conj_prog_, st_.cells_, 1.0f / static_cast<float>(st_.cells_));
            multiply(gl, half_buf_);
        }
    }

    // Exact three-shear rotation of psi about coordinate `axis` by theta (the
    // magnetic paramagnetic (B/2)L_axis term), GPU transcription of
    // core/rotation.hpp rotate_axis. In place.
    void rotate_axis_shear(Gl& gl, int axis, double theta) {
        gl.glBindBufferBase(GL_SHADER_STORAGE_BUFFER, bind::kPsi, st_.psi_buf_);
        const int b = (axis + 1) % 3;  // in-plane axes (b x c = axis)
        const int c = (axis + 2) % 3;
        const double t = std::tan(0.5 * theta);
        const double s = std::sin(theta);
        apply_shear(gl, b, c, -t);
        apply_shear(gl, c, b, s);
        apply_shear(gl, b, c, -t);
    }
    void rotate_z_shear(Gl& gl, double theta) { rotate_axis_shear(gl, 2, theta); }

    // Magnetic Strang step: R(a) . core-step . R(a), a = (B/2)(dt/2), rotating
    // about the field `axis`. The core step uses half_buf_, which the caller
    // must have set to the diamagnetic-augmented V + (B^2/8)rho_perp^2
    // (set_half_potential). Chained half-rotations between steps merge into
    // the per-step Larmor angle (B/2) dt.
    void magnetic_step(Gl& gl, int axis, double half_angle, int nsteps) {
        for (int s = 0; s < nsteps; ++s) {
            rotate_axis_shear(gl, axis, half_angle);
            step(gl, 1);
            rotate_axis_shear(gl, axis, half_angle);
        }
    }

    // One dipole half-kick on psi (T3). theta = E0 cos(omega t) dt/2 is
    // computed by the caller in double (core/drive.hpp's formula); the
    // kernel only does the pointwise complex rotation.
    void dipole_kick(Gl& gl, const ses::Vec3d& axis, double theta) {
        gl.glBindBufferBase(GL_SHADER_STORAGE_BUFFER, bind::kPsi, st_.psi_buf_);
        gl.glUseProgram(kick_prog_);
        gl.glUniform1ui(gl.glGetUniformLocation(kick_prog_, "n"),
                        static_cast<GLuint>(st_.cells_));
        gl.glUniform1i(gl.glGetUniformLocation(kick_prog_, "nx"), st_.grid_.x.n);
        gl.glUniform1i(gl.glGetUniformLocation(kick_prog_, "ny"), st_.grid_.y.n);
        gl.glUniform3f(gl.glGetUniformLocation(kick_prog_, "box_min"),
                       static_cast<float>(st_.grid_.x.xmin), static_cast<float>(st_.grid_.y.xmin),
                       static_cast<float>(st_.grid_.z.xmin));
        gl.glUniform3f(gl.glGetUniformLocation(kick_prog_, "cell_h"),
                       static_cast<float>(st_.grid_.x.spacing()),
                       static_cast<float>(st_.grid_.y.spacing()),
                       static_cast<float>(st_.grid_.z.spacing()));
        gl.glUniform3f(gl.glGetUniformLocation(kick_prog_, "axis"),
                       static_cast<float>(axis.x), static_cast<float>(axis.y),
                       static_cast<float>(axis.z));
        gl.glUniform1f(gl.glGetUniformLocation(kick_prog_, "theta"),
                       static_cast<float>(theta));
        gl.glDispatchCompute(static_cast<GLuint>((st_.cells_ + 255) / 256), 1, 1);
        gl.glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    }

    // Driven Strang steps (T3): the tested core/drive.hpp composition
    //   psi <- kick(t+dt) . halfV . IFFT . kinetic . FFT . halfV . kick(t) psi
    // around the untouched static tables; t0 is the physical time at the
    // start of the first step, dt must match the tables' time step.
    void driven_step(Gl& gl, const ses::DipoleDrive& d, double t0, double dt, int nsteps) {
        gl.glBindBufferBase(GL_SHADER_STORAGE_BUFFER, bind::kPsi, st_.psi_buf_);
        for (int s = 0; s < nsteps; ++s) {
            const double t = t0 + s * dt;
            dipole_kick(gl, d.axis, d.amplitude * std::cos(d.omega * t) * 0.5 * dt);
            multiply(gl, half_buf_);
            run_fft3(gl, st_.grid_, st_.fft_progs_);
            multiply(gl, kin_buf_);
            run_conj_scale(gl, st_.conj_prog_, st_.cells_, 1.0f);
            run_fft3(gl, st_.grid_, st_.fft_progs_);
            run_conj_scale(gl, st_.conj_prog_, st_.cells_, 1.0f / static_cast<float>(st_.cells_));
            multiply(gl, half_buf_);
            dipole_kick(gl, d.axis,
                        d.amplitude * std::cos(d.omega * (t + dt)) * 0.5 * dt);
        }
    }

    // Write psi into the volume renderer's RG32F 3D texture, GPU-to-GPU.
    void write_psi_texture(Gl& gl, GLuint texture) {
        write_texture_from(gl, st_.psi_buf_, texture);
    }

    // Bridge an SSBO (psi or a rotated copy) into the RG32F volume texture.
    void write_texture_from(Gl& gl, GLuint src_buf, GLuint texture) {
        gl.glBindBufferBase(GL_SHADER_STORAGE_BUFFER, bind::kPsi, src_buf);
        gl.glBindImageTexture(0, texture, 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_RG32F);
        gl.glUseProgram(st_.bridge_prog_);
        gl.glUniform1i(gl.glGetUniformLocation(st_.bridge_prog_, "nx"), st_.grid_.x.n);
        gl.glUniform1i(gl.glGetUniformLocation(st_.bridge_prog_, "ny"), st_.grid_.y.n);
        gl.glUniform1i(gl.glGetUniformLocation(st_.bridge_prog_, "nz"), st_.grid_.z.n);
        gl.glDispatchCompute(static_cast<GLuint>((st_.cells_ + 255) / 256), 1, 1);
        gl.glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT | GL_TEXTURE_UPDATE_BARRIER_BIT);
    }


private:
    const ses::Grid1D& axis_grid(int a) const {
        return a == 0 ? st_.grid_.x : (a == 1 ? st_.grid_.y : st_.grid_.z);
    }

    // One shear of the three-shear rotation: FFT along freq_axis (0=x,1=y,2=z),
    // shift each line by coeff*(its coord_axis coordinate) via the phase kernel,
    // then inverse FFT along that axis (conj -> fft -> conj/N). In place on
    // whatever is bound at binding 0 (rotate_axis_shear binds psi).
    void apply_shear(Gl& gl, int freq_axis, int coord_axis, double coeff) {
        run_fft_axis(gl, st_.grid_, st_.fft_progs_, freq_axis);
        const ses::Grid1D& fa = axis_grid(freq_axis);
        const ses::Grid1D& ca = axis_grid(coord_axis);
        const double two_pi = 6.283185307179586;
        gl.glUseProgram(shear_prog_);
        gl.glUniform1ui(gl.glGetUniformLocation(shear_prog_, "n"),
                        static_cast<GLuint>(st_.cells_));
        gl.glUniform1i(gl.glGetUniformLocation(shear_prog_, "nx"), st_.grid_.x.n);
        gl.glUniform1i(gl.glGetUniformLocation(shear_prog_, "ny"), st_.grid_.y.n);
        gl.glUniform1i(gl.glGetUniformLocation(shear_prog_, "nz"), st_.grid_.z.n);
        gl.glUniform1i(gl.glGetUniformLocation(shear_prog_, "freq_axis"), freq_axis);
        gl.glUniform1i(gl.glGetUniformLocation(shear_prog_, "coord_axis"), coord_axis);
        gl.glUniform1i(gl.glGetUniformLocation(shear_prog_, "nf"), fa.n);
        gl.glUniform1f(gl.glGetUniformLocation(shear_prog_, "kscale"),
                       static_cast<float>(two_pi / (fa.xmax - fa.xmin)));
        gl.glUniform1f(gl.glGetUniformLocation(shear_prog_, "cmin"),
                       static_cast<float>(ca.xmin));
        gl.glUniform1f(gl.glGetUniformLocation(shear_prog_, "ch"),
                       static_cast<float>(ca.spacing()));
        gl.glUniform1f(gl.glGetUniformLocation(shear_prog_, "coeff"),
                       static_cast<float>(coeff));
        gl.glDispatchCompute(static_cast<GLuint>((st_.cells_ + 255) / 256), 1, 1);
        gl.glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        run_conj_scale(gl, st_.conj_prog_, st_.cells_, 1.0f);
        run_fft_axis(gl, st_.grid_, st_.fft_progs_, freq_axis);
        run_conj_scale(gl, st_.conj_prog_, st_.cells_, 1.0f / static_cast<float>(fa.n));
    }

    // One e^{-H dtau} Strang pass over psi (no normalization).
    void strang_imaginary(Gl& gl) {
        multiply(gl, relax_half_buf_);
        run_fft3(gl, st_.grid_, st_.fft_progs_);
        multiply(gl, relax_kin_buf_);
        run_conj_scale(gl, st_.conj_prog_, st_.cells_, 1.0f);
        run_fft3(gl, st_.grid_, st_.fft_progs_);
        run_conj_scale(gl, st_.conj_prog_, st_.cells_, 1.0f / static_cast<float>(st_.cells_));
        multiply(gl, relax_half_buf_);
    }

    RelaxStats renormalize_and_estimate(Gl& gl) {
        const NormPeak np = run_norm_peak(gl, st_.norm_prog_, st_.partials_buf_, st_.cells_);
        const double norm_sq = np.sum * st_.grid_.cell_volume();
        RelaxStats stats;
        stats.energy = -std::log(norm_sq) / (2.0 * relax_dtau_);
        stats.peak = np.peak / norm_sq;
        run_scale(gl, st_.scale_prog_, st_.cells_, static_cast<float>(1.0 / std::sqrt(norm_sq)));
        return stats;
    }

    // Lazily allocate a reusable fp32 scratch buffer (for unpacking an fp16
    // state before a tested fp32 consumer reads it). Never uploaded from host.
    GLuint ensure_scratch(Gl& gl, GLuint& slot) {
        if (slot == 0) {
            gl.glGenBuffers(1, &slot);
            gl.glBindBuffer(GL_SHADER_STORAGE_BUFFER, slot);
            gl.glBufferData(GL_SHADER_STORAGE_BUFFER,
                            static_cast<GLsizeiptr>(2 * st_.cells_ * sizeof(float)), nullptr,
                            GL_DYNAMIC_COPY);
        }
        return slot;
    }

    void multiply(Gl& gl, GLuint phase_buf) {
        gl.glBindBufferBase(GL_SHADER_STORAGE_BUFFER, bind::kPhase, phase_buf);
        gl.glUseProgram(st_.mul_prog_);
        gl.glUniform1ui(gl.glGetUniformLocation(st_.mul_prog_, "n"),
                        static_cast<GLuint>(st_.cells_));
        gl.glDispatchCompute(static_cast<GLuint>((st_.cells_ + 255) / 256), 1, 1);
        gl.glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    }

    GpuState st_;  // shared spine (grid, psi, core programs)
    GLuint half_buf_ = 0;
    GLuint kin_buf_ = 0;
    GLuint relax_half_buf_ = 0;
    GLuint relax_kin_buf_ = 0;
    double relax_dtau_ = 0.0;
    GLuint axpy_prog_ = 0;
    GLuint kick_prog_ = 0;
    GLuint shear_prog_ = 0;
    GLuint scratch_a_ = 0;  // fp32 unpack scratch for fp16 consumers
    GLuint scratch_b_ = 0;  // second scratch (dipole needs two operands)
    Observables obs_;           // <a|b> / mean_force / dipole (gpu_observables.hpp)
    OrbitalSynth orbital_;      // orbital synthesis + fp16 codec (gpu_orbital.hpp)
    Projector projector_;       // orbital-free angular projection (gpu_projection.hpp)
};

}  // namespace ses_gpu
