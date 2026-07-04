// sesolver_gpucheck -- the verification harness for GPU compute kernels
// (docs/GPU_PLAN.md). Compute shaders cannot be gtest-unit-tested from the
// pure core, so each kernel is compared here, element by element, against
// the unit-tested CPU double implementation on deterministic inputs.
//
// Exit codes: 0 = all kernels match, 1 = mismatch, 77 = no GL 4.3 context
// (registered to ctest as SKIP).
//
// G1 kernel: in-place pointwise complex multiply -- the e^{-iVdt/2} and
// e^{-ik^2dt/2} phase applications of the split-operator step.

#include <core/complex.hpp>

#include <QGuiApplication>
#include <QOffscreenSurface>
#include <QOpenGLContext>
#include <QOpenGLFunctions_4_3_Core>
#include <QSurfaceFormat>

#include <cmath>
#include <cstdio>
#include <cstddef>
#include <vector>

namespace {

constexpr int kSkipExitCode = 77;

const char* kPhaseMultiplySrc = R"(#version 430 core
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

struct Gl : QOpenGLFunctions_4_3_Core {};

GLuint compile_compute(Gl& gl, const char* src) {
    const GLuint shader = gl.glCreateShader(GL_COMPUTE_SHADER);
    gl.glShaderSource(shader, 1, &src, nullptr);
    gl.glCompileShader(shader);
    GLint ok = GL_FALSE;
    gl.glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (ok != GL_TRUE) {
        char log[2048];
        gl.glGetShaderInfoLog(shader, sizeof(log), nullptr, log);
        std::fprintf(stderr, "compute shader compile failed:\n%s\n", log);
        std::exit(1);
    }
    const GLuint prog = gl.glCreateProgram();
    gl.glAttachShader(prog, shader);
    gl.glLinkProgram(prog);
    gl.glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (ok != GL_TRUE) {
        char log[2048];
        gl.glGetProgramInfoLog(prog, sizeof(log), nullptr, log);
        std::fprintf(stderr, "compute program link failed:\n%s\n", log);
        std::exit(1);
    }
    gl.glDeleteShader(shader);
    return prog;
}

GLuint make_ssbo(Gl& gl, GLuint binding, const std::vector<float>& data) {
    GLuint buf = 0;
    gl.glGenBuffers(1, &buf);
    gl.glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding, buf);
    gl.glBufferData(GL_SHADER_STORAGE_BUFFER,
                    static_cast<GLsizeiptr>(data.size() * sizeof(float)), data.data(),
                    GL_DYNAMIC_COPY);
    return buf;
}

// G1: GPU pointwise complex multiply vs the CPU double reference.
bool check_phase_multiply(Gl& gl) {
    const std::size_t n = 4096;

    // Deterministic inputs spanning magnitudes and all phase quadrants.
    std::vector<float> psi_f(2 * n);
    std::vector<float> phase_f(2 * n);
    std::vector<ses::Complex<double>> psi_d(n);
    std::vector<ses::Complex<double>> phase_d(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double x = static_cast<double>(i);
        psi_d[i] = ses::Complex<double>{std::sin(0.37 * x) + 0.2, std::cos(1.13 * x) - 0.1};
        const double th = 2.9 * x;
        phase_d[i] = ses::Complex<double>{std::cos(th), std::sin(th)};
        psi_f[2 * i] = static_cast<float>(psi_d[i].re);
        psi_f[2 * i + 1] = static_cast<float>(psi_d[i].im);
        phase_f[2 * i] = static_cast<float>(phase_d[i].re);
        phase_f[2 * i + 1] = static_cast<float>(phase_d[i].im);
    }

    const GLuint prog = compile_compute(gl, kPhaseMultiplySrc);
    const GLuint psi_buf = make_ssbo(gl, 0, psi_f);
    make_ssbo(gl, 1, phase_f);

    gl.glUseProgram(prog);
    gl.glUniform1ui(gl.glGetUniformLocation(prog, "n"), static_cast<GLuint>(n));
    gl.glDispatchCompute(static_cast<GLuint>((n + 255) / 256), 1, 1);
    gl.glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);

    std::vector<float> out(2 * n);
    gl.glBindBuffer(GL_SHADER_STORAGE_BUFFER, psi_buf);
    gl.glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                          static_cast<GLsizeiptr>(out.size() * sizeof(float)), out.data());

    double max_err = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const ses::Complex<double> expected = psi_d[i] * phase_d[i];
        max_err = std::max(max_err, std::abs(out[2 * i] - expected.re));
        max_err = std::max(max_err, std::abs(out[2 * i + 1] - expected.im));
    }
    // Values are O(1); fp32 input quantization alone is ~1e-7, product ~1e-6.
    const bool pass = max_err < 1e-5;
    std::printf("phase-multiply kernel: max |gpu - cpu| = %.3e  [%s]\n", max_err,
                pass ? "PASS" : "FAIL");
    return pass;
}

}  // namespace

int main(int argc, char** argv) {
    QGuiApplication app(argc, argv);

    QSurfaceFormat format;
    format.setVersion(4, 3);
    format.setProfile(QSurfaceFormat::CoreProfile);

    QOpenGLContext ctx;
    ctx.setFormat(format);
    QOffscreenSurface surface;
    surface.setFormat(format);
    surface.create();
    if (!ctx.create() || !ctx.makeCurrent(&surface)) {
        std::printf("no OpenGL context available - SKIP\n");
        return kSkipExitCode;
    }
    if (ctx.format().majorVersion() < 4 ||
        (ctx.format().majorVersion() == 4 && ctx.format().minorVersion() < 3)) {
        std::printf("OpenGL %d.%d < 4.3 - SKIP\n", ctx.format().majorVersion(),
                    ctx.format().minorVersion());
        return kSkipExitCode;
    }

    Gl gl;
    if (!gl.initializeOpenGLFunctions()) {
        std::printf("could not resolve GL 4.3 functions - SKIP\n");
        return kSkipExitCode;
    }

    const bool ok = check_phase_multiply(gl);
    return ok ? 0 : 1;
}
