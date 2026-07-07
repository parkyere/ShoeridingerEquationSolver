// sesolver_gpucheck -- the verification harness for GPU compute kernels
// (docs/GPU_PLAN.md). Compute shaders cannot be gtest-unit-tested from the
// pure core, so every kernel -- and the production GpuEngine composition --
// is compared here against the unit-tested CPU double implementation on
// deterministic inputs. The kernels under test are the SAME sources the app
// uses (ses_gpu, single source of truth).
//
// Exit codes: 0 = all match, 1 = mismatch, 77 = no GL 4.3 context (SKIP).
//
// Checks:
//  G1: pointwise complex phase multiply;
//  G2+G3: 3-axis line FFT vs CPU fft3 on 16x8x4 (distinct dims: axis
//         mapping) and 64^3, plus GPU inverse round-trip;
//  G4: GpuEngine::step x20 vs SplitOperator3D::step x20 (soft Coulomb),
//      plus the SSBO -> RG32F texture bridge;
//  G6: norm/peak tree reduction vs CPU double sums, and the scale kernel
//      (fp32 renormalization) scaling the reduced norm exactly by s^2;
//  G7: GPU imaginary-time relax x50 vs ImaginaryTimePropagator3D x50, and
//      the free ITP energy estimator converging to the harmonic 3w/2;
//  T1: complex inner-product reduction vs ses::inner_product, and the
//      deflated relax vs the CPU relax_deflated, converging to E1 = 5w/2;
//  T3: the dipole-kick kernel vs apply_dipole_halfkick, and the driven
//      engine step x20 vs core driven_step (skew axis, nonzero t0).

#include "gpu_engine.hpp"

#include <core/complex.hpp>
#include <core/drive.hpp>
#include <core/fft.hpp>
#include <core/field.hpp>
#include <core/grid.hpp>
#include <core/imaginary_time.hpp>
#include <core/magnetic.hpp>
#include <core/potential.hpp>
#include <core/propagator.hpp>
#include <core/sampling.hpp>
#include <core/vec.hpp>
#include <core/wavepacket.hpp>

#include <QGuiApplication>
#include <QOffscreenSurface>
#include <QOpenGLContext>
#include <QSurfaceFormat>

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <string>
#include <vector>

namespace {

constexpr int kSkipExitCode = 77;

using ses_gpu::Gl;

GLuint make_ssbo(Gl& gl, GLuint binding, const std::vector<float>& data) {
    GLuint buf = 0;
    gl.glGenBuffers(1, &buf);
    gl.glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding, buf);
    gl.glBufferData(GL_SHADER_STORAGE_BUFFER,
                    static_cast<GLsizeiptr>(data.size() * sizeof(float)), data.data(),
                    GL_DYNAMIC_COPY);
    return buf;
}

std::vector<float> read_ssbo(Gl& gl, GLuint buf, std::size_t floats) {
    std::vector<float> out(floats);
    // Spec-required visibility of incoherent compute writes to the
    // buffer-update API (matches the production readback path).
    gl.glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
    gl.glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf);
    gl.glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                          static_cast<GLsizeiptr>(floats * sizeof(float)), out.data());
    return out;
}

ses::Field3D deterministic_field(const ses::Grid3D& g) {
    ses::Field3D f{g};
    for (int i = 0; i < f.size(); ++i) {
        const double x = static_cast<double>(i);
        f.data()[static_cast<std::size_t>(i)] =
            ses::Complex<double>{std::sin(0.61 * x) + 0.15, std::cos(1.27 * x) - 0.2};
    }
    return f;
}

bool compare(const char* label, const std::vector<float>& gpu, const ses::Field3D& cpu,
             double abs_tol) {
    double max_err = 0.0;
    double max_mag = 0.0;
    for (std::size_t i = 0; i < cpu.data().size(); ++i) {
        max_err = std::max(max_err, std::abs(gpu[2 * i] - cpu.data()[i].real()));
        max_err = std::max(max_err, std::abs(gpu[2 * i + 1] - cpu.data()[i].imag()));
        max_mag = std::max(max_mag, std::abs(cpu.data()[i].real()));
        max_mag = std::max(max_mag, std::abs(cpu.data()[i].imag()));
    }
    const double tol = abs_tol + 1e-5 * max_mag;
    const bool pass = max_err < tol;
    std::printf("%s: max |gpu - cpu| = %.3e (tol %.3e)  [%s]\n", label, max_err, tol,
                pass ? "PASS" : "FAIL");
    return pass;
}

// G1: pointwise complex multiply vs CPU doubles.
bool check_phase_multiply(Gl& gl) {
    const std::size_t n = 4096;
    std::vector<ses::Complex<double>> psi_d(n);
    std::vector<ses::Complex<double>> phase_d(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double x = static_cast<double>(i);
        psi_d[i] = ses::Complex<double>{std::sin(0.37 * x) + 0.2, std::cos(1.13 * x) - 0.1};
        phase_d[i] = ses::Complex<double>{std::cos(2.9 * x), std::sin(2.9 * x)};
    }

    const GLuint prog = ses_gpu::build_program(gl, ses_gpu::kPhaseMultiplySrc, "G1");
    const GLuint psi_buf = make_ssbo(gl, 0, ses_gpu::to_rg32f(psi_d));
    make_ssbo(gl, 1, ses_gpu::to_rg32f(phase_d));
    gl.glUseProgram(prog);
    gl.glUniform1ui(gl.glGetUniformLocation(prog, "n"), static_cast<GLuint>(n));
    gl.glDispatchCompute(static_cast<GLuint>((n + 255) / 256), 1, 1);
    gl.glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);

    const std::vector<float> out = read_ssbo(gl, psi_buf, 2 * n);
    double max_err = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const ses::Complex<double> expected = psi_d[i] * phase_d[i];
        max_err = std::max(max_err, std::abs(out[2 * i] - expected.real()));
        max_err = std::max(max_err, std::abs(out[2 * i + 1] - expected.imag()));
    }
    const bool pass = max_err < 1e-5;
    std::printf("phase-multiply kernel: max |gpu - cpu| = %.3e  [%s]\n", max_err,
                pass ? "PASS" : "FAIL");
    return pass;
}

// G2+G3: GPU 3-axis FFT vs CPU double fft3, plus GPU inverse round-trip.
bool check_fft3(Gl& gl, const ses::Grid3D& g, const char* label) {
    const ses::Field3D original = deterministic_field(g);
    const std::size_t cells = original.data().size();

    const GLuint buf = make_ssbo(gl, 0, ses_gpu::to_rg32f(original.data()));
    const ses_gpu::FftPrograms progs = ses_gpu::build_fft_programs(gl, g);
    const GLuint conj_prog = ses_gpu::build_program(gl, ses_gpu::kConjScaleSrc, "conj");

    ses_gpu::run_fft3(gl, g, progs);
    ses::Field3D cpu = original;
    ses::fft(cpu);
    const std::string fwd = std::string{label} + " forward";
    bool ok = compare(fwd.c_str(), read_ssbo(gl, buf, 2 * cells), cpu, 1e-4);

    ses_gpu::run_conj_scale(gl, conj_prog, cells, 1.0f);
    ses_gpu::run_fft3(gl, g, progs);
    ses_gpu::run_conj_scale(gl, conj_prog, cells, 1.0f / static_cast<float>(cells));
    const std::string rt = std::string{label} + " round-trip";
    ok = compare(rt.c_str(), read_ssbo(gl, buf, 2 * cells), original, 1e-4) && ok;

    gl.glDeleteBuffers(1, &buf);
    return ok;
}

// G4: the production GpuEngine, stepped 20x, vs the CPU propagator, plus the
// SSBO -> RG32F texture bridge the volume renderer consumes.
bool check_engine_step(Gl& gl) {
    const ses::Grid1D axis{-8.0, 8.0, 32};
    const ses::Grid3D g{axis, axis, axis};
    const std::vector<double> v = ses::soft_coulomb_potential(g, 1.0, 1.0, ses::Vec3d{});
    const double dt = 0.02;
    const ses::SplitOperator3D cpu_prop{g, v, dt};

    ses::Field3D psi0 = ses::gaussian_wavepacket(g, ses::Vec3d{2.0, 0.0, 0.0},
                                                 ses::Vec3d{1.2, 1.2, 1.2},
                                                 ses::Vec3d{0.0, 0.5, 0.0});

    ses_gpu::GpuEngine engine;
    if (!engine.initialize(gl, g, cpu_prop.half_potential_phase(), cpu_prop.kinetic_phase(),
                           psi0)) {
        std::printf("engine init: FAIL\n");
        return false;
    }
    engine.step(gl, 20);

    ses::Field3D cpu = psi0;
    cpu_prop.step(cpu, 20);

    std::vector<float> gpu_out;
    engine.readback(gl, gpu_out);
    bool ok = compare("engine 20 steps", gpu_out, cpu, 1e-4);

    // Bridge: the texture must contain exactly the SSBO contents.
    GLuint tex = 0;
    gl.glGenTextures(1, &tex);
    gl.glBindTexture(GL_TEXTURE_3D, tex);
    // Without these the default mipmapping MIN_FILTER leaves the texture
    // INCOMPLETE and every imageStore is silently dropped (reads back zeros).
    gl.glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    gl.glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    gl.glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAX_LEVEL, 0);
    gl.glTexImage3D(GL_TEXTURE_3D, 0, GL_RG32F, g.x.n, g.y.n, g.z.n, 0, GL_RG, GL_FLOAT,
                    nullptr);
    engine.write_psi_texture(gl, tex);
    std::vector<float> tex_out(2 * gpu_out.size() / 2, 0.0f);
    gl.glBindTexture(GL_TEXTURE_3D, tex);
    gl.glGetTexImage(GL_TEXTURE_3D, 0, GL_RG, GL_FLOAT, tex_out.data());
    double bridge_err = 0.0;
    for (std::size_t i = 0; i < tex_out.size(); ++i) {
        bridge_err = std::max(bridge_err, static_cast<double>(std::abs(tex_out[i] - gpu_out[i])));
    }
    const bool bridge_ok = bridge_err == 0.0;  // pure copy: must be bitwise
    std::printf("texture bridge: max |tex - ssbo| = %.3e  [%s]\n", bridge_err,
                bridge_ok ? "PASS" : "FAIL");
    return ok && bridge_ok;
}

// G6: the norm/peak reduction and the scale kernel vs CPU double reference.
bool check_norm_reduction(Gl& gl) {
    const ses::Grid1D axis{-8.0, 8.0, 32};
    const ses::Grid3D g{axis, axis, axis};
    const ses::Field3D f = deterministic_field(g);
    const std::size_t cells = f.data().size();

    double cpu_sum = 0.0;
    double cpu_peak = 0.0;
    for (const ses::Complex<double>& z : f.data()) {
        const double d = ses::norm_sq(z);
        cpu_sum += d;
        cpu_peak = std::max(cpu_peak, d);
    }

    const GLuint norm_prog = ses_gpu::build_program(gl, ses_gpu::kNormPeakSrc, "G6 norm");
    const GLuint scale_prog = ses_gpu::build_program(gl, ses_gpu::kScaleSrc, "G6 scale");
    const GLuint psi_buf = make_ssbo(gl, 0, ses_gpu::to_rg32f(f.data()));
    const GLuint partials = make_ssbo(
        gl, 2, std::vector<float>(2 * ses_gpu::kNormPeakGroups, 0.0f));

    const ses_gpu::NormPeak np = ses_gpu::run_norm_peak(gl, norm_prog, partials, cells);
    const double sum_rel = std::abs(np.sum - cpu_sum) / cpu_sum;
    const double peak_rel = std::abs(np.peak - cpu_peak) / cpu_peak;
    bool ok = sum_rel < 1e-5 && peak_rel < 1e-6;
    std::printf("norm/peak reduce: rel err sum %.3e, peak %.3e  [%s]\n", sum_rel, peak_rel,
                ok ? "PASS" : "FAIL");

    // scale by s must scale the reduced |psi|^2 sum by s^2.
    gl.glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, psi_buf);
    ses_gpu::run_scale(gl, scale_prog, cells, 0.5f);
    const ses_gpu::NormPeak scaled = ses_gpu::run_norm_peak(gl, norm_prog, partials, cells);
    const double scale_rel = std::abs(scaled.sum - 0.25 * np.sum) / (0.25 * np.sum);
    const bool scale_ok = scale_rel < 1e-6;
    std::printf("scale kernel: rel err %.3e  [%s]\n", scale_rel, scale_ok ? "PASS" : "FAIL");

    gl.glDeleteBuffers(1, &psi_buf);
    return ok && scale_ok;
}

// G7: GPU imaginary-time relaxation vs the CPU relaxer, plus the free ITP
// energy estimator at convergence.
bool check_relax(Gl& gl) {
    const ses::Grid1D axis{-8.0, 8.0, 32};
    const ses::Grid3D g{axis, axis, axis};
    const std::vector<double> v = ses::harmonic_potential(g, 1.0, ses::Vec3d{});
    const double dtau = 0.05;
    const ses::SplitOperator3D real_prop{g, v, 0.02};  // engine init needs them
    const ses::ImaginaryTimePropagator3D cpu_relaxer{g, v, dtau};

    const ses::Field3D psi0 = ses::gaussian_wavepacket(
        g, ses::Vec3d{1.5, 0.0, 0.0}, ses::Vec3d{2.0, 2.0, 2.0}, ses::Vec3d{});

    ses_gpu::GpuEngine engine;
    if (!engine.initialize(gl, g, real_prop.half_potential_phase(),
                           real_prop.kinetic_phase(), psi0)) {
        std::printf("engine init: FAIL\n");
        return false;
    }
    engine.set_relax_tables(gl, cpu_relaxer.half_potential_weight(),
                            cpu_relaxer.kinetic_weight(), dtau);

    engine.relax_step(gl, 50);
    ses::Field3D cpu = psi0;
    cpu_relaxer.relax(cpu, 50);
    std::vector<float> gpu_out;
    engine.readback(gl, gpu_out);
    bool ok = compare("relax 50 steps", gpu_out, cpu, 1e-4);

    // Converge (tau = 30 total) and check the free energy estimator against
    // the known 3D harmonic ground energy 3w/2 = 1.5.
    const ses_gpu::GpuEngine::RelaxStats stats = engine.relax_step(gl, 550);
    const bool e_ok = std::abs(stats.energy - 1.5) < 0.02;
    std::printf("relax energy estimator: E = %.4f (expect 1.5)  [%s]\n", stats.energy,
                e_ok ? "PASS" : "FAIL");
    return ok && e_ok;
}

// T1-GPU: the inner-product kernel and the deflated relaxation vs CPU.
bool check_deflation(Gl& gl) {
    const ses::Grid1D axis{-8.0, 8.0, 32};
    const ses::Grid3D g{axis, axis, axis};
    const std::vector<double> v = ses::harmonic_potential(g, 1.0, ses::Vec3d{});
    const double dtau = 0.05;
    const ses::SplitOperator3D real_prop{g, v, 0.02};
    const ses::ImaginaryTimePropagator3D cpu_relaxer{g, v, dtau};

    // CPU ground state (the deflation target), plus a mixed-parity guess.
    ses::Field3D ground = ses::gaussian_wavepacket(g, ses::Vec3d{},
                                                   ses::Vec3d{1.2, 1.2, 1.2}, ses::Vec3d{});
    cpu_relaxer.relax(ground, 600);
    const ses::Field3D guess = ses::gaussian_wavepacket(
        g, ses::Vec3d{1.0, 0.5, 0.0}, ses::Vec3d{1.2, 1.2, 1.2}, ses::Vec3d{});

    ses_gpu::GpuEngine engine;
    if (!engine.initialize(gl, g, real_prop.half_potential_phase(),
                           real_prop.kinetic_phase(), guess)) {
        std::printf("engine init: FAIL\n");
        return false;
    }
    engine.set_relax_tables(gl, cpu_relaxer.half_potential_weight(),
                            cpu_relaxer.kinetic_weight(), dtau);
    const GLuint ground_buf = engine.create_state_buffer(gl, ground);

    // Inner-product kernel vs the CPU double reference.
    const ses::Complex<double> cpu_ip = ses::inner_product(ground, guess);
    const ses_gpu::NormPeak gpu_ip = engine.inner_with_psi(gl, ground_buf);
    const double ip_err = std::max(std::abs(gpu_ip.sum - cpu_ip.real()),
                                   std::abs(gpu_ip.peak - cpu_ip.imag()));
    const bool ip_ok = ip_err < 1e-6;
    std::printf("inner-product kernel: max err %.3e  [%s]\n", ip_err,
                ip_ok ? "PASS" : "FAIL");

    // Deflated relax: 50 GPU steps vs 50 CPU steps.
    engine.relax_deflated_step(gl, {ground_buf}, 50);
    ses::Field3D cpu = guess;
    cpu_relaxer.relax_deflated(cpu, {&ground}, 50);
    std::vector<float> gpu_out;
    engine.readback(gl, gpu_out);
    bool ok = compare("deflated relax 50 steps", gpu_out, cpu, 1e-4);

    // Convergence: the estimator must land on the first excited level 5w/2.
    const ses_gpu::GpuEngine::RelaxStats stats =
        engine.relax_deflated_step(gl, {ground_buf}, 550);
    const bool e_ok = std::abs(stats.energy - 2.5) < 0.02;
    std::printf("deflated energy estimator: E = %.4f (expect 2.5)  [%s]\n", stats.energy,
                e_ok ? "PASS" : "FAIL");

    // copy_into_psi (quantum-jump collapse path): bitwise buffer copy.
    engine.copy_into_psi(gl, ground_buf);
    std::vector<float> copied;
    engine.readback(gl, copied);
    const std::vector<float> ground_staged = ses_gpu::to_rg32f(ground.data());
    double copy_err = 0.0;
    for (std::size_t i = 0; i < copied.size(); ++i) {
        copy_err = std::max(copy_err,
                            static_cast<double>(std::abs(copied[i] - ground_staged[i])));
    }
    const bool copy_ok = copy_err == 0.0;
    std::printf("copy_into_psi: max err %.3e  [%s]\n", copy_err, copy_ok ? "PASS" : "FAIL");
    return ip_ok && ok && e_ok && copy_ok;
}

// T3-GPU: the dipole-kick kernel and the driven engine step vs core/drive.hpp.
// The drive is deliberately adversarial: skew (non-unit) polarization axis,
// nonzero frequency, and a nonzero start time exercise every uniform path.
bool check_driven_step(Gl& gl) {
    const ses::Grid1D axis{-8.0, 8.0, 32};
    const ses::Grid3D g{axis, axis, axis};
    const std::vector<double> v = ses::soft_coulomb_potential(g, 1.0, 1.0, ses::Vec3d{});
    const double dt = 0.02;
    const ses::SplitOperator3D cpu_prop{g, v, dt};
    const ses::DipoleDrive drive{ses::Vec3d{0.3, -0.2, 1.0}, 0.5, 0.6};

    const ses::Field3D psi0 = ses::gaussian_wavepacket(g, ses::Vec3d{2.0, 0.0, 0.0},
                                                       ses::Vec3d{1.2, 1.2, 1.2},
                                                       ses::Vec3d{0.0, 0.5, 0.0});

    ses_gpu::GpuEngine engine;
    if (!engine.initialize(gl, g, cpu_prop.half_potential_phase(), cpu_prop.kinetic_phase(),
                           psi0)) {
        std::printf("engine init: FAIL\n");
        return false;
    }

    // Kick kernel alone vs the CPU half-kick at a fixed edge time.
    const double t_kick = 0.37;
    engine.dipole_kick(gl, drive.axis,
                       drive.amplitude * std::cos(drive.omega * t_kick) * 0.5 * dt);
    ses::Field3D cpu = psi0;
    ses::apply_dipole_halfkick(cpu, drive, t_kick, dt);
    std::vector<float> gpu_out;
    engine.readback(gl, gpu_out);
    bool ok = compare("dipole kick kernel", gpu_out, cpu, 1e-5);

    // Full driven composition, 20 steps from t0 = 1.3.
    engine.upload_state(gl, psi0);
    engine.driven_step(gl, drive, 1.3, dt, 20);
    cpu = psi0;
    ses::driven_step(cpu, cpu_prop, drive, 1.3, 20);
    engine.readback(gl, gpu_out);
    ok = compare("driven engine 20 steps", gpu_out, cpu, 1e-4) && ok;
    return ok;
}

// B-field: the rigid-rotation kernel (Larmor precession) vs core rotate_field.
bool check_rotate(Gl& gl) {
    const ses::Grid1D axis{-8.0, 8.0, 32};
    const ses::Grid3D g{axis, axis, axis};
    const std::vector<double> v = ses::soft_coulomb_potential(g, 1.0, 1.0, ses::Vec3d{});
    const ses::SplitOperator3D cpu_prop{g, v, 0.02};  // only for the phase tables
    const ses::Field3D psi0 = ses::gaussian_wavepacket(
        g, ses::Vec3d{2.0, 1.0, 0.0}, ses::Vec3d{1.2, 1.2, 1.2}, ses::Vec3d{0.3, 0.0, 0.0});
    ses_gpu::GpuEngine engine;
    if (!engine.initialize(gl, g, cpu_prop.half_potential_phase(),
                           cpu_prop.kinetic_phase(), psi0)) {
        std::printf("engine init: FAIL\n");
        return false;
    }
    bool ok = true;
    for (int ax = 0; ax <= 2; ax += 2) {  // about x, then z
        engine.upload_state(gl, psi0);
        engine.rotate_psi(gl, ax, 0.7);
        const ses::Field3D cpu = ses::rotate_field(psi0, ax, 0.7);
        std::vector<float> gpu_out;
        engine.readback(gl, gpu_out);
        ok = compare(ax == 0 ? "rotate about x" : "rotate about z", gpu_out, cpu, 1e-4) && ok;
    }
    return ok;
}

// B-field PROPER solve: the exact three-shear rotate_z on the GPU psi SSBO vs
// core ses::rotate_z, and the full magnetic Strang step vs core
// MagneticPropagator3D (diamagnetic potential + paramagnetic rotation).
bool check_magnetic(Gl& gl) {
    const ses::Grid1D axis{-8.0, 8.0, 32};
    const ses::Grid3D g{axis, axis, axis};
    const std::vector<double> v = ses::soft_coulomb_potential(g, 1.0, 1.0, ses::Vec3d{});
    const double dt = 0.02;
    const double b = 0.5;
    const ses::Field3D psi0 = ses::gaussian_wavepacket(
        g, ses::Vec3d{2.0, 0.0, 0.5}, ses::Vec3d{1.4, 1.4, 1.4}, ses::Vec3d{0.0, 0.4, 0.0});

    const ses::SplitOperator3D base{g, v, dt};
    ses_gpu::GpuEngine eng;
    if (!eng.initialize(gl, g, base.half_potential_phase(), base.kinetic_phase(), psi0)) {
        std::printf("engine init: FAIL\n");
        return false;
    }

    // Exact three-shear rotation vs core rotate_z.
    eng.rotate_z_shear(gl, 0.6);
    std::vector<float> gpu_out;
    eng.readback(gl, gpu_out);
    ses::Field3D cpu = psi0;
    ses::rotate_z(cpu, 0.6);
    bool ok = compare("magnetic rotate_z (three-shear)", gpu_out, cpu, 1e-3);

    // Full magnetic step vs core MagneticPropagator3D, for a field along z
    // and along x. Kinetic table is B-independent (already correct); only the
    // half-potential gains the (axis-dependent) diamagnetic term.
    for (int fa = 2; fa >= 0; fa -= 2) {  // axis z then x
        const ses::MagneticPropagator3D mprop{g, v, dt, b, fa};
        const ses::SplitOperator3D core_diamag{g, mprop.effective_potential(), dt};
        eng.upload_state(gl, psi0);
        eng.set_half_potential(gl, core_diamag.half_potential_phase());
        eng.magnetic_step(gl, fa, 0.5 * b * (0.5 * dt), 20);
        eng.readback(gl, gpu_out);
        ses::Field3D cpu2 = psi0;
        mprop.step(cpu2, 20);
        ok = compare(fa == 2 ? "magnetic step z" : "magnetic step x", gpu_out, cpu2,
                     2e-3) &&
             ok;
    }
    return ok;
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

    bool ok = check_phase_multiply(gl);

    const ses::Grid3D small{ses::Grid1D{0.0, 1.0, 16}, ses::Grid1D{0.0, 1.0, 8},
                            ses::Grid1D{0.0, 1.0, 4}};
    ok = check_fft3(gl, small, "fft3 16x8x4") && ok;

    const ses::Grid1D axis64{0.0, 1.0, 64};
    ok = check_fft3(gl, ses::Grid3D{axis64, axis64, axis64}, "fft3 64^3") && ok;

    ok = check_engine_step(gl) && ok;
    ok = check_norm_reduction(gl) && ok;
    ok = check_relax(gl) && ok;
    ok = check_deflation(gl) && ok;
    ok = check_driven_step(gl) && ok;
    ok = check_rotate(gl) && ok;
    ok = check_magnetic(gl) && ok;

    return ok ? 0 : 1;
}
