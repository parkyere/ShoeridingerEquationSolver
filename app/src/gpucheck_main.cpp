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
#include <core/decay.hpp>
#include <core/drive.hpp>
#include <core/fft.hpp>
#include <core/field.hpp>
#include <core/grid.hpp>
#include <core/emission.hpp>
#include <core/harmonics.hpp>
#include <core/imaginary_time.hpp>
#include <core/magnetic.hpp>
#include <core/radial.hpp>
#include <core/potential.hpp>
#include <core/projection.hpp>
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

// Radiation: the GPU mean-force reduction <grad V> vs core
// mean_potential_gradient (the Ehrenfest dipole acceleration).
bool check_mean_force(Gl& gl) {
    const ses::Grid1D axis{-8.0, 8.0, 32};
    const ses::Grid3D g{axis, axis, axis};
    const std::vector<double> v = ses::soft_coulomb_potential(g, 1.0, 1.0, ses::Vec3d{});
    const ses::SplitOperator3D prop{g, v, 0.02};
    const ses::Field3D psi = ses::gaussian_wavepacket(
        g, ses::Vec3d{2.0, 1.0, -0.5}, ses::Vec3d{1.4, 1.4, 1.4}, ses::Vec3d{0.0, 0.3, 0.0});
    ses_gpu::GpuEngine eng;
    if (!eng.initialize(gl, g, prop.half_potential_phase(), prop.kinetic_phase(), psi)) {
        std::printf("engine init: FAIL\n");
        return false;
    }
    eng.set_potential_gradient(gl, v);
    const ses::Vec3d gpu = eng.mean_force(gl);
    const ses::Vec3d cpu = ses::mean_potential_gradient(psi, v, g);
    const double err = std::max({std::abs(gpu.x - cpu.x), std::abs(gpu.y - cpu.y),
                                 std::abs(gpu.z - cpu.z)});
    const bool ok = err < 1e-5;
    std::printf("mean force <grad V>: max |gpu - cpu| = %.3e  [%s]\n", err,
                ok ? "PASS" : "FAIL");
    return ok;
}

// T10: the atlas dipole integral <to| r |from> reduced on the GPU from two
// resident state buffers vs ses::dipole_matrix_element. Compare against the
// core sum over the SAME fp32 data the GPU integrates (state buffers are
// fp32), so the residual is pure tree-vs-sequential reduction order.
bool check_dipole(Gl& gl) {
    const ses::Grid1D axis{-8.0, 8.0, 32};
    const ses::Grid3D g{axis, axis, axis};
    const std::vector<double> v = ses::soft_coulomb_potential(g, 1.0, 1.0, ses::Vec3d{});
    const ses::SplitOperator3D prop{g, v, 0.02};
    const ses::Field3D seed = ses::gaussian_wavepacket(
        g, ses::Vec3d{0.0, 0.0, 0.0}, ses::Vec3d{1.5, 1.5, 1.5}, ses::Vec3d{});
    ses_gpu::GpuEngine eng;
    if (!eng.initialize(gl, g, prop.half_potential_phase(), prop.kinetic_phase(), seed)) {
        std::printf("engine init: FAIL\n");
        return false;
    }
    // Two distinct states; the "from" carries a momentum kick so both real
    // and imaginary parts of the matrix element are exercised.
    const ses::Field3D fto = ses::gaussian_wavepacket(
        g, ses::Vec3d{1.0, -0.5, 0.3}, ses::Vec3d{1.3, 1.3, 1.3}, ses::Vec3d{});
    const ses::Field3D ffrom = ses::gaussian_wavepacket(
        g, ses::Vec3d{-0.7, 0.4, -0.2}, ses::Vec3d{1.1, 1.1, 1.1}, ses::Vec3d{0.3, 0.0, 0.2});
    const GLuint to_buf = eng.create_state_buffer(gl, fto);
    const GLuint from_buf = eng.create_state_buffer(gl, ffrom);
    const ses::DipoleMatrixElement gpu = eng.dipole_between(gl, to_buf, from_buf);
    // fp32-truncate to match the uploaded buffers exactly.
    auto trunc = [](const ses::Field3D& f) {
        ses::Field3D o{f.grid()};
        const auto& s = f.data();
        auto& d = o.data();
        for (std::size_t i = 0; i < s.size(); ++i) {
            d[i] = ses::Complex<double>{static_cast<double>(static_cast<float>(s[i].real())),
                                        static_cast<double>(static_cast<float>(s[i].imag()))};
        }
        return o;
    };
    const ses::DipoleMatrixElement cpu =
        ses::dipole_matrix_element(trunc(fto), trunc(ffrom));
    const double err = std::max({std::abs(gpu.x.real() - cpu.x.real()),
                                 std::abs(gpu.x.imag() - cpu.x.imag()),
                                 std::abs(gpu.y.real() - cpu.y.real()),
                                 std::abs(gpu.y.imag() - cpu.y.imag()),
                                 std::abs(gpu.z.real() - cpu.z.real()),
                                 std::abs(gpu.z.imag() - cpu.z.imag())});
    const bool ok = err < 1e-5;
    std::printf("dipole <to|r|from> reduce: max |gpu - cpu| = %.3e  [%s]\n", err,
                ok ? "PASS" : "FAIL");
    return ok;
}

// T10: orbital synthesis psi = (u/r) Y_lm on the GPU vs core synthesize_orbital.
// The radial u is solved once on the CPU (harmonic trap, exercises l up to 5)
// and fed to BOTH paths; the GPU builds + normalizes on-device while the core
// builds in double, so the residual is the fp32-vs-double synthesis error.
bool check_synth(Gl& gl) {
    const ses::Grid1D axis{-8.0, 8.0, 32};
    const ses::Grid3D g{axis, axis, axis};
    const std::vector<double> v = ses::soft_coulomb_potential(g, 1.0, 1.0, ses::Vec3d{});
    const ses::SplitOperator3D prop{g, v, 0.02};
    const ses::Field3D seed = ses::gaussian_wavepacket(
        g, ses::Vec3d{}, ses::Vec3d{1.5, 1.5, 1.5}, ses::Vec3d{});
    ses_gpu::GpuEngine eng;
    if (!eng.initialize(gl, g, prop.half_potential_phase(), prop.kinetic_phase(), seed)) {
        std::printf("engine init: FAIL\n");
        return false;
    }
    const ses::RadialGrid rg{8.0, 1599};
    std::vector<double> vr(static_cast<std::size_t>(rg.n));
    for (int i = 0; i < rg.n; ++i) {
        vr[static_cast<std::size_t>(i)] = 0.5 * rg.r(i) * rg.r(i);
    }
    struct SynCase {
        int l, k, m;
    };
    const SynCase cases[] = {{0, 0, 0}, {1, 0, -1}, {2, 0, 1},
                             {3, 0, -2}, {4, 0, 3}, {5, 0, 5}, {5, 0, 0}};
    double worst = 0.0;
    for (const SynCase& c : cases) {
        const ses::RadialState st =
            ses::radial_eigenstate(rg, ses::radial_hamiltonian(rg, vr, c.l), c.k);
        const ses::Field3D cpu = ses::synthesize_orbital(g, rg, st.u, c.l, c.m);
        const GLuint buf =
            eng.synthesize_state(gl, st.u, c.l, c.m, rg.h(), rg.rmax, rg.n);
        eng.copy_into_psi(gl, buf);
        std::vector<float> gpu;
        eng.readback(gl, gpu);
        double err = 0.0;
        for (std::size_t i = 0; i < cpu.data().size(); ++i) {
            err = std::max(err, std::abs(static_cast<double>(gpu[2 * i]) -
                                         cpu.data()[i].real()));
            err = std::max(err, std::abs(static_cast<double>(gpu[2 * i + 1]) -
                                         cpu.data()[i].imag()));
        }
        worst = std::max(worst, err);
        gl.glDeleteBuffers(1, &buf);
    }
    const bool ok = worst < 1e-4;
    std::printf("orbital synthesis (u/r)Ylm: max |gpu - cpu| = %.3e  [%s]\n", worst,
                ok ? "PASS" : "FAIL");
    return ok;
}

// fp16 (half) atlas storage: the small-VRAM fallback packs each eigenstate to
// fp16 and unpacks it on demand. Two properties pin the pack/unpack kernels:
//  (a) a round-trip matches the original to fp16 precision (~1e-3 for O(1)
//      values) -- it really is storing at half precision, and
//  (b) it is IDEMPOTENT: a SECOND round-trip is bit-exact, because the first
//      unpack already lands on the fp16 value lattice -- proving pack/unpack
//      are true inverses (a broken kernel fails this even if (a) passes).
bool check_fp16(Gl& gl) {
    const ses::Grid1D axis{-4.0, 4.0, 16};
    const ses::Grid3D g{axis, axis, axis};
    const std::vector<double> v = ses::soft_coulomb_potential(g, 1.0, 1.0, ses::Vec3d{});
    const ses::SplitOperator3D prop{g, v, 0.02};
    const ses::Field3D seed = deterministic_field(g);
    ses_gpu::GpuEngine eng;
    if (!eng.initialize(gl, g, prop.half_potential_phase(), prop.kinetic_phase(), seed)) {
        std::printf("engine init: FAIL\n");
        return false;
    }
    const ses::Field3D field = deterministic_field(g);

    // Round-trip 1: field (fp32) -> half -> psi (fp32).
    const GLuint src = eng.create_state_buffer(gl, field);
    const GLuint half1 = eng.make_half_state_buffer(gl);
    eng.pack_to_half(gl, src, half1);
    eng.unpack_into_psi(gl, half1);
    std::vector<float> rec1;
    eng.readback(gl, rec1);

    // Round-trip 2: rec1 -> half -> psi. Must reproduce rec1 bit-for-bit.
    ses::Field3D rec1_field{g};
    for (std::size_t i = 0; i < rec1_field.data().size(); ++i) {
        rec1_field.data()[i] = ses::Complex<double>{rec1[2 * i], rec1[2 * i + 1]};
    }
    const GLuint src2 = eng.create_state_buffer(gl, rec1_field);
    const GLuint half2 = eng.make_half_state_buffer(gl);
    eng.pack_to_half(gl, src2, half2);
    eng.unpack_into_psi(gl, half2);
    std::vector<float> rec2;
    eng.readback(gl, rec2);

    double rt_err = 0.0;    // vs the original: bounded by fp16 precision
    double idem_err = 0.0;  // second round-trip vs first: must be exactly 0
    for (std::size_t i = 0; i < field.data().size(); ++i) {
        rt_err = std::max({rt_err,
                           std::abs(static_cast<double>(rec1[2 * i]) - field.data()[i].real()),
                           std::abs(static_cast<double>(rec1[2 * i + 1]) - field.data()[i].imag())});
        idem_err = std::max({idem_err, std::abs(static_cast<double>(rec2[2 * i] - rec1[2 * i])),
                             std::abs(static_cast<double>(rec2[2 * i + 1] - rec1[2 * i + 1]))});
    }
    for (const GLuint b : {src, half1, src2, half2}) {
        gl.glDeleteBuffers(1, &b);
    }
    const bool ok = rt_err < 1e-3 && idem_err == 0.0;
    std::printf("fp16 half pack/unpack: round-trip = %.3e (fp16), idempotent = %.3e  [%s]\n",
                rt_err, idem_err, ok ? "PASS" : "FAIL");
    return ok;
}

// fp16 decode-on-use consumers: an fp16-stored eigenstate, unpacked to a
// scratch fp32 buffer, must feed the tested inner-product and dipole reductions
// to fp16 precision -- and mixed precision (an fp32 operand with an fp16 one,
// as the channel table pairs an fp16 p-state with an fp32 s-state) must work.
bool check_fp16_consumers(Gl& gl) {
    const ses::Grid1D axis{-8.0, 8.0, 32};
    const ses::Grid3D g{axis, axis, axis};
    const std::vector<double> v = ses::soft_coulomb_potential(g, 1.0, 1.0, ses::Vec3d{});
    const ses::SplitOperator3D prop{g, v, 0.02};
    const ses::Field3D seed = ses::gaussian_wavepacket(
        g, ses::Vec3d{}, ses::Vec3d{1.5, 1.5, 1.5}, ses::Vec3d{});
    ses_gpu::GpuEngine eng;
    if (!eng.initialize(gl, g, prop.half_potential_phase(), prop.kinetic_phase(), seed)) {
        std::printf("engine init: FAIL\n");
        return false;
    }
    const ses::RadialGrid rg{8.0, 1599};
    std::vector<double> vr(static_cast<std::size_t>(rg.n));
    for (int i = 0; i < rg.n; ++i) {
        vr[static_cast<std::size_t>(i)] = 0.5 * rg.r(i) * rg.r(i);
    }
    const ses::RadialState st =
        ses::radial_eigenstate(rg, ses::radial_hamiltonian(rg, vr, 1), 0);
    const GLuint s32 = eng.synthesize_state(gl, st.u, 1, 0, rg.h(), rg.rmax, rg.n);
    const GLuint s16 = eng.synthesize_state_half(gl, st.u, 1, 0, rg.h(), rg.rmax, rg.n);

    const ses::Field3D testpsi = ses::gaussian_wavepacket(
        g, ses::Vec3d{1.0, 0.5, -0.4}, ses::Vec3d{1.7, 1.7, 1.7}, ses::Vec3d{});
    eng.upload_state(gl, testpsi);

    const ses_gpu::NormPeak ip32 = eng.inner_with_psi(gl, s32);
    const ses_gpu::NormPeak ip16 = eng.inner_with_psi(gl, ses_gpu::StateHandle{s16, true});
    const double inner_err =
        std::max(std::abs(ip32.sum - ip16.sum), std::abs(ip32.peak - ip16.peak));

    const ses::DipoleMatrixElement d32 = eng.dipole_between(gl, s32, s32);
    const ses::DipoleMatrixElement d16 =
        eng.dipole_between(gl, ses_gpu::StateHandle{s16, true}, ses_gpu::StateHandle{s16, true});
    const ses::DipoleMatrixElement dmix =
        eng.dipole_between(gl, ses_gpu::StateHandle{s32, false}, ses_gpu::StateHandle{s16, true});
    auto dip_err = [](const ses::DipoleMatrixElement& a, const ses::DipoleMatrixElement& b) {
        return std::max({std::abs(a.x.real() - b.x.real()), std::abs(a.y.real() - b.y.real()),
                         std::abs(a.z.real() - b.z.real()), std::abs(a.x.imag() - b.x.imag()),
                         std::abs(a.y.imag() - b.y.imag()), std::abs(a.z.imag() - b.z.imag())});
    };
    const double d16_err = dip_err(d32, d16);
    const double dmix_err = dip_err(d32, dmix);

    for (const GLuint b : {s32, s16}) {
        gl.glDeleteBuffers(1, &b);
    }
    const double worst = std::max({inner_err, d16_err, dmix_err});
    const bool ok = worst < 3e-3;
    std::printf("fp16 consumers (inner/dipole/mixed vs fp32): max = %.3e  [%s]\n", worst,
                ok ? "PASS" : "FAIL");
    return ok;
}

// Orbital-free angular-decomposition projection: the GPU deposit (sorted-gather
// per radial bin, fp32) + CPU-double 1-D radial dot vs the CPU oracle
// project_radial_angular (which equals the direct grid inner product to 1e-11).
// The gap is the fp32 deposit + fp32 bin-key rounding vs the double oracle.
bool check_project(Gl& gl) {
    const ses::Grid1D axis{-8.0, 8.0, 32};
    const ses::Grid3D g{axis, axis, axis};
    const std::vector<double> v = ses::soft_coulomb_potential(g, 1.0, 1.0, ses::Vec3d{});
    const ses::SplitOperator3D prop{g, v, 0.02};
    const ses::Field3D seed = ses::gaussian_wavepacket(
        g, ses::Vec3d{}, ses::Vec3d{1.5, 1.5, 1.5}, ses::Vec3d{});
    ses_gpu::GpuEngine eng;
    if (!eng.initialize(gl, g, prop.half_potential_phase(), prop.kinetic_phase(), seed)) {
        std::printf("engine init: FAIL\n");
        return false;
    }
    const ses::RadialGrid rg{8.0, 200};
    std::vector<std::vector<double>> u_by_level(2);
    for (int lev = 0; lev < 2; ++lev) {
        u_by_level[static_cast<std::size_t>(lev)].resize(static_cast<std::size_t>(rg.n));
        for (int i = 0; i < rg.n; ++i) {
            const double r = rg.r(i);
            u_by_level[static_cast<std::size_t>(lev)][static_cast<std::size_t>(i)] =
                (lev == 0) ? r * std::exp(-r) : r * r * std::exp(-0.5 * r);
        }
    }
    const int l_max = 5;
    const ses::RadialBinIndex idx = ses::build_radial_bin_index(g, rg);
    eng.set_projection_index(gl, idx.sorted_cell, idx.bin_off, rg.n, rg.h(), l_max);

    const ses::Field3D psi = ses::gaussian_wavepacket(
        g, ses::Vec3d{1.0, 0.5, -0.4}, ses::Vec3d{1.7, 1.7, 1.7}, ses::Vec3d{0.3, -0.2, 0.1});
    eng.upload_state(gl, psi);
    eng.project_psi(gl);

    const std::vector<ses::ProjectorState> states = {
        {0, 0, 0}, {1, 1, 0}, {1, 1, 1}, {0, 2, -1}, {1, 4, 2}, {0, 5, 3}};
    const ses::RadialAngularProjection cpu =
        ses::project_radial_angular(psi, rg, u_by_level, states, l_max);
    double worst = 0.0;
    for (std::size_t s = 0; s < states.size(); ++s) {
        const ses::ProjectorState& st = states[s];
        const ses::Complex<double> gpu = eng.project_amplitude(
            u_by_level[static_cast<std::size_t>(st.level)], st.l, st.m);
        const ses::Complex<double> ref =
            cpu.amp[s] * std::sqrt(cpu.norm2[static_cast<std::size_t>(s)]);
        const double e = std::max(std::abs(gpu.real() - ref.real()),
                                  std::abs(gpu.imag() - ref.imag())) /
                         (1.0 + std::abs(ref));
        worst = std::max(worst, e);
    }
    // Determinism: a second deposit reproduces the amplitude bit-for-bit (no
    // atomics; fixed-order tree reduction + deterministic double finish).
    const ses::Complex<double> a1 = eng.project_amplitude(u_by_level[1], 1, 0);
    eng.project_psi(gl);
    const ses::Complex<double> a2 = eng.project_amplitude(u_by_level[1], 1, 0);
    const bool deterministic = (a1 == a2);
    const bool ok = worst < 1e-3 && deterministic;
    std::printf("orbital-free projection <n|psi>: max rel |gpu - cpu| = %.3e, "
                "deterministic = %d  [%s]\n",
                worst, deterministic ? 1 : 0, ok ? "PASS" : "FAIL");
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
    ok = check_magnetic(gl) && ok;
    ok = check_mean_force(gl) && ok;
    ok = check_dipole(gl) && ok;
    ok = check_synth(gl) && ok;
    ok = check_fp16(gl) && ok;
    ok = check_fp16_consumers(gl) && ok;
    ok = check_project(gl) && ok;

    return ok ? 0 : 1;
}
