#pragma once

// Read-only observables reduced on the GPU (Stage 3): overlaps <a|b>, the
// Ehrenfest force <grad V>, and dipole matrix elements <to|r|from> -- each a
// GPU reduction that reads state buffers and finishes the 256 partial pairs on
// the CPU in double. Extracted from GpuEngine, verified by sesolver_gpucheck's
// inner-product / mean-force / dipole checks. NOTE: the psi-MUTATING deflation
// op (subtract_projection / axpy) stays on GpuEngine -- it is a state-evolution
// primitive, not an observable.
//
// NOT a standalone header: gpu_engine.hpp #includes this INSIDE namespace
// ses_gpu, after its includes and shared utilities (bind::, build_program,
// make_buffer, NormPeak, kNormPeakGroups, GpuState, and kInnerProductSrc /
// kMeanForceSrc / kDipoleSrc from gpu_shaders.hpp) are in scope.

struct Observables {
    GLuint inner_prog_ = 0;
    GLuint force_prog_ = 0;
    GLuint dipole_prog_ = 0;
    GLuint force_partials_buf_ = 0;    // vec4 partials for mean_force
    GLuint dipole_partials_buf_ = 0;   // 6-float partials for dipole_between
    GLuint grad_v_buf_ = 0;            // grad V field (vec4 per cell)

    bool build(Gl& gl) {
        inner_prog_ = build_program(gl, kInnerProductSrc, "inner product");
        force_prog_ = build_program(gl, kMeanForceSrc, "mean force");
        dipole_prog_ = build_program(gl, kDipoleSrc, "dipole");
        return inner_prog_ != 0 && force_prog_ != 0 && dipole_prog_ != 0;
    }

    // The wider (vec4 / 6-float) partials buffers this module reduces into.
    // st.partials_buf_ (the 2-float pair) is shared and allocated by the engine.
    void alloc_partials(Gl& gl) {
        force_partials_buf_ = make_buffer(gl, std::vector<float>(4 * kNormPeakGroups, 0.0f));
        dipole_partials_buf_ = make_buffer(gl, std::vector<float>(6 * kNormPeakGroups, 0.0f));
    }

    // <a|b> via the GPU reduction; returned as NormPeak{re, im} (the partials
    // pair is reused as a complex accumulator). Includes dV. Reduces into the
    // shared st.partials_buf_.
    NormPeak inner(Gl& gl, const GpuState& st, GLuint a_buf, GLuint b_buf) {
        gl.glBindBufferBase(GL_SHADER_STORAGE_BUFFER, bind::kPsi, a_buf);
        gl.glBindBufferBase(GL_SHADER_STORAGE_BUFFER, bind::kPartials, st.partials_buf_);
        gl.glBindBufferBase(GL_SHADER_STORAGE_BUFFER, bind::kPhi, b_buf);
        gl.glUseProgram(inner_prog_);
        gl.glUniform1ui(gl.glGetUniformLocation(inner_prog_, "n"),
                        static_cast<GLuint>(st.cells_));
        gl.glDispatchCompute(kNormPeakGroups, 1, 1);
        gl.glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

        float partials[2 * kNormPeakGroups];
        gl.glBindBuffer(GL_SHADER_STORAGE_BUFFER, st.partials_buf_);
        gl.glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(partials), partials);
        double re = 0.0;
        double im = 0.0;
        for (int g = 0; g < kNormPeakGroups; ++g) {
            re += partials[2 * g];
            im += partials[2 * g + 1];
        }
        const double dv = st.grid_.cell_volume();
        return NormPeak{re * dv, im * dv};
    }

    // Upload the potential-gradient field grad V (central differences on the
    // periodic grid, packed vec4 (gx,gy,gz,0) fp32) so mean_force can reduce
    // against it. Rebuild when the potential changes.
    void set_potential_gradient(Gl& gl, const GpuState& st, const std::vector<double>& v) {
        const int nx = st.grid_.x.n;
        const int ny = st.grid_.y.n;
        const int nz = st.grid_.z.n;
        const double i2hx = 1.0 / (2.0 * st.grid_.x.spacing());
        const double i2hy = 1.0 / (2.0 * st.grid_.y.spacing());
        const double i2hz = 1.0 / (2.0 * st.grid_.z.spacing());
        std::vector<float> packed(4 * st.cells_, 0.0f);
        for (int k = 0; k < nz; ++k) {
            const int kp = (k + 1) % nz;
            const int km = (k - 1 + nz) % nz;
            for (int j = 0; j < ny; ++j) {
                const int jp = (j + 1) % ny;
                const int jm = (j - 1 + ny) % ny;
                for (int i = 0; i < nx; ++i) {
                    const int ip = (i + 1) % nx;
                    const int im = (i - 1 + nx) % nx;
                    const std::size_t idx = static_cast<std::size_t>(st.grid_.flat(i, j, k));
                    packed[4 * idx + 0] = static_cast<float>(
                        (v[static_cast<std::size_t>(st.grid_.flat(ip, j, k))] -
                         v[static_cast<std::size_t>(st.grid_.flat(im, j, k))]) * i2hx);
                    packed[4 * idx + 1] = static_cast<float>(
                        (v[static_cast<std::size_t>(st.grid_.flat(i, jp, k))] -
                         v[static_cast<std::size_t>(st.grid_.flat(i, jm, k))]) * i2hy);
                    packed[4 * idx + 2] = static_cast<float>(
                        (v[static_cast<std::size_t>(st.grid_.flat(i, j, kp))] -
                         v[static_cast<std::size_t>(st.grid_.flat(i, j, km))]) * i2hz);
                }
            }
        }
        if (grad_v_buf_ == 0) {
            grad_v_buf_ = make_buffer(gl, packed);
        } else {
            gl.glBindBuffer(GL_SHADER_STORAGE_BUFFER, grad_v_buf_);
            gl.glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                               static_cast<GLsizeiptr>(packed.size() * sizeof(float)),
                               packed.data());
        }
    }

    // <grad V> = sum |psi|^2 grad V * dV -- the Ehrenfest dipole acceleration
    // for the semiclassical radiated power. Zero if no gradient was uploaded.
    ses::Vec3d mean_force(Gl& gl, const GpuState& st, GLuint psi_buf) {
        if (grad_v_buf_ == 0) {
            return ses::Vec3d{};
        }
        gl.glBindBufferBase(GL_SHADER_STORAGE_BUFFER, bind::kPsi, psi_buf);
        gl.glBindBufferBase(GL_SHADER_STORAGE_BUFFER, bind::kGradV, grad_v_buf_);
        gl.glBindBufferBase(GL_SHADER_STORAGE_BUFFER, bind::kPartials, force_partials_buf_);
        gl.glUseProgram(force_prog_);
        gl.glUniform1ui(gl.glGetUniformLocation(force_prog_, "n"),
                        static_cast<GLuint>(st.cells_));
        gl.glDispatchCompute(kNormPeakGroups, 1, 1);
        gl.glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

        float partials[4 * kNormPeakGroups];
        gl.glBindBuffer(GL_SHADER_STORAGE_BUFFER, force_partials_buf_);
        gl.glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(partials), partials);
        double gx = 0.0;
        double gy = 0.0;
        double gz = 0.0;
        for (int g = 0; g < kNormPeakGroups; ++g) {
            gx += partials[4 * g + 0];
            gy += partials[4 * g + 1];
            gz += partials[4 * g + 2];
        }
        const double dv = st.grid_.cell_volume();
        return ses::Vec3d{gx * dv, gy * dv, gz * dv};
    }

    // <to| r |from> = sum conj(to) * (x,y,z) * from * dV, straight from two
    // resident state buffers -- the atlas dipole integrals with NO CPU copy of
    // the states (they already live on the GPU). Component-wise complex.
    ses::DipoleMatrixElement dipole_between(Gl& gl, const GpuState& st, GLuint to_buf,
                                            GLuint from_buf) {
        gl.glBindBufferBase(GL_SHADER_STORAGE_BUFFER, bind::kPsi, to_buf);
        gl.glBindBufferBase(GL_SHADER_STORAGE_BUFFER, bind::kPhi, from_buf);
        gl.glBindBufferBase(GL_SHADER_STORAGE_BUFFER, bind::kPartials, dipole_partials_buf_);
        gl.glUseProgram(dipole_prog_);
        gl.glUniform1ui(gl.glGetUniformLocation(dipole_prog_, "n"),
                        static_cast<GLuint>(st.cells_));
        gl.glUniform1i(gl.glGetUniformLocation(dipole_prog_, "nx"), st.grid_.x.n);
        gl.glUniform1i(gl.glGetUniformLocation(dipole_prog_, "ny"), st.grid_.y.n);
        gl.glUniform3f(gl.glGetUniformLocation(dipole_prog_, "box_min"),
                       static_cast<float>(st.grid_.x.xmin), static_cast<float>(st.grid_.y.xmin),
                       static_cast<float>(st.grid_.z.xmin));
        gl.glUniform3f(gl.glGetUniformLocation(dipole_prog_, "cell_h"),
                       static_cast<float>(st.grid_.x.spacing()),
                       static_cast<float>(st.grid_.y.spacing()),
                       static_cast<float>(st.grid_.z.spacing()));
        gl.glDispatchCompute(kNormPeakGroups, 1, 1);
        gl.glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

        float partials[6 * kNormPeakGroups];
        gl.glBindBuffer(GL_SHADER_STORAGE_BUFFER, dipole_partials_buf_);
        gl.glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(partials), partials);
        double dxr = 0.0, dxi = 0.0, dyr = 0.0, dyi = 0.0, dzr = 0.0, dzi = 0.0;
        for (int g = 0; g < kNormPeakGroups; ++g) {
            dxr += partials[6 * g + 0];
            dxi += partials[6 * g + 1];
            dyr += partials[6 * g + 2];
            dyi += partials[6 * g + 3];
            dzr += partials[6 * g + 4];
            dzi += partials[6 * g + 5];
        }
        const double dv = st.grid_.cell_volume();
        return ses::DipoleMatrixElement{
            ses::Complex<double>{dxr * dv, dxi * dv},
            ses::Complex<double>{dyr * dv, dyi * dv},
            ses::Complex<double>{dzr * dv, dzi * dv}};
    }
};
