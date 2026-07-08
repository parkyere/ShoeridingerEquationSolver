#pragma once

// Orbital-free angular-decomposition projection on the GPU (Phase 3): one grid
// pass deposits psi -> g_lm radial bins (independent of the state count), then a
// 1-D radial dot per state gives <n|psi>. Extracted from GpuEngine (Stage 3) as
// a self-contained concern verified by sesolver_gpucheck's projection test.
//
// NOT a standalone header: gpu_engine.hpp #includes this INSIDE namespace ses_gpu,
// after its includes (core/complex.hpp, core/grid.hpp, <vector>, <algorithm>,
// <cstdint>) and shared utilities (bind::, build_program, make_buffer,
// make_u32_buffer, and kProjectDepositSrc from gpu_shaders.hpp) are in scope.

struct Projector {
    GLuint prog_ = 0;
    GLuint sorted_buf_ = 0;   // static counting-sort: cells grouped by radial bin
    GLuint binoff_buf_ = 0;   // static CSR offsets
    GLuint glm_buf_ = 0;      // g_lm[ncomp*nr] deposit output (fp32)
    int nr_ = 0;              // radial bins
    int ncomp_ = 0;           // (l_max+1)^2
    double h_radial_ = 0.0;
    std::vector<std::vector<ses::Complex<double>>> glm_host_;  // last project() readback

    bool build(Gl& gl) {
        prog_ = build_program(gl, kProjectDepositSrc, "projection deposit");
        return prog_ != 0;
    }

    // Upload the static counting-sort geometry (ses::build_radial_bin_index) and
    // allocate g_lm. Call once after the radial grid is fixed.
    void set_index(Gl& gl, const std::vector<std::uint32_t>& sorted_cell,
                   const std::vector<std::uint32_t>& bin_off, int n_radial,
                   double h_radial, int l_max) {
        nr_ = n_radial;
        ncomp_ = (l_max + 1) * (l_max + 1);
        h_radial_ = h_radial;
        sorted_buf_ = make_u32_buffer(gl, sorted_cell);
        binoff_buf_ = make_u32_buffer(gl, bin_off);
        glm_buf_ = make_buffer(gl, std::vector<float>(
            static_cast<std::size_t>(2 * ncomp_ * nr_), 0.0f));
    }

    // Deposit psi -> g_lm (read back to host as double). ONE grid pass over psi,
    // INDEPENDENT of the number of states; then call amplitude() per state.
    void project(Gl& gl, GLuint psi_buf, const ses::Grid3D& grid) {
        gl.glBindBufferBase(GL_SHADER_STORAGE_BUFFER, bind::kPsi, psi_buf);
        gl.glBindBufferBase(GL_SHADER_STORAGE_BUFFER, bind::kAux, sorted_buf_);
        gl.glBindBufferBase(GL_SHADER_STORAGE_BUFFER, bind::kBinOff, binoff_buf_);
        gl.glBindBufferBase(GL_SHADER_STORAGE_BUFFER, bind::kGlm, glm_buf_);
        gl.glUseProgram(prog_);
        gl.glUniform1i(gl.glGetUniformLocation(prog_, "nx"), grid.x.n);
        gl.glUniform1i(gl.glGetUniformLocation(prog_, "ny"), grid.y.n);
        gl.glUniform3f(gl.glGetUniformLocation(prog_, "box_min"),
                       static_cast<float>(grid.x.xmin), static_cast<float>(grid.y.xmin),
                       static_cast<float>(grid.z.xmin));
        gl.glUniform3f(gl.glGetUniformLocation(prog_, "cell_h"),
                       static_cast<float>(grid.x.spacing()),
                       static_cast<float>(grid.y.spacing()),
                       static_cast<float>(grid.z.spacing()));
        gl.glUniform1f(gl.glGetUniformLocation(prog_, "h_radial"),
                       static_cast<float>(h_radial_));
        gl.glUniform1f(gl.glGetUniformLocation(prog_, "dv"),
                       static_cast<float>(grid.cell_volume()));
        gl.glUniform1i(gl.glGetUniformLocation(prog_, "nr"), nr_);
        gl.glDispatchCompute(static_cast<GLuint>(nr_), 1, 1);
        gl.glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

        std::vector<float> raw(static_cast<std::size_t>(2 * ncomp_ * nr_));
        gl.glBindBuffer(GL_SHADER_STORAGE_BUFFER, glm_buf_);
        gl.glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                              static_cast<GLsizeiptr>(raw.size() * sizeof(float)),
                              raw.data());
        glm_host_.assign(static_cast<std::size_t>(ncomp_),
                         std::vector<ses::Complex<double>>(static_cast<std::size_t>(nr_)));
        for (int c = 0; c < ncomp_; ++c) {
            for (int j = 0; j < nr_; ++j) {
                const std::size_t o = 2 * (static_cast<std::size_t>(c) * nr_ +
                                           static_cast<std::size_t>(j));
                glm_host_[static_cast<std::size_t>(c)][static_cast<std::size_t>(j)] =
                    ses::Complex<double>{raw[o], raw[o + 1]};
            }
        }
    }

    // <n|psi> raw amplitude = sum_j u_nl[j] g_lm[lm(l,m)][j] (double CPU finish).
    // Population = norm_sq(raw) / (sum_j u^2 h). Needs a prior project().
    ses::Complex<double> amplitude(const std::vector<double>& u, int l, int m) const {
        const std::vector<ses::Complex<double>>& gc =
            glm_host_[static_cast<std::size_t>(l * l + (l + m))];
        ses::Complex<double> raw{};
        const int n = std::min(static_cast<int>(u.size()), nr_);
        for (int j = 0; j < n; ++j) {
            raw += u[static_cast<std::size_t>(j)] * gc[static_cast<std::size_t>(j)];
        }
        return raw;
    }
};
