// Humble Object shell -- the Qt + hand-written OpenGL boundary.
//
// NO domain logic lives here (docs/ARCHITECTURE.md). The TDSE runs in core's
// WavepacketSimulation; matrices, colormaps, marching cubes, and ALL the
// volume-rendering math (ray_box, Beer-Lambert alpha, front-to-back
// compositing, phase LUT) live in core and are unit-tested -- the GLSL below
// is a line-by-line transcription of those verified formulas.
//
// v5 deliverable: THE STATIONARY-STATE DEMO ARC on top of the volume-
// rendered cloud. Key 2 switches stepping to imaginary time: you WATCH the
// packet cool into the 1s ground state (energy readout converging). Key 1
// returns to real time: the density freezes -- only the hue cycles at
// e^{-i E0 t} -- the definition of a stationary state, on screen. Key R
// releases a fresh wavepacket.
// Controls: drag = orbit, wheel = zoom, space = pause, Tab = cloud/surface,
// 1 = real time, 2 = relax (imaginary time), R = reset packet,
// [ ] = thinner/denser cloud.

#include <core/camera.hpp>
#include <core/colormap.hpp>
#include <core/decay.hpp>
#include <core/field.hpp>
#include <core/grid.hpp>
#include <core/imaginary_time.hpp>
#include <core/marching_cubes.hpp>
#include <core/observables.hpp>
#include <core/potential.hpp>
#include <core/sampling.hpp>
#include <core/simulation.hpp>
#include <core/sphere.hpp>
#include <core/vec.hpp>
#include <core/volume.hpp>

#include "gpu_engine.hpp"

#include <QApplication>
#include <QKeyEvent>
#include <QMainWindow>
#include <QMessageBox>
#include <QMouseEvent>
#include <QOpenGLFunctions_4_3_Core>
#include <QOpenGLWidget>
#include <QString>
#include <QSurfaceFormat>
#include <QTimer>
#include <QToolBar>
#include <QWheelEvent>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <optional>
#include <random>
#include <utility>
#include <vector>

namespace {

[[noreturn]] void fatal_gl_error(const char* stage, const QString& detail) {
    qCritical("%s: %s", stage, qPrintable(detail));
    QMessageBox::critical(nullptr, QStringLiteral("OpenGL error"),
                          QStringLiteral("%1\n\n%2").arg(QLatin1String(stage), detail));
    std::exit(EXIT_FAILURE);
}

ses::WavepacketSimulation make_simulation() {
    // 128^3: real-time stepping runs on the GPU engine (docs/GPU_PLAN.md G5);
    // the CPU session stays the double-precision truth for relax / measure /
    // surface meshing, synced on demand.
    const ses::Grid1D axis{-12.0, 12.0, 128};
    const ses::Grid3D grid{axis, axis, axis};
    return ses::WavepacketSimulation{ses::WavepacketSimulation::Config{
        grid,
        ses::soft_coulomb_potential(grid, 1.0, 1.0, ses::Vec3d{}),
        ses::Vec3d{3.0, 0.0, 0.0},  // r0: beside the nucleus
        ses::Vec3d{1.5, 1.5, 1.5},  // sigma
        ses::Vec3d{0.0, 0.4, 0.0},  // k0: tangential kick
        0.04,                       // dt
    }};
}

constexpr int kStepsPerTick = 1;
constexpr int kMaxPendingGpuSteps = 8;  // backlog cap: a stalled paint cannot spiral
constexpr int kRelaxStepsPerTick = 1;
constexpr double kRelaxDtau = 0.05;
constexpr int kTickMs = 16;
constexpr double kIsoFraction = 0.25;
constexpr int kPhaseLutSize = 256;
constexpr double kMeasureSigma = 0.8;  // measurement resolution (Bohr)
// Display decay rate: the TRUE Einstein-A lifetime is ~1e8 a.u. (unwatchable);
// this gives tau_display ~ 8 a.u. (~3 s of wall time). The title reports the
// true lifetime and the acceleration factor honestly.
constexpr double kDecayGammaDisplay = 0.125;
constexpr double kProtonMarkerRadius = 0.35;  // symbolic (a real proton is ~1e-5 Bohr)

// ---- isosurface (mesh) shaders ----

const char* kMeshVertexShader = R"(#version 430 core
layout(location = 0) in vec3 pos;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec3 color;
uniform mat4 mvp;
out vec3 v_normal;
out vec3 v_pos;
out vec3 v_color;
void main() {
    gl_Position = mvp * vec4(pos, 1.0);
    v_normal = normal;
    v_pos = pos;
    v_color = color;
}
)";

const char* kMeshFragmentShader = R"(#version 430 core
in vec3 v_normal;
in vec3 v_pos;
in vec3 v_color;
uniform vec3 eye;
out vec4 frag;
void main() {
    vec3 n = normalize(v_normal);
    vec3 vdir = normalize(eye - v_pos);
    float diffuse = abs(dot(n, vdir));
    float spec = pow(max(dot(n, vdir), 0.0), 32.0);
    vec3 c = v_color * (0.20 + 0.80 * diffuse) + vec3(0.25) * spec;
    frag = vec4(c, 1.0);
}
)";

// ---- volume (ray-marching) shaders ----

const char* kVolumeVertexShader = R"(#version 430 core
layout(location = 0) in vec3 pos;
uniform mat4 mvp;
out vec3 v_world;
void main() {
    gl_Position = mvp * vec4(pos, 1.0);
    v_world = pos;
}
)";

// Transcription of the TESTED core formulas (core/volume.hpp):
// ray_box (slab), sample_alpha (Beer-Lambert), front-to-back compositing.
// Colors come from the tested phase_lut via a 1D texture; density and phase
// are derived from the complex psi stored in an RG 3D texture, so the GPU
// interpolates re/im exactly like core's sample_trilinear.
const char* kVolumeFragmentShader = R"(#version 430 core
in vec3 v_world;
uniform vec3 eye;
uniform vec3 box_min;
uniform vec3 box_max;
uniform float inv_peak;
uniform float absorbance;
uniform vec3 proton_center;
uniform float proton_radius;
uniform vec3 proton_color;
uniform sampler3D psi_tex;
uniform sampler1D phase_tex;
out vec4 frag;

vec2 ray_box(vec3 o, vec3 d) {
    vec3 inv = 1.0 / d;
    vec3 t1 = (box_min - o) * inv;
    vec3 t2 = (box_max - o) * inv;
    vec3 tmin = min(t1, t2);
    vec3 tmax = max(t1, t2);
    return vec2(max(max(tmin.x, tmin.y), tmin.z),
                min(min(tmax.x, tmax.y), tmax.z));
}

// Mirrors the tested ses::ray_sphere (unit direction, raw interval).
vec2 ray_sphere(vec3 o, vec3 d) {
    vec3 oc = o - proton_center;
    float b = dot(oc, d);
    float c = dot(oc, oc) - proton_radius * proton_radius;
    float disc = b * b - c;
    if (disc < 0.0) {
        return vec2(1.0, -1.0);  // empty interval = miss
    }
    float s = sqrt(disc);
    return vec2(-b - s, -b + s);
}

void main() {
    vec3 dir = normalize(v_world - eye);
    vec2 t = ray_box(eye, dir);
    float tn = max(t.x, 0.0);
    if (t.y <= tn) {
        discard;
    }

    // Terminate the march at the proton sphere: cloud IN FRONT still fogs
    // it, cloud BEHIND is correctly occluded (the earlier mesh marker was
    // fogged from both sides and vanished at the density peak).
    float t_stop = t.y;
    bool hit_proton = false;
    vec2 sp = ray_sphere(eye, dir);
    if (sp.x <= sp.y && sp.x > tn && sp.x < t_stop) {
        hit_proton = true;
        t_stop = sp.x;
    }

    const int kSteps = 160;
    float step_len = (t_stop - tn) / float(kSteps);
    // Per-pixel jitter of the ray start kills wood-grain banding.
    float jitter = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453);

    vec3 C = vec3(0.0);
    float A = 0.0;
    for (int i = 0; i < kSteps; ++i) {
        vec3 p = eye + (tn + (float(i) + jitter) * step_len) * dir;
        vec3 uvw = (p - box_min) / (box_max - box_min);
        vec2 s = texture(psi_tex, uvw).rg;
        float dens = dot(s, s) * inv_peak;
        float alpha = 1.0 - exp(-absorbance * dens * step_len);
        float phase01 = (atan(s.g, s.r) + 3.14159265358979) / 6.28318530717959;
        vec3 col = texture(phase_tex, phase01).rgb;
        float w = (1.0 - A) * alpha;
        C += w * col;
        A += w;
        if (A >= 0.999) {
            break;
        }
    }

    if (hit_proton) {
        // Shade the sphere point with the same headlight model as the mesh.
        vec3 p = eye + sp.x * dir;
        vec3 n = (p - proton_center) / proton_radius;
        float diffuse = max(dot(n, -dir), 0.0);
        float spec = pow(diffuse, 32.0);
        vec3 shaded = proton_color * (0.25 + 0.75 * diffuse) + vec3(0.2) * spec;
        C += (1.0 - A) * shaded;
        A = 1.0;
    }

    frag = vec4(C, A);  // premultiplied; blended over the clear color
}
)";

enum class ViewMode { Cloud, Surface };
enum class Stepping { RealTime, Relaxing, RelaxingExcited };

class Viewport : public QOpenGLWidget, protected QOpenGLFunctions_4_3_Core {
public:
    explicit Viewport(QWidget* parent = nullptr)
        : QOpenGLWidget(parent), sim_(make_simulation()) {
        setFocusPolicy(Qt::StrongFocus);
        remesh();
        stage_volume();
        connect(&timer_, &QTimer::timeout, this, &Viewport::tick);
        timer_.start(kTickMs);
    }

protected:
    // GPU stepping covers the Cloud view in BOTH real and imaginary time
    // (G7); measure and surface meshing run on the CPU double session,
    // synced through the single cpu_is_truth_ invariant.
    bool use_gpu_path() const { return gpu_ok_ && mode_ == ViewMode::Cloud; }

    void initializeGL() override {
        if (!initializeOpenGLFunctions()) {
            const QSurfaceFormat got = context()->format();
            fatal_gl_error("OpenGL 4.3 core profile is required",
                           QStringLiteral("The driver provided only %1.%2. Update the GPU "
                                          "driver or run on hardware with OpenGL 4.3 support.")
                               .arg(got.majorVersion())
                               .arg(got.minorVersion()));
        }
        glClearColor(0.04f, 0.05f, 0.09f, 1.0f);

        // -- mesh pipeline --
        mesh_program_ = link_program(kMeshVertexShader, kMeshFragmentShader);
        mesh_mvp_loc_ = glGetUniformLocation(mesh_program_, "mvp");
        mesh_eye_loc_ = glGetUniformLocation(mesh_program_, "eye");
        glGenVertexArrays(1, &mesh_vao_);
        glBindVertexArray(mesh_vao_);
        glGenBuffers(1, &mesh_vbo_);
        glBindBuffer(GL_ARRAY_BUFFER, mesh_vbo_);
        constexpr GLsizei kStride = 9 * sizeof(float);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, kStride, reinterpret_cast<void*>(0));
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, kStride,
                              reinterpret_cast<void*>(3 * sizeof(float)));
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, kStride,
                              reinterpret_cast<void*>(6 * sizeof(float)));
        glEnableVertexAttribArray(2);
        glBindVertexArray(0);

        upload_proton_marker();

        // -- GPU propagation engine (fp32 transcription of the tested CPU
        //    tables; verified by sesolver_gpucheck). Falls back to CPU
        //    stepping when compute programs fail to build. --
        gpu_ok_ = engine_.initialize(*this, sim_.grid(),
                                     sim_.propagator().half_potential_phase(),
                                     sim_.propagator().kinetic_phase(), sim_.psi());
        if (gpu_ok_) {
            // Imaginary-time weights from the tested CPU relaxer (G7).
            const ses::ImaginaryTimePropagator3D relaxer{sim_.grid(), sim_.potential(),
                                                         kRelaxDtau};
            engine_.set_relax_tables(*this, relaxer.half_potential_weight(),
                                     relaxer.kinetic_weight(), kRelaxDtau);
        }

        // -- volume pipeline --
        volume_program_ = link_program(kVolumeVertexShader, kVolumeFragmentShader);
        vol_mvp_loc_ = glGetUniformLocation(volume_program_, "mvp");
        vol_eye_loc_ = glGetUniformLocation(volume_program_, "eye");
        vol_boxmin_loc_ = glGetUniformLocation(volume_program_, "box_min");
        vol_boxmax_loc_ = glGetUniformLocation(volume_program_, "box_max");
        vol_invpeak_loc_ = glGetUniformLocation(volume_program_, "inv_peak");
        vol_absorb_loc_ = glGetUniformLocation(volume_program_, "absorbance");
        vol_pcenter_loc_ = glGetUniformLocation(volume_program_, "proton_center");
        vol_pradius_loc_ = glGetUniformLocation(volume_program_, "proton_radius");
        vol_pcolor_loc_ = glGetUniformLocation(volume_program_, "proton_color");
        glUseProgram(volume_program_);
        glUniform1i(glGetUniformLocation(volume_program_, "psi_tex"), 0);
        glUniform1i(glGetUniformLocation(volume_program_, "phase_tex"), 1);

        upload_box_cube();
        create_textures();

        mesh_dirty_ = true;
        volume_dirty_ = true;
    }

    void paintGL() override {
        // Photon flash: a brief warm background right after a quantum jump.
        if (flash_ticks_ > 0) {
            const float w = static_cast<float>(flash_ticks_) / 25.0f;
            glClearColor(0.04f + 0.55f * w, 0.05f + 0.42f * w, 0.09f + 0.18f * w, 1.0f);
            --flash_ticks_;
        } else {
            glClearColor(0.04f, 0.05f, 0.09f, 1.0f);
        }
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        const double aspect = static_cast<double>(width()) / std::max(1, height());
        const ses::Mat4 proj =
            ses::perspective(45.0 * 3.14159265358979323846 / 180.0, aspect, 0.1, 200.0);
        const ses::Vec3d eye = ses::orbit_eye(azimuth_, elevation_, distance_, ses::Vec3d{});
        const ses::Mat4 view = ses::look_at(eye, ses::Vec3d{}, ses::Vec3d{0.0, 1.0, 0.0});
        const ses::Mat4 mvp = proj * view;
        float mvp_f[16];
        for (int i = 0; i < 16; ++i) {
            mvp_f[i] = static_cast<float>(mvp.m[i]);
        }

        // GPU stepping happens here, where the context is guaranteed current.
        if (use_gpu_path()) {
            if (cpu_is_truth_) {
                // The CPU state is authoritative here: refresh the brightness
                // normalizer from it (covers post-M collapse, post-R reset).
                double pk = 0.0;
                for (const ses::Complex<double>& z : sim_.psi().data()) {
                    pk = std::max(pk, ses::norm_sq(z));
                }
                if (pk > 0.0) {
                    peak_ = pk;
                }
                engine_.upload_state(*this, sim_.psi());
                cpu_is_truth_ = false;
                volume_dirty_ = false;  // texture comes from the bridge now
                // Bridge immediately: with an empty step queue (paused R/M,
                // first frame) the block below would never refresh psi_tex_
                // and the screen would keep the stale (or undefined) cloud.
                engine_.write_psi_texture(*this, psi_tex_);
            }
            if (pending_gpu_steps_ > 0) {
                if (stepping_ == Stepping::RealTime) {
                    if (gpu_title_due_) {
                        // GPU-reduced norm+peak (2 KB readback), taken BEFORE
                        // enqueueing new steps so the implicit sync waits
                        // only on long-finished work.
                        const ses_gpu::NormPeak np = engine_.norm_and_peak(*this);
                        norm_display_ = np.sum;
                        if (np.peak > 0.0) {
                            peak_ = np.peak;  // brightness tracks the cloud
                        }
                        // fp32 drift renormalization: the split-operator is
                        // unitary in exact arithmetic; pinning the norm back
                        // to 1 removes pure numerical decay.
                        if (np.sum > 0.0 && std::abs(np.sum - 1.0) > 1e-4) {
                            engine_.scale(*this,
                                          static_cast<float>(1.0 / std::sqrt(np.sum)));
                        }
                    }
                    engine_.step(*this, pending_gpu_steps_);
                    // Time is credited where steps EXECUTE, so a stalled or
                    // occluded paint cannot desync the clock from the state.
                    gpu_time_ += pending_gpu_steps_ * sim_.dt();

                    // T4: one Poisson decay trial per executed batch. The
                    // jump probability is weighted by the LIVE excited
                    // population (GPU inner product); on a jump the state
                    // collapses onto the cached ground state ("photon out").
                    if (decay_on_ && excited_buf_ != 0) {
                        const ses_gpu::NormPeak ip =
                            engine_.inner_with_psi(*this, excited_buf_);
                        const double p_e = ip.sum * ip.sum + ip.peak * ip.peak;
                        const double p_jump =
                            1.0 - std::exp(-kDecayGammaDisplay * p_e *
                                           pending_gpu_steps_ * sim_.dt());
                        std::uniform_real_distribution<double> uniform(0.0, 1.0);
                        if (uniform(rng_) < p_jump) {
                            engine_.copy_into_psi(*this, ground_buf_);
                            flash_ticks_ = 25;
                            ++photon_count_;
                            refresh_title();
                        }
                    }
                } else {
                    // GPU imaginary time (G7/T1): renormalized every step;
                    // the ITP estimator gives the convergence readout free.
                    // The excited flavor deflates the cached ground state.
                    const ses_gpu::GpuEngine::RelaxStats stats =
                        (stepping_ == Stepping::RelaxingExcited && ground_buf_ != 0)
                            ? engine_.relax_deflated_step(*this, {ground_buf_},
                                                          pending_gpu_steps_)
                            : engine_.relax_step(*this, pending_gpu_steps_);
                    relax_energy_display_ = stats.energy;
                    if (stats.peak > 0.0) {
                        peak_ = stats.peak;
                    }
                    norm_display_ = 1.0;  // pinned by per-step renormalization
                }
                pending_gpu_steps_ = 0;
                engine_.write_psi_texture(*this, psi_tex_);
                volume_dirty_ = false;
                if (gpu_title_due_) {
                    gpu_title_due_ = false;
                    refresh_title();
                }
            }
        }

        if (mode_ == ViewMode::Cloud) {
            if (volume_dirty_) {
                upload_volume();
                volume_dirty_ = false;
            }
            glDisable(GL_DEPTH_TEST);
            glEnable(GL_BLEND);
            glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);  // premultiplied over clear color
            glEnable(GL_CULL_FACE);
            glCullFace(GL_FRONT);  // raster the BACK faces: works from inside too

            glUseProgram(volume_program_);
            glUniformMatrix4fv(vol_mvp_loc_, 1, GL_FALSE, mvp_f);
            glUniform3f(vol_eye_loc_, static_cast<float>(eye.x), static_cast<float>(eye.y),
                        static_cast<float>(eye.z));
            const ses::Grid3D& g = sim_.grid();
            glUniform3f(vol_boxmin_loc_, static_cast<float>(g.x.xmin),
                        static_cast<float>(g.y.xmin), static_cast<float>(g.z.xmin));
            glUniform3f(vol_boxmax_loc_, static_cast<float>(g.x.xmax),
                        static_cast<float>(g.y.xmax), static_cast<float>(g.z.xmax));
            glUniform1f(vol_invpeak_loc_, static_cast<float>(peak_ > 0.0 ? 1.0 / peak_ : 0.0));
            glUniform1f(vol_absorb_loc_, static_cast<float>(absorbance_));
            // The nucleus lives INSIDE the ray marcher now (analytic sphere):
            // fogged by cloud in front, occluding cloud behind.
            glUniform3f(vol_pcenter_loc_, 0.0f, 0.0f, 0.0f);
            glUniform1f(vol_pradius_loc_, static_cast<float>(kProtonMarkerRadius));
            glUniform3f(vol_pcolor_loc_, 1.0f, 0.45f, 0.2f);

            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_3D, psi_tex_);
            glActiveTexture(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_1D, phase_tex_);

            glBindVertexArray(cube_vao_);
            glDrawArrays(GL_TRIANGLES, 0, 36);
            glBindVertexArray(0);

            glDisable(GL_CULL_FACE);
            glDisable(GL_BLEND);
        } else {
            if (mesh_dirty_) {
                upload_mesh();
                mesh_dirty_ = false;
            }
            // Surface mode keeps the mesh proton (depth-tested against the
            // opaque isosurface, so occlusion is already correct there).
            draw_proton(mvp_f, eye);
            glEnable(GL_DEPTH_TEST);
            glUseProgram(mesh_program_);
            glUniformMatrix4fv(mesh_mvp_loc_, 1, GL_FALSE, mvp_f);
            glUniform3f(mesh_eye_loc_, static_cast<float>(eye.x), static_cast<float>(eye.y),
                        static_cast<float>(eye.z));
            glBindVertexArray(mesh_vao_);
            glDrawArrays(GL_TRIANGLES, 0, vertex_count_);
            glBindVertexArray(0);
        }
    }

    void draw_proton(const float* mvp_f, const ses::Vec3d& eye) {
        glEnable(GL_DEPTH_TEST);
        glUseProgram(mesh_program_);
        glUniformMatrix4fv(mesh_mvp_loc_, 1, GL_FALSE, mvp_f);
        glUniform3f(mesh_eye_loc_, static_cast<float>(eye.x), static_cast<float>(eye.y),
                    static_cast<float>(eye.z));
        glBindVertexArray(proton_vao_);
        glDrawArrays(GL_TRIANGLES, 0, proton_vertex_count_);
        glBindVertexArray(0);
    }

    void mousePressEvent(QMouseEvent* e) override { last_pos_ = e->position(); }

    void mouseMoveEvent(QMouseEvent* e) override {
        const QPointF delta = e->position() - last_pos_;
        last_pos_ = e->position();
        azimuth_ -= 0.01 * delta.x();
        elevation_ += 0.01 * delta.y();
        elevation_ = std::clamp(elevation_, -1.5, 1.5);
        update();
    }

    void wheelEvent(QWheelEvent* e) override {
        distance_ *= std::pow(0.999, e->angleDelta().y());
        distance_ = std::clamp(distance_, 4.0, 100.0);
        update();
    }

public:
    // Control entry points, shared by the keyboard and the toolbar buttons.
    void toggle_pause() {
        paused_ = !paused_;
        after_control();
    }
    void set_real_time() {
        stepping_ = Stepping::RealTime;
        after_control();
    }
    void set_relaxing() {
        stepping_ = Stepping::Relaxing;
        if (!use_gpu_path()) {
            ensure_cpu_current();  // CPU relax (Surface view / no GPU)
        }
        after_control();
    }
    void reset_simulation() {
        sim_ = make_simulation();
        stepping_ = Stepping::RealTime;
        cpu_is_truth_ = true;  // GPU state discarded with the reset
        gpu_time_ = 0.0;
        pending_gpu_steps_ = 0;
        stage_active_view();
        after_control();
    }
    // Soft position measurement: sample from |psi|^2 (RNG lives here in the
    // shell; core takes the uniform draw) and let the sharpened packet
    // re-evolve.
    void measure_now() {
        ensure_cpu_current();
        std::uniform_real_distribution<double> uniform(0.0, 1.0);
        sim_.measure(uniform(rng_), kMeasureSigma);
        stepping_ = Stepping::RealTime;
        stage_active_view();
        after_control();
    }
    void toggle_view_mode() {
        mode_ = (mode_ == ViewMode::Cloud) ? ViewMode::Surface : ViewMode::Cloud;
        // Re-stage for the newly selected mode: its data may be stale (tick
        // only stages the active mode, and we may be paused).
        if (mode_ == ViewMode::Surface) {
            ensure_cpu_current();  // meshing reads the CPU field
            if (stepping_ == Stepping::RelaxingExcited) {
                stepping_ = Stepping::Relaxing;  // deflation is GPU-only (v1)
            }
        }
        stage_active_view();
        after_control();
    }

    // Transitions arc T1: relax into the z-aligned first excited state.
    // The ground state is computed and cached on first use (a few seconds,
    // one-time); the z-odd seed keeps the whole flow in the odd-parity
    // sector, so it converges to the 2p_z-like state deterministically.
    void relax_to_excited() {
        if (!gpu_ok_) {
            return;  // deflation runs on the GPU path only (v1)
        }
        if (mode_ != ViewMode::Cloud) {
            mode_ = ViewMode::Cloud;
        }
        if (!prepare_ground_cache()) {
            return;
        }
        sim_.set_psi(make_z_odd_seed());
        cpu_is_truth_ = true;
        stepping_ = Stepping::RelaxingExcited;
        after_control();
    }

    // Transitions arc T4: toggle spontaneous decay (quantum jumps). The
    // decay rate comes from OUR spectrum: Gamma = einstein_a(gap, |<f|r|i>|^2)
    // with both eigenstates cached; the true lifetime is ~1e8 a.u., so the
    // display runs an accelerated Gamma (title shows both, honestly).
    void toggle_decay() {
        if (!gpu_ok_) {
            return;
        }
        if (!decay_on_ && !prepare_excited_cache()) {
            return;
        }
        decay_on_ = !decay_on_;
        after_control();
    }

    long long photon_count() const { return photon_count_; }

protected:
    void after_control() {
        refresh_title();
        update();
    }

    ses::Field3D make_z_odd_seed() const {
        const ses::Grid3D& g = sim_.grid();
        ses::Field3D seed{g};
        for (int k = 0; k < g.z.n; ++k) {
            for (int j = 0; j < g.y.n; ++j) {
                for (int i = 0; i < g.x.n; ++i) {
                    const double x = g.x.coord(i);
                    const double y = g.y.coord(j);
                    const double z = g.z.coord(k);
                    const double env =
                        std::exp(-(x * x + y * y + z * z) / (4.0 * 2.0 * 2.0));
                    seed(i, j, k) = ses::Complex<double>{z * env, 0.0};
                }
            }
        }
        ses::normalize(seed);
        return seed;
    }

    // One-time ground-state computation into a cached GPU buffer (and a CPU
    // copy for later CPU-side use). Blocks for a few seconds on first call.
    bool prepare_ground_cache() {
        if (ground_buf_ != 0) {
            return true;
        }
        ensure_cpu_current();  // snapshot the user's state
        const ses::Field3D user_state = sim_.psi();

        sim_.set_psi(ses::gaussian_wavepacket(sim_.grid(), ses::Vec3d{},
                                              ses::Vec3d{2.0, 2.0, 2.0}, ses::Vec3d{}));
        makeCurrent();
        engine_.upload_state(*this, sim_.psi());
        ground_energy_ = engine_.relax_step(*this, 700).energy;
        engine_.readback(*this, readback_buf_);
        ses::Field3D ground{sim_.grid()};
        for (std::size_t i = 0; i < ground.data().size(); ++i) {
            ground.data()[i] =
                ses::Complex<double>{readback_buf_[2 * i], readback_buf_[2 * i + 1]};
        }
        ses::normalize(ground);  // fp32 round-trip polish
        ground_buf_ = engine_.create_state_buffer(*this, ground);
        doneCurrent();
        ground_cpu_.emplace(std::move(ground));

        sim_.set_psi(user_state);  // restore; next paint re-uploads
        cpu_is_truth_ = true;
        return ground_buf_ != 0;
    }

    // One-time excited-state (2p_z) cache + first-principles decay rate:
    // gap from the two ITP energy estimates, dipole strength from the cached
    // wavefunctions, Gamma_true = einstein_a(gap, strength).
    bool prepare_excited_cache() {
        if (excited_buf_ != 0) {
            return true;
        }
        if (!prepare_ground_cache()) {
            return false;
        }
        ensure_cpu_current();
        const ses::Field3D user_state = sim_.psi();

        sim_.set_psi(make_z_odd_seed());
        makeCurrent();
        engine_.upload_state(*this, sim_.psi());
        excited_energy_ =
            engine_.relax_deflated_step(*this, {ground_buf_}, 700).energy;
        engine_.readback(*this, readback_buf_);
        ses::Field3D excited{sim_.grid()};
        for (std::size_t i = 0; i < excited.data().size(); ++i) {
            excited.data()[i] =
                ses::Complex<double>{readback_buf_[2 * i], readback_buf_[2 * i + 1]};
        }
        ses::normalize(excited);
        excited_buf_ = engine_.create_state_buffer(*this, excited);
        doneCurrent();
        excited_cpu_.emplace(std::move(excited));

        const double gap = excited_energy_ - ground_energy_;
        const ses::DipoleMatrixElement d =
            ses::dipole_matrix_element(*excited_cpu_, *ground_cpu_);
        gamma_true_ = ses::einstein_a(gap, ses::dipole_strength_sq(d));

        sim_.set_psi(user_state);
        cpu_is_truth_ = true;
        return excited_buf_ != 0;
    }

    void keyPressEvent(QKeyEvent* e) override {
        switch (e->key()) {
            case Qt::Key_Space:
                toggle_pause();
                break;
            case Qt::Key_1:
                set_real_time();
                break;
            case Qt::Key_2:
                set_relaxing();
                break;
            case Qt::Key_3:
                relax_to_excited();
                break;
            case Qt::Key_R:
                reset_simulation();
                break;
            case Qt::Key_M:
                measure_now();
                break;
            case Qt::Key_D:
                toggle_decay();
                break;
            case Qt::Key_Tab:
                toggle_view_mode();
                break;
            case Qt::Key_BracketLeft:
                absorbance_ = std::max(0.1, absorbance_ / 1.3);
                after_control();
                break;
            case Qt::Key_BracketRight:
                absorbance_ = std::min(50.0, absorbance_ * 1.3);
                after_control();
                break;
            default:
                QOpenGLWidget::keyPressEvent(e);
                return;
        }
    }

private:
    void tick() {
        if (paused_) {
            return;
        }
        if (use_gpu_path()) {
            // Steps run in paintGL (context current there). Cap the backlog
            // so a stalled paint cannot spiral; time is credited at
            // execution, so dropped ticks drop cleanly.
            pending_gpu_steps_ = std::min(pending_gpu_steps_ + kStepsPerTick,
                                          kMaxPendingGpuSteps);
            if (++ticks_ % 10 == 0) {
                gpu_title_due_ = true;
            }
            update();
            return;
        }
        ensure_cpu_current();
        if (stepping_ == Stepping::RealTime) {
            sim_.advance(kStepsPerTick);
        } else {
            sim_.relax(kRelaxStepsPerTick, kRelaxDtau);
        }
        stage_active_view();
        if (++ticks_ % 10 == 0) {
            norm_display_ = ses::norm_sq(sim_.psi());
            refresh_title();
        }
        update();
    }

    // Pull the GPU-evolved state back into the CPU double session (fp32
    // precision at the handoff -- the display path's documented tradeoff).
    void ensure_cpu_current() {
        // Queued-but-unexecuted steps were never credited to gpu_time_;
        // discard them so they cannot fire later against a different state.
        pending_gpu_steps_ = 0;
        if (cpu_is_truth_ || !gpu_ok_) {
            return;
        }
        makeCurrent();
        engine_.readback(*this, readback_buf_);
        doneCurrent();
        ses::Field3D f{sim_.grid()};
        for (std::size_t i = 0; i < f.data().size(); ++i) {
            f.data()[i] = ses::Complex<double>{readback_buf_[2 * i], readback_buf_[2 * i + 1]};
        }
        sim_.set_psi(f);
        cpu_is_truth_ = true;
    }

    void stage_active_view() {
        if (mode_ == ViewMode::Cloud) {
            if (use_gpu_path()) {
                return;  // paintGL uploads the state and bridges the texture
            }
            stage_volume();
            volume_dirty_ = true;
        } else {
            remesh();
            mesh_dirty_ = true;
        }
    }

    // ---- CPU staging (no GL context needed) ----

    void remesh() {
        mesh_ = ses::marching_cubes_at_fraction(sim_.density(), sim_.grid(), kIsoFraction);
        colors_ = ses::phase_colors(mesh_, sim_.psi());
    }

    // Pack complex psi into RG float pairs and track the density peak.
    void stage_volume() {
        const auto& data = sim_.psi().data();
        psi_staging_.resize(data.size() * 2);
        double peak = 0.0;
        for (std::size_t i = 0; i < data.size(); ++i) {
            psi_staging_[2 * i] = static_cast<float>(data[i].real());
            psi_staging_[2 * i + 1] = static_cast<float>(data[i].imag());
            peak = std::max(peak, ses::norm_sq(data[i]));
        }
        peak_ = peak;
    }

    void refresh_title() {
        // Convergence readout while relaxing: exact <H> on the CPU session,
        // or the free ITP estimator (-ln||psi||^2 / 2 dtau) on the GPU path.
        QString energy;
        if (stepping_ != Stepping::RealTime) {
            energy = cpu_is_truth_
                         ? QStringLiteral("E = %1 Ha   ")
                               .arg(ses::mean_energy(sim_.psi(), sim_.potential()), 0,
                                    'f', 4)
                         : QStringLiteral("E ~ %1 Ha   ")
                               .arg(relax_energy_display_, 0, 'f', 4);
        }
        window()->setWindowTitle(
            QStringLiteral("Electron near a soft-Coulomb nucleus   t = %1   %2"
                           "norm = %3   [%4, %5, %6]  1=real 2=relax R=reset tab=view "
                           "[ ]=density M=measure")
                .arg(sim_.time() + gpu_time_, 0, 'f', 2)
                .arg(energy)
                .arg(norm_display_, 0, 'f', 6)
                .arg(mode_ == ViewMode::Cloud ? QStringLiteral("cloud")
                                              : QStringLiteral("surface"))
                .arg(stepping_ == Stepping::RealTime
                         ? QStringLiteral("real-time")
                         : (stepping_ == Stepping::Relaxing
                                ? QStringLiteral("relaxing->1s")
                                : QStringLiteral("relaxing->2p")))
                .arg(use_gpu_path() ? QStringLiteral("gpu 128^3")
                                    : QStringLiteral("cpu 128^3")) +
            (decay_on_
                 ? QStringLiteral("  decay ON: tau_true %1 au, x%2, photons %3")
                       .arg(gamma_true_ > 0.0 ? 1.0 / gamma_true_ : 0.0, 0, 'e', 2)
                       .arg(gamma_true_ > 0.0 ? kDecayGammaDisplay / gamma_true_ : 0.0,
                            0, 'e', 1)
                       .arg(photon_count_)
                 : QString()));
    }

    // ---- GL uploads (current context required: called from initializeGL/paintGL) ----

    // Static warm-colored sphere at the nucleus, in the mesh vertex format.
    void upload_proton_marker() {
        const ses::Mesh sphere = ses::sphere_mesh(ses::Vec3d{}, kProtonMarkerRadius, 16, 24);
        std::vector<float> interleaved;
        interleaved.reserve(sphere.vertices.size() * 9);
        for (std::size_t i = 0; i < sphere.vertices.size(); ++i) {
            const ses::Vec3d& p = sphere.vertices[i];
            const ses::Vec3d& n = sphere.normals[i];
            interleaved.push_back(static_cast<float>(p.x));
            interleaved.push_back(static_cast<float>(p.y));
            interleaved.push_back(static_cast<float>(p.z));
            interleaved.push_back(static_cast<float>(n.x));
            interleaved.push_back(static_cast<float>(n.y));
            interleaved.push_back(static_cast<float>(n.z));
            interleaved.push_back(1.0f);   // warm proton color
            interleaved.push_back(0.45f);
            interleaved.push_back(0.20f);
        }
        proton_vertex_count_ = static_cast<int>(sphere.vertices.size());
        glGenVertexArrays(1, &proton_vao_);
        glBindVertexArray(proton_vao_);
        glGenBuffers(1, &proton_vbo_);
        glBindBuffer(GL_ARRAY_BUFFER, proton_vbo_);
        glBufferData(GL_ARRAY_BUFFER,
                     static_cast<GLsizeiptr>(interleaved.size() * sizeof(float)),
                     interleaved.data(), GL_STATIC_DRAW);
        constexpr GLsizei kStride = 9 * sizeof(float);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, kStride, reinterpret_cast<void*>(0));
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, kStride,
                              reinterpret_cast<void*>(3 * sizeof(float)));
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, kStride,
                              reinterpret_cast<void*>(6 * sizeof(float)));
        glEnableVertexAttribArray(2);
        glBindVertexArray(0);
    }

    void upload_box_cube() {
        const ses::Grid3D& g = sim_.grid();
        const float lo[3] = {static_cast<float>(g.x.xmin), static_cast<float>(g.y.xmin),
                             static_cast<float>(g.z.xmin)};
        const float hi[3] = {static_cast<float>(g.x.xmax), static_cast<float>(g.y.xmax),
                             static_cast<float>(g.z.xmax)};
        // 12 triangles, CCW seen from OUTSIDE (verified per-face normals).
        const float v[] = {
            lo[0],lo[1],lo[2], lo[0],lo[1],hi[2], lo[0],hi[1],hi[2],
            lo[0],lo[1],lo[2], lo[0],hi[1],hi[2], lo[0],hi[1],lo[2],
            hi[0],lo[1],lo[2], hi[0],hi[1],lo[2], hi[0],hi[1],hi[2],
            hi[0],lo[1],lo[2], hi[0],hi[1],hi[2], hi[0],lo[1],hi[2],
            lo[0],lo[1],lo[2], hi[0],lo[1],lo[2], hi[0],lo[1],hi[2],
            lo[0],lo[1],lo[2], hi[0],lo[1],hi[2], lo[0],lo[1],hi[2],
            lo[0],hi[1],lo[2], lo[0],hi[1],hi[2], hi[0],hi[1],hi[2],
            lo[0],hi[1],lo[2], hi[0],hi[1],hi[2], hi[0],hi[1],lo[2],
            lo[0],lo[1],lo[2], lo[0],hi[1],lo[2], hi[0],hi[1],lo[2],
            lo[0],lo[1],lo[2], hi[0],hi[1],lo[2], hi[0],lo[1],lo[2],
            lo[0],lo[1],hi[2], hi[0],lo[1],hi[2], hi[0],hi[1],hi[2],
            lo[0],lo[1],hi[2], hi[0],hi[1],hi[2], lo[0],hi[1],hi[2],
        };
        glGenVertexArrays(1, &cube_vao_);
        glBindVertexArray(cube_vao_);
        glGenBuffers(1, &cube_vbo_);
        glBindBuffer(GL_ARRAY_BUFFER, cube_vbo_);
        glBufferData(GL_ARRAY_BUFFER, sizeof(v), v, GL_STATIC_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float),
                              reinterpret_cast<void*>(0));
        glEnableVertexAttribArray(0);
        glBindVertexArray(0);
    }

    void create_textures() {
        const ses::Grid3D& g = sim_.grid();

        glGenTextures(1, &psi_tex_);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_3D, psi_tex_);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
        glTexImage3D(GL_TEXTURE_3D, 0, GL_RG32F, g.x.n, g.y.n, g.z.n, 0, GL_RG, GL_FLOAT,
                     nullptr);

        // The TESTED cyclic phase colormap, baked to a repeating 1D texture.
        const std::vector<ses::Rgb> lut = ses::phase_lut(kPhaseLutSize);
        std::vector<float> lut_f;
        lut_f.reserve(lut.size() * 3);
        for (const ses::Rgb& c : lut) {
            lut_f.push_back(static_cast<float>(c.r));
            lut_f.push_back(static_cast<float>(c.g));
            lut_f.push_back(static_cast<float>(c.b));
        }
        glGenTextures(1, &phase_tex_);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_1D, phase_tex_);
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_REPEAT);  // cyclic!
        glTexImage1D(GL_TEXTURE_1D, 0, GL_RGB32F, kPhaseLutSize, 0, GL_RGB, GL_FLOAT,
                     lut_f.data());
    }

    void upload_volume() {
        const ses::Grid3D& g = sim_.grid();
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_3D, psi_tex_);
        glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, g.x.n, g.y.n, g.z.n, GL_RG, GL_FLOAT,
                        psi_staging_.data());
    }

    void upload_mesh() {
        std::vector<float> interleaved;
        interleaved.reserve(mesh_.vertices.size() * 9);
        for (std::size_t i = 0; i < mesh_.vertices.size(); ++i) {
            const ses::Vec3d& p = mesh_.vertices[i];
            const ses::Vec3d& n = mesh_.normals[i];
            const ses::Rgb& c = colors_[i];
            interleaved.push_back(static_cast<float>(p.x));
            interleaved.push_back(static_cast<float>(p.y));
            interleaved.push_back(static_cast<float>(p.z));
            interleaved.push_back(static_cast<float>(n.x));
            interleaved.push_back(static_cast<float>(n.y));
            interleaved.push_back(static_cast<float>(n.z));
            interleaved.push_back(static_cast<float>(c.r));
            interleaved.push_back(static_cast<float>(c.g));
            interleaved.push_back(static_cast<float>(c.b));
        }
        vertex_count_ = static_cast<int>(mesh_.vertices.size());
        glBindBuffer(GL_ARRAY_BUFFER, mesh_vbo_);
        glBufferData(GL_ARRAY_BUFFER,
                     static_cast<GLsizeiptr>(interleaved.size() * sizeof(float)),
                     interleaved.data(), GL_DYNAMIC_DRAW);
    }

    GLuint compile_shader(GLenum type, const char* src) {
        const GLuint shader = glCreateShader(type);
        glShaderSource(shader, 1, &src, nullptr);
        glCompileShader(shader);
        GLint ok = GL_FALSE;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
        if (ok != GL_TRUE) {
            char log[2048];
            glGetShaderInfoLog(shader, sizeof(log), nullptr, log);
            fatal_gl_error("shader compile failed", QString::fromLatin1(log));
        }
        return shader;
    }

    GLuint link_program(const char* vs_src, const char* fs_src) {
        const GLuint vs = compile_shader(GL_VERTEX_SHADER, vs_src);
        const GLuint fs = compile_shader(GL_FRAGMENT_SHADER, fs_src);
        const GLuint prog = glCreateProgram();
        glAttachShader(prog, vs);
        glAttachShader(prog, fs);
        glLinkProgram(prog);
        GLint ok = GL_FALSE;
        glGetProgramiv(prog, GL_LINK_STATUS, &ok);
        if (ok != GL_TRUE) {
            char log[2048];
            glGetProgramInfoLog(prog, sizeof(log), nullptr, log);
            fatal_gl_error("program link failed", QString::fromLatin1(log));
        }
        glDeleteShader(vs);
        glDeleteShader(fs);
        return prog;
    }

    ses::WavepacketSimulation sim_;
    ViewMode mode_ = ViewMode::Cloud;
    Stepping stepping_ = Stepping::RealTime;

    // GPU stepping state (docs/GPU_PLAN.md G5). cpu_is_truth_ is the single
    // sync invariant: true -> sim_.psi() is current, false -> the engine's
    // SSBO is ahead and must be read back before any CPU-side operation.
    ses_gpu::GpuEngine engine_;
    bool gpu_ok_ = false;
    bool cpu_is_truth_ = true;
    int pending_gpu_steps_ = 0;
    bool gpu_title_due_ = false;
    double gpu_time_ = 0.0;
    double norm_display_ = 1.0;
    double relax_energy_display_ = 0.0;
    std::vector<float> readback_buf_;

    // Transitions arc T1/T4: cached eigenstates and decay bookkeeping.
    GLuint ground_buf_ = 0;
    std::optional<ses::Field3D> ground_cpu_;
    GLuint excited_buf_ = 0;
    std::optional<ses::Field3D> excited_cpu_;
    double ground_energy_ = 0.0;
    double excited_energy_ = 0.0;
    double gamma_true_ = 0.0;
    bool decay_on_ = false;
    int flash_ticks_ = 0;
    long long photon_count_ = 0;

    ses::Mesh mesh_;
    std::vector<ses::Rgb> colors_;
    std::vector<float> psi_staging_;
    double peak_ = 0.0;
    double absorbance_ = 1.5;

    QTimer timer_;
    bool paused_ = false;
    bool mesh_dirty_ = false;
    bool volume_dirty_ = false;
    long long ticks_ = 0;

    GLuint mesh_program_ = 0;
    GLuint mesh_vao_ = 0;
    GLuint mesh_vbo_ = 0;
    GLint mesh_mvp_loc_ = -1;
    GLint mesh_eye_loc_ = -1;
    int vertex_count_ = 0;

    GLuint proton_vao_ = 0;
    GLuint proton_vbo_ = 0;
    int proton_vertex_count_ = 0;
    std::mt19937 rng_{std::random_device{}()};

    GLuint volume_program_ = 0;
    GLuint cube_vao_ = 0;
    GLuint cube_vbo_ = 0;
    GLuint psi_tex_ = 0;
    GLuint phase_tex_ = 0;
    GLint vol_mvp_loc_ = -1;
    GLint vol_eye_loc_ = -1;
    GLint vol_boxmin_loc_ = -1;
    GLint vol_boxmax_loc_ = -1;
    GLint vol_invpeak_loc_ = -1;
    GLint vol_absorb_loc_ = -1;
    GLint vol_pcenter_loc_ = -1;
    GLint vol_pradius_loc_ = -1;
    GLint vol_pcolor_loc_ = -1;

    double azimuth_ = 0.6;
    double elevation_ = 0.4;
    double distance_ = 28.0;
    QPointF last_pos_;
};

}  // namespace

int main(int argc, char** argv) {
    QSurfaceFormat format;
    format.setVersion(4, 3);
    format.setProfile(QSurfaceFormat::CoreProfile);
    format.setDepthBufferSize(24);
    format.setSamples(4);
    QSurfaceFormat::setDefaultFormat(format);

    QApplication app(argc, argv);

    QMainWindow window;
    window.setWindowTitle(QStringLiteral("Electron wavepacket near a soft-Coulomb nucleus"));
    auto* viewport = new Viewport();
    window.setCentralWidget(viewport);

    // Discoverable controls: toolbar buttons mirroring the hotkeys (the
    // buttons keep Qt::NoFocus by default, so the keys stay live).
    QToolBar* controls = window.addToolBar(QStringLiteral("Controls"));
    controls->setMovable(false);
    controls->addAction(QStringLiteral("Measure (M)"), viewport,
                        [viewport] { viewport->measure_now(); });
    controls->addSeparator();
    controls->addAction(QStringLiteral("Real time (1)"), viewport,
                        [viewport] { viewport->set_real_time(); });
    controls->addAction(QStringLiteral("Relax to ground state (2)"), viewport,
                        [viewport] { viewport->set_relaxing(); });
    controls->addAction(QStringLiteral("Relax to 2p (3)"), viewport,
                        [viewport] { viewport->relax_to_excited(); });
    controls->addAction(QStringLiteral("Decay (D)"), viewport,
                        [viewport] { viewport->toggle_decay(); });
    controls->addSeparator();
    controls->addAction(QStringLiteral("Reset packet (R)"), viewport,
                        [viewport] { viewport->reset_simulation(); });
    controls->addAction(QStringLiteral("Cloud/Surface (Tab)"), viewport,
                        [viewport] { viewport->toggle_view_mode(); });
    controls->addAction(QStringLiteral("Pause (Space)"), viewport,
                        [viewport] { viewport->toggle_pause(); });

    window.resize(1024, 768);
    window.show();
    // StrongFocus only ACCEPTS focus; grab it so the 1/2/R/M/space keys work
    // immediately without a click.
    viewport->setFocus();

    // Headless-ish regression of the decay demo arc: prepare 2p, return to
    // real time, enable decay, and require at least one quantum jump.
    // (After the first jump the atom sits in 1s with P_e ~ 0, so exactly one
    // photon is the physically expected outcome without a re-pump laser.)
    if (app.arguments().contains(QStringLiteral("--selftest-decay"))) {
        QTimer::singleShot(500, viewport, [viewport] { viewport->relax_to_excited(); });
        QTimer::singleShot(14000, viewport, [viewport] { viewport->set_real_time(); });
        QTimer::singleShot(15000, viewport, [viewport] { viewport->toggle_decay(); });
        QTimer::singleShot(45000, viewport, [viewport, &app] {
            const long long photons = viewport->photon_count();
            std::fprintf(stderr, "selftest-decay: photons = %lld  [%s]\n", photons,
                         photons >= 1 ? "PASS" : "FAIL");
            app.exit(photons >= 1 ? 0 : 1);
        });
    }

    return app.exec();
}
