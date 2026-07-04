// Humble Object shell -- the Qt + hand-written OpenGL boundary.
//
// NO domain logic lives here (docs/ARCHITECTURE.md): the electron wavepacket
// evolves in core's WavepacketSimulation (split-operator TDSE), the
// isosurface mesh comes from core's marching cubes, matrices from core's
// camera math, colors from core's colormaps. This file owns the window, the
// GL context, buffer uploads, shaders, the frame timer, and input glue. It
// is verified by eye, not by unit tests (the Humble Object pattern).
//
// v3 deliverable: REAL-TIME DYNAMICS WITH PHASE COLORING -- a Gaussian
// electron wavepacket released beside a soft-Coulomb nucleus, re-meshed
// every frame, with arg(psi) painted on the surface through the cyclic
// colormap: the packet's momentum shows up as color stripes, and the
// stationary parts cycle hue at the local energy (e^{-iEt}).
// Controls: drag = orbit, wheel = zoom, space = pause/resume.

#include <core/camera.hpp>
#include <core/colormap.hpp>
#include <core/field.hpp>
#include <core/grid.hpp>
#include <core/marching_cubes.hpp>
#include <core/potential.hpp>
#include <core/sampling.hpp>
#include <core/simulation.hpp>
#include <core/vec.hpp>

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
#include <QWheelEvent>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <utility>
#include <vector>

namespace {

// Fatal GL-setup failure: log to stderr (terminal launches) AND show a modal
// dialog (double-click launches, where the console vanishes with the abort),
// then exit. A QApplication is always live by the time GL init runs.
[[noreturn]] void fatal_gl_error(const char* stage, const QString& detail) {
    qCritical("%s: %s", stage, qPrintable(detail));
    QMessageBox::critical(nullptr, QStringLiteral("OpenGL error"),
                          QStringLiteral("%1\n\n%2").arg(QLatin1String(stage), detail));
    std::exit(EXIT_FAILURE);
}

// The scenario, built entirely from core pieces: a bound-ish packet released
// 3 Bohr from a soft-Coulomb nucleus with a little tangential momentum.
ses::WavepacketSimulation make_simulation() {
    const ses::Grid1D axis{-12.0, 12.0, 32};
    const ses::Grid3D grid{axis, axis, axis};
    return ses::WavepacketSimulation{ses::WavepacketSimulation::Config{
        grid,
        ses::soft_coulomb_potential(grid, 1.0, 1.0, ses::Vec3d{}),
        ses::Vec3d{3.0, 0.0, 0.0},  // r0: beside the nucleus
        ses::Vec3d{1.5, 1.5, 1.5},  // sigma
        ses::Vec3d{0.0, 0.4, 0.0},  // k0: tangential kick
        0.02,                       // dt
    }};
}

constexpr int kStepsPerTick = 2;
constexpr int kTickMs = 16;
constexpr double kIsoFraction = 0.25;

const char* kVertexShader = R"(#version 430 core
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

const char* kFragmentShader = R"(#version 430 core
in vec3 v_normal;
in vec3 v_pos;
in vec3 v_color;
uniform vec3 eye;
out vec4 frag;
void main() {
    vec3 n = normalize(v_normal);
    vec3 vdir = normalize(eye - v_pos);
    float diffuse = abs(dot(n, vdir));               // two-sided headlight
    float spec = pow(max(dot(n, vdir), 0.0), 32.0);  // light rides the camera
    vec3 c = v_color * (0.20 + 0.80 * diffuse) + vec3(0.25) * spec;
    frag = vec4(c, 1.0);
}
)";

class Viewport : public QOpenGLWidget, protected QOpenGLFunctions_4_3_Core {
public:
    explicit Viewport(QWidget* parent = nullptr)
        : QOpenGLWidget(parent), sim_(make_simulation()) {
        setFocusPolicy(Qt::StrongFocus);  // receive the spacebar
        remesh();
        connect(&timer_, &QTimer::timeout, this, &Viewport::tick);
        timer_.start(kTickMs);
    }

protected:
    void initializeGL() override {
        // setVersion(4,3) is only a REQUEST: drivers without 4.3 core (RDP,
        // VMs, old iGPUs) silently deliver less, initializeOpenGLFunctions()
        // returns false, and every wrapped GL call would then dereference a
        // null backend pointer. Fail loudly instead.
        if (!initializeOpenGLFunctions()) {
            const QSurfaceFormat got = context()->format();
            fatal_gl_error("OpenGL 4.3 core profile is required",
                           QStringLiteral("The driver provided only %1.%2. Update the GPU "
                                          "driver or run on hardware with OpenGL 4.3 support.")
                               .arg(got.majorVersion())
                               .arg(got.minorVersion()));
        }
        glEnable(GL_DEPTH_TEST);
        glClearColor(0.04f, 0.05f, 0.09f, 1.0f);

        program_ = link_program(kVertexShader, kFragmentShader);
        mvp_loc_ = glGetUniformLocation(program_, "mvp");
        eye_loc_ = glGetUniformLocation(program_, "eye");

        glGenVertexArrays(1, &vao_);
        glBindVertexArray(vao_);
        glGenBuffers(1, &vbo_);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_);
        constexpr GLsizei kStride = 9 * sizeof(float);  // pos3 + normal3 + color3
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, kStride,
                              reinterpret_cast<void*>(0));
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, kStride,
                              reinterpret_cast<void*>(3 * sizeof(float)));
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, kStride,
                              reinterpret_cast<void*>(6 * sizeof(float)));
        glEnableVertexAttribArray(2);
        glBindVertexArray(0);
        mesh_dirty_ = true;
    }

    void paintGL() override {
        if (mesh_dirty_) {
            upload_mesh();
            mesh_dirty_ = false;
        }

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        const double aspect =
            static_cast<double>(width()) / std::max(1, height());
        const ses::Mat4 proj = ses::perspective(45.0 * 3.14159265358979323846 / 180.0,
                                                aspect, 0.1, 200.0);
        const ses::Vec3d eye = ses::orbit_eye(azimuth_, elevation_, distance_, ses::Vec3d{});
        const ses::Mat4 view = ses::look_at(eye, ses::Vec3d{}, ses::Vec3d{0.0, 1.0, 0.0});
        const ses::Mat4 mvp = proj * view;

        float mvp_f[16];
        for (int i = 0; i < 16; ++i) {
            mvp_f[i] = static_cast<float>(mvp.m[i]);
        }

        glUseProgram(program_);
        glUniformMatrix4fv(mvp_loc_, 1, GL_FALSE, mvp_f);  // column-major, no transpose
        glUniform3f(eye_loc_, static_cast<float>(eye.x), static_cast<float>(eye.y),
                    static_cast<float>(eye.z));

        glBindVertexArray(vao_);
        glDrawArrays(GL_TRIANGLES, 0, vertex_count_);
        glBindVertexArray(0);
    }

    void mousePressEvent(QMouseEvent* e) override { last_pos_ = e->position(); }

    void mouseMoveEvent(QMouseEvent* e) override {
        const QPointF delta = e->position() - last_pos_;
        last_pos_ = e->position();
        azimuth_ -= 0.01 * delta.x();
        elevation_ += 0.01 * delta.y();
        elevation_ = std::clamp(elevation_, -1.5, 1.5);  // stay off the poles
        update();
    }

    void wheelEvent(QWheelEvent* e) override {
        distance_ *= std::pow(0.999, e->angleDelta().y());
        distance_ = std::clamp(distance_, 4.0, 100.0);
        update();
    }

    void keyPressEvent(QKeyEvent* e) override {
        if (e->key() == Qt::Key_Space) {
            paused_ = !paused_;
            refresh_title();
        } else {
            QOpenGLWidget::keyPressEvent(e);
        }
    }

private:
    void tick() {
        if (paused_) {
            return;
        }
        sim_.advance(kStepsPerTick);
        remesh();
        mesh_dirty_ = true;
        if (++ticks_ % 10 == 0) {
            refresh_title();
        }
        update();
    }

    // CPU side only -- safe to call with no GL context current.
    void remesh() {
        mesh_ = ses::marching_cubes_at_fraction(sim_.density(), sim_.grid(), kIsoFraction);
        colors_ = ses::phase_colors(mesh_, sim_.psi());
    }

    void refresh_title() {
        window()->setWindowTitle(
            QStringLiteral("Electron wavepacket near a soft-Coulomb nucleus   "
                           "t = %1 a.u.   norm = %2   %3")
                .arg(sim_.time(), 0, 'f', 2)
                .arg(ses::norm_sq(sim_.psi()), 0, 'f', 9)
                .arg(paused_ ? QStringLiteral("[paused - space resumes]")
                             : QStringLiteral("[space pauses]")));
    }

    // Requires a current GL context (called from paintGL only).
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
        glBindBuffer(GL_ARRAY_BUFFER, vbo_);
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
    ses::Mesh mesh_;
    std::vector<ses::Rgb> colors_;
    QTimer timer_;
    bool paused_ = false;
    bool mesh_dirty_ = false;
    long long ticks_ = 0;

    GLuint program_ = 0;
    GLuint vao_ = 0;
    GLuint vbo_ = 0;
    GLint mvp_loc_ = -1;
    GLint eye_loc_ = -1;
    int vertex_count_ = 0;

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
    window.setCentralWidget(new Viewport());
    window.resize(1024, 768);
    window.show();

    return app.exec();
}
