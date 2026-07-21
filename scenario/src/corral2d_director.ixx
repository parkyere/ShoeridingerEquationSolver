module;
#include <algorithm>
#include <cstddef>
#include <cmath>
#include <complex>
#include <memory>
#include <string>
#include <utility>
#include <vector>
export module ses.scenario.corral2d_director;
export import ses.scenario.lattice2d_director;
import ses.heightfield;
import ses.potential;
import ses.imaginary_time;
import ses.observables;
import ses.propagator;
import ses.parallel;
import ses.vk.engine_blobs;


// Quantum corral (Crommie, Lutz & Eigler, IBM 1993): 48 Fe adatoms on
// Cu(111). B = 0 -> spectral split-operator (physics-first; not Peierls).
// Contracts: tests/corral2d_test.cpp.


export namespace ses_shell {

constexpr double kCr2dBox = 16.0;
constexpr double kHaToEv = 27.211386;  // atomic-unit energy -> eV
constexpr int kCr2dN = 512;
constexpr int kCr2dNz = 4;
constexpr double kCr2dZHalf = 2.0;
constexpr double kCr2dSurfH = 6.0;
constexpr int kCr2dMeshStride = 1;
constexpr double kCr2dDt = 0.02;
// Anneal coarse -> fine: fine dtau polishes the Strang [T,V] bias before capture.
constexpr double kCr2dCoarseDtau = 0.05;
constexpr double kCr2dFineDtau = 0.005;
constexpr int kCr2dAtoms = 48;
constexpr double kCr2dR = 10.0;
constexpr double kCr2dRMin = 6.0;
constexpr double kCr2dRMax = 12.0;
constexpr double kCr2dBumpA = 1.5;
constexpr double kCr2dBumpSigma = 0.6;
// Cu(111) surface-state effective mass (Crommie et al.), atomic units.
constexpr double kCr2dMass = 0.38;
// outer absorber width (Bohr)
constexpr double kCr2dEdgeW = 3.0;
// leaky-fence absorber peak (partial absorption)
constexpr double kCr2dFenceW0 = 0.8;
constexpr int kCr2dMaxStates = 6;
constexpr double kCr2dConvTol = 1e-7;
constexpr double kCr2dCoarseTol = 1e-5;

class Corral2DDirector final : public Lattice2DDirectorBase,
                               public CorralApi {
public:
    Corral2DDirector()
        : Lattice2DDirectorBase(kCr2dBox, kCr2dN, kCr2dNz, kCr2dZHalf) {
        rebuild_potential();
        start_relax(0);
    }

    // B = 0 spectral scene with the Cu(111) effective mass m* = 0.38: GPU is
    // the mass-aware split-operator Engine (real time, damp absorber) + its
    // deflated ITP (relax). Not the base Peierls engine. The relax tables come
    // from the CPU ITP (mass baked); the disc-cut projection runs on the CPU
    // between chunks (re-upload). vkcheck: check_engine_planar_mass + _relax +
    // _deflation + _project.
    void init_compute(ses_vk::DeviceContext& ctx, bool device_ok,
                      std::int64_t /*free_vram*/) override {
        compute_attempted_ = true;
        if (device_ok &&
            eng_.initialize(ctx, phys_grid_, ses_vk::engine_blobs(kCr2dN), v_,
                            kCr2dDt, psi_.data(), kCr2dMass)) {
            eng_.set_absorber(damp_);  // real-time fence + open boundary
            eng_ok_ = true;
            eng_dirty_ = false;
        }
    }
    void release_gpu() override {
        for (int h : defl_) {
            eng_.release_state(h);
        }
        defl_.clear();
        eng_.destroy();
        eng_ok_ = false;
        eng_dirty_ = true;
        relax_tables_stage_ = -1;
    }

    CorralApi* corral() override { return this; }

    // ---- CorralApi ----
    void set_radius(double r) override {
        radius_ = std::clamp(r, kCr2dRMin, kCr2dRMax);
        rebuild_potential();
        start_relax(0);
    }
    double radius() const override { return radius_; }
    double mass() const override { return kCr2dMass; }
    void relax_next() override {
        if (!relaxing_ &&
            static_cast<int>(captured_.size()) < kCr2dMaxStates) {
            start_relax(static_cast<int>(captured_.size()));
        }
    }
    int captured() const override {
        return static_cast<int>(captured_.size());
    }
    double energy(int k) const override {
        return k >= 0 && k < static_cast<int>(energies_.size())
                   ? energies_[static_cast<std::size_t>(k)]
                   : 0.0;
    }
    double confinement() const override {
        double inside = 0.0;
        double total = 0.0;
        for (int j = 0; j < kCr2dN; ++j) {
            for (int i = 0; i < kCr2dN; ++i) {
                const double w = std::norm(psi_(i, j, 0));
                total += w;
                if (std::hypot(phys_grid_.x.coord(i),
                               phys_grid_.y.coord(j)) < radius_ - 1.0) {
                    inside += w;
                }
            }
        }
        return total > 0.0 ? inside / total : 0.0;
    }
    bool relaxing() const override { return relaxing_; }
    void fire_packet() override {
        disp_peak_ = 0.0;  // re-snap surface normalizer
        relaxing_ = false;
        ses::parallel_for(kCr2dN, [&](int j) {
            const double y = phys_grid_.y.coord(j);
            for (int i = 0; i < kCr2dN; ++i) {
                const double x = phys_grid_.x.coord(i);
                psi_(i, j, 0) =
                    std::exp(-(x * x + y * y) / (4.0 * 1.5 * 1.5));
            }
        });
        ses::normalize(psi_);
        eng_dirty_ = true;  // packet built on the CPU: re-upload before stepping
        track_peak();
        mark_fired();
    }

    // CONTRACT: tests/corral2d_test.cpp FermiWaveIsQuasiStationary.
    void fermi_wave() override {
        // Node count fence-limited: highest J0 zero the fence still quasi-confines.
        const double j0_n = 10.1734681350627;  // 3rd zero of J0
        const double kf = j0_n / radius_;
        disp_peak_ = 0.0;  // re-snap surface normalizer
        relaxing_ = false;
        ses::parallel_for(kCr2dN, [&](int j) {
            const double y = phys_grid_.y.coord(j);
            for (int i = 0; i < kCr2dN; ++i) {
                const double x = phys_grid_.x.coord(i);
                const double r = std::sqrt(x * x + y * y);
                double w = 1.0;
                if (r > radius_) {
                    const double t = (r - radius_) / (2.0 * kCr2dBumpSigma);
                    w = t >= 1.0 ? 0.0 : std::cos(0.5 * 3.14159265358979323846 * t);
                    w *= w;
                }
                psi_(i, j, 0) = w * std::cyl_bessel_j(0.0, kf * r);
            }
        });
        ses::normalize(psi_);
        eng_dirty_ = true;  // wave built on the CPU: re-upload before stepping
        track_peak();
        mark_fired();
        title_dirty_ = true;
    }

    void reset_simulation() override {
        captured_.clear();
        energies_.clear();
        release_gpu_deflation();
        start_relax(0);
    }

    bool handle_key(char key) override {
        if (key == '5') {
            fermi_wave();
            return true;
        }
        if (key == '2') {
            reset_simulation();
            return true;
        }
        if (key == '3') {
            relax_next();
            return true;
        }
        if (key == 'F') {
            fire_packet();
            return true;
        }
        return false;
    }

    double sim_dt() const override { return kCr2dDt; }

    // ---- STM-style surface display ----
    bool cloud() const override { return false; }
    const ses::Mesh& mesh() const override { return hf_.mesh; }
    const std::vector<ses::Rgb>& colors() const override {
        return hf_.colors;
    }
    bool take_mesh_dirty() override {
        return std::exchange(mesh_dirty_, false);
    }

    // tilted so the height relief reads
    double default_camera_azimuth() const override { return 0.35; }
    double default_camera_elevation() const override { return 0.95; }
    double default_camera_distance() const override { return 60.0; }

    int marker_count() const override { return kCr2dAtoms; }
    SceneMarker marker(int i) const override {
        const double pi = 3.14159265358979323846;
        const double th = 2.0 * pi * i / kCr2dAtoms;
        return {static_cast<float>(radius_ * std::cos(th)),
                static_cast<float>(radius_ * std::sin(th)),
                0.0f,
                0.45f,
                0.88f,
                0.42f,
                0.15f};
    }

    std::string title_text() override {
        std::string s = strf(
            "Quantum corral (48 adatoms, IBM 1993)  |  t = %.1f au  "
            "R = %.1f  states %d",
            sim_time_, radius_, captured());
        for (std::size_t k = 0; k < energies_.size(); ++k) {
            s += strf("  E%zu = %.3f eV", k, energies_[k] * kHaToEv);
        }
        if (relaxing_) {
            s += "  [relaxing...]";
        }
        s += "  keys: 2 ground / 3 next state / 5 Fermi wave / F packet";
        return s;
    }

protected:
    // Surface normalizer snaps to the first observed max then 0.98-decays;
    // the base peak_ decays too slow (2%/frame) to reach the ~1e-2 relax scale.
    void rebuild_display() override {
        double cur = 0.0;
        for (int j = 0; j < kCr2dN; ++j) {
            for (int i = 0; i < kCr2dN; ++i) {
                cur = std::max(cur, std::norm(psi_(i, j, 0)));
            }
        }
        disp_peak_ = disp_peak_ <= 0.0 ? cur
                                       : std::max(cur, 0.98 * disp_peak_);
        hf_ = ses::heightfield_surface(psi_, kCr2dSurfH, disp_peak_,
                                       kCr2dMeshStride);
        mesh_dirty_ = true;
    }

    void do_steps(int n) override {
        if (relaxing_) {
            // Disc projection between chunks makes prep the closed-corral
            // Dirichlet problem; without it the leaky fence drains relax into
            // the outside continuum and no plateau forms. Real time (F) is
            // the leaky H, unprojected.
            const int chunk = fine_ ? 100 : 20;  // tau ~ 0.5 / 1.0 per cut
            int left = 4 * n;  // relax gets 4x the tick budget (the wait)
            if (eng_ok_) {
                ensure_gpu_relax_tables();
                sync_gpu_deflation();
                if (eng_dirty_) {
                    eng_.upload_state(psi_.data());
                    eng_dirty_ = false;
                }
                while (left > 0) {
                    const int c = std::min(left, chunk);
                    if (defl_.empty()) {
                        eng_.relax_step(c);
                    } else {
                        eng_.relax_deflated_step(defl_, c);
                    }
                    eng_.readback(rb_);
                    store_readback();
                    project_disc();  // CPU Dirichlet cut + renorm
                    eng_.upload_state(psi_.data());  // re-sync the projection
                    left -= c;
                }
            } else {
                std::vector<const ses::Field3D*> lower;
                lower.reserve(captured_.size());
                for (const ses::Field3D& s : captured_) {
                    lower.push_back(&s);
                }
                while (left > 0) {
                    const int c = std::min(left, chunk);
                    (fine_ ? itp_fine_ : itp_coarse_)
                        ->relax_deflated(psi_, lower, c);
                    project_disc();
                    left -= c;
                }
            }
            const double e = ses::mean_energy(psi_, v_, kCr2dMass);
            const double tol = fine_ ? kCr2dConvTol : kCr2dCoarseTol;
            if (std::abs(e - last_e_) < tol * std::max(1.0, e)) {
                if (!fine_) {
                    fine_ = true;  // coarse settled: polish at small dtau
                    conv_streak_ = 0;
                } else if (++conv_streak_ >= 3) {
                    capture(e);
                }
            } else {
                conv_streak_ = 0;
            }
            last_e_ = e;
            track_peak();
            return;
        }
        // Damp (fence + open boundary); the surviving norm = escaped fraction.
        if (eng_ok_) {
            if (eng_dirty_) {
                eng_.upload_state(psi_.data());
                eng_dirty_ = false;
            }
            int left = n;
            while (left > 0) {
                const int c = std::min(left, 8);
                eng_.step(c, true);  // absorb=true applies damp_ each step
                left -= c;
            }
            eng_.readback(rb_);
            store_readback();
            ses::normalize(psi_);  // display/confinement copy (GPU state left as is)
        } else {
            for (int s2 = 0; s2 < n; ++s2) {
                prop_->step(psi_, 1);
                ses::parallel_for(static_cast<int>(damp_.size()), [&](int i) {
                    psi_.data()[static_cast<std::size_t>(i)] *=
                        damp_[static_cast<std::size_t>(i)];
                });
            }
            ses::normalize(psi_);
        }
        sim_time_ += n * kCr2dDt;
        track_peak();
    }

private:
    void store_readback() {
        std::vector<std::complex<double>>& d = psi_.data();
        for (std::size_t i = 0; i < d.size(); ++i) {
            d[i] = std::complex<double>{static_cast<double>(rb_[2 * i]),
                                        static_cast<double>(rb_[2 * i + 1])};
        }
    }
    // Load the coarse or fine ITP weights (m* baked in) into the engine.
    void ensure_gpu_relax_tables() {
        const int want = fine_ ? 1 : 0;
        if (relax_tables_stage_ == want) {
            return;
        }
        const ses::ImaginaryTimePropagator3D& itp =
            fine_ ? *itp_fine_ : *itp_coarse_;
        eng_.set_relax_tables(itp.half_potential_weight(),
                              itp.kinetic_weight(),
                              fine_ ? kCr2dFineDtau : kCr2dCoarseDtau,
                              phys_grid_.cell_volume());
        relax_tables_stage_ = want;
    }
    // GPU deflation buffers track captured_ (the Gram-Schmidt lower states).
    void sync_gpu_deflation() {
        while (defl_.size() < captured_.size()) {
            defl_.push_back(
                eng_.create_state_buffer(captured_[defl_.size()].data()));
        }
    }
    void release_gpu_deflation() {
        if (!eng_ok_) {
            return;
        }
        for (int h : defl_) {
            eng_.release_state(h);
        }
        defl_.clear();
    }

    void rebuild_potential() {
        const double pi = 3.14159265358979323846;
        std::vector<double> v(
            static_cast<std::size_t>(phys_grid_.size()), 0.0);
        for (int a = 0; a < kCr2dAtoms; ++a) {
            const double th = 2.0 * pi * a / kCr2dAtoms;
            const double ax = radius_ * std::cos(th);
            const double ay = radius_ * std::sin(th);
            for (int j = 0; j < kCr2dN; ++j) {
                const double dy = phys_grid_.y.coord(j) - ay;
                if (std::abs(dy) > 4.0 * kCr2dBumpSigma) {
                    continue;
                }
                for (int i = 0; i < kCr2dN; ++i) {
                    const double dx = phys_grid_.x.coord(i) - ax;
                    if (std::abs(dx) > 4.0 * kCr2dBumpSigma) {
                        continue;
                    }
                    v[static_cast<std::size_t>(
                        phys_grid_.flat(i, j, 0))] +=
                        kCr2dBumpA *
                        std::exp(-(dx * dx + dy * dy) /
                                 (2.0 * kCr2dBumpSigma * kCr2dBumpSigma));
                }
            }
        }
        // damp = open-boundary absorber x fence exp(-W dt). Mirrors
        // tests/corral2d_test.cpp.
        const std::vector<double> edge =
            ses::absorbing_mask(phys_grid_, kCr2dEdgeW);
        damp_.assign(v.size(), 1.0);
        for (std::size_t i = 0; i < v.size(); ++i) {
            damp_[i] = edge[i] * std::exp(-kCr2dFenceW0 *
                                          (v[i] / kCr2dBumpA) * kCr2dDt);
        }
        v_ = std::move(v);
        prop_ = std::make_unique<ses::SplitOperator3D>(phys_grid_, v_,
                                                      kCr2dDt, kCr2dMass);
        itp_coarse_ = std::make_unique<ses::ImaginaryTimePropagator3D>(
            phys_grid_, v_, kCr2dCoarseDtau, kCr2dMass);
        itp_fine_ = std::make_unique<ses::ImaginaryTimePropagator3D>(
            phys_grid_, v_, kCr2dFineDtau, kCr2dMass);
        captured_.clear();
        energies_.clear();
        if (eng_ok_) {
            eng_.set_potential(v_);       // real-time fence
            eng_.set_absorber(damp_);     // real-time damp
            release_gpu_deflation();      // captured basis dropped
            relax_tables_stage_ = -1;     // itp weights changed with v_
        }
    }

    void start_relax(int k) {
        // asymmetric seed to overlap the next mode; deflation cleans the rest.
        const double off = k == 0 ? 0.0 : 0.35 * radius_;
        ses::parallel_for(kCr2dN, [&](int j) {
            const double y = phys_grid_.y.coord(j);
            for (int i = 0; i < kCr2dN; ++i) {
                const double x = phys_grid_.x.coord(i);
                const double dx = x - off;
                psi_(i, j, 0) =
                    std::exp(-(dx * dx + y * y) / (4.0 * 4.0 * 4.0));
            }
        });
        ses::normalize(psi_);
        deflate();
        relaxing_ = true;
        fine_ = false;
        disp_peak_ = 0.0;  // re-snap surface normalizer
        conv_streak_ = 0;
        last_e_ = 1e30;
        eng_dirty_ = true;  // seed built on the CPU: re-upload before relaxing
        mark_fired();
    }

    // prep-only Dirichlet cut (zero psi outside R); never in real time.
    void project_disc() {
        ses::parallel_for(kCr2dN, [&](int j) {
            const double y = phys_grid_.y.coord(j);
            for (int i = 0; i < kCr2dN; ++i) {
                if (std::hypot(phys_grid_.x.coord(i), y) > radius_) {
                    psi_(i, j, 0) = 0.0;
                }
            }
        });
        ses::normalize(psi_);
    }

    // Gram-Schmidt against captured states.
    void deflate() {
        const double cell = phys_grid_.x.spacing() *
                            phys_grid_.y.spacing() *
                            phys_grid_.z.spacing();
        for (const ses::Field3D& s : captured_) {
            std::complex<double> ov{};
            for (int i = 0; i < phys_grid_.size(); ++i) {
                ov += std::conj(s.data()[static_cast<std::size_t>(i)]) *
                      psi_.data()[static_cast<std::size_t>(i)];
            }
            ov *= cell;
            for (int i = 0; i < phys_grid_.size(); ++i) {
                psi_.data()[static_cast<std::size_t>(i)] -=
                    ov * s.data()[static_cast<std::size_t>(i)];
            }
        }
        ses::normalize(psi_);
    }

    void capture(double e) {
        captured_.push_back(psi_);
        energies_.push_back(e);
        relaxing_ = false;
        title_dirty_ = true;
    }

    std::vector<double> v_;  // fence potential
    std::vector<double> damp_;
    std::unique_ptr<ses::SplitOperator3D> prop_;
    std::unique_ptr<ses::ImaginaryTimePropagator3D> itp_coarse_;
    std::unique_ptr<ses::ImaginaryTimePropagator3D> itp_fine_;
    ses::Heightfield hf_;
    bool mesh_dirty_ = false;
    std::vector<ses::Field3D> captured_;
    std::vector<double> energies_;
    double radius_ = kCr2dR;
    double last_e_ = 1e30;
    double disp_peak_ = 0.0;  // surface height normalizer
    int conv_streak_ = 0;
    bool relaxing_ = false;
    bool fine_ = false;  // anneal stage: coarse -> fine

    ses_vk::Engine eng_;           // GPU: mass-aware split-operator + deflated ITP
    bool eng_ok_ = false;
    bool eng_dirty_ = true;
    std::vector<float> rb_;        // readback scratch
    std::vector<int> defl_;        // GPU deflation handles (parallel to captured_)
    int relax_tables_stage_ = -1;  // -1 none, 0 coarse, 1 fine
};

}  // namespace ses_shell
