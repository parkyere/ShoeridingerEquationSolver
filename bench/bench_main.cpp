// Micro-benchmark for the per-frame hot path. NOT part of ctest (timing is
// machine-dependent); correctness is owned by the unit tests. Run manually:
//     sesolver_bench
// prints per-operation wall times at 32^3 and 64^3.

#include <core/field.hpp>
import ses.grid;
#include <core/marching_cubes.hpp>
#include <core/potential.hpp>
#include <core/simulation.hpp>
import ses.vec;

#include <chrono>
#include <cstdio>
#include <functional>

namespace {

using Clock = std::chrono::steady_clock;

double time_ms(int reps, const std::function<void()>& fn) {
    fn();  // warm-up
    const auto t0 = Clock::now();
    for (int i = 0; i < reps; ++i) {
        fn();
    }
    const auto t1 = Clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
}

void bench_size(int n) {
    const ses::Grid1D axis{-12.0, 12.0, n};
    const ses::Grid3D grid{axis, axis, axis};
    ses::WavepacketSimulation sim{ses::WavepacketSimulation::Config{
        grid,
        ses::soft_coulomb_potential(grid, 1.0, 1.0, ses::Vec3d{}),
        ses::Vec3d{3.0, 0.0, 0.0},
        ses::Vec3d{1.5, 1.5, 1.5},
        ses::Vec3d{0.0, 0.4, 0.0},
        0.02,
    }};

    ses::Field3D f = sim.psi();
    const double t_fft = time_ms(5, [&] {
        ses::fft(f);
        ses::ifft(f);
    }) / 2.0;

    const double t_advance = time_ms(5, [&] { sim.advance(1); });
    const double t_relax = time_ms(5, [&] { sim.relax(1, 0.05); });

    const std::vector<double> rho = sim.density();
    const double t_density = time_ms(5, [&] { (void)sim.density(); });
    const double t_mc = time_ms(5, [&] {
        (void)ses::marching_cubes_at_fraction(rho, grid, 0.25);
    });

    std::printf("%d^3 | fft3 %7.2f ms | advance(1) %7.2f ms | relax(1) %7.2f ms | "
                "density %6.2f ms | marching cubes %7.2f ms\n",
                n, t_fft, t_advance, t_relax, t_density, t_mc);
}

}  // namespace

int main() {
    std::printf("hot-path timings (avg of 5, after warm-up)\n");
    bench_size(32);
    bench_size(64);
    return 0;
}
