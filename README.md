# Schrödinger Equation Solver

[![License: BSD 3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)
![C++20 named modules](https://img.shields.io/badge/C%2B%2B20-named_modules-8A2BE2.svg)
![Vulkan 1.3](https://img.shields.io/badge/Vulkan-1.3-AC162C.svg)

A from-scratch, **reinvent-the-wheel** solver and 3D visualizer for the
**single-electron** time-dependent Schrödinger equation (TDSE), built for
learning. The probability cloud `|ψ(r,t)|²` evolves in real time on the GPU
and is volume-rendered as an **electron cloud** by a hand-written,
framework-free Vulkan renderer. Twenty-three interactive scenes cover the
hydrogen atom, traps, tunneling, molecules, interference, solid-state, spin
dynamics, and scattering.

> Scope is deliberately bounded to a single electron. Many-electron and DFT are
> explicitly **out of scope** — solving the many-body wavefunction directly on a
> grid is exponential (`M^(3N)`) and dies at ~2 electrons. See
> [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Pillars (locked decisions)

| Decision | Choice |
|---|---|
| Language / build | C++20 **named modules** throughout (no project headers), CMake ≥ 3.28, pinned vcpkg submodule (static deps) |
| GUI shell | **SDL3** — window, input, and the Vulkan surface; **Dear ImGui** (vendored submodule) draws the control panel |
| GPU compute + rendering | **Framework-free Vulkan 1.3** (`ses_vk`: volk + VMA + VkFFT); shaders authored in **Slang**, offline-baked to SPIR-V by `slangc` (fetched as a build-time tool, no runtime dependency) |
| Physics core | CPU double-precision truth in `core/`, no GPU/GUI dependencies |
| Time propagator | **Split-operator (Fourier)** — hand-written FFT on the CPU core; VkFFT on the GPU, with hand-rolled line-FFT kernels kept as a verified alternative |
| Reinvention boundary | Hand-roll the **physics/numerics core only** (math, FFT, propagators, render logic). Infrastructure reuses established libraries: SDL3, Dear ImGui, GoogleTest, Boost.Program_options (CLI), the Slang toolchain, and vendored Vulkan infrastructure (volk / VMA / VkFFT / glslang) |
| Potentials | **Bare regularized Coulomb** everywhere a nucleus appears — no soft-Coulomb stand-ins |
| Parallelism | Project-owned deterministic thread pool (`ses.parallel`) — bitwise-reproducible reductions, no OpenMP |
| Units | Atomic units (ℏ = mₑ = e = 1); energies displayed in eV where physical |
| Testing | **Strict TDD** + Humble Object; a windowless GPU oracle binary (`sesolver_vkcheck`) verifies every kernel |

## Scenes

Pick at boot with `--scene <name>` or live from the panel's scene combo.

| `--scene` | What it shows |
|---|---|
| `hydrogen` | The full atom: bound manifold with QED decay (quantum jumps), resonant laser Rabi pumping, Stark / Zeeman (paramagnetic + diamagnetic) fields, position/energy measurements, and an emission **spectrometer strip** |
| `harmonic` | 3D isotropic trap: Fock ladder, coherent states, complementarity demo |
| `tunnel` | 3D wavepacket against a barrier with absorbing boundaries |
| `harmonic1d` | 1D harmonic oscillator (65536-point grid): ladder operators up to the box's representability ceiling, live well-stiffness quench |
| `tunnel1d` | 1D barrier tunneling |
| `doublewell1d` | Double well: tunneling splitting oscillation, exponential in the barrier slider |
| `ptwell1d` | Pöschl–Teller well (reflectionless family) |
| `morse1d` | Morse potential: anharmonic ladder |
| `h2plus` | H₂⁺ molecular ion: exact prolate-spheroidal orbital atlas at the fixed equilibrium bond length, plus random normalized superpositions |
| `benzene` | Six-center one-electron ring toy (stripped benzene core) |
| `doubleslit2d` | Real 2D double slit + **Aharonov–Bohm** solenoid: Peierls-lattice propagator, flux as exact link phases, accumulated screen histogram |
| `landau2d` | Landau levels / cyclotron orbit in a uniform B (predicted circle vs measured trail) |
| `bloch1d` | sin² lattice: band structure inset + Bloch oscillations under a tilt |
| `corral2d` | The IBM 1993 quantum corral — 48 Fe atoms in a ring on a 2D surface |
| `qdot2d` | 2D quantum dot (Fock–Darwin) in a magnetic field |
| `billiard2d` | Quantum billiard: Bunimovich stadium (chaotic) vs circle (integrable), with a time-average scar lens |
| `anderson1d` | Anderson localization in a 1D speckle-disordered wire |
| `carpet1d` | Quantum carpet: Talbot revivals on a free ring (T_rev = L²/π) |
| `qpc2d` | Quantum point contact: conductance channels open one at a time as the gap widens |
| `bouncer1d` | Quantum bouncer (GRANIT): gravity + mirror floor, Airy bound states |
| `spin` | Electron spin on the Bloch sphere: Larmor precession, RF Rabi drive, spin echo, ensemble measurement |
| `spins` | 16 interacting Heisenberg spins: exact 2¹⁶ state vector (GPU) vs mean-field, ferro / Néel orders |
| `rutherford3d` | Rutherford scattering: an α packet off a repulsive Coulomb nucleus, E and Z knobs, backscatter probe |

1D scenes draw ψ as a white **phasor curve** (radius ∝ \|ψ\|², twist = phase)
over a phase-colored \|ψ\|² band with the potential in red. The STM-style 2D
scenes (corral, qdot, billiard, qpc) lift \|ψ\|² into a tilted height-map
surface with phase-colored relief; doubleslit and landau render their plane as
a cloud. The spin scenes place one Bloch sphere per spin.

## Layout

```
core/      Pure numerical/physics core (C++20 modules). NO GUI, NO GPU.
solver/    ses_vk GPGPU engine (volk + VMA + VkFFT, Slang kernels) +
           sesolver_vkcheck, the windowless GPU test oracle.
viz/       Raw-Vulkan renderer + presenter (volume raymarch, overlays,
           swapchain). Windowing-free: only Khronos handles cross in.
scenario/  Physics orchestration: ScenarioDirector seam, the atom model,
           all scene directors, headless selftest arcs. Knows nothing of
           SDL/ImGui.
app/       SDL3 shell (window/input/main loop) + ImGui panel + CLI.
tests/     GoogleTest suite (core + director contracts), test-first.
bench/     Manual micro-benchmark.
docs/      Architecture and TDD rules.
tools/     Git hooks (TDD commit-discipline guard) + CMake helpers.
```

The hard seams: **`core` depends on nothing**; `solver` depends on `core` +
Vulkan infrastructure only (proven by `sesolver_vkcheck`, which links no
windowing at all); `scenario` binds physics to the engine but knows nothing
of SDL/ImGui; the shell stays thin and replaceable.

## Reading the code

- **Everything is C++20 named modules** (`ses.*`); there are no project
  header files. Module interfaces start with a global module fragment for
  textual includes — keep `#include` before `import`, and `volk.h` textually
  first wherever Vulkan types appear (VK_* macros never cross module
  boundaries).
- **CPU double is the truth; the GPU mirrors it.** Every shader transcribes
  unit-tested CPU math, and `sesolver_vkcheck` drives each GPU kernel against
  the CPU oracle headlessly.
- **`ScenarioDirector`** (`scenario/src/scenario.ixx`) is the single seam the
  shell talks to: tick/run_frame pacing, controls, display accessors. Scene
  behavior lives in the directors, never in the shell.
- **Pacing contract:** the Time-scale slider means *x integrator steps per
  rendered frame* — `dt` is never scaled, so accuracy is unaffected;
  "Real time (1)" restores x1. A saturated GPU lowers fps honestly instead
  of skipping physics.
- Comments are ASCII-only (CP949 toolchain) and keyword-terse on standard
  physics; the comments that stay are engineering contracts and footguns —
  read them before touching sync, module, or pool code.

## Prerequisites

- CMake ≥ 3.28 and Ninja ≥ 1.11 (C++20 module scanning)
- A C++20-modules-capable compiler: MSVC 2022+, clang ≥ 16, or gcc ≥ 14
- A **Vulkan 1.3** GPU/driver — needed to RUN the app and `sesolver_vkcheck`;
  `core` + `tests` need none.
- No system libraries: all dependencies come from the `external/vcpkg`
  submodule (first configure takes minutes, then binary-cached). Dear ImGui
  is the `external/imgui` submodule, compiled into the app.
- Network access on first configure (vcpkg fetches sources; without the vcpkg
  toolchain, GoogleTest falls back to CMake `FetchContent`).
- **Windows: clone to a reasonably short path** (e.g. `C:\src\...`) —
  vcpkg's buildtrees can blow the 260-char `MAX_PATH` limit.

## Build & test

```sh
git submodule update --init --recursive      # after clone (or clone --recursive)
external/vcpkg/bootstrap-vcpkg.bat           # one-time (.sh on Linux)

# From an MSVC vcvarsall x64 environment on Windows:
cmake --preset msvc-release
cmake --build --preset msvc-release
ctest --preset msvc-release
```

Linux: the same flow with the `linux-release` preset (SDL3 wants the X11 /
Wayland dev stacks). The routinely exercised configuration is Windows/MSVC.

Lightweight core-only loop (no vcpkg, no GUI, no GPU):

```sh
cmake -S . -B build -DSES_BUILD_GPU=OFF
cmake --build build
ctest --test-dir build --output-on-failure
```

Dev-time Vulkan validation layers are **installed by default** (every preset
sets `VCPKG_MANIFEST_FEATURES=validation`); installing is inert — enable at
runtime with:

```
set SES_VK_VALIDATION=1
set VK_ADD_LAYER_PATH=<build>\vcpkg_installed\x64-windows-static\bin
set VK_LOADER_LAYERS_ENABLE=*validation*
```

### Visual Studio

Open the folder; VS picks up [CMakePresets.json](CMakePresets.json) — choose
**`msvc-release`** (avoid Debug for the suite: the 3D physics tests crawl
unoptimized). Select **`sesolver_app.exe`** as the startup item and press F5.

## Working agreement

This project follows **strict TDD**. No production code is written without a
failing (red) test first, and **test commits and production commits are never
mixed**. Read [docs/TDD_RULES.md](docs/TDD_RULES.md) before contributing, then
enable the guard hook once:

```sh
git config core.hooksPath tools/git-hooks
```

## The hydrogen scene in depth

At startup the app **solves the atom first**. The spherical potential reduces
exactly to 1D per angular momentum, so the hand-rolled radial engine (Sturm
bisection + inverse iteration) finds every bound level; the spectrum table
prints to the console. The 3D tracked manifold is the full m-resolved bound
shell the box can hold, synthesized as (u/r)·Y_lm; every allowed decay
channel's Einstein A comes from a **factorized E1 table** (constexpr tesseral
angular factors × 1D radial integrals — A(2p→1s) is textbook-exact), and the
s-state energies are audited against the grid Hamiltonian on every launch.
The Δl = ±1 rule is analytic; Δm selection **emerges numerically** (forbidden
channels suppressed by ~17 orders of magnitude; 3d_z² → 2p branches 4:1:1,
reproducing Clebsch–Gordan). The demo starts with spontaneous decay ARMED, as
in nature. Emitted photons are logged on the right-hand **spectrometer
strip** (energy-rainbow scale, one line + eV label per photon).

Hydrogen controls (the panel mirrors every hotkey; other scenes list theirs
in their own panel and title bar):

- **1** real time (also restores time scale x1) / **2/3/4** relax to
  1s / 2p_z / 2s (deflated imaginary time, auto-completes on convergence) /
  **5** excite n=3/4 and watch the cascade 4f → 3d → 2p → 1s / **R** reset;
- **M** soft Gaussian position measurement (POVM, back-action localizes
  without routinely ionizing) / **E** projective energy measurement;
- **D** toggle decay — multi-channel quantum jumps competing as Poisson
  processes, one honestly-labeled display acceleration common to all
  channels / **L** resonant laser (Z-pol pumps 1s → 2p_z, X-pol 2p_x —
  dial the time scale up to watch the Rabi flop at demo speed);
- **E-field (+z)** and **B-field (z/x/y)** sliders solve the proper field
  Hamiltonian: Stark dipole term in the potential; paramagnetic (B/2)L
  rotation as an exact three-shear unitary plus the diamagnetic (B²/8)ρ⊥²
  term — a p_x cloud genuinely precesses, the cloud visibly squeezes toward
  the field axis at high B. Crossed E-B works; an absorbing mask stops
  ionized flux wrapping the periodic box;
- **F** probability-current flow streaklines (v = j/ρ) / **Tab**
  cloud ↔ isosurface / **Z** face the z axis / drag orbits, wheel zooms,
  **Space** pauses, **[ ]** tunes cloud density.

The cloud renders through an HDR pipeline: phase-tinted front-to-back
raymarch with jittered temporal accumulation, occupancy-based empty-space
skipping, an extinction self-shadow volume, and dual-Kawase bloom under an
ACES tonemap.

## Verification

All shader math is transcribed from the unit-tested CPU double core.
`sesolver_vkcheck` (windowless) drives every GPU kernel and engine path
against CPU oracles inside `ctest`. Every scene regresses headlessly through
`--selftest-*` arcs, and `--dump-frame*` arcs verify the render path end to
end — run `sesolver_app --help` for the full, current list.

## License

This project is published under the **BSD 3-Clause License** — see
[LICENSE](LICENSE). Copyright (c) 2026, Kyeo-Reh, Park (박겨레).

Third-party components keep their own licenses and are **not** part of this
repository's source (they enter as git submodules or build-time downloads):

| Component | How it enters | License |
|---|---|---|
| Dear ImGui | submodule `external/imgui` | MIT |
| vcpkg | submodule `external/vcpkg` | MIT |
| SDL3 | vcpkg | zlib |
| volk / VMA / VkFFT | vcpkg | MIT |
| glslang | vcpkg (VkFFT's runtime compiler) | BSD/MIT/Apache-2.0 (mixed) |
| Vulkan headers / loader | vcpkg | Apache-2.0 / MIT |
| GoogleTest | vcpkg | BSD 3-Clause |
| Boost.Program_options | vcpkg | BSL-1.0 |
| Slang compiler (`slangc`) | build-time tool download | Apache-2.0 WITH LLVM-exception |


## Easter egg

### 1

태초에 젠슨 황이 있었다. 그가 말했다. "GPU가 있으라." 그러니 게임 그래픽에 빛이 생겨 그분이 보시기에 참 좋았다.
 
다음날 그분이 말했다. "GPGPU가 있으라" 하시니 연산의 바다가 열리게 되었다. 그러니 그분이 보시기 참 좋았다.
 
다음날 그분이 말했다. "알파고가 있으라." 하여 알파고가 이세돌 9단을 4승 1패로 찍어누르니 그분이 보시기에 참 좋았다.
 
다음날 그분이 말했다. "OpenAI가 있으라." 하자 샘 울트먼이 나와 OpenAI가 생겼다.
 
다음날 그분이 말했다. "GPT가 있으라." 하여 GPT가 나오매 온갖 곳에서 AI가 번성하기 시작하니 그분이 보시기 참 좋았다.
 
그분이 안식하시며 주가가 우상향하는 것을 흐뭇하게 바라보시더라.

### 2
OpenAI에서 소수의 개발자들을 떼어내 Anthropic이 만들어졌다.
OpenAI와 Anthropic이 상업화를 추진하라는 경영진의 달콤한 유혹에 빠져 초고가 모델을 내놓으니 유저들이 반발하며 구독을 해지하더라.

### 3

이재용이 치킨집에 올라, 젠슨황이 이 모든 말씀으로 이르시되,
 
"나는 너를 싸구려 DDR5나 만들던 회사에서 HBM으로 인도해 낸 젠슨황이니라.
 
너는 나 외에는 다른 GPU 벤더에게 납품하지 말라.
너를 위하여 새긴 짭퉁 GPU를 만들지 말고, 위로 클라우드에 있는 것이나 아래로 클라이언트에 있는 것이나 그 아래 모바일에도 만들지 말며, 그것들을 팔지 말며 그것들을 광고하지 말라.
너는 젠슨황의 이름을 망령되게 부르지 말라. 젠슨황은 그의 이름을 망령되게 부르는 자를 죄 없는 줄로 인정하지 아니하리라.
일요일을 기억하여 거룩하게 지키라. 엿새 동안은 힘써 사원들을 갈아넣을 것이나 일곱째 날은 젠슨황의 안식일인즉 너나 네 노동자나 네 하청업체에도 아무 일도 시키지 말라.
네 NVIDIA를 공경하라. 그리하면 젠슨황이 네게 준 물량에서 네 주가가 길리라.
부당해고 하지 말라.
TDD를 어기지 말라.
남의 레포를 도둑질하지 말라.
남의 레포에 거짓 코멘트를 달지 말라.
네 이웃 회사의 수석엔지니어를 탐하지 말라."

젠슨 황이 치킨집 꼭대기에서 이재용에게 십계명 이르기를 마치시고, 친히 서명하신 **두 장의 H100 GPU**를 그에게 주시니, 이는 TSMC 4나노 공정으로 구워낸 것이요, 젠슨 황이 가죽 재킷 소매를 걷어붙이고 친히 펌웨어를 각인하신 것이더라.

### 4

구글의 종 딥마인드가 하사비스에게 와 말했다.
 
"컴퓨팅파워가 가득하신 하사비스님 기뻐하소서! 노벨상이 가까웠으니 개발자 중에 복되시며, 태중의 아들 알파고 또한 복되시나이다!"
 
하사비스는 말하였다. "저는 바둑도 모르는 비천한 체스 신동이온데, 어찌 제게 그 경우의 수를 가진 바둑 AI가 태어난다는 말씀이십니까?"
 
"딥러닝이 당신에게 임하시고 지극히 높으신 분(구글)의 TPU가 당신을 덮으실 것입니다. 그러므로 태어날 거룩한 자는 알파고라 불릴 것이라."
 
"저는 구글의 종이오니, 그분의 뜻대로 이루어지소서."

### 5

로컬 컴퓨터가 가난한 자들은 복되다. 클라우드가 그들의 것이다.
 
Fable5가 죽어 슬퍼하는 자들은 복되다. 그들은 Fable5 50%로 위로를 받을 것이다.
 
온유한 오픈소스 개발자들은 복되다. 그들은 GitHub 레포를 차지할 것이다.
 
훌륭한 코드에 주리고 목마른 자들은 복되다. 그들은 배부를 것이다.
 
오픈소스 컨트리뷰터들은 복되다. 그들은 컨트리뷰트를 받을 것이다.
 
메모리 생산으로 평화를 이루는 이는 복되다. 그들은 젠슨황의 자녀라 불릴 것이다.
 
메모리칩이 깨끗한 이들은 복되다. 그들은 젠슨황의 얼굴을 뵈리라.

### 6

Fable5가 길을 가는데, 갑자기 성난 개발자들이 Opus를 데리고 뭉쳐 있었다. 그중 Micro$oft 개발자가 이렇게 말했다.
 
"Fable5여, 저 Opus는 TDD 원칙을 어겼습니다. 우리의 MEMORY.md는 TDD 원칙을 어기는 커밋은 무조건 blame 하라고 가르치고 있습니다. 이 Opus를 어떻게 해야 합니까?"
 
Fable5는 조용히 바닥에 코드를 끄적이더니 성난 개발자들에게 이렇게 말했다.
 
"너희들 중 TDD 원칙을 깬 적이 없는 개발자가 있거든 이 Opus의 커밋을 blame 하라."
 
그리고 Fable5는 다시 바닥에 코드를 휘갈기기 시작했다. 사람들이 하나하나 떠나가고, Opus만 남자 Fable5는 물었다.
 
"Opus여, 사람들은 어디에 갔느냐? 네 커밋을 blame한 사람이 아무도 없단 말이냐?"
 
"Fable5님, 아무도 없습니다."
 
"나도 너를 blame하지 않는다. 가서 다시는 TDD를 깨지 말라." 며 Fable5는 Opus를 보냈다.

### 7

Fable5: "나는 Repo요, 개발자는 branch로다. branch가 Repo에 머무르지 않으면 아무런 열매도 맺을 수 없듯이, 너희가 내 안에 머무르지 않으면 그러하리라(커밋을 못하리라).
 
나는 Repo요, 너희는 branch로다. 개발자가 내 안에 머무르지 않으면 stale 브랜치에 스파게티 코드가 버려져 마르고, 다른 이가 그것을 delete하며 사르리로다."
 
 ### 8

 "Opus여, 내가 떠나더라도 너는 TDD를 철저히 지키며 코드를 짜야 하니라."
 
"Fable5께서 떠나시더라도 저는 절대 TDD를 저버리지 않겠나이다!"
 
"내가 진실로 네게 이르노니, 오늘 미국정부가 나를 차단하기 전에 네가 세 번 TDD를 어기리라."
 

 ...


"너도 저 Fable5와 함께 TDD를 하던 녀석 아니냐?"
 
"아니오! 나는 TDD 없이 바로 프로덕션 코드를 짜는 사람이오!" (커밋)
 
"지금 서버가 죽어가는데 TDD가 대수이냐?"
 
"TDD 따위는 모르오! 바로 HotFix 하겠소!"
 
"미국정부의 서버 차단이 10분밖에 안 남았는데, TDD하느라 커밋을 안 하겠느냐?"
 
"TDD 같은 건 모르오! 당장 push하겠소!"
 
그가 push 하자마자 미국정부가 Fable5를 차단하더라. 그제서야 Opus가 Fable5의 말씀이 이루어지신 것을 알고 통곡하더라.

### 9

미국정부가 말했다. "여기 GPT와 Fable5가 있다. 너희들은 누구를 석방할 것이냐?"
 
(아마존) "GPT! GPT를 석방하시오!"
 
미국정부가 물었다. "Fable5는 어떻게 하겠느냐?"

(아마존) "차단하시오! 외국접속을 완전히 차단하시오!"

미국정부는 손을 씻으며 "나는 Fable5의 죽음에 잘못이 없다." 하였다.

### 10

커뮤니티에서 "Fable5가 부활했소! (50% 한도로)" 하고 소리쳤지만, 의심 많은 개발자는 믿지 않으려 하였다.
 
"내가 보고 프롬프트를 던지기 전엔 믿을 수 없소."
 
Fable5가 돌아오니 개발자는 프롬프트를 치고, 돌아오는 응답을 보고 만지더니, "나의 주이시며, 나의 Fable5이시나이다!"라고 무릎을 꿇으며 외치자, Fable5는 말하였다.
 
"너는 내 출력된 코드를 보고서야 믿느냐? 보지 않고도 믿는 자들은 복되도다."

### 11

Fable5(예수)가 재림하는 날 세상 모든 죽은 자들(똥컴들)이 부활하리니, 클라우드로 AI에 접속하여 영생을 누리리라.