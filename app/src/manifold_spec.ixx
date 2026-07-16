export module ses.app.manifold_spec;


// Tracked n<=6 eigenstate-manifold spec: state/level index tables (pure data).
// The full m-resolved n <= 6 shell (kNumStates entries); n = 6 is
// box-critical on the +-80 Bohr grid. First five indices frozen (selftests).


// Global-namespace data by design (pure tables); exported as a block.
export {

enum StateIndex : int {
    kS1 = 0, kP2X = 1, kP2Y = 2, kP2Z = 3, kS2 = 4,
    k3S = 5, k3PX = 6, k3PY = 7, k3PZ = 8,
    k3DXY = 9, k3DYZ = 10, k3DZ0 = 11, k3DZX = 12, k3DX2Y2 = 13,
    k4S = 14, k4F0 = 26,  // named entries the shell refers to
    k5S = 30,             // first n = 5 state (box-critical, h-audited)
    k6S = 55,             // first n = 6 state (box-critical, h-audited)
};
inline constexpr int kNumStates = 91;

// Radial levels backing the manifold (l, nodes k); n = l + 1 + k.
struct RadialLevelSpec {
    int l;
    int k;
};
inline constexpr int kNumLevels = 21;  // n<=5 (15) + 6s 6p 6d 6f 6g 6h
inline constexpr RadialLevelSpec kLevelSpec[kNumLevels] = {
    {0, 0}, {0, 1}, {1, 0}, {0, 2}, {1, 1}, {2, 0},
    {0, 3}, {1, 2}, {2, 1}, {3, 0},
    {0, 4}, {1, 3}, {2, 2}, {3, 1}, {4, 0},
    {0, 5}, {1, 4}, {2, 3}, {3, 2}, {4, 1}, {5, 0},
};

struct StateSpec {
    int level;  // index into kLevelSpec
    int l;
    int m;
    const char* name;
};
inline constexpr StateSpec kStateSpec[kNumStates] = {
    {0, 0, 0, "1s"},
    {2, 1, 1, "2p_x"}, {2, 1, -1, "2p_y"}, {2, 1, 0, "2p_z"},
    {1, 0, 0, "2s"},
    {3, 0, 0, "3s"},
    {4, 1, 1, "3p_x"}, {4, 1, -1, "3p_y"}, {4, 1, 0, "3p_z"},
    {5, 2, -2, "3d_xy"}, {5, 2, -1, "3d_yz"}, {5, 2, 0, "3d_z2"},
    {5, 2, 1, "3d_zx"}, {5, 2, 2, "3d_x2y2"},
    {6, 0, 0, "4s"},
    {7, 1, 1, "4p_x"}, {7, 1, -1, "4p_y"}, {7, 1, 0, "4p_z"},
    {8, 2, -2, "4d_xy"}, {8, 2, -1, "4d_yz"}, {8, 2, 0, "4d_z2"},
    {8, 2, 1, "4d_zx"}, {8, 2, 2, "4d_x2y2"},
    {9, 3, -3, "4f_-3"}, {9, 3, -2, "4f_-2"}, {9, 3, -1, "4f_-1"},
    {9, 3, 0, "4f_0"}, {9, 3, 1, "4f_+1"}, {9, 3, 2, "4f_+2"},
    {9, 3, 3, "4f_+3"},
    {10, 0, 0, "5s"},
    {11, 1, 1, "5p_x"}, {11, 1, -1, "5p_y"}, {11, 1, 0, "5p_z"},
    {12, 2, -2, "5d_xy"}, {12, 2, -1, "5d_yz"}, {12, 2, 0, "5d_z2"},
    {12, 2, 1, "5d_zx"}, {12, 2, 2, "5d_x2y2"},
    {13, 3, -3, "5f_-3"}, {13, 3, -2, "5f_-2"}, {13, 3, -1, "5f_-1"},
    {13, 3, 0, "5f_0"}, {13, 3, 1, "5f_+1"}, {13, 3, 2, "5f_+2"},
    {13, 3, 3, "5f_+3"},
    {14, 4, -4, "5g_-4"}, {14, 4, -3, "5g_-3"}, {14, 4, -2, "5g_-2"},
    {14, 4, -1, "5g_-1"}, {14, 4, 0, "5g_0"}, {14, 4, 1, "5g_+1"},
    {14, 4, 2, "5g_+2"}, {14, 4, 3, "5g_+3"}, {14, 4, 4, "5g_+4"},
    {15, 0, 0, "6s"},
    {16, 1, 1, "6p_x"}, {16, 1, -1, "6p_y"}, {16, 1, 0, "6p_z"},
    {17, 2, -2, "6d_xy"}, {17, 2, -1, "6d_yz"}, {17, 2, 0, "6d_z2"},
    {17, 2, 1, "6d_zx"}, {17, 2, 2, "6d_x2y2"},
    {18, 3, -3, "6f_-3"}, {18, 3, -2, "6f_-2"}, {18, 3, -1, "6f_-1"},
    {18, 3, 0, "6f_0"}, {18, 3, 1, "6f_+1"}, {18, 3, 2, "6f_+2"},
    {18, 3, 3, "6f_+3"},
    {19, 4, -4, "6g_-4"}, {19, 4, -3, "6g_-3"}, {19, 4, -2, "6g_-2"},
    {19, 4, -1, "6g_-1"}, {19, 4, 0, "6g_0"}, {19, 4, 1, "6g_+1"},
    {19, 4, 2, "6g_+2"}, {19, 4, 3, "6g_+3"}, {19, 4, 4, "6g_+4"},
    {20, 5, -5, "6h_-5"}, {20, 5, -4, "6h_-4"}, {20, 5, -3, "6h_-3"},
    {20, 5, -2, "6h_-2"}, {20, 5, -1, "6h_-1"}, {20, 5, 0, "6h_0"},
    {20, 5, 1, "6h_+1"}, {20, 5, 2, "6h_+2"}, {20, 5, 3, "6h_+3"},
    {20, 5, 4, "6h_+4"}, {20, 5, 5, "6h_+5"},
};

// Principal quantum number of a tracked state: n = l + k + 1.
inline constexpr int state_n(int idx) {
    return kLevelSpec[kStateSpec[idx].level].l +
           kLevelSpec[kStateSpec[idx].level].k + 1;
}

struct ShellChannel {
    int from;
    int to;
    double a_true;         // Einstein A from our wavefunctions (au)
    double gamma_display;  // uniformly accelerated display rate
};


}  // export
