// ylm.slang -- real spherical harmonics Y_lm (l = 0..5), the single source of
// truth shared by synth / mcwf_axpy / project_deposit (was copy-pasted three
// times). Pure scalar float math; inlines to the exact ops the kernels wrote,
// so the oracles are unchanged. Import with `import ylm;`.
module ylm;

public static const float PI = 3.14159265358979323846;

// Real Y_lm(theta,phi) evaluated at Cartesian (x,y,z); returns 0 at the origin
// for l>0 (the r=0 singularity the callers pin separately).
public float real_Ylm(int l, int m, float x, float y, float z) {
    float r2 = x * x + y * y + z * z;
    if (l == 0) return 1.0 / (2.0 * sqrt(PI));
    if (r2 == 0.0) return 0.0;
    if (l == 1) {
        float c = sqrt(3.0 / (4.0 * PI)) / sqrt(r2);
        if (m == -1) return c * y;
        if (m == 0) return c * z;
        return c * x;
    }
    if (l == 2) {
        float c = sqrt(15.0 / PI) / r2;
        if (m == -2) return 0.5 * c * x * y;
        if (m == -1) return 0.5 * c * y * z;
        if (m == 0) return 0.25 * sqrt(5.0 / PI) * (3.0 * z * z - r2) / r2;
        if (m == 1) return 0.5 * c * z * x;
        return 0.25 * c * (x * x - y * y);
    }
    if (l == 3) {
        float r3 = r2 * sqrt(r2);
        if (m == -3) return 0.25 * sqrt(35.0 / (2.0 * PI)) * y * (3.0 * x * x - y * y) / r3;
        if (m == -2) return 0.5 * sqrt(105.0 / PI) * x * y * z / r3;
        if (m == -1) return 0.25 * sqrt(21.0 / (2.0 * PI)) * y * (5.0 * z * z - r2) / r3;
        if (m == 0) return 0.25 * sqrt(7.0 / PI) * z * (5.0 * z * z - 3.0 * r2) / r3;
        if (m == 1) return 0.25 * sqrt(21.0 / (2.0 * PI)) * x * (5.0 * z * z - r2) / r3;
        if (m == 2) return 0.25 * sqrt(105.0 / PI) * z * (x * x - y * y) / r3;
        return 0.25 * sqrt(35.0 / (2.0 * PI)) * x * (x * x - 3.0 * y * y) / r3;
    }
    if (l == 4) {
        float r4 = r2 * r2;
        if (m == -4) return 0.75 * sqrt(35.0 / PI) * x * y * (x * x - y * y) / r4;
        if (m == -3) return 0.75 * sqrt(35.0 / (2.0 * PI)) * y * z * (3.0 * x * x - y * y) / r4;
        if (m == -2) return 0.75 * sqrt(5.0 / PI) * x * y * (7.0 * z * z - r2) / r4;
        if (m == -1) return 0.75 * sqrt(5.0 / (2.0 * PI)) * y * z * (7.0 * z * z - 3.0 * r2) / r4;
        if (m == 0) return (3.0 / 16.0) * sqrt(1.0 / PI) *
                           (35.0 * z * z * z * z - 30.0 * z * z * r2 + 3.0 * r2 * r2) / r4;
        if (m == 1) return 0.75 * sqrt(5.0 / (2.0 * PI)) * x * z * (7.0 * z * z - 3.0 * r2) / r4;
        if (m == 2) return 0.375 * sqrt(5.0 / PI) * (x * x - y * y) * (7.0 * z * z - r2) / r4;
        if (m == 3) return 0.75 * sqrt(35.0 / (2.0 * PI)) * x * z * (x * x - 3.0 * y * y) / r4;
        return (3.0 / 16.0) * sqrt(35.0 / PI) *
               (x * x * x * x - 6.0 * x * x * y * y + y * y * y * y) / r4;
    }
    // l == 5
    float r5 = r2 * r2 * sqrt(r2);
    float x2 = x * x;
    float y2 = y * y;
    float z2 = z * z;
    float zpoly = 21.0 * z2 * z2 - 14.0 * z2 * r2 + r2 * r2;
    if (m == -5) return (1.0 / 32.0) * sqrt(1386.0 / PI) *
                        y * (5.0 * x2 * x2 - 10.0 * x2 * y2 + y2 * y2) / r5;
    if (m == -4) return (1.0 / 16.0) * sqrt(3465.0 / PI) * 4.0 * x * y * (x2 - y2) * z / r5;
    if (m == -3) return (1.0 / 32.0) * sqrt(770.0 / PI) * y * (3.0 * x2 - y2) * (9.0 * z2 - r2) / r5;
    if (m == -2) return (1.0 / 8.0) * sqrt(1155.0 / PI) * 2.0 * x * y * z * (3.0 * z2 - r2) / r5;
    if (m == -1) return (1.0 / 16.0) * sqrt(165.0 / PI) * y * zpoly / r5;
    if (m == 0) return (1.0 / 16.0) * sqrt(11.0 / PI) *
                       z * (63.0 * z2 * z2 - 70.0 * z2 * r2 + 15.0 * r2 * r2) / r5;
    if (m == 1) return (1.0 / 16.0) * sqrt(165.0 / PI) * x * zpoly / r5;
    if (m == 2) return (1.0 / 8.0) * sqrt(1155.0 / PI) * (x2 - y2) * z * (3.0 * z2 - r2) / r5;
    if (m == 3) return (1.0 / 32.0) * sqrt(770.0 / PI) * x * (x2 - 3.0 * y2) * (9.0 * z2 - r2) / r5;
    if (m == 4) return (1.0 / 16.0) * sqrt(3465.0 / PI) *
                       (x2 * x2 - 6.0 * x2 * y2 + y2 * y2) * z / r5;
    return (1.0 / 32.0) * sqrt(1386.0 / PI) * x * (x2 * x2 - 10.0 * x2 * y2 + 5.0 * y2 * y2) / r5;
}
