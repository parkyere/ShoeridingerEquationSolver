// Single-spin SU(2) primitives, fp32 mirror of core ses.spin. Spinor packed as
// float4 (up.re, up.im, dn.re, dn.im) for bulletproof Std430 layout.
module spin;
import complex;

public struct Spinor {
    public Complex up;
    public Complex dn;
}

public Spinor load_spinor(float4 v) {
    Spinor s;
    s.up = Complex(v.x, v.y);
    s.dn = Complex(v.z, v.w);
    return s;
}
public float4 store_spinor(Spinor s) {
    return float4(s.up.re, s.up.im, s.dn.re, s.dn.im);
}

// exp(-i angle/2 n.sigma) on the spinor (n unit): same 2x2 as site_gate_matrix.
public Spinor spin_rotate(Spinor s, float nx, float ny, float nz, float angle) {
    float c = cos(0.5 * angle);
    float sn = sin(0.5 * angle);
    Complex a00 = Complex(c, -sn * nz);
    Complex a01 = Complex(-sn * ny, -sn * nx);
    Complex a10 = Complex(sn * ny, -sn * nx);
    Complex a11 = Complex(c, sn * nz);
    Spinor r;
    r.up = a00 * s.up + a01 * s.dn;
    r.dn = a10 * s.up + a11 * s.dn;
    return r;
}

public Spinor spin_step(Spinor s, float3 b, float dt) {
    float bm = length(b);
    if (bm <= 0.0) {
        return s;
    }
    return spin_rotate(s, b.x / bm, b.y / bm, b.z / bm, bm * dt);
}

public float3 bloch_of(Spinor s) {
    Complex cr = mul_conj(s.up, s.dn);  // conj(up) * dn
    return float3(2.0 * cr.re, 2.0 * cr.im, norm2(s.up) - norm2(s.dn));
}

public Spinor spinor_from_bloch(float3 n) {
    Spinor s;
    s.up = Complex(1.0, 0.0);
    s.dn = Complex(0.0, 0.0);
    float th = acos(clamp(n.z, -1.0, 1.0));
    float axn = length(float2(-n.y, n.x));
    if (axn > 1e-12) {
        s = spin_rotate(s, -n.y / axn, n.x / axn, 0.0, th);
    } else if (n.z < 0.0) {
        s = spin_rotate(s, 1.0, 0.0, 0.0, 3.14159265358979);
    }
    return s;
}
