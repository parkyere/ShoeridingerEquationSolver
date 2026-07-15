// complex.slang -- a zero-overhead complex-scalar type shared by the ses_vk
// kernels. A Complex IS a bare float2 (re, im); every operator inlines to the
// exact float ops the hand-written kernels used, so the baked SPIR-V -- and
// every oracle number -- is unchanged. Import with `import complex;`.
module complex;

public struct Complex {
    public float2 v;  // (re, im)

    public __init(float2 f) { v = f; }
    public __init(float re, float im) { v = float2(re, im); }

    public property float re {
        get { return v.x; }
        set { v.x = newValue; }
    }
    public property float im {
        get { return v.y; }
        set { v.y = newValue; }
    }
}

// z1 +/- z2, componentwise.
public Complex operator+(Complex a, Complex b) { return Complex(a.v + b.v); }
public Complex operator-(Complex a, Complex b) { return Complex(a.v - b.v); }

// z1 * z2 (the exact (ac-bd, ad+bc) the kernels wrote by hand).
public Complex operator*(Complex a, Complex b) {
    return Complex(a.v.x * b.v.x - a.v.y * b.v.y,
                   a.v.x * b.v.y + a.v.y * b.v.x);
}

// Real scale (both orders) and real divide.
public Complex operator*(Complex a, float s) { return Complex(a.v * s); }
public Complex operator*(float s, Complex a) { return Complex(a.v * s); }
public Complex operator/(Complex a, float s) { return Complex(a.v / s); }

public Complex conj(Complex a) { return Complex(a.v.x, -a.v.y); }
public float norm2(Complex a) { return dot(a.v, a.v); }         // |z|^2
public float arg(Complex a) { return atan2(a.v.y, a.v.x); }     // phase in (-pi, pi]

// Componentwise lerp (delegates to the builtin float2 lerp -- bit-identical to
// a hand-written lerp on the re/im pair).
public Complex mix(Complex a, Complex b, float t) { return Complex(lerp(a.v, b.v, t)); }

// Hermitian product conj(a)*b, written as the reduction kernels did --
// (a.x b.x + a.y b.y, a.x b.y - a.y b.x) -- so no stray negation can perturb
// FMA contraction (bit-identical to the hand-written form).
public Complex mul_conj(Complex a, Complex b) {
    return Complex(a.v.x * b.v.x + a.v.y * b.v.y,
                   a.v.x * b.v.y - a.v.y * b.v.x);
}
public Complex cis(float theta) { return Complex(cos(theta), sin(theta)); }  // e^{i theta}
