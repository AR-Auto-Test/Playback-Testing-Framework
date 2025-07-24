package b.o.a;

import b.o.a.b;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import com.google.android.material.shadow.ShadowDrawableWrapper;

/* compiled from: SpringForce.java */
/* loaded from: classes.dex */
public final class e {

    /* renamed from: a  reason: collision with root package name */
    public double f2362a;

    /* renamed from: b  reason: collision with root package name */
    public double f2363b;

    /* renamed from: c  reason: collision with root package name */
    public boolean f2364c;

    /* renamed from: d  reason: collision with root package name */
    public double f2365d;

    /* renamed from: e  reason: collision with root package name */
    public double f2366e;

    /* renamed from: f  reason: collision with root package name */
    public double f2367f;

    /* renamed from: g  reason: collision with root package name */
    public double f2368g;

    /* renamed from: h  reason: collision with root package name */
    public double f2369h;
    public double i;
    public final b.o j;

    public e() {
        this.f2362a = Math.sqrt(1500.0d);
        this.f2363b = 0.5d;
        this.f2364c = false;
        this.i = Double.MAX_VALUE;
        this.j = new b.o();
    }

    public e a(float f2) {
        if (f2 > StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
            this.f2362a = Math.sqrt(f2);
            this.f2364c = false;
            return this;
        }
        throw new IllegalArgumentException("Spring stiffness constant must be positive.");
    }

    public b.o b(double d2, double d3, long j) {
        double cos;
        double d4;
        if (!this.f2364c) {
            if (this.i != Double.MAX_VALUE) {
                double d5 = this.f2363b;
                if (d5 > 1.0d) {
                    double d6 = this.f2362a;
                    this.f2367f = (Math.sqrt((d5 * d5) - 1.0d) * d6) + ((-d5) * d6);
                    double d7 = this.f2363b;
                    double d8 = this.f2362a;
                    this.f2368g = ((-d7) * d8) - (Math.sqrt((d7 * d7) - 1.0d) * d8);
                } else if (d5 >= ShadowDrawableWrapper.COS_45 && d5 < 1.0d) {
                    this.f2369h = Math.sqrt(1.0d - (d5 * d5)) * this.f2362a;
                }
                this.f2364c = true;
            } else {
                throw new IllegalStateException("Error: Final position of the spring must be set before the animation starts");
            }
        }
        double d9 = j / 1000.0d;
        double d10 = d2 - this.i;
        double d11 = this.f2363b;
        if (d11 > 1.0d) {
            double d12 = this.f2368g;
            double d13 = this.f2367f;
            double d14 = d10 - (((d12 * d10) - d3) / (d12 - d13));
            double d15 = ((d10 * d12) - d3) / (d12 - d13);
            d4 = (Math.pow(2.718281828459045d, this.f2367f * d9) * d15) + (Math.pow(2.718281828459045d, d12 * d9) * d14);
            double d16 = this.f2368g;
            double pow = Math.pow(2.718281828459045d, d16 * d9) * d14 * d16;
            double d17 = this.f2367f;
            cos = (Math.pow(2.718281828459045d, d17 * d9) * d15 * d17) + pow;
        } else if (d11 == 1.0d) {
            double d18 = this.f2362a;
            double d19 = (d18 * d10) + d3;
            double d20 = (d19 * d9) + d10;
            double pow2 = Math.pow(2.718281828459045d, (-d18) * d9) * d20;
            double pow3 = Math.pow(2.718281828459045d, (-this.f2362a) * d9) * d20;
            double d21 = this.f2362a;
            cos = (Math.pow(2.718281828459045d, (-d21) * d9) * d19) + (pow3 * (-d21));
            d4 = pow2;
        } else {
            double d22 = 1.0d / this.f2369h;
            double d23 = this.f2362a;
            double d24 = ((d11 * d23 * d10) + d3) * d22;
            double sin = ((Math.sin(this.f2369h * d9) * d24) + (Math.cos(this.f2369h * d9) * d10)) * Math.pow(2.718281828459045d, (-d11) * d23 * d9);
            double d25 = this.f2362a;
            double d26 = this.f2363b;
            double d27 = (-d25) * sin * d26;
            double pow4 = Math.pow(2.718281828459045d, (-d26) * d25 * d9);
            double d28 = this.f2369h;
            double d29 = (-d28) * d10;
            double d30 = this.f2369h;
            cos = (((Math.cos(d30 * d9) * d24 * d30) + (Math.sin(d28 * d9) * d29)) * pow4) + d27;
            d4 = sin;
        }
        b.o oVar = this.j;
        oVar.f2359a = (float) (d4 + this.i);
        oVar.f2360b = (float) cos;
        return oVar;
    }

    public e(float f2) {
        this.f2362a = Math.sqrt(1500.0d);
        this.f2363b = 0.5d;
        this.f2364c = false;
        this.i = Double.MAX_VALUE;
        this.j = new b.o();
        this.i = f2;
    }
}