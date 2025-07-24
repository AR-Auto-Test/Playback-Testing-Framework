package c.e.b.p000if;

import android.view.animation.Interpolator;

/* compiled from: CustomBounceInterpolator.java */
/* renamed from: c.e.b.if.i  reason: invalid package */
/* loaded from: classes2.dex */
public class i implements Interpolator {

    /* renamed from: a  reason: collision with root package name */
    public double f4877a;

    /* renamed from: b  reason: collision with root package name */
    public double f4878b;

    public i(double d2, double d3) {
        this.f4877a = 1.0d;
        this.f4878b = 10.0d;
        this.f4877a = d2;
        this.f4878b = d3;
    }

    @Override // android.animation.TimeInterpolator
    public float getInterpolation(float f2) {
        return (float) ((Math.cos(this.f4878b * f2) * Math.pow(2.718281828459045d, (-f2) / this.f4877a) * (-1.0d)) + 1.0d);
    }
}