package c.a.a.d0;

import android.graphics.PointF;
import android.view.animation.Interpolator;
import com.google.android.material.internal.StaticLayoutBuilderCompat;

/* compiled from: Keyframe.java */
/* loaded from: classes.dex */
public class a<T> {

    /* renamed from: a  reason: collision with root package name */
    public final c.a.a.d f3045a;

    /* renamed from: b  reason: collision with root package name */
    public final T f3046b;

    /* renamed from: c  reason: collision with root package name */
    public T f3047c;

    /* renamed from: d  reason: collision with root package name */
    public final Interpolator f3048d;

    /* renamed from: e  reason: collision with root package name */
    public final float f3049e;

    /* renamed from: f  reason: collision with root package name */
    public Float f3050f;

    /* renamed from: g  reason: collision with root package name */
    public float f3051g;

    /* renamed from: h  reason: collision with root package name */
    public float f3052h;
    public int i;
    public int j;
    public float k;
    public float l;
    public PointF m;
    public PointF n;

    public a(c.a.a.d dVar, T t, T t2, Interpolator interpolator, float f2, Float f3) {
        this.f3051g = -3987645.8f;
        this.f3052h = -3987645.8f;
        this.i = 784923401;
        this.j = 784923401;
        this.k = Float.MIN_VALUE;
        this.l = Float.MIN_VALUE;
        this.m = null;
        this.n = null;
        this.f3045a = dVar;
        this.f3046b = t;
        this.f3047c = t2;
        this.f3048d = interpolator;
        this.f3049e = f2;
        this.f3050f = f3;
    }

    public boolean a(float f2) {
        return f2 >= c() && f2 < b();
    }

    public float b() {
        if (this.f3045a == null) {
            return 1.0f;
        }
        if (this.l == Float.MIN_VALUE) {
            if (this.f3050f == null) {
                this.l = 1.0f;
            } else {
                this.l = ((this.f3050f.floatValue() - this.f3049e) / this.f3045a.c()) + c();
            }
        }
        return this.l;
    }

    public float c() {
        c.a.a.d dVar = this.f3045a;
        if (dVar == null) {
            return StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        }
        if (this.k == Float.MIN_VALUE) {
            this.k = (this.f3049e - dVar.k) / dVar.c();
        }
        return this.k;
    }

    public boolean d() {
        return this.f3048d == null;
    }

    public String toString() {
        StringBuilder x = c.b.a.a.a.x("Keyframe{startValue=");
        x.append(this.f3046b);
        x.append(", endValue=");
        x.append(this.f3047c);
        x.append(", startFrame=");
        x.append(this.f3049e);
        x.append(", endFrame=");
        x.append(this.f3050f);
        x.append(", interpolator=");
        x.append(this.f3048d);
        x.append('}');
        return x.toString();
    }

    public a(T t) {
        this.f3051g = -3987645.8f;
        this.f3052h = -3987645.8f;
        this.i = 784923401;
        this.j = 784923401;
        this.k = Float.MIN_VALUE;
        this.l = Float.MIN_VALUE;
        this.m = null;
        this.n = null;
        this.f3045a = null;
        this.f3046b = t;
        this.f3047c = t;
        this.f3048d = null;
        this.f3049e = Float.MIN_VALUE;
        this.f3050f = Float.valueOf(Float.MAX_VALUE);
    }
}