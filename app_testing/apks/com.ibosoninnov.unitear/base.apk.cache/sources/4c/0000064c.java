package c.a.a.c0;

import android.animation.Animator;
import android.graphics.PointF;
import android.view.Choreographer;
import com.google.android.material.internal.StaticLayoutBuilderCompat;

/* compiled from: LottieValueAnimator.java */
/* loaded from: classes.dex */
public class d extends a implements Choreographer.FrameCallback {
    public c.a.a.d k;

    /* renamed from: d  reason: collision with root package name */
    public float f3023d = 1.0f;

    /* renamed from: e  reason: collision with root package name */
    public boolean f3024e = false;

    /* renamed from: f  reason: collision with root package name */
    public long f3025f = 0;

    /* renamed from: g  reason: collision with root package name */
    public float f3026g = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;

    /* renamed from: h  reason: collision with root package name */
    public int f3027h = 0;
    public float i = -2.14748365E9f;
    public float j = 2.14748365E9f;
    public boolean l = false;

    public void c() {
        i();
        a(g());
    }

    @Override // android.animation.ValueAnimator, android.animation.Animator
    public void cancel() {
        for (Animator.AnimatorListener animatorListener : this.f3020c) {
            animatorListener.onAnimationCancel(this);
        }
        i();
    }

    public float d() {
        c.a.a.d dVar = this.k;
        if (dVar == null) {
            return StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        }
        float f2 = this.f3026g;
        float f3 = dVar.k;
        return (f2 - f3) / (dVar.l - f3);
    }

    @Override // android.view.Choreographer.FrameCallback
    public void doFrame(long j) {
        h();
        c.a.a.d dVar = this.k;
        if (dVar == null || !this.l) {
            return;
        }
        long j2 = this.f3025f;
        float abs = ((float) (j2 != 0 ? j - j2 : 0L)) / ((1.0E9f / dVar.m) / Math.abs(this.f3023d));
        float f2 = this.f3026g;
        if (g()) {
            abs = -abs;
        }
        float f3 = f2 + abs;
        this.f3026g = f3;
        float f4 = f();
        float e2 = e();
        PointF pointF = f.f3030a;
        boolean z = !(f3 >= f4 && f3 <= e2);
        this.f3026g = f.b(this.f3026g, f(), e());
        this.f3025f = j;
        b();
        if (z) {
            if (getRepeatCount() != -1 && this.f3027h >= getRepeatCount()) {
                this.f3026g = this.f3023d < StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD ? f() : e();
                i();
                a(g());
            } else {
                for (Animator.AnimatorListener animatorListener : this.f3020c) {
                    animatorListener.onAnimationRepeat(this);
                }
                this.f3027h++;
                if (getRepeatMode() == 2) {
                    this.f3024e = !this.f3024e;
                    this.f3023d = -this.f3023d;
                } else {
                    this.f3026g = g() ? e() : f();
                }
                this.f3025f = j;
            }
        }
        if (this.k != null) {
            float f5 = this.f3026g;
            if (f5 < this.i || f5 > this.j) {
                throw new IllegalStateException(String.format("Frame must be [%f,%f]. It is %f", Float.valueOf(this.i), Float.valueOf(this.j), Float.valueOf(this.f3026g)));
            }
        }
        c.a.a.c.a("LottieValueAnimator#doFrame");
    }

    public float e() {
        c.a.a.d dVar = this.k;
        if (dVar == null) {
            return StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        }
        float f2 = this.j;
        return f2 == 2.14748365E9f ? dVar.l : f2;
    }

    public float f() {
        c.a.a.d dVar = this.k;
        if (dVar == null) {
            return StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        }
        float f2 = this.i;
        return f2 == -2.14748365E9f ? dVar.k : f2;
    }

    public final boolean g() {
        return this.f3023d < StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
    }

    @Override // android.animation.ValueAnimator
    public float getAnimatedFraction() {
        float f2;
        float e2;
        float f3;
        if (this.k == null) {
            return StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        }
        if (g()) {
            f2 = e() - this.f3026g;
            e2 = e();
            f3 = f();
        } else {
            f2 = this.f3026g - f();
            e2 = e();
            f3 = f();
        }
        return f2 / (e2 - f3);
    }

    @Override // android.animation.ValueAnimator
    public Object getAnimatedValue() {
        return Float.valueOf(d());
    }

    @Override // android.animation.ValueAnimator, android.animation.Animator
    public long getDuration() {
        c.a.a.d dVar = this.k;
        if (dVar == null) {
            return 0L;
        }
        return dVar.b();
    }

    public void h() {
        if (this.l) {
            Choreographer.getInstance().removeFrameCallback(this);
            Choreographer.getInstance().postFrameCallback(this);
        }
    }

    public void i() {
        Choreographer.getInstance().removeFrameCallback(this);
        this.l = false;
    }

    @Override // android.animation.ValueAnimator, android.animation.Animator
    public boolean isRunning() {
        return this.l;
    }

    public void j(float f2) {
        if (this.f3026g == f2) {
            return;
        }
        this.f3026g = f.b(f2, f(), e());
        this.f3025f = 0L;
        b();
    }

    public void k(float f2, float f3) {
        if (f2 <= f3) {
            c.a.a.d dVar = this.k;
            float f4 = dVar == null ? -3.4028235E38f : dVar.k;
            float f5 = dVar == null ? Float.MAX_VALUE : dVar.l;
            this.i = f.b(f2, f4, f5);
            this.j = f.b(f3, f4, f5);
            j((int) f.b(this.f3026g, f2, f3));
            return;
        }
        throw new IllegalArgumentException(String.format("minFrame (%s) must be <= maxFrame (%s)", Float.valueOf(f2), Float.valueOf(f3)));
    }

    @Override // android.animation.ValueAnimator
    public void setRepeatMode(int i) {
        super.setRepeatMode(i);
        if (i == 2 || !this.f3024e) {
            return;
        }
        this.f3024e = false;
        this.f3023d = -this.f3023d;
    }
}