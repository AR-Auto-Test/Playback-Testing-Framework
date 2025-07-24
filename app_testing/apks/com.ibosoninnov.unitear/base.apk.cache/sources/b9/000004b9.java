package b.j.k;

import android.content.res.Resources;
import android.os.SystemClock;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewConfiguration;
import android.view.animation.AccelerateInterpolator;
import android.view.animation.AnimationUtils;
import android.view.animation.Interpolator;
import android.widget.ListView;
import b.j.j.q;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicInteger;

/* compiled from: AutoScrollHelper.java */
/* loaded from: classes.dex */
public abstract class a implements View.OnTouchListener {

    /* renamed from: b  reason: collision with root package name */
    public static final int f2281b = ViewConfiguration.getTapTimeout();

    /* renamed from: c  reason: collision with root package name */
    public final C0039a f2282c;

    /* renamed from: d  reason: collision with root package name */
    public final Interpolator f2283d;

    /* renamed from: e  reason: collision with root package name */
    public final View f2284e;

    /* renamed from: f  reason: collision with root package name */
    public Runnable f2285f;

    /* renamed from: g  reason: collision with root package name */
    public float[] f2286g;

    /* renamed from: h  reason: collision with root package name */
    public float[] f2287h;
    public int i;
    public int j;
    public float[] k;
    public float[] l;
    public float[] m;
    public boolean n;
    public boolean o;
    public boolean p;
    public boolean q;
    public boolean r;

    /* compiled from: AutoScrollHelper.java */
    /* renamed from: b.j.k.a$a  reason: collision with other inner class name */
    /* loaded from: classes.dex */
    public static class C0039a {

        /* renamed from: a  reason: collision with root package name */
        public int f2288a;

        /* renamed from: b  reason: collision with root package name */
        public int f2289b;

        /* renamed from: c  reason: collision with root package name */
        public float f2290c;

        /* renamed from: d  reason: collision with root package name */
        public float f2291d;
        public float j;
        public int k;

        /* renamed from: e  reason: collision with root package name */
        public long f2292e = Long.MIN_VALUE;
        public long i = -1;

        /* renamed from: f  reason: collision with root package name */
        public long f2293f = 0;

        /* renamed from: g  reason: collision with root package name */
        public int f2294g = 0;

        /* renamed from: h  reason: collision with root package name */
        public int f2295h = 0;

        public final float a(long j) {
            long j2 = this.f2292e;
            if (j < j2) {
                return StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
            }
            long j3 = this.i;
            if (j3 >= 0 && j >= j3) {
                float f2 = this.j;
                return (a.b(((float) (j - j3)) / this.k, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 1.0f) * f2) + (1.0f - f2);
            }
            return a.b(((float) (j - j2)) / this.f2288a, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 1.0f) * 0.5f;
        }
    }

    /* compiled from: AutoScrollHelper.java */
    /* loaded from: classes.dex */
    public class b implements Runnable {
        public b() {
        }

        @Override // java.lang.Runnable
        public void run() {
            a aVar = a.this;
            if (aVar.q) {
                if (aVar.o) {
                    aVar.o = false;
                    C0039a c0039a = aVar.f2282c;
                    Objects.requireNonNull(c0039a);
                    long currentAnimationTimeMillis = AnimationUtils.currentAnimationTimeMillis();
                    c0039a.f2292e = currentAnimationTimeMillis;
                    c0039a.i = -1L;
                    c0039a.f2293f = currentAnimationTimeMillis;
                    c0039a.j = 0.5f;
                    c0039a.f2294g = 0;
                    c0039a.f2295h = 0;
                }
                C0039a c0039a2 = a.this.f2282c;
                if (!(c0039a2.i > 0 && AnimationUtils.currentAnimationTimeMillis() > c0039a2.i + ((long) c0039a2.k)) && a.this.e()) {
                    a aVar2 = a.this;
                    if (aVar2.p) {
                        aVar2.p = false;
                        Objects.requireNonNull(aVar2);
                        long uptimeMillis = SystemClock.uptimeMillis();
                        MotionEvent obtain = MotionEvent.obtain(uptimeMillis, uptimeMillis, 3, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 0);
                        aVar2.f2284e.onTouchEvent(obtain);
                        obtain.recycle();
                    }
                    if (c0039a2.f2293f != 0) {
                        long currentAnimationTimeMillis2 = AnimationUtils.currentAnimationTimeMillis();
                        float a2 = c0039a2.a(currentAnimationTimeMillis2);
                        c0039a2.f2293f = currentAnimationTimeMillis2;
                        float f2 = ((float) (currentAnimationTimeMillis2 - c0039a2.f2293f)) * ((a2 * 4.0f) + ((-4.0f) * a2 * a2));
                        c0039a2.f2294g = (int) (c0039a2.f2290c * f2);
                        int i = (int) (f2 * c0039a2.f2291d);
                        c0039a2.f2295h = i;
                        ((c) a.this).s.scrollListBy(i);
                        View view = a.this.f2284e;
                        AtomicInteger atomicInteger = q.f2214a;
                        view.postOnAnimation(this);
                        return;
                    }
                    throw new RuntimeException("Cannot compute scroll delta before calling start()");
                }
                a.this.q = false;
            }
        }
    }

    public a(View view) {
        C0039a c0039a = new C0039a();
        this.f2282c = c0039a;
        this.f2283d = new AccelerateInterpolator();
        this.f2286g = new float[]{StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD};
        this.f2287h = new float[]{Float.MAX_VALUE, Float.MAX_VALUE};
        this.k = new float[]{StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD};
        this.l = new float[]{StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD};
        this.m = new float[]{Float.MAX_VALUE, Float.MAX_VALUE};
        this.f2284e = view;
        float f2 = Resources.getSystem().getDisplayMetrics().density;
        float[] fArr = this.m;
        float f3 = ((int) ((1575.0f * f2) + 0.5f)) / 1000.0f;
        fArr[0] = f3;
        fArr[1] = f3;
        float[] fArr2 = this.l;
        float f4 = ((int) ((f2 * 315.0f) + 0.5f)) / 1000.0f;
        fArr2[0] = f4;
        fArr2[1] = f4;
        this.i = 1;
        float[] fArr3 = this.f2287h;
        fArr3[0] = Float.MAX_VALUE;
        fArr3[1] = Float.MAX_VALUE;
        float[] fArr4 = this.f2286g;
        fArr4[0] = 0.2f;
        fArr4[1] = 0.2f;
        float[] fArr5 = this.k;
        fArr5[0] = 0.001f;
        fArr5[1] = 0.001f;
        this.j = f2281b;
        c0039a.f2288a = 500;
        c0039a.f2289b = 500;
    }

    public static float b(float f2, float f3, float f4) {
        return f2 > f4 ? f4 : f2 < f3 ? f3 : f2;
    }

    /* JADX WARN: Removed duplicated region for block: B:12:0x003d A[RETURN] */
    /* JADX WARN: Removed duplicated region for block: B:13:0x003e  */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public final float a(int i, float f2, float f3, float f4) {
        float f5;
        float interpolation;
        int i2;
        float b2 = b(this.f2286g[i] * f3, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, this.f2287h[i]);
        float c2 = c(f3 - f2, b2) - c(f2, b2);
        if (c2 < StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
            interpolation = -this.f2283d.getInterpolation(-c2);
        } else if (c2 <= StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
            f5 = 0.0f;
            i2 = (f5 > StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD ? 1 : (f5 == StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD ? 0 : -1));
            if (i2 != 0) {
                return StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
            }
            float f6 = this.k[i];
            float f7 = this.l[i];
            float f8 = this.m[i];
            float f9 = f6 * f4;
            if (i2 > 0) {
                return b(f5 * f9, f7, f8);
            }
            return -b((-f5) * f9, f7, f8);
        } else {
            interpolation = this.f2283d.getInterpolation(c2);
        }
        f5 = b(interpolation, -1.0f, 1.0f);
        i2 = (f5 > StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD ? 1 : (f5 == StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD ? 0 : -1));
        if (i2 != 0) {
        }
    }

    public final float c(float f2, float f3) {
        if (f3 == StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
            return StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        }
        int i = this.i;
        if (i == 0 || i == 1) {
            if (f2 < f3) {
                if (f2 >= StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                    return 1.0f - (f2 / f3);
                }
                if (this.q && i == 1) {
                    return 1.0f;
                }
            }
        } else if (i == 2 && f2 < StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
            return f2 / (-f3);
        }
        return StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
    }

    public final void d() {
        int i = 0;
        if (this.o) {
            this.q = false;
            return;
        }
        C0039a c0039a = this.f2282c;
        Objects.requireNonNull(c0039a);
        long currentAnimationTimeMillis = AnimationUtils.currentAnimationTimeMillis();
        int i2 = (int) (currentAnimationTimeMillis - c0039a.f2292e);
        int i3 = c0039a.f2289b;
        if (i2 > i3) {
            i = i3;
        } else if (i2 >= 0) {
            i = i2;
        }
        c0039a.k = i;
        c0039a.j = c0039a.a(currentAnimationTimeMillis);
        c0039a.i = currentAnimationTimeMillis;
    }

    /* JADX WARN: Removed duplicated region for block: B:22:? A[RETURN, SYNTHETIC] */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public boolean e() {
        boolean z;
        C0039a c0039a = this.f2282c;
        float f2 = c0039a.f2291d;
        int abs = (int) (f2 / Math.abs(f2));
        float f3 = c0039a.f2290c;
        int abs2 = (int) (f3 / Math.abs(f3));
        if (abs != 0) {
            ListView listView = ((c) this).s;
            int count = listView.getCount();
            if (count != 0) {
                int childCount = listView.getChildCount();
                int firstVisiblePosition = listView.getFirstVisiblePosition();
                int i = firstVisiblePosition + childCount;
                if (abs <= 0 ? !(abs >= 0 || (firstVisiblePosition <= 0 && listView.getChildAt(0).getTop() >= 0)) : !(i >= count && listView.getChildAt(childCount - 1).getBottom() <= listView.getHeight())) {
                    z = true;
                    if (z) {
                        return true;
                    }
                }
            }
            z = false;
            if (z) {
            }
        }
        return false;
    }

    /* JADX WARN: Code restructure failed: missing block: B:11:0x0013, code lost:
        if (r0 != 3) goto L12;
     */
    @Override // android.view.View.OnTouchListener
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public boolean onTouch(View view, MotionEvent motionEvent) {
        int i;
        if (this.r) {
            int actionMasked = motionEvent.getActionMasked();
            if (actionMasked != 0) {
                if (actionMasked != 1) {
                    if (actionMasked != 2) {
                    }
                }
                d();
                return false;
            }
            this.p = true;
            this.n = false;
            float a2 = a(0, motionEvent.getX(), view.getWidth(), this.f2284e.getWidth());
            float a3 = a(1, motionEvent.getY(), view.getHeight(), this.f2284e.getHeight());
            C0039a c0039a = this.f2282c;
            c0039a.f2290c = a2;
            c0039a.f2291d = a3;
            if (!this.q && e()) {
                if (this.f2285f == null) {
                    this.f2285f = new b();
                }
                this.q = true;
                this.o = true;
                if (!this.n && (i = this.j) > 0) {
                    View view2 = this.f2284e;
                    Runnable runnable = this.f2285f;
                    long j = i;
                    AtomicInteger atomicInteger = q.f2214a;
                    view2.postOnAnimationDelayed(runnable, j);
                } else {
                    this.f2285f.run();
                }
                this.n = true;
            }
            return false;
        }
        return false;
    }
}