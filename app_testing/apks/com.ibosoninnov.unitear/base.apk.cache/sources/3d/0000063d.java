package c.a.a.b0;

import android.graphics.PathMeasure;
import android.graphics.PointF;
import android.view.animation.Interpolator;
import android.view.animation.LinearInterpolator;
import android.view.animation.PathInterpolator;
import c.a.a.b0.h0.c;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import java.lang.ref.WeakReference;

/* compiled from: KeyframeParser.java */
/* loaded from: classes.dex */
public class p {

    /* renamed from: b  reason: collision with root package name */
    public static b.f.i<WeakReference<Interpolator>> f3001b;

    /* renamed from: a  reason: collision with root package name */
    public static final Interpolator f3000a = new LinearInterpolator();

    /* renamed from: c  reason: collision with root package name */
    public static c.a f3002c = c.a.a("t", "s", "e", "o", "i", "h", "to", "ti");

    public static <T> c.a.a.d0.a<T> a(c.a.a.b0.h0.c cVar, c.a.a.d dVar, float f2, g0<T> g0Var, boolean z) {
        Interpolator interpolator;
        Interpolator interpolator2;
        T t;
        WeakReference<Interpolator> e2;
        if (z) {
            cVar.C();
            float f3 = 0.0f;
            boolean z2 = false;
            PointF pointF = null;
            PointF pointF2 = null;
            T t2 = null;
            T t3 = null;
            PointF pointF3 = null;
            PointF pointF4 = null;
            while (cVar.G()) {
                switch (cVar.O(f3002c)) {
                    case 0:
                        f3 = (float) cVar.I();
                        break;
                    case 1:
                        t3 = g0Var.a(cVar, f2);
                        break;
                    case 2:
                        t2 = g0Var.a(cVar, f2);
                        break;
                    case 3:
                        pointF = o.b(cVar, f2);
                        break;
                    case 4:
                        pointF2 = o.b(cVar, f2);
                        break;
                    case 5:
                        if (cVar.J() != 1) {
                            z2 = false;
                            break;
                        } else {
                            z2 = true;
                            break;
                        }
                    case 6:
                        pointF4 = o.b(cVar, f2);
                        break;
                    case 7:
                        pointF3 = o.b(cVar, f2);
                        break;
                    default:
                        cVar.Q();
                        break;
                }
            }
            cVar.E();
            if (z2) {
                interpolator2 = f3000a;
                t = t3;
            } else {
                if (pointF != null && pointF2 != null) {
                    float f4 = -f2;
                    pointF.x = c.a.a.c0.f.b(pointF.x, f4, f2);
                    pointF.y = c.a.a.c0.f.b(pointF.y, -100.0f, 100.0f);
                    pointF2.x = c.a.a.c0.f.b(pointF2.x, f4, f2);
                    float b2 = c.a.a.c0.f.b(pointF2.y, -100.0f, 100.0f);
                    pointF2.y = b2;
                    float f5 = pointF.x;
                    float f6 = pointF.y;
                    float f7 = pointF2.x;
                    PathMeasure pathMeasure = c.a.a.c0.g.f3031a;
                    int i = f5 != StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD ? (int) (527 * f5) : 17;
                    if (f6 != StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                        i = (int) (i * 31 * f6);
                    }
                    if (f7 != StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                        i = (int) (i * 31 * f7);
                    }
                    if (b2 != StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                        i = (int) (i * 31 * b2);
                    }
                    synchronized (p.class) {
                        if (f3001b == null) {
                            f3001b = new b.f.i<>(10);
                        }
                        e2 = f3001b.e(i, null);
                    }
                    interpolator = e2 != null ? e2.get() : null;
                    if (e2 == null || interpolator == null) {
                        pointF.x /= f2;
                        pointF.y /= f2;
                        float f8 = pointF2.x / f2;
                        pointF2.x = f8;
                        float f9 = pointF2.y / f2;
                        pointF2.y = f9;
                        try {
                            interpolator = new PathInterpolator(pointF.x, pointF.y, f8, f9);
                        } catch (IllegalArgumentException e3) {
                            if (e3.getMessage().equals("The Path cannot loop back on itself.")) {
                                interpolator = new PathInterpolator(Math.min(pointF.x, 1.0f), pointF.y, Math.max(pointF2.x, (float) StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD), pointF2.y);
                            } else {
                                interpolator = new LinearInterpolator();
                            }
                        }
                        try {
                            WeakReference<Interpolator> weakReference = new WeakReference<>(interpolator);
                            synchronized (p.class) {
                                f3001b.g(i, weakReference);
                            }
                        } catch (ArrayIndexOutOfBoundsException unused) {
                        }
                    }
                } else {
                    interpolator = f3000a;
                }
                interpolator2 = interpolator;
                t = t2;
            }
            c.a.a.d0.a<T> aVar = new c.a.a.d0.a<>(dVar, t3, t, interpolator2, f3, null);
            aVar.m = pointF4;
            aVar.n = pointF3;
            return aVar;
        }
        return new c.a.a.d0.a<>(g0Var.a(cVar, f2));
    }
}