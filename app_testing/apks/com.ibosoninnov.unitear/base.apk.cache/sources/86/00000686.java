package c.a.a.x.b;

import android.graphics.Canvas;
import android.graphics.ColorFilter;
import android.graphics.LinearGradient;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.PointF;
import android.graphics.RadialGradient;
import android.graphics.RectF;
import android.graphics.Shader;
import c.a.a.x.c.a;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import java.util.ArrayList;
import java.util.List;

/* compiled from: GradientFillContent.java */
/* loaded from: classes.dex */
public class h implements e, a.b, k {

    /* renamed from: a  reason: collision with root package name */
    public final String f3174a;

    /* renamed from: b  reason: collision with root package name */
    public final boolean f3175b;

    /* renamed from: c  reason: collision with root package name */
    public final c.a.a.z.l.b f3176c;

    /* renamed from: d  reason: collision with root package name */
    public final b.f.e<LinearGradient> f3177d = new b.f.e<>(10);

    /* renamed from: e  reason: collision with root package name */
    public final b.f.e<RadialGradient> f3178e = new b.f.e<>(10);

    /* renamed from: f  reason: collision with root package name */
    public final Path f3179f;

    /* renamed from: g  reason: collision with root package name */
    public final Paint f3180g;

    /* renamed from: h  reason: collision with root package name */
    public final RectF f3181h;
    public final List<m> i;
    public final int j;
    public final c.a.a.x.c.a<c.a.a.z.k.c, c.a.a.z.k.c> k;
    public final c.a.a.x.c.a<Integer, Integer> l;
    public final c.a.a.x.c.a<PointF, PointF> m;
    public final c.a.a.x.c.a<PointF, PointF> n;
    public c.a.a.x.c.a<ColorFilter, ColorFilter> o;
    public c.a.a.x.c.p p;
    public final c.a.a.j q;
    public final int r;

    public h(c.a.a.j jVar, c.a.a.z.l.b bVar, c.a.a.z.k.d dVar) {
        Path path = new Path();
        this.f3179f = path;
        this.f3180g = new c.a.a.x.a(1);
        this.f3181h = new RectF();
        this.i = new ArrayList();
        this.f3176c = bVar;
        this.f3174a = dVar.f3315g;
        this.f3175b = dVar.f3316h;
        this.q = jVar;
        this.j = dVar.f3309a;
        path.setFillType(dVar.f3310b);
        this.r = (int) (jVar.f3075c.b() / 32.0f);
        c.a.a.x.c.a<c.a.a.z.k.c, c.a.a.z.k.c> a2 = dVar.f3311c.a();
        this.k = a2;
        a2.f3223a.add(this);
        bVar.e(a2);
        c.a.a.x.c.a<Integer, Integer> a3 = dVar.f3312d.a();
        this.l = a3;
        a3.f3223a.add(this);
        bVar.e(a3);
        c.a.a.x.c.a<PointF, PointF> a4 = dVar.f3313e.a();
        this.m = a4;
        a4.f3223a.add(this);
        bVar.e(a4);
        c.a.a.x.c.a<PointF, PointF> a5 = dVar.f3314f.a();
        this.n = a5;
        a5.f3223a.add(this);
        bVar.e(a5);
    }

    @Override // c.a.a.x.c.a.b
    public void a() {
        this.q.invalidateSelf();
    }

    @Override // c.a.a.x.b.c
    public void b(List<c> list, List<c> list2) {
        for (int i = 0; i < list2.size(); i++) {
            c cVar = list2.get(i);
            if (cVar instanceof m) {
                this.i.add((m) cVar);
            }
        }
    }

    @Override // c.a.a.z.f
    public void c(c.a.a.z.e eVar, int i, List<c.a.a.z.e> list, c.a.a.z.e eVar2) {
        c.a.a.c0.f.f(eVar, i, list, eVar2, this);
    }

    @Override // c.a.a.x.b.e
    public void d(RectF rectF, Matrix matrix, boolean z) {
        this.f3179f.reset();
        for (int i = 0; i < this.i.size(); i++) {
            this.f3179f.addPath(this.i.get(i).g(), matrix);
        }
        this.f3179f.computeBounds(rectF, false);
        rectF.set(rectF.left - 1.0f, rectF.top - 1.0f, rectF.right + 1.0f, rectF.bottom + 1.0f);
    }

    public final int[] e(int[] iArr) {
        c.a.a.x.c.p pVar = this.p;
        if (pVar != null) {
            Integer[] numArr = (Integer[]) pVar.e();
            int i = 0;
            if (iArr.length == numArr.length) {
                while (i < iArr.length) {
                    iArr[i] = numArr[i].intValue();
                    i++;
                }
            } else {
                iArr = new int[numArr.length];
                while (i < numArr.length) {
                    iArr[i] = numArr[i].intValue();
                    i++;
                }
            }
        }
        return iArr;
    }

    /* JADX DEBUG: Multi-variable search result rejected for r4v12, resolved type: b.f.e<android.graphics.RadialGradient> */
    /* JADX WARN: Multi-variable type inference failed */
    @Override // c.a.a.x.b.e
    public void f(Canvas canvas, Matrix matrix, int i) {
        RadialGradient d2;
        if (this.f3175b) {
            return;
        }
        this.f3179f.reset();
        for (int i2 = 0; i2 < this.i.size(); i2++) {
            this.f3179f.addPath(this.i.get(i2).g(), matrix);
        }
        this.f3179f.computeBounds(this.f3181h, false);
        if (this.j == 1) {
            long i3 = i();
            d2 = this.f3177d.d(i3);
            if (d2 == null) {
                PointF e2 = this.m.e();
                PointF e3 = this.n.e();
                c.a.a.z.k.c e4 = this.k.e();
                LinearGradient linearGradient = new LinearGradient(e2.x, e2.y, e3.x, e3.y, e(e4.f3308b), e4.f3307a, Shader.TileMode.CLAMP);
                this.f3177d.g(i3, linearGradient);
                d2 = linearGradient;
            }
        } else {
            long i4 = i();
            d2 = this.f3178e.d(i4);
            if (d2 == null) {
                PointF e5 = this.m.e();
                PointF e6 = this.n.e();
                c.a.a.z.k.c e7 = this.k.e();
                int[] e8 = e(e7.f3308b);
                float[] fArr = e7.f3307a;
                float f2 = e5.x;
                float f3 = e5.y;
                float hypot = (float) Math.hypot(e6.x - f2, e6.y - f3);
                if (hypot <= StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                    hypot = 0.001f;
                }
                d2 = new RadialGradient(f2, f3, hypot, e8, fArr, Shader.TileMode.CLAMP);
                this.f3178e.g(i4, d2);
            }
        }
        d2.setLocalMatrix(matrix);
        this.f3180g.setShader(d2);
        c.a.a.x.c.a<ColorFilter, ColorFilter> aVar = this.o;
        if (aVar != null) {
            this.f3180g.setColorFilter(aVar.e());
        }
        this.f3180g.setAlpha(c.a.a.c0.f.c((int) ((((i / 255.0f) * this.l.e().intValue()) / 100.0f) * 255.0f), 0, 255));
        canvas.drawPath(this.f3179f, this.f3180g);
        c.a.a.c.a("GradientFillContent#draw");
    }

    @Override // c.a.a.x.b.c
    public String getName() {
        return this.f3174a;
    }

    /* JADX DEBUG: Multi-variable search result rejected for r0v2, resolved type: java.lang.Integer[] */
    /* JADX DEBUG: Multi-variable search result rejected for r4v0, resolved type: c.a.a.d0.c<T> */
    /* JADX WARN: Multi-variable type inference failed */
    @Override // c.a.a.z.f
    public <T> void h(T t, c.a.a.d0.c<T> cVar) {
        if (t == c.a.a.o.f3117d) {
            c.a.a.x.c.a<Integer, Integer> aVar = this.l;
            c.a.a.d0.c<Integer> cVar2 = aVar.f3227e;
            aVar.f3227e = cVar;
        } else if (t == c.a.a.o.C) {
            c.a.a.x.c.a<ColorFilter, ColorFilter> aVar2 = this.o;
            if (aVar2 != null) {
                this.f3176c.u.remove(aVar2);
            }
            if (cVar == 0) {
                this.o = null;
                return;
            }
            c.a.a.x.c.p pVar = new c.a.a.x.c.p(cVar, null);
            this.o = pVar;
            pVar.f3223a.add(this);
            this.f3176c.e(this.o);
        } else if (t == c.a.a.o.D) {
            c.a.a.x.c.p pVar2 = this.p;
            if (pVar2 != null) {
                this.f3176c.u.remove(pVar2);
            }
            if (cVar == 0) {
                this.p = null;
                return;
            }
            c.a.a.x.c.p pVar3 = new c.a.a.x.c.p(cVar, null);
            this.p = pVar3;
            pVar3.f3223a.add(this);
            this.f3176c.e(this.p);
        }
    }

    public final int i() {
        int round = Math.round(this.m.f3226d * this.r);
        int round2 = Math.round(this.n.f3226d * this.r);
        int round3 = Math.round(this.k.f3226d * this.r);
        int i = round != 0 ? 527 * round : 17;
        if (round2 != 0) {
            i = i * 31 * round2;
        }
        return round3 != 0 ? i * 31 * round3 : i;
    }
}