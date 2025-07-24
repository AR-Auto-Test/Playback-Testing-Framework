package c.a.a.x.b;

import android.graphics.Canvas;
import android.graphics.LinearGradient;
import android.graphics.Matrix;
import android.graphics.PointF;
import android.graphics.RadialGradient;
import android.graphics.RectF;
import android.graphics.Shader;
import b.d.b.m0;

/* compiled from: GradientStrokeContent.java */
/* loaded from: classes.dex */
public class i extends a {
    public final String o;
    public final boolean p;
    public final b.f.e<LinearGradient> q;
    public final b.f.e<RadialGradient> r;
    public final RectF s;
    public final int t;
    public final int u;
    public final c.a.a.x.c.a<c.a.a.z.k.c, c.a.a.z.k.c> v;
    public final c.a.a.x.c.a<PointF, PointF> w;
    public final c.a.a.x.c.a<PointF, PointF> x;
    public c.a.a.x.c.p y;

    public i(c.a.a.j jVar, c.a.a.z.l.b bVar, c.a.a.z.k.e eVar) {
        super(jVar, bVar, m0.h(eVar.f3324h), m0.i(eVar.i), eVar.j, eVar.f3320d, eVar.f3323g, eVar.k, eVar.l);
        this.q = new b.f.e<>(10);
        this.r = new b.f.e<>(10);
        this.s = new RectF();
        this.o = eVar.f3317a;
        this.t = eVar.f3318b;
        this.p = eVar.m;
        this.u = (int) (jVar.f3075c.b() / 32.0f);
        c.a.a.x.c.a<c.a.a.z.k.c, c.a.a.z.k.c> a2 = eVar.f3319c.a();
        this.v = a2;
        a2.f3223a.add(this);
        bVar.e(a2);
        c.a.a.x.c.a<PointF, PointF> a3 = eVar.f3321e.a();
        this.w = a3;
        a3.f3223a.add(this);
        bVar.e(a3);
        c.a.a.x.c.a<PointF, PointF> a4 = eVar.f3322f.a();
        this.x = a4;
        a4.f3223a.add(this);
        bVar.e(a4);
    }

    public final int[] e(int[] iArr) {
        c.a.a.x.c.p pVar = this.y;
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

    /* JADX DEBUG: Multi-variable search result rejected for r3v14, resolved type: b.f.e<android.graphics.LinearGradient> */
    /* JADX DEBUG: Multi-variable search result rejected for r3v8, resolved type: b.f.e<android.graphics.RadialGradient> */
    /* JADX WARN: Multi-variable type inference failed */
    @Override // c.a.a.x.b.a, c.a.a.x.b.e
    public void f(Canvas canvas, Matrix matrix, int i) {
        RadialGradient d2;
        float f2;
        float f3;
        if (this.p) {
            return;
        }
        d(this.s, matrix, false);
        if (this.t == 1) {
            long i2 = i();
            d2 = this.q.d(i2);
            if (d2 == null) {
                PointF e2 = this.w.e();
                PointF e3 = this.x.e();
                c.a.a.z.k.c e4 = this.v.e();
                d2 = new LinearGradient(e2.x, e2.y, e3.x, e3.y, e(e4.f3308b), e4.f3307a, Shader.TileMode.CLAMP);
                this.q.g(i2, d2);
            }
        } else {
            long i3 = i();
            d2 = this.r.d(i3);
            if (d2 == null) {
                PointF e5 = this.w.e();
                PointF e6 = this.x.e();
                c.a.a.z.k.c e7 = this.v.e();
                int[] e8 = e(e7.f3308b);
                float[] fArr = e7.f3307a;
                d2 = new RadialGradient(e5.x, e5.y, (float) Math.hypot(e6.x - f2, e6.y - f3), e8, fArr, Shader.TileMode.CLAMP);
                this.r.g(i3, d2);
            }
        }
        d2.setLocalMatrix(matrix);
        this.i.setShader(d2);
        super.f(canvas, matrix, i);
    }

    @Override // c.a.a.x.b.c
    public String getName() {
        return this.o;
    }

    /* JADX DEBUG: Multi-variable search result rejected for r0v0, resolved type: java.lang.Integer[] */
    /* JADX WARN: Multi-variable type inference failed */
    @Override // c.a.a.x.b.a, c.a.a.z.f
    public <T> void h(T t, c.a.a.d0.c<T> cVar) {
        super.h(t, cVar);
        if (t == c.a.a.o.D) {
            c.a.a.x.c.p pVar = this.y;
            if (pVar != null) {
                this.f3144f.u.remove(pVar);
            }
            if (cVar == null) {
                this.y = null;
                return;
            }
            c.a.a.x.c.p pVar2 = new c.a.a.x.c.p(cVar, null);
            this.y = pVar2;
            pVar2.f3223a.add(this);
            this.f3144f.e(this.y);
        }
    }

    public final int i() {
        int round = Math.round(this.w.f3226d * this.u);
        int round2 = Math.round(this.x.f3226d * this.u);
        int round3 = Math.round(this.v.f3226d * this.u);
        int i = round != 0 ? 527 * round : 17;
        if (round2 != 0) {
            i = i * 31 * round2;
        }
        return round3 != 0 ? i * 31 * round3 : i;
    }
}