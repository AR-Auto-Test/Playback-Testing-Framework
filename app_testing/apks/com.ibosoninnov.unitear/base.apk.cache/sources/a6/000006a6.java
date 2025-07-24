package c.a.a.x.c;

import android.graphics.Matrix;
import android.graphics.PointF;
import c.a.a.x.c.a;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import java.util.Collections;

/* compiled from: TransformKeyframeAnimation.java */
/* loaded from: classes.dex */
public class o {

    /* renamed from: a  reason: collision with root package name */
    public final Matrix f3240a = new Matrix();

    /* renamed from: b  reason: collision with root package name */
    public final Matrix f3241b;

    /* renamed from: c  reason: collision with root package name */
    public final Matrix f3242c;

    /* renamed from: d  reason: collision with root package name */
    public final Matrix f3243d;

    /* renamed from: e  reason: collision with root package name */
    public final float[] f3244e;

    /* renamed from: f  reason: collision with root package name */
    public a<PointF, PointF> f3245f;

    /* renamed from: g  reason: collision with root package name */
    public a<?, PointF> f3246g;

    /* renamed from: h  reason: collision with root package name */
    public a<c.a.a.d0.d, c.a.a.d0.d> f3247h;
    public a<Float, Float> i;
    public a<Integer, Integer> j;
    public c k;
    public c l;
    public a<?, Float> m;
    public a<?, Float> n;

    public o(c.a.a.z.j.l lVar) {
        c.a.a.z.j.e eVar = lVar.f3293a;
        this.f3245f = eVar == null ? null : eVar.a();
        c.a.a.z.j.m<PointF, PointF> mVar = lVar.f3294b;
        this.f3246g = mVar == null ? null : mVar.a();
        c.a.a.z.j.g gVar = lVar.f3295c;
        this.f3247h = gVar == null ? null : gVar.a();
        c.a.a.z.j.b bVar = lVar.f3296d;
        this.i = bVar == null ? null : bVar.a();
        c.a.a.z.j.b bVar2 = lVar.f3298f;
        c cVar = bVar2 == null ? null : (c) bVar2.a();
        this.k = cVar;
        if (cVar != null) {
            this.f3241b = new Matrix();
            this.f3242c = new Matrix();
            this.f3243d = new Matrix();
            this.f3244e = new float[9];
        } else {
            this.f3241b = null;
            this.f3242c = null;
            this.f3243d = null;
            this.f3244e = null;
        }
        c.a.a.z.j.b bVar3 = lVar.f3299g;
        this.l = bVar3 == null ? null : (c) bVar3.a();
        c.a.a.z.j.d dVar = lVar.f3297e;
        if (dVar != null) {
            this.j = dVar.a();
        }
        c.a.a.z.j.b bVar4 = lVar.f3300h;
        if (bVar4 != null) {
            this.m = bVar4.a();
        } else {
            this.m = null;
        }
        c.a.a.z.j.b bVar5 = lVar.i;
        if (bVar5 != null) {
            this.n = bVar5.a();
        } else {
            this.n = null;
        }
    }

    public void a(c.a.a.z.l.b bVar) {
        bVar.e(this.j);
        bVar.e(this.m);
        bVar.e(this.n);
        bVar.e(this.f3245f);
        bVar.e(this.f3246g);
        bVar.e(this.f3247h);
        bVar.e(this.i);
        bVar.e(this.k);
        bVar.e(this.l);
    }

    public void b(a.b bVar) {
        a<Integer, Integer> aVar = this.j;
        if (aVar != null) {
            aVar.f3223a.add(bVar);
        }
        a<?, Float> aVar2 = this.m;
        if (aVar2 != null) {
            aVar2.f3223a.add(bVar);
        }
        a<?, Float> aVar3 = this.n;
        if (aVar3 != null) {
            aVar3.f3223a.add(bVar);
        }
        a<PointF, PointF> aVar4 = this.f3245f;
        if (aVar4 != null) {
            aVar4.f3223a.add(bVar);
        }
        a<?, PointF> aVar5 = this.f3246g;
        if (aVar5 != null) {
            aVar5.f3223a.add(bVar);
        }
        a<c.a.a.d0.d, c.a.a.d0.d> aVar6 = this.f3247h;
        if (aVar6 != null) {
            aVar6.f3223a.add(bVar);
        }
        a<Float, Float> aVar7 = this.i;
        if (aVar7 != null) {
            aVar7.f3223a.add(bVar);
        }
        c cVar = this.k;
        if (cVar != null) {
            cVar.f3223a.add(bVar);
        }
        c cVar2 = this.l;
        if (cVar2 != null) {
            cVar2.f3223a.add(bVar);
        }
    }

    /* JADX DEBUG: Multi-variable search result rejected for r5v0, resolved type: c.a.a.d0.c<T> */
    /* JADX WARN: Multi-variable type inference failed */
    public <T> boolean c(T t, c.a.a.d0.c<T> cVar) {
        c cVar2;
        c cVar3;
        a<?, Float> aVar;
        a<?, Float> aVar2;
        if (t == c.a.a.o.f3118e) {
            a<PointF, PointF> aVar3 = this.f3245f;
            if (aVar3 == null) {
                this.f3245f = new p(cVar, new PointF());
                return true;
            }
            c.a.a.d0.c<PointF> cVar4 = aVar3.f3227e;
            aVar3.f3227e = cVar;
            return true;
        } else if (t == c.a.a.o.f3119f) {
            a<?, PointF> aVar4 = this.f3246g;
            if (aVar4 == null) {
                this.f3246g = new p(cVar, new PointF());
                return true;
            }
            c.a.a.d0.c<PointF> cVar5 = aVar4.f3227e;
            aVar4.f3227e = cVar;
            return true;
        } else if (t == c.a.a.o.k) {
            a<c.a.a.d0.d, c.a.a.d0.d> aVar5 = this.f3247h;
            if (aVar5 == null) {
                this.f3247h = new p(cVar, new c.a.a.d0.d());
                return true;
            }
            c.a.a.d0.c<c.a.a.d0.d> cVar6 = aVar5.f3227e;
            aVar5.f3227e = cVar;
            return true;
        } else if (t == c.a.a.o.l) {
            a<Float, Float> aVar6 = this.i;
            if (aVar6 == null) {
                this.i = new p(cVar, Float.valueOf((float) StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD));
                return true;
            }
            c.a.a.d0.c<Float> cVar7 = aVar6.f3227e;
            aVar6.f3227e = cVar;
            return true;
        } else if (t == c.a.a.o.f3116c) {
            a<Integer, Integer> aVar7 = this.j;
            if (aVar7 == null) {
                this.j = new p(cVar, 100);
                return true;
            }
            c.a.a.d0.c<Integer> cVar8 = aVar7.f3227e;
            aVar7.f3227e = cVar;
            return true;
        } else if (t == c.a.a.o.y && (aVar2 = this.m) != null) {
            if (aVar2 == null) {
                this.m = new p(cVar, 100);
                return true;
            }
            c.a.a.d0.c<Float> cVar9 = aVar2.f3227e;
            aVar2.f3227e = cVar;
            return true;
        } else if (t == c.a.a.o.z && (aVar = this.n) != null) {
            if (aVar == null) {
                this.n = new p(cVar, 100);
                return true;
            }
            c.a.a.d0.c<Float> cVar10 = aVar.f3227e;
            aVar.f3227e = cVar;
            return true;
        } else if (t == c.a.a.o.m && (cVar3 = this.k) != null) {
            if (cVar3 == null) {
                this.k = new c(Collections.singletonList(new c.a.a.d0.a(Float.valueOf((float) StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD))));
            }
            c cVar11 = this.k;
            Object obj = cVar11.f3227e;
            cVar11.f3227e = cVar;
            return true;
        } else if (t != c.a.a.o.n || (cVar2 = this.l) == null) {
            return false;
        } else {
            if (cVar2 == null) {
                this.l = new c(Collections.singletonList(new c.a.a.d0.a(Float.valueOf((float) StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD))));
            }
            c cVar12 = this.l;
            Object obj2 = cVar12.f3227e;
            cVar12.f3227e = cVar;
            return true;
        }
    }

    public final void d() {
        for (int i = 0; i < 9; i++) {
            this.f3244e[i] = 0.0f;
        }
    }

    public Matrix e() {
        float j;
        this.f3240a.reset();
        a<?, PointF> aVar = this.f3246g;
        if (aVar != null) {
            PointF e2 = aVar.e();
            float f2 = e2.x;
            if (f2 != StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD || e2.y != StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                this.f3240a.preTranslate(f2, e2.y);
            }
        }
        a<Float, Float> aVar2 = this.i;
        if (aVar2 != null) {
            if (aVar2 instanceof p) {
                j = aVar2.e().floatValue();
            } else {
                j = ((c) aVar2).j();
            }
            if (j != StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                this.f3240a.preRotate(j);
            }
        }
        if (this.k != null) {
            c cVar = this.l;
            float cos = cVar == null ? 0.0f : (float) Math.cos(Math.toRadians((-cVar.j()) + 90.0f));
            c cVar2 = this.l;
            float sin = cVar2 == null ? 1.0f : (float) Math.sin(Math.toRadians((-cVar2.j()) + 90.0f));
            d();
            float[] fArr = this.f3244e;
            fArr[0] = cos;
            fArr[1] = sin;
            float f3 = -sin;
            fArr[3] = f3;
            fArr[4] = cos;
            fArr[8] = 1.0f;
            this.f3241b.setValues(fArr);
            d();
            float[] fArr2 = this.f3244e;
            fArr2[0] = 1.0f;
            fArr2[3] = (float) Math.tan(Math.toRadians(this.k.j()));
            fArr2[4] = 1.0f;
            fArr2[8] = 1.0f;
            this.f3242c.setValues(fArr2);
            d();
            float[] fArr3 = this.f3244e;
            fArr3[0] = cos;
            fArr3[1] = f3;
            fArr3[3] = sin;
            fArr3[4] = cos;
            fArr3[8] = 1.0f;
            this.f3243d.setValues(fArr3);
            this.f3242c.preConcat(this.f3241b);
            this.f3243d.preConcat(this.f3242c);
            this.f3240a.preConcat(this.f3243d);
        }
        a<c.a.a.d0.d, c.a.a.d0.d> aVar3 = this.f3247h;
        if (aVar3 != null) {
            c.a.a.d0.d e3 = aVar3.e();
            float f4 = e3.f3057a;
            if (f4 != 1.0f || e3.f3058b != 1.0f) {
                this.f3240a.preScale(f4, e3.f3058b);
            }
        }
        a<PointF, PointF> aVar4 = this.f3245f;
        if (aVar4 != null) {
            PointF e4 = aVar4.e();
            float f5 = e4.x;
            if (f5 != StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD || e4.y != StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                this.f3240a.preTranslate(-f5, -e4.y);
            }
        }
        return this.f3240a;
    }

    public Matrix f(float f2) {
        a<?, PointF> aVar = this.f3246g;
        PointF e2 = aVar == null ? null : aVar.e();
        a<c.a.a.d0.d, c.a.a.d0.d> aVar2 = this.f3247h;
        c.a.a.d0.d e3 = aVar2 == null ? null : aVar2.e();
        this.f3240a.reset();
        if (e2 != null) {
            this.f3240a.preTranslate(e2.x * f2, e2.y * f2);
        }
        if (e3 != null) {
            double d2 = f2;
            this.f3240a.preScale((float) Math.pow(e3.f3057a, d2), (float) Math.pow(e3.f3058b, d2));
        }
        a<Float, Float> aVar3 = this.i;
        if (aVar3 != null) {
            float floatValue = aVar3.e().floatValue();
            a<PointF, PointF> aVar4 = this.f3245f;
            PointF e4 = aVar4 != null ? aVar4.e() : null;
            Matrix matrix = this.f3240a;
            float f3 = floatValue * f2;
            float f4 = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
            float f5 = e4 == null ? 0.0f : e4.x;
            if (e4 != null) {
                f4 = e4.y;
            }
            matrix.preRotate(f3, f5, f4);
        }
        return this.f3240a;
    }
}