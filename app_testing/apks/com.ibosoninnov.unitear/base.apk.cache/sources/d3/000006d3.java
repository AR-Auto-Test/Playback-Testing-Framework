package c.a.a.z.l;

import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.PathMeasure;
import android.graphics.PointF;
import android.graphics.PorterDuff;
import android.graphics.PorterDuffXfermode;
import android.graphics.RectF;
import android.os.Build;
import b.d.b.m0;
import c.a.a.j;
import c.a.a.s;
import c.a.a.x.c.a;
import c.a.a.x.c.o;
import c.a.a.z.j.l;
import c.a.a.z.k.k;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/* compiled from: BaseLayer.java */
/* loaded from: classes.dex */
public abstract class b implements c.a.a.x.b.e, a.b, c.a.a.z.f {

    /* renamed from: a  reason: collision with root package name */
    public final Path f3387a = new Path();

    /* renamed from: b  reason: collision with root package name */
    public final Matrix f3388b = new Matrix();

    /* renamed from: c  reason: collision with root package name */
    public final Paint f3389c = new c.a.a.x.a(1);

    /* renamed from: d  reason: collision with root package name */
    public final Paint f3390d = new c.a.a.x.a(1, PorterDuff.Mode.DST_IN);

    /* renamed from: e  reason: collision with root package name */
    public final Paint f3391e = new c.a.a.x.a(1, PorterDuff.Mode.DST_OUT);

    /* renamed from: f  reason: collision with root package name */
    public final Paint f3392f;

    /* renamed from: g  reason: collision with root package name */
    public final Paint f3393g;

    /* renamed from: h  reason: collision with root package name */
    public final RectF f3394h;
    public final RectF i;
    public final RectF j;
    public final RectF k;
    public final String l;
    public final Matrix m;
    public final j n;
    public final e o;
    public c.a.a.x.c.g p;
    public c.a.a.x.c.c q;
    public b r;
    public b s;
    public List<b> t;
    public final List<c.a.a.x.c.a<?, ?>> u;
    public final o v;
    public boolean w;

    public b(j jVar, e eVar) {
        c.a.a.x.a aVar = new c.a.a.x.a(1);
        this.f3392f = aVar;
        this.f3393g = new c.a.a.x.a(PorterDuff.Mode.CLEAR);
        this.f3394h = new RectF();
        this.i = new RectF();
        this.j = new RectF();
        this.k = new RectF();
        this.m = new Matrix();
        this.u = new ArrayList();
        this.w = true;
        this.n = jVar;
        this.o = eVar;
        this.l = c.b.a.a.a.v(new StringBuilder(), eVar.f3397c, "#draw");
        if (eVar.u == 3) {
            aVar.setXfermode(new PorterDuffXfermode(PorterDuff.Mode.DST_OUT));
        } else {
            aVar.setXfermode(new PorterDuffXfermode(PorterDuff.Mode.DST_IN));
        }
        l lVar = eVar.i;
        Objects.requireNonNull(lVar);
        o oVar = new o(lVar);
        this.v = oVar;
        oVar.b(this);
        List<c.a.a.z.k.f> list = eVar.f3402h;
        if (list != null && !list.isEmpty()) {
            c.a.a.x.c.g gVar = new c.a.a.x.c.g(eVar.f3402h);
            this.p = gVar;
            for (c.a.a.x.c.a<k, Path> aVar2 : gVar.f3237a) {
                aVar2.f3223a.add(this);
            }
            for (c.a.a.x.c.a<Integer, Integer> aVar3 : this.p.f3238b) {
                e(aVar3);
                aVar3.f3223a.add(this);
            }
        }
        if (!this.o.t.isEmpty()) {
            c.a.a.x.c.c cVar = new c.a.a.x.c.c(this.o.t);
            this.q = cVar;
            cVar.f3224b = true;
            cVar.f3223a.add(new a(this));
            q(this.q.e().floatValue() == 1.0f);
            e(this.q);
            return;
        }
        q(true);
    }

    @Override // c.a.a.x.c.a.b
    public void a() {
        this.n.invalidateSelf();
    }

    @Override // c.a.a.x.b.c
    public void b(List<c.a.a.x.b.c> list, List<c.a.a.x.b.c> list2) {
    }

    @Override // c.a.a.z.f
    public void c(c.a.a.z.e eVar, int i, List<c.a.a.z.e> list, c.a.a.z.e eVar2) {
        if (eVar.e(this.o.f3397c, i)) {
            if (!"__container".equals(this.o.f3397c)) {
                eVar2 = eVar2.a(this.o.f3397c);
                if (eVar.c(this.o.f3397c, i)) {
                    list.add(eVar2.g(this));
                }
            }
            if (eVar.f(this.o.f3397c, i)) {
                o(eVar, eVar.d(this.o.f3397c, i) + i, list, eVar2);
            }
        }
    }

    @Override // c.a.a.x.b.e
    public void d(RectF rectF, Matrix matrix, boolean z) {
        this.f3394h.set(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
        i();
        this.m.set(matrix);
        if (z) {
            List<b> list = this.t;
            if (list != null) {
                for (int size = list.size() - 1; size >= 0; size--) {
                    this.m.preConcat(this.t.get(size).v.e());
                }
            } else {
                b bVar = this.s;
                if (bVar != null) {
                    this.m.preConcat(bVar.v.e());
                }
            }
        }
        this.m.preConcat(this.v.e());
    }

    public void e(c.a.a.x.c.a<?, ?> aVar) {
        if (aVar == null) {
            return;
        }
        this.u.add(aVar);
    }

    /* JADX WARN: Removed duplicated region for block: B:120:0x03bd A[SYNTHETIC] */
    /* JADX WARN: Removed duplicated region for block: B:45:0x012a  */
    /* JADX WARN: Removed duplicated region for block: B:46:0x0132  */
    /* JADX WARN: Removed duplicated region for block: B:85:0x024f  */
    @Override // c.a.a.x.b.e
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public void f(Canvas canvas, Matrix matrix, int i) {
        c.a.a.x.c.a<Integer, Integer> aVar;
        boolean z;
        String str = this.l;
        if (this.w && !this.o.v) {
            i();
            this.f3388b.reset();
            this.f3388b.set(matrix);
            int i2 = 1;
            for (int size = this.t.size() - 1; size >= 0; size--) {
                this.f3388b.preConcat(this.t.get(size).v.e());
            }
            c.a.a.c.a("Layer#parentMatrix");
            int intValue = (int) ((((i / 255.0f) * (this.v.j == null ? 100 : aVar.e().intValue())) / 100.0f) * 255.0f);
            boolean m = m();
            float f2 = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
            if (!m && !l()) {
                this.f3388b.preConcat(this.v.e());
                k(canvas, this.f3388b, intValue);
                c.a.a.c.a("Layer#drawLayer");
                c.a.a.c.a(this.l);
                n(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
                return;
            }
            boolean z2 = false;
            d(this.f3394h, this.f3388b, false);
            RectF rectF = this.f3394h;
            int i3 = 3;
            if (m() && this.o.u != 3) {
                this.j.set(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
                this.r.d(this.j, matrix, true);
                if (!rectF.intersect(this.j)) {
                    rectF.set(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
                }
            }
            this.f3388b.preConcat(this.v.e());
            RectF rectF2 = this.f3394h;
            Matrix matrix2 = this.f3388b;
            this.i.set(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
            int i4 = 2;
            if (l()) {
                int size2 = this.p.f3239c.size();
                int i5 = 0;
                while (true) {
                    if (i5 < size2) {
                        c.a.a.z.k.f fVar = this.p.f3239c.get(i5);
                        this.f3387a.set(this.p.f3237a.get(i5).e());
                        this.f3387a.transform(matrix2);
                        int f3 = m0.f(fVar.f3325a);
                        if (f3 != 0) {
                            if (f3 == 1) {
                                break;
                            } else if (f3 != i4) {
                                if (f3 == i3) {
                                    break;
                                }
                                this.f3387a.computeBounds(this.k, z2);
                                if (i5 != 0) {
                                    this.i.set(this.k);
                                } else {
                                    RectF rectF3 = this.i;
                                    rectF3.set(Math.min(rectF3.left, this.k.left), Math.min(this.i.top, this.k.top), Math.max(this.i.right, this.k.right), Math.max(this.i.bottom, this.k.bottom));
                                }
                                i5++;
                                z2 = false;
                                i3 = 3;
                                i4 = 2;
                            }
                        }
                        if (fVar.f3328d) {
                            break;
                        }
                        this.f3387a.computeBounds(this.k, z2);
                        if (i5 != 0) {
                        }
                        i5++;
                        z2 = false;
                        i3 = 3;
                        i4 = 2;
                    } else if (!rectF2.intersect(this.i)) {
                        f2 = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
                        rectF2.set(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
                    }
                }
                f2 = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
            }
            if (!this.f3394h.intersect(f2, f2, canvas.getWidth(), canvas.getHeight())) {
                this.f3394h.set(f2, f2, f2, f2);
            }
            c.a.a.c.a("Layer#computeBounds");
            if (!this.f3394h.isEmpty()) {
                this.f3389c.setAlpha(255);
                RectF rectF4 = this.f3394h;
                Paint paint = this.f3389c;
                PathMeasure pathMeasure = c.a.a.c0.g.f3031a;
                canvas.saveLayer(rectF4, paint);
                c.a.a.c.a("Utils#saveLayer");
                c.a.a.c.a("Layer#saveLayer");
                j(canvas);
                k(canvas, this.f3388b, intValue);
                c.a.a.c.a("Layer#drawLayer");
                if (l()) {
                    Matrix matrix3 = this.f3388b;
                    canvas.saveLayer(this.f3394h, this.f3390d);
                    c.a.a.c.a("Utils#saveLayer");
                    if (Build.VERSION.SDK_INT < 28) {
                        j(canvas);
                    }
                    c.a.a.c.a("Layer#saveLayer");
                    int i6 = 0;
                    while (i6 < this.p.f3239c.size()) {
                        c.a.a.z.k.f fVar2 = this.p.f3239c.get(i6);
                        c.a.a.x.c.a<k, Path> aVar2 = this.p.f3237a.get(i6);
                        c.a.a.x.c.a<Integer, Integer> aVar3 = this.p.f3238b.get(i6);
                        int f4 = m0.f(fVar2.f3325a);
                        if (f4 != 0) {
                            if (f4 == i2) {
                                if (i6 == 0) {
                                    this.f3389c.setColor(-16777216);
                                    this.f3389c.setAlpha(255);
                                    canvas.drawRect(this.f3394h, this.f3389c);
                                }
                                if (fVar2.f3328d) {
                                    RectF rectF5 = this.f3394h;
                                    Paint paint2 = this.f3391e;
                                    PathMeasure pathMeasure2 = c.a.a.c0.g.f3031a;
                                    canvas.saveLayer(rectF5, paint2);
                                    c.a.a.c.a("Utils#saveLayer");
                                    canvas.drawRect(this.f3394h, this.f3389c);
                                    this.f3391e.setAlpha((int) (aVar3.e().intValue() * 2.55f));
                                    this.f3387a.set(aVar2.e());
                                    this.f3387a.transform(matrix3);
                                    canvas.drawPath(this.f3387a, this.f3391e);
                                    canvas.restore();
                                } else {
                                    this.f3387a.set(aVar2.e());
                                    this.f3387a.transform(matrix3);
                                    canvas.drawPath(this.f3387a, this.f3391e);
                                }
                            } else if (f4 != 2) {
                                if (f4 == 3) {
                                    if (!this.p.f3237a.isEmpty()) {
                                        for (int i7 = 0; i7 < this.p.f3239c.size(); i7++) {
                                            if (this.p.f3239c.get(i7).f3325a == 4) {
                                            }
                                        }
                                        z = true;
                                        if (!z) {
                                            this.f3389c.setAlpha(255);
                                            canvas.drawRect(this.f3394h, this.f3389c);
                                        }
                                    }
                                    z = false;
                                    if (!z) {
                                    }
                                }
                            } else if (fVar2.f3328d) {
                                RectF rectF6 = this.f3394h;
                                Paint paint3 = this.f3390d;
                                PathMeasure pathMeasure3 = c.a.a.c0.g.f3031a;
                                canvas.saveLayer(rectF6, paint3);
                                c.a.a.c.a("Utils#saveLayer");
                                canvas.drawRect(this.f3394h, this.f3389c);
                                this.f3391e.setAlpha((int) (aVar3.e().intValue() * 2.55f));
                                this.f3387a.set(aVar2.e());
                                this.f3387a.transform(matrix3);
                                canvas.drawPath(this.f3387a, this.f3391e);
                                canvas.restore();
                            } else {
                                RectF rectF7 = this.f3394h;
                                Paint paint4 = this.f3390d;
                                PathMeasure pathMeasure4 = c.a.a.c0.g.f3031a;
                                canvas.saveLayer(rectF7, paint4);
                                c.a.a.c.a("Utils#saveLayer");
                                this.f3387a.set(aVar2.e());
                                this.f3387a.transform(matrix3);
                                this.f3389c.setAlpha((int) (aVar3.e().intValue() * 2.55f));
                                canvas.drawPath(this.f3387a, this.f3389c);
                                canvas.restore();
                            }
                        } else if (fVar2.f3328d) {
                            RectF rectF8 = this.f3394h;
                            Paint paint5 = this.f3389c;
                            PathMeasure pathMeasure5 = c.a.a.c0.g.f3031a;
                            canvas.saveLayer(rectF8, paint5);
                            c.a.a.c.a("Utils#saveLayer");
                            canvas.drawRect(this.f3394h, this.f3389c);
                            this.f3387a.set(aVar2.e());
                            this.f3387a.transform(matrix3);
                            this.f3389c.setAlpha((int) (aVar3.e().intValue() * 2.55f));
                            canvas.drawPath(this.f3387a, this.f3391e);
                            canvas.restore();
                        } else {
                            this.f3387a.set(aVar2.e());
                            this.f3387a.transform(matrix3);
                            this.f3389c.setAlpha((int) (aVar3.e().intValue() * 2.55f));
                            canvas.drawPath(this.f3387a, this.f3389c);
                        }
                        i6++;
                        i2 = 1;
                    }
                    canvas.restore();
                    c.a.a.c.a("Layer#restoreLayer");
                }
                if (m()) {
                    canvas.saveLayer(this.f3394h, this.f3392f);
                    c.a.a.c.a("Utils#saveLayer");
                    c.a.a.c.a("Layer#saveLayer");
                    j(canvas);
                    this.r.f(canvas, matrix, intValue);
                    canvas.restore();
                    c.a.a.c.a("Layer#restoreLayer");
                    c.a.a.c.a("Layer#drawMatte");
                }
                canvas.restore();
                c.a.a.c.a("Layer#restoreLayer");
            }
            c.a.a.c.a(this.l);
            n(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
            return;
        }
        c.a.a.c.a(str);
    }

    @Override // c.a.a.x.b.c
    public String getName() {
        return this.o.f3397c;
    }

    @Override // c.a.a.z.f
    public <T> void h(T t, c.a.a.d0.c<T> cVar) {
        this.v.c(t, cVar);
    }

    public final void i() {
        if (this.t != null) {
            return;
        }
        if (this.s == null) {
            this.t = Collections.emptyList();
            return;
        }
        this.t = new ArrayList();
        for (b bVar = this.s; bVar != null; bVar = bVar.s) {
            this.t.add(bVar);
        }
    }

    public final void j(Canvas canvas) {
        RectF rectF = this.f3394h;
        canvas.drawRect(rectF.left - 1.0f, rectF.top - 1.0f, rectF.right + 1.0f, rectF.bottom + 1.0f, this.f3393g);
        c.a.a.c.a("Layer#clearLayer");
    }

    public abstract void k(Canvas canvas, Matrix matrix, int i);

    public boolean l() {
        c.a.a.x.c.g gVar = this.p;
        return (gVar == null || gVar.f3237a.isEmpty()) ? false : true;
    }

    public boolean m() {
        return this.r != null;
    }

    public final void n(float f2) {
        s sVar = this.n.f3075c.f3037a;
        String str = this.o.f3397c;
        if (sVar.f3131a) {
            c.a.a.c0.e eVar = sVar.f3133c.get(str);
            if (eVar == null) {
                eVar = new c.a.a.c0.e();
                sVar.f3133c.put(str, eVar);
            }
            float f3 = eVar.f3028a + f2;
            eVar.f3028a = f3;
            int i = eVar.f3029b + 1;
            eVar.f3029b = i;
            if (i == Integer.MAX_VALUE) {
                eVar.f3028a = f3 / 2.0f;
                eVar.f3029b = i / 2;
            }
            if (str.equals("__container")) {
                for (s.a aVar : sVar.f3132b) {
                    aVar.a(f2);
                }
            }
        }
    }

    public void o(c.a.a.z.e eVar, int i, List<c.a.a.z.e> list, c.a.a.z.e eVar2) {
    }

    public void p(float f2) {
        o oVar = this.v;
        c.a.a.x.c.a<Integer, Integer> aVar = oVar.j;
        if (aVar != null) {
            aVar.h(f2);
        }
        c.a.a.x.c.a<?, Float> aVar2 = oVar.m;
        if (aVar2 != null) {
            aVar2.h(f2);
        }
        c.a.a.x.c.a<?, Float> aVar3 = oVar.n;
        if (aVar3 != null) {
            aVar3.h(f2);
        }
        c.a.a.x.c.a<PointF, PointF> aVar4 = oVar.f3245f;
        if (aVar4 != null) {
            aVar4.h(f2);
        }
        c.a.a.x.c.a<?, PointF> aVar5 = oVar.f3246g;
        if (aVar5 != null) {
            aVar5.h(f2);
        }
        c.a.a.x.c.a<c.a.a.d0.d, c.a.a.d0.d> aVar6 = oVar.f3247h;
        if (aVar6 != null) {
            aVar6.h(f2);
        }
        c.a.a.x.c.a<Float, Float> aVar7 = oVar.i;
        if (aVar7 != null) {
            aVar7.h(f2);
        }
        c.a.a.x.c.c cVar = oVar.k;
        if (cVar != null) {
            cVar.h(f2);
        }
        c.a.a.x.c.c cVar2 = oVar.l;
        if (cVar2 != null) {
            cVar2.h(f2);
        }
        if (this.p != null) {
            for (int i = 0; i < this.p.f3237a.size(); i++) {
                this.p.f3237a.get(i).h(f2);
            }
        }
        float f3 = this.o.m;
        if (f3 != StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
            f2 /= f3;
        }
        c.a.a.x.c.c cVar3 = this.q;
        if (cVar3 != null) {
            cVar3.h(f2 / f3);
        }
        b bVar = this.r;
        if (bVar != null) {
            bVar.p(bVar.o.m * f2);
        }
        for (int i2 = 0; i2 < this.u.size(); i2++) {
            this.u.get(i2).h(f2);
        }
    }

    public final void q(boolean z) {
        if (z != this.w) {
            this.w = z;
            this.n.invalidateSelf();
        }
    }
}