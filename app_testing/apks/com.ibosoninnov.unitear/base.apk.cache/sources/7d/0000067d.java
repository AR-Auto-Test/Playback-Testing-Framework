package c.a.a.x.b;

import android.graphics.Canvas;
import android.graphics.ColorFilter;
import android.graphics.DashPathEffect;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.PathMeasure;
import android.graphics.RectF;
import c.a.a.x.c.a;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import java.util.ArrayList;
import java.util.List;

/* compiled from: BaseStrokeContent.java */
/* loaded from: classes.dex */
public abstract class a implements a.b, k, e {

    /* renamed from: e  reason: collision with root package name */
    public final c.a.a.j f3143e;

    /* renamed from: f  reason: collision with root package name */
    public final c.a.a.z.l.b f3144f;

    /* renamed from: h  reason: collision with root package name */
    public final float[] f3146h;
    public final Paint i;
    public final c.a.a.x.c.a<?, Float> j;
    public final c.a.a.x.c.a<?, Integer> k;
    public final List<c.a.a.x.c.a<?, Float>> l;
    public final c.a.a.x.c.a<?, Float> m;
    public c.a.a.x.c.a<ColorFilter, ColorFilter> n;

    /* renamed from: a  reason: collision with root package name */
    public final PathMeasure f3139a = new PathMeasure();

    /* renamed from: b  reason: collision with root package name */
    public final Path f3140b = new Path();

    /* renamed from: c  reason: collision with root package name */
    public final Path f3141c = new Path();

    /* renamed from: d  reason: collision with root package name */
    public final RectF f3142d = new RectF();

    /* renamed from: g  reason: collision with root package name */
    public final List<b> f3145g = new ArrayList();

    /* compiled from: BaseStrokeContent.java */
    /* loaded from: classes.dex */
    public static final class b {

        /* renamed from: a  reason: collision with root package name */
        public final List<m> f3147a = new ArrayList();

        /* renamed from: b  reason: collision with root package name */
        public final s f3148b;

        public b(s sVar, C0060a c0060a) {
            this.f3148b = sVar;
        }
    }

    public a(c.a.a.j jVar, c.a.a.z.l.b bVar, Paint.Cap cap, Paint.Join join, float f2, c.a.a.z.j.d dVar, c.a.a.z.j.b bVar2, List<c.a.a.z.j.b> list, c.a.a.z.j.b bVar3) {
        c.a.a.x.a aVar = new c.a.a.x.a(1);
        this.i = aVar;
        this.f3143e = jVar;
        this.f3144f = bVar;
        aVar.setStyle(Paint.Style.STROKE);
        aVar.setStrokeCap(cap);
        aVar.setStrokeJoin(join);
        aVar.setStrokeMiter(f2);
        this.k = dVar.a();
        this.j = bVar2.a();
        if (bVar3 == null) {
            this.m = null;
        } else {
            this.m = bVar3.a();
        }
        this.l = new ArrayList(list.size());
        this.f3146h = new float[list.size()];
        for (int i = 0; i < list.size(); i++) {
            this.l.add(list.get(i).a());
        }
        bVar.e(this.k);
        bVar.e(this.j);
        for (int i2 = 0; i2 < this.l.size(); i2++) {
            bVar.e(this.l.get(i2));
        }
        c.a.a.x.c.a<?, Float> aVar2 = this.m;
        if (aVar2 != null) {
            bVar.e(aVar2);
        }
        this.k.f3223a.add(this);
        this.j.f3223a.add(this);
        for (int i3 = 0; i3 < list.size(); i3++) {
            this.l.get(i3).f3223a.add(this);
        }
        c.a.a.x.c.a<?, Float> aVar3 = this.m;
        if (aVar3 != null) {
            aVar3.f3223a.add(this);
        }
    }

    @Override // c.a.a.x.c.a.b
    public void a() {
        this.f3143e.invalidateSelf();
    }

    @Override // c.a.a.x.b.c
    public void b(List<c> list, List<c> list2) {
        s sVar = null;
        for (int size = list.size() - 1; size >= 0; size--) {
            c cVar = list.get(size);
            if (cVar instanceof s) {
                s sVar2 = (s) cVar;
                if (sVar2.f3219c == 2) {
                    sVar = sVar2;
                }
            }
        }
        if (sVar != null) {
            sVar.f3218b.add(this);
        }
        b bVar = null;
        for (int size2 = list2.size() - 1; size2 >= 0; size2--) {
            c cVar2 = list2.get(size2);
            if (cVar2 instanceof s) {
                s sVar3 = (s) cVar2;
                if (sVar3.f3219c == 2) {
                    if (bVar != null) {
                        this.f3145g.add(bVar);
                    }
                    bVar = new b(sVar3, null);
                    sVar3.f3218b.add(this);
                }
            }
            if (cVar2 instanceof m) {
                if (bVar == null) {
                    bVar = new b(sVar, null);
                }
                bVar.f3147a.add((m) cVar2);
            }
        }
        if (bVar != null) {
            this.f3145g.add(bVar);
        }
    }

    @Override // c.a.a.z.f
    public void c(c.a.a.z.e eVar, int i, List<c.a.a.z.e> list, c.a.a.z.e eVar2) {
        c.a.a.c0.f.f(eVar, i, list, eVar2, this);
    }

    @Override // c.a.a.x.b.e
    public void d(RectF rectF, Matrix matrix, boolean z) {
        this.f3140b.reset();
        for (int i = 0; i < this.f3145g.size(); i++) {
            b bVar = this.f3145g.get(i);
            for (int i2 = 0; i2 < bVar.f3147a.size(); i2++) {
                this.f3140b.addPath(bVar.f3147a.get(i2).g(), matrix);
            }
        }
        this.f3140b.computeBounds(this.f3142d, false);
        float j = ((c.a.a.x.c.c) this.j).j();
        RectF rectF2 = this.f3142d;
        float f2 = j / 2.0f;
        rectF2.set(rectF2.left - f2, rectF2.top - f2, rectF2.right + f2, rectF2.bottom + f2);
        rectF.set(this.f3142d);
        rectF.set(rectF.left - 1.0f, rectF.top - 1.0f, rectF.right + 1.0f, rectF.bottom + 1.0f);
        c.a.a.c.a("StrokeContent#getBounds");
    }

    @Override // c.a.a.x.b.e
    public void f(Canvas canvas, Matrix matrix, int i) {
        float[] fArr = c.a.a.c0.g.f3034d;
        boolean z = false;
        fArr[0] = 0.0f;
        fArr[1] = 0.0f;
        fArr[2] = 37394.73f;
        fArr[3] = 39575.234f;
        matrix.mapPoints(fArr);
        if (fArr[0] == fArr[2] || fArr[1] == fArr[3]) {
            c.a.a.c.a("StrokeContent#draw");
            return;
        }
        c.a.a.x.c.e eVar = (c.a.a.x.c.e) this.k;
        float j = (i / 255.0f) * eVar.j(eVar.a(), eVar.c());
        float f2 = 100.0f;
        this.i.setAlpha(c.a.a.c0.f.c((int) ((j / 100.0f) * 255.0f), 0, 255));
        this.i.setStrokeWidth(c.a.a.c0.g.d(matrix) * ((c.a.a.x.c.c) this.j).j());
        if (this.i.getStrokeWidth() <= StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
            c.a.a.c.a("StrokeContent#draw");
            return;
        }
        float f3 = 1.0f;
        if (this.l.isEmpty()) {
            c.a.a.c.a("StrokeContent#applyDashPattern");
        } else {
            float d2 = c.a.a.c0.g.d(matrix);
            for (int i2 = 0; i2 < this.l.size(); i2++) {
                this.f3146h[i2] = this.l.get(i2).e().floatValue();
                if (i2 % 2 == 0) {
                    float[] fArr2 = this.f3146h;
                    if (fArr2[i2] < 1.0f) {
                        fArr2[i2] = 1.0f;
                    }
                } else {
                    float[] fArr3 = this.f3146h;
                    if (fArr3[i2] < 0.1f) {
                        fArr3[i2] = 0.1f;
                    }
                }
                float[] fArr4 = this.f3146h;
                fArr4[i2] = fArr4[i2] * d2;
            }
            c.a.a.x.c.a<?, Float> aVar = this.m;
            this.i.setPathEffect(new DashPathEffect(this.f3146h, aVar == null ? 0.0f : aVar.e().floatValue() * d2));
            c.a.a.c.a("StrokeContent#applyDashPattern");
        }
        c.a.a.x.c.a<ColorFilter, ColorFilter> aVar2 = this.n;
        if (aVar2 != null) {
            this.i.setColorFilter(aVar2.e());
        }
        int i3 = 0;
        while (i3 < this.f3145g.size()) {
            b bVar = this.f3145g.get(i3);
            s sVar = bVar.f3148b;
            if (sVar == null) {
                this.f3140b.reset();
                for (int size = bVar.f3147a.size() - 1; size >= 0; size--) {
                    this.f3140b.addPath(bVar.f3147a.get(size).g(), matrix);
                }
                c.a.a.c.a("StrokeContent#buildPath");
                canvas.drawPath(this.f3140b, this.i);
                c.a.a.c.a("StrokeContent#drawPath");
            } else if (sVar == null) {
                c.a.a.c.a("StrokeContent#applyTrimPath");
            } else {
                this.f3140b.reset();
                int size2 = bVar.f3147a.size();
                while (true) {
                    size2--;
                    if (size2 < 0) {
                        break;
                    }
                    this.f3140b.addPath(bVar.f3147a.get(size2).g(), matrix);
                }
                this.f3139a.setPath(this.f3140b, z);
                float length = this.f3139a.getLength();
                while (this.f3139a.nextContour()) {
                    length += this.f3139a.getLength();
                }
                float floatValue = (bVar.f3148b.f3222f.e().floatValue() * length) / 360.0f;
                float floatValue2 = ((bVar.f3148b.f3220d.e().floatValue() * length) / f2) + floatValue;
                float floatValue3 = ((bVar.f3148b.f3221e.e().floatValue() * length) / f2) + floatValue;
                int size3 = bVar.f3147a.size() - 1;
                float f4 = 0.0f;
                while (size3 >= 0) {
                    this.f3141c.set(bVar.f3147a.get(size3).g());
                    this.f3141c.transform(matrix);
                    this.f3139a.setPath(this.f3141c, z);
                    float length2 = this.f3139a.getLength();
                    if (floatValue3 > length) {
                        float f5 = floatValue3 - length;
                        if (f5 < f4 + length2 && f4 < f5) {
                            c.a.a.c0.g.a(this.f3141c, floatValue2 > length ? (floatValue2 - length) / length2 : 0.0f, Math.min(f5 / length2, f3), StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
                            canvas.drawPath(this.f3141c, this.i);
                            f4 += length2;
                            size3--;
                            z = false;
                            f3 = 1.0f;
                        }
                    }
                    float f6 = f4 + length2;
                    if (f6 >= floatValue2 && f4 <= floatValue3) {
                        if (f6 <= floatValue3 && floatValue2 < f4) {
                            canvas.drawPath(this.f3141c, this.i);
                        } else {
                            c.a.a.c0.g.a(this.f3141c, floatValue2 < f4 ? 0.0f : (floatValue2 - f4) / length2, floatValue3 > f6 ? 1.0f : (floatValue3 - f4) / length2, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
                            canvas.drawPath(this.f3141c, this.i);
                        }
                    }
                    f4 += length2;
                    size3--;
                    z = false;
                    f3 = 1.0f;
                }
                c.a.a.c.a("StrokeContent#applyTrimPath");
            }
            i3++;
            z = false;
            f2 = 100.0f;
            f3 = 1.0f;
        }
        c.a.a.c.a("StrokeContent#draw");
    }

    /* JADX DEBUG: Multi-variable search result rejected for r3v0, resolved type: c.a.a.d0.c<T> */
    /* JADX WARN: Multi-variable type inference failed */
    @Override // c.a.a.z.f
    public <T> void h(T t, c.a.a.d0.c<T> cVar) {
        if (t == c.a.a.o.f3117d) {
            c.a.a.x.c.a<?, Integer> aVar = this.k;
            c.a.a.d0.c<Integer> cVar2 = aVar.f3227e;
            aVar.f3227e = cVar;
        } else if (t == c.a.a.o.o) {
            c.a.a.x.c.a<?, Float> aVar2 = this.j;
            c.a.a.d0.c<Float> cVar3 = aVar2.f3227e;
            aVar2.f3227e = cVar;
        } else if (t == c.a.a.o.C) {
            c.a.a.x.c.a<ColorFilter, ColorFilter> aVar3 = this.n;
            if (aVar3 != null) {
                this.f3144f.u.remove(aVar3);
            }
            if (cVar == 0) {
                this.n = null;
                return;
            }
            c.a.a.x.c.p pVar = new c.a.a.x.c.p(cVar, null);
            this.n = pVar;
            pVar.f3223a.add(this);
            this.f3144f.e(this.n);
        }
    }
}