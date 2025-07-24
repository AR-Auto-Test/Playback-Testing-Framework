package c.a.a.z.l;

import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.PathMeasure;
import android.graphics.RectF;
import b.d.b.m0;
import c.a.a.j;
import c.a.a.o;
import c.a.a.x.c.p;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import java.util.ArrayList;
import java.util.List;

/* compiled from: CompositionLayer.java */
/* loaded from: classes.dex */
public class c extends b {
    public final RectF A;
    public Paint B;
    public c.a.a.x.c.a<Float, Float> x;
    public final List<b> y;
    public final RectF z;

    public c(j jVar, e eVar, List<e> list, c.a.a.d dVar) {
        super(jVar, eVar);
        int i;
        b bVar;
        b cVar;
        this.y = new ArrayList();
        this.z = new RectF();
        this.A = new RectF();
        this.B = new Paint();
        c.a.a.z.j.b bVar2 = eVar.s;
        if (bVar2 != null) {
            c.a.a.x.c.a<Float, Float> a2 = bVar2.a();
            this.x = a2;
            e(a2);
            this.x.f3223a.add(this);
        } else {
            this.x = null;
        }
        b.f.e eVar2 = new b.f.e(dVar.i.size());
        int size = list.size() - 1;
        b bVar3 = null;
        while (true) {
            if (size < 0) {
                break;
            }
            e eVar3 = list.get(size);
            int ordinal = eVar3.f3399e.ordinal();
            if (ordinal == 0) {
                cVar = new c(jVar, eVar3, dVar.f3039c.get(eVar3.f3401g), dVar);
            } else if (ordinal == 1) {
                cVar = new h(jVar, eVar3);
            } else if (ordinal == 2) {
                cVar = new d(jVar, eVar3);
            } else if (ordinal == 3) {
                cVar = new f(jVar, eVar3);
            } else if (ordinal == 4) {
                cVar = new g(jVar, eVar3);
            } else if (ordinal != 5) {
                StringBuilder x = c.b.a.a.a.x("Unknown layer type ");
                x.append(eVar3.f3399e);
                c.a.a.c0.c.b(x.toString());
                cVar = null;
            } else {
                cVar = new i(jVar, eVar3);
            }
            if (cVar != null) {
                eVar2.g(cVar.o.f3398d, cVar);
                if (bVar3 != null) {
                    bVar3.r = cVar;
                    bVar3 = null;
                } else {
                    this.y.add(0, cVar);
                    int f2 = m0.f(eVar3.u);
                    if (f2 == 1 || f2 == 2) {
                        bVar3 = cVar;
                    }
                }
            }
            size--;
        }
        for (i = 0; i < eVar2.h(); i++) {
            b bVar4 = (b) eVar2.d(eVar2.f(i));
            if (bVar4 != null && (bVar = (b) eVar2.d(bVar4.o.f3400f)) != null) {
                bVar4.s = bVar;
            }
        }
    }

    @Override // c.a.a.z.l.b, c.a.a.x.b.e
    public void d(RectF rectF, Matrix matrix, boolean z) {
        super.d(rectF, matrix, z);
        for (int size = this.y.size() - 1; size >= 0; size--) {
            this.z.set(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
            this.y.get(size).d(this.z, this.m, true);
            rectF.union(this.z);
        }
    }

    @Override // c.a.a.z.l.b, c.a.a.z.f
    public <T> void h(T t, c.a.a.d0.c<T> cVar) {
        this.v.c(t, cVar);
        if (t == o.A) {
            if (cVar == null) {
                c.a.a.x.c.a<Float, Float> aVar = this.x;
                if (aVar != null) {
                    aVar.i(null);
                    return;
                }
                return;
            }
            p pVar = new p(cVar, null);
            this.x = pVar;
            pVar.f3223a.add(this);
            e(this.x);
        }
    }

    @Override // c.a.a.z.l.b
    public void k(Canvas canvas, Matrix matrix, int i) {
        RectF rectF = this.A;
        e eVar = this.o;
        rectF.set(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, eVar.o, eVar.p);
        matrix.mapRect(this.A);
        boolean z = this.n.s && this.y.size() > 1 && i != 255;
        if (z) {
            this.B.setAlpha(i);
            RectF rectF2 = this.A;
            Paint paint = this.B;
            PathMeasure pathMeasure = c.a.a.c0.g.f3031a;
            canvas.saveLayer(rectF2, paint);
            c.a.a.c.a("Utils#saveLayer");
        } else {
            canvas.save();
        }
        if (z) {
            i = 255;
        }
        for (int size = this.y.size() - 1; size >= 0; size--) {
            if (!this.A.isEmpty() ? canvas.clipRect(this.A) : true) {
                this.y.get(size).f(canvas, matrix, i);
            }
        }
        canvas.restore();
        c.a.a.c.a("CompositionLayer#draw");
    }

    @Override // c.a.a.z.l.b
    public void o(c.a.a.z.e eVar, int i, List<c.a.a.z.e> list, c.a.a.z.e eVar2) {
        for (int i2 = 0; i2 < this.y.size(); i2++) {
            this.y.get(i2).c(eVar, i, list, eVar2);
        }
    }

    @Override // c.a.a.z.l.b
    public void p(float f2) {
        super.p(f2);
        if (this.x != null) {
            f2 = ((this.x.e().floatValue() * this.o.f3396b.m) - this.o.f3396b.k) / (this.n.f3075c.c() + 0.01f);
        }
        if (this.x == null) {
            e eVar = this.o;
            f2 -= eVar.n / eVar.f3396b.c();
        }
        float f3 = this.o.m;
        if (f3 != StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
            f2 /= f3;
        }
        int size = this.y.size();
        while (true) {
            size--;
            if (size < 0) {
                return;
            }
            this.y.get(size).p(f2);
        }
    }
}