package c.a.a.x.b;

import android.graphics.Canvas;
import android.graphics.ColorFilter;
import android.graphics.Matrix;
import android.graphics.Paint;
import b.d.b.m0;

/* compiled from: StrokeContent.java */
/* loaded from: classes.dex */
public class r extends a {
    public final c.a.a.z.l.b o;
    public final String p;
    public final boolean q;
    public final c.a.a.x.c.a<Integer, Integer> r;
    public c.a.a.x.c.a<ColorFilter, ColorFilter> s;

    public r(c.a.a.j jVar, c.a.a.z.l.b bVar, c.a.a.z.k.o oVar) {
        super(jVar, bVar, m0.h(oVar.f3378g), m0.i(oVar.f3379h), oVar.i, oVar.f3376e, oVar.f3377f, oVar.f3374c, oVar.f3373b);
        this.o = bVar;
        this.p = oVar.f3372a;
        this.q = oVar.j;
        c.a.a.x.c.a<Integer, Integer> a2 = oVar.f3375d.a();
        this.r = a2;
        a2.f3223a.add(this);
        bVar.e(a2);
    }

    @Override // c.a.a.x.b.a, c.a.a.x.b.e
    public void f(Canvas canvas, Matrix matrix, int i) {
        if (this.q) {
            return;
        }
        Paint paint = this.i;
        c.a.a.x.c.b bVar = (c.a.a.x.c.b) this.r;
        paint.setColor(bVar.j(bVar.a(), bVar.c()));
        c.a.a.x.c.a<ColorFilter, ColorFilter> aVar = this.s;
        if (aVar != null) {
            this.i.setColorFilter(aVar.e());
        }
        super.f(canvas, matrix, i);
    }

    @Override // c.a.a.x.b.c
    public String getName() {
        return this.p;
    }

    /* JADX DEBUG: Multi-variable search result rejected for r3v0, resolved type: c.a.a.d0.c<T> */
    /* JADX WARN: Multi-variable type inference failed */
    @Override // c.a.a.x.b.a, c.a.a.z.f
    public <T> void h(T t, c.a.a.d0.c<T> cVar) {
        super.h(t, cVar);
        if (t == c.a.a.o.f3115b) {
            c.a.a.x.c.a<Integer, Integer> aVar = this.r;
            c.a.a.d0.c<Integer> cVar2 = aVar.f3227e;
            aVar.f3227e = cVar;
        } else if (t == c.a.a.o.C) {
            c.a.a.x.c.a<ColorFilter, ColorFilter> aVar2 = this.s;
            if (aVar2 != null) {
                this.o.u.remove(aVar2);
            }
            if (cVar == 0) {
                this.s = null;
                return;
            }
            c.a.a.x.c.p pVar = new c.a.a.x.c.p(cVar, null);
            this.s = pVar;
            pVar.f3223a.add(this);
            this.o.e(this.r);
        }
    }
}