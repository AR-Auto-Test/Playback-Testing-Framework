package c.a.a.x.b;

import android.graphics.Canvas;
import android.graphics.ColorFilter;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.RectF;
import c.a.a.x.c.a;
import java.util.ArrayList;
import java.util.List;

/* compiled from: FillContent.java */
/* loaded from: classes.dex */
public class g implements e, a.b, k {

    /* renamed from: a  reason: collision with root package name */
    public final Path f3166a;

    /* renamed from: b  reason: collision with root package name */
    public final Paint f3167b;

    /* renamed from: c  reason: collision with root package name */
    public final c.a.a.z.l.b f3168c;

    /* renamed from: d  reason: collision with root package name */
    public final String f3169d;

    /* renamed from: e  reason: collision with root package name */
    public final boolean f3170e;

    /* renamed from: f  reason: collision with root package name */
    public final List<m> f3171f;

    /* renamed from: g  reason: collision with root package name */
    public final c.a.a.x.c.a<Integer, Integer> f3172g;

    /* renamed from: h  reason: collision with root package name */
    public final c.a.a.x.c.a<Integer, Integer> f3173h;
    public c.a.a.x.c.a<ColorFilter, ColorFilter> i;
    public final c.a.a.j j;

    public g(c.a.a.j jVar, c.a.a.z.l.b bVar, c.a.a.z.k.l lVar) {
        Path path = new Path();
        this.f3166a = path;
        this.f3167b = new c.a.a.x.a(1);
        this.f3171f = new ArrayList();
        this.f3168c = bVar;
        this.f3169d = lVar.f3361c;
        this.f3170e = lVar.f3364f;
        this.j = jVar;
        if (lVar.f3362d != null && lVar.f3363e != null) {
            path.setFillType(lVar.f3360b);
            c.a.a.x.c.a<Integer, Integer> a2 = lVar.f3362d.a();
            this.f3172g = a2;
            a2.f3223a.add(this);
            bVar.e(a2);
            c.a.a.x.c.a<Integer, Integer> a3 = lVar.f3363e.a();
            this.f3173h = a3;
            a3.f3223a.add(this);
            bVar.e(a3);
            return;
        }
        this.f3172g = null;
        this.f3173h = null;
    }

    @Override // c.a.a.x.c.a.b
    public void a() {
        this.j.invalidateSelf();
    }

    @Override // c.a.a.x.b.c
    public void b(List<c> list, List<c> list2) {
        for (int i = 0; i < list2.size(); i++) {
            c cVar = list2.get(i);
            if (cVar instanceof m) {
                this.f3171f.add((m) cVar);
            }
        }
    }

    @Override // c.a.a.z.f
    public void c(c.a.a.z.e eVar, int i, List<c.a.a.z.e> list, c.a.a.z.e eVar2) {
        c.a.a.c0.f.f(eVar, i, list, eVar2, this);
    }

    @Override // c.a.a.x.b.e
    public void d(RectF rectF, Matrix matrix, boolean z) {
        this.f3166a.reset();
        for (int i = 0; i < this.f3171f.size(); i++) {
            this.f3166a.addPath(this.f3171f.get(i).g(), matrix);
        }
        this.f3166a.computeBounds(rectF, false);
        rectF.set(rectF.left - 1.0f, rectF.top - 1.0f, rectF.right + 1.0f, rectF.bottom + 1.0f);
    }

    @Override // c.a.a.x.b.e
    public void f(Canvas canvas, Matrix matrix, int i) {
        if (this.f3170e) {
            return;
        }
        Paint paint = this.f3167b;
        c.a.a.x.c.b bVar = (c.a.a.x.c.b) this.f3172g;
        paint.setColor(bVar.j(bVar.a(), bVar.c()));
        this.f3167b.setAlpha(c.a.a.c0.f.c((int) ((((i / 255.0f) * this.f3173h.e().intValue()) / 100.0f) * 255.0f), 0, 255));
        c.a.a.x.c.a<ColorFilter, ColorFilter> aVar = this.i;
        if (aVar != null) {
            this.f3167b.setColorFilter(aVar.e());
        }
        this.f3166a.reset();
        for (int i2 = 0; i2 < this.f3171f.size(); i2++) {
            this.f3166a.addPath(this.f3171f.get(i2).g(), matrix);
        }
        canvas.drawPath(this.f3166a, this.f3167b);
        c.a.a.c.a("FillContent#draw");
    }

    @Override // c.a.a.x.b.c
    public String getName() {
        return this.f3169d;
    }

    /* JADX DEBUG: Multi-variable search result rejected for r3v0, resolved type: c.a.a.d0.c<T> */
    /* JADX WARN: Multi-variable type inference failed */
    @Override // c.a.a.z.f
    public <T> void h(T t, c.a.a.d0.c<T> cVar) {
        if (t == c.a.a.o.f3114a) {
            c.a.a.x.c.a<Integer, Integer> aVar = this.f3172g;
            c.a.a.d0.c<Integer> cVar2 = aVar.f3227e;
            aVar.f3227e = cVar;
        } else if (t == c.a.a.o.f3117d) {
            c.a.a.x.c.a<Integer, Integer> aVar2 = this.f3173h;
            c.a.a.d0.c<Integer> cVar3 = aVar2.f3227e;
            aVar2.f3227e = cVar;
        } else if (t == c.a.a.o.C) {
            c.a.a.x.c.a<ColorFilter, ColorFilter> aVar3 = this.i;
            if (aVar3 != null) {
                this.f3168c.u.remove(aVar3);
            }
            if (cVar == 0) {
                this.i = null;
                return;
            }
            c.a.a.x.c.p pVar = new c.a.a.x.c.p(cVar, null);
            this.i = pVar;
            pVar.f3223a.add(this);
            this.f3168c.e(this.i);
        }
    }
}