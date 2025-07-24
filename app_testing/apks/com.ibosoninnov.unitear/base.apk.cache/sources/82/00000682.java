package c.a.a.x.b;

import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.PathMeasure;
import android.graphics.RectF;
import c.a.a.x.c.a;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import java.util.ArrayList;
import java.util.List;

/* compiled from: ContentGroup.java */
/* loaded from: classes.dex */
public class d implements e, m, a.b, c.a.a.z.f {

    /* renamed from: a  reason: collision with root package name */
    public Paint f3150a;

    /* renamed from: b  reason: collision with root package name */
    public RectF f3151b;

    /* renamed from: c  reason: collision with root package name */
    public final Matrix f3152c;

    /* renamed from: d  reason: collision with root package name */
    public final Path f3153d;

    /* renamed from: e  reason: collision with root package name */
    public final RectF f3154e;

    /* renamed from: f  reason: collision with root package name */
    public final String f3155f;

    /* renamed from: g  reason: collision with root package name */
    public final boolean f3156g;

    /* renamed from: h  reason: collision with root package name */
    public final List<c> f3157h;
    public final c.a.a.j i;
    public List<m> j;
    public c.a.a.x.c.o k;

    /* JADX WARN: Illegal instructions before constructor call */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public d(c.a.a.j jVar, c.a.a.z.l.b bVar, c.a.a.z.k.m mVar) {
        this(jVar, bVar, r3, r4, r5, r6);
        c.a.a.z.j.l lVar;
        String str = mVar.f3365a;
        boolean z = mVar.f3367c;
        List<c.a.a.z.k.b> list = mVar.f3366b;
        ArrayList arrayList = new ArrayList(list.size());
        int i = 0;
        for (int i2 = 0; i2 < list.size(); i2++) {
            c a2 = list.get(i2).a(jVar, bVar);
            if (a2 != null) {
                arrayList.add(a2);
            }
        }
        List<c.a.a.z.k.b> list2 = mVar.f3366b;
        while (true) {
            if (i >= list2.size()) {
                lVar = null;
                break;
            }
            c.a.a.z.k.b bVar2 = list2.get(i);
            if (bVar2 instanceof c.a.a.z.j.l) {
                lVar = (c.a.a.z.j.l) bVar2;
                break;
            }
            i++;
        }
    }

    @Override // c.a.a.x.c.a.b
    public void a() {
        this.i.invalidateSelf();
    }

    @Override // c.a.a.x.b.c
    public void b(List<c> list, List<c> list2) {
        ArrayList arrayList = new ArrayList(this.f3157h.size() + list.size());
        arrayList.addAll(list);
        for (int size = this.f3157h.size() - 1; size >= 0; size--) {
            c cVar = this.f3157h.get(size);
            cVar.b(arrayList, this.f3157h.subList(0, size));
            arrayList.add(cVar);
        }
    }

    @Override // c.a.a.z.f
    public void c(c.a.a.z.e eVar, int i, List<c.a.a.z.e> list, c.a.a.z.e eVar2) {
        if (eVar.e(this.f3155f, i)) {
            if (!"__container".equals(this.f3155f)) {
                eVar2 = eVar2.a(this.f3155f);
                if (eVar.c(this.f3155f, i)) {
                    list.add(eVar2.g(this));
                }
            }
            if (eVar.f(this.f3155f, i)) {
                int d2 = eVar.d(this.f3155f, i) + i;
                for (int i2 = 0; i2 < this.f3157h.size(); i2++) {
                    c cVar = this.f3157h.get(i2);
                    if (cVar instanceof c.a.a.z.f) {
                        ((c.a.a.z.f) cVar).c(eVar, d2, list, eVar2);
                    }
                }
            }
        }
    }

    @Override // c.a.a.x.b.e
    public void d(RectF rectF, Matrix matrix, boolean z) {
        this.f3152c.set(matrix);
        c.a.a.x.c.o oVar = this.k;
        if (oVar != null) {
            this.f3152c.preConcat(oVar.e());
        }
        this.f3154e.set(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
        for (int size = this.f3157h.size() - 1; size >= 0; size--) {
            c cVar = this.f3157h.get(size);
            if (cVar instanceof e) {
                ((e) cVar).d(this.f3154e, this.f3152c, z);
                rectF.union(this.f3154e);
            }
        }
    }

    public List<m> e() {
        if (this.j == null) {
            this.j = new ArrayList();
            for (int i = 0; i < this.f3157h.size(); i++) {
                c cVar = this.f3157h.get(i);
                if (cVar instanceof m) {
                    this.j.add((m) cVar);
                }
            }
        }
        return this.j;
    }

    @Override // c.a.a.x.b.e
    public void f(Canvas canvas, Matrix matrix, int i) {
        boolean z;
        c.a.a.x.c.a<Integer, Integer> aVar;
        if (this.f3156g) {
            return;
        }
        this.f3152c.set(matrix);
        c.a.a.x.c.o oVar = this.k;
        if (oVar != null) {
            this.f3152c.preConcat(oVar.e());
            i = (int) (((((this.k.j == null ? 100 : aVar.e().intValue()) / 100.0f) * i) / 255.0f) * 255.0f);
        }
        boolean z2 = false;
        if (this.i.s) {
            int i2 = 0;
            int i3 = 0;
            while (true) {
                if (i2 >= this.f3157h.size()) {
                    z = false;
                    break;
                } else if ((this.f3157h.get(i2) instanceof e) && (i3 = i3 + 1) >= 2) {
                    z = true;
                    break;
                } else {
                    i2++;
                }
            }
            if (z && i != 255) {
                z2 = true;
            }
        }
        if (z2) {
            this.f3151b.set(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
            d(this.f3151b, this.f3152c, true);
            this.f3150a.setAlpha(i);
            RectF rectF = this.f3151b;
            Paint paint = this.f3150a;
            PathMeasure pathMeasure = c.a.a.c0.g.f3031a;
            canvas.saveLayer(rectF, paint);
            c.a.a.c.a("Utils#saveLayer");
        }
        if (z2) {
            i = 255;
        }
        for (int size = this.f3157h.size() - 1; size >= 0; size--) {
            c cVar = this.f3157h.get(size);
            if (cVar instanceof e) {
                ((e) cVar).f(canvas, this.f3152c, i);
            }
        }
        if (z2) {
            canvas.restore();
        }
    }

    @Override // c.a.a.x.b.m
    public Path g() {
        this.f3152c.reset();
        c.a.a.x.c.o oVar = this.k;
        if (oVar != null) {
            this.f3152c.set(oVar.e());
        }
        this.f3153d.reset();
        if (this.f3156g) {
            return this.f3153d;
        }
        for (int size = this.f3157h.size() - 1; size >= 0; size--) {
            c cVar = this.f3157h.get(size);
            if (cVar instanceof m) {
                this.f3153d.addPath(((m) cVar).g(), this.f3152c);
            }
        }
        return this.f3153d;
    }

    @Override // c.a.a.x.b.c
    public String getName() {
        return this.f3155f;
    }

    @Override // c.a.a.z.f
    public <T> void h(T t, c.a.a.d0.c<T> cVar) {
        c.a.a.x.c.o oVar = this.k;
        if (oVar != null) {
            oVar.c(t, cVar);
        }
    }

    public d(c.a.a.j jVar, c.a.a.z.l.b bVar, String str, boolean z, List<c> list, c.a.a.z.j.l lVar) {
        this.f3150a = new c.a.a.x.a();
        this.f3151b = new RectF();
        this.f3152c = new Matrix();
        this.f3153d = new Path();
        this.f3154e = new RectF();
        this.f3155f = str;
        this.i = jVar;
        this.f3156g = z;
        this.f3157h = list;
        if (lVar != null) {
            c.a.a.x.c.o oVar = new c.a.a.x.c.o(lVar);
            this.k = oVar;
            oVar.a(bVar);
            this.k.b(this);
        }
        ArrayList arrayList = new ArrayList();
        int size = list.size();
        while (true) {
            size--;
            if (size < 0) {
                break;
            }
            c cVar = list.get(size);
            if (cVar instanceof j) {
                arrayList.add((j) cVar);
            }
        }
        int size2 = arrayList.size();
        while (true) {
            size2--;
            if (size2 < 0) {
                return;
            }
            ((j) arrayList.get(size2)).e(list.listIterator(list.size()));
        }
    }
}