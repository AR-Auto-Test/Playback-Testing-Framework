package c.a.a.x.b;

import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.Path;
import android.graphics.RectF;
import c.a.a.x.c.a;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.ListIterator;
import java.util.Objects;

/* compiled from: RepeaterContent.java */
/* loaded from: classes.dex */
public class p implements e, m, j, a.b, k {

    /* renamed from: a  reason: collision with root package name */
    public final Matrix f3203a = new Matrix();

    /* renamed from: b  reason: collision with root package name */
    public final Path f3204b = new Path();

    /* renamed from: c  reason: collision with root package name */
    public final c.a.a.j f3205c;

    /* renamed from: d  reason: collision with root package name */
    public final c.a.a.z.l.b f3206d;

    /* renamed from: e  reason: collision with root package name */
    public final String f3207e;

    /* renamed from: f  reason: collision with root package name */
    public final boolean f3208f;

    /* renamed from: g  reason: collision with root package name */
    public final c.a.a.x.c.a<Float, Float> f3209g;

    /* renamed from: h  reason: collision with root package name */
    public final c.a.a.x.c.a<Float, Float> f3210h;
    public final c.a.a.x.c.o i;
    public d j;

    public p(c.a.a.j jVar, c.a.a.z.l.b bVar, c.a.a.z.k.j jVar2) {
        this.f3205c = jVar;
        this.f3206d = bVar;
        this.f3207e = jVar2.f3351a;
        this.f3208f = jVar2.f3355e;
        c.a.a.x.c.a<Float, Float> a2 = jVar2.f3352b.a();
        this.f3209g = a2;
        bVar.e(a2);
        a2.f3223a.add(this);
        c.a.a.x.c.a<Float, Float> a3 = jVar2.f3353c.a();
        this.f3210h = a3;
        bVar.e(a3);
        a3.f3223a.add(this);
        c.a.a.z.j.l lVar = jVar2.f3354d;
        Objects.requireNonNull(lVar);
        c.a.a.x.c.o oVar = new c.a.a.x.c.o(lVar);
        this.i = oVar;
        oVar.a(bVar);
        oVar.b(this);
    }

    @Override // c.a.a.x.c.a.b
    public void a() {
        this.f3205c.invalidateSelf();
    }

    @Override // c.a.a.x.b.c
    public void b(List<c> list, List<c> list2) {
        this.j.b(list, list2);
    }

    @Override // c.a.a.z.f
    public void c(c.a.a.z.e eVar, int i, List<c.a.a.z.e> list, c.a.a.z.e eVar2) {
        c.a.a.c0.f.f(eVar, i, list, eVar2, this);
    }

    @Override // c.a.a.x.b.e
    public void d(RectF rectF, Matrix matrix, boolean z) {
        this.j.d(rectF, matrix, z);
    }

    @Override // c.a.a.x.b.j
    public void e(ListIterator<c> listIterator) {
        if (this.j != null) {
            return;
        }
        while (listIterator.hasPrevious() && listIterator.previous() != this) {
        }
        ArrayList arrayList = new ArrayList();
        while (listIterator.hasPrevious()) {
            arrayList.add(listIterator.previous());
            listIterator.remove();
        }
        Collections.reverse(arrayList);
        this.j = new d(this.f3205c, this.f3206d, "Repeater", this.f3208f, arrayList, null);
    }

    @Override // c.a.a.x.b.e
    public void f(Canvas canvas, Matrix matrix, int i) {
        float floatValue = this.f3209g.e().floatValue();
        float floatValue2 = this.f3210h.e().floatValue();
        float floatValue3 = this.i.m.e().floatValue() / 100.0f;
        float floatValue4 = this.i.n.e().floatValue() / 100.0f;
        for (int i2 = ((int) floatValue) - 1; i2 >= 0; i2--) {
            this.f3203a.set(matrix);
            float f2 = i2;
            this.f3203a.preConcat(this.i.f(f2 + floatValue2));
            this.j.f(canvas, this.f3203a, (int) (c.a.a.c0.f.e(floatValue3, floatValue4, f2 / floatValue) * i));
        }
    }

    @Override // c.a.a.x.b.m
    public Path g() {
        Path g2 = this.j.g();
        this.f3204b.reset();
        float floatValue = this.f3209g.e().floatValue();
        float floatValue2 = this.f3210h.e().floatValue();
        for (int i = ((int) floatValue) - 1; i >= 0; i--) {
            this.f3203a.set(this.i.f(i + floatValue2));
            this.f3204b.addPath(g2, this.f3203a);
        }
        return this.f3204b;
    }

    @Override // c.a.a.x.b.c
    public String getName() {
        return this.f3207e;
    }

    /* JADX DEBUG: Multi-variable search result rejected for r3v0, resolved type: c.a.a.d0.c<T> */
    /* JADX WARN: Multi-variable type inference failed */
    @Override // c.a.a.z.f
    public <T> void h(T t, c.a.a.d0.c<T> cVar) {
        if (this.i.c(t, cVar)) {
            return;
        }
        if (t == c.a.a.o.q) {
            c.a.a.x.c.a<Float, Float> aVar = this.f3209g;
            c.a.a.d0.c<Float> cVar2 = aVar.f3227e;
            aVar.f3227e = cVar;
        } else if (t == c.a.a.o.r) {
            c.a.a.x.c.a<Float, Float> aVar2 = this.f3210h;
            c.a.a.d0.c<Float> cVar3 = aVar2.f3227e;
            aVar2.f3227e = cVar;
        }
    }
}