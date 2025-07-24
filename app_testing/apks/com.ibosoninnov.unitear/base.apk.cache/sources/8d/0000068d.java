package c.a.a.x.b;

import android.graphics.Path;
import android.graphics.PointF;
import android.graphics.RectF;
import c.a.a.x.c.a;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import java.util.List;

/* compiled from: RectangleContent.java */
/* loaded from: classes.dex */
public class o implements a.b, k, m {

    /* renamed from: c  reason: collision with root package name */
    public final String f3197c;

    /* renamed from: d  reason: collision with root package name */
    public final boolean f3198d;

    /* renamed from: e  reason: collision with root package name */
    public final c.a.a.j f3199e;

    /* renamed from: f  reason: collision with root package name */
    public final c.a.a.x.c.a<?, PointF> f3200f;

    /* renamed from: g  reason: collision with root package name */
    public final c.a.a.x.c.a<?, PointF> f3201g;

    /* renamed from: h  reason: collision with root package name */
    public final c.a.a.x.c.a<?, Float> f3202h;
    public boolean j;

    /* renamed from: a  reason: collision with root package name */
    public final Path f3195a = new Path();

    /* renamed from: b  reason: collision with root package name */
    public final RectF f3196b = new RectF();
    public b i = new b();

    public o(c.a.a.j jVar, c.a.a.z.l.b bVar, c.a.a.z.k.i iVar) {
        this.f3197c = iVar.f3346a;
        this.f3198d = iVar.f3350e;
        this.f3199e = jVar;
        c.a.a.x.c.a<PointF, PointF> a2 = iVar.f3347b.a();
        this.f3200f = a2;
        c.a.a.x.c.a<PointF, PointF> a3 = iVar.f3348c.a();
        this.f3201g = a3;
        c.a.a.x.c.a<Float, Float> a4 = iVar.f3349d.a();
        this.f3202h = a4;
        bVar.e(a2);
        bVar.e(a3);
        bVar.e(a4);
        a2.f3223a.add(this);
        a3.f3223a.add(this);
        a4.f3223a.add(this);
    }

    @Override // c.a.a.x.c.a.b
    public void a() {
        this.j = false;
        this.f3199e.invalidateSelf();
    }

    @Override // c.a.a.x.b.c
    public void b(List<c> list, List<c> list2) {
        for (int i = 0; i < list.size(); i++) {
            c cVar = list.get(i);
            if (cVar instanceof s) {
                s sVar = (s) cVar;
                if (sVar.f3219c == 1) {
                    this.i.f3149a.add(sVar);
                    sVar.f3218b.add(this);
                }
            }
        }
    }

    @Override // c.a.a.z.f
    public void c(c.a.a.z.e eVar, int i, List<c.a.a.z.e> list, c.a.a.z.e eVar2) {
        c.a.a.c0.f.f(eVar, i, list, eVar2, this);
    }

    @Override // c.a.a.x.b.m
    public Path g() {
        if (this.j) {
            return this.f3195a;
        }
        this.f3195a.reset();
        if (this.f3198d) {
            this.j = true;
            return this.f3195a;
        }
        PointF e2 = this.f3201g.e();
        float f2 = e2.x / 2.0f;
        float f3 = e2.y / 2.0f;
        c.a.a.x.c.a<?, Float> aVar = this.f3202h;
        float j = aVar == null ? 0.0f : ((c.a.a.x.c.c) aVar).j();
        float min = Math.min(f2, f3);
        if (j > min) {
            j = min;
        }
        PointF e3 = this.f3200f.e();
        this.f3195a.moveTo(e3.x + f2, (e3.y - f3) + j);
        this.f3195a.lineTo(e3.x + f2, (e3.y + f3) - j);
        int i = (j > StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD ? 1 : (j == StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD ? 0 : -1));
        if (i > 0) {
            RectF rectF = this.f3196b;
            float f4 = e3.x;
            float f5 = j * 2.0f;
            float f6 = e3.y;
            rectF.set((f4 + f2) - f5, (f6 + f3) - f5, f4 + f2, f6 + f3);
            this.f3195a.arcTo(this.f3196b, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 90.0f, false);
        }
        this.f3195a.lineTo((e3.x - f2) + j, e3.y + f3);
        if (i > 0) {
            RectF rectF2 = this.f3196b;
            float f7 = e3.x;
            float f8 = e3.y;
            float f9 = j * 2.0f;
            rectF2.set(f7 - f2, (f8 + f3) - f9, (f7 - f2) + f9, f8 + f3);
            this.f3195a.arcTo(this.f3196b, 90.0f, 90.0f, false);
        }
        this.f3195a.lineTo(e3.x - f2, (e3.y - f3) + j);
        if (i > 0) {
            RectF rectF3 = this.f3196b;
            float f10 = e3.x;
            float f11 = e3.y;
            float f12 = j * 2.0f;
            rectF3.set(f10 - f2, f11 - f3, (f10 - f2) + f12, (f11 - f3) + f12);
            this.f3195a.arcTo(this.f3196b, 180.0f, 90.0f, false);
        }
        this.f3195a.lineTo((e3.x + f2) - j, e3.y - f3);
        if (i > 0) {
            RectF rectF4 = this.f3196b;
            float f13 = e3.x;
            float f14 = j * 2.0f;
            float f15 = e3.y;
            rectF4.set((f13 + f2) - f14, f15 - f3, f13 + f2, (f15 - f3) + f14);
            this.f3195a.arcTo(this.f3196b, 270.0f, 90.0f, false);
        }
        this.f3195a.close();
        this.i.a(this.f3195a);
        this.j = true;
        return this.f3195a;
    }

    @Override // c.a.a.x.b.c
    public String getName() {
        return this.f3197c;
    }

    /* JADX DEBUG: Multi-variable search result rejected for r3v0, resolved type: c.a.a.d0.c<T> */
    /* JADX WARN: Multi-variable type inference failed */
    @Override // c.a.a.z.f
    public <T> void h(T t, c.a.a.d0.c<T> cVar) {
        if (t == c.a.a.o.f3121h) {
            c.a.a.x.c.a<?, PointF> aVar = this.f3201g;
            c.a.a.d0.c<PointF> cVar2 = aVar.f3227e;
            aVar.f3227e = cVar;
        } else if (t == c.a.a.o.j) {
            c.a.a.x.c.a<?, PointF> aVar2 = this.f3200f;
            c.a.a.d0.c<PointF> cVar3 = aVar2.f3227e;
            aVar2.f3227e = cVar;
        } else if (t == c.a.a.o.i) {
            c.a.a.x.c.a<?, Float> aVar3 = this.f3202h;
            c.a.a.d0.c<Float> cVar4 = aVar3.f3227e;
            aVar3.f3227e = cVar;
        }
    }
}