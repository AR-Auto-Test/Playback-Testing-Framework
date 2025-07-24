package c.a.a.x.b;

import android.graphics.Path;
import android.graphics.PointF;
import c.a.a.x.c.a;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import java.util.List;

/* compiled from: EllipseContent.java */
/* loaded from: classes.dex */
public class f implements m, a.b, k {

    /* renamed from: b  reason: collision with root package name */
    public final String f3159b;

    /* renamed from: c  reason: collision with root package name */
    public final c.a.a.j f3160c;

    /* renamed from: d  reason: collision with root package name */
    public final c.a.a.x.c.a<?, PointF> f3161d;

    /* renamed from: e  reason: collision with root package name */
    public final c.a.a.x.c.a<?, PointF> f3162e;

    /* renamed from: f  reason: collision with root package name */
    public final c.a.a.z.k.a f3163f;

    /* renamed from: h  reason: collision with root package name */
    public boolean f3165h;

    /* renamed from: a  reason: collision with root package name */
    public final Path f3158a = new Path();

    /* renamed from: g  reason: collision with root package name */
    public b f3164g = new b();

    public f(c.a.a.j jVar, c.a.a.z.l.b bVar, c.a.a.z.k.a aVar) {
        this.f3159b = aVar.f3302a;
        this.f3160c = jVar;
        c.a.a.x.c.a<PointF, PointF> a2 = aVar.f3304c.a();
        this.f3161d = a2;
        c.a.a.x.c.a<PointF, PointF> a3 = aVar.f3303b.a();
        this.f3162e = a3;
        this.f3163f = aVar;
        bVar.e(a2);
        bVar.e(a3);
        a2.f3223a.add(this);
        a3.f3223a.add(this);
    }

    @Override // c.a.a.x.c.a.b
    public void a() {
        this.f3165h = false;
        this.f3160c.invalidateSelf();
    }

    @Override // c.a.a.x.b.c
    public void b(List<c> list, List<c> list2) {
        for (int i = 0; i < list.size(); i++) {
            c cVar = list.get(i);
            if (cVar instanceof s) {
                s sVar = (s) cVar;
                if (sVar.f3219c == 1) {
                    this.f3164g.f3149a.add(sVar);
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
        if (this.f3165h) {
            return this.f3158a;
        }
        this.f3158a.reset();
        if (this.f3163f.f3306e) {
            this.f3165h = true;
            return this.f3158a;
        }
        PointF e2 = this.f3161d.e();
        float f2 = e2.x / 2.0f;
        float f3 = e2.y / 2.0f;
        float f4 = f2 * 0.55228f;
        float f5 = 0.55228f * f3;
        this.f3158a.reset();
        if (this.f3163f.f3305d) {
            float f6 = -f3;
            this.f3158a.moveTo(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, f6);
            Path path = this.f3158a;
            float f7 = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD - f4;
            float f8 = -f2;
            float f9 = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD - f5;
            path.cubicTo(f7, f6, f8, f9, f8, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
            Path path2 = this.f3158a;
            float f10 = f5 + StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
            path2.cubicTo(f8, f10, f7, f3, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, f3);
            Path path3 = this.f3158a;
            float f11 = f4 + StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
            path3.cubicTo(f11, f3, f2, f10, f2, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
            this.f3158a.cubicTo(f2, f9, f11, f6, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, f6);
        } else {
            float f12 = -f3;
            this.f3158a.moveTo(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, f12);
            Path path4 = this.f3158a;
            float f13 = f4 + StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
            float f14 = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD - f5;
            path4.cubicTo(f13, f12, f2, f14, f2, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
            Path path5 = this.f3158a;
            float f15 = f5 + StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
            path5.cubicTo(f2, f15, f13, f3, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, f3);
            Path path6 = this.f3158a;
            float f16 = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD - f4;
            float f17 = -f2;
            path6.cubicTo(f16, f3, f17, f15, f17, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
            this.f3158a.cubicTo(f17, f14, f16, f12, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, f12);
        }
        PointF e3 = this.f3162e.e();
        this.f3158a.offset(e3.x, e3.y);
        this.f3158a.close();
        this.f3164g.a(this.f3158a);
        this.f3165h = true;
        return this.f3158a;
    }

    @Override // c.a.a.x.b.c
    public String getName() {
        return this.f3159b;
    }

    /* JADX DEBUG: Multi-variable search result rejected for r3v0, resolved type: c.a.a.d0.c<T> */
    /* JADX WARN: Multi-variable type inference failed */
    @Override // c.a.a.z.f
    public <T> void h(T t, c.a.a.d0.c<T> cVar) {
        if (t == c.a.a.o.f3120g) {
            c.a.a.x.c.a<?, PointF> aVar = this.f3161d;
            c.a.a.d0.c<PointF> cVar2 = aVar.f3227e;
            aVar.f3227e = cVar;
        } else if (t == c.a.a.o.j) {
            c.a.a.x.c.a<?, PointF> aVar2 = this.f3162e;
            c.a.a.d0.c<PointF> cVar3 = aVar2.f3227e;
            aVar2.f3227e = cVar;
        }
    }
}