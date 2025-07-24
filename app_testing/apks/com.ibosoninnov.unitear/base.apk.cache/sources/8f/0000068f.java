package c.a.a.x.b;

import android.graphics.Path;
import c.a.a.x.c.a;
import java.util.List;

/* compiled from: ShapeContent.java */
/* loaded from: classes.dex */
public class q implements m, a.b {

    /* renamed from: b  reason: collision with root package name */
    public final boolean f3212b;

    /* renamed from: c  reason: collision with root package name */
    public final c.a.a.j f3213c;

    /* renamed from: d  reason: collision with root package name */
    public final c.a.a.x.c.a<?, Path> f3214d;

    /* renamed from: e  reason: collision with root package name */
    public boolean f3215e;

    /* renamed from: a  reason: collision with root package name */
    public final Path f3211a = new Path();

    /* renamed from: f  reason: collision with root package name */
    public b f3216f = new b();

    public q(c.a.a.j jVar, c.a.a.z.l.b bVar, c.a.a.z.k.n nVar) {
        this.f3212b = nVar.f3371d;
        this.f3213c = jVar;
        c.a.a.x.c.a<c.a.a.z.k.k, Path> a2 = nVar.f3370c.a();
        this.f3214d = a2;
        bVar.e(a2);
        a2.f3223a.add(this);
    }

    @Override // c.a.a.x.c.a.b
    public void a() {
        this.f3215e = false;
        this.f3213c.invalidateSelf();
    }

    @Override // c.a.a.x.b.c
    public void b(List<c> list, List<c> list2) {
        for (int i = 0; i < list.size(); i++) {
            c cVar = list.get(i);
            if (cVar instanceof s) {
                s sVar = (s) cVar;
                if (sVar.f3219c == 1) {
                    this.f3216f.f3149a.add(sVar);
                    sVar.f3218b.add(this);
                }
            }
        }
    }

    @Override // c.a.a.x.b.m
    public Path g() {
        if (this.f3215e) {
            return this.f3211a;
        }
        this.f3211a.reset();
        if (this.f3212b) {
            this.f3215e = true;
            return this.f3211a;
        }
        this.f3211a.set(this.f3214d.e());
        this.f3211a.setFillType(Path.FillType.EVEN_ODD);
        this.f3216f.a(this.f3211a);
        this.f3215e = true;
        return this.f3211a;
    }
}