package c.a.a.x.b;

import c.a.a.x.c.a;
import java.util.ArrayList;
import java.util.List;

/* compiled from: TrimPathContent.java */
/* loaded from: classes.dex */
public class s implements c, a.b {

    /* renamed from: a  reason: collision with root package name */
    public final boolean f3217a;

    /* renamed from: b  reason: collision with root package name */
    public final List<a.b> f3218b = new ArrayList();

    /* renamed from: c  reason: collision with root package name */
    public final int f3219c;

    /* renamed from: d  reason: collision with root package name */
    public final c.a.a.x.c.a<?, Float> f3220d;

    /* renamed from: e  reason: collision with root package name */
    public final c.a.a.x.c.a<?, Float> f3221e;

    /* renamed from: f  reason: collision with root package name */
    public final c.a.a.x.c.a<?, Float> f3222f;

    public s(c.a.a.z.l.b bVar, c.a.a.z.k.p pVar) {
        this.f3217a = pVar.f3385f;
        this.f3219c = pVar.f3381b;
        c.a.a.x.c.a<Float, Float> a2 = pVar.f3382c.a();
        this.f3220d = a2;
        c.a.a.x.c.a<Float, Float> a3 = pVar.f3383d.a();
        this.f3221e = a3;
        c.a.a.x.c.a<Float, Float> a4 = pVar.f3384e.a();
        this.f3222f = a4;
        bVar.e(a2);
        bVar.e(a3);
        bVar.e(a4);
        a2.f3223a.add(this);
        a3.f3223a.add(this);
        a4.f3223a.add(this);
    }

    @Override // c.a.a.x.c.a.b
    public void a() {
        for (int i = 0; i < this.f3218b.size(); i++) {
            this.f3218b.get(i).a();
        }
    }

    @Override // c.a.a.x.b.c
    public void b(List<c> list, List<c> list2) {
    }
}