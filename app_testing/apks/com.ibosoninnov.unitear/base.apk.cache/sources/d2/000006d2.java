package c.a.a.z.l;

import c.a.a.x.c.a;

/* compiled from: BaseLayer.java */
/* loaded from: classes.dex */
public class a implements a.b {

    /* renamed from: a  reason: collision with root package name */
    public final /* synthetic */ b f3386a;

    public a(b bVar) {
        this.f3386a = bVar;
    }

    @Override // c.a.a.x.c.a.b
    public void a() {
        b bVar = this.f3386a;
        boolean z = bVar.q.j() == 1.0f;
        if (z != bVar.w) {
            bVar.w = z;
            bVar.n.invalidateSelf();
        }
    }
}