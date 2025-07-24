package c.c.a.m.v;

import c.c.a.m.u.d;
import c.c.a.m.v.g;
import c.c.a.m.w.n;

/* compiled from: SourceGenerator.java */
/* loaded from: classes.dex */
public class a0 implements d.a<Object> {

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ n.a f3596b;

    /* renamed from: c  reason: collision with root package name */
    public final /* synthetic */ b0 f3597c;

    public a0(b0 b0Var, n.a aVar) {
        this.f3597c = b0Var;
        this.f3596b = aVar;
    }

    @Override // c.c.a.m.u.d.a
    public void c(Exception exc) {
        b0 b0Var = this.f3597c;
        n.a<?> aVar = this.f3596b;
        n.a<?> aVar2 = b0Var.f3604g;
        if (aVar2 != null && aVar2 == aVar) {
            b0 b0Var2 = this.f3597c;
            n.a aVar3 = this.f3596b;
            g.a aVar4 = b0Var2.f3600c;
            c.c.a.m.m mVar = b0Var2.f3605h;
            c.c.a.m.u.d<Data> dVar = aVar3.f3865c;
            aVar4.a(mVar, exc, dVar, dVar.d());
        }
    }

    @Override // c.c.a.m.u.d.a
    public void f(Object obj) {
        b0 b0Var = this.f3597c;
        n.a<?> aVar = this.f3596b;
        n.a<?> aVar2 = b0Var.f3604g;
        if (aVar2 != null && aVar2 == aVar) {
            b0 b0Var2 = this.f3597c;
            n.a aVar3 = this.f3596b;
            k kVar = b0Var2.f3599b.p;
            if (obj != null && kVar.c(aVar3.f3865c.d())) {
                b0Var2.f3603f = obj;
                b0Var2.f3600c.c();
                return;
            }
            g.a aVar4 = b0Var2.f3600c;
            c.c.a.m.m mVar = aVar3.f3863a;
            c.c.a.m.u.d<Data> dVar = aVar3.f3865c;
            aVar4.d(mVar, obj, dVar, dVar.d(), b0Var2.f3605h);
        }
    }
}