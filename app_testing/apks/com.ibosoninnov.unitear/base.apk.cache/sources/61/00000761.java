package c.c.a.m.v.d0;

import c.c.a.m.m;
import c.c.a.m.v.d0.i;
import c.c.a.m.v.l;
import c.c.a.m.v.w;

/* compiled from: LruResourceCache.java */
/* loaded from: classes.dex */
public class h extends c.c.a.s.g<m, w<?>> implements i {

    /* renamed from: d  reason: collision with root package name */
    public i.a f3663d;

    public h(long j) {
        super(j);
    }

    /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object] */
    @Override // c.c.a.s.g
    public int b(w<?> wVar) {
        w<?> wVar2 = wVar;
        if (wVar2 == null) {
            return 1;
        }
        return wVar2.c();
    }

    /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object, java.lang.Object] */
    @Override // c.c.a.s.g
    public void c(m mVar, w<?> wVar) {
        w<?> wVar2 = wVar;
        i.a aVar = this.f3663d;
        if (aVar == null || wVar2 == null) {
            return;
        }
        ((l) aVar).f3739f.a(wVar2, true);
    }
}