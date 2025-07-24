package androidx.lifecycle;

import b.t.d;
import b.t.e;
import b.t.f;
import b.t.h;
import b.t.l;

/* loaded from: classes.dex */
public class CompositeGeneratedAdaptersObserver implements f {

    /* renamed from: a  reason: collision with root package name */
    public final d[] f308a;

    public CompositeGeneratedAdaptersObserver(d[] dVarArr) {
        this.f308a = dVarArr;
    }

    @Override // b.t.f
    public void e(h hVar, e.a aVar) {
        l lVar = new l();
        for (d dVar : this.f308a) {
            dVar.a(hVar, aVar, false, lVar);
        }
        for (d dVar2 : this.f308a) {
            dVar2.a(hVar, aVar, true, lVar);
        }
    }
}