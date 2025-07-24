package b.m;

import b.m.g;

/* compiled from: BaseObservable.java */
/* loaded from: classes.dex */
public class a implements g {

    /* renamed from: b  reason: collision with root package name */
    public transient i f2328b;

    @Override // b.m.g
    public void a(g.a aVar) {
        synchronized (this) {
            if (this.f2328b == null) {
                this.f2328b = new i();
            }
        }
        i iVar = this.f2328b;
        synchronized (iVar) {
            int lastIndexOf = iVar.f2329b.lastIndexOf(aVar);
            if (lastIndexOf < 0 || iVar.a(lastIndexOf)) {
                iVar.f2329b.add(aVar);
            }
        }
    }

    @Override // b.m.g
    public void b(g.a aVar) {
        synchronized (this) {
            i iVar = this.f2328b;
            if (iVar == null) {
                return;
            }
            synchronized (iVar) {
                if (iVar.f2332e == 0) {
                    iVar.f2329b.remove(aVar);
                } else {
                    int lastIndexOf = iVar.f2329b.lastIndexOf(aVar);
                    if (lastIndexOf >= 0) {
                        iVar.f(lastIndexOf);
                    }
                }
            }
        }
    }
}