package b.h.b.i.l;

import b.h.b.i.l.f;

/* compiled from: DimensionDependency.java */
/* loaded from: classes.dex */
public class g extends f {
    public int m;

    public g(o oVar) {
        super(oVar);
        if (oVar instanceof k) {
            this.f1907e = f.a.HORIZONTAL_DIMENSION;
        } else {
            this.f1907e = f.a.VERTICAL_DIMENSION;
        }
    }

    @Override // b.h.b.i.l.f
    public void c(int i) {
        if (this.j) {
            return;
        }
        this.j = true;
        this.f1909g = i;
        for (d dVar : this.k) {
            dVar.a(dVar);
        }
    }
}