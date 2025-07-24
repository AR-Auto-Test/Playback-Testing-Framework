package b.h.b.i;

import b.h.b.i.l.n;
import java.util.ArrayList;

/* compiled from: HelperWidget.java */
/* loaded from: classes.dex */
public class h extends d implements g {
    public d[] l0 = new d[4];
    public int m0 = 0;

    public void P(ArrayList<n> arrayList, int i, n nVar) {
        for (int i2 = 0; i2 < this.m0; i2++) {
            nVar.a(this.l0[i2]);
        }
        for (int i3 = 0; i3 < this.m0; i3++) {
            b.e.a.b(this.l0[i3], i, arrayList, nVar);
        }
    }

    @Override // b.h.b.i.g
    public void a(e eVar) {
    }
}