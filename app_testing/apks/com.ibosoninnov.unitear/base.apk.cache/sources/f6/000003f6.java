package b.h.b.i;

import java.util.ArrayList;

/* compiled from: WidgetContainer.java */
/* loaded from: classes.dex */
public class k extends d {
    public ArrayList<d> l0 = new ArrayList<>();

    @Override // b.h.b.i.d
    public void B() {
        this.l0.clear();
        super.B();
    }

    @Override // b.h.b.i.d
    public void D(b.h.b.c cVar) {
        super.D(cVar);
        int size = this.l0.size();
        for (int i = 0; i < size; i++) {
            this.l0.get(i).D(cVar);
        }
    }

    public void P() {
        ArrayList<d> arrayList = this.l0;
        if (arrayList == null) {
            return;
        }
        int size = arrayList.size();
        for (int i = 0; i < size; i++) {
            d dVar = this.l0.get(i);
            if (dVar instanceof k) {
                ((k) dVar).P();
            }
        }
    }
}