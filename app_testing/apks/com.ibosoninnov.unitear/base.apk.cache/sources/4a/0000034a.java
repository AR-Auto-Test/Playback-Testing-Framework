package b.d.b.d1;

import java.util.ArrayList;
import java.util.List;

/* compiled from: LensFacingCameraFilter.java */
/* loaded from: classes.dex */
public class q0 implements b.d.b.h0 {

    /* renamed from: a  reason: collision with root package name */
    public int f1585a;

    public q0(int i) {
        this.f1585a = i;
    }

    @Override // b.d.b.h0
    public List<b.d.b.i0> a(List<b.d.b.i0> list) {
        ArrayList arrayList = new ArrayList();
        for (b.d.b.i0 i0Var : list) {
            b.j.b.d.e(i0Var instanceof z, "The camera info doesn't contain internal implementation.");
            Integer c2 = ((z) i0Var).c();
            if (c2 != null && c2.intValue() == this.f1585a) {
                arrayList.add(i0Var);
            }
        }
        return arrayList;
    }
}