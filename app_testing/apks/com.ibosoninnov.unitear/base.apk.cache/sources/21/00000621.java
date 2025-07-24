package c.a.a.b0;

import android.graphics.PointF;
import c.a.a.b0.h0.c;
import java.util.ArrayList;

/* compiled from: AnimatablePathValueParser.java */
/* loaded from: classes.dex */
public class a {

    /* renamed from: a  reason: collision with root package name */
    public static c.a f2954a = c.a.a("k", "x", "y");

    public static c.a.a.z.j.e a(c.a.a.b0.h0.c cVar, c.a.a.d dVar) {
        ArrayList arrayList = new ArrayList();
        if (cVar.M() == c.b.BEGIN_ARRAY) {
            cVar.B();
            while (cVar.G()) {
                arrayList.add(new c.a.a.x.c.h(dVar, p.a(cVar, dVar, c.a.a.c0.g.c(), u.f3012a, cVar.M() == c.b.BEGIN_OBJECT)));
            }
            cVar.D();
            q.b(arrayList);
        } else {
            arrayList.add(new c.a.a.d0.a(o.b(cVar, c.a.a.c0.g.c())));
        }
        return new c.a.a.z.j.e(arrayList);
    }

    public static c.a.a.z.j.m<PointF, PointF> b(c.a.a.b0.h0.c cVar, c.a.a.d dVar) {
        c.b bVar = c.b.STRING;
        cVar.C();
        c.a.a.z.j.e eVar = null;
        c.a.a.z.j.b bVar2 = null;
        boolean z = false;
        c.a.a.z.j.b bVar3 = null;
        while (cVar.M() != c.b.END_OBJECT) {
            int O = cVar.O(f2954a);
            if (O == 0) {
                eVar = a(cVar, dVar);
            } else if (O != 1) {
                if (O != 2) {
                    cVar.P();
                    cVar.Q();
                } else if (cVar.M() == bVar) {
                    cVar.Q();
                    z = true;
                } else {
                    bVar2 = b.v.u.c.s(cVar, dVar);
                }
            } else if (cVar.M() == bVar) {
                cVar.Q();
                z = true;
            } else {
                bVar3 = b.v.u.c.s(cVar, dVar);
            }
        }
        cVar.E();
        if (z) {
            dVar.a("Lottie doesn't support expressions.");
        }
        return eVar != null ? eVar : new c.a.a.z.j.i(bVar3, bVar2);
    }
}