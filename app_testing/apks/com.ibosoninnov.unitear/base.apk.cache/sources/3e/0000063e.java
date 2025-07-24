package c.a.a.b0;

import c.a.a.b0.h0.c;
import java.util.ArrayList;
import java.util.List;

/* compiled from: KeyframesParser.java */
/* loaded from: classes.dex */
public class q {

    /* renamed from: a  reason: collision with root package name */
    public static c.a f3003a = c.a.a("k");

    public static <T> List<c.a.a.d0.a<T>> a(c.a.a.b0.h0.c cVar, c.a.a.d dVar, float f2, g0<T> g0Var) {
        ArrayList arrayList = new ArrayList();
        if (cVar.M() == c.b.STRING) {
            dVar.a("Lottie doesn't support expressions.");
            return arrayList;
        }
        cVar.C();
        while (cVar.G()) {
            if (cVar.O(f3003a) != 0) {
                cVar.Q();
            } else if (cVar.M() == c.b.BEGIN_ARRAY) {
                cVar.B();
                if (cVar.M() == c.b.NUMBER) {
                    arrayList.add(p.a(cVar, dVar, f2, g0Var, false));
                } else {
                    while (cVar.G()) {
                        arrayList.add(p.a(cVar, dVar, f2, g0Var, true));
                    }
                }
                cVar.D();
            } else {
                arrayList.add(p.a(cVar, dVar, f2, g0Var, false));
            }
        }
        cVar.E();
        b(arrayList);
        return arrayList;
    }

    public static <T> void b(List<? extends c.a.a.d0.a<T>> list) {
        int i;
        T t;
        int size = list.size();
        int i2 = 0;
        while (true) {
            i = size - 1;
            if (i2 >= i) {
                break;
            }
            c.a.a.d0.a<T> aVar = list.get(i2);
            i2++;
            c.a.a.d0.a<T> aVar2 = list.get(i2);
            aVar.f3050f = Float.valueOf(aVar2.f3049e);
            if (aVar.f3047c == null && (t = aVar2.f3046b) != null) {
                aVar.f3047c = t;
                if (aVar instanceof c.a.a.x.c.h) {
                    ((c.a.a.x.c.h) aVar).e();
                }
            }
        }
        c.a.a.d0.a<T> aVar3 = list.get(i);
        if ((aVar3.f3046b == null || aVar3.f3047c == null) && list.size() > 1) {
            list.remove(aVar3);
        }
    }
}