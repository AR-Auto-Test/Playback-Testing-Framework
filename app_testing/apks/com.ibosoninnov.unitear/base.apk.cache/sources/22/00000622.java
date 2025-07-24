package c.a.a.b0;

import android.graphics.PointF;
import c.a.a.b0.h0.c;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/* compiled from: ShapeDataParser.java */
/* loaded from: classes.dex */
public class a0 implements g0<c.a.a.z.k.k> {

    /* renamed from: a  reason: collision with root package name */
    public static final a0 f2955a = new a0();

    /* renamed from: b  reason: collision with root package name */
    public static final c.a f2956b = c.a.a("c", "v", "i", "o");

    /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
    @Override // c.a.a.b0.g0
    public c.a.a.z.k.k a(c.a.a.b0.h0.c cVar, float f2) {
        if (cVar.M() == c.b.BEGIN_ARRAY) {
            cVar.B();
        }
        cVar.C();
        List<PointF> list = null;
        List<PointF> list2 = null;
        List<PointF> list3 = null;
        boolean z = false;
        while (cVar.G()) {
            int O = cVar.O(f2956b);
            if (O == 0) {
                z = cVar.H();
            } else if (O == 1) {
                list = o.c(cVar, f2);
            } else if (O == 2) {
                list2 = o.c(cVar, f2);
            } else if (O != 3) {
                cVar.P();
                cVar.Q();
            } else {
                list3 = o.c(cVar, f2);
            }
        }
        cVar.E();
        if (cVar.M() == c.b.END_ARRAY) {
            cVar.D();
        }
        if (list != null && list2 != null && list3 != null) {
            if (list.isEmpty()) {
                return new c.a.a.z.k.k(new PointF(), false, Collections.emptyList());
            }
            int size = list.size();
            PointF pointF = list.get(0);
            ArrayList arrayList = new ArrayList(size);
            for (int i = 1; i < size; i++) {
                PointF pointF2 = list.get(i);
                int i2 = i - 1;
                arrayList.add(new c.a.a.z.a(c.a.a.c0.f.a(list.get(i2), list3.get(i2)), c.a.a.c0.f.a(pointF2, list2.get(i)), pointF2));
            }
            if (z) {
                PointF pointF3 = list.get(0);
                int i3 = size - 1;
                arrayList.add(new c.a.a.z.a(c.a.a.c0.f.a(list.get(i3), list3.get(i3)), c.a.a.c0.f.a(pointF3, list2.get(0)), pointF3));
            }
            return new c.a.a.z.k.k(pointF, z, arrayList);
        }
        throw new IllegalArgumentException("Shape data was missing information.");
    }
}