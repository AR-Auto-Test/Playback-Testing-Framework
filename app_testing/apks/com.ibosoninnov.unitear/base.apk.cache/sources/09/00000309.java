package b.d.b.d1;

import android.util.ArrayMap;
import java.util.ArrayList;
import java.util.HashSet;

/* compiled from: CaptureStage.java */
/* loaded from: classes.dex */
public interface h0 {

    /* compiled from: CaptureStage.java */
    /* loaded from: classes.dex */
    public static final class a implements h0 {
        public a() {
            HashSet hashSet = new HashSet();
            u0 y = u0.y();
            ArrayList arrayList = new ArrayList();
            v0 v0Var = new v0(new ArrayMap());
            ArrayList arrayList2 = new ArrayList(hashSet);
            w0 x = w0.x(y);
            g1 g1Var = g1.f1479a;
            ArrayMap arrayMap = new ArrayMap();
            for (String str : v0Var.f1480b.keySet()) {
                arrayMap.put(str, v0Var.a(str));
            }
            new f0(arrayList2, x, -1, arrayList, false, new g1(arrayMap));
        }
    }
}