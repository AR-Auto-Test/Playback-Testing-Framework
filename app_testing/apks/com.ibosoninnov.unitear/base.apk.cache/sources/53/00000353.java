package b.d.b.d1;

import android.util.ArrayMap;
import b.d.b.d1.i0;
import java.util.Collections;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

/* compiled from: MutableOptionsBundle.java */
/* loaded from: classes.dex */
public final class u0 extends w0 implements t0 {
    public u0(TreeMap<i0.a<?>, Map<i0.c, Object>> treeMap) {
        super(treeMap);
    }

    public static u0 y() {
        return new u0(new TreeMap(i.f1490b));
    }

    public static u0 z(i0 i0Var) {
        TreeMap treeMap = new TreeMap(i.f1490b);
        for (i0.a<?> aVar : i0Var.e()) {
            Set<i0.c> h2 = i0Var.h(aVar);
            ArrayMap arrayMap = new ArrayMap();
            for (i0.c cVar : h2) {
                arrayMap.put(cVar, i0Var.d(aVar, cVar));
            }
            treeMap.put(aVar, arrayMap);
        }
        return new u0(treeMap);
    }

    public <ValueT> void A(i0.a<ValueT> aVar, i0.c cVar, ValueT valuet) {
        i0.c cVar2;
        Map<i0.c, Object> map = this.r.get(aVar);
        if (map == null) {
            ArrayMap arrayMap = new ArrayMap();
            this.r.put(aVar, arrayMap);
            arrayMap.put(cVar, valuet);
            return;
        }
        i0.c cVar3 = (i0.c) Collections.min(map.keySet());
        if (!map.get(cVar3).equals(valuet)) {
            i0.c cVar4 = i0.c.ALWAYS_OVERRIDE;
            boolean z = true;
            if ((cVar3 != cVar4 || cVar != cVar4) && (cVar3 != (cVar2 = i0.c.REQUIRED) || cVar != cVar2)) {
                z = false;
            }
            if (z) {
                StringBuilder x = c.b.a.a.a.x("Option values conflicts: ");
                x.append(aVar.a());
                x.append(", existing value (");
                x.append(cVar3);
                x.append(")=");
                x.append(map.get(cVar3));
                x.append(", conflicting (");
                x.append(cVar);
                x.append(")=");
                x.append(valuet);
                throw new IllegalArgumentException(x.toString());
            }
        }
        map.put(cVar, valuet);
    }
}