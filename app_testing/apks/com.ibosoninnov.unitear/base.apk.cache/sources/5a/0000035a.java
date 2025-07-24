package b.d.b.d1;

import android.util.ArrayMap;
import b.d.a.f.i;
import b.d.b.d1.i0;
import java.util.Collections;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

/* compiled from: OptionsBundle.java */
/* loaded from: classes.dex */
public class w0 implements i0 {
    public static final w0 q = new w0(new TreeMap(i.f1490b));
    public final TreeMap<i0.a<?>, Map<i0.c, Object>> r;

    public w0(TreeMap<i0.a<?>, Map<i0.c, Object>> treeMap) {
        this.r = treeMap;
    }

    public static w0 x(i0 i0Var) {
        if (w0.class.equals(i0Var.getClass())) {
            return (w0) i0Var;
        }
        TreeMap treeMap = new TreeMap(i.f1490b);
        w0 w0Var = (w0) i0Var;
        for (i0.a<?> aVar : w0Var.e()) {
            Set<i0.c> h2 = w0Var.h(aVar);
            ArrayMap arrayMap = new ArrayMap();
            for (i0.c cVar : h2) {
                arrayMap.put(cVar, w0Var.d(aVar, cVar));
            }
            treeMap.put(aVar, arrayMap);
        }
        return new w0(treeMap);
    }

    @Override // b.d.b.d1.i0
    public <ValueT> ValueT a(i0.a<ValueT> aVar) {
        Map<i0.c, Object> map = this.r.get(aVar);
        if (map != null) {
            return (ValueT) map.get((i0.c) Collections.min(map.keySet()));
        }
        throw new IllegalArgumentException("Option does not exist: " + aVar);
    }

    @Override // b.d.b.d1.i0
    public boolean b(i0.a<?> aVar) {
        return this.r.containsKey(aVar);
    }

    @Override // b.d.b.d1.i0
    public void c(String str, i0.b bVar) {
        for (Map.Entry<i0.a<?>, Map<i0.c, Object>> entry : this.r.tailMap(new n(str, Void.class, null)).entrySet()) {
            if (!entry.getKey().a().startsWith(str)) {
                return;
            }
            i0.a<?> key = entry.getKey();
            b.d.a.f.g gVar = (b.d.a.f.g) bVar;
            i.a aVar = gVar.f1366a;
            i0 i0Var = gVar.f1367b;
            aVar.f1376a.A(key, i0Var.g(key), i0Var.a(key));
        }
    }

    @Override // b.d.b.d1.i0
    public <ValueT> ValueT d(i0.a<ValueT> aVar, i0.c cVar) {
        Map<i0.c, Object> map = this.r.get(aVar);
        if (map != null) {
            if (map.containsKey(cVar)) {
                return (ValueT) map.get(cVar);
            }
            throw new IllegalArgumentException("Option does not exist: " + aVar + " with priority=" + cVar);
        }
        throw new IllegalArgumentException("Option does not exist: " + aVar);
    }

    @Override // b.d.b.d1.i0
    public Set<i0.a<?>> e() {
        return Collections.unmodifiableSet(this.r.keySet());
    }

    @Override // b.d.b.d1.i0
    public <ValueT> ValueT f(i0.a<ValueT> aVar, ValueT valuet) {
        try {
            return (ValueT) a(aVar);
        } catch (IllegalArgumentException unused) {
            return valuet;
        }
    }

    @Override // b.d.b.d1.i0
    public i0.c g(i0.a<?> aVar) {
        Map<i0.c, Object> map = this.r.get(aVar);
        if (map != null) {
            return (i0.c) Collections.min(map.keySet());
        }
        throw new IllegalArgumentException("Option does not exist: " + aVar);
    }

    @Override // b.d.b.d1.i0
    public Set<i0.c> h(i0.a<?> aVar) {
        Map<i0.c, Object> map = this.r.get(aVar);
        if (map == null) {
            return Collections.emptySet();
        }
        return Collections.unmodifiableSet(map.keySet());
    }
}