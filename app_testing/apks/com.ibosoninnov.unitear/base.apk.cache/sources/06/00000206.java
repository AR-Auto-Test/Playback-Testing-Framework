package b.c.a.b;

import b.c.a.b.b;
import java.util.HashMap;

/* compiled from: FastSafeIterableMap.java */
/* loaded from: classes.dex */
public class a<K, V> extends b<K, V> {

    /* renamed from: f  reason: collision with root package name */
    public HashMap<K, b.c<K, V>> f991f = new HashMap<>();

    @Override // b.c.a.b.b
    public b.c<K, V> a(K k) {
        return this.f991f.get(k);
    }

    public boolean contains(K k) {
        return this.f991f.containsKey(k);
    }

    @Override // b.c.a.b.b
    public V d(K k, V v) {
        b.c<K, V> cVar = this.f991f.get(k);
        if (cVar != null) {
            return cVar.f997c;
        }
        this.f991f.put(k, c(k, v));
        return null;
    }

    @Override // b.c.a.b.b
    public V e(K k) {
        V v = (V) super.e(k);
        this.f991f.remove(k);
        return v;
    }
}