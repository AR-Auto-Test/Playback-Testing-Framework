package b.f;

import b.f.g;
import java.util.Collection;
import java.util.Map;
import java.util.Set;

/* compiled from: ArrayMap.java */
/* loaded from: classes.dex */
public class a<K, V> extends h<K, V> implements Map<K, V> {
    public g<K, V> i;

    /* compiled from: ArrayMap.java */
    /* renamed from: b.f.a$a  reason: collision with other inner class name */
    /* loaded from: classes.dex */
    public class C0026a extends g<K, V> {
        public C0026a() {
        }

        @Override // b.f.g
        public void a() {
            a.this.clear();
        }

        @Override // b.f.g
        public Object b(int i, int i2) {
            return a.this.f1774g[(i << 1) + i2];
        }

        @Override // b.f.g
        public Map<K, V> c() {
            return a.this;
        }

        @Override // b.f.g
        public int d() {
            return a.this.f1775h;
        }

        @Override // b.f.g
        public int e(Object obj) {
            return a.this.e(obj);
        }

        @Override // b.f.g
        public int f(Object obj) {
            return a.this.g(obj);
        }

        @Override // b.f.g
        public void g(K k, V v) {
            a.this.put(k, v);
        }

        @Override // b.f.g
        public void h(int i) {
            a.this.j(i);
        }

        @Override // b.f.g
        public V i(int i, V v) {
            return a.this.k(i, v);
        }
    }

    public a() {
    }

    @Override // java.util.Map
    public Set<Map.Entry<K, V>> entrySet() {
        g<K, V> m = m();
        if (m.f1754a == null) {
            m.f1754a = new g.b();
        }
        return m.f1754a;
    }

    @Override // java.util.Map
    public Set<K> keySet() {
        g<K, V> m = m();
        if (m.f1755b == null) {
            m.f1755b = new g.c();
        }
        return m.f1755b;
    }

    public final g<K, V> m() {
        if (this.i == null) {
            this.i = new C0026a();
        }
        return this.i;
    }

    @Override // java.util.Map
    public void putAll(Map<? extends K, ? extends V> map) {
        b(map.size() + this.f1775h);
        for (Map.Entry<? extends K, ? extends V> entry : map.entrySet()) {
            put(entry.getKey(), entry.getValue());
        }
    }

    @Override // java.util.Map
    public Collection<V> values() {
        g<K, V> m = m();
        if (m.f1756c == null) {
            m.f1756c = new g.e();
        }
        return m.f1756c;
    }

    public a(int i) {
        super(i);
    }

    public a(h hVar) {
        if (hVar != null) {
            i(hVar);
        }
    }
}