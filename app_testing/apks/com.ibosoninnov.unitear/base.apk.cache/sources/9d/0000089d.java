package c.c.a.s;

/* compiled from: CachedHashCodeArrayMap.java */
/* loaded from: classes.dex */
public final class b<K, V> extends b.f.a<K, V> {
    public int j;

    @Override // b.f.h, java.util.Map
    public void clear() {
        this.j = 0;
        super.clear();
    }

    @Override // b.f.h, java.util.Map
    public int hashCode() {
        if (this.j == 0) {
            this.j = super.hashCode();
        }
        return this.j;
    }

    @Override // b.f.h
    public void i(b.f.h<? extends K, ? extends V> hVar) {
        this.j = 0;
        super.i(hVar);
    }

    @Override // b.f.h
    public V j(int i) {
        this.j = 0;
        return (V) super.j(i);
    }

    @Override // b.f.h
    public V k(int i, V v) {
        this.j = 0;
        int i2 = (i << 1) + 1;
        Object[] objArr = this.f1774g;
        V v2 = (V) objArr[i2];
        objArr[i2] = v;
        return v2;
    }

    @Override // b.f.h, java.util.Map
    public V put(K k, V v) {
        this.j = 0;
        return (V) super.put(k, v);
    }
}