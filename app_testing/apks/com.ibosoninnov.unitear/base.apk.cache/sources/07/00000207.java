package b.c.a.b;

import java.util.Iterator;
import java.util.Map;
import java.util.WeakHashMap;

/* compiled from: SafeIterableMap.java */
/* loaded from: classes.dex */
public class b<K, V> implements Iterable<Map.Entry<K, V>> {

    /* renamed from: b  reason: collision with root package name */
    public c<K, V> f992b;

    /* renamed from: c  reason: collision with root package name */
    public c<K, V> f993c;

    /* renamed from: d  reason: collision with root package name */
    public WeakHashMap<f<K, V>, Boolean> f994d = new WeakHashMap<>();

    /* renamed from: e  reason: collision with root package name */
    public int f995e = 0;

    /* compiled from: SafeIterableMap.java */
    /* loaded from: classes.dex */
    public static class a<K, V> extends e<K, V> {
        public a(c<K, V> cVar, c<K, V> cVar2) {
            super(cVar, cVar2);
        }

        @Override // b.c.a.b.b.e
        public c<K, V> b(c<K, V> cVar) {
            return cVar.f999e;
        }

        @Override // b.c.a.b.b.e
        public c<K, V> c(c<K, V> cVar) {
            return cVar.f998d;
        }
    }

    /* compiled from: SafeIterableMap.java */
    /* renamed from: b.c.a.b.b$b  reason: collision with other inner class name */
    /* loaded from: classes.dex */
    public static class C0011b<K, V> extends e<K, V> {
        public C0011b(c<K, V> cVar, c<K, V> cVar2) {
            super(cVar, cVar2);
        }

        @Override // b.c.a.b.b.e
        public c<K, V> b(c<K, V> cVar) {
            return cVar.f998d;
        }

        @Override // b.c.a.b.b.e
        public c<K, V> c(c<K, V> cVar) {
            return cVar.f999e;
        }
    }

    /* compiled from: SafeIterableMap.java */
    /* loaded from: classes.dex */
    public static class c<K, V> implements Map.Entry<K, V> {

        /* renamed from: b  reason: collision with root package name */
        public final K f996b;

        /* renamed from: c  reason: collision with root package name */
        public final V f997c;

        /* renamed from: d  reason: collision with root package name */
        public c<K, V> f998d;

        /* renamed from: e  reason: collision with root package name */
        public c<K, V> f999e;

        public c(K k, V v) {
            this.f996b = k;
            this.f997c = v;
        }

        @Override // java.util.Map.Entry
        public boolean equals(Object obj) {
            if (obj == this) {
                return true;
            }
            if (obj instanceof c) {
                c cVar = (c) obj;
                return this.f996b.equals(cVar.f996b) && this.f997c.equals(cVar.f997c);
            }
            return false;
        }

        @Override // java.util.Map.Entry
        public K getKey() {
            return this.f996b;
        }

        @Override // java.util.Map.Entry
        public V getValue() {
            return this.f997c;
        }

        @Override // java.util.Map.Entry
        public int hashCode() {
            return this.f996b.hashCode() ^ this.f997c.hashCode();
        }

        @Override // java.util.Map.Entry
        public V setValue(V v) {
            throw new UnsupportedOperationException("An entry modification is not supported");
        }

        public String toString() {
            return this.f996b + "=" + this.f997c;
        }
    }

    /* compiled from: SafeIterableMap.java */
    /* loaded from: classes.dex */
    public class d implements Iterator<Map.Entry<K, V>>, f<K, V> {

        /* renamed from: b  reason: collision with root package name */
        public c<K, V> f1000b;

        /* renamed from: c  reason: collision with root package name */
        public boolean f1001c = true;

        public d() {
        }

        @Override // b.c.a.b.b.f
        public void a(c<K, V> cVar) {
            c<K, V> cVar2 = this.f1000b;
            if (cVar == cVar2) {
                c<K, V> cVar3 = cVar2.f999e;
                this.f1000b = cVar3;
                this.f1001c = cVar3 == null;
            }
        }

        @Override // java.util.Iterator
        public boolean hasNext() {
            if (this.f1001c) {
                return b.this.f992b != null;
            }
            c<K, V> cVar = this.f1000b;
            return (cVar == null || cVar.f998d == null) ? false : true;
        }

        @Override // java.util.Iterator
        public Object next() {
            if (this.f1001c) {
                this.f1001c = false;
                this.f1000b = b.this.f992b;
            } else {
                c<K, V> cVar = this.f1000b;
                this.f1000b = cVar != null ? cVar.f998d : null;
            }
            return this.f1000b;
        }
    }

    /* compiled from: SafeIterableMap.java */
    /* loaded from: classes.dex */
    public static abstract class e<K, V> implements Iterator<Map.Entry<K, V>>, f<K, V> {

        /* renamed from: b  reason: collision with root package name */
        public c<K, V> f1003b;

        /* renamed from: c  reason: collision with root package name */
        public c<K, V> f1004c;

        public e(c<K, V> cVar, c<K, V> cVar2) {
            this.f1003b = cVar2;
            this.f1004c = cVar;
        }

        @Override // b.c.a.b.b.f
        public void a(c<K, V> cVar) {
            c<K, V> cVar2 = null;
            if (this.f1003b == cVar && cVar == this.f1004c) {
                this.f1004c = null;
                this.f1003b = null;
            }
            c<K, V> cVar3 = this.f1003b;
            if (cVar3 == cVar) {
                this.f1003b = b(cVar3);
            }
            c<K, V> cVar4 = this.f1004c;
            if (cVar4 == cVar) {
                c<K, V> cVar5 = this.f1003b;
                if (cVar4 != cVar5 && cVar5 != null) {
                    cVar2 = c(cVar4);
                }
                this.f1004c = cVar2;
            }
        }

        public abstract c<K, V> b(c<K, V> cVar);

        public abstract c<K, V> c(c<K, V> cVar);

        @Override // java.util.Iterator
        public boolean hasNext() {
            return this.f1004c != null;
        }

        @Override // java.util.Iterator
        public Object next() {
            c<K, V> cVar = this.f1004c;
            c<K, V> cVar2 = this.f1003b;
            this.f1004c = (cVar == cVar2 || cVar2 == null) ? null : c(cVar);
            return cVar;
        }
    }

    /* compiled from: SafeIterableMap.java */
    /* loaded from: classes.dex */
    public interface f<K, V> {
        void a(c<K, V> cVar);
    }

    public c<K, V> a(K k) {
        c<K, V> cVar = this.f992b;
        while (cVar != null && !cVar.f996b.equals(k)) {
            cVar = cVar.f998d;
        }
        return cVar;
    }

    public b<K, V>.d b() {
        b<K, V>.d dVar = new d();
        this.f994d.put(dVar, Boolean.FALSE);
        return dVar;
    }

    public c<K, V> c(K k, V v) {
        c<K, V> cVar = new c<>(k, v);
        this.f995e++;
        c<K, V> cVar2 = this.f993c;
        if (cVar2 == null) {
            this.f992b = cVar;
            this.f993c = cVar;
            return cVar;
        }
        cVar2.f998d = cVar;
        cVar.f999e = cVar2;
        this.f993c = cVar;
        return cVar;
    }

    public V d(K k, V v) {
        c<K, V> a2 = a(k);
        if (a2 != null) {
            return a2.f997c;
        }
        c(k, v);
        return null;
    }

    public V e(K k) {
        c<K, V> a2 = a(k);
        if (a2 == null) {
            return null;
        }
        this.f995e--;
        if (!this.f994d.isEmpty()) {
            for (f<K, V> fVar : this.f994d.keySet()) {
                fVar.a(a2);
            }
        }
        c<K, V> cVar = a2.f999e;
        if (cVar != null) {
            cVar.f998d = a2.f998d;
        } else {
            this.f992b = a2.f998d;
        }
        c<K, V> cVar2 = a2.f998d;
        if (cVar2 != null) {
            cVar2.f999e = cVar;
        } else {
            this.f993c = cVar;
        }
        a2.f998d = null;
        a2.f999e = null;
        return a2.f997c;
    }

    /* JADX WARN: Code restructure failed: missing block: B:24:0x0048, code lost:
        if (r3.hasNext() != false) goto L35;
     */
    /* JADX WARN: Code restructure failed: missing block: B:26:0x0050, code lost:
        if (((b.c.a.b.b.e) r7).hasNext() != false) goto L35;
     */
    /* JADX WARN: Code restructure failed: missing block: B:29:0x0054, code lost:
        return false;
     */
    /* JADX WARN: Code restructure failed: missing block: B:38:?, code lost:
        return true;
     */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public boolean equals(Object obj) {
        if (obj == this) {
            return true;
        }
        if (obj instanceof b) {
            b bVar = (b) obj;
            if (this.f995e != bVar.f995e) {
                return false;
            }
            Iterator<Map.Entry<K, V>> it = iterator();
            Iterator<Map.Entry<K, V>> it2 = bVar.iterator();
            while (true) {
                e eVar = (e) it;
                if (!eVar.hasNext()) {
                    break;
                }
                e eVar2 = (e) it2;
                if (!eVar2.hasNext()) {
                    break;
                }
                Map.Entry entry = (Map.Entry) eVar.next();
                Object next = eVar2.next();
                if ((entry != null || next == null) && (entry == null || entry.equals(next))) {
                }
            }
            return false;
        }
        return false;
    }

    public int hashCode() {
        Iterator<Map.Entry<K, V>> it = iterator();
        int i = 0;
        while (true) {
            e eVar = (e) it;
            if (!eVar.hasNext()) {
                return i;
            }
            i += ((Map.Entry) eVar.next()).hashCode();
        }
    }

    @Override // java.lang.Iterable
    public Iterator<Map.Entry<K, V>> iterator() {
        a aVar = new a(this.f992b, this.f993c);
        this.f994d.put(aVar, Boolean.FALSE);
        return aVar;
    }

    public String toString() {
        StringBuilder x = c.b.a.a.a.x("[");
        Iterator<Map.Entry<K, V>> it = iterator();
        while (true) {
            e eVar = (e) it;
            if (eVar.hasNext()) {
                x.append(((Map.Entry) eVar.next()).toString());
                if (eVar.hasNext()) {
                    x.append(", ");
                }
            } else {
                x.append("]");
                return x.toString();
            }
        }
    }
}