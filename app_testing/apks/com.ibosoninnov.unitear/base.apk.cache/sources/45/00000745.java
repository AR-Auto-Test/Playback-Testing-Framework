package c.c.a.m.v.c0;

import c.c.a.m.v.c0.l;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/* compiled from: GroupedLinkedMap.java */
/* loaded from: classes.dex */
public class g<K extends l, V> {

    /* renamed from: a  reason: collision with root package name */
    public final a<K, V> f3607a = new a<>(null);

    /* renamed from: b  reason: collision with root package name */
    public final Map<K, a<K, V>> f3608b = new HashMap();

    /* compiled from: GroupedLinkedMap.java */
    /* loaded from: classes.dex */
    public static class a<K, V> {

        /* renamed from: a  reason: collision with root package name */
        public final K f3609a;

        /* renamed from: b  reason: collision with root package name */
        public List<V> f3610b;

        /* renamed from: c  reason: collision with root package name */
        public a<K, V> f3611c;

        /* renamed from: d  reason: collision with root package name */
        public a<K, V> f3612d;

        public a() {
            this(null);
        }

        public V a() {
            List<V> list = this.f3610b;
            int size = list != null ? list.size() : 0;
            if (size > 0) {
                return this.f3610b.remove(size - 1);
            }
            return null;
        }

        public a(K k) {
            this.f3612d = this;
            this.f3611c = this;
            this.f3609a = k;
        }
    }

    public V a(K k) {
        a<K, V> aVar = this.f3608b.get(k);
        if (aVar == null) {
            aVar = new a<>(k);
            this.f3608b.put(k, aVar);
        } else {
            k.a();
        }
        a<K, V> aVar2 = aVar.f3612d;
        aVar2.f3611c = aVar.f3611c;
        aVar.f3611c.f3612d = aVar2;
        a<K, V> aVar3 = this.f3607a;
        aVar.f3612d = aVar3;
        a<K, V> aVar4 = aVar3.f3611c;
        aVar.f3611c = aVar4;
        aVar4.f3612d = aVar;
        aVar.f3612d.f3611c = aVar;
        return aVar.a();
    }

    public void b(K k, V v) {
        a<K, V> aVar = this.f3608b.get(k);
        if (aVar == null) {
            aVar = new a<>(k);
            a<K, V> aVar2 = aVar.f3612d;
            aVar2.f3611c = aVar.f3611c;
            aVar.f3611c.f3612d = aVar2;
            a<K, V> aVar3 = this.f3607a;
            aVar.f3612d = aVar3.f3612d;
            aVar.f3611c = aVar3;
            aVar3.f3612d = aVar;
            aVar.f3612d.f3611c = aVar;
            this.f3608b.put(k, aVar);
        } else {
            k.a();
        }
        if (aVar.f3610b == null) {
            aVar.f3610b = new ArrayList();
        }
        aVar.f3610b.add(v);
    }

    public V c() {
        for (a aVar = this.f3607a.f3612d; !aVar.equals(this.f3607a); aVar = aVar.f3612d) {
            V v = (V) aVar.a();
            if (v != null) {
                return v;
            }
            a<K, V> aVar2 = aVar.f3612d;
            aVar2.f3611c = aVar.f3611c;
            aVar.f3611c.f3612d = aVar2;
            this.f3608b.remove(aVar.f3609a);
            ((l) aVar.f3609a).a();
        }
        return null;
    }

    public String toString() {
        StringBuilder sb = new StringBuilder("GroupedLinkedMap( ");
        boolean z = false;
        for (a aVar = this.f3607a.f3611c; !aVar.equals(this.f3607a); aVar = aVar.f3611c) {
            z = true;
            sb.append('{');
            sb.append(aVar.f3609a);
            sb.append(':');
            List<V> list = aVar.f3610b;
            sb.append(list != null ? list.size() : 0);
            sb.append("}, ");
        }
        if (z) {
            sb.delete(sb.length() - 2, sb.length());
        }
        sb.append(" )");
        return sb.toString();
    }
}