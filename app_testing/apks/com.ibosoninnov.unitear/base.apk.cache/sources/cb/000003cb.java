package b.f;

import java.util.ConcurrentModificationException;
import java.util.Map;

/* compiled from: SimpleArrayMap.java */
/* loaded from: classes.dex */
public class h<K, V> {

    /* renamed from: b  reason: collision with root package name */
    public static Object[] f1769b;

    /* renamed from: c  reason: collision with root package name */
    public static int f1770c;

    /* renamed from: d  reason: collision with root package name */
    public static Object[] f1771d;

    /* renamed from: e  reason: collision with root package name */
    public static int f1772e;

    /* renamed from: f  reason: collision with root package name */
    public int[] f1773f;

    /* renamed from: g  reason: collision with root package name */
    public Object[] f1774g;

    /* renamed from: h  reason: collision with root package name */
    public int f1775h;

    public h() {
        this.f1773f = d.f1746a;
        this.f1774g = d.f1748c;
        this.f1775h = 0;
    }

    public static void c(int[] iArr, Object[] objArr, int i) {
        if (iArr.length == 8) {
            synchronized (h.class) {
                if (f1772e < 10) {
                    objArr[0] = f1771d;
                    objArr[1] = iArr;
                    for (int i2 = (i << 1) - 1; i2 >= 2; i2--) {
                        objArr[i2] = null;
                    }
                    f1771d = objArr;
                    f1772e++;
                }
            }
        } else if (iArr.length == 4) {
            synchronized (h.class) {
                if (f1770c < 10) {
                    objArr[0] = f1769b;
                    objArr[1] = iArr;
                    for (int i3 = (i << 1) - 1; i3 >= 2; i3--) {
                        objArr[i3] = null;
                    }
                    f1769b = objArr;
                    f1770c++;
                }
            }
        }
    }

    public final void a(int i) {
        if (i == 8) {
            synchronized (h.class) {
                Object[] objArr = f1771d;
                if (objArr != null) {
                    this.f1774g = objArr;
                    f1771d = (Object[]) objArr[0];
                    this.f1773f = (int[]) objArr[1];
                    objArr[1] = null;
                    objArr[0] = null;
                    f1772e--;
                    return;
                }
            }
        } else if (i == 4) {
            synchronized (h.class) {
                Object[] objArr2 = f1769b;
                if (objArr2 != null) {
                    this.f1774g = objArr2;
                    f1769b = (Object[]) objArr2[0];
                    this.f1773f = (int[]) objArr2[1];
                    objArr2[1] = null;
                    objArr2[0] = null;
                    f1770c--;
                    return;
                }
            }
        }
        this.f1773f = new int[i];
        this.f1774g = new Object[i << 1];
    }

    public void b(int i) {
        int i2 = this.f1775h;
        int[] iArr = this.f1773f;
        if (iArr.length < i) {
            Object[] objArr = this.f1774g;
            a(i);
            if (this.f1775h > 0) {
                System.arraycopy(iArr, 0, this.f1773f, 0, i2);
                System.arraycopy(objArr, 0, this.f1774g, 0, i2 << 1);
            }
            c(iArr, objArr, i2);
        }
        if (this.f1775h != i2) {
            throw new ConcurrentModificationException();
        }
    }

    public void clear() {
        int i = this.f1775h;
        if (i > 0) {
            int[] iArr = this.f1773f;
            Object[] objArr = this.f1774g;
            this.f1773f = d.f1746a;
            this.f1774g = d.f1748c;
            this.f1775h = 0;
            c(iArr, objArr, i);
        }
        if (this.f1775h > 0) {
            throw new ConcurrentModificationException();
        }
    }

    public boolean containsKey(Object obj) {
        return e(obj) >= 0;
    }

    public boolean containsValue(Object obj) {
        return g(obj) >= 0;
    }

    public int d(Object obj, int i) {
        int i2 = this.f1775h;
        if (i2 == 0) {
            return -1;
        }
        try {
            int a2 = d.a(this.f1773f, i2, i);
            if (a2 >= 0 && !obj.equals(this.f1774g[a2 << 1])) {
                int i3 = a2 + 1;
                while (i3 < i2 && this.f1773f[i3] == i) {
                    if (obj.equals(this.f1774g[i3 << 1])) {
                        return i3;
                    }
                    i3++;
                }
                for (int i4 = a2 - 1; i4 >= 0 && this.f1773f[i4] == i; i4--) {
                    if (obj.equals(this.f1774g[i4 << 1])) {
                        return i4;
                    }
                }
                return ~i3;
            }
            return a2;
        } catch (ArrayIndexOutOfBoundsException unused) {
            throw new ConcurrentModificationException();
        }
    }

    public int e(Object obj) {
        return obj == null ? f() : d(obj, obj.hashCode());
    }

    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }
        if (obj instanceof h) {
            h hVar = (h) obj;
            if (this.f1775h != hVar.f1775h) {
                return false;
            }
            for (int i = 0; i < this.f1775h; i++) {
                try {
                    K h2 = h(i);
                    V l = l(i);
                    Object obj2 = hVar.get(h2);
                    if (l == null) {
                        if (obj2 != null || !hVar.containsKey(h2)) {
                            return false;
                        }
                    } else if (!l.equals(obj2)) {
                        return false;
                    }
                } catch (ClassCastException | NullPointerException unused) {
                    return false;
                }
            }
            return true;
        }
        if (obj instanceof Map) {
            Map map = (Map) obj;
            if (this.f1775h != map.size()) {
                return false;
            }
            for (int i2 = 0; i2 < this.f1775h; i2++) {
                try {
                    K h3 = h(i2);
                    V l2 = l(i2);
                    Object obj3 = map.get(h3);
                    if (l2 == null) {
                        if (obj3 != null || !map.containsKey(h3)) {
                            return false;
                        }
                    } else if (!l2.equals(obj3)) {
                        return false;
                    }
                } catch (ClassCastException | NullPointerException unused2) {
                }
            }
            return true;
        }
        return false;
    }

    public int f() {
        int i = this.f1775h;
        if (i == 0) {
            return -1;
        }
        try {
            int a2 = d.a(this.f1773f, i, 0);
            if (a2 >= 0 && this.f1774g[a2 << 1] != null) {
                int i2 = a2 + 1;
                while (i2 < i && this.f1773f[i2] == 0) {
                    if (this.f1774g[i2 << 1] == null) {
                        return i2;
                    }
                    i2++;
                }
                for (int i3 = a2 - 1; i3 >= 0 && this.f1773f[i3] == 0; i3--) {
                    if (this.f1774g[i3 << 1] == null) {
                        return i3;
                    }
                }
                return ~i2;
            }
            return a2;
        } catch (ArrayIndexOutOfBoundsException unused) {
            throw new ConcurrentModificationException();
        }
    }

    public int g(Object obj) {
        int i = this.f1775h * 2;
        Object[] objArr = this.f1774g;
        if (obj == null) {
            for (int i2 = 1; i2 < i; i2 += 2) {
                if (objArr[i2] == null) {
                    return i2 >> 1;
                }
            }
            return -1;
        }
        for (int i3 = 1; i3 < i; i3 += 2) {
            if (obj.equals(objArr[i3])) {
                return i3 >> 1;
            }
        }
        return -1;
    }

    public V get(Object obj) {
        return getOrDefault(obj, null);
    }

    public V getOrDefault(Object obj, V v) {
        int e2 = e(obj);
        return e2 >= 0 ? (V) this.f1774g[(e2 << 1) + 1] : v;
    }

    public K h(int i) {
        return (K) this.f1774g[i << 1];
    }

    public int hashCode() {
        int[] iArr = this.f1773f;
        Object[] objArr = this.f1774g;
        int i = this.f1775h;
        int i2 = 1;
        int i3 = 0;
        int i4 = 0;
        while (i3 < i) {
            Object obj = objArr[i2];
            i4 += (obj == null ? 0 : obj.hashCode()) ^ iArr[i3];
            i3++;
            i2 += 2;
        }
        return i4;
    }

    public void i(h<? extends K, ? extends V> hVar) {
        int i = hVar.f1775h;
        b(this.f1775h + i);
        if (this.f1775h != 0) {
            for (int i2 = 0; i2 < i; i2++) {
                put(hVar.h(i2), hVar.l(i2));
            }
        } else if (i > 0) {
            System.arraycopy(hVar.f1773f, 0, this.f1773f, 0, i);
            System.arraycopy(hVar.f1774g, 0, this.f1774g, 0, i << 1);
            this.f1775h = i;
        }
    }

    public boolean isEmpty() {
        return this.f1775h <= 0;
    }

    public V j(int i) {
        Object[] objArr = this.f1774g;
        int i2 = i << 1;
        V v = (V) objArr[i2 + 1];
        int i3 = this.f1775h;
        int i4 = 0;
        if (i3 <= 1) {
            c(this.f1773f, objArr, i3);
            this.f1773f = d.f1746a;
            this.f1774g = d.f1748c;
        } else {
            int i5 = i3 - 1;
            int[] iArr = this.f1773f;
            if (iArr.length > 8 && i3 < iArr.length / 3) {
                a(i3 > 8 ? i3 + (i3 >> 1) : 8);
                if (i3 != this.f1775h) {
                    throw new ConcurrentModificationException();
                }
                if (i > 0) {
                    System.arraycopy(iArr, 0, this.f1773f, 0, i);
                    System.arraycopy(objArr, 0, this.f1774g, 0, i2);
                }
                if (i < i5) {
                    int i6 = i + 1;
                    int i7 = i5 - i;
                    System.arraycopy(iArr, i6, this.f1773f, i, i7);
                    System.arraycopy(objArr, i6 << 1, this.f1774g, i2, i7 << 1);
                }
            } else {
                if (i < i5) {
                    int i8 = i + 1;
                    int i9 = i5 - i;
                    System.arraycopy(iArr, i8, iArr, i, i9);
                    Object[] objArr2 = this.f1774g;
                    System.arraycopy(objArr2, i8 << 1, objArr2, i2, i9 << 1);
                }
                Object[] objArr3 = this.f1774g;
                int i10 = i5 << 1;
                objArr3[i10] = null;
                objArr3[i10 + 1] = null;
            }
            i4 = i5;
        }
        if (i3 == this.f1775h) {
            this.f1775h = i4;
            return v;
        }
        throw new ConcurrentModificationException();
    }

    public V k(int i, V v) {
        int i2 = (i << 1) + 1;
        Object[] objArr = this.f1774g;
        V v2 = (V) objArr[i2];
        objArr[i2] = v;
        return v2;
    }

    public V l(int i) {
        return (V) this.f1774g[(i << 1) + 1];
    }

    public V put(K k, V v) {
        int i;
        int d2;
        int i2 = this.f1775h;
        if (k == null) {
            d2 = f();
            i = 0;
        } else {
            int hashCode = k.hashCode();
            i = hashCode;
            d2 = d(k, hashCode);
        }
        if (d2 >= 0) {
            int i3 = (d2 << 1) + 1;
            Object[] objArr = this.f1774g;
            V v2 = (V) objArr[i3];
            objArr[i3] = v;
            return v2;
        }
        int i4 = ~d2;
        int[] iArr = this.f1773f;
        if (i2 >= iArr.length) {
            int i5 = 4;
            if (i2 >= 8) {
                i5 = (i2 >> 1) + i2;
            } else if (i2 >= 4) {
                i5 = 8;
            }
            Object[] objArr2 = this.f1774g;
            a(i5);
            if (i2 == this.f1775h) {
                int[] iArr2 = this.f1773f;
                if (iArr2.length > 0) {
                    System.arraycopy(iArr, 0, iArr2, 0, iArr.length);
                    System.arraycopy(objArr2, 0, this.f1774g, 0, objArr2.length);
                }
                c(iArr, objArr2, i2);
            } else {
                throw new ConcurrentModificationException();
            }
        }
        if (i4 < i2) {
            int[] iArr3 = this.f1773f;
            int i6 = i4 + 1;
            System.arraycopy(iArr3, i4, iArr3, i6, i2 - i4);
            Object[] objArr3 = this.f1774g;
            System.arraycopy(objArr3, i4 << 1, objArr3, i6 << 1, (this.f1775h - i4) << 1);
        }
        int i7 = this.f1775h;
        if (i2 == i7) {
            int[] iArr4 = this.f1773f;
            if (i4 < iArr4.length) {
                iArr4[i4] = i;
                Object[] objArr4 = this.f1774g;
                int i8 = i4 << 1;
                objArr4[i8] = k;
                objArr4[i8 + 1] = v;
                this.f1775h = i7 + 1;
                return null;
            }
        }
        throw new ConcurrentModificationException();
    }

    public V putIfAbsent(K k, V v) {
        V orDefault = getOrDefault(k, null);
        return orDefault == null ? put(k, v) : orDefault;
    }

    public V remove(Object obj) {
        int e2 = e(obj);
        if (e2 >= 0) {
            return j(e2);
        }
        return null;
    }

    public V replace(K k, V v) {
        int e2 = e(k);
        if (e2 >= 0) {
            return k(e2, v);
        }
        return null;
    }

    public int size() {
        return this.f1775h;
    }

    public String toString() {
        if (isEmpty()) {
            return "{}";
        }
        StringBuilder sb = new StringBuilder(this.f1775h * 28);
        sb.append('{');
        for (int i = 0; i < this.f1775h; i++) {
            if (i > 0) {
                sb.append(", ");
            }
            K h2 = h(i);
            if (h2 != this) {
                sb.append(h2);
            } else {
                sb.append("(this Map)");
            }
            sb.append('=');
            V l = l(i);
            if (l != this) {
                sb.append(l);
            } else {
                sb.append("(this Map)");
            }
        }
        sb.append('}');
        return sb.toString();
    }

    public boolean remove(Object obj, Object obj2) {
        int e2 = e(obj);
        if (e2 >= 0) {
            V l = l(e2);
            if (obj2 == l || (obj2 != null && obj2.equals(l))) {
                j(e2);
                return true;
            }
            return false;
        }
        return false;
    }

    public boolean replace(K k, V v, V v2) {
        int e2 = e(k);
        if (e2 >= 0) {
            V l = l(e2);
            if (l == v || (v != null && v.equals(l))) {
                k(e2, v2);
                return true;
            }
            return false;
        }
        return false;
    }

    public h(int i) {
        if (i == 0) {
            this.f1773f = d.f1746a;
            this.f1774g = d.f1748c;
        } else {
            a(i);
        }
        this.f1775h = 0;
    }
}