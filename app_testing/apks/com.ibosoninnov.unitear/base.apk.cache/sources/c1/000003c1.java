package b.f;

import b.f.g;
import java.lang.reflect.Array;
import java.util.Collection;
import java.util.Iterator;
import java.util.Set;

/* compiled from: ArraySet.java */
/* loaded from: classes.dex */
public final class c<E> implements Collection<E>, Set<E> {

    /* renamed from: b  reason: collision with root package name */
    public static final int[] f1739b = new int[0];

    /* renamed from: c  reason: collision with root package name */
    public static final Object[] f1740c = new Object[0];

    /* renamed from: d  reason: collision with root package name */
    public static Object[] f1741d;

    /* renamed from: e  reason: collision with root package name */
    public static int f1742e;

    /* renamed from: f  reason: collision with root package name */
    public static Object[] f1743f;

    /* renamed from: g  reason: collision with root package name */
    public static int f1744g;

    /* renamed from: h  reason: collision with root package name */
    public int[] f1745h;
    public Object[] i;
    public int j;
    public g<E, E> k;

    public c() {
        this(0);
    }

    public static void b(int[] iArr, Object[] objArr, int i) {
        if (iArr.length == 8) {
            synchronized (c.class) {
                if (f1744g < 10) {
                    objArr[0] = f1743f;
                    objArr[1] = iArr;
                    for (int i2 = i - 1; i2 >= 2; i2--) {
                        objArr[i2] = null;
                    }
                    f1743f = objArr;
                    f1744g++;
                }
            }
        } else if (iArr.length == 4) {
            synchronized (c.class) {
                if (f1742e < 10) {
                    objArr[0] = f1741d;
                    objArr[1] = iArr;
                    for (int i3 = i - 1; i3 >= 2; i3--) {
                        objArr[i3] = null;
                    }
                    f1741d = objArr;
                    f1742e++;
                }
            }
        }
    }

    public final void a(int i) {
        if (i == 8) {
            synchronized (c.class) {
                Object[] objArr = f1743f;
                if (objArr != null) {
                    this.i = objArr;
                    f1743f = (Object[]) objArr[0];
                    this.f1745h = (int[]) objArr[1];
                    objArr[1] = null;
                    objArr[0] = null;
                    f1744g--;
                    return;
                }
            }
        } else if (i == 4) {
            synchronized (c.class) {
                Object[] objArr2 = f1741d;
                if (objArr2 != null) {
                    this.i = objArr2;
                    f1741d = (Object[]) objArr2[0];
                    this.f1745h = (int[]) objArr2[1];
                    objArr2[1] = null;
                    objArr2[0] = null;
                    f1742e--;
                    return;
                }
            }
        }
        this.f1745h = new int[i];
        this.i = new Object[i];
    }

    @Override // java.util.Collection, java.util.Set
    public boolean add(E e2) {
        int i;
        int c2;
        if (e2 == null) {
            c2 = d();
            i = 0;
        } else {
            int hashCode = e2.hashCode();
            i = hashCode;
            c2 = c(e2, hashCode);
        }
        if (c2 >= 0) {
            return false;
        }
        int i2 = ~c2;
        int i3 = this.j;
        int[] iArr = this.f1745h;
        if (i3 >= iArr.length) {
            int i4 = 4;
            if (i3 >= 8) {
                i4 = (i3 >> 1) + i3;
            } else if (i3 >= 4) {
                i4 = 8;
            }
            Object[] objArr = this.i;
            a(i4);
            int[] iArr2 = this.f1745h;
            if (iArr2.length > 0) {
                System.arraycopy(iArr, 0, iArr2, 0, iArr.length);
                System.arraycopy(objArr, 0, this.i, 0, objArr.length);
            }
            b(iArr, objArr, this.j);
        }
        int i5 = this.j;
        if (i2 < i5) {
            int[] iArr3 = this.f1745h;
            int i6 = i2 + 1;
            System.arraycopy(iArr3, i2, iArr3, i6, i5 - i2);
            Object[] objArr2 = this.i;
            System.arraycopy(objArr2, i2, objArr2, i6, this.j - i2);
        }
        this.f1745h[i2] = i;
        this.i[i2] = e2;
        this.j++;
        return true;
    }

    @Override // java.util.Collection, java.util.Set
    public boolean addAll(Collection<? extends E> collection) {
        int size = collection.size() + this.j;
        int[] iArr = this.f1745h;
        boolean z = false;
        if (iArr.length < size) {
            Object[] objArr = this.i;
            a(size);
            int i = this.j;
            if (i > 0) {
                System.arraycopy(iArr, 0, this.f1745h, 0, i);
                System.arraycopy(objArr, 0, this.i, 0, this.j);
            }
            b(iArr, objArr, this.j);
        }
        for (E e2 : collection) {
            z |= add(e2);
        }
        return z;
    }

    public final int c(Object obj, int i) {
        int i2 = this.j;
        if (i2 == 0) {
            return -1;
        }
        int a2 = d.a(this.f1745h, i2, i);
        if (a2 >= 0 && !obj.equals(this.i[a2])) {
            int i3 = a2 + 1;
            while (i3 < i2 && this.f1745h[i3] == i) {
                if (obj.equals(this.i[i3])) {
                    return i3;
                }
                i3++;
            }
            for (int i4 = a2 - 1; i4 >= 0 && this.f1745h[i4] == i; i4--) {
                if (obj.equals(this.i[i4])) {
                    return i4;
                }
            }
            return ~i3;
        }
        return a2;
    }

    @Override // java.util.Collection, java.util.Set
    public void clear() {
        int i = this.j;
        if (i != 0) {
            b(this.f1745h, this.i, i);
            this.f1745h = f1739b;
            this.i = f1740c;
            this.j = 0;
        }
    }

    @Override // java.util.Collection, java.util.Set
    public boolean contains(Object obj) {
        return indexOf(obj) >= 0;
    }

    @Override // java.util.Collection, java.util.Set
    public boolean containsAll(Collection<?> collection) {
        Iterator<?> it = collection.iterator();
        while (it.hasNext()) {
            if (!contains(it.next())) {
                return false;
            }
        }
        return true;
    }

    public final int d() {
        int i = this.j;
        if (i == 0) {
            return -1;
        }
        int a2 = d.a(this.f1745h, i, 0);
        if (a2 >= 0 && this.i[a2] != null) {
            int i2 = a2 + 1;
            while (i2 < i && this.f1745h[i2] == 0) {
                if (this.i[i2] == null) {
                    return i2;
                }
                i2++;
            }
            for (int i3 = a2 - 1; i3 >= 0 && this.f1745h[i3] == 0; i3--) {
                if (this.i[i3] == null) {
                    return i3;
                }
            }
            return ~i2;
        }
        return a2;
    }

    public E e(int i) {
        Object[] objArr = this.i;
        E e2 = (E) objArr[i];
        int i2 = this.j;
        if (i2 <= 1) {
            b(this.f1745h, objArr, i2);
            this.f1745h = f1739b;
            this.i = f1740c;
            this.j = 0;
        } else {
            int[] iArr = this.f1745h;
            if (iArr.length > 8 && i2 < iArr.length / 3) {
                a(i2 > 8 ? i2 + (i2 >> 1) : 8);
                this.j--;
                if (i > 0) {
                    System.arraycopy(iArr, 0, this.f1745h, 0, i);
                    System.arraycopy(objArr, 0, this.i, 0, i);
                }
                int i3 = this.j;
                if (i < i3) {
                    int i4 = i + 1;
                    System.arraycopy(iArr, i4, this.f1745h, i, i3 - i);
                    System.arraycopy(objArr, i4, this.i, i, this.j - i);
                }
            } else {
                int i5 = i2 - 1;
                this.j = i5;
                if (i < i5) {
                    int i6 = i + 1;
                    System.arraycopy(iArr, i6, iArr, i, i5 - i);
                    Object[] objArr2 = this.i;
                    System.arraycopy(objArr2, i6, objArr2, i, this.j - i);
                }
                this.i[this.j] = null;
            }
        }
        return e2;
    }

    @Override // java.util.Collection, java.util.Set
    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }
        if (obj instanceof Set) {
            Set set = (Set) obj;
            if (this.j != set.size()) {
                return false;
            }
            for (int i = 0; i < this.j; i++) {
                try {
                    if (!set.contains(this.i[i])) {
                        return false;
                    }
                } catch (ClassCastException | NullPointerException unused) {
                }
            }
            return true;
        }
        return false;
    }

    @Override // java.util.Collection, java.util.Set
    public int hashCode() {
        int[] iArr = this.f1745h;
        int i = this.j;
        int i2 = 0;
        for (int i3 = 0; i3 < i; i3++) {
            i2 += iArr[i3];
        }
        return i2;
    }

    public int indexOf(Object obj) {
        return obj == null ? d() : c(obj, obj.hashCode());
    }

    @Override // java.util.Collection, java.util.Set
    public boolean isEmpty() {
        return this.j <= 0;
    }

    @Override // java.util.Collection, java.lang.Iterable, java.util.Set
    public Iterator<E> iterator() {
        if (this.k == null) {
            this.k = new b(this);
        }
        g<E, E> gVar = this.k;
        if (gVar.f1755b == null) {
            gVar.f1755b = new g.c();
        }
        return gVar.f1755b.iterator();
    }

    @Override // java.util.Collection, java.util.Set
    public boolean remove(Object obj) {
        int indexOf = indexOf(obj);
        if (indexOf >= 0) {
            e(indexOf);
            return true;
        }
        return false;
    }

    @Override // java.util.Collection, java.util.Set
    public boolean removeAll(Collection<?> collection) {
        Iterator<?> it = collection.iterator();
        boolean z = false;
        while (it.hasNext()) {
            z |= remove(it.next());
        }
        return z;
    }

    @Override // java.util.Collection, java.util.Set
    public boolean retainAll(Collection<?> collection) {
        boolean z = false;
        for (int i = this.j - 1; i >= 0; i--) {
            if (!collection.contains(this.i[i])) {
                e(i);
                z = true;
            }
        }
        return z;
    }

    @Override // java.util.Collection, java.util.Set
    public int size() {
        return this.j;
    }

    @Override // java.util.Collection, java.util.Set
    public Object[] toArray() {
        int i = this.j;
        Object[] objArr = new Object[i];
        System.arraycopy(this.i, 0, objArr, 0, i);
        return objArr;
    }

    public String toString() {
        if (isEmpty()) {
            return "{}";
        }
        StringBuilder sb = new StringBuilder(this.j * 14);
        sb.append('{');
        for (int i = 0; i < this.j; i++) {
            if (i > 0) {
                sb.append(", ");
            }
            Object obj = this.i[i];
            if (obj != this) {
                sb.append(obj);
            } else {
                sb.append("(this Set)");
            }
        }
        sb.append('}');
        return sb.toString();
    }

    public c(int i) {
        if (i == 0) {
            this.f1745h = f1739b;
            this.i = f1740c;
        } else {
            a(i);
        }
        this.j = 0;
    }

    @Override // java.util.Collection, java.util.Set
    public <T> T[] toArray(T[] tArr) {
        if (tArr.length < this.j) {
            tArr = (T[]) ((Object[]) Array.newInstance(tArr.getClass().getComponentType(), this.j));
        }
        System.arraycopy(this.i, 0, tArr, 0, this.j);
        int length = tArr.length;
        int i = this.j;
        if (length > i) {
            tArr[i] = null;
        }
        return tArr;
    }
}