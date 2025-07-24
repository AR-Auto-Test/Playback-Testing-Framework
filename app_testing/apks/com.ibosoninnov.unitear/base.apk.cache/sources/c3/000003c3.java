package b.f;

/* compiled from: LongSparseArray.java */
/* loaded from: classes.dex */
public class e<E> implements Cloneable {

    /* renamed from: b  reason: collision with root package name */
    public static final Object f1749b = new Object();

    /* renamed from: c  reason: collision with root package name */
    public boolean f1750c;

    /* renamed from: d  reason: collision with root package name */
    public long[] f1751d;

    /* renamed from: e  reason: collision with root package name */
    public Object[] f1752e;

    /* renamed from: f  reason: collision with root package name */
    public int f1753f;

    public e() {
        this(10);
    }

    public void a() {
        int i = this.f1753f;
        Object[] objArr = this.f1752e;
        for (int i2 = 0; i2 < i; i2++) {
            objArr[i2] = null;
        }
        this.f1753f = 0;
        this.f1750c = false;
    }

    /* JADX DEBUG: Method merged with bridge method */
    /* renamed from: b */
    public e<E> clone() {
        try {
            e<E> eVar = (e) super.clone();
            eVar.f1751d = (long[]) this.f1751d.clone();
            eVar.f1752e = (Object[]) this.f1752e.clone();
            return eVar;
        } catch (CloneNotSupportedException e2) {
            throw new AssertionError(e2);
        }
    }

    public final void c() {
        int i = this.f1753f;
        long[] jArr = this.f1751d;
        Object[] objArr = this.f1752e;
        int i2 = 0;
        for (int i3 = 0; i3 < i; i3++) {
            Object obj = objArr[i3];
            if (obj != f1749b) {
                if (i3 != i2) {
                    jArr[i2] = jArr[i3];
                    objArr[i2] = obj;
                    objArr[i3] = null;
                }
                i2++;
            }
        }
        this.f1750c = false;
        this.f1753f = i2;
    }

    public E d(long j) {
        return e(j, null);
    }

    public E e(long j, E e2) {
        int b2 = d.b(this.f1751d, this.f1753f, j);
        if (b2 >= 0) {
            Object[] objArr = this.f1752e;
            if (objArr[b2] != f1749b) {
                return (E) objArr[b2];
            }
        }
        return e2;
    }

    public long f(int i) {
        if (this.f1750c) {
            c();
        }
        return this.f1751d[i];
    }

    public void g(long j, E e2) {
        int b2 = d.b(this.f1751d, this.f1753f, j);
        if (b2 >= 0) {
            this.f1752e[b2] = e2;
            return;
        }
        int i = ~b2;
        int i2 = this.f1753f;
        if (i < i2) {
            Object[] objArr = this.f1752e;
            if (objArr[i] == f1749b) {
                this.f1751d[i] = j;
                objArr[i] = e2;
                return;
            }
        }
        if (this.f1750c && i2 >= this.f1751d.length) {
            c();
            i = ~d.b(this.f1751d, this.f1753f, j);
        }
        int i3 = this.f1753f;
        if (i3 >= this.f1751d.length) {
            int f2 = d.f(i3 + 1);
            long[] jArr = new long[f2];
            Object[] objArr2 = new Object[f2];
            long[] jArr2 = this.f1751d;
            System.arraycopy(jArr2, 0, jArr, 0, jArr2.length);
            Object[] objArr3 = this.f1752e;
            System.arraycopy(objArr3, 0, objArr2, 0, objArr3.length);
            this.f1751d = jArr;
            this.f1752e = objArr2;
        }
        int i4 = this.f1753f;
        if (i4 - i != 0) {
            long[] jArr3 = this.f1751d;
            int i5 = i + 1;
            System.arraycopy(jArr3, i, jArr3, i5, i4 - i);
            Object[] objArr4 = this.f1752e;
            System.arraycopy(objArr4, i, objArr4, i5, this.f1753f - i);
        }
        this.f1751d[i] = j;
        this.f1752e[i] = e2;
        this.f1753f++;
    }

    public int h() {
        if (this.f1750c) {
            c();
        }
        return this.f1753f;
    }

    public E i(int i) {
        if (this.f1750c) {
            c();
        }
        return (E) this.f1752e[i];
    }

    public String toString() {
        if (h() <= 0) {
            return "{}";
        }
        StringBuilder sb = new StringBuilder(this.f1753f * 28);
        sb.append('{');
        for (int i = 0; i < this.f1753f; i++) {
            if (i > 0) {
                sb.append(", ");
            }
            sb.append(f(i));
            sb.append('=');
            E i2 = i(i);
            if (i2 != this) {
                sb.append(i2);
            } else {
                sb.append("(this Map)");
            }
        }
        sb.append('}');
        return sb.toString();
    }

    public e(int i) {
        this.f1750c = false;
        if (i == 0) {
            this.f1751d = d.f1747b;
            this.f1752e = d.f1748c;
            return;
        }
        int f2 = d.f(i);
        this.f1751d = new long[f2];
        this.f1752e = new Object[f2];
    }
}