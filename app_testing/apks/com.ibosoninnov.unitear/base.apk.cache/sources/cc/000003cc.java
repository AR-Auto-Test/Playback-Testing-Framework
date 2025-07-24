package b.f;

/* compiled from: SparseArrayCompat.java */
/* loaded from: classes.dex */
public class i<E> implements Cloneable {

    /* renamed from: b  reason: collision with root package name */
    public static final Object f1776b = new Object();

    /* renamed from: c  reason: collision with root package name */
    public boolean f1777c;

    /* renamed from: d  reason: collision with root package name */
    public int[] f1778d;

    /* renamed from: e  reason: collision with root package name */
    public Object[] f1779e;

    /* renamed from: f  reason: collision with root package name */
    public int f1780f;

    public i() {
        this(10);
    }

    public void a(int i, E e2) {
        int i2 = this.f1780f;
        if (i2 != 0 && i <= this.f1778d[i2 - 1]) {
            g(i, e2);
            return;
        }
        if (this.f1777c && i2 >= this.f1778d.length) {
            c();
        }
        int i3 = this.f1780f;
        if (i3 >= this.f1778d.length) {
            int e3 = d.e(i3 + 1);
            int[] iArr = new int[e3];
            Object[] objArr = new Object[e3];
            int[] iArr2 = this.f1778d;
            System.arraycopy(iArr2, 0, iArr, 0, iArr2.length);
            Object[] objArr2 = this.f1779e;
            System.arraycopy(objArr2, 0, objArr, 0, objArr2.length);
            this.f1778d = iArr;
            this.f1779e = objArr;
        }
        this.f1778d[i3] = i;
        this.f1779e[i3] = e2;
        this.f1780f = i3 + 1;
    }

    /* JADX DEBUG: Method merged with bridge method */
    /* renamed from: b */
    public i<E> clone() {
        try {
            i<E> iVar = (i) super.clone();
            iVar.f1778d = (int[]) this.f1778d.clone();
            iVar.f1779e = (Object[]) this.f1779e.clone();
            return iVar;
        } catch (CloneNotSupportedException e2) {
            throw new AssertionError(e2);
        }
    }

    public final void c() {
        int i = this.f1780f;
        int[] iArr = this.f1778d;
        Object[] objArr = this.f1779e;
        int i2 = 0;
        for (int i3 = 0; i3 < i; i3++) {
            Object obj = objArr[i3];
            if (obj != f1776b) {
                if (i3 != i2) {
                    iArr[i2] = iArr[i3];
                    objArr[i2] = obj;
                    objArr[i3] = null;
                }
                i2++;
            }
        }
        this.f1777c = false;
        this.f1780f = i2;
    }

    public E d(int i) {
        return e(i, null);
    }

    public E e(int i, E e2) {
        int a2 = d.a(this.f1778d, this.f1780f, i);
        if (a2 >= 0) {
            Object[] objArr = this.f1779e;
            if (objArr[a2] != f1776b) {
                return (E) objArr[a2];
            }
        }
        return e2;
    }

    public int f(int i) {
        if (this.f1777c) {
            c();
        }
        return this.f1778d[i];
    }

    public void g(int i, E e2) {
        int a2 = d.a(this.f1778d, this.f1780f, i);
        if (a2 >= 0) {
            this.f1779e[a2] = e2;
            return;
        }
        int i2 = ~a2;
        int i3 = this.f1780f;
        if (i2 < i3) {
            Object[] objArr = this.f1779e;
            if (objArr[i2] == f1776b) {
                this.f1778d[i2] = i;
                objArr[i2] = e2;
                return;
            }
        }
        if (this.f1777c && i3 >= this.f1778d.length) {
            c();
            i2 = ~d.a(this.f1778d, this.f1780f, i);
        }
        int i4 = this.f1780f;
        if (i4 >= this.f1778d.length) {
            int e3 = d.e(i4 + 1);
            int[] iArr = new int[e3];
            Object[] objArr2 = new Object[e3];
            int[] iArr2 = this.f1778d;
            System.arraycopy(iArr2, 0, iArr, 0, iArr2.length);
            Object[] objArr3 = this.f1779e;
            System.arraycopy(objArr3, 0, objArr2, 0, objArr3.length);
            this.f1778d = iArr;
            this.f1779e = objArr2;
        }
        int i5 = this.f1780f;
        if (i5 - i2 != 0) {
            int[] iArr3 = this.f1778d;
            int i6 = i2 + 1;
            System.arraycopy(iArr3, i2, iArr3, i6, i5 - i2);
            Object[] objArr4 = this.f1779e;
            System.arraycopy(objArr4, i2, objArr4, i6, this.f1780f - i2);
        }
        this.f1778d[i2] = i;
        this.f1779e[i2] = e2;
        this.f1780f++;
    }

    public void h(int i) {
        int a2 = d.a(this.f1778d, this.f1780f, i);
        if (a2 >= 0) {
            Object[] objArr = this.f1779e;
            Object obj = objArr[a2];
            Object obj2 = f1776b;
            if (obj != obj2) {
                objArr[a2] = obj2;
                this.f1777c = true;
            }
        }
    }

    public int i() {
        if (this.f1777c) {
            c();
        }
        return this.f1780f;
    }

    public E j(int i) {
        if (this.f1777c) {
            c();
        }
        return (E) this.f1779e[i];
    }

    public String toString() {
        if (i() <= 0) {
            return "{}";
        }
        StringBuilder sb = new StringBuilder(this.f1780f * 28);
        sb.append('{');
        for (int i = 0; i < this.f1780f; i++) {
            if (i > 0) {
                sb.append(", ");
            }
            sb.append(f(i));
            sb.append('=');
            E j = j(i);
            if (j != this) {
                sb.append(j);
            } else {
                sb.append("(this Map)");
            }
        }
        sb.append('}');
        return sb.toString();
    }

    public i(int i) {
        this.f1777c = false;
        if (i == 0) {
            this.f1778d = d.f1746a;
            this.f1779e = d.f1748c;
            return;
        }
        int e2 = d.e(i);
        this.f1778d = new int[e2];
        this.f1779e = new Object[e2];
    }
}