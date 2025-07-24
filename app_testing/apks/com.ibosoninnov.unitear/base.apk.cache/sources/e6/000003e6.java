package b.h.b;

/* compiled from: Pools.java */
/* loaded from: classes.dex */
public class f<T> {

    /* renamed from: a  reason: collision with root package name */
    public final Object[] f1837a;

    /* renamed from: b  reason: collision with root package name */
    public int f1838b;

    public f(int i) {
        if (i > 0) {
            this.f1837a = new Object[i];
            return;
        }
        throw new IllegalArgumentException("The max pool size must be > 0");
    }

    public T a() {
        int i = this.f1838b;
        if (i > 0) {
            int i2 = i - 1;
            Object[] objArr = this.f1837a;
            T t = (T) objArr[i2];
            objArr[i2] = null;
            this.f1838b = i - 1;
            return t;
        }
        return null;
    }

    public boolean b(T t) {
        int i = this.f1838b;
        Object[] objArr = this.f1837a;
        if (i < objArr.length) {
            objArr[i] = t;
            this.f1838b = i + 1;
            return true;
        }
        return false;
    }
}