package b.j.i;

/* compiled from: Pools.java */
/* loaded from: classes.dex */
public class e<T> implements d<T> {

    /* renamed from: a  reason: collision with root package name */
    public final Object[] f2194a;

    /* renamed from: b  reason: collision with root package name */
    public int f2195b;

    public e(int i) {
        if (i > 0) {
            this.f2194a = new Object[i];
            return;
        }
        throw new IllegalArgumentException("The max pool size must be > 0");
    }

    @Override // b.j.i.d
    public boolean a(T t) {
        int i;
        boolean z;
        int i2 = 0;
        while (true) {
            i = this.f2195b;
            if (i2 >= i) {
                z = false;
                break;
            } else if (this.f2194a[i2] == t) {
                z = true;
                break;
            } else {
                i2++;
            }
        }
        if (!z) {
            Object[] objArr = this.f2194a;
            if (i < objArr.length) {
                objArr[i] = t;
                this.f2195b = i + 1;
                return true;
            }
            return false;
        }
        throw new IllegalStateException("Already in the pool!");
    }

    @Override // b.j.i.d
    public T b() {
        int i = this.f2195b;
        if (i > 0) {
            int i2 = i - 1;
            Object[] objArr = this.f2194a;
            T t = (T) objArr[i2];
            objArr[i2] = null;
            this.f2195b = i - 1;
            return t;
        }
        return null;
    }
}