package b.m;

import java.util.ArrayList;
import java.util.List;

/* compiled from: CallbackRegistry.java */
/* loaded from: classes.dex */
public class c<C, T, A> implements Cloneable {

    /* renamed from: b  reason: collision with root package name */
    public List<C> f2329b = new ArrayList();

    /* renamed from: c  reason: collision with root package name */
    public long f2330c = 0;

    /* renamed from: d  reason: collision with root package name */
    public long[] f2331d;

    /* renamed from: e  reason: collision with root package name */
    public int f2332e;

    /* renamed from: f  reason: collision with root package name */
    public final a<C, T, A> f2333f;

    /* compiled from: CallbackRegistry.java */
    /* loaded from: classes.dex */
    public static abstract class a<C, T, A> {
        public abstract void a(C c2, T t, int i, A a2);
    }

    public c(a<C, T, A> aVar) {
        this.f2333f = aVar;
    }

    public final boolean a(int i) {
        int i2;
        if (i < 64) {
            return ((1 << i) & this.f2330c) != 0;
        }
        long[] jArr = this.f2331d;
        if (jArr != null && (i2 = (i / 64) - 1) < jArr.length) {
            return ((1 << (i % 64)) & jArr[i2]) != 0;
        }
        return false;
    }

    public synchronized void b(T t, int i, A a2) {
        long[] jArr;
        this.f2332e++;
        int size = this.f2329b.size();
        int length = this.f2331d == null ? -1 : jArr.length - 1;
        d(t, i, null, length);
        c(t, i, null, (length + 2) * 64, size, 0L);
        int i2 = this.f2332e - 1;
        this.f2332e = i2;
        if (i2 == 0) {
            long[] jArr2 = this.f2331d;
            if (jArr2 != null) {
                for (int length2 = jArr2.length - 1; length2 >= 0; length2--) {
                    long j = this.f2331d[length2];
                    if (j != 0) {
                        e((length2 + 1) * 64, j);
                        this.f2331d[length2] = 0;
                    }
                }
            }
            long j2 = this.f2330c;
            if (j2 != 0) {
                e(0, j2);
                this.f2330c = 0L;
            }
        }
    }

    public final void c(T t, int i, A a2, int i2, int i3, long j) {
        long j2 = 1;
        while (i2 < i3) {
            if ((j & j2) == 0) {
                this.f2333f.a(this.f2329b.get(i2), t, i, a2);
            }
            j2 <<= 1;
            i2++;
        }
    }

    public Object clone() {
        c cVar;
        CloneNotSupportedException e2;
        synchronized (this) {
            try {
                cVar = (c) super.clone();
            } catch (CloneNotSupportedException e3) {
                cVar = null;
                e2 = e3;
            }
            try {
                cVar.f2330c = 0L;
                cVar.f2331d = null;
                cVar.f2332e = 0;
                cVar.f2329b = new ArrayList();
                int size = this.f2329b.size();
                for (int i = 0; i < size; i++) {
                    if (!a(i)) {
                        cVar.f2329b.add(this.f2329b.get(i));
                    }
                }
            } catch (CloneNotSupportedException e4) {
                e2 = e4;
                e2.printStackTrace();
                return cVar;
            }
        }
        return cVar;
    }

    public final void d(T t, int i, A a2, int i2) {
        if (i2 < 0) {
            c(t, i, a2, 0, Math.min(64, this.f2329b.size()), this.f2330c);
            return;
        }
        long j = this.f2331d[i2];
        int i3 = (i2 + 1) * 64;
        int min = Math.min(this.f2329b.size(), i3 + 64);
        d(t, i, a2, i2 - 1);
        c(t, i, a2, i3, min, j);
    }

    public final void e(int i, long j) {
        long j2 = Long.MIN_VALUE;
        for (int i2 = (i + 64) - 1; i2 >= i; i2--) {
            if ((j & j2) != 0) {
                this.f2329b.remove(i2);
            }
            j2 >>>= 1;
        }
    }

    public final void f(int i) {
        if (i < 64) {
            this.f2330c = (1 << i) | this.f2330c;
            return;
        }
        int i2 = (i / 64) - 1;
        long[] jArr = this.f2331d;
        if (jArr == null) {
            this.f2331d = new long[this.f2329b.size() / 64];
        } else if (jArr.length <= i2) {
            long[] jArr2 = new long[this.f2329b.size() / 64];
            long[] jArr3 = this.f2331d;
            System.arraycopy(jArr3, 0, jArr2, 0, jArr3.length);
            this.f2331d = jArr2;
        }
        long j = 1 << (i % 64);
        long[] jArr4 = this.f2331d;
        jArr4[i2] = j | jArr4[i2];
    }
}