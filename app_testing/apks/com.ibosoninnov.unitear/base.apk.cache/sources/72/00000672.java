package c.a.a;

import java.util.Arrays;

/* compiled from: LottieResult.java */
/* loaded from: classes.dex */
public final class p<V> {

    /* renamed from: a  reason: collision with root package name */
    public final V f3122a;

    /* renamed from: b  reason: collision with root package name */
    public final Throwable f3123b;

    public p(V v) {
        this.f3122a = v;
        this.f3123b = null;
    }

    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }
        if (obj instanceof p) {
            p pVar = (p) obj;
            V v = this.f3122a;
            if (v == null || !v.equals(pVar.f3122a)) {
                Throwable th = this.f3123b;
                if (th == null || pVar.f3123b == null) {
                    return false;
                }
                return th.toString().equals(this.f3123b.toString());
            }
            return true;
        }
        return false;
    }

    public int hashCode() {
        return Arrays.hashCode(new Object[]{this.f3122a, this.f3123b});
    }

    public p(Throwable th) {
        this.f3123b = th;
        this.f3122a = null;
    }
}