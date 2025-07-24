package b.d.b.d1;

import java.util.HashMap;

/* compiled from: LiveDataObservable.java */
/* loaded from: classes.dex */
public final class r0<T> {

    /* renamed from: a  reason: collision with root package name */
    public final b.t.m<a<T>> f1586a = new b.t.m<>();

    /* compiled from: LiveDataObservable.java */
    /* loaded from: classes.dex */
    public static final class a<T> {

        /* renamed from: a  reason: collision with root package name */
        public T f1587a;

        public a(T t, Throwable th) {
            this.f1587a = t;
        }

        public String toString() {
            StringBuilder x = c.b.a.a.a.x("[Result: <");
            StringBuilder x2 = c.b.a.a.a.x("Value: ");
            x2.append(this.f1587a);
            x.append(x2.toString());
            x.append(">]");
            return x.toString();
        }
    }

    public r0() {
        new HashMap();
    }
}