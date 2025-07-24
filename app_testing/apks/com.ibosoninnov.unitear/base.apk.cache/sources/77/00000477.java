package b.j.i;

/* compiled from: Pools.java */
/* loaded from: classes.dex */
public class f<T> extends e<T> {

    /* renamed from: c  reason: collision with root package name */
    public final Object f2196c;

    public f(int i) {
        super(i);
        this.f2196c = new Object();
    }

    @Override // b.j.i.e, b.j.i.d
    public boolean a(T t) {
        boolean a2;
        synchronized (this.f2196c) {
            a2 = super.a(t);
        }
        return a2;
    }

    @Override // b.j.i.e, b.j.i.d
    public T b() {
        T t;
        synchronized (this.f2196c) {
            t = (T) super.b();
        }
        return t;
    }
}