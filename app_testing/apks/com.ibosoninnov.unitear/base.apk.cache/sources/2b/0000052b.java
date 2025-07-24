package b.q.b;

/* compiled from: FragmentViewLifecycleOwner.java */
/* loaded from: classes.dex */
public class l0 implements b.t.h {

    /* renamed from: b  reason: collision with root package name */
    public b.t.i f2487b = null;

    @Override // b.t.h
    public b.t.e getLifecycle() {
        if (this.f2487b == null) {
            this.f2487b = new b.t.i(this);
        }
        return this.f2487b;
    }
}