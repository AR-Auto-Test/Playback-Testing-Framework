package g;

/* compiled from: ForwardingSource.java */
/* loaded from: classes2.dex */
public abstract class j implements x {

    /* renamed from: b  reason: collision with root package name */
    public final x f6184b;

    public j(x xVar) {
        if (xVar != null) {
            this.f6184b = xVar;
            return;
        }
        throw new IllegalArgumentException("delegate == null");
    }

    @Override // g.x
    public y b() {
        return this.f6184b.b();
    }

    public String toString() {
        return getClass().getSimpleName() + "(" + this.f6184b.toString() + ")";
    }
}