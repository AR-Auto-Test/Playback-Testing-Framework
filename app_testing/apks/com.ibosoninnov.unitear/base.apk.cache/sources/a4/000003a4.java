package b.d.b;

/* compiled from: SettableImageProxy.java */
/* loaded from: classes.dex */
public final class y0 extends p0 {

    /* renamed from: d  reason: collision with root package name */
    public final q0 f1694d;

    /* renamed from: e  reason: collision with root package name */
    public final int f1695e;

    /* renamed from: f  reason: collision with root package name */
    public final int f1696f;

    public y0(r0 r0Var, q0 q0Var) {
        super(r0Var);
        int width;
        int height;
        synchronized (this) {
            width = this.f1662b.getWidth();
        }
        this.f1695e = width;
        synchronized (this) {
            height = this.f1662b.getHeight();
        }
        this.f1696f = height;
        this.f1694d = q0Var;
    }

    @Override // b.d.b.r0
    public synchronized int getHeight() {
        return this.f1696f;
    }

    @Override // b.d.b.r0
    public synchronized int getWidth() {
        return this.f1695e;
    }

    @Override // b.d.b.r0
    public q0 n() {
        return this.f1694d;
    }
}