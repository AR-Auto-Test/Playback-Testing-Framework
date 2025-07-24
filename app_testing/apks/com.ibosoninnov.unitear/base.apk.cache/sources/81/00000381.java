package b.d.b;

/* compiled from: CameraX.java */
/* loaded from: classes.dex */
public class l0 implements b.d.b.d1.k1.c.d<Void> {

    /* renamed from: a  reason: collision with root package name */
    public final /* synthetic */ b.g.a.b f1635a;

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ n0 f1636b;

    public l0(b.g.a.b bVar, n0 n0Var) {
        this.f1635a = bVar;
        this.f1636b = n0Var;
    }

    @Override // b.d.b.d1.k1.c.d
    public void onFailure(Throwable th) {
        u0.d("CameraX", "CameraX initialize() failed", th);
        synchronized (n0.f1647a) {
            if (n0.f1648b == this.f1636b) {
                n0.f();
            }
        }
        this.f1635a.c(th);
    }

    /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object] */
    @Override // b.d.b.d1.k1.c.d
    public void onSuccess(Void r2) {
        this.f1635a.a(null);
    }
}