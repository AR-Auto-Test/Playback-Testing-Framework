package b.j.f;

/* compiled from: CancellationSignal.java */
/* loaded from: classes.dex */
public final class b {

    /* renamed from: a  reason: collision with root package name */
    public boolean f2117a;

    /* renamed from: b  reason: collision with root package name */
    public a f2118b;

    /* renamed from: c  reason: collision with root package name */
    public boolean f2119c;

    /* compiled from: CancellationSignal.java */
    /* loaded from: classes.dex */
    public interface a {
        void a();
    }

    public void a(a aVar) {
        synchronized (this) {
            while (this.f2119c) {
                try {
                    wait();
                } catch (InterruptedException unused) {
                }
            }
            if (this.f2118b == aVar) {
                return;
            }
            this.f2118b = aVar;
            if (this.f2117a) {
                aVar.a();
            }
        }
    }
}