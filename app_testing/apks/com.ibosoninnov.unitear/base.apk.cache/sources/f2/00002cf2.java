package f.g0;

/* compiled from: NamedRunnable.java */
/* loaded from: classes2.dex */
public abstract class b implements Runnable {

    /* renamed from: b  reason: collision with root package name */
    public final String f5772b;

    public b(String str, Object... objArr) {
        this.f5772b = c.n(str, objArr);
    }

    public abstract void a();

    @Override // java.lang.Runnable
    public final void run() {
        String name = Thread.currentThread().getName();
        Thread.currentThread().setName(this.f5772b);
        try {
            a();
        } finally {
            Thread.currentThread().setName(name);
        }
    }
}