package b.c.a.a;

/* compiled from: ArchTaskExecutor.java */
/* loaded from: classes.dex */
public class a extends c {

    /* renamed from: a  reason: collision with root package name */
    public static volatile a f984a;

    /* renamed from: b  reason: collision with root package name */
    public c f985b;

    /* renamed from: c  reason: collision with root package name */
    public c f986c;

    public a() {
        b bVar = new b();
        this.f986c = bVar;
        this.f985b = bVar;
    }

    public static a c() {
        if (f984a != null) {
            return f984a;
        }
        synchronized (a.class) {
            if (f984a == null) {
                f984a = new a();
            }
        }
        return f984a;
    }

    @Override // b.c.a.a.c
    public boolean a() {
        return this.f985b.a();
    }

    @Override // b.c.a.a.c
    public void b(Runnable runnable) {
        this.f985b.b(runnable);
    }
}