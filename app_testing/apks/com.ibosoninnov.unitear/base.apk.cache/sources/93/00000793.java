package c.c.a.m.v;

import java.util.Objects;

/* compiled from: EngineResource.java */
/* loaded from: classes.dex */
public class q<Z> implements w<Z> {

    /* renamed from: b  reason: collision with root package name */
    public final boolean f3780b;

    /* renamed from: c  reason: collision with root package name */
    public final boolean f3781c;

    /* renamed from: d  reason: collision with root package name */
    public final w<Z> f3782d;

    /* renamed from: e  reason: collision with root package name */
    public final a f3783e;

    /* renamed from: f  reason: collision with root package name */
    public final c.c.a.m.m f3784f;

    /* renamed from: g  reason: collision with root package name */
    public int f3785g;

    /* renamed from: h  reason: collision with root package name */
    public boolean f3786h;

    /* compiled from: EngineResource.java */
    /* loaded from: classes.dex */
    public interface a {
        void a(c.c.a.m.m mVar, q<?> qVar);
    }

    public q(w<Z> wVar, boolean z, boolean z2, c.c.a.m.m mVar, a aVar) {
        Objects.requireNonNull(wVar, "Argument must not be null");
        this.f3782d = wVar;
        this.f3780b = z;
        this.f3781c = z2;
        this.f3784f = mVar;
        Objects.requireNonNull(aVar, "Argument must not be null");
        this.f3783e = aVar;
    }

    @Override // c.c.a.m.v.w
    public synchronized void a() {
        if (this.f3785g <= 0) {
            if (!this.f3786h) {
                this.f3786h = true;
                if (this.f3781c) {
                    this.f3782d.a();
                }
            } else {
                throw new IllegalStateException("Cannot recycle a resource that has already been recycled");
            }
        } else {
            throw new IllegalStateException("Cannot recycle a resource while it is still acquired");
        }
    }

    public synchronized void b() {
        if (!this.f3786h) {
            this.f3785g++;
        } else {
            throw new IllegalStateException("Cannot acquire a recycled resource");
        }
    }

    @Override // c.c.a.m.v.w
    public int c() {
        return this.f3782d.c();
    }

    @Override // c.c.a.m.v.w
    public Class<Z> d() {
        return this.f3782d.d();
    }

    public void e() {
        boolean z;
        synchronized (this) {
            int i = this.f3785g;
            if (i > 0) {
                z = true;
                int i2 = i - 1;
                this.f3785g = i2;
                if (i2 != 0) {
                    z = false;
                }
            } else {
                throw new IllegalStateException("Cannot release a recycled or not yet acquired resource");
            }
        }
        if (z) {
            this.f3783e.a(this.f3784f, this);
        }
    }

    @Override // c.c.a.m.v.w
    public Z get() {
        return this.f3782d.get();
    }

    public synchronized String toString() {
        return "EngineResource{isMemoryCacheable=" + this.f3780b + ", listener=" + this.f3783e + ", key=" + this.f3784f + ", acquired=" + this.f3785g + ", isRecycled=" + this.f3786h + ", resource=" + this.f3782d + '}';
    }
}