package c.c.a.m.v;

import c.c.a.s.k.a;
import c.c.a.s.k.d;
import java.util.Objects;

/* compiled from: LockedResource.java */
/* loaded from: classes.dex */
public final class v<Z> implements w<Z>, a.d {

    /* renamed from: b  reason: collision with root package name */
    public static final b.j.i.d<v<?>> f3800b = c.c.a.s.k.a.a(20, new a());

    /* renamed from: c  reason: collision with root package name */
    public final c.c.a.s.k.d f3801c = new d.b();

    /* renamed from: d  reason: collision with root package name */
    public w<Z> f3802d;

    /* renamed from: e  reason: collision with root package name */
    public boolean f3803e;

    /* renamed from: f  reason: collision with root package name */
    public boolean f3804f;

    /* compiled from: LockedResource.java */
    /* loaded from: classes.dex */
    public class a implements a.b<v<?>> {
        /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
        @Override // c.c.a.s.k.a.b
        public v<?> a() {
            return new v<>();
        }
    }

    public static <Z> v<Z> e(w<Z> wVar) {
        v<Z> vVar = (v<Z>) f3800b.b();
        Objects.requireNonNull(vVar, "Argument must not be null");
        vVar.f3804f = false;
        vVar.f3803e = true;
        vVar.f3802d = wVar;
        return vVar;
    }

    @Override // c.c.a.m.v.w
    public synchronized void a() {
        this.f3801c.a();
        this.f3804f = true;
        if (!this.f3803e) {
            this.f3802d.a();
            this.f3802d = null;
            f3800b.a(this);
        }
    }

    @Override // c.c.a.s.k.a.d
    public c.c.a.s.k.d b() {
        return this.f3801c;
    }

    @Override // c.c.a.m.v.w
    public int c() {
        return this.f3802d.c();
    }

    @Override // c.c.a.m.v.w
    public Class<Z> d() {
        return this.f3802d.d();
    }

    public synchronized void f() {
        this.f3801c.a();
        if (this.f3803e) {
            this.f3803e = false;
            if (this.f3804f) {
                a();
            }
        } else {
            throw new IllegalStateException("Already unlocked");
        }
    }

    @Override // c.c.a.m.v.w
    public Z get() {
        return this.f3802d.get();
    }
}