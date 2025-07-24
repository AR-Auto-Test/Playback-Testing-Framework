package c.c.a.m.v;

import android.os.SystemClock;
import android.util.Log;
import c.c.a.m.v.g;
import c.c.a.m.w.n;
import java.util.Collections;
import java.util.List;

/* compiled from: SourceGenerator.java */
/* loaded from: classes.dex */
public class b0 implements g, g.a {

    /* renamed from: b  reason: collision with root package name */
    public final h<?> f3599b;

    /* renamed from: c  reason: collision with root package name */
    public final g.a f3600c;

    /* renamed from: d  reason: collision with root package name */
    public int f3601d;

    /* renamed from: e  reason: collision with root package name */
    public d f3602e;

    /* renamed from: f  reason: collision with root package name */
    public Object f3603f;

    /* renamed from: g  reason: collision with root package name */
    public volatile n.a<?> f3604g;

    /* renamed from: h  reason: collision with root package name */
    public e f3605h;

    public b0(h<?> hVar, g.a aVar) {
        this.f3599b = hVar;
        this.f3600c = aVar;
    }

    @Override // c.c.a.m.v.g.a
    public void a(c.c.a.m.m mVar, Exception exc, c.c.a.m.u.d<?> dVar, c.c.a.m.a aVar) {
        this.f3600c.a(mVar, exc, dVar, this.f3604g.f3865c.d());
    }

    @Override // c.c.a.m.v.g
    public boolean b() {
        Object obj = this.f3603f;
        if (obj != null) {
            this.f3603f = null;
            int i = c.c.a.s.f.f4187b;
            long elapsedRealtimeNanos = SystemClock.elapsedRealtimeNanos();
            try {
                c.c.a.m.d<X> e2 = this.f3599b.e(obj);
                f fVar = new f(e2, obj, this.f3599b.i);
                c.c.a.m.m mVar = this.f3604g.f3863a;
                h<?> hVar = this.f3599b;
                this.f3605h = new e(mVar, hVar.n);
                hVar.b().a(this.f3605h, fVar);
                if (Log.isLoggable("SourceGenerator", 2)) {
                    Log.v("SourceGenerator", "Finished encoding source to cache, key: " + this.f3605h + ", data: " + obj + ", encoder: " + e2 + ", duration: " + c.c.a.s.f.a(elapsedRealtimeNanos));
                }
                this.f3604g.f3865c.b();
                this.f3602e = new d(Collections.singletonList(this.f3604g.f3863a), this.f3599b, this);
            } catch (Throwable th) {
                this.f3604g.f3865c.b();
                throw th;
            }
        }
        d dVar = this.f3602e;
        if (dVar == null || !dVar.b()) {
            this.f3602e = null;
            this.f3604g = null;
            boolean z = false;
            while (!z) {
                if (!(this.f3601d < this.f3599b.c().size())) {
                    break;
                }
                List<n.a<?>> c2 = this.f3599b.c();
                int i2 = this.f3601d;
                this.f3601d = i2 + 1;
                this.f3604g = c2.get(i2);
                if (this.f3604g != null && (this.f3599b.p.c(this.f3604g.f3865c.d()) || this.f3599b.g(this.f3604g.f3865c.a()))) {
                    this.f3604g.f3865c.e(this.f3599b.o, new a0(this, this.f3604g));
                    z = true;
                }
            }
            return z;
        }
        return true;
    }

    @Override // c.c.a.m.v.g.a
    public void c() {
        throw new UnsupportedOperationException();
    }

    @Override // c.c.a.m.v.g
    public void cancel() {
        n.a<?> aVar = this.f3604g;
        if (aVar != null) {
            aVar.f3865c.cancel();
        }
    }

    @Override // c.c.a.m.v.g.a
    public void d(c.c.a.m.m mVar, Object obj, c.c.a.m.u.d<?> dVar, c.c.a.m.a aVar, c.c.a.m.m mVar2) {
        this.f3600c.d(mVar, obj, dVar, this.f3604g.f3865c.d(), mVar);
    }
}