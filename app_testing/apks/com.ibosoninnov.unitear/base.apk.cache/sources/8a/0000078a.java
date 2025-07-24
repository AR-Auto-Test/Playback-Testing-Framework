package c.c.a.m.v;

import c.c.a.m.v.i;
import c.c.a.m.v.q;
import c.c.a.s.k.a;
import c.c.a.s.k.d;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.Executor;
import java.util.concurrent.atomic.AtomicInteger;

/* compiled from: EngineJob.java */
/* loaded from: classes.dex */
public class m<R> implements i.a<R>, a.d {

    /* renamed from: b  reason: collision with root package name */
    public static final c f3759b = new c();
    public boolean A;

    /* renamed from: c  reason: collision with root package name */
    public final e f3760c;

    /* renamed from: d  reason: collision with root package name */
    public final c.c.a.s.k.d f3761d;

    /* renamed from: e  reason: collision with root package name */
    public final q.a f3762e;

    /* renamed from: f  reason: collision with root package name */
    public final b.j.i.d<m<?>> f3763f;

    /* renamed from: g  reason: collision with root package name */
    public final c f3764g;

    /* renamed from: h  reason: collision with root package name */
    public final n f3765h;
    public final c.c.a.m.v.e0.a i;
    public final c.c.a.m.v.e0.a j;
    public final c.c.a.m.v.e0.a k;
    public final c.c.a.m.v.e0.a l;
    public final AtomicInteger m;
    public c.c.a.m.m n;
    public boolean o;
    public boolean p;
    public boolean q;
    public boolean r;
    public w<?> s;
    public c.c.a.m.a t;
    public boolean u;
    public r v;
    public boolean w;
    public q<?> x;
    public i<R> y;
    public volatile boolean z;

    /* compiled from: EngineJob.java */
    /* loaded from: classes.dex */
    public class a implements Runnable {

        /* renamed from: b  reason: collision with root package name */
        public final c.c.a.q.g f3766b;

        public a(c.c.a.q.g gVar) {
            this.f3766b = gVar;
        }

        @Override // java.lang.Runnable
        public void run() {
            c.c.a.q.h hVar = (c.c.a.q.h) this.f3766b;
            hVar.f4141c.a();
            synchronized (hVar.f4142d) {
                synchronized (m.this) {
                    if (m.this.f3760c.f3772b.contains(new d(this.f3766b, c.c.a.s.e.f4185b))) {
                        m mVar = m.this;
                        c.c.a.q.g gVar = this.f3766b;
                        Objects.requireNonNull(mVar);
                        ((c.c.a.q.h) gVar).n(mVar.v, 5);
                    }
                    m.this.d();
                }
            }
        }
    }

    /* compiled from: EngineJob.java */
    /* loaded from: classes.dex */
    public class b implements Runnable {

        /* renamed from: b  reason: collision with root package name */
        public final c.c.a.q.g f3768b;

        public b(c.c.a.q.g gVar) {
            this.f3768b = gVar;
        }

        @Override // java.lang.Runnable
        public void run() {
            c.c.a.q.h hVar = (c.c.a.q.h) this.f3768b;
            hVar.f4141c.a();
            synchronized (hVar.f4142d) {
                synchronized (m.this) {
                    if (m.this.f3760c.f3772b.contains(new d(this.f3768b, c.c.a.s.e.f4185b))) {
                        m.this.x.b();
                        m mVar = m.this;
                        c.c.a.q.g gVar = this.f3768b;
                        Objects.requireNonNull(mVar);
                        ((c.c.a.q.h) gVar).o(mVar.x, mVar.t, mVar.A);
                        m.this.h(this.f3768b);
                    }
                    m.this.d();
                }
            }
        }
    }

    /* compiled from: EngineJob.java */
    /* loaded from: classes.dex */
    public static class c {
    }

    /* compiled from: EngineJob.java */
    /* loaded from: classes.dex */
    public static final class d {

        /* renamed from: a  reason: collision with root package name */
        public final c.c.a.q.g f3770a;

        /* renamed from: b  reason: collision with root package name */
        public final Executor f3771b;

        public d(c.c.a.q.g gVar, Executor executor) {
            this.f3770a = gVar;
            this.f3771b = executor;
        }

        public boolean equals(Object obj) {
            if (obj instanceof d) {
                return this.f3770a.equals(((d) obj).f3770a);
            }
            return false;
        }

        public int hashCode() {
            return this.f3770a.hashCode();
        }
    }

    /* compiled from: EngineJob.java */
    /* loaded from: classes.dex */
    public static final class e implements Iterable<d> {

        /* renamed from: b  reason: collision with root package name */
        public final List<d> f3772b = new ArrayList(2);

        public boolean isEmpty() {
            return this.f3772b.isEmpty();
        }

        @Override // java.lang.Iterable
        public Iterator<d> iterator() {
            return this.f3772b.iterator();
        }
    }

    public m(c.c.a.m.v.e0.a aVar, c.c.a.m.v.e0.a aVar2, c.c.a.m.v.e0.a aVar3, c.c.a.m.v.e0.a aVar4, n nVar, q.a aVar5, b.j.i.d<m<?>> dVar) {
        c cVar = f3759b;
        this.f3760c = new e();
        this.f3761d = new d.b();
        this.m = new AtomicInteger();
        this.i = aVar;
        this.j = aVar2;
        this.k = aVar3;
        this.l = aVar4;
        this.f3765h = nVar;
        this.f3762e = aVar5;
        this.f3763f = dVar;
        this.f3764g = cVar;
    }

    public synchronized void a(c.c.a.q.g gVar, Executor executor) {
        this.f3761d.a();
        this.f3760c.f3772b.add(new d(gVar, executor));
        boolean z = true;
        if (this.u) {
            e(1);
            executor.execute(new b(gVar));
        } else if (this.w) {
            e(1);
            executor.execute(new a(gVar));
        } else {
            if (this.z) {
                z = false;
            }
            b.v.u.c.d(z, "Cannot add callbacks to a cancelled EngineJob");
        }
    }

    @Override // c.c.a.s.k.a.d
    public c.c.a.s.k.d b() {
        return this.f3761d;
    }

    public void c() {
        if (f()) {
            return;
        }
        this.z = true;
        i<R> iVar = this.y;
        iVar.F = true;
        g gVar = iVar.D;
        if (gVar != null) {
            gVar.cancel();
        }
        n nVar = this.f3765h;
        c.c.a.m.m mVar = this.n;
        l lVar = (l) nVar;
        synchronized (lVar) {
            t tVar = lVar.f3735b;
            Objects.requireNonNull(tVar);
            Map<c.c.a.m.m, m<?>> a2 = tVar.a(this.r);
            if (equals(a2.get(mVar))) {
                a2.remove(mVar);
            }
        }
    }

    public void d() {
        q<?> qVar;
        synchronized (this) {
            this.f3761d.a();
            b.v.u.c.d(f(), "Not yet complete!");
            int decrementAndGet = this.m.decrementAndGet();
            b.v.u.c.d(decrementAndGet >= 0, "Can't decrement below 0");
            if (decrementAndGet == 0) {
                qVar = this.x;
                g();
            } else {
                qVar = null;
            }
        }
        if (qVar != null) {
            qVar.e();
        }
    }

    public synchronized void e(int i) {
        q<?> qVar;
        b.v.u.c.d(f(), "Not yet complete!");
        if (this.m.getAndAdd(i) == 0 && (qVar = this.x) != null) {
            qVar.b();
        }
    }

    public final boolean f() {
        return this.w || this.u || this.z;
    }

    public final synchronized void g() {
        boolean a2;
        if (this.n != null) {
            this.f3760c.f3772b.clear();
            this.n = null;
            this.x = null;
            this.s = null;
            this.w = false;
            this.z = false;
            this.u = false;
            this.A = false;
            i<R> iVar = this.y;
            i.e eVar = iVar.f3706h;
            synchronized (eVar) {
                eVar.f3712a = true;
                a2 = eVar.a(false);
            }
            if (a2) {
                iVar.l();
            }
            this.y = null;
            this.v = null;
            this.t = null;
            this.f3763f.a(this);
        } else {
            throw new IllegalArgumentException();
        }
    }

    public synchronized void h(c.c.a.q.g gVar) {
        boolean z;
        this.f3761d.a();
        this.f3760c.f3772b.remove(new d(gVar, c.c.a.s.e.f4185b));
        if (this.f3760c.isEmpty()) {
            c();
            if (!this.u && !this.w) {
                z = false;
                if (z && this.m.get() == 0) {
                    g();
                }
            }
            z = true;
            if (z) {
                g();
            }
        }
    }

    public void i(i<?> iVar) {
        c.c.a.m.v.e0.a aVar;
        if (this.p) {
            aVar = this.k;
        } else {
            aVar = this.q ? this.l : this.j;
        }
        aVar.f3682d.execute(iVar);
    }
}