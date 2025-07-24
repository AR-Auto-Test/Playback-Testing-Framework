package c.c.a.m.v;

import android.os.SystemClock;
import android.util.Log;
import c.c.a.m.v.a;
import c.c.a.m.v.d0.a;
import c.c.a.m.v.d0.i;
import c.c.a.m.v.i;
import c.c.a.m.v.q;
import c.c.a.s.g;
import c.c.a.s.k.a;
import java.io.File;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.Executor;

/* compiled from: Engine.java */
/* loaded from: classes.dex */
public class l implements n, i.a, q.a {

    /* renamed from: a  reason: collision with root package name */
    public static final boolean f3734a = Log.isLoggable("Engine", 2);

    /* renamed from: b  reason: collision with root package name */
    public final t f3735b;

    /* renamed from: c  reason: collision with root package name */
    public final p f3736c;

    /* renamed from: d  reason: collision with root package name */
    public final c.c.a.m.v.d0.i f3737d;

    /* renamed from: e  reason: collision with root package name */
    public final b f3738e;

    /* renamed from: f  reason: collision with root package name */
    public final z f3739f;

    /* renamed from: g  reason: collision with root package name */
    public final c f3740g;

    /* renamed from: h  reason: collision with root package name */
    public final a f3741h;
    public final c.c.a.m.v.a i;

    /* compiled from: Engine.java */
    /* loaded from: classes.dex */
    public static class a {

        /* renamed from: a  reason: collision with root package name */
        public final i.d f3742a;

        /* renamed from: b  reason: collision with root package name */
        public final b.j.i.d<i<?>> f3743b = c.c.a.s.k.a.a(150, new C0072a());

        /* renamed from: c  reason: collision with root package name */
        public int f3744c;

        /* compiled from: Engine.java */
        /* renamed from: c.c.a.m.v.l$a$a  reason: collision with other inner class name */
        /* loaded from: classes.dex */
        public class C0072a implements a.b<i<?>> {
            public C0072a() {
            }

            /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
            @Override // c.c.a.s.k.a.b
            public i<?> a() {
                a aVar = a.this;
                return new i<>(aVar.f3742a, aVar.f3743b);
            }
        }

        public a(i.d dVar) {
            this.f3742a = dVar;
        }
    }

    /* compiled from: Engine.java */
    /* loaded from: classes.dex */
    public static class b {

        /* renamed from: a  reason: collision with root package name */
        public final c.c.a.m.v.e0.a f3746a;

        /* renamed from: b  reason: collision with root package name */
        public final c.c.a.m.v.e0.a f3747b;

        /* renamed from: c  reason: collision with root package name */
        public final c.c.a.m.v.e0.a f3748c;

        /* renamed from: d  reason: collision with root package name */
        public final c.c.a.m.v.e0.a f3749d;

        /* renamed from: e  reason: collision with root package name */
        public final n f3750e;

        /* renamed from: f  reason: collision with root package name */
        public final q.a f3751f;

        /* renamed from: g  reason: collision with root package name */
        public final b.j.i.d<m<?>> f3752g = c.c.a.s.k.a.a(150, new a());

        /* compiled from: Engine.java */
        /* loaded from: classes.dex */
        public class a implements a.b<m<?>> {
            public a() {
            }

            /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
            @Override // c.c.a.s.k.a.b
            public m<?> a() {
                b bVar = b.this;
                return new m<>(bVar.f3746a, bVar.f3747b, bVar.f3748c, bVar.f3749d, bVar.f3750e, bVar.f3751f, bVar.f3752g);
            }
        }

        public b(c.c.a.m.v.e0.a aVar, c.c.a.m.v.e0.a aVar2, c.c.a.m.v.e0.a aVar3, c.c.a.m.v.e0.a aVar4, n nVar, q.a aVar5) {
            this.f3746a = aVar;
            this.f3747b = aVar2;
            this.f3748c = aVar3;
            this.f3749d = aVar4;
            this.f3750e = nVar;
            this.f3751f = aVar5;
        }
    }

    /* compiled from: Engine.java */
    /* loaded from: classes.dex */
    public static class c implements i.d {

        /* renamed from: a  reason: collision with root package name */
        public final a.InterfaceC0068a f3754a;

        /* renamed from: b  reason: collision with root package name */
        public volatile c.c.a.m.v.d0.a f3755b;

        public c(a.InterfaceC0068a interfaceC0068a) {
            this.f3754a = interfaceC0068a;
        }

        public c.c.a.m.v.d0.a a() {
            if (this.f3755b == null) {
                synchronized (this) {
                    if (this.f3755b == null) {
                        c.c.a.m.v.d0.d dVar = (c.c.a.m.v.d0.d) this.f3754a;
                        c.c.a.m.v.d0.f fVar = (c.c.a.m.v.d0.f) dVar.f3655b;
                        File cacheDir = fVar.f3661a.getCacheDir();
                        c.c.a.m.v.d0.e eVar = null;
                        if (cacheDir == null) {
                            cacheDir = null;
                        } else if (fVar.f3662b != null) {
                            cacheDir = new File(cacheDir, fVar.f3662b);
                        }
                        if (cacheDir != null && (cacheDir.isDirectory() || cacheDir.mkdirs())) {
                            eVar = new c.c.a.m.v.d0.e(cacheDir, dVar.f3654a);
                        }
                        this.f3755b = eVar;
                    }
                    if (this.f3755b == null) {
                        this.f3755b = new c.c.a.m.v.d0.b();
                    }
                }
            }
            return this.f3755b;
        }
    }

    /* compiled from: Engine.java */
    /* loaded from: classes.dex */
    public class d {

        /* renamed from: a  reason: collision with root package name */
        public final m<?> f3756a;

        /* renamed from: b  reason: collision with root package name */
        public final c.c.a.q.g f3757b;

        public d(c.c.a.q.g gVar, m<?> mVar) {
            this.f3757b = gVar;
            this.f3756a = mVar;
        }
    }

    public l(c.c.a.m.v.d0.i iVar, a.InterfaceC0068a interfaceC0068a, c.c.a.m.v.e0.a aVar, c.c.a.m.v.e0.a aVar2, c.c.a.m.v.e0.a aVar3, c.c.a.m.v.e0.a aVar4, boolean z) {
        this.f3737d = iVar;
        c cVar = new c(interfaceC0068a);
        this.f3740g = cVar;
        c.c.a.m.v.a aVar5 = new c.c.a.m.v.a(z);
        this.i = aVar5;
        synchronized (this) {
            synchronized (aVar5) {
                aVar5.f3591e = this;
            }
        }
        this.f3736c = new p();
        this.f3735b = new t();
        this.f3738e = new b(aVar, aVar2, aVar3, aVar4, this, this);
        this.f3741h = new a(cVar);
        this.f3739f = new z();
        ((c.c.a.m.v.d0.h) iVar).f3663d = this;
    }

    public static void d(String str, long j, c.c.a.m.m mVar) {
        StringBuilder A = c.b.a.a.a.A(str, " in ");
        A.append(c.c.a.s.f.a(j));
        A.append("ms, key: ");
        A.append(mVar);
        Log.v("Engine", A.toString());
    }

    @Override // c.c.a.m.v.q.a
    public void a(c.c.a.m.m mVar, q<?> qVar) {
        c.c.a.m.v.a aVar = this.i;
        synchronized (aVar) {
            a.b remove = aVar.f3589c.remove(mVar);
            if (remove != null) {
                remove.f3595c = null;
                remove.clear();
            }
        }
        if (qVar.f3780b) {
            ((c.c.a.m.v.d0.h) this.f3737d).d(mVar, qVar);
        } else {
            this.f3739f.a(qVar, false);
        }
    }

    public <R> d b(c.c.a.d dVar, Object obj, c.c.a.m.m mVar, int i, int i2, Class<?> cls, Class<R> cls2, c.c.a.f fVar, k kVar, Map<Class<?>, c.c.a.m.t<?>> map, boolean z, boolean z2, c.c.a.m.p pVar, boolean z3, boolean z4, boolean z5, boolean z6, c.c.a.q.g gVar, Executor executor) {
        long j;
        if (f3734a) {
            int i3 = c.c.a.s.f.f4187b;
            j = SystemClock.elapsedRealtimeNanos();
        } else {
            j = 0;
        }
        long j2 = j;
        Objects.requireNonNull(this.f3736c);
        o oVar = new o(obj, mVar, i, i2, map, cls, cls2, pVar);
        synchronized (this) {
            q<?> c2 = c(oVar, z3, j2);
            if (c2 == null) {
                return g(dVar, obj, mVar, i, i2, cls, cls2, fVar, kVar, map, z, z2, pVar, z3, z4, z5, z6, gVar, executor, oVar, j2);
            }
            ((c.c.a.q.h) gVar).o(c2, c.c.a.m.a.MEMORY_CACHE, false);
            return null;
        }
    }

    /* JADX DEBUG: Multi-variable search result rejected for r1v6, resolved type: Y */
    /* JADX WARN: Multi-variable type inference failed */
    public final q<?> c(o oVar, boolean z, long j) {
        q<?> qVar;
        w wVar;
        q<?> qVar2;
        if (z) {
            c.c.a.m.v.a aVar = this.i;
            synchronized (aVar) {
                a.b bVar = aVar.f3589c.get(oVar);
                if (bVar == null) {
                    qVar = null;
                } else {
                    qVar = bVar.get();
                    if (qVar == null) {
                        aVar.b(bVar);
                    }
                }
            }
            if (qVar != null) {
                qVar.b();
            }
            if (qVar != null) {
                if (f3734a) {
                    d("Loaded resource from active resources", j, oVar);
                }
                return qVar;
            }
            c.c.a.m.v.d0.h hVar = (c.c.a.m.v.d0.h) this.f3737d;
            synchronized (hVar) {
                g.a aVar2 = (g.a) hVar.f4188a.remove(oVar);
                if (aVar2 == null) {
                    wVar = null;
                } else {
                    hVar.f4190c -= aVar2.f4192b;
                    wVar = aVar2.f4191a;
                }
            }
            w wVar2 = wVar;
            if (wVar2 == null) {
                qVar2 = null;
            } else if (wVar2 instanceof q) {
                qVar2 = (q) wVar2;
            } else {
                qVar2 = new q<>(wVar2, true, true, oVar, this);
            }
            if (qVar2 != null) {
                qVar2.b();
                this.i.a(oVar, qVar2);
            }
            if (qVar2 != null) {
                if (f3734a) {
                    d("Loaded resource from cache", j, oVar);
                }
                return qVar2;
            }
            return null;
        }
        return null;
    }

    public synchronized void e(m<?> mVar, c.c.a.m.m mVar2, q<?> qVar) {
        if (qVar != null) {
            if (qVar.f3780b) {
                this.i.a(mVar2, qVar);
            }
        }
        t tVar = this.f3735b;
        Objects.requireNonNull(tVar);
        Map<c.c.a.m.m, m<?>> a2 = tVar.a(mVar.r);
        if (mVar.equals(a2.get(mVar2))) {
            a2.remove(mVar2);
        }
    }

    public void f(w<?> wVar) {
        if (wVar instanceof q) {
            ((q) wVar).e();
            return;
        }
        throw new IllegalArgumentException("Cannot release anything but an EngineResource");
    }

    /* JADX DEBUG: Multi-variable search result rejected for r23v0, resolved type: java.lang.Class<R> */
    /* JADX WARN: Multi-variable type inference failed */
    /* JADX WARN: Removed duplicated region for block: B:27:0x00e9 A[Catch: all -> 0x0113, TryCatch #0 {, blocks: (B:19:0x00d3, B:21:0x00df, B:27:0x00e9, B:35:0x00fc, B:28:0x00ec, B:30:0x00f0, B:31:0x00f3, B:33:0x00f7, B:34:0x00fa), top: B:48:0x00d3 }] */
    /* JADX WARN: Removed duplicated region for block: B:28:0x00ec A[Catch: all -> 0x0113, TryCatch #0 {, blocks: (B:19:0x00d3, B:21:0x00df, B:27:0x00e9, B:35:0x00fc, B:28:0x00ec, B:30:0x00f0, B:31:0x00f3, B:33:0x00f7, B:34:0x00fa), top: B:48:0x00d3 }] */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public final <R> d g(c.c.a.d dVar, Object obj, c.c.a.m.m mVar, int i, int i2, Class<?> cls, Class<R> cls2, c.c.a.f fVar, k kVar, Map<Class<?>, c.c.a.m.t<?>> map, boolean z, boolean z2, c.c.a.m.p pVar, boolean z3, boolean z4, boolean z5, boolean z6, c.c.a.q.g gVar, Executor executor, o oVar, long j) {
        boolean z7;
        c.c.a.m.v.e0.a aVar;
        t tVar = this.f3735b;
        m<?> mVar2 = (z6 ? tVar.f3796b : tVar.f3795a).get(oVar);
        if (mVar2 != null) {
            mVar2.a(gVar, executor);
            if (f3734a) {
                d("Added to existing load", j, oVar);
            }
            return new d(gVar, mVar2);
        }
        m<?> b2 = this.f3738e.f3752g.b();
        Objects.requireNonNull(b2, "Argument must not be null");
        synchronized (b2) {
            b2.n = oVar;
            b2.o = z3;
            b2.p = z4;
            b2.q = z5;
            b2.r = z6;
        }
        a aVar2 = this.f3741h;
        i<R> iVar = (i<R>) aVar2.f3743b.b();
        Objects.requireNonNull(iVar, "Argument must not be null");
        int i3 = aVar2.f3744c;
        aVar2.f3744c = i3 + 1;
        h<R> hVar = iVar.f3700b;
        i.d dVar2 = iVar.f3703e;
        hVar.f3694c = dVar;
        hVar.f3695d = obj;
        hVar.n = mVar;
        hVar.f3696e = i;
        hVar.f3697f = i2;
        hVar.p = kVar;
        hVar.f3698g = cls;
        hVar.f3699h = dVar2;
        hVar.k = cls2;
        hVar.o = fVar;
        hVar.i = pVar;
        hVar.j = map;
        hVar.q = z;
        hVar.r = z2;
        iVar.i = dVar;
        iVar.j = mVar;
        iVar.k = fVar;
        iVar.l = oVar;
        iVar.m = i;
        iVar.n = i2;
        iVar.o = kVar;
        iVar.v = z6;
        iVar.p = pVar;
        iVar.q = b2;
        iVar.r = i3;
        iVar.t = i.f.INITIALIZE;
        iVar.w = obj;
        t tVar2 = this.f3735b;
        Objects.requireNonNull(tVar2);
        tVar2.a(b2.r).put(oVar, b2);
        b2.a(gVar, executor);
        synchronized (b2) {
            b2.y = iVar;
            i.g i4 = iVar.i(i.g.INITIALIZE);
            if (i4 != i.g.RESOURCE_CACHE && i4 != i.g.DATA_CACHE) {
                z7 = false;
                if (!z7) {
                    aVar = b2.i;
                } else if (b2.p) {
                    aVar = b2.k;
                } else {
                    aVar = b2.q ? b2.l : b2.j;
                }
                aVar.f3682d.execute(iVar);
            }
            z7 = true;
            if (!z7) {
            }
            aVar.f3682d.execute(iVar);
        }
        if (f3734a) {
            d("Started new load", j, oVar);
        }
        return new d(gVar, b2);
    }
}