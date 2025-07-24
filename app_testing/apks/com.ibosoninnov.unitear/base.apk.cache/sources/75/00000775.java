package c.c.a.m.v;

import android.os.Build;
import android.os.SystemClock;
import android.util.Log;
import c.c.a.m.u.e;
import c.c.a.m.v.g;
import c.c.a.m.v.j;
import c.c.a.m.v.l;
import c.c.a.m.v.m;
import c.c.a.m.v.q;
import c.c.a.s.k.a;
import c.c.a.s.k.d;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Objects;

/* compiled from: DecodeJob.java */
/* loaded from: classes.dex */
public class i<R> implements g.a, Runnable, Comparable<i<?>>, a.d {
    public Object A;
    public c.c.a.m.a B;
    public c.c.a.m.u.d<?> C;
    public volatile c.c.a.m.v.g D;
    public volatile boolean E;
    public volatile boolean F;
    public boolean G;

    /* renamed from: e  reason: collision with root package name */
    public final d f3703e;

    /* renamed from: f  reason: collision with root package name */
    public final b.j.i.d<i<?>> f3704f;
    public c.c.a.d i;
    public c.c.a.m.m j;
    public c.c.a.f k;
    public o l;
    public int m;
    public int n;
    public k o;
    public c.c.a.m.p p;
    public a<R> q;
    public int r;
    public g s;
    public f t;
    public long u;
    public boolean v;
    public Object w;
    public Thread x;
    public c.c.a.m.m y;
    public c.c.a.m.m z;

    /* renamed from: b  reason: collision with root package name */
    public final h<R> f3700b = new h<>();

    /* renamed from: c  reason: collision with root package name */
    public final List<Throwable> f3701c = new ArrayList();

    /* renamed from: d  reason: collision with root package name */
    public final c.c.a.s.k.d f3702d = new d.b();

    /* renamed from: g  reason: collision with root package name */
    public final c<?> f3705g = new c<>();

    /* renamed from: h  reason: collision with root package name */
    public final e f3706h = new e();

    /* compiled from: DecodeJob.java */
    /* loaded from: classes.dex */
    public interface a<R> {
    }

    /* compiled from: DecodeJob.java */
    /* loaded from: classes.dex */
    public final class b<Z> implements j.a<Z> {

        /* renamed from: a  reason: collision with root package name */
        public final c.c.a.m.a f3707a;

        public b(c.c.a.m.a aVar) {
            this.f3707a = aVar;
        }
    }

    /* compiled from: DecodeJob.java */
    /* loaded from: classes.dex */
    public static class c<Z> {

        /* renamed from: a  reason: collision with root package name */
        public c.c.a.m.m f3709a;

        /* renamed from: b  reason: collision with root package name */
        public c.c.a.m.s<Z> f3710b;

        /* renamed from: c  reason: collision with root package name */
        public v<Z> f3711c;
    }

    /* compiled from: DecodeJob.java */
    /* loaded from: classes.dex */
    public interface d {
    }

    /* compiled from: DecodeJob.java */
    /* loaded from: classes.dex */
    public static class e {

        /* renamed from: a  reason: collision with root package name */
        public boolean f3712a;

        /* renamed from: b  reason: collision with root package name */
        public boolean f3713b;

        /* renamed from: c  reason: collision with root package name */
        public boolean f3714c;

        public final boolean a(boolean z) {
            return (this.f3714c || z || this.f3713b) && this.f3712a;
        }
    }

    /* compiled from: DecodeJob.java */
    /* loaded from: classes.dex */
    public enum f {
        INITIALIZE,
        SWITCH_TO_SOURCE_SERVICE,
        DECODE_DATA
    }

    /* compiled from: DecodeJob.java */
    /* loaded from: classes.dex */
    public enum g {
        INITIALIZE,
        RESOURCE_CACHE,
        DATA_CACHE,
        SOURCE,
        ENCODE,
        FINISHED
    }

    public i(d dVar, b.j.i.d<i<?>> dVar2) {
        this.f3703e = dVar;
        this.f3704f = dVar2;
    }

    @Override // c.c.a.m.v.g.a
    public void a(c.c.a.m.m mVar, Exception exc, c.c.a.m.u.d<?> dVar, c.c.a.m.a aVar) {
        dVar.b();
        r rVar = new r("Fetching data failed", exc);
        Class<?> a2 = dVar.a();
        rVar.f3789d = mVar;
        rVar.f3790e = aVar;
        rVar.f3791f = a2;
        this.f3701c.add(rVar);
        if (Thread.currentThread() != this.x) {
            this.t = f.SWITCH_TO_SOURCE_SERVICE;
            ((m) this.q).i(this);
            return;
        }
        m();
    }

    @Override // c.c.a.s.k.a.d
    public c.c.a.s.k.d b() {
        return this.f3702d;
    }

    @Override // c.c.a.m.v.g.a
    public void c() {
        this.t = f.SWITCH_TO_SOURCE_SERVICE;
        ((m) this.q).i(this);
    }

    /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object] */
    @Override // java.lang.Comparable
    public int compareTo(i<?> iVar) {
        i<?> iVar2 = iVar;
        int ordinal = this.k.ordinal() - iVar2.k.ordinal();
        return ordinal == 0 ? this.r - iVar2.r : ordinal;
    }

    @Override // c.c.a.m.v.g.a
    public void d(c.c.a.m.m mVar, Object obj, c.c.a.m.u.d<?> dVar, c.c.a.m.a aVar, c.c.a.m.m mVar2) {
        this.y = mVar;
        this.A = obj;
        this.C = dVar;
        this.B = aVar;
        this.z = mVar2;
        this.G = mVar != this.f3700b.a().get(0);
        if (Thread.currentThread() != this.x) {
            this.t = f.DECODE_DATA;
            ((m) this.q).i(this);
            return;
        }
        g();
    }

    public final <Data> w<R> e(c.c.a.m.u.d<?> dVar, Data data, c.c.a.m.a aVar) {
        if (data == null) {
            return null;
        }
        try {
            int i = c.c.a.s.f.f4187b;
            long elapsedRealtimeNanos = SystemClock.elapsedRealtimeNanos();
            w<R> f2 = f(data, aVar);
            if (Log.isLoggable("DecodeJob", 2)) {
                j("Decoded result " + f2, elapsedRealtimeNanos, null);
            }
            return f2;
        } finally {
            dVar.b();
        }
    }

    public final <Data> w<R> f(Data data, c.c.a.m.a aVar) {
        c.c.a.m.u.e<Data> b2;
        u<Data, ?, R> d2 = this.f3700b.d(data.getClass());
        c.c.a.m.p pVar = this.p;
        if (Build.VERSION.SDK_INT >= 26) {
            boolean z = aVar == c.c.a.m.a.RESOURCE_DISK_CACHE || this.f3700b.r;
            c.c.a.m.o<Boolean> oVar = c.c.a.m.x.c.m.f3979d;
            Boolean bool = (Boolean) pVar.c(oVar);
            if (bool == null || (bool.booleanValue() && !z)) {
                pVar = new c.c.a.m.p();
                pVar.d(this.p);
                pVar.f3544b.put(oVar, Boolean.valueOf(z));
            }
        }
        c.c.a.m.p pVar2 = pVar;
        c.c.a.m.u.f fVar = this.i.f3427c.f3444e;
        synchronized (fVar) {
            e.a<?> aVar2 = fVar.f3556b.get(data.getClass());
            if (aVar2 == null) {
                Iterator<e.a<?>> it = fVar.f3556b.values().iterator();
                while (true) {
                    if (!it.hasNext()) {
                        break;
                    }
                    e.a<?> next = it.next();
                    if (next.a().isAssignableFrom(data.getClass())) {
                        aVar2 = next;
                        break;
                    }
                }
            }
            if (aVar2 == null) {
                aVar2 = c.c.a.m.u.f.f3555a;
            }
            b2 = aVar2.b(data);
        }
        try {
            return d2.a(b2, pVar2, this.m, this.n, new b(aVar));
        } finally {
            b2.b();
        }
    }

    /* JADX DEBUG: Another duplicated slice has different insns count: {[IF]}, finally: {[IF, INVOKE] complete} */
    /* JADX DEBUG: Failed to insert an additional move for type inference into block B:18:0x006b */
    /* JADX DEBUG: Type inference failed for r1v2. Raw type applied. Possible types: c.c.a.m.v.w<R> */
    /* JADX WARN: Multi-variable type inference failed */
    public final void g() {
        w wVar;
        boolean a2;
        if (Log.isLoggable("DecodeJob", 2)) {
            long j = this.u;
            StringBuilder x = c.b.a.a.a.x("data: ");
            x.append(this.A);
            x.append(", cache key: ");
            x.append(this.y);
            x.append(", fetcher: ");
            x.append(this.C);
            j("Retrieved data", j, x.toString());
        }
        v vVar = null;
        try {
            wVar = (w<R>) e(this.C, this.A, this.B);
        } catch (r e2) {
            c.c.a.m.m mVar = this.z;
            c.c.a.m.a aVar = this.B;
            e2.f3789d = mVar;
            e2.f3790e = aVar;
            e2.f3791f = null;
            this.f3701c.add(e2);
            wVar = (w<R>) null;
        }
        if (wVar != null) {
            c.c.a.m.a aVar2 = this.B;
            boolean z = this.G;
            if (wVar instanceof s) {
                ((s) wVar).initialize();
            }
            if (this.f3705g.f3711c != null) {
                vVar = v.e(wVar);
                wVar = vVar;
            }
            o();
            m<?> mVar2 = (m) this.q;
            synchronized (mVar2) {
                mVar2.s = wVar;
                mVar2.t = aVar2;
                mVar2.A = z;
            }
            synchronized (mVar2) {
                mVar2.f3761d.a();
                if (mVar2.z) {
                    mVar2.s.a();
                    mVar2.g();
                } else if (!mVar2.f3760c.isEmpty()) {
                    if (!mVar2.u) {
                        m.c cVar = mVar2.f3764g;
                        w<?> wVar2 = mVar2.s;
                        boolean z2 = mVar2.o;
                        c.c.a.m.m mVar3 = mVar2.n;
                        q.a aVar3 = mVar2.f3762e;
                        Objects.requireNonNull(cVar);
                        mVar2.x = new q<>(wVar2, z2, true, mVar3, aVar3);
                        mVar2.u = true;
                        m.e eVar = mVar2.f3760c;
                        Objects.requireNonNull(eVar);
                        ArrayList arrayList = new ArrayList(eVar.f3772b);
                        mVar2.e(arrayList.size() + 1);
                        ((l) mVar2.f3765h).e(mVar2, mVar2.n, mVar2.x);
                        Iterator it = arrayList.iterator();
                        while (it.hasNext()) {
                            m.d dVar = (m.d) it.next();
                            dVar.f3771b.execute(new m.b(dVar.f3770a));
                        }
                        mVar2.d();
                    } else {
                        throw new IllegalStateException("Already have resource");
                    }
                } else {
                    throw new IllegalStateException("Received a resource without any callbacks to notify");
                }
            }
            this.s = g.ENCODE;
            try {
                c<?> cVar2 = this.f3705g;
                if (cVar2.f3711c != null) {
                    ((l.c) this.f3703e).a().a(cVar2.f3709a, new c.c.a.m.v.f(cVar2.f3710b, cVar2.f3711c, this.p));
                    cVar2.f3711c.f();
                }
                e eVar2 = this.f3706h;
                synchronized (eVar2) {
                    eVar2.f3713b = true;
                    a2 = eVar2.a(false);
                }
                if (a2) {
                    l();
                    return;
                }
                return;
            } finally {
                if (vVar != null) {
                    vVar.f();
                }
            }
        }
        m();
    }

    public final c.c.a.m.v.g h() {
        int ordinal = this.s.ordinal();
        if (ordinal != 1) {
            if (ordinal != 2) {
                if (ordinal != 3) {
                    if (ordinal == 5) {
                        return null;
                    }
                    StringBuilder x = c.b.a.a.a.x("Unrecognized stage: ");
                    x.append(this.s);
                    throw new IllegalStateException(x.toString());
                }
                return new b0(this.f3700b, this);
            }
            return new c.c.a.m.v.d(this.f3700b, this);
        }
        return new x(this.f3700b, this);
    }

    public final g i(g gVar) {
        g gVar2 = g.RESOURCE_CACHE;
        g gVar3 = g.DATA_CACHE;
        g gVar4 = g.FINISHED;
        int ordinal = gVar.ordinal();
        if (ordinal == 0) {
            return this.o.b() ? gVar2 : i(gVar2);
        } else if (ordinal == 1) {
            return this.o.a() ? gVar3 : i(gVar3);
        } else if (ordinal == 2) {
            return this.v ? gVar4 : g.SOURCE;
        } else if (ordinal == 3 || ordinal == 5) {
            return gVar4;
        } else {
            throw new IllegalArgumentException("Unrecognized stage: " + gVar);
        }
    }

    public final void j(String str, long j, String str2) {
        StringBuilder A = c.b.a.a.a.A(str, " in ");
        A.append(c.c.a.s.f.a(j));
        A.append(", load key: ");
        A.append(this.l);
        A.append(str2 != null ? c.b.a.a.a.q(", ", str2) : "");
        A.append(", thread: ");
        A.append(Thread.currentThread().getName());
        Log.v("DecodeJob", A.toString());
    }

    public final void k() {
        boolean a2;
        o();
        r rVar = new r("Failed to load resource", new ArrayList(this.f3701c));
        m<?> mVar = (m) this.q;
        synchronized (mVar) {
            mVar.v = rVar;
        }
        synchronized (mVar) {
            mVar.f3761d.a();
            if (mVar.z) {
                mVar.g();
            } else if (!mVar.f3760c.isEmpty()) {
                if (!mVar.w) {
                    mVar.w = true;
                    c.c.a.m.m mVar2 = mVar.n;
                    m.e eVar = mVar.f3760c;
                    Objects.requireNonNull(eVar);
                    ArrayList arrayList = new ArrayList(eVar.f3772b);
                    mVar.e(arrayList.size() + 1);
                    ((l) mVar.f3765h).e(mVar, mVar2, null);
                    Iterator it = arrayList.iterator();
                    while (it.hasNext()) {
                        m.d dVar = (m.d) it.next();
                        dVar.f3771b.execute(new m.a(dVar.f3770a));
                    }
                    mVar.d();
                } else {
                    throw new IllegalStateException("Already failed once");
                }
            } else {
                throw new IllegalStateException("Received an exception without any callbacks to notify");
            }
        }
        e eVar2 = this.f3706h;
        synchronized (eVar2) {
            eVar2.f3714c = true;
            a2 = eVar2.a(false);
        }
        if (a2) {
            l();
        }
    }

    public final void l() {
        e eVar = this.f3706h;
        synchronized (eVar) {
            eVar.f3713b = false;
            eVar.f3712a = false;
            eVar.f3714c = false;
        }
        c<?> cVar = this.f3705g;
        cVar.f3709a = null;
        cVar.f3710b = null;
        cVar.f3711c = null;
        h<R> hVar = this.f3700b;
        hVar.f3694c = null;
        hVar.f3695d = null;
        hVar.n = null;
        hVar.f3698g = null;
        hVar.k = null;
        hVar.i = null;
        hVar.o = null;
        hVar.j = null;
        hVar.p = null;
        hVar.f3692a.clear();
        hVar.l = false;
        hVar.f3693b.clear();
        hVar.m = false;
        this.E = false;
        this.i = null;
        this.j = null;
        this.p = null;
        this.k = null;
        this.l = null;
        this.q = null;
        this.s = null;
        this.D = null;
        this.x = null;
        this.y = null;
        this.A = null;
        this.B = null;
        this.C = null;
        this.u = 0L;
        this.F = false;
        this.w = null;
        this.f3701c.clear();
        this.f3704f.a(this);
    }

    public final void m() {
        this.x = Thread.currentThread();
        int i = c.c.a.s.f.f4187b;
        this.u = SystemClock.elapsedRealtimeNanos();
        boolean z = false;
        while (!this.F && this.D != null && !(z = this.D.b())) {
            this.s = i(this.s);
            this.D = h();
            if (this.s == g.SOURCE) {
                this.t = f.SWITCH_TO_SOURCE_SERVICE;
                ((m) this.q).i(this);
                return;
            }
        }
        if ((this.s == g.FINISHED || this.F) && !z) {
            k();
        }
    }

    public final void n() {
        int ordinal = this.t.ordinal();
        if (ordinal == 0) {
            this.s = i(g.INITIALIZE);
            this.D = h();
            m();
        } else if (ordinal == 1) {
            m();
        } else if (ordinal == 2) {
            g();
        } else {
            StringBuilder x = c.b.a.a.a.x("Unrecognized run reason: ");
            x.append(this.t);
            throw new IllegalStateException(x.toString());
        }
    }

    public final void o() {
        Throwable th;
        this.f3702d.a();
        if (this.E) {
            if (this.f3701c.isEmpty()) {
                th = null;
            } else {
                List<Throwable> list = this.f3701c;
                th = list.get(list.size() - 1);
            }
            throw new IllegalStateException("Already notified", th);
        }
        this.E = true;
    }

    @Override // java.lang.Runnable
    public void run() {
        c.c.a.m.u.d<?> dVar = this.C;
        try {
            try {
                if (this.F) {
                    k();
                    if (dVar != null) {
                        dVar.b();
                        return;
                    }
                    return;
                }
                n();
                if (dVar != null) {
                    dVar.b();
                }
            } catch (c.c.a.m.v.c e2) {
                throw e2;
            }
        }
    }
}