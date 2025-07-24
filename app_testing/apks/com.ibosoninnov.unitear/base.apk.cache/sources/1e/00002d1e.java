package f.g0.i;

import f.g0.i.o;
import java.io.Closeable;
import java.io.IOException;
import java.io.InterruptedIOException;
import java.net.Socket;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledThreadPoolExecutor;
import java.util.concurrent.SynchronousQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import org.opencv.calib3d.Calib3d;

/* compiled from: Http2Connection.java */
/* loaded from: classes2.dex */
public final class g implements Closeable {

    /* renamed from: b  reason: collision with root package name */
    public static final ExecutorService f5914b;

    /* renamed from: c  reason: collision with root package name */
    public final boolean f5915c;

    /* renamed from: d  reason: collision with root package name */
    public final d f5916d;

    /* renamed from: f  reason: collision with root package name */
    public final String f5918f;

    /* renamed from: g  reason: collision with root package name */
    public int f5919g;

    /* renamed from: h  reason: collision with root package name */
    public int f5920h;
    public boolean i;
    public final ScheduledExecutorService j;
    public final ExecutorService k;
    public final s l;
    public boolean m;
    public long o;
    public final t q;
    public boolean r;
    public final Socket s;
    public final q t;
    public final f u;
    public final Set<Integer> v;

    /* renamed from: e  reason: collision with root package name */
    public final Map<Integer, p> f5917e = new LinkedHashMap();
    public long n = 0;
    public t p = new t();

    /* compiled from: Http2Connection.java */
    /* loaded from: classes2.dex */
    public class a extends f.g0.b {

        /* renamed from: c  reason: collision with root package name */
        public final /* synthetic */ int f5921c;

        /* renamed from: d  reason: collision with root package name */
        public final /* synthetic */ f.g0.i.b f5922d;

        /* JADX WARN: 'super' call moved to the top of the method (can break code semantics) */
        public a(String str, Object[] objArr, int i, f.g0.i.b bVar) {
            super(str, objArr);
            this.f5921c = i;
            this.f5922d = bVar;
        }

        @Override // f.g0.b
        public void a() {
            try {
                g gVar = g.this;
                gVar.t.H(this.f5921c, this.f5922d);
            } catch (IOException unused) {
                g.B(g.this);
            }
        }
    }

    /* compiled from: Http2Connection.java */
    /* loaded from: classes2.dex */
    public class b extends f.g0.b {

        /* renamed from: c  reason: collision with root package name */
        public final /* synthetic */ int f5924c;

        /* renamed from: d  reason: collision with root package name */
        public final /* synthetic */ long f5925d;

        /* JADX WARN: 'super' call moved to the top of the method (can break code semantics) */
        public b(String str, Object[] objArr, int i, long j) {
            super(str, objArr);
            this.f5924c = i;
            this.f5925d = j;
        }

        @Override // f.g0.b
        public void a() {
            try {
                g.this.t.I(this.f5924c, this.f5925d);
            } catch (IOException unused) {
                g.B(g.this);
            }
        }
    }

    /* compiled from: Http2Connection.java */
    /* loaded from: classes2.dex */
    public static class c {

        /* renamed from: a  reason: collision with root package name */
        public Socket f5927a;

        /* renamed from: b  reason: collision with root package name */
        public String f5928b;

        /* renamed from: c  reason: collision with root package name */
        public g.g f5929c;

        /* renamed from: d  reason: collision with root package name */
        public g.f f5930d;

        /* renamed from: e  reason: collision with root package name */
        public d f5931e = d.f5933a;

        /* renamed from: f  reason: collision with root package name */
        public int f5932f;

        public c(boolean z) {
        }
    }

    /* compiled from: Http2Connection.java */
    /* loaded from: classes2.dex */
    public static abstract class d {

        /* renamed from: a  reason: collision with root package name */
        public static final d f5933a = new a();

        /* compiled from: Http2Connection.java */
        /* loaded from: classes2.dex */
        public class a extends d {
            @Override // f.g0.i.g.d
            public void b(p pVar) {
                pVar.c(f.g0.i.b.REFUSED_STREAM);
            }
        }

        public void a(g gVar) {
        }

        public abstract void b(p pVar);
    }

    /* compiled from: Http2Connection.java */
    /* loaded from: classes2.dex */
    public final class e extends f.g0.b {

        /* renamed from: c  reason: collision with root package name */
        public final boolean f5934c;

        /* renamed from: d  reason: collision with root package name */
        public final int f5935d;

        /* renamed from: e  reason: collision with root package name */
        public final int f5936e;

        public e(boolean z, int i, int i2) {
            super("OkHttp %s ping %08x%08x", g.this.f5918f, Integer.valueOf(i), Integer.valueOf(i2));
            this.f5934c = z;
            this.f5935d = i;
            this.f5936e = i2;
        }

        /* JADX WARN: Can't wrap try/catch for region: R(6:2|(2:f|(3:9|10|11))|18|19|10|11) */
        /* JADX WARN: Code restructure failed: missing block: B:15:0x0025, code lost:
            r0.C(r4, r4);
         */
        @Override // f.g0.b
        /*
            Code decompiled incorrectly, please refer to instructions dump.
        */
        public void a() {
            boolean z;
            g gVar = g.this;
            boolean z2 = this.f5934c;
            int i = this.f5935d;
            int i2 = this.f5936e;
            Objects.requireNonNull(gVar);
            f.g0.i.b bVar = f.g0.i.b.PROTOCOL_ERROR;
            try {
                if (!z2) {
                    synchronized (gVar) {
                        z = gVar.m;
                        gVar.m = true;
                    }
                    if (z) {
                        gVar.C(bVar, bVar);
                    }
                }
                gVar.t.G(z2, i, i2);
            } catch (IOException unused) {
            }
        }
    }

    /* compiled from: Http2Connection.java */
    /* loaded from: classes2.dex */
    public class f extends f.g0.b implements o.b {

        /* renamed from: c  reason: collision with root package name */
        public final o f5938c;

        public f(o oVar) {
            super("OkHttp %s", g.this.f5918f);
            this.f5938c = oVar;
        }

        @Override // f.g0.b
        public void a() {
            f.g0.i.b bVar;
            f.g0.i.b bVar2 = f.g0.i.b.INTERNAL_ERROR;
            try {
                try {
                    this.f5938c.D(this);
                    while (this.f5938c.C(false, this)) {
                    }
                    bVar = f.g0.i.b.NO_ERROR;
                } catch (IOException unused) {
                    bVar = bVar2;
                } catch (Throwable th) {
                    th = th;
                    bVar = bVar2;
                    try {
                        g.this.C(bVar, bVar2);
                    } catch (IOException unused2) {
                    }
                    f.g0.c.f(this.f5938c);
                    throw th;
                }
                try {
                    try {
                        g.this.C(bVar, f.g0.i.b.CANCEL);
                    } catch (Throwable th2) {
                        th = th2;
                        g.this.C(bVar, bVar2);
                        f.g0.c.f(this.f5938c);
                        throw th;
                    }
                } catch (IOException unused3) {
                    f.g0.i.b bVar3 = f.g0.i.b.PROTOCOL_ERROR;
                    g.this.C(bVar3, bVar3);
                    f.g0.c.f(this.f5938c);
                }
            } catch (IOException unused4) {
            }
            f.g0.c.f(this.f5938c);
        }
    }

    static {
        TimeUnit timeUnit = TimeUnit.SECONDS;
        SynchronousQueue synchronousQueue = new SynchronousQueue();
        byte[] bArr = f.g0.c.f5773a;
        f5914b = new ThreadPoolExecutor(0, Integer.MAX_VALUE, 60L, timeUnit, synchronousQueue, new f.g0.d("OkHttp Http2Connection", true));
    }

    public g(c cVar) {
        t tVar = new t();
        this.q = tVar;
        this.r = false;
        this.v = new LinkedHashSet();
        this.l = s.f6003a;
        this.f5915c = true;
        this.f5916d = cVar.f5931e;
        this.f5920h = 1;
        this.f5920h = 3;
        this.p.b(7, 16777216);
        String str = cVar.f5928b;
        this.f5918f = str;
        ScheduledThreadPoolExecutor scheduledThreadPoolExecutor = new ScheduledThreadPoolExecutor(1, new f.g0.d(f.g0.c.n("OkHttp %s Writer", str), false));
        this.j = scheduledThreadPoolExecutor;
        if (cVar.f5932f != 0) {
            e eVar = new e(false, 0, 0);
            long j = cVar.f5932f;
            scheduledThreadPoolExecutor.scheduleAtFixedRate(eVar, j, j, TimeUnit.MILLISECONDS);
        }
        this.k = new ThreadPoolExecutor(0, 1, 60L, TimeUnit.SECONDS, new LinkedBlockingQueue(), new f.g0.d(f.g0.c.n("OkHttp %s Push Observer", str), true));
        tVar.b(7, 65535);
        tVar.b(5, Calib3d.CALIB_RATIONAL_MODEL);
        this.o = tVar.a();
        this.s = cVar.f5927a;
        this.t = new q(cVar.f5930d, true);
        this.u = new f(new o(cVar.f5929c, true));
    }

    public static void B(g gVar) {
        Objects.requireNonNull(gVar);
        try {
            f.g0.i.b bVar = f.g0.i.b.PROTOCOL_ERROR;
            gVar.C(bVar, bVar);
        } catch (IOException unused) {
        }
    }

    public void C(f.g0.i.b bVar, f.g0.i.b bVar2) {
        p[] pVarArr = null;
        try {
            H(bVar);
            e = null;
        } catch (IOException e2) {
            e = e2;
        }
        synchronized (this) {
            if (!this.f5917e.isEmpty()) {
                pVarArr = (p[]) this.f5917e.values().toArray(new p[this.f5917e.size()]);
                this.f5917e.clear();
            }
        }
        if (pVarArr != null) {
            for (p pVar : pVarArr) {
                try {
                    pVar.c(bVar2);
                } catch (IOException e3) {
                    if (e != null) {
                        e = e3;
                    }
                }
            }
        }
        try {
            this.t.close();
        } catch (IOException e4) {
            if (e == null) {
                e = e4;
            }
        }
        try {
            this.s.close();
        } catch (IOException e5) {
            e = e5;
        }
        this.j.shutdown();
        this.k.shutdown();
        if (e != null) {
            throw e;
        }
    }

    public synchronized p D(int i) {
        return this.f5917e.get(Integer.valueOf(i));
    }

    public synchronized int E() {
        t tVar;
        tVar = this.q;
        return (tVar.f6004a & 16) != 0 ? tVar.f6005b[4] : Integer.MAX_VALUE;
    }

    public boolean F(int i) {
        return i != 0 && (i & 1) == 0;
    }

    public synchronized p G(int i) {
        p remove;
        remove = this.f5917e.remove(Integer.valueOf(i));
        notifyAll();
        return remove;
    }

    public void H(f.g0.i.b bVar) {
        synchronized (this.t) {
            synchronized (this) {
                if (this.i) {
                    return;
                }
                this.i = true;
                this.t.E(this.f5919g, bVar, f.g0.c.f5773a);
            }
        }
    }

    /* JADX WARN: Code restructure failed: missing block: B:15:0x002f, code lost:
        throw new java.io.IOException("stream closed");
     */
    /* JADX WARN: Code restructure failed: missing block: B:16:0x0030, code lost:
        r2 = java.lang.Math.min((int) java.lang.Math.min(r12, r4), r8.t.f5993f);
        r6 = r2;
        r8.o -= r6;
     */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public void I(int i, boolean z, g.e eVar, long j) {
        int min;
        long j2;
        if (j == 0) {
            this.t.C(z, i, eVar, 0);
            return;
        }
        loop0: while (j > 0) {
            synchronized (this) {
                while (true) {
                    try {
                        long j3 = this.o;
                        if (j3 > 0) {
                            break;
                        } else if (!this.f5917e.containsKey(Integer.valueOf(i))) {
                            break loop0;
                        } else {
                            wait();
                        }
                    } catch (InterruptedException unused) {
                        throw new InterruptedIOException();
                    }
                }
            }
            j -= j2;
            this.t.C(z && j == 0, i, eVar, min);
        }
    }

    public void J(int i, f.g0.i.b bVar) {
        try {
            this.j.execute(new a("OkHttp %s stream %d", new Object[]{this.f5918f, Integer.valueOf(i)}, i, bVar));
        } catch (RejectedExecutionException unused) {
        }
    }

    public void K(int i, long j) {
        try {
            this.j.execute(new b("OkHttp Window Update %s stream %d", new Object[]{this.f5918f, Integer.valueOf(i)}, i, j));
        } catch (RejectedExecutionException unused) {
        }
    }

    @Override // java.io.Closeable, java.lang.AutoCloseable
    public void close() {
        C(f.g0.i.b.NO_ERROR, f.g0.i.b.CANCEL);
    }

    public void flush() {
        this.t.flush();
    }
}