package f.g0.i;

import g.w;
import g.x;
import g.y;
import java.io.IOException;
import java.io.InterruptedIOException;
import java.net.SocketTimeoutException;
import java.util.List;
import java.util.Objects;

/* compiled from: Http2Stream.java */
/* loaded from: classes2.dex */
public final class p {

    /* renamed from: b  reason: collision with root package name */
    public long f5972b;

    /* renamed from: c  reason: collision with root package name */
    public final int f5973c;

    /* renamed from: d  reason: collision with root package name */
    public final g f5974d;

    /* renamed from: e  reason: collision with root package name */
    public final List<f.g0.i.c> f5975e;

    /* renamed from: f  reason: collision with root package name */
    public List<f.g0.i.c> f5976f;

    /* renamed from: g  reason: collision with root package name */
    public boolean f5977g;

    /* renamed from: h  reason: collision with root package name */
    public final b f5978h;
    public final a i;

    /* renamed from: a  reason: collision with root package name */
    public long f5971a = 0;
    public final c j = new c();
    public final c k = new c();
    public f.g0.i.b l = null;

    /* compiled from: Http2Stream.java */
    /* loaded from: classes2.dex */
    public final class a implements w {

        /* renamed from: b  reason: collision with root package name */
        public final g.e f5979b = new g.e();

        /* renamed from: c  reason: collision with root package name */
        public boolean f5980c;

        /* renamed from: d  reason: collision with root package name */
        public boolean f5981d;

        public a() {
        }

        public final void B(boolean z) {
            p pVar;
            long min;
            p pVar2;
            synchronized (p.this) {
                p.this.k.i();
                while (true) {
                    pVar = p.this;
                    if (pVar.f5972b > 0 || this.f5981d || this.f5980c || pVar.l != null) {
                        break;
                    }
                    pVar.j();
                }
                pVar.k.n();
                p.this.b();
                min = Math.min(p.this.f5972b, this.f5979b.f6176d);
                pVar2 = p.this;
                pVar2.f5972b -= min;
            }
            pVar2.k.i();
            try {
                p pVar3 = p.this;
                pVar3.f5974d.I(pVar3.f5973c, z && min == this.f5979b.f6176d, this.f5979b, min);
            } finally {
                p.this.k.n();
            }
        }

        @Override // g.w
        public y b() {
            return p.this.k;
        }

        @Override // g.w, java.io.Closeable, java.lang.AutoCloseable
        public void close() {
            synchronized (p.this) {
                if (this.f5980c) {
                    return;
                }
                p pVar = p.this;
                if (!pVar.i.f5981d) {
                    if (this.f5979b.f6176d > 0) {
                        while (this.f5979b.f6176d > 0) {
                            B(true);
                        }
                    } else {
                        pVar.f5974d.I(pVar.f5973c, true, null, 0L);
                    }
                }
                synchronized (p.this) {
                    this.f5980c = true;
                }
                p.this.f5974d.t.flush();
                p.this.a();
            }
        }

        @Override // g.w, java.io.Flushable
        public void flush() {
            synchronized (p.this) {
                p.this.b();
            }
            while (this.f5979b.f6176d > 0) {
                B(false);
                p.this.f5974d.flush();
            }
        }

        @Override // g.w
        public void l(g.e eVar, long j) {
            this.f5979b.l(eVar, j);
            while (this.f5979b.f6176d >= 16384) {
                B(false);
            }
        }
    }

    /* compiled from: Http2Stream.java */
    /* loaded from: classes2.dex */
    public final class b implements x {

        /* renamed from: b  reason: collision with root package name */
        public final g.e f5983b = new g.e();

        /* renamed from: c  reason: collision with root package name */
        public final g.e f5984c = new g.e();

        /* renamed from: d  reason: collision with root package name */
        public final long f5985d;

        /* renamed from: e  reason: collision with root package name */
        public boolean f5986e;

        /* renamed from: f  reason: collision with root package name */
        public boolean f5987f;

        public b(long j) {
            this.f5985d = j;
        }

        public final void B() {
            p.this.j.i();
            while (this.f5984c.f6176d == 0 && !this.f5987f && !this.f5986e) {
                try {
                    p pVar = p.this;
                    if (pVar.l != null) {
                        break;
                    }
                    pVar.j();
                } finally {
                    p.this.j.n();
                }
            }
        }

        @Override // g.x
        public y b() {
            return p.this.j;
        }

        @Override // g.x, java.io.Closeable, java.lang.AutoCloseable
        public void close() {
            synchronized (p.this) {
                this.f5986e = true;
                this.f5984c.B();
                p.this.notifyAll();
            }
            p.this.a();
        }

        @Override // g.x
        public long u(g.e eVar, long j) {
            if (j >= 0) {
                synchronized (p.this) {
                    B();
                    if (!this.f5986e) {
                        if (p.this.l == null) {
                            g.e eVar2 = this.f5984c;
                            long j2 = eVar2.f6176d;
                            if (j2 == 0) {
                                return -1L;
                            }
                            long u = eVar2.u(eVar, Math.min(j, j2));
                            p pVar = p.this;
                            long j3 = pVar.f5971a + u;
                            pVar.f5971a = j3;
                            if (j3 >= pVar.f5974d.p.a() / 2) {
                                p pVar2 = p.this;
                                pVar2.f5974d.K(pVar2.f5973c, pVar2.f5971a);
                                p.this.f5971a = 0L;
                            }
                            synchronized (p.this.f5974d) {
                                g gVar = p.this.f5974d;
                                long j4 = gVar.n + u;
                                gVar.n = j4;
                                if (j4 >= gVar.p.a() / 2) {
                                    g gVar2 = p.this.f5974d;
                                    gVar2.K(0, gVar2.n);
                                    p.this.f5974d.n = 0L;
                                }
                            }
                            return u;
                        }
                        throw new u(p.this.l);
                    }
                    throw new IOException("stream closed");
                }
            }
            throw new IllegalArgumentException(c.b.a.a.a.l("byteCount < 0: ", j));
        }
    }

    /* compiled from: Http2Stream.java */
    /* loaded from: classes2.dex */
    public class c extends g.c {
        public c() {
        }

        @Override // g.c
        public IOException l(IOException iOException) {
            SocketTimeoutException socketTimeoutException = new SocketTimeoutException("timeout");
            if (iOException != null) {
                socketTimeoutException.initCause(iOException);
            }
            return socketTimeoutException;
        }

        @Override // g.c
        public void m() {
            p.this.e(f.g0.i.b.CANCEL);
        }

        public void n() {
            if (k()) {
                throw l(null);
            }
        }
    }

    public p(int i, g gVar, boolean z, boolean z2, List<f.g0.i.c> list) {
        Objects.requireNonNull(gVar, "connection == null");
        Objects.requireNonNull(list, "requestHeaders == null");
        this.f5973c = i;
        this.f5974d = gVar;
        this.f5972b = gVar.q.a();
        b bVar = new b(gVar.p.a());
        this.f5978h = bVar;
        a aVar = new a();
        this.i = aVar;
        bVar.f5987f = z2;
        aVar.f5981d = z;
        this.f5975e = list;
    }

    public void a() {
        boolean z;
        boolean h2;
        synchronized (this) {
            b bVar = this.f5978h;
            if (!bVar.f5987f && bVar.f5986e) {
                a aVar = this.i;
                if (aVar.f5981d || aVar.f5980c) {
                    z = true;
                    h2 = h();
                }
            }
            z = false;
            h2 = h();
        }
        if (z) {
            c(f.g0.i.b.CANCEL);
        } else if (h2) {
        } else {
            this.f5974d.G(this.f5973c);
        }
    }

    public void b() {
        a aVar = this.i;
        if (!aVar.f5980c) {
            if (!aVar.f5981d) {
                if (this.l != null) {
                    throw new u(this.l);
                }
                return;
            }
            throw new IOException("stream finished");
        }
        throw new IOException("stream closed");
    }

    public void c(f.g0.i.b bVar) {
        if (d(bVar)) {
            g gVar = this.f5974d;
            gVar.t.H(this.f5973c, bVar);
        }
    }

    public final boolean d(f.g0.i.b bVar) {
        synchronized (this) {
            if (this.l != null) {
                return false;
            }
            if (this.f5978h.f5987f && this.i.f5981d) {
                return false;
            }
            this.l = bVar;
            notifyAll();
            this.f5974d.G(this.f5973c);
            return true;
        }
    }

    public void e(f.g0.i.b bVar) {
        if (d(bVar)) {
            this.f5974d.J(this.f5973c, bVar);
        }
    }

    public w f() {
        synchronized (this) {
            if (!this.f5977g && !g()) {
                throw new IllegalStateException("reply before requesting the sink");
            }
        }
        return this.i;
    }

    public boolean g() {
        return this.f5974d.f5915c == ((this.f5973c & 1) == 1);
    }

    public synchronized boolean h() {
        if (this.l != null) {
            return false;
        }
        b bVar = this.f5978h;
        if (bVar.f5987f || bVar.f5986e) {
            a aVar = this.i;
            if (aVar.f5981d || aVar.f5980c) {
                if (this.f5977g) {
                    return false;
                }
            }
        }
        return true;
    }

    public void i() {
        boolean h2;
        synchronized (this) {
            this.f5978h.f5987f = true;
            h2 = h();
            notifyAll();
        }
        if (h2) {
            return;
        }
        this.f5974d.G(this.f5973c);
    }

    public void j() {
        try {
            wait();
        } catch (InterruptedException unused) {
            throw new InterruptedIOException();
        }
    }
}