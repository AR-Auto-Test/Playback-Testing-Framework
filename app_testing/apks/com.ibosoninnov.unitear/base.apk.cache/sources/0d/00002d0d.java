package f.g0.h;

import com.google.common.net.HttpHeaders;
import f.b0;
import f.d0;
import f.g0.g.i;
import f.q;
import f.r;
import f.v;
import g.k;
import g.o;
import g.s;
import g.w;
import g.x;
import g.y;
import java.io.EOFException;
import java.io.IOException;
import java.net.ProtocolException;
import java.net.Proxy;
import java.util.Objects;
import java.util.concurrent.TimeUnit;
import java.util.logging.Logger;

/* compiled from: Http1Codec.java */
/* loaded from: classes2.dex */
public final class a implements f.g0.g.c {

    /* renamed from: a  reason: collision with root package name */
    public final v f5844a;

    /* renamed from: b  reason: collision with root package name */
    public final f.g0.f.g f5845b;

    /* renamed from: c  reason: collision with root package name */
    public final g.g f5846c;

    /* renamed from: d  reason: collision with root package name */
    public final g.f f5847d;

    /* renamed from: e  reason: collision with root package name */
    public int f5848e = 0;

    /* renamed from: f  reason: collision with root package name */
    public long f5849f = 262144;

    /* compiled from: Http1Codec.java */
    /* loaded from: classes2.dex */
    public abstract class b implements x {

        /* renamed from: b  reason: collision with root package name */
        public final k f5850b;

        /* renamed from: c  reason: collision with root package name */
        public boolean f5851c;

        /* renamed from: d  reason: collision with root package name */
        public long f5852d = 0;

        public b(C0127a c0127a) {
            this.f5850b = new k(a.this.f5846c.b());
        }

        public final void B(boolean z, IOException iOException) {
            a aVar = a.this;
            int i = aVar.f5848e;
            if (i == 6) {
                return;
            }
            if (i == 5) {
                aVar.g(this.f5850b);
                a aVar2 = a.this;
                aVar2.f5848e = 6;
                f.g0.f.g gVar = aVar2.f5845b;
                if (gVar != null) {
                    gVar.i(!z, aVar2, this.f5852d, iOException);
                    return;
                }
                return;
            }
            StringBuilder x = c.b.a.a.a.x("state: ");
            x.append(a.this.f5848e);
            throw new IllegalStateException(x.toString());
        }

        @Override // g.x
        public y b() {
            return this.f5850b;
        }

        @Override // g.x
        public long u(g.e eVar, long j) {
            try {
                long u = a.this.f5846c.u(eVar, j);
                if (u > 0) {
                    this.f5852d += u;
                }
                return u;
            } catch (IOException e2) {
                B(false, e2);
                throw e2;
            }
        }
    }

    /* compiled from: Http1Codec.java */
    /* loaded from: classes2.dex */
    public final class c implements w {

        /* renamed from: b  reason: collision with root package name */
        public final k f5854b;

        /* renamed from: c  reason: collision with root package name */
        public boolean f5855c;

        public c() {
            this.f5854b = new k(a.this.f5847d.b());
        }

        @Override // g.w
        public y b() {
            return this.f5854b;
        }

        @Override // g.w, java.io.Closeable, java.lang.AutoCloseable
        public synchronized void close() {
            if (this.f5855c) {
                return;
            }
            this.f5855c = true;
            a.this.f5847d.i("0\r\n\r\n");
            a.this.g(this.f5854b);
            a.this.f5848e = 3;
        }

        @Override // g.w, java.io.Flushable
        public synchronized void flush() {
            if (this.f5855c) {
                return;
            }
            a.this.f5847d.flush();
        }

        @Override // g.w
        public void l(g.e eVar, long j) {
            if (this.f5855c) {
                throw new IllegalStateException("closed");
            }
            if (j == 0) {
                return;
            }
            a.this.f5847d.m(j);
            a.this.f5847d.i("\r\n");
            a.this.f5847d.l(eVar, j);
            a.this.f5847d.i("\r\n");
        }
    }

    /* compiled from: Http1Codec.java */
    /* loaded from: classes2.dex */
    public class d extends b {

        /* renamed from: f  reason: collision with root package name */
        public final r f5857f;

        /* renamed from: g  reason: collision with root package name */
        public long f5858g;

        /* renamed from: h  reason: collision with root package name */
        public boolean f5859h;

        public d(r rVar) {
            super(null);
            this.f5858g = -1L;
            this.f5859h = true;
            this.f5857f = rVar;
        }

        @Override // g.x, java.io.Closeable, java.lang.AutoCloseable
        public void close() {
            if (this.f5851c) {
                return;
            }
            if (this.f5859h && !f.g0.c.l(this, 100, TimeUnit.MILLISECONDS)) {
                B(false, null);
            }
            this.f5851c = true;
        }

        @Override // f.g0.h.a.b, g.x
        public long u(g.e eVar, long j) {
            if (j >= 0) {
                if (!this.f5851c) {
                    if (this.f5859h) {
                        long j2 = this.f5858g;
                        if (j2 == 0 || j2 == -1) {
                            if (j2 != -1) {
                                a.this.f5846c.p();
                            }
                            try {
                                this.f5858g = a.this.f5846c.y();
                                String trim = a.this.f5846c.p().trim();
                                if (this.f5858g >= 0 && (trim.isEmpty() || trim.startsWith(";"))) {
                                    if (this.f5858g == 0) {
                                        this.f5859h = false;
                                        a aVar = a.this;
                                        f.g0.g.e.d(aVar.f5844a.k, this.f5857f, aVar.j());
                                        B(true, null);
                                    }
                                    if (!this.f5859h) {
                                        return -1L;
                                    }
                                } else {
                                    throw new ProtocolException("expected chunk size and optional extensions but was \"" + this.f5858g + trim + "\"");
                                }
                            } catch (NumberFormatException e2) {
                                throw new ProtocolException(e2.getMessage());
                            }
                        }
                        long u = super.u(eVar, Math.min(j, this.f5858g));
                        if (u != -1) {
                            this.f5858g -= u;
                            return u;
                        }
                        ProtocolException protocolException = new ProtocolException("unexpected end of stream");
                        B(false, protocolException);
                        throw protocolException;
                    }
                    return -1L;
                }
                throw new IllegalStateException("closed");
            }
            throw new IllegalArgumentException(c.b.a.a.a.l("byteCount < 0: ", j));
        }
    }

    /* compiled from: Http1Codec.java */
    /* loaded from: classes2.dex */
    public final class e implements w {

        /* renamed from: b  reason: collision with root package name */
        public final k f5860b;

        /* renamed from: c  reason: collision with root package name */
        public boolean f5861c;

        /* renamed from: d  reason: collision with root package name */
        public long f5862d;

        public e(long j) {
            this.f5860b = new k(a.this.f5847d.b());
            this.f5862d = j;
        }

        @Override // g.w
        public y b() {
            return this.f5860b;
        }

        @Override // g.w, java.io.Closeable, java.lang.AutoCloseable
        public void close() {
            if (this.f5861c) {
                return;
            }
            this.f5861c = true;
            if (this.f5862d <= 0) {
                a.this.g(this.f5860b);
                a.this.f5848e = 3;
                return;
            }
            throw new ProtocolException("unexpected end of stream");
        }

        @Override // g.w, java.io.Flushable
        public void flush() {
            if (this.f5861c) {
                return;
            }
            a.this.f5847d.flush();
        }

        @Override // g.w
        public void l(g.e eVar, long j) {
            if (!this.f5861c) {
                f.g0.c.e(eVar.f6176d, 0L, j);
                if (j <= this.f5862d) {
                    a.this.f5847d.l(eVar, j);
                    this.f5862d -= j;
                    return;
                }
                StringBuilder x = c.b.a.a.a.x("expected ");
                x.append(this.f5862d);
                x.append(" bytes but received ");
                x.append(j);
                throw new ProtocolException(x.toString());
            }
            throw new IllegalStateException("closed");
        }
    }

    /* compiled from: Http1Codec.java */
    /* loaded from: classes2.dex */
    public class f extends b {

        /* renamed from: f  reason: collision with root package name */
        public long f5864f;

        public f(a aVar, long j) {
            super(null);
            this.f5864f = j;
            if (j == 0) {
                B(true, null);
            }
        }

        @Override // g.x, java.io.Closeable, java.lang.AutoCloseable
        public void close() {
            if (this.f5851c) {
                return;
            }
            if (this.f5864f != 0 && !f.g0.c.l(this, 100, TimeUnit.MILLISECONDS)) {
                B(false, null);
            }
            this.f5851c = true;
        }

        @Override // f.g0.h.a.b, g.x
        public long u(g.e eVar, long j) {
            if (j >= 0) {
                if (!this.f5851c) {
                    long j2 = this.f5864f;
                    if (j2 == 0) {
                        return -1L;
                    }
                    long u = super.u(eVar, Math.min(j2, j));
                    if (u != -1) {
                        long j3 = this.f5864f - u;
                        this.f5864f = j3;
                        if (j3 == 0) {
                            B(true, null);
                        }
                        return u;
                    }
                    ProtocolException protocolException = new ProtocolException("unexpected end of stream");
                    B(false, protocolException);
                    throw protocolException;
                }
                throw new IllegalStateException("closed");
            }
            throw new IllegalArgumentException(c.b.a.a.a.l("byteCount < 0: ", j));
        }
    }

    /* compiled from: Http1Codec.java */
    /* loaded from: classes2.dex */
    public class g extends b {

        /* renamed from: f  reason: collision with root package name */
        public boolean f5865f;

        public g(a aVar) {
            super(null);
        }

        @Override // g.x, java.io.Closeable, java.lang.AutoCloseable
        public void close() {
            if (this.f5851c) {
                return;
            }
            if (!this.f5865f) {
                B(false, null);
            }
            this.f5851c = true;
        }

        @Override // f.g0.h.a.b, g.x
        public long u(g.e eVar, long j) {
            if (j >= 0) {
                if (!this.f5851c) {
                    if (this.f5865f) {
                        return -1L;
                    }
                    long u = super.u(eVar, j);
                    if (u == -1) {
                        this.f5865f = true;
                        B(true, null);
                        return -1L;
                    }
                    return u;
                }
                throw new IllegalStateException("closed");
            }
            throw new IllegalArgumentException(c.b.a.a.a.l("byteCount < 0: ", j));
        }
    }

    public a(v vVar, f.g0.f.g gVar, g.g gVar2, g.f fVar) {
        this.f5844a = vVar;
        this.f5845b = gVar;
        this.f5846c = gVar2;
        this.f5847d = fVar;
    }

    @Override // f.g0.g.c
    public void a() {
        this.f5847d.flush();
    }

    @Override // f.g0.g.c
    public void b(f.y yVar) {
        Proxy.Type type = this.f5845b.b().f5791c.f5751b.type();
        StringBuilder sb = new StringBuilder();
        sb.append(yVar.f6151b);
        sb.append(' ');
        if (!yVar.f6150a.f6087b.equals("https") && type == Proxy.Type.HTTP) {
            sb.append(yVar.f6150a);
        } else {
            sb.append(b.v.u.c.y(yVar.f6150a));
        }
        sb.append(" HTTP/1.1");
        k(yVar.f6152c, sb.toString());
    }

    @Override // f.g0.g.c
    public d0 c(b0 b0Var) {
        Objects.requireNonNull(this.f5845b.f5815f);
        String a2 = b0Var.f5729g.a(HttpHeaders.CONTENT_TYPE);
        if (a2 == null) {
            a2 = null;
        }
        if (!f.g0.g.e.b(b0Var)) {
            x h2 = h(0L);
            Logger logger = o.f6197a;
            return new f.g0.g.g(a2, 0L, new s(h2));
        }
        String a3 = b0Var.f5729g.a(HttpHeaders.TRANSFER_ENCODING);
        if ("chunked".equalsIgnoreCase(a3 != null ? a3 : null)) {
            r rVar = b0Var.f5724b.f6150a;
            if (this.f5848e == 4) {
                this.f5848e = 5;
                d dVar = new d(rVar);
                Logger logger2 = o.f6197a;
                return new f.g0.g.g(a2, -1L, new s(dVar));
            }
            StringBuilder x = c.b.a.a.a.x("state: ");
            x.append(this.f5848e);
            throw new IllegalStateException(x.toString());
        }
        long a4 = f.g0.g.e.a(b0Var);
        if (a4 != -1) {
            x h3 = h(a4);
            Logger logger3 = o.f6197a;
            return new f.g0.g.g(a2, a4, new s(h3));
        } else if (this.f5848e == 4) {
            f.g0.f.g gVar = this.f5845b;
            if (gVar != null) {
                this.f5848e = 5;
                gVar.f();
                g gVar2 = new g(this);
                Logger logger4 = o.f6197a;
                return new f.g0.g.g(a2, -1L, new s(gVar2));
            }
            throw new IllegalStateException("streamAllocation == null");
        } else {
            StringBuilder x2 = c.b.a.a.a.x("state: ");
            x2.append(this.f5848e);
            throw new IllegalStateException(x2.toString());
        }
    }

    @Override // f.g0.g.c
    public void cancel() {
        f.g0.f.c b2 = this.f5845b.b();
        if (b2 != null) {
            f.g0.c.g(b2.f5792d);
        }
    }

    @Override // f.g0.g.c
    public b0.a d(boolean z) {
        int i = this.f5848e;
        if (i != 1 && i != 3) {
            StringBuilder x = c.b.a.a.a.x("state: ");
            x.append(this.f5848e);
            throw new IllegalStateException(x.toString());
        }
        try {
            i a2 = i.a(i());
            b0.a aVar = new b0.a();
            aVar.f5732b = a2.f5841a;
            aVar.f5733c = a2.f5842b;
            aVar.f5734d = a2.f5843c;
            aVar.d(j());
            if (z && a2.f5842b == 100) {
                return null;
            }
            if (a2.f5842b == 100) {
                this.f5848e = 3;
                return aVar;
            }
            this.f5848e = 4;
            return aVar;
        } catch (EOFException e2) {
            StringBuilder x2 = c.b.a.a.a.x("unexpected end of stream on ");
            x2.append(this.f5845b);
            IOException iOException = new IOException(x2.toString());
            iOException.initCause(e2);
            throw iOException;
        }
    }

    @Override // f.g0.g.c
    public void e() {
        this.f5847d.flush();
    }

    @Override // f.g0.g.c
    public w f(f.y yVar, long j) {
        if ("chunked".equalsIgnoreCase(yVar.f6152c.a(HttpHeaders.TRANSFER_ENCODING))) {
            if (this.f5848e == 1) {
                this.f5848e = 2;
                return new c();
            }
            StringBuilder x = c.b.a.a.a.x("state: ");
            x.append(this.f5848e);
            throw new IllegalStateException(x.toString());
        } else if (j != -1) {
            if (this.f5848e == 1) {
                this.f5848e = 2;
                return new e(j);
            }
            StringBuilder x2 = c.b.a.a.a.x("state: ");
            x2.append(this.f5848e);
            throw new IllegalStateException(x2.toString());
        } else {
            throw new IllegalStateException("Cannot stream a request body without chunked encoding or a known content length!");
        }
    }

    public void g(k kVar) {
        y yVar = kVar.f6185e;
        kVar.f6185e = y.f6220a;
        yVar.a();
        yVar.b();
    }

    public x h(long j) {
        if (this.f5848e == 4) {
            this.f5848e = 5;
            return new f(this, j);
        }
        StringBuilder x = c.b.a.a.a.x("state: ");
        x.append(this.f5848e);
        throw new IllegalStateException(x.toString());
    }

    public final String i() {
        String h2 = this.f5846c.h(this.f5849f);
        this.f5849f -= h2.length();
        return h2;
    }

    public q j() {
        q.a aVar = new q.a();
        while (true) {
            String i = i();
            if (i.length() != 0) {
                Objects.requireNonNull((v.a) f.g0.a.f5771a);
                int indexOf = i.indexOf(":", 1);
                if (indexOf != -1) {
                    aVar.a(i.substring(0, indexOf), i.substring(indexOf + 1));
                } else if (i.startsWith(":")) {
                    String substring = i.substring(1);
                    aVar.f6085a.add("");
                    aVar.f6085a.add(substring.trim());
                } else {
                    aVar.f6085a.add("");
                    aVar.f6085a.add(i.trim());
                }
            } else {
                return new q(aVar);
            }
        }
    }

    public void k(q qVar, String str) {
        if (this.f5848e == 0) {
            this.f5847d.i(str).i("\r\n");
            int d2 = qVar.d();
            for (int i = 0; i < d2; i++) {
                this.f5847d.i(qVar.b(i)).i(": ").i(qVar.e(i)).i("\r\n");
            }
            this.f5847d.i("\r\n");
            this.f5848e = 1;
            return;
        }
        StringBuilder x = c.b.a.a.a.x("state: ");
        x.append(this.f5848e);
        throw new IllegalStateException(x.toString());
    }
}