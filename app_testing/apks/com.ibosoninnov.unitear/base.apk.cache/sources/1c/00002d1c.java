package f.g0.i;

import com.google.common.net.HttpHeaders;
import f.b0;
import f.d0;
import f.g0.i.p;
import f.q;
import f.s;
import f.v;
import f.w;
import f.y;
import g.x;
import java.io.IOException;
import java.net.ProtocolException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.concurrent.TimeUnit;
import java.util.logging.Logger;

/* compiled from: Http2Codec.java */
/* loaded from: classes2.dex */
public final class f implements f.g0.g.c {

    /* renamed from: a  reason: collision with root package name */
    public static final g.h f5903a;

    /* renamed from: b  reason: collision with root package name */
    public static final g.h f5904b;

    /* renamed from: c  reason: collision with root package name */
    public static final g.h f5905c;

    /* renamed from: d  reason: collision with root package name */
    public static final g.h f5906d;

    /* renamed from: e  reason: collision with root package name */
    public static final g.h f5907e;

    /* renamed from: f  reason: collision with root package name */
    public static final g.h f5908f;

    /* renamed from: g  reason: collision with root package name */
    public static final g.h f5909g;

    /* renamed from: h  reason: collision with root package name */
    public static final g.h f5910h;
    public static final List<g.h> i;
    public static final List<g.h> j;
    public final s.a k;
    public final f.g0.f.g l;
    public final g m;
    public p n;

    /* compiled from: Http2Codec.java */
    /* loaded from: classes2.dex */
    public class a extends g.j {

        /* renamed from: c  reason: collision with root package name */
        public boolean f5911c;

        /* renamed from: d  reason: collision with root package name */
        public long f5912d;

        public a(x xVar) {
            super(xVar);
            this.f5911c = false;
            this.f5912d = 0L;
        }

        public final void B(IOException iOException) {
            if (this.f5911c) {
                return;
            }
            this.f5911c = true;
            f fVar = f.this;
            fVar.l.i(false, fVar, this.f5912d, iOException);
        }

        @Override // g.x, java.io.Closeable, java.lang.AutoCloseable
        public void close() {
            this.f6184b.close();
            B(null);
        }

        @Override // g.x
        public long u(g.e eVar, long j) {
            try {
                long u = this.f6184b.u(eVar, j);
                if (u > 0) {
                    this.f5912d += u;
                }
                return u;
            } catch (IOException e2) {
                B(e2);
                throw e2;
            }
        }
    }

    static {
        g.h e2 = g.h.e("connection");
        f5903a = e2;
        g.h e3 = g.h.e("host");
        f5904b = e3;
        g.h e4 = g.h.e("keep-alive");
        f5905c = e4;
        g.h e5 = g.h.e("proxy-connection");
        f5906d = e5;
        g.h e6 = g.h.e("transfer-encoding");
        f5907e = e6;
        g.h e7 = g.h.e("te");
        f5908f = e7;
        g.h e8 = g.h.e("encoding");
        f5909g = e8;
        g.h e9 = g.h.e("upgrade");
        f5910h = e9;
        i = f.g0.c.q(e2, e3, e4, e5, e7, e6, e8, e9, c.f5875c, c.f5876d, c.f5877e, c.f5878f);
        j = f.g0.c.q(e2, e3, e4, e5, e7, e6, e8, e9);
    }

    public f(v vVar, s.a aVar, f.g0.f.g gVar, g gVar2) {
        this.k = aVar;
        this.l = gVar;
        this.m = gVar2;
    }

    @Override // f.g0.g.c
    public void a() {
        ((p.a) this.n.f()).close();
    }

    @Override // f.g0.g.c
    public void b(y yVar) {
        int i2;
        p pVar;
        if (this.n != null) {
            return;
        }
        boolean z = false;
        boolean z2 = yVar.f6153d != null;
        f.q qVar = yVar.f6152c;
        ArrayList arrayList = new ArrayList(qVar.d() + 4);
        arrayList.add(new c(c.f5875c, yVar.f6151b));
        arrayList.add(new c(c.f5876d, b.v.u.c.y(yVar.f6150a)));
        String a2 = yVar.f6152c.a(HttpHeaders.HOST);
        if (a2 != null) {
            arrayList.add(new c(c.f5878f, a2));
        }
        arrayList.add(new c(c.f5877e, yVar.f6150a.f6087b));
        int d2 = qVar.d();
        for (int i3 = 0; i3 < d2; i3++) {
            g.h e2 = g.h.e(qVar.b(i3).toLowerCase(Locale.US));
            if (!i.contains(e2)) {
                arrayList.add(new c(e2, qVar.e(i3)));
            }
        }
        g gVar = this.m;
        boolean z3 = !z2;
        synchronized (gVar.t) {
            synchronized (gVar) {
                if (gVar.f5920h > 1073741823) {
                    gVar.H(b.REFUSED_STREAM);
                }
                if (!gVar.i) {
                    i2 = gVar.f5920h;
                    gVar.f5920h = i2 + 2;
                    pVar = new p(i2, gVar, z3, false, arrayList);
                    z = (!z2 || gVar.o == 0 || pVar.f5972b == 0) ? true : true;
                    if (pVar.h()) {
                        gVar.f5917e.put(Integer.valueOf(i2), pVar);
                    }
                } else {
                    throw new f.g0.i.a();
                }
            }
            q qVar2 = gVar.t;
            synchronized (qVar2) {
                if (!qVar2.f5994g) {
                    qVar2.F(z3, i2, arrayList);
                } else {
                    throw new IOException("closed");
                }
            }
        }
        if (z) {
            gVar.t.flush();
        }
        this.n = pVar;
        TimeUnit timeUnit = TimeUnit.MILLISECONDS;
        pVar.j.g(((f.g0.g.f) this.k).j, timeUnit);
        this.n.k.g(((f.g0.g.f) this.k).k, timeUnit);
    }

    @Override // f.g0.g.c
    public d0 c(b0 b0Var) {
        Objects.requireNonNull(this.l.f5815f);
        String a2 = b0Var.f5729g.a(HttpHeaders.CONTENT_TYPE);
        if (a2 == null) {
            a2 = null;
        }
        long a3 = f.g0.g.e.a(b0Var);
        a aVar = new a(this.n.f5978h);
        Logger logger = g.o.f6197a;
        return new f.g0.g.g(a2, a3, new g.s(aVar));
    }

    @Override // f.g0.g.c
    public void cancel() {
        p pVar = this.n;
        if (pVar != null) {
            pVar.e(b.CANCEL);
        }
    }

    @Override // f.g0.g.c
    public b0.a d(boolean z) {
        List<c> list;
        p pVar = this.n;
        synchronized (pVar) {
            if (pVar.g()) {
                pVar.j.i();
                while (pVar.f5976f == null && pVar.l == null) {
                    pVar.j();
                }
                pVar.j.n();
                list = pVar.f5976f;
                if (list != null) {
                    pVar.f5976f = null;
                } else {
                    throw new u(pVar.l);
                }
            } else {
                throw new IllegalStateException("servers cannot read response headers");
            }
        }
        q.a aVar = new q.a();
        int size = list.size();
        f.g0.g.i iVar = null;
        for (int i2 = 0; i2 < size; i2++) {
            c cVar = list.get(i2);
            if (cVar == null) {
                if (iVar != null && iVar.f5842b == 100) {
                    aVar = new q.a();
                    iVar = null;
                }
            } else {
                g.h hVar = cVar.f5879g;
                String p = cVar.f5880h.p();
                if (hVar.equals(c.f5874b)) {
                    iVar = f.g0.g.i.a("HTTP/1.1 " + p);
                } else if (!j.contains(hVar)) {
                    f.g0.a.f5771a.a(aVar, hVar.p(), p);
                }
            }
        }
        if (iVar != null) {
            b0.a aVar2 = new b0.a();
            aVar2.f5732b = w.HTTP_2;
            aVar2.f5733c = iVar.f5842b;
            aVar2.f5734d = iVar.f5843c;
            List<String> list2 = aVar.f6085a;
            q.a aVar3 = new q.a();
            Collections.addAll(aVar3.f6085a, (String[]) list2.toArray(new String[list2.size()]));
            aVar2.f5736f = aVar3;
            if (z) {
                Objects.requireNonNull((v.a) f.g0.a.f5771a);
                if (aVar2.f5733c == 100) {
                    return null;
                }
            }
            return aVar2;
        }
        throw new ProtocolException("Expected ':status' header not present");
    }

    @Override // f.g0.g.c
    public void e() {
        this.m.t.flush();
    }

    @Override // f.g0.g.c
    public g.w f(y yVar, long j2) {
        return this.n.f();
    }
}