package f.g0.f;

import androidx.recyclerview.widget.RecyclerView;
import com.google.common.net.HttpHeaders;
import f.b0;
import f.e0;
import f.g0.h.a;
import f.g0.i.g;
import f.g0.i.t;
import f.h;
import f.i;
import f.n;
import f.p;
import f.q;
import f.s;
import f.v;
import f.w;
import f.y;
import g.o;
import g.r;
import g.s;
import g.x;
import java.io.IOException;
import java.io.InterruptedIOException;
import java.lang.ref.Reference;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.net.ConnectException;
import java.net.InetSocketAddress;
import java.net.ProtocolException;
import java.net.Proxy;
import java.net.Socket;
import java.net.UnknownServiceException;
import java.security.cert.CertificateException;
import java.security.cert.X509Certificate;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.TimeUnit;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.net.ssl.SSLHandshakeException;
import javax.net.ssl.SSLPeerUnverifiedException;
import javax.net.ssl.SSLProtocolException;
import javax.net.ssl.SSLSession;
import javax.net.ssl.SSLSocket;
import javax.net.ssl.SSLSocketFactory;

/* compiled from: RealConnection.java */
/* loaded from: classes2.dex */
public final class c extends g.d {

    /* renamed from: b  reason: collision with root package name */
    public final h f5790b;

    /* renamed from: c  reason: collision with root package name */
    public final e0 f5791c;

    /* renamed from: d  reason: collision with root package name */
    public Socket f5792d;

    /* renamed from: e  reason: collision with root package name */
    public Socket f5793e;

    /* renamed from: f  reason: collision with root package name */
    public p f5794f;

    /* renamed from: g  reason: collision with root package name */
    public w f5795g;

    /* renamed from: h  reason: collision with root package name */
    public f.g0.i.g f5796h;
    public g.g i;
    public g.f j;
    public boolean k;
    public int l;
    public int m = 1;
    public final List<Reference<g>> n = new ArrayList();
    public long o = RecyclerView.FOREVER_NS;

    public c(h hVar, e0 e0Var) {
        this.f5790b = hVar;
        this.f5791c = e0Var;
    }

    @Override // f.g0.i.g.d
    public void a(f.g0.i.g gVar) {
        synchronized (this.f5790b) {
            this.m = gVar.E();
        }
    }

    @Override // f.g0.i.g.d
    public void b(f.g0.i.p pVar) {
        pVar.c(f.g0.i.b.REFUSED_STREAM);
    }

    /* JADX WARN: Removed duplicated region for block: B:35:0x00a4  */
    /* JADX WARN: Removed duplicated region for block: B:43:0x00b4 A[ORIG_RETURN, RETURN] */
    /* JADX WARN: Removed duplicated region for block: B:52:0x00e4  */
    /* JADX WARN: Removed duplicated region for block: B:53:0x00ea  */
    /* JADX WARN: Removed duplicated region for block: B:58:0x00fb  */
    /* JADX WARN: Removed duplicated region for block: B:97:0x0129 A[EDGE_INSN: B:97:0x0129->B:81:0x0129 ?: BREAK  , SYNTHETIC] */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public void c(int i, int i2, int i3, int i4, boolean z, f.d dVar, n nVar) {
        boolean z2;
        if (this.f5795g == null) {
            f.a aVar = this.f5791c.f5750a;
            List<i> list = aVar.f5720f;
            b bVar = new b(list);
            if (aVar.i == null) {
                if (list.contains(i.f6055c)) {
                    String str = this.f5791c.f5750a.f5715a.f6090e;
                    if (!f.g0.j.f.f6032a.j(str)) {
                        throw new e(new UnknownServiceException(c.b.a.a.a.r("CLEARTEXT communication to ", str, " not permitted by network security policy")));
                    }
                } else {
                    throw new e(new UnknownServiceException("CLEARTEXT communication not enabled for client"));
                }
            }
            e eVar = null;
            do {
                try {
                } catch (IOException e2) {
                    e = e2;
                }
                try {
                    if (this.f5791c.a()) {
                        e(i, i2, i3, dVar, nVar);
                        if (this.f5792d == null) {
                            if (!this.f5791c.a() && this.f5792d == null) {
                                throw new e(new ProtocolException("Too many tunnel connections attempted: 21"));
                            }
                            if (this.f5796h == null) {
                                synchronized (this.f5790b) {
                                    this.m = this.f5796h.E();
                                }
                                return;
                            }
                            return;
                        }
                    } else {
                        try {
                            d(i, i2, dVar, nVar);
                        } catch (IOException e3) {
                            e = e3;
                            f.g0.c.g(this.f5793e);
                            f.g0.c.g(this.f5792d);
                            this.f5793e = null;
                            this.f5792d = null;
                            this.i = null;
                            this.j = null;
                            this.f5794f = null;
                            this.f5795g = null;
                            this.f5796h = null;
                            InetSocketAddress inetSocketAddress = this.f5791c.f5752c;
                            Objects.requireNonNull(nVar);
                            z2 = false;
                            if (eVar != null) {
                                eVar = new e(e);
                            } else {
                                IOException iOException = eVar.f5799c;
                                Method method = e.f5798b;
                                if (method != null) {
                                    try {
                                        method.invoke(e, iOException);
                                    } catch (IllegalAccessException | InvocationTargetException unused) {
                                    }
                                }
                                eVar.f5799c = e;
                            }
                            if (z) {
                                break;
                            }
                            bVar.f5789d = true;
                            if (bVar.f5788c) {
                                if (e instanceof ProtocolException) {
                                    continue;
                                } else if (e instanceof InterruptedIOException) {
                                    continue;
                                } else {
                                    boolean z3 = e instanceof SSLHandshakeException;
                                    if (z3 && (e.getCause() instanceof CertificateException)) {
                                        continue;
                                    } else if (e instanceof SSLPeerUnverifiedException) {
                                        continue;
                                    } else if (z3 || (e instanceof SSLProtocolException)) {
                                        z2 = true;
                                        continue;
                                    } else {
                                        continue;
                                    }
                                }
                            }
                            if (!z2) {
                            }
                            throw eVar;
                        }
                    }
                    f(bVar, i4, dVar, nVar);
                    InetSocketAddress inetSocketAddress2 = this.f5791c.f5752c;
                    Objects.requireNonNull(nVar);
                    if (!this.f5791c.a()) {
                    }
                    if (this.f5796h == null) {
                    }
                } catch (IOException e4) {
                    e = e4;
                    f.g0.c.g(this.f5793e);
                    f.g0.c.g(this.f5792d);
                    this.f5793e = null;
                    this.f5792d = null;
                    this.i = null;
                    this.j = null;
                    this.f5794f = null;
                    this.f5795g = null;
                    this.f5796h = null;
                    InetSocketAddress inetSocketAddress3 = this.f5791c.f5752c;
                    Objects.requireNonNull(nVar);
                    z2 = false;
                    if (eVar != null) {
                    }
                    if (z) {
                    }
                    throw eVar;
                }
            } while (!z2);
            throw eVar;
        }
        throw new IllegalStateException("already connected");
    }

    public final void d(int i, int i2, f.d dVar, n nVar) {
        Socket createSocket;
        e0 e0Var = this.f5791c;
        Proxy proxy = e0Var.f5751b;
        f.a aVar = e0Var.f5750a;
        if (proxy.type() != Proxy.Type.DIRECT && proxy.type() != Proxy.Type.HTTP) {
            createSocket = new Socket(proxy);
        } else {
            createSocket = aVar.f5717c.createSocket();
        }
        this.f5792d = createSocket;
        InetSocketAddress inetSocketAddress = this.f5791c.f5752c;
        Objects.requireNonNull(nVar);
        this.f5792d.setSoTimeout(i2);
        try {
            f.g0.j.f.f6032a.f(this.f5792d, this.f5791c.f5752c, i);
            try {
                this.i = new s(o.e(this.f5792d));
                this.j = new r(o.b(this.f5792d));
            } catch (NullPointerException e2) {
                if ("throw with null exception".equals(e2.getMessage())) {
                    throw new IOException(e2);
                }
            }
        } catch (ConnectException e3) {
            StringBuilder x = c.b.a.a.a.x("Failed to connect to ");
            x.append(this.f5791c.f5752c);
            ConnectException connectException = new ConnectException(x.toString());
            connectException.initCause(e3);
            throw connectException;
        }
    }

    public final void e(int i, int i2, int i3, f.d dVar, n nVar) {
        y.a aVar = new y.a();
        aVar.e(this.f5791c.f5750a.f5715a);
        aVar.b(HttpHeaders.HOST, f.g0.c.o(this.f5791c.f5750a.f5715a, true));
        q.a aVar2 = aVar.f6158c;
        aVar2.b("Proxy-Connection", "Keep-Alive");
        aVar2.c("Proxy-Connection");
        aVar2.f6085a.add("Proxy-Connection");
        aVar2.f6085a.add("Keep-Alive");
        q.a aVar3 = aVar.f6158c;
        aVar3.b("User-Agent", "okhttp/3.10.0");
        aVar3.c("User-Agent");
        aVar3.f6085a.add("User-Agent");
        aVar3.f6085a.add("okhttp/3.10.0");
        y a2 = aVar.a();
        f.r rVar = a2.f6150a;
        d(i, i2, dVar, nVar);
        String str = "CONNECT " + f.g0.c.o(rVar, true) + " HTTP/1.1";
        g.g gVar = this.i;
        g.f fVar = this.j;
        f.g0.h.a aVar4 = new f.g0.h.a(null, null, gVar, fVar);
        g.y b2 = gVar.b();
        long j = i2;
        TimeUnit timeUnit = TimeUnit.MILLISECONDS;
        b2.g(j, timeUnit);
        this.j.b().g(i3, timeUnit);
        aVar4.k(a2.f6152c, str);
        fVar.flush();
        b0.a d2 = aVar4.d(false);
        d2.f5731a = a2;
        b0 a3 = d2.a();
        long a4 = f.g0.g.e.a(a3);
        if (a4 == -1) {
            a4 = 0;
        }
        x h2 = aVar4.h(a4);
        f.g0.c.v(h2, Integer.MAX_VALUE, timeUnit);
        ((a.f) h2).close();
        int i4 = a3.f5726d;
        if (i4 == 200) {
            if (!this.i.a().f() || !this.j.a().f()) {
                throw new IOException("TLS tunnel buffered too many bytes!");
            }
        } else if (i4 == 407) {
            Objects.requireNonNull(this.f5791c.f5750a.f5718d);
            throw new IOException("Failed to authenticate with proxy");
        } else {
            StringBuilder x = c.b.a.a.a.x("Unexpected response code for CONNECT: ");
            x.append(a3.f5726d);
            throw new IOException(x.toString());
        }
    }

    public final void f(b bVar, int i, f.d dVar, n nVar) {
        SSLSocket sSLSocket;
        int a2;
        w wVar = w.HTTP_1_1;
        if (this.f5791c.f5750a.i == null) {
            this.f5795g = wVar;
            this.f5793e = this.f5792d;
            return;
        }
        Objects.requireNonNull(nVar);
        f.a aVar = this.f5791c.f5750a;
        SSLSocketFactory sSLSocketFactory = aVar.i;
        try {
            try {
                Socket socket = this.f5792d;
                f.r rVar = aVar.f5715a;
                sSLSocket = (SSLSocket) sSLSocketFactory.createSocket(socket, rVar.f6090e, rVar.f6091f, true);
                try {
                    i a3 = bVar.a(sSLSocket);
                    if (a3.f6057e) {
                        f.g0.j.f.f6032a.e(sSLSocket, aVar.f5715a.f6090e, aVar.f5719e);
                    }
                    sSLSocket.startHandshake();
                    SSLSession session = sSLSocket.getSession();
                    if (("NONE".equals(session.getProtocol()) || "SSL_NULL_WITH_NULL_NULL".equals(session.getCipherSuite())) ? false : true) {
                        p a4 = p.a(session);
                        if (aVar.j.verify(aVar.f5715a.f6090e, session)) {
                            aVar.k.a(aVar.f5715a.f6090e, a4.f6082c);
                            String h2 = a3.f6057e ? f.g0.j.f.f6032a.h(sSLSocket) : null;
                            this.f5793e = sSLSocket;
                            this.i = new s(o.e(sSLSocket));
                            this.j = new r(o.b(this.f5793e));
                            this.f5794f = a4;
                            if (h2 != null) {
                                wVar = w.a(h2);
                            }
                            this.f5795g = wVar;
                            f.g0.j.f.f6032a.a(sSLSocket);
                            if (this.f5795g == w.HTTP_2) {
                                this.f5793e.setSoTimeout(0);
                                g.c cVar = new g.c(true);
                                Socket socket2 = this.f5793e;
                                String str = this.f5791c.f5750a.f5715a.f6090e;
                                g.g gVar = this.i;
                                g.f fVar = this.j;
                                cVar.f5927a = socket2;
                                cVar.f5928b = str;
                                cVar.f5929c = gVar;
                                cVar.f5930d = fVar;
                                cVar.f5931e = this;
                                cVar.f5932f = i;
                                f.g0.i.g gVar2 = new f.g0.i.g(cVar);
                                this.f5796h = gVar2;
                                f.g0.i.q qVar = gVar2.t;
                                synchronized (qVar) {
                                    if (!qVar.f5994g) {
                                        if (qVar.f5991d) {
                                            Logger logger = f.g0.i.q.f5989b;
                                            if (logger.isLoggable(Level.FINE)) {
                                                logger.fine(f.g0.c.n(">> CONNECTION %s", f.g0.i.e.f5899a.g()));
                                            }
                                            qVar.f5990c.write(f.g0.i.e.f5899a.o());
                                            qVar.f5990c.flush();
                                        }
                                    } else {
                                        throw new IOException("closed");
                                    }
                                }
                                f.g0.i.q qVar2 = gVar2.t;
                                t tVar = gVar2.p;
                                synchronized (qVar2) {
                                    if (!qVar2.f5994g) {
                                        qVar2.D(0, Integer.bitCount(tVar.f6004a) * 6, (byte) 4, (byte) 0);
                                        int i2 = 0;
                                        while (i2 < 10) {
                                            if (((1 << i2) & tVar.f6004a) != 0) {
                                                qVar2.f5990c.writeShort(i2 == 4 ? 3 : i2 == 7 ? 4 : i2);
                                                qVar2.f5990c.writeInt(tVar.f6005b[i2]);
                                            }
                                            i2++;
                                        }
                                        qVar2.f5990c.flush();
                                    } else {
                                        throw new IOException("closed");
                                    }
                                }
                                if (gVar2.p.a() != 65535) {
                                    gVar2.t.I(0, a2 - 65535);
                                }
                                new Thread(gVar2.u).start();
                                return;
                            }
                            return;
                        }
                        X509Certificate x509Certificate = (X509Certificate) a4.f6082c.get(0);
                        throw new SSLPeerUnverifiedException("Hostname " + aVar.f5715a.f6090e + " not verified:\n    certificate: " + f.f.b(x509Certificate) + "\n    DN: " + x509Certificate.getSubjectDN().getName() + "\n    subjectAltNames: " + f.g0.l.d.a(x509Certificate));
                    }
                    throw new IOException("a valid ssl session was not established");
                } catch (AssertionError e2) {
                    e = e2;
                    if (!f.g0.c.t(e)) {
                        throw e;
                    }
                    throw new IOException(e);
                } catch (Throwable th) {
                    th = th;
                    if (sSLSocket != null) {
                        f.g0.j.f.f6032a.a(sSLSocket);
                    }
                    f.g0.c.g(sSLSocket);
                    throw th;
                }
            } catch (AssertionError e3) {
                e = e3;
            }
        } catch (Throwable th2) {
            th = th2;
            sSLSocket = null;
        }
    }

    public boolean g(f.a aVar, e0 e0Var) {
        if (this.n.size() < this.m && !this.k) {
            f.g0.a aVar2 = f.g0.a.f5771a;
            f.a aVar3 = this.f5791c.f5750a;
            Objects.requireNonNull((v.a) aVar2);
            if (!aVar3.a(aVar)) {
                return false;
            }
            if (aVar.f5715a.f6090e.equals(this.f5791c.f5750a.f5715a.f6090e)) {
                return true;
            }
            if (this.f5796h == null || e0Var == null || e0Var.f5751b.type() != Proxy.Type.DIRECT || this.f5791c.f5751b.type() != Proxy.Type.DIRECT || !this.f5791c.f5752c.equals(e0Var.f5752c) || e0Var.f5750a.j != f.g0.l.d.f6044a || !j(aVar.f5715a)) {
                return false;
            }
            try {
                aVar.k.a(aVar.f5715a.f6090e, this.f5794f.f6082c);
                return true;
            } catch (SSLPeerUnverifiedException unused) {
            }
        }
        return false;
    }

    public boolean h() {
        return this.f5796h != null;
    }

    public f.g0.g.c i(v vVar, s.a aVar, g gVar) {
        if (this.f5796h != null) {
            return new f.g0.i.f(vVar, aVar, gVar, this.f5796h);
        }
        f.g0.g.f fVar = (f.g0.g.f) aVar;
        this.f5793e.setSoTimeout(fVar.j);
        TimeUnit timeUnit = TimeUnit.MILLISECONDS;
        this.i.b().g(fVar.j, timeUnit);
        this.j.b().g(fVar.k, timeUnit);
        return new f.g0.h.a(vVar, gVar, this.i, this.j);
    }

    public boolean j(f.r rVar) {
        int i = rVar.f6091f;
        f.r rVar2 = this.f5791c.f5750a.f5715a;
        if (i != rVar2.f6091f) {
            return false;
        }
        if (rVar.f6090e.equals(rVar2.f6090e)) {
            return true;
        }
        p pVar = this.f5794f;
        return pVar != null && f.g0.l.d.f6044a.c(rVar.f6090e, (X509Certificate) pVar.f6082c.get(0));
    }

    public String toString() {
        StringBuilder x = c.b.a.a.a.x("Connection{");
        x.append(this.f5791c.f5750a.f5715a.f6090e);
        x.append(":");
        x.append(this.f5791c.f5750a.f5715a.f6091f);
        x.append(", proxy=");
        x.append(this.f5791c.f5751b);
        x.append(" hostAddress=");
        x.append(this.f5791c.f5752c);
        x.append(" cipherSuite=");
        p pVar = this.f5794f;
        x.append(pVar != null ? pVar.f6081b : "none");
        x.append(" protocol=");
        x.append(this.f5795g);
        x.append('}');
        return x.toString();
    }
}