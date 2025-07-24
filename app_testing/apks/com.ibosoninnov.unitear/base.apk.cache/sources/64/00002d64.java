package f;

import f.n;
import f.q;
import java.net.ProxySelector;
import java.net.Socket;
import java.security.GeneralSecurityException;
import java.security.KeyStore;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.TimeUnit;
import javax.net.SocketFactory;
import javax.net.ssl.HostnameVerifier;
import javax.net.ssl.SSLContext;
import javax.net.ssl.SSLSocketFactory;
import javax.net.ssl.TrustManager;
import javax.net.ssl.TrustManagerFactory;
import javax.net.ssl.X509TrustManager;

/* compiled from: OkHttpClient.java */
/* loaded from: classes2.dex */
public class v implements Cloneable {

    /* renamed from: b  reason: collision with root package name */
    public static final List<w> f6120b = f.g0.c.q(w.HTTP_2, w.HTTP_1_1);

    /* renamed from: c  reason: collision with root package name */
    public static final List<i> f6121c = f.g0.c.q(i.f6054b, i.f6055c);

    /* renamed from: d  reason: collision with root package name */
    public final l f6122d;

    /* renamed from: e  reason: collision with root package name */
    public final List<w> f6123e;

    /* renamed from: f  reason: collision with root package name */
    public final List<i> f6124f;

    /* renamed from: g  reason: collision with root package name */
    public final List<s> f6125g;

    /* renamed from: h  reason: collision with root package name */
    public final List<s> f6126h;
    public final n.b i;
    public final ProxySelector j;
    public final k k;
    public final SocketFactory l;
    public final SSLSocketFactory m;
    public final f.g0.l.c n;
    public final HostnameVerifier o;
    public final f p;
    public final f.b q;
    public final f.b r;
    public final h s;
    public final m t;
    public final boolean u;
    public final boolean v;
    public final boolean w;
    public final int x;
    public final int y;
    public final int z;

    /* compiled from: OkHttpClient.java */
    /* loaded from: classes2.dex */
    public class a extends f.g0.a {
        @Override // f.g0.a
        public void a(q.a aVar, String str, String str2) {
            aVar.f6085a.add(str);
            aVar.f6085a.add(str2.trim());
        }

        @Override // f.g0.a
        public Socket b(h hVar, f.a aVar, f.g0.f.g gVar) {
            for (f.g0.f.c cVar : hVar.f6049e) {
                if (cVar.g(aVar, null) && cVar.h() && cVar != gVar.b()) {
                    if (gVar.n == null && gVar.j.n.size() == 1) {
                        Socket c2 = gVar.c(true, false, false);
                        gVar.j = cVar;
                        cVar.n.add(gVar.j.n.get(0));
                        return c2;
                    }
                    throw new IllegalStateException();
                }
            }
            return null;
        }

        @Override // f.g0.a
        public f.g0.f.c c(h hVar, f.a aVar, f.g0.f.g gVar, e0 e0Var) {
            for (f.g0.f.c cVar : hVar.f6049e) {
                if (cVar.g(aVar, e0Var)) {
                    gVar.a(cVar, true);
                    return cVar;
                }
            }
            return null;
        }
    }

    /* compiled from: OkHttpClient.java */
    /* loaded from: classes2.dex */
    public static final class b {
        public f.b l;
        public f.b m;
        public h n;
        public m o;
        public boolean p;
        public boolean q;
        public boolean r;
        public int s;
        public int t;
        public int u;

        /* renamed from: d  reason: collision with root package name */
        public final List<s> f6130d = new ArrayList();

        /* renamed from: e  reason: collision with root package name */
        public final List<s> f6131e = new ArrayList();

        /* renamed from: a  reason: collision with root package name */
        public l f6127a = new l();

        /* renamed from: b  reason: collision with root package name */
        public List<w> f6128b = v.f6120b;

        /* renamed from: c  reason: collision with root package name */
        public List<i> f6129c = v.f6121c;

        /* renamed from: f  reason: collision with root package name */
        public n.b f6132f = new o(n.f6078a);

        /* renamed from: g  reason: collision with root package name */
        public ProxySelector f6133g = ProxySelector.getDefault();

        /* renamed from: h  reason: collision with root package name */
        public k f6134h = k.f6072a;
        public SocketFactory i = SocketFactory.getDefault();
        public HostnameVerifier j = f.g0.l.d.f6044a;
        public f k = f.f5753a;

        public b() {
            f.b bVar = f.b.f5723a;
            this.l = bVar;
            this.m = bVar;
            this.n = new h();
            this.o = m.f6077a;
            this.p = true;
            this.q = true;
            this.r = true;
            this.s = 10000;
            this.t = 10000;
            this.u = 10000;
        }

        public b a(long j, TimeUnit timeUnit) {
            this.s = f.g0.c.d("timeout", j, timeUnit);
            return this;
        }

        public b b(long j, TimeUnit timeUnit) {
            this.t = f.g0.c.d("timeout", j, timeUnit);
            return this;
        }

        public b c(long j, TimeUnit timeUnit) {
            this.u = f.g0.c.d("timeout", j, timeUnit);
            return this;
        }
    }

    static {
        f.g0.a.f5771a = new a();
    }

    public v() {
        this(new b());
    }

    public d a(y yVar) {
        x xVar = new x(this, yVar, false);
        xVar.f6144d = ((o) this.i).f6079a;
        return xVar;
    }

    public v(b bVar) {
        boolean z;
        this.f6122d = bVar.f6127a;
        this.f6123e = bVar.f6128b;
        List<i> list = bVar.f6129c;
        this.f6124f = list;
        this.f6125g = f.g0.c.p(bVar.f6130d);
        this.f6126h = f.g0.c.p(bVar.f6131e);
        this.i = bVar.f6132f;
        this.j = bVar.f6133g;
        this.k = bVar.f6134h;
        this.l = bVar.i;
        loop0: while (true) {
            z = false;
            for (i iVar : list) {
                z = (z || iVar.f6056d) ? true : z;
            }
        }
        if (!z) {
            this.m = null;
            this.n = null;
        } else {
            try {
                TrustManagerFactory trustManagerFactory = TrustManagerFactory.getInstance(TrustManagerFactory.getDefaultAlgorithm());
                trustManagerFactory.init((KeyStore) null);
                TrustManager[] trustManagers = trustManagerFactory.getTrustManagers();
                if (trustManagers.length == 1 && (trustManagers[0] instanceof X509TrustManager)) {
                    X509TrustManager x509TrustManager = (X509TrustManager) trustManagers[0];
                    try {
                        f.g0.j.f fVar = f.g0.j.f.f6032a;
                        SSLContext g2 = fVar.g();
                        g2.init(null, new TrustManager[]{x509TrustManager}, null);
                        this.m = g2.getSocketFactory();
                        this.n = fVar.c(x509TrustManager);
                    } catch (GeneralSecurityException e2) {
                        throw f.g0.c.a("No System TLS", e2);
                    }
                } else {
                    throw new IllegalStateException("Unexpected default trust managers:" + Arrays.toString(trustManagers));
                }
            } catch (GeneralSecurityException e3) {
                throw f.g0.c.a("No System TLS", e3);
            }
        }
        this.o = bVar.j;
        f fVar2 = bVar.k;
        f.g0.l.c cVar = this.n;
        this.p = f.g0.c.m(fVar2.f5755c, cVar) ? fVar2 : new f(fVar2.f5754b, cVar);
        this.q = bVar.l;
        this.r = bVar.m;
        this.s = bVar.n;
        this.t = bVar.o;
        this.u = bVar.p;
        this.v = bVar.q;
        this.w = bVar.r;
        this.x = bVar.s;
        this.y = bVar.t;
        this.z = bVar.u;
        if (!this.f6125g.contains(null)) {
            if (this.f6126h.contains(null)) {
                StringBuilder x = c.b.a.a.a.x("Null network interceptor: ");
                x.append(this.f6126h);
                throw new IllegalStateException(x.toString());
            }
            return;
        }
        StringBuilder x2 = c.b.a.a.a.x("Null interceptor: ");
        x2.append(this.f6125g);
        throw new IllegalStateException(x2.toString());
    }
}