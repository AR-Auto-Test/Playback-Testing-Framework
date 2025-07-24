package f.g0.g;

import com.google.common.net.HttpHeaders;
import f.a0;
import f.b0;
import f.e0;
import f.g0.f.f;
import f.n;
import f.r;
import f.s;
import f.v;
import f.y;
import java.io.IOException;
import java.io.InterruptedIOException;
import java.net.ProtocolException;
import java.net.Proxy;
import java.net.SocketTimeoutException;
import java.security.cert.CertificateException;
import java.util.Objects;
import javax.net.ssl.HostnameVerifier;
import javax.net.ssl.SSLHandshakeException;
import javax.net.ssl.SSLPeerUnverifiedException;
import javax.net.ssl.SSLSocketFactory;

/* compiled from: RetryAndFollowUpInterceptor.java */
/* loaded from: classes2.dex */
public final class h implements s {

    /* renamed from: a  reason: collision with root package name */
    public final v f5836a;

    /* renamed from: b  reason: collision with root package name */
    public final boolean f5837b;

    /* renamed from: c  reason: collision with root package name */
    public volatile f.g0.f.g f5838c;

    /* renamed from: d  reason: collision with root package name */
    public Object f5839d;

    /* renamed from: e  reason: collision with root package name */
    public volatile boolean f5840e;

    public h(v vVar, boolean z) {
        this.f5836a = vVar;
        this.f5837b = z;
    }

    /* JADX WARN: Code restructure failed: missing block: B:59:0x00e4, code lost:
        if (r5.equals("HEAD") == false) goto L93;
     */
    /* JADX WARN: Removed duplicated region for block: B:105:0x0191  */
    /* JADX WARN: Removed duplicated region for block: B:148:0x0189 A[SYNTHETIC] */
    /* JADX WARN: Removed duplicated region for block: B:64:0x00ee  */
    @Override // f.s
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public b0 a(s.a aVar) {
        b0 b2;
        e0 e0Var;
        int i;
        String str;
        y yVar;
        c cVar;
        y yVar2;
        Proxy proxy;
        b0 b0Var;
        f fVar = (f) aVar;
        y yVar3 = fVar.f5830f;
        f.d dVar = fVar.f5831g;
        n nVar = fVar.f5832h;
        f.g0.f.g gVar = new f.g0.f.g(this.f5836a.s, b(yVar3.f6150a), dVar, nVar, this.f5839d);
        this.f5838c = gVar;
        int i2 = 0;
        b0 b0Var2 = null;
        while (!this.f5840e) {
            boolean z = true;
            try {
                try {
                    b2 = fVar.b(yVar3, gVar, null, null);
                    if (b0Var2 != null) {
                        b0.a aVar2 = new b0.a(b2);
                        b0.a aVar3 = new b0.a(b0Var2);
                        aVar3.f5737g = null;
                        b0 a2 = aVar3.a();
                        if (a2.f5730h == null) {
                            aVar2.j = a2;
                            b2 = aVar2.a();
                        } else {
                            throw new IllegalArgumentException("priorResponse.body != null");
                        }
                    }
                    e0Var = gVar.f5812c;
                    i = b2.f5726d;
                    str = b2.f5724b.f6151b;
                } catch (f.g0.f.e e2) {
                    if (!c(e2.f5799c, gVar, false, yVar3)) {
                        throw e2.f5799c;
                    }
                } catch (IOException e3) {
                    if (e3 instanceof f.g0.i.a) {
                        z = false;
                    }
                    if (!c(e3, gVar, z, yVar3)) {
                        throw e3;
                    }
                }
                if (i == 307 || i == 308) {
                    if (!str.equals("GET")) {
                    }
                    if (this.f5836a.v) {
                        String a3 = b2.f5729g.a(HttpHeaders.LOCATION);
                        if (a3 == null) {
                            a3 = null;
                        }
                        if (a3 != null) {
                            r rVar = b2.f5724b.f6150a;
                            Objects.requireNonNull(rVar);
                            r.a aVar4 = new r.a();
                            if (aVar4.c(rVar, a3) != 1) {
                                aVar4 = null;
                            }
                            r a4 = aVar4 != null ? aVar4.a() : null;
                            if (a4 != null && (a4.f6087b.equals(b2.f5724b.f6150a.f6087b) || this.f5836a.u)) {
                                y.a aVar5 = new y.a(b2.f5724b);
                                if (b.v.u.c.w(str)) {
                                    boolean equals = str.equals("PROPFIND");
                                    if (true ^ str.equals("PROPFIND")) {
                                        aVar5.c("GET", null);
                                    } else {
                                        aVar5.c(str, equals ? b2.f5724b.f6153d : null);
                                    }
                                    if (!equals) {
                                        aVar5.f6158c.c(HttpHeaders.TRANSFER_ENCODING);
                                        aVar5.f6158c.c(HttpHeaders.CONTENT_LENGTH);
                                        aVar5.f6158c.c(HttpHeaders.CONTENT_TYPE);
                                    }
                                }
                                if (!e(b2, a4)) {
                                    aVar5.f6158c.c(HttpHeaders.AUTHORIZATION);
                                }
                                aVar5.e(a4);
                                yVar2 = aVar5.a();
                                yVar = yVar2;
                                if (yVar != null) {
                                    if (!this.f5837b) {
                                        gVar.g();
                                    }
                                    return b2;
                                }
                                f.g0.c.f(b2.f5730h);
                                int i3 = i2 + 1;
                                if (i3 <= 20) {
                                    if (!e(b2, yVar.f6150a)) {
                                        gVar.g();
                                        gVar = new f.g0.f.g(this.f5836a.s, b(yVar.f6150a), dVar, nVar, this.f5839d);
                                        this.f5838c = gVar;
                                    } else {
                                        synchronized (gVar.f5813d) {
                                            cVar = gVar.n;
                                        }
                                        if (cVar != null) {
                                            throw new IllegalStateException("Closing the body of " + b2 + " didn't close its backing stream. Bad interceptor?");
                                        }
                                    }
                                    b0Var2 = b2;
                                    yVar3 = yVar;
                                    i2 = i3;
                                } else {
                                    gVar.g();
                                    throw new ProtocolException(c.b.a.a.a.j("Too many follow-up requests: ", i3));
                                }
                            }
                        }
                    }
                    yVar = null;
                    if (yVar != null) {
                    }
                } else {
                    if (i != 401) {
                        if (i == 503) {
                            b0 b0Var3 = b2.k;
                            if ((b0Var3 == null || b0Var3.f5726d != 503) && d(b2, Integer.MAX_VALUE) == 0) {
                                yVar2 = b2.f5724b;
                                yVar = yVar2;
                            }
                        } else if (i == 407) {
                            if (e0Var != null) {
                                proxy = e0Var.f5751b;
                            } else {
                                Objects.requireNonNull(this.f5836a);
                                proxy = null;
                            }
                            if (proxy.type() == Proxy.Type.HTTP) {
                                Objects.requireNonNull(this.f5836a.q);
                            } else {
                                throw new ProtocolException("Received HTTP_PROXY_AUTH (407) code while not using proxy");
                            }
                        } else if (i == 408) {
                            if (this.f5836a.w && (((b0Var = b2.k) == null || b0Var.f5726d != 408) && d(b2, 0) <= 0)) {
                                yVar2 = b2.f5724b;
                                yVar = yVar2;
                            }
                        } else {
                            switch (i) {
                                case 300:
                                case 301:
                                case 302:
                                case 303:
                                    if (this.f5836a.v) {
                                    }
                                    break;
                                default:
                                    yVar = null;
                                    break;
                            }
                        }
                        if (yVar != null) {
                        }
                    } else {
                        Objects.requireNonNull(this.f5836a.r);
                    }
                    yVar = null;
                    if (yVar != null) {
                    }
                }
            } catch (Throwable th) {
                gVar.h(null);
                gVar.g();
                throw th;
            }
        }
        gVar.g();
        throw new IOException("Canceled");
    }

    public final f.a b(r rVar) {
        SSLSocketFactory sSLSocketFactory;
        HostnameVerifier hostnameVerifier;
        f.f fVar;
        if (rVar.f6087b.equals("https")) {
            v vVar = this.f5836a;
            SSLSocketFactory sSLSocketFactory2 = vVar.m;
            HostnameVerifier hostnameVerifier2 = vVar.o;
            fVar = vVar.p;
            sSLSocketFactory = sSLSocketFactory2;
            hostnameVerifier = hostnameVerifier2;
        } else {
            sSLSocketFactory = null;
            hostnameVerifier = null;
            fVar = null;
        }
        String str = rVar.f6090e;
        int i = rVar.f6091f;
        v vVar2 = this.f5836a;
        return new f.a(str, i, vVar2.t, vVar2.l, sSLSocketFactory, hostnameVerifier, fVar, vVar2.q, null, vVar2.f6123e, vVar2.f6124f, vVar2.j);
    }

    public final boolean c(IOException iOException, f.g0.f.g gVar, boolean z, y yVar) {
        f.a aVar;
        gVar.h(iOException);
        if (this.f5836a.w) {
            if (z) {
                a0 a0Var = yVar.f6153d;
            }
            if (!(iOException instanceof ProtocolException) && (!(iOException instanceof InterruptedIOException) ? ((iOException instanceof SSLHandshakeException) && (iOException.getCause() instanceof CertificateException)) || (iOException instanceof SSLPeerUnverifiedException) : !((iOException instanceof SocketTimeoutException) && !z))) {
                return gVar.f5812c != null || (((aVar = gVar.f5811b) != null && aVar.a()) || gVar.f5817h.b());
            }
            return false;
        }
        return false;
    }

    public final int d(b0 b0Var, int i) {
        String a2 = b0Var.f5729g.a(HttpHeaders.RETRY_AFTER);
        if (a2 == null) {
            a2 = null;
        }
        if (a2 == null) {
            return i;
        }
        if (a2.matches("\\d+")) {
            return Integer.valueOf(a2).intValue();
        }
        return Integer.MAX_VALUE;
    }

    public final boolean e(b0 b0Var, r rVar) {
        r rVar2 = b0Var.f5724b.f6150a;
        return rVar2.f6090e.equals(rVar.f6090e) && rVar2.f6091f == rVar.f6091f && rVar2.f6087b.equals(rVar.f6087b);
    }
}