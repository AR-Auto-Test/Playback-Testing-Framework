package f.g0.j;

import f.v;
import f.w;
import java.net.InetSocketAddress;
import java.net.Socket;
import java.security.NoSuchAlgorithmException;
import java.security.Security;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.net.ssl.SSLContext;
import javax.net.ssl.SSLParameters;
import javax.net.ssl.SSLSocket;
import javax.net.ssl.X509TrustManager;

/* compiled from: Platform.java */
/* loaded from: classes2.dex */
public class f {

    /* renamed from: a  reason: collision with root package name */
    public static final f f6032a;

    /* renamed from: b  reason: collision with root package name */
    public static final Logger f6033b;

    static {
        Class<?> cls;
        f fVar;
        boolean z;
        e eVar;
        e eVar2;
        d dVar = null;
        try {
            try {
                cls = Class.forName("com.android.org.conscrypt.SSLParametersImpl");
            } catch (ClassNotFoundException unused) {
                cls = Class.forName("org.apache.harmony.xnet.provider.jsse.SSLParametersImpl");
            }
            Class<?> cls2 = cls;
            e eVar3 = new e(null, "setUseSessionTickets", Boolean.TYPE);
            e eVar4 = new e(null, "setHostname", String.class);
            if (Security.getProvider("GMSCore_OpenSSL") == null) {
                try {
                    Class.forName("android.net.Network");
                } catch (ClassNotFoundException unused2) {
                    z = false;
                }
            }
            z = true;
            if (z) {
                eVar = new e(byte[].class, "getAlpnSelectedProtocol", new Class[0]);
                eVar2 = new e(null, "setAlpnProtocols", byte[].class);
            } else {
                eVar = null;
                eVar2 = null;
            }
            fVar = new a(cls2, eVar3, eVar4, eVar, eVar2);
        } catch (ClassNotFoundException unused3) {
            fVar = null;
        }
        if (fVar == null) {
            if (!("conscrypt".equals(System.getProperty("okhttp.platform")) ? true : "Conscrypt".equals(Security.getProviders()[0].getName())) || (fVar = b.m()) == null) {
                try {
                    fVar = new c(SSLParameters.class.getMethod("setApplicationProtocols", String[].class), SSLSocket.class.getMethod("getApplicationProtocol", new Class[0]));
                } catch (NoSuchMethodException unused4) {
                    fVar = null;
                }
                if (fVar == null) {
                    try {
                        Class<?> cls3 = Class.forName("org.eclipse.jetty.alpn.ALPN");
                        dVar = new d(cls3.getMethod("put", SSLSocket.class, Class.forName("org.eclipse.jetty.alpn.ALPN$Provider")), cls3.getMethod("get", SSLSocket.class), cls3.getMethod("remove", SSLSocket.class), Class.forName("org.eclipse.jetty.alpn.ALPN$ClientProvider"), Class.forName("org.eclipse.jetty.alpn.ALPN$ServerProvider"));
                    } catch (ClassNotFoundException | NoSuchMethodException unused5) {
                    }
                    fVar = dVar != null ? dVar : new f();
                }
            }
        }
        f6032a = fVar;
        f6033b = Logger.getLogger(v.class.getName());
    }

    public static List<String> b(List<w> list) {
        ArrayList arrayList = new ArrayList(list.size());
        int size = list.size();
        for (int i = 0; i < size; i++) {
            w wVar = list.get(i);
            if (wVar != w.HTTP_1_0) {
                arrayList.add(wVar.f6141h);
            }
        }
        return arrayList;
    }

    public void a(SSLSocket sSLSocket) {
    }

    public f.g0.l.c c(X509TrustManager x509TrustManager) {
        return new f.g0.l.a(d(x509TrustManager));
    }

    public f.g0.l.e d(X509TrustManager x509TrustManager) {
        return new f.g0.l.b(x509TrustManager.getAcceptedIssuers());
    }

    public void e(SSLSocket sSLSocket, String str, List<w> list) {
    }

    public void f(Socket socket, InetSocketAddress inetSocketAddress, int i) {
        socket.connect(inetSocketAddress, i);
    }

    public SSLContext g() {
        try {
            return SSLContext.getInstance("TLS");
        } catch (NoSuchAlgorithmException e2) {
            throw new IllegalStateException("No TLS provider", e2);
        }
    }

    public String h(SSLSocket sSLSocket) {
        return null;
    }

    public Object i(String str) {
        if (f6033b.isLoggable(Level.FINE)) {
            return new Throwable(str);
        }
        return null;
    }

    public boolean j(String str) {
        return true;
    }

    public void k(int i, String str, Throwable th) {
        f6033b.log(i == 5 ? Level.WARNING : Level.INFO, str, th);
    }

    public void l(String str, Object obj) {
        if (obj == null) {
            str = c.b.a.a.a.q(str, " To see where this was allocated, set the OkHttpClient logger level to FINE: Logger.getLogger(OkHttpClient.class.getName()).setLevel(Level.FINE);");
        }
        k(5, str, (Throwable) obj);
    }
}