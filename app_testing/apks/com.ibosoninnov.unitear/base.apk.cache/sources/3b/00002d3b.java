package f.g0.j;

import android.os.Build;
import android.util.Log;
import f.w;
import java.io.EOFException;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.net.InetSocketAddress;
import java.net.Socket;
import java.security.cert.Certificate;
import java.security.cert.TrustAnchor;
import java.security.cert.X509Certificate;
import java.util.List;
import java.util.Objects;
import javax.net.ssl.SSLPeerUnverifiedException;
import javax.net.ssl.SSLSocket;
import javax.net.ssl.X509TrustManager;

/* compiled from: AndroidPlatform.java */
/* loaded from: classes2.dex */
public class a extends f {

    /* renamed from: c  reason: collision with root package name */
    public final e<Socket> f6007c;

    /* renamed from: d  reason: collision with root package name */
    public final e<Socket> f6008d;

    /* renamed from: e  reason: collision with root package name */
    public final e<Socket> f6009e;

    /* renamed from: f  reason: collision with root package name */
    public final e<Socket> f6010f;

    /* renamed from: g  reason: collision with root package name */
    public final c f6011g;

    /* compiled from: AndroidPlatform.java */
    /* renamed from: f.g0.j.a$a  reason: collision with other inner class name */
    /* loaded from: classes2.dex */
    public static final class C0128a extends f.g0.l.c {

        /* renamed from: a  reason: collision with root package name */
        public final Object f6012a;

        /* renamed from: b  reason: collision with root package name */
        public final Method f6013b;

        public C0128a(Object obj, Method method) {
            this.f6012a = obj;
            this.f6013b = method;
        }

        @Override // f.g0.l.c
        public List<Certificate> a(List<Certificate> list, String str) {
            try {
                return (List) this.f6013b.invoke(this.f6012a, (X509Certificate[]) list.toArray(new X509Certificate[list.size()]), "RSA", str);
            } catch (IllegalAccessException e2) {
                throw new AssertionError(e2);
            } catch (InvocationTargetException e3) {
                SSLPeerUnverifiedException sSLPeerUnverifiedException = new SSLPeerUnverifiedException(e3.getMessage());
                sSLPeerUnverifiedException.initCause(e3);
                throw sSLPeerUnverifiedException;
            }
        }

        public boolean equals(Object obj) {
            return obj instanceof C0128a;
        }

        public int hashCode() {
            return 0;
        }
    }

    /* compiled from: AndroidPlatform.java */
    /* loaded from: classes2.dex */
    public static final class b implements f.g0.l.e {

        /* renamed from: a  reason: collision with root package name */
        public final X509TrustManager f6014a;

        /* renamed from: b  reason: collision with root package name */
        public final Method f6015b;

        public b(X509TrustManager x509TrustManager, Method method) {
            this.f6015b = method;
            this.f6014a = x509TrustManager;
        }

        @Override // f.g0.l.e
        public X509Certificate a(X509Certificate x509Certificate) {
            try {
                TrustAnchor trustAnchor = (TrustAnchor) this.f6015b.invoke(this.f6014a, x509Certificate);
                if (trustAnchor != null) {
                    return trustAnchor.getTrustedCert();
                }
                return null;
            } catch (IllegalAccessException e2) {
                throw f.g0.c.a("unable to get issues and signature", e2);
            } catch (InvocationTargetException unused) {
                return null;
            }
        }

        public boolean equals(Object obj) {
            if (obj == this) {
                return true;
            }
            if (obj instanceof b) {
                b bVar = (b) obj;
                return this.f6014a.equals(bVar.f6014a) && this.f6015b.equals(bVar.f6015b);
            }
            return false;
        }

        public int hashCode() {
            return (this.f6015b.hashCode() * 31) + this.f6014a.hashCode();
        }
    }

    /* compiled from: AndroidPlatform.java */
    /* loaded from: classes2.dex */
    public static final class c {

        /* renamed from: a  reason: collision with root package name */
        public final Method f6016a;

        /* renamed from: b  reason: collision with root package name */
        public final Method f6017b;

        /* renamed from: c  reason: collision with root package name */
        public final Method f6018c;

        public c(Method method, Method method2, Method method3) {
            this.f6016a = method;
            this.f6017b = method2;
            this.f6018c = method3;
        }
    }

    public a(Class<?> cls, e<Socket> eVar, e<Socket> eVar2, e<Socket> eVar3, e<Socket> eVar4) {
        Method method;
        Method method2;
        Method method3 = null;
        try {
            Class<?> cls2 = Class.forName("dalvik.system.CloseGuard");
            Method method4 = cls2.getMethod("get", new Class[0]);
            method2 = cls2.getMethod("open", String.class);
            method = cls2.getMethod("warnIfOpen", new Class[0]);
            method3 = method4;
        } catch (Exception unused) {
            method = null;
            method2 = null;
        }
        this.f6011g = new c(method3, method2, method);
        this.f6007c = eVar;
        this.f6008d = eVar2;
        this.f6009e = eVar3;
        this.f6010f = eVar4;
    }

    @Override // f.g0.j.f
    public f.g0.l.c c(X509TrustManager x509TrustManager) {
        try {
            Class<?> cls = Class.forName("android.net.http.X509TrustManagerExtensions");
            return new C0128a(cls.getConstructor(X509TrustManager.class).newInstance(x509TrustManager), cls.getMethod("checkServerTrusted", X509Certificate[].class, String.class, String.class));
        } catch (Exception unused) {
            return new f.g0.l.a(d(x509TrustManager));
        }
    }

    @Override // f.g0.j.f
    public f.g0.l.e d(X509TrustManager x509TrustManager) {
        try {
            Method declaredMethod = x509TrustManager.getClass().getDeclaredMethod("findTrustAnchorByIssuerAndSignature", X509Certificate.class);
            declaredMethod.setAccessible(true);
            return new b(x509TrustManager, declaredMethod);
        } catch (NoSuchMethodException unused) {
            return new f.g0.l.b(x509TrustManager.getAcceptedIssuers());
        }
    }

    @Override // f.g0.j.f
    public void e(SSLSocket sSLSocket, String str, List<w> list) {
        if (str != null) {
            this.f6007c.c(sSLSocket, Boolean.TRUE);
            this.f6008d.c(sSLSocket, str);
        }
        e<Socket> eVar = this.f6010f;
        if (eVar != null) {
            if (eVar.a(sSLSocket.getClass()) != null) {
                Object[] objArr = new Object[1];
                g.e eVar2 = new g.e();
                int size = list.size();
                for (int i = 0; i < size; i++) {
                    w wVar = list.get(i);
                    if (wVar != w.HTTP_1_0) {
                        eVar2.T(wVar.f6141h.length());
                        eVar2.Y(wVar.f6141h);
                    }
                }
                try {
                    objArr[0] = eVar2.r(eVar2.f6176d);
                    this.f6010f.d(sSLSocket, objArr);
                } catch (EOFException e2) {
                    throw new AssertionError(e2);
                }
            }
        }
    }

    @Override // f.g0.j.f
    public void f(Socket socket, InetSocketAddress inetSocketAddress, int i) {
        try {
            socket.connect(inetSocketAddress, i);
        } catch (AssertionError e2) {
            if (!f.g0.c.t(e2)) {
                throw e2;
            }
            throw new IOException(e2);
        } catch (ClassCastException e3) {
            if (Build.VERSION.SDK_INT == 26) {
                IOException iOException = new IOException("Exception in connect");
                iOException.initCause(e3);
                throw iOException;
            }
            throw e3;
        } catch (SecurityException e4) {
            IOException iOException2 = new IOException("Exception in connect");
            iOException2.initCause(e4);
            throw iOException2;
        }
    }

    @Override // f.g0.j.f
    public String h(SSLSocket sSLSocket) {
        byte[] bArr;
        e<Socket> eVar = this.f6009e;
        if (eVar == null) {
            return null;
        }
        if ((eVar.a(sSLSocket.getClass()) != null) && (bArr = (byte[]) this.f6009e.d(sSLSocket, new Object[0])) != null) {
            return new String(bArr, f.g0.c.i);
        }
        return null;
    }

    @Override // f.g0.j.f
    public Object i(String str) {
        c cVar = this.f6011g;
        Method method = cVar.f6016a;
        if (method != null) {
            try {
                Object invoke = method.invoke(null, new Object[0]);
                cVar.f6017b.invoke(invoke, str);
                return invoke;
            } catch (Exception unused) {
                return null;
            }
        }
        return null;
    }

    @Override // f.g0.j.f
    public boolean j(String str) {
        try {
            Class<?> cls = Class.forName("android.security.NetworkSecurityPolicy");
            return m(str, cls, cls.getMethod("getInstance", new Class[0]).invoke(null, new Object[0]));
        } catch (ClassNotFoundException | NoSuchMethodException unused) {
            return true;
        } catch (IllegalAccessException e2) {
            e = e2;
            throw f.g0.c.a("unable to determine cleartext support", e);
        } catch (IllegalArgumentException e3) {
            e = e3;
            throw f.g0.c.a("unable to determine cleartext support", e);
        } catch (InvocationTargetException e4) {
            e = e4;
            throw f.g0.c.a("unable to determine cleartext support", e);
        }
    }

    @Override // f.g0.j.f
    public void k(int i, String str, Throwable th) {
        int min;
        int i2 = i != 5 ? 3 : 5;
        if (th != null) {
            str = str + '\n' + Log.getStackTraceString(th);
        }
        int i3 = 0;
        int length = str.length();
        while (i3 < length) {
            int indexOf = str.indexOf(10, i3);
            if (indexOf == -1) {
                indexOf = length;
            }
            while (true) {
                min = Math.min(indexOf, i3 + 4000);
                Log.println(i2, "OkHttp", str.substring(i3, min));
                if (min >= indexOf) {
                    break;
                }
                i3 = min;
            }
            i3 = min + 1;
        }
    }

    @Override // f.g0.j.f
    public void l(String str, Object obj) {
        c cVar = this.f6011g;
        Objects.requireNonNull(cVar);
        boolean z = false;
        if (obj != null) {
            try {
                cVar.f6018c.invoke(obj, new Object[0]);
                z = true;
            } catch (Exception unused) {
            }
        }
        if (z) {
            return;
        }
        k(5, str, null);
    }

    public final boolean m(String str, Class<?> cls, Object obj) {
        try {
            try {
                return ((Boolean) cls.getMethod("isCleartextTrafficPermitted", String.class).invoke(obj, str)).booleanValue();
            } catch (NoSuchMethodException unused) {
                return true;
            }
        } catch (NoSuchMethodException unused2) {
            return ((Boolean) cls.getMethod("isCleartextTrafficPermitted", new Class[0]).invoke(obj, new Object[0])).booleanValue();
        }
    }
}