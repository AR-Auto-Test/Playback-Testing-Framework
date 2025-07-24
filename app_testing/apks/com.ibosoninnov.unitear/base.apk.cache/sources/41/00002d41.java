package f.g0.j;

import f.w;
import java.lang.reflect.InvocationHandler;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.lang.reflect.Proxy;
import java.util.List;
import javax.net.ssl.SSLSocket;

/* compiled from: JdkWithJettyBootPlatform.java */
/* loaded from: classes2.dex */
public class d extends f {

    /* renamed from: c  reason: collision with root package name */
    public final Method f6021c;

    /* renamed from: d  reason: collision with root package name */
    public final Method f6022d;

    /* renamed from: e  reason: collision with root package name */
    public final Method f6023e;

    /* renamed from: f  reason: collision with root package name */
    public final Class<?> f6024f;

    /* renamed from: g  reason: collision with root package name */
    public final Class<?> f6025g;

    /* compiled from: JdkWithJettyBootPlatform.java */
    /* loaded from: classes2.dex */
    public static class a implements InvocationHandler {

        /* renamed from: a  reason: collision with root package name */
        public final List<String> f6026a;

        /* renamed from: b  reason: collision with root package name */
        public boolean f6027b;

        /* renamed from: c  reason: collision with root package name */
        public String f6028c;

        public a(List<String> list) {
            this.f6026a = list;
        }

        @Override // java.lang.reflect.InvocationHandler
        public Object invoke(Object obj, Method method, Object[] objArr) {
            String name = method.getName();
            Class<?> returnType = method.getReturnType();
            if (objArr == null) {
                objArr = f.g0.c.f5774b;
            }
            if (name.equals("supports") && Boolean.TYPE == returnType) {
                return Boolean.TRUE;
            }
            if (name.equals("unsupported") && Void.TYPE == returnType) {
                this.f6027b = true;
                return null;
            } else if (name.equals("protocols") && objArr.length == 0) {
                return this.f6026a;
            } else {
                if ((name.equals("selectProtocol") || name.equals("select")) && String.class == returnType && objArr.length == 1 && (objArr[0] instanceof List)) {
                    List list = (List) objArr[0];
                    int size = list.size();
                    for (int i = 0; i < size; i++) {
                        if (this.f6026a.contains(list.get(i))) {
                            String str = (String) list.get(i);
                            this.f6028c = str;
                            return str;
                        }
                    }
                    String str2 = this.f6026a.get(0);
                    this.f6028c = str2;
                    return str2;
                } else if ((name.equals("protocolSelected") || name.equals("selected")) && objArr.length == 1) {
                    this.f6028c = (String) objArr[0];
                    return null;
                } else {
                    return method.invoke(this, objArr);
                }
            }
        }
    }

    public d(Method method, Method method2, Method method3, Class<?> cls, Class<?> cls2) {
        this.f6021c = method;
        this.f6022d = method2;
        this.f6023e = method3;
        this.f6024f = cls;
        this.f6025g = cls2;
    }

    @Override // f.g0.j.f
    public void a(SSLSocket sSLSocket) {
        try {
            this.f6023e.invoke(null, sSLSocket);
        } catch (IllegalAccessException | InvocationTargetException e2) {
            throw f.g0.c.a("unable to remove alpn", e2);
        }
    }

    @Override // f.g0.j.f
    public void e(SSLSocket sSLSocket, String str, List<w> list) {
        try {
            this.f6021c.invoke(null, sSLSocket, Proxy.newProxyInstance(f.class.getClassLoader(), new Class[]{this.f6024f, this.f6025g}, new a(f.b(list))));
        } catch (IllegalAccessException | InvocationTargetException e2) {
            throw f.g0.c.a("unable to set alpn", e2);
        }
    }

    @Override // f.g0.j.f
    public String h(SSLSocket sSLSocket) {
        try {
            a aVar = (a) Proxy.getInvocationHandler(this.f6022d.invoke(null, sSLSocket));
            boolean z = aVar.f6027b;
            if (!z && aVar.f6028c == null) {
                f.f6032a.k(4, "ALPN callback dropped: HTTP/2 is disabled. Is alpn-boot on the boot class path?", null);
                return null;
            } else if (z) {
                return null;
            } else {
                return aVar.f6028c;
            }
        } catch (IllegalAccessException | InvocationTargetException e2) {
            throw f.g0.c.a("unable to get selected protocol", e2);
        }
    }
}