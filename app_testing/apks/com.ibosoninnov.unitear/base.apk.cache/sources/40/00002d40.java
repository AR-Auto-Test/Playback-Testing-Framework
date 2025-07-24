package f.g0.j;

import f.w;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;
import javax.net.ssl.SSLParameters;
import javax.net.ssl.SSLSocket;

/* compiled from: Jdk9Platform.java */
/* loaded from: classes2.dex */
public final class c extends f {

    /* renamed from: c  reason: collision with root package name */
    public final Method f6019c;

    /* renamed from: d  reason: collision with root package name */
    public final Method f6020d;

    public c(Method method, Method method2) {
        this.f6019c = method;
        this.f6020d = method2;
    }

    @Override // f.g0.j.f
    public void e(SSLSocket sSLSocket, String str, List<w> list) {
        try {
            SSLParameters sSLParameters = sSLSocket.getSSLParameters();
            ArrayList arrayList = (ArrayList) f.b(list);
            this.f6019c.invoke(sSLParameters, arrayList.toArray(new String[arrayList.size()]));
            sSLSocket.setSSLParameters(sSLParameters);
        } catch (IllegalAccessException | InvocationTargetException e2) {
            throw f.g0.c.a("unable to set ssl parameters", e2);
        }
    }

    @Override // f.g0.j.f
    public String h(SSLSocket sSLSocket) {
        try {
            String str = (String) this.f6020d.invoke(sSLSocket, new Object[0]);
            if (str != null) {
                if (str.equals("")) {
                    return null;
                }
                return str;
            }
            return null;
        } catch (IllegalAccessException | InvocationTargetException e2) {
            throw f.g0.c.a("unable to get selected protocols", e2);
        }
    }
}