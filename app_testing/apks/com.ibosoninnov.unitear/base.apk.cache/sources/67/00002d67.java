package f;

import java.io.IOException;

/* compiled from: Protocol.java */
/* loaded from: classes2.dex */
public enum w {
    HTTP_1_0("http/1.0"),
    HTTP_1_1("http/1.1"),
    SPDY_3("spdy/3.1"),
    HTTP_2("h2"),
    QUIC("quic");
    

    /* renamed from: h  reason: collision with root package name */
    public final String f6141h;

    w(String str) {
        this.f6141h = str;
    }

    public static w a(String str) {
        w wVar = HTTP_1_0;
        if (str.equals("http/1.0")) {
            return wVar;
        }
        w wVar2 = HTTP_1_1;
        if (str.equals("http/1.1")) {
            return wVar2;
        }
        w wVar3 = HTTP_2;
        if (str.equals("h2")) {
            return wVar3;
        }
        w wVar4 = SPDY_3;
        if (str.equals("spdy/3.1")) {
            return wVar4;
        }
        w wVar5 = QUIC;
        if (str.equals("quic")) {
            return wVar5;
        }
        throw new IOException(c.b.a.a.a.q("Unexpected protocol: ", str));
    }

    @Override // java.lang.Enum
    public String toString() {
        return this.f6141h;
    }
}