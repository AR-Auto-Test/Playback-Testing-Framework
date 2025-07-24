package f.g0.g;

import f.w;
import java.net.ProtocolException;

/* compiled from: StatusLine.java */
/* loaded from: classes2.dex */
public final class i {

    /* renamed from: a  reason: collision with root package name */
    public final w f5841a;

    /* renamed from: b  reason: collision with root package name */
    public final int f5842b;

    /* renamed from: c  reason: collision with root package name */
    public final String f5843c;

    public i(w wVar, int i, String str) {
        this.f5841a = wVar;
        this.f5842b = i;
        this.f5843c = str;
    }

    public static i a(String str) {
        String str2;
        w wVar = w.HTTP_1_0;
        int i = 9;
        if (str.startsWith("HTTP/1.")) {
            if (str.length() >= 9 && str.charAt(8) == ' ') {
                int charAt = str.charAt(7) - '0';
                if (charAt != 0) {
                    if (charAt == 1) {
                        wVar = w.HTTP_1_1;
                    } else {
                        throw new ProtocolException(c.b.a.a.a.q("Unexpected status line: ", str));
                    }
                }
            } else {
                throw new ProtocolException(c.b.a.a.a.q("Unexpected status line: ", str));
            }
        } else if (!str.startsWith("ICY ")) {
            throw new ProtocolException(c.b.a.a.a.q("Unexpected status line: ", str));
        } else {
            i = 4;
        }
        int i2 = i + 3;
        if (str.length() >= i2) {
            try {
                int parseInt = Integer.parseInt(str.substring(i, i2));
                if (str.length() <= i2) {
                    str2 = "";
                } else if (str.charAt(i2) == ' ') {
                    str2 = str.substring(i + 4);
                } else {
                    throw new ProtocolException(c.b.a.a.a.q("Unexpected status line: ", str));
                }
                return new i(wVar, parseInt, str2);
            } catch (NumberFormatException unused) {
                throw new ProtocolException(c.b.a.a.a.q("Unexpected status line: ", str));
            }
        }
        throw new ProtocolException(c.b.a.a.a.q("Unexpected status line: ", str));
    }

    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append(this.f5841a == w.HTTP_1_0 ? "HTTP/1.0" : "HTTP/1.1");
        sb.append(' ');
        sb.append(this.f5842b);
        if (this.f5843c != null) {
            sb.append(' ');
            sb.append(this.f5843c);
        }
        return sb.toString();
    }
}