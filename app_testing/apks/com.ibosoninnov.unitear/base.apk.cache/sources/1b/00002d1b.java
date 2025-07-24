package f.g0.i;

import java.io.IOException;

/* compiled from: Http2.java */
/* loaded from: classes2.dex */
public final class e {

    /* renamed from: a  reason: collision with root package name */
    public static final g.h f5899a = g.h.e("PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n");

    /* renamed from: b  reason: collision with root package name */
    public static final String[] f5900b = {"DATA", "HEADERS", "PRIORITY", "RST_STREAM", "SETTINGS", "PUSH_PROMISE", "PING", "GOAWAY", "WINDOW_UPDATE", "CONTINUATION"};

    /* renamed from: c  reason: collision with root package name */
    public static final String[] f5901c = new String[64];

    /* renamed from: d  reason: collision with root package name */
    public static final String[] f5902d = new String[256];

    static {
        int i = 0;
        int i2 = 0;
        while (true) {
            String[] strArr = f5902d;
            if (i2 >= strArr.length) {
                break;
            }
            strArr[i2] = f.g0.c.n("%8s", Integer.toBinaryString(i2)).replace(' ', '0');
            i2++;
        }
        String[] strArr2 = f5901c;
        strArr2[0] = "";
        strArr2[1] = "END_STREAM";
        int[] iArr = {1};
        strArr2[8] = "PADDED";
        for (int i3 = 0; i3 < 1; i3++) {
            int i4 = iArr[i3];
            String[] strArr3 = f5901c;
            strArr3[i4 | 8] = c.b.a.a.a.v(new StringBuilder(), strArr3[i4], "|PADDED");
        }
        String[] strArr4 = f5901c;
        strArr4[4] = "END_HEADERS";
        strArr4[32] = "PRIORITY";
        strArr4[36] = "END_HEADERS|PRIORITY";
        int[] iArr2 = {4, 32, 36};
        for (int i5 = 0; i5 < 3; i5++) {
            int i6 = iArr2[i5];
            for (int i7 = 0; i7 < 1; i7++) {
                int i8 = iArr[i7];
                String[] strArr5 = f5901c;
                int i9 = i8 | i6;
                strArr5[i9] = strArr5[i8] + '|' + strArr5[i6];
                StringBuilder sb = new StringBuilder();
                sb.append(strArr5[i8]);
                sb.append('|');
                strArr5[i9 | 8] = c.b.a.a.a.v(sb, strArr5[i6], "|PADDED");
            }
        }
        while (true) {
            String[] strArr6 = f5901c;
            if (i >= strArr6.length) {
                return;
            }
            if (strArr6[i] == null) {
                strArr6[i] = f5902d[i];
            }
            i++;
        }
    }

    public static String a(boolean z, int i, int i2, byte b2, byte b3) {
        String str;
        String[] strArr = f5900b;
        String n = b2 < strArr.length ? strArr[b2] : f.g0.c.n("0x%02x", Byte.valueOf(b2));
        if (b3 == 0) {
            str = "";
        } else {
            if (b2 != 2 && b2 != 3) {
                if (b2 == 4 || b2 == 6) {
                    str = b3 == 1 ? "ACK" : f5902d[b3];
                } else if (b2 != 7 && b2 != 8) {
                    String[] strArr2 = f5901c;
                    String str2 = b3 < strArr2.length ? strArr2[b3] : f5902d[b3];
                    if (b2 == 5 && (b3 & 4) != 0) {
                        str = str2.replace("HEADERS", "PUSH_PROMISE");
                    } else {
                        str = (b2 != 0 || (b3 & 32) == 0) ? str2 : str2.replace("PRIORITY", "COMPRESSED");
                    }
                }
            }
            str = f5902d[b3];
        }
        Object[] objArr = new Object[5];
        objArr[0] = z ? "<<" : ">>";
        objArr[1] = Integer.valueOf(i);
        objArr[2] = Integer.valueOf(i2);
        objArr[3] = n;
        objArr[4] = str;
        return f.g0.c.n("%s 0x%08x %5d %-13s %s", objArr);
    }

    public static IllegalArgumentException b(String str, Object... objArr) {
        throw new IllegalArgumentException(f.g0.c.n(str, objArr));
    }

    public static IOException c(String str, Object... objArr) {
        throw new IOException(f.g0.c.n(str, objArr));
    }
}