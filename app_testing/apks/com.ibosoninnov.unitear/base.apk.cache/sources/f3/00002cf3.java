package f.g0;

import androidx.recyclerview.widget.RecyclerView;
import com.google.common.primitives.UnsignedBytes;
import f.c0;
import f.d0;
import f.r;
import g.e;
import g.g;
import g.h;
import g.x;
import java.io.Closeable;
import java.io.IOException;
import java.io.InterruptedIOException;
import java.net.IDN;
import java.net.InetAddress;
import java.net.Socket;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.TimeZone;
import java.util.concurrent.TimeUnit;
import java.util.regex.Pattern;

/* compiled from: Util.java */
/* loaded from: classes2.dex */
public final class c {

    /* renamed from: a  reason: collision with root package name */
    public static final byte[] f5773a;

    /* renamed from: b  reason: collision with root package name */
    public static final String[] f5774b = new String[0];

    /* renamed from: c  reason: collision with root package name */
    public static final d0 f5775c;

    /* renamed from: d  reason: collision with root package name */
    public static final h f5776d;

    /* renamed from: e  reason: collision with root package name */
    public static final h f5777e;

    /* renamed from: f  reason: collision with root package name */
    public static final h f5778f;

    /* renamed from: g  reason: collision with root package name */
    public static final h f5779g;

    /* renamed from: h  reason: collision with root package name */
    public static final h f5780h;
    public static final Charset i;
    public static final Charset j;
    public static final Charset k;
    public static final Charset l;
    public static final Charset m;
    public static final Charset n;
    public static final TimeZone o;
    public static final Comparator<String> p;
    public static final Pattern q;

    /* compiled from: Util.java */
    /* loaded from: classes2.dex */
    public class a implements Comparator<String> {
        /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object, java.lang.Object] */
        @Override // java.util.Comparator
        public int compare(String str, String str2) {
            return str.compareTo(str2);
        }
    }

    static {
        byte[] bArr = new byte[0];
        f5773a = bArr;
        e eVar = new e();
        eVar.Q(bArr);
        long j2 = 0;
        f5775c = new c0(null, j2, eVar);
        e(j2, j2, j2);
        f5776d = h.b("efbbbf");
        f5777e = h.b("feff");
        f5778f = h.b("fffe");
        f5779g = h.b("0000ffff");
        f5780h = h.b("ffff0000");
        i = Charset.forName("UTF-8");
        j = Charset.forName("ISO-8859-1");
        k = Charset.forName("UTF-16BE");
        l = Charset.forName("UTF-16LE");
        m = Charset.forName("UTF-32BE");
        n = Charset.forName("UTF-32LE");
        o = TimeZone.getTimeZone("GMT");
        p = new a();
        q = Pattern.compile("([0-9a-fA-F]*:[0-9a-fA-F:.]*)|([\\d.]+)");
    }

    public static AssertionError a(String str, Exception exc) {
        AssertionError assertionError = new AssertionError(str);
        try {
            assertionError.initCause(exc);
        } catch (IllegalStateException unused) {
        }
        return assertionError;
    }

    public static Charset b(g gVar, Charset charset) {
        h hVar = f5776d;
        if (gVar.j(0L, hVar)) {
            gVar.c(hVar.l());
            return i;
        }
        h hVar2 = f5777e;
        if (gVar.j(0L, hVar2)) {
            gVar.c(hVar2.l());
            return k;
        }
        h hVar3 = f5778f;
        if (gVar.j(0L, hVar3)) {
            gVar.c(hVar3.l());
            return l;
        }
        h hVar4 = f5779g;
        if (gVar.j(0L, hVar4)) {
            gVar.c(hVar4.l());
            return m;
        }
        h hVar5 = f5780h;
        if (gVar.j(0L, hVar5)) {
            gVar.c(hVar5.l());
            return n;
        }
        return charset;
    }

    public static String c(String str) {
        InetAddress i2;
        int i3 = -1;
        int i4 = 0;
        if (str.contains(":")) {
            if (str.startsWith("[") && str.endsWith("]")) {
                i2 = i(str, 1, str.length() - 1);
            } else {
                i2 = i(str, 0, str.length());
            }
            if (i2 == null) {
                return null;
            }
            byte[] address = i2.getAddress();
            if (address.length == 16) {
                int i5 = 0;
                int i6 = 0;
                while (i5 < address.length) {
                    int i7 = i5;
                    while (i7 < 16 && address[i7] == 0 && address[i7 + 1] == 0) {
                        i7 += 2;
                    }
                    int i8 = i7 - i5;
                    if (i8 > i6 && i8 >= 4) {
                        i3 = i5;
                        i6 = i8;
                    }
                    i5 = i7 + 2;
                }
                e eVar = new e();
                while (i4 < address.length) {
                    if (i4 == i3) {
                        eVar.T(58);
                        i4 += i6;
                        if (i4 == 16) {
                            eVar.T(58);
                        }
                    } else {
                        if (i4 > 0) {
                            eVar.T(58);
                        }
                        eVar.m(((address[i4] & UnsignedBytes.MAX_VALUE) << 8) | (address[i4 + 1] & UnsignedBytes.MAX_VALUE));
                        i4 += 2;
                    }
                }
                return eVar.K();
            }
            throw new AssertionError(c.b.a.a.a.r("Invalid IPv6 address: '", str, "'"));
        }
        try {
            String lowerCase = IDN.toASCII(str).toLowerCase(Locale.US);
            if (lowerCase.isEmpty()) {
                return null;
            }
            for (int i9 = 0; i9 < lowerCase.length(); i9++) {
                char charAt = lowerCase.charAt(i9);
                if (charAt > 31 && charAt < 127 && " #%/:?@[\\]".indexOf(charAt) == -1) {
                }
                i4 = 1;
            }
            if (i4 != 0) {
                return null;
            }
            return lowerCase;
        } catch (IllegalArgumentException unused) {
            return null;
        }
    }

    public static int d(String str, long j2, TimeUnit timeUnit) {
        int i2 = (j2 > 0L ? 1 : (j2 == 0L ? 0 : -1));
        if (i2 >= 0) {
            Objects.requireNonNull(timeUnit, "unit == null");
            long millis = timeUnit.toMillis(j2);
            if (millis <= 2147483647L) {
                if (millis != 0 || i2 <= 0) {
                    return (int) millis;
                }
                throw new IllegalArgumentException(c.b.a.a.a.q(str, " too small."));
            }
            throw new IllegalArgumentException(c.b.a.a.a.q(str, " too large."));
        }
        throw new IllegalArgumentException(c.b.a.a.a.q(str, " < 0"));
    }

    public static void e(long j2, long j3, long j4) {
        if ((j3 | j4) < 0 || j3 > j2 || j2 - j3 < j4) {
            throw new ArrayIndexOutOfBoundsException();
        }
    }

    public static void f(Closeable closeable) {
        if (closeable != null) {
            try {
                closeable.close();
            } catch (RuntimeException e2) {
                throw e2;
            } catch (Exception unused) {
            }
        }
    }

    public static void g(Socket socket) {
        if (socket != null) {
            try {
                socket.close();
            } catch (AssertionError e2) {
                if (!t(e2)) {
                    throw e2;
                }
            } catch (RuntimeException e3) {
                throw e3;
            } catch (Exception unused) {
            }
        }
    }

    public static int h(char c2) {
        if (c2 < '0' || c2 > '9') {
            char c3 = 'a';
            if (c2 < 'a' || c2 > 'f') {
                c3 = 'A';
                if (c2 < 'A' || c2 > 'F') {
                    return -1;
                }
            }
            return (c2 - c3) + 10;
        }
        return c2 - '0';
    }

    /* JADX WARN: Code restructure failed: missing block: B:30:0x005d, code lost:
        r14 = r5;
     */
    /* JADX WARN: Code restructure failed: missing block: B:72:0x00db, code lost:
        if (r7 == r0) goto L35;
     */
    /* JADX WARN: Code restructure failed: missing block: B:74:0x00de, code lost:
        if (r8 != (-1)) goto L34;
     */
    /* JADX WARN: Code restructure failed: missing block: B:75:0x00e0, code lost:
        return null;
     */
    /* JADX WARN: Code restructure failed: missing block: B:76:0x00e1, code lost:
        r1 = r7 - r8;
        java.lang.System.arraycopy(r3, r8, r3, 16 - r1, r1);
        java.util.Arrays.fill(r3, r8, (16 - r7) + r8, (byte) 0);
     */
    /* JADX WARN: Code restructure failed: missing block: B:78:0x00f3, code lost:
        return java.net.InetAddress.getByAddress(r3);
     */
    /* JADX WARN: Code restructure failed: missing block: B:80:0x00f9, code lost:
        throw new java.lang.AssertionError();
     */
    /* JADX WARN: Removed duplicated region for block: B:61:0x00ab  */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public static InetAddress i(String str, int i2, int i3) {
        int i4;
        int i5;
        int i6;
        int i7 = 16;
        byte[] bArr = new byte[16];
        int i8 = -1;
        int i9 = 0;
        int i10 = i2;
        int i11 = -1;
        int i12 = -1;
        int i13 = 0;
        while (true) {
            if (i10 >= i3) {
                i4 = i7;
                break;
            } else if (i13 != i7) {
                int i14 = i10 + 2;
                if (i14 <= i3 && str.regionMatches(i10, "::", i9, 2)) {
                    if (i11 == i8) {
                        i13 += 2;
                        if (i14 != i3) {
                            i11 = i13;
                            i12 = i14;
                            i10 = i12;
                            int i15 = 0;
                            while (i10 < i3) {
                            }
                            i6 = i10 - i12;
                            if (i6 == 0) {
                                break;
                            }
                            break;
                        }
                        i4 = i7;
                        i11 = i13;
                        break;
                    }
                    return null;
                }
                if (i13 != 0) {
                    if (str.regionMatches(i10, ":", i9, 1)) {
                        i10++;
                    } else if (!str.regionMatches(i10, ".", i9, 1)) {
                        return null;
                    } else {
                        int i16 = i13 - 2;
                        int i17 = i16;
                        loop2: while (true) {
                            if (i12 < i3) {
                                if (i17 == i7) {
                                    break;
                                }
                                if (i17 != i16) {
                                    if (str.charAt(i12) != '.') {
                                        break;
                                    }
                                    i12++;
                                }
                                int i18 = i9;
                                int i19 = i12;
                                while (i19 < i3) {
                                    char charAt = str.charAt(i19);
                                    if (charAt < '0' || charAt > '9') {
                                        break;
                                    } else if ((i18 == 0 && i12 != i19) || (i18 = ((i18 * 10) + charAt) - 48) > 255) {
                                        break loop2;
                                    } else {
                                        i19++;
                                    }
                                }
                                if (i19 - i12 == 0) {
                                    break;
                                }
                                bArr[i17] = (byte) i18;
                                i17++;
                                i12 = i19;
                                i7 = 16;
                                i9 = 0;
                            } else if (i17 == i16 + 4) {
                                i5 = 1;
                            }
                        }
                        i5 = 0;
                        if (i5 == 0) {
                            return null;
                        }
                        i13 += 2;
                        i4 = 16;
                    }
                }
                i12 = i10;
                i10 = i12;
                int i152 = 0;
                while (i10 < i3) {
                    int h2 = h(str.charAt(i10));
                    if (h2 == -1) {
                        break;
                    }
                    i152 = (i152 << 4) + h2;
                    i10++;
                }
                i6 = i10 - i12;
                if (i6 == 0 || i6 > 4) {
                    break;
                }
                int i20 = i13 + 1;
                bArr[i13] = (byte) ((i152 >>> 8) & 255);
                i13 = i20 + 1;
                bArr[i20] = (byte) (i152 & 255);
                i7 = 16;
                i8 = -1;
                i9 = 0;
            } else {
                return null;
            }
        }
        return null;
    }

    public static int j(String str, int i2, int i3, char c2) {
        while (i2 < i3) {
            if (str.charAt(i2) == c2) {
                return i2;
            }
            i2++;
        }
        return i3;
    }

    public static int k(String str, int i2, int i3, String str2) {
        while (i2 < i3) {
            if (str2.indexOf(str.charAt(i2)) != -1) {
                return i2;
            }
            i2++;
        }
        return i3;
    }

    public static boolean l(x xVar, int i2, TimeUnit timeUnit) {
        try {
            return v(xVar, i2, timeUnit);
        } catch (IOException unused) {
            return false;
        }
    }

    public static boolean m(Object obj, Object obj2) {
        return obj == obj2 || (obj != null && obj.equals(obj2));
    }

    public static String n(String str, Object... objArr) {
        return String.format(Locale.US, str, objArr);
    }

    public static String o(r rVar, boolean z) {
        String str;
        if (rVar.f6090e.contains(":")) {
            str = c.b.a.a.a.v(c.b.a.a.a.x("["), rVar.f6090e, "]");
        } else {
            str = rVar.f6090e;
        }
        if (z || rVar.f6091f != r.c(rVar.f6087b)) {
            StringBuilder A = c.b.a.a.a.A(str, ":");
            A.append(rVar.f6091f);
            return A.toString();
        }
        return str;
    }

    public static <T> List<T> p(List<T> list) {
        return Collections.unmodifiableList(new ArrayList(list));
    }

    public static <T> List<T> q(T... tArr) {
        return Collections.unmodifiableList(Arrays.asList((Object[]) tArr.clone()));
    }

    public static int r(String str) {
        int length = str.length();
        for (int i2 = 0; i2 < length; i2++) {
            char charAt = str.charAt(i2);
            if (charAt <= 31 || charAt >= 127) {
                return i2;
            }
        }
        return -1;
    }

    public static String[] s(Comparator<? super String> comparator, String[] strArr, String[] strArr2) {
        ArrayList arrayList = new ArrayList();
        for (String str : strArr) {
            int length = strArr2.length;
            int i2 = 0;
            while (true) {
                if (i2 >= length) {
                    break;
                } else if (comparator.compare(str, strArr2[i2]) == 0) {
                    arrayList.add(str);
                    break;
                } else {
                    i2++;
                }
            }
        }
        return (String[]) arrayList.toArray(new String[arrayList.size()]);
    }

    public static boolean t(AssertionError assertionError) {
        return (assertionError.getCause() == null || assertionError.getMessage() == null || !assertionError.getMessage().contains("getsockname failed")) ? false : true;
    }

    public static boolean u(Comparator<String> comparator, String[] strArr, String[] strArr2) {
        if (strArr != null && strArr2 != null && strArr.length != 0 && strArr2.length != 0) {
            for (String str : strArr) {
                for (String str2 : strArr2) {
                    if (comparator.compare(str, str2) == 0) {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    /* JADX DEBUG: Another duplicated slice has different insns count: {[CMP_L]}, finally: {[CMP_L, INVOKE, INVOKE, INVOKE, ARITH, INVOKE, IF] complete} */
    public static boolean v(x xVar, int i2, TimeUnit timeUnit) {
        long nanoTime = System.nanoTime();
        long c2 = xVar.b().e() ? xVar.b().c() - nanoTime : Long.MAX_VALUE;
        xVar.b().d(Math.min(c2, timeUnit.toNanos(i2)) + nanoTime);
        try {
            e eVar = new e();
            while (xVar.u(eVar, 8192L) != -1) {
                eVar.B();
            }
            if (c2 == RecyclerView.FOREVER_NS) {
                xVar.b().a();
            } else {
                xVar.b().d(nanoTime + c2);
            }
            return true;
        } catch (InterruptedIOException unused) {
            if (c2 == RecyclerView.FOREVER_NS) {
                xVar.b().a();
            } else {
                xVar.b().d(nanoTime + c2);
            }
            return false;
        } catch (Throwable th) {
            if (c2 == RecyclerView.FOREVER_NS) {
                xVar.b().a();
            } else {
                xVar.b().d(nanoTime + c2);
            }
            throw th;
        }
    }

    public static int w(String str, int i2, int i3) {
        while (i2 < i3) {
            char charAt = str.charAt(i2);
            if (charAt != '\t' && charAt != '\n' && charAt != '\f' && charAt != '\r' && charAt != ' ') {
                return i2;
            }
            i2++;
        }
        return i3;
    }

    public static int x(String str, int i2, int i3) {
        for (int i4 = i3 - 1; i4 >= i2; i4--) {
            char charAt = str.charAt(i4);
            if (charAt != '\t' && charAt != '\n' && charAt != '\f' && charAt != '\r' && charAt != ' ') {
                return i4 + 1;
            }
        }
        return i2;
    }

    public static String y(String str, int i2, int i3) {
        int w = w(str, i2, i3);
        return str.substring(w, x(str, w, i3));
    }
}