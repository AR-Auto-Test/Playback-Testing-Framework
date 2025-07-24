package f.g0.g;

import androidx.recyclerview.widget.RecyclerView;
import com.google.common.net.HttpHeaders;
import f.b0;
import f.j;
import f.k;
import f.q;
import f.r;
import java.io.IOException;
import java.io.InterruptedIOException;
import java.net.IDN;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.regex.Pattern;

/* compiled from: HttpHeaders.java */
/* loaded from: classes2.dex */
public final class e {
    static {
        Pattern.compile(" +([^ \"=]*)=(:?\"([^\"]*)\"|([^ \"=]*)) *(:?,|$)");
    }

    public static long a(b0 b0Var) {
        String a2 = b0Var.f5729g.a(HttpHeaders.CONTENT_LENGTH);
        if (a2 != null) {
            try {
                return Long.parseLong(a2);
            } catch (NumberFormatException unused) {
                return -1L;
            }
        }
        return -1L;
    }

    public static boolean b(b0 b0Var) {
        if (b0Var.f5724b.f6151b.equals("HEAD")) {
            return false;
        }
        int i = b0Var.f5726d;
        if (((i >= 100 && i < 200) || i == 204 || i == 304) && a(b0Var) == -1) {
            String a2 = b0Var.f5729g.a(HttpHeaders.TRANSFER_ENCODING);
            if (a2 == null) {
                a2 = null;
            }
            if (!"chunked".equalsIgnoreCase(a2)) {
                return false;
            }
        }
        return true;
    }

    public static int c(String str, int i) {
        try {
            long parseLong = Long.parseLong(str);
            if (parseLong > 2147483647L) {
                return Integer.MAX_VALUE;
            }
            if (parseLong < 0) {
                return 0;
            }
            return (int) parseLong;
        } catch (NumberFormatException unused) {
            return i;
        }
    }

    /* JADX DEBUG: Multi-variable search result rejected for r0v53, resolved type: java.util.concurrent.atomic.AtomicBoolean */
    /* JADX WARN: Code restructure failed: missing block: B:110:0x01c3, code lost:
        if (((r0.equals(r10) || (r0.endsWith(r10) && r0.charAt((r0.length() - r10.length()) - 1) == '.' && !f.g0.c.q.matcher(r0).matches())) ? true : r4) == false) goto L23;
     */
    /* JADX WARN: Code restructure failed: missing block: B:120:0x01f4, code lost:
        if (r13 != false) goto L195;
     */
    /* JADX WARN: Code restructure failed: missing block: B:127:0x0203, code lost:
        if (r13 == false) goto L118;
     */
    /* JADX WARN: Code restructure failed: missing block: B:128:0x0205, code lost:
        java.lang.Thread.currentThread().interrupt();
     */
    /* JADX WARN: Code restructure failed: missing block: B:45:0x00e0, code lost:
        if (r20 <= 0) goto L85;
     */
    /* JADX WARN: Multi-variable type inference failed */
    /* JADX WARN: Removed duplicated region for block: B:198:0x0303  */
    /* JADX WARN: Type inference failed for: r4v0 */
    /* JADX WARN: Type inference failed for: r4v1, types: [int, boolean] */
    /* JADX WARN: Type inference failed for: r4v29 */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public static void d(k kVar, r rVar, q qVar) {
        List emptyList;
        List emptyList2;
        r rVar2;
        long j;
        boolean z;
        String str;
        String str2;
        j jVar;
        String str3;
        String str4;
        String str5;
        String[] strArr;
        String[] strArr2;
        int i;
        int length;
        int length2;
        String sb;
        if (kVar == k.f6072a) {
            return;
        }
        Pattern pattern = j.f6064a;
        int d2 = qVar.d();
        ?? r4 = 0;
        ArrayList arrayList = null;
        for (int i2 = 0; i2 < d2; i2++) {
            if (HttpHeaders.SET_COOKIE.equalsIgnoreCase(qVar.b(i2))) {
                if (arrayList == null) {
                    arrayList = new ArrayList(2);
                }
                arrayList.add(qVar.e(i2));
            }
        }
        if (arrayList != null) {
            emptyList = Collections.unmodifiableList(arrayList);
        } else {
            emptyList = Collections.emptyList();
        }
        List list = emptyList;
        int size = list.size();
        int i3 = 0;
        ArrayList arrayList2 = null;
        while (i3 < size) {
            String str6 = (String) list.get(i3);
            long currentTimeMillis = System.currentTimeMillis();
            int length3 = str6.length();
            char c2 = ';';
            int j2 = f.g0.c.j(str6, r4, length3, ';');
            int j3 = f.g0.c.j(str6, r4, j2, '=');
            if (j3 != j2) {
                String y = f.g0.c.y(str6, r4, j3);
                if (!y.isEmpty() && f.g0.c.r(y) == -1) {
                    String y2 = f.g0.c.y(str6, j3 + 1, j2);
                    if (f.g0.c.r(y2) == -1) {
                        int i4 = j2 + 1;
                        boolean z2 = r4;
                        boolean z3 = z2;
                        boolean z4 = z3;
                        boolean z5 = true;
                        long j4 = -1;
                        long j5 = 253402300799999L;
                        String str7 = null;
                        String str8 = null;
                        while (true) {
                            long j6 = RecyclerView.FOREVER_NS;
                            if (i4 < length3) {
                                int j7 = f.g0.c.j(str6, i4, length3, c2);
                                int j8 = f.g0.c.j(str6, i4, j7, '=');
                                String y3 = f.g0.c.y(str6, i4, j8);
                                String y4 = j8 < j7 ? f.g0.c.y(str6, j8 + 1, j7) : "";
                                if (y3.equalsIgnoreCase("expires")) {
                                    try {
                                        j5 = j.b(y4, r4, y4.length());
                                        z4 = true;
                                    } catch (NumberFormatException | IllegalArgumentException unused) {
                                    }
                                    i4 = j7 + 1;
                                    c2 = ';';
                                } else if (y3.equalsIgnoreCase("max-age")) {
                                    try {
                                        j4 = Long.parseLong(y4);
                                    } catch (NumberFormatException e2) {
                                        if (y4.matches("-?\\d+")) {
                                            if (!y4.startsWith("-")) {
                                                j4 = Long.MAX_VALUE;
                                            }
                                            j4 = Long.MIN_VALUE;
                                        } else {
                                            throw e2;
                                        }
                                    }
                                } else {
                                    if (y3.equalsIgnoreCase("domain")) {
                                        if (!y4.endsWith(".")) {
                                            if (y4.startsWith(".")) {
                                                y4 = y4.substring(1);
                                            }
                                            String c3 = f.g0.c.c(y4);
                                            if (c3 == null) {
                                                throw new IllegalArgumentException();
                                            }
                                            str7 = c3;
                                            z5 = r4;
                                        } else {
                                            throw new IllegalArgumentException();
                                        }
                                    } else if (y3.equalsIgnoreCase("path")) {
                                        str8 = y4;
                                    } else if (y3.equalsIgnoreCase("secure")) {
                                        z2 = true;
                                    } else if (y3.equalsIgnoreCase("httponly")) {
                                        z3 = true;
                                    }
                                    i4 = j7 + 1;
                                    c2 = ';';
                                }
                            } else {
                                if (j4 == Long.MIN_VALUE) {
                                    rVar2 = rVar;
                                    j = Long.MIN_VALUE;
                                } else if (j4 != -1) {
                                    if (j4 <= 9223372036854775L) {
                                        j6 = j4 * 1000;
                                    }
                                    long j9 = currentTimeMillis + j6;
                                    if (j9 < currentTimeMillis || j9 > 253402300799999L) {
                                        j = 253402300799999L;
                                        rVar2 = rVar;
                                    } else {
                                        rVar2 = rVar;
                                        j = j9;
                                    }
                                } else {
                                    rVar2 = rVar;
                                    j = j5;
                                }
                                String str9 = rVar2.f6090e;
                                String str10 = str7;
                                if (str10 == null) {
                                    str10 = str9;
                                }
                                if (str9.length() != str10.length()) {
                                    f.g0.k.a aVar = f.g0.k.a.f6037d;
                                    Objects.requireNonNull(aVar);
                                    String[] split = IDN.toUnicode(str10).split("\\.");
                                    if (aVar.f6038e.get() || !aVar.f6038e.compareAndSet(r4, true)) {
                                        try {
                                            aVar.f6039f.await();
                                        } catch (InterruptedException unused2) {
                                        }
                                    } else {
                                        boolean z6 = r4;
                                        while (true) {
                                            try {
                                                try {
                                                    aVar.b();
                                                    break;
                                                } catch (InterruptedIOException unused3) {
                                                    z6 = true;
                                                } catch (IOException e3) {
                                                    f.g0.j.f.f6032a.k(5, "Failed to read public suffix list", e3);
                                                }
                                            } catch (Throwable th) {
                                                if (z6) {
                                                    Thread.currentThread().interrupt();
                                                }
                                                throw th;
                                            }
                                        }
                                    }
                                    synchronized (aVar) {
                                        if (aVar.f6040g == null) {
                                            throw new IllegalStateException("Unable to load publicsuffixes.gz resource from the classpath.");
                                        }
                                    }
                                    int length4 = split.length;
                                    byte[][] bArr = new byte[length4];
                                    for (int i5 = r4; i5 < split.length; i5++) {
                                        bArr[i5] = split[i5].getBytes(f.g0.c.i);
                                    }
                                    int i6 = r4;
                                    while (true) {
                                        if (i6 >= length4) {
                                            str3 = null;
                                            break;
                                        }
                                        str3 = f.g0.k.a.a(aVar.f6040g, bArr, i6);
                                        if (str3 != null) {
                                            break;
                                        }
                                        i6++;
                                    }
                                    if (length4 > 1) {
                                        byte[][] bArr2 = (byte[][]) bArr.clone();
                                        for (int i7 = r4; i7 < bArr2.length - 1; i7++) {
                                            bArr2[i7] = f.g0.k.a.f6034a;
                                            str4 = f.g0.k.a.a(aVar.f6040g, bArr2, i7);
                                            if (str4 != null) {
                                                break;
                                            }
                                        }
                                    }
                                    str4 = null;
                                    if (str4 != null) {
                                        for (int i8 = 0; i8 < length4 - 1; i8++) {
                                            str5 = f.g0.k.a.a(aVar.f6041h, bArr, i8);
                                            if (str5 != null) {
                                                break;
                                            }
                                        }
                                    }
                                    str5 = null;
                                    if (str5 != null) {
                                        strArr = ("!" + str5).split("\\.");
                                    } else if (str3 == null && str4 == null) {
                                        strArr = f.g0.k.a.f6036c;
                                    } else {
                                        if (str3 != null) {
                                            strArr = str3.split("\\.");
                                        } else {
                                            strArr = f.g0.k.a.f6035b;
                                        }
                                        if (str4 != null) {
                                            strArr2 = str4.split("\\.");
                                        } else {
                                            strArr2 = f.g0.k.a.f6035b;
                                        }
                                        if (strArr.length <= strArr2.length) {
                                            strArr = strArr2;
                                        }
                                    }
                                    if (split.length == strArr.length) {
                                        i = 0;
                                        if (strArr[0].charAt(0) != '!') {
                                            sb = null;
                                            z = sb == null ? false : false;
                                        }
                                    } else {
                                        i = 0;
                                    }
                                    if (strArr[i].charAt(i) == '!') {
                                        length = split.length;
                                        length2 = strArr.length;
                                    } else {
                                        length = split.length;
                                        length2 = strArr.length + 1;
                                    }
                                    StringBuilder sb2 = new StringBuilder();
                                    String[] split2 = str10.split("\\.");
                                    for (int i9 = length - length2; i9 < split2.length; i9++) {
                                        sb2.append(split2[i9]);
                                        sb2.append('.');
                                    }
                                    sb2.deleteCharAt(sb2.length() - 1);
                                    sb = sb2.toString();
                                    if (sb == null) {
                                    }
                                }
                                String str11 = str8;
                                if (str11 == null || !str11.startsWith("/")) {
                                    String e4 = rVar.e();
                                    int lastIndexOf = e4.lastIndexOf(47);
                                    if (lastIndexOf != 0) {
                                        z = false;
                                        str = e4.substring(0, lastIndexOf);
                                    } else {
                                        z = false;
                                        str = "/";
                                    }
                                    str2 = str;
                                } else {
                                    str2 = str11;
                                    z = false;
                                }
                                jVar = new j(y, y2, j, str10, str2, z2, z3, z5, z4);
                            }
                        }
                    }
                }
            }
            z = r4;
            jVar = null;
            if (jVar != null) {
                if (arrayList2 == null) {
                    arrayList2 = new ArrayList();
                }
                arrayList2.add(jVar);
            }
            i3++;
            r4 = z;
        }
        if (arrayList2 != null) {
            emptyList2 = Collections.unmodifiableList(arrayList2);
        } else {
            emptyList2 = Collections.emptyList();
        }
        if (emptyList2.isEmpty()) {
            return;
        }
        Objects.requireNonNull((k.a) kVar);
    }

    public static int e(String str, int i, String str2) {
        while (i < str.length() && str2.indexOf(str.charAt(i)) == -1) {
            i++;
        }
        return i;
    }
}