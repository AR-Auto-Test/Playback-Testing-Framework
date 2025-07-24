package f;

import com.google.android.material.badge.BadgeDrawable;
import com.google.common.primitives.UnsignedBytes;
import java.net.URI;
import java.net.URISyntaxException;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/* compiled from: HttpUrl.java */
/* loaded from: classes2.dex */
public final class r {

    /* renamed from: a  reason: collision with root package name */
    public static final char[] f6086a = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F'};

    /* renamed from: b  reason: collision with root package name */
    public final String f6087b;

    /* renamed from: c  reason: collision with root package name */
    public final String f6088c;

    /* renamed from: d  reason: collision with root package name */
    public final String f6089d;

    /* renamed from: e  reason: collision with root package name */
    public final String f6090e;

    /* renamed from: f  reason: collision with root package name */
    public final int f6091f;

    /* renamed from: g  reason: collision with root package name */
    public final List<String> f6092g;

    /* renamed from: h  reason: collision with root package name */
    public final List<String> f6093h;
    public final String i;
    public final String j;

    /* compiled from: HttpUrl.java */
    /* loaded from: classes2.dex */
    public static final class a {

        /* renamed from: a  reason: collision with root package name */
        public String f6094a;

        /* renamed from: d  reason: collision with root package name */
        public String f6097d;

        /* renamed from: f  reason: collision with root package name */
        public final List<String> f6099f;

        /* renamed from: g  reason: collision with root package name */
        public List<String> f6100g;

        /* renamed from: h  reason: collision with root package name */
        public String f6101h;

        /* renamed from: b  reason: collision with root package name */
        public String f6095b = "";

        /* renamed from: c  reason: collision with root package name */
        public String f6096c = "";

        /* renamed from: e  reason: collision with root package name */
        public int f6098e = -1;

        public a() {
            ArrayList arrayList = new ArrayList();
            this.f6099f = arrayList;
            arrayList.add("");
        }

        public r a() {
            if (this.f6094a != null) {
                if (this.f6097d != null) {
                    return new r(this);
                }
                throw new IllegalStateException("host == null");
            }
            throw new IllegalStateException("scheme == null");
        }

        public a b(String str) {
            this.f6100g = str != null ? r.n(r.b(str, " \"'<>#", true, false, true, true)) : null;
            return this;
        }

        /* JADX WARN: Code restructure failed: missing block: B:109:0x01f0, code lost:
            if (r1 <= 65535) goto L170;
         */
        /* JADX WARN: Code restructure failed: missing block: B:30:0x005b, code lost:
            if (r7 == ':') goto L30;
         */
        /* JADX WARN: Removed duplicated region for block: B:136:0x025c  */
        /* JADX WARN: Unsupported multi-entry loop pattern (BACK_EDGE: B:134:0x0259 -> B:135:0x025a). Please submit an issue!!! */
        /*
            Code decompiled incorrectly, please refer to instructions dump.
        */
        public int c(r rVar, String str) {
            int i;
            char c2;
            char c3;
            int k;
            boolean z;
            int i2;
            int i3;
            int i4;
            a aVar;
            a aVar2;
            int i5;
            String str2;
            String str3;
            int i6;
            String str4;
            a aVar3;
            String str5;
            List<String> list;
            List<String> list2;
            List<String> list3;
            List<String> list4;
            char c4;
            char charAt;
            int w = f.g0.c.w(str, 0, str.length());
            int x = f.g0.c.x(str, w, str.length());
            char c5 = 65535;
            boolean z2 = true;
            if (x - w >= 2) {
                char charAt2 = str.charAt(w);
                char c6 = 'z';
                if ((charAt2 >= 'a' && charAt2 <= 'z') || (charAt2 >= 'A' && charAt2 <= 'Z')) {
                    i = w;
                    while (true) {
                        i++;
                        if (i >= x) {
                            break;
                        }
                        char charAt3 = str.charAt(i);
                        if ((charAt3 >= 'a' && charAt3 <= c6) || ((charAt3 >= 'A' && charAt3 <= 'Z') || ((charAt3 >= '0' && charAt3 <= '9') || charAt3 == '+' || charAt3 == '-' || charAt3 == '.'))) {
                            c6 = 'z';
                        }
                    }
                }
            }
            i = -1;
            if (i != -1) {
                if (str.regionMatches(true, w, "https:", 0, 6)) {
                    this.f6094a = "https";
                    w += 6;
                } else if (!str.regionMatches(true, w, "http:", 0, 5)) {
                    return 3;
                } else {
                    this.f6094a = "http";
                    w += 5;
                }
            } else if (rVar == null) {
                return 2;
            } else {
                this.f6094a = rVar.f6087b;
            }
            int i7 = w;
            int i8 = 0;
            while (true) {
                c2 = '\\';
                c3 = '/';
                if (i7 >= x || !((charAt = str.charAt(i7)) == '\\' || charAt == '/')) {
                    break;
                }
                i8++;
                i7++;
            }
            char c7 = '?';
            char c8 = '#';
            if (i8 < 2 && rVar != null && rVar.f6087b.equals(this.f6094a)) {
                this.f6095b = rVar.h();
                this.f6096c = rVar.d();
                this.f6097d = rVar.f6090e;
                this.f6098e = rVar.f6091f;
                this.f6099f.clear();
                this.f6099f.addAll(rVar.f());
                if (w == x || str.charAt(w) == '#') {
                    b(rVar.g());
                }
                z = false;
            } else {
                int i9 = w + i8;
                boolean z3 = false;
                boolean z4 = false;
                while (true) {
                    k = f.g0.c.k(str, i9, x, "@/\\?#");
                    char charAt4 = k != x ? str.charAt(k) : c5;
                    if (charAt4 == c5 || charAt4 == c8 || charAt4 == c3 || charAt4 == c2 || charAt4 == c7) {
                        break;
                    }
                    if (charAt4 == '@') {
                        if (!z3) {
                            int j = f.g0.c.j(str, i9, k, ':');
                            i4 = k;
                            String a2 = r.a(str, i9, j, " \"':;<=>@[]^`{}|/\\?#", true, false, false, true, null);
                            if (z4) {
                                a2 = this.f6095b + "%40" + a2;
                            }
                            this.f6095b = a2;
                            if (j != i4) {
                                this.f6096c = r.a(str, j + 1, i4, " \"':;<=>@[]^`{}|/\\?#", true, false, false, true, null);
                                z3 = z2;
                            }
                            z4 = z2;
                        } else {
                            i4 = k;
                            this.f6096c += "%40" + r.a(str, i9, i4, " \"':;<=>@[]^`{}|/\\?#", true, false, false, true, null);
                        }
                        i9 = i4 + 1;
                    }
                    c8 = '#';
                    c7 = '?';
                    c3 = '/';
                    c2 = '\\';
                    c5 = 65535;
                    z2 = true;
                }
                int i10 = i9;
                while (true) {
                    if (i10 < k) {
                        char charAt5 = str.charAt(i10);
                        if (charAt5 == ':') {
                            break;
                        }
                        if (charAt5 != '[') {
                            i3 = 1;
                        } else {
                            i3 = 1;
                            do {
                                i10++;
                                if (i10 < k) {
                                }
                            } while (str.charAt(i10) != ']');
                        }
                        i10 += i3;
                    } else {
                        i10 = k;
                        break;
                    }
                }
                int i11 = i10 + 1;
                if (i11 < k) {
                    this.f6097d = f.g0.c.c(r.j(str, i9, i10, false));
                    try {
                        i2 = Integer.parseInt(r.a(str, i11, k, "", false, false, false, true, null));
                        if (i2 > 0) {
                        }
                    } catch (NumberFormatException unused) {
                    }
                    i2 = -1;
                    this.f6098e = i2;
                    if (i2 == -1) {
                        return 4;
                    }
                    z = false;
                } else {
                    z = false;
                    this.f6097d = f.g0.c.c(r.j(str, i9, i10, false));
                    this.f6098e = r.c(this.f6094a);
                }
                if (this.f6097d == null) {
                    return 5;
                }
                w = k;
            }
            int k2 = f.g0.c.k(str, w, x, "?#");
            if (w == k2) {
                aVar3 = this;
                str5 = str;
                str3 = str5;
            } else {
                char charAt6 = str.charAt(w);
                if (charAt6 != '/' && charAt6 != '\\') {
                    List<String> list5 = this.f6099f;
                    list5.set(list5.size() - 1, "");
                    aVar = this;
                    aVar2 = aVar;
                    i5 = k2;
                    str2 = "";
                    str3 = str;
                    i6 = i5;
                    str4 = str3;
                    while (w < i5) {
                    }
                    k2 = i6;
                    aVar3 = aVar;
                    str5 = str;
                } else {
                    int i12 = 1;
                    this.f6099f.clear();
                    this.f6099f.add("");
                    aVar = this;
                    aVar2 = aVar;
                    i5 = k2;
                    str2 = "";
                    str3 = str;
                    i6 = i5;
                    str4 = str3;
                    w += i12;
                    while (w < i5) {
                        int k3 = f.g0.c.k(str4, w, i5, "/\\");
                        boolean z5 = k3 < i5 ? true : z;
                        String a3 = r.a(str4, w, k3, " \"<>^`{}|/\\?#", true, false, false, true, null);
                        if (!((a3.equals(".") || a3.equalsIgnoreCase("%2e")) ? true : z)) {
                            if ((a3.equals("..") || a3.equalsIgnoreCase("%2e.") || a3.equalsIgnoreCase(".%2e") || a3.equalsIgnoreCase("%2e%2e")) ? true : z) {
                                if (aVar2.f6099f.remove(list3.size() - 1).isEmpty() && !aVar2.f6099f.isEmpty()) {
                                    aVar2.f6099f.set(list4.size() - 1, str2);
                                } else {
                                    aVar2.f6099f.add(str2);
                                }
                            } else {
                                if (aVar2.f6099f.get(list.size() - 1).isEmpty()) {
                                    aVar2.f6099f.set(list2.size() - 1, a3);
                                } else {
                                    aVar2.f6099f.add(a3);
                                }
                                if (z5) {
                                    aVar2.f6099f.add(str2);
                                }
                            }
                        }
                        w = k3;
                        z = false;
                        if (z5) {
                            i12 = 1;
                            w += i12;
                            while (w < i5) {
                            }
                        }
                    }
                    k2 = i6;
                    aVar3 = aVar;
                    str5 = str;
                }
            }
            if (k2 >= x || str3.charAt(k2) != '?') {
                c4 = '#';
            } else {
                c4 = '#';
                int j2 = f.g0.c.j(str3, k2, x, '#');
                aVar3.f6100g = r.n(r.a(str5, k2 + 1, j2, " \"'<>#", true, false, true, true, null));
                k2 = j2;
            }
            if (k2 >= x || str3.charAt(k2) != c4) {
                return 1;
            }
            aVar3.f6101h = r.a(str5, k2 + 1, x, "", true, false, false, false, null);
            return 1;
        }

        public String toString() {
            StringBuilder sb = new StringBuilder();
            sb.append(this.f6094a);
            sb.append("://");
            if (!this.f6095b.isEmpty() || !this.f6096c.isEmpty()) {
                sb.append(this.f6095b);
                if (!this.f6096c.isEmpty()) {
                    sb.append(':');
                    sb.append(this.f6096c);
                }
                sb.append('@');
            }
            if (this.f6097d.indexOf(58) != -1) {
                sb.append('[');
                sb.append(this.f6097d);
                sb.append(']');
            } else {
                sb.append(this.f6097d);
            }
            int i = this.f6098e;
            if (i == -1) {
                i = r.c(this.f6094a);
            }
            if (i != r.c(this.f6094a)) {
                sb.append(':');
                sb.append(i);
            }
            List<String> list = this.f6099f;
            int size = list.size();
            for (int i2 = 0; i2 < size; i2++) {
                sb.append('/');
                sb.append(list.get(i2));
            }
            if (this.f6100g != null) {
                sb.append('?');
                r.i(sb, this.f6100g);
            }
            if (this.f6101h != null) {
                sb.append('#');
                sb.append(this.f6101h);
            }
            return sb.toString();
        }
    }

    public r(a aVar) {
        this.f6087b = aVar.f6094a;
        this.f6088c = k(aVar.f6095b, false);
        this.f6089d = k(aVar.f6096c, false);
        this.f6090e = aVar.f6097d;
        int i = aVar.f6098e;
        this.f6091f = i == -1 ? c(aVar.f6094a) : i;
        this.f6092g = l(aVar.f6099f, false);
        List<String> list = aVar.f6100g;
        this.f6093h = list != null ? l(list, true) : null;
        String str = aVar.f6101h;
        this.i = str != null ? j(str, 0, str.length(), false) : null;
        this.j = aVar.toString();
    }

    public static String a(String str, int i, int i2, String str2, boolean z, boolean z2, boolean z3, boolean z4, Charset charset) {
        int i3 = i;
        while (i3 < i2) {
            int codePointAt = str.codePointAt(i3);
            if (codePointAt >= 32 && codePointAt != 127 && ((codePointAt < 128 || !z4) && str2.indexOf(codePointAt) == -1 && ((codePointAt != 37 || (z && (!z2 || m(str, i3, i2)))) && (codePointAt != 43 || !z3)))) {
                i3 += Character.charCount(codePointAt);
            } else {
                g.e eVar = new g.e();
                eVar.Z(str, i, i3);
                g.e eVar2 = null;
                while (i3 < i2) {
                    int codePointAt2 = str.codePointAt(i3);
                    if (!z || (codePointAt2 != 9 && codePointAt2 != 10 && codePointAt2 != 12 && codePointAt2 != 13)) {
                        if (codePointAt2 == 43 && z3) {
                            eVar.Y(z ? BadgeDrawable.DEFAULT_EXCEED_MAX_BADGE_NUMBER_SUFFIX : "%2B");
                        } else if (codePointAt2 >= 32 && codePointAt2 != 127 && ((codePointAt2 < 128 || !z4) && str2.indexOf(codePointAt2) == -1 && (codePointAt2 != 37 || (z && (!z2 || m(str, i3, i2)))))) {
                            eVar.a0(codePointAt2);
                        } else {
                            if (eVar2 == null) {
                                eVar2 = new g.e();
                            }
                            eVar2.a0(codePointAt2);
                            while (!eVar2.f()) {
                                int readByte = eVar2.readByte() & UnsignedBytes.MAX_VALUE;
                                eVar.T(37);
                                char[] cArr = f6086a;
                                eVar.T(cArr[(readByte >> 4) & 15]);
                                eVar.T(cArr[readByte & 15]);
                            }
                        }
                    }
                    i3 += Character.charCount(codePointAt2);
                }
                return eVar.K();
            }
        }
        return str.substring(i, i2);
    }

    public static String b(String str, String str2, boolean z, boolean z2, boolean z3, boolean z4) {
        return a(str, 0, str.length(), str2, z, z2, z3, z4, null);
    }

    public static int c(String str) {
        if (str.equals("http")) {
            return 80;
        }
        return str.equals("https") ? 443 : -1;
    }

    public static void i(StringBuilder sb, List<String> list) {
        int size = list.size();
        for (int i = 0; i < size; i += 2) {
            String str = list.get(i);
            String str2 = list.get(i + 1);
            if (i > 0) {
                sb.append('&');
            }
            sb.append(str);
            if (str2 != null) {
                sb.append('=');
                sb.append(str2);
            }
        }
    }

    public static String j(String str, int i, int i2, boolean z) {
        int i3;
        int i4 = i;
        while (i4 < i2) {
            char charAt = str.charAt(i4);
            if (charAt == '%' || (charAt == '+' && z)) {
                g.e eVar = new g.e();
                eVar.Z(str, i, i4);
                while (i4 < i2) {
                    int codePointAt = str.codePointAt(i4);
                    if (codePointAt == 37 && (i3 = i4 + 2) < i2) {
                        int h2 = f.g0.c.h(str.charAt(i4 + 1));
                        int h3 = f.g0.c.h(str.charAt(i3));
                        if (h2 != -1 && h3 != -1) {
                            eVar.T((h2 << 4) + h3);
                            i4 = i3;
                        }
                        eVar.a0(codePointAt);
                    } else {
                        if (codePointAt == 43 && z) {
                            eVar.T(32);
                        }
                        eVar.a0(codePointAt);
                    }
                    i4 += Character.charCount(codePointAt);
                }
                return eVar.K();
            }
            i4++;
        }
        return str.substring(i, i2);
    }

    public static String k(String str, boolean z) {
        return j(str, 0, str.length(), z);
    }

    public static boolean m(String str, int i, int i2) {
        int i3 = i + 2;
        return i3 < i2 && str.charAt(i) == '%' && f.g0.c.h(str.charAt(i + 1)) != -1 && f.g0.c.h(str.charAt(i3)) != -1;
    }

    public static List<String> n(String str) {
        ArrayList arrayList = new ArrayList();
        int i = 0;
        while (i <= str.length()) {
            int indexOf = str.indexOf(38, i);
            if (indexOf == -1) {
                indexOf = str.length();
            }
            int indexOf2 = str.indexOf(61, i);
            if (indexOf2 != -1 && indexOf2 <= indexOf) {
                arrayList.add(str.substring(i, indexOf2));
                arrayList.add(str.substring(indexOf2 + 1, indexOf));
            } else {
                arrayList.add(str.substring(i, indexOf));
                arrayList.add(null);
            }
            i = indexOf + 1;
        }
        return arrayList;
    }

    public String d() {
        if (this.f6089d.isEmpty()) {
            return "";
        }
        int indexOf = this.j.indexOf(64);
        return this.j.substring(this.j.indexOf(58, this.f6087b.length() + 3) + 1, indexOf);
    }

    public String e() {
        int indexOf = this.j.indexOf(47, this.f6087b.length() + 3);
        String str = this.j;
        return this.j.substring(indexOf, f.g0.c.k(str, indexOf, str.length(), "?#"));
    }

    public boolean equals(Object obj) {
        return (obj instanceof r) && ((r) obj).j.equals(this.j);
    }

    public List<String> f() {
        int indexOf = this.j.indexOf(47, this.f6087b.length() + 3);
        String str = this.j;
        int k = f.g0.c.k(str, indexOf, str.length(), "?#");
        ArrayList arrayList = new ArrayList();
        while (indexOf < k) {
            int i = indexOf + 1;
            int j = f.g0.c.j(this.j, i, k, '/');
            arrayList.add(this.j.substring(i, j));
            indexOf = j;
        }
        return arrayList;
    }

    public String g() {
        if (this.f6093h == null) {
            return null;
        }
        int indexOf = this.j.indexOf(63) + 1;
        String str = this.j;
        return this.j.substring(indexOf, f.g0.c.j(str, indexOf, str.length(), '#'));
    }

    public String h() {
        if (this.f6088c.isEmpty()) {
            return "";
        }
        int length = this.f6087b.length() + 3;
        String str = this.j;
        return this.j.substring(length, f.g0.c.k(str, length, str.length(), ":@"));
    }

    public int hashCode() {
        return this.j.hashCode();
    }

    public final List<String> l(List<String> list, boolean z) {
        int size = list.size();
        ArrayList arrayList = new ArrayList(size);
        for (int i = 0; i < size; i++) {
            String str = list.get(i);
            arrayList.add(str != null ? j(str, 0, str.length(), z) : null);
        }
        return Collections.unmodifiableList(arrayList);
    }

    public URI o() {
        a aVar = new a();
        aVar.f6094a = this.f6087b;
        aVar.f6095b = h();
        aVar.f6096c = d();
        aVar.f6097d = this.f6090e;
        aVar.f6098e = this.f6091f != c(this.f6087b) ? this.f6091f : -1;
        aVar.f6099f.clear();
        aVar.f6099f.addAll(f());
        aVar.b(g());
        aVar.f6101h = this.i == null ? null : this.j.substring(this.j.indexOf(35) + 1);
        int size = aVar.f6099f.size();
        for (int i = 0; i < size; i++) {
            aVar.f6099f.set(i, b(aVar.f6099f.get(i), "[]", true, true, false, true));
        }
        List<String> list = aVar.f6100g;
        if (list != null) {
            int size2 = list.size();
            for (int i2 = 0; i2 < size2; i2++) {
                String str = aVar.f6100g.get(i2);
                if (str != null) {
                    aVar.f6100g.set(i2, b(str, "\\^`{|}", true, true, true, true));
                }
            }
        }
        String str2 = aVar.f6101h;
        if (str2 != null) {
            aVar.f6101h = b(str2, " \"#<>\\^`{|}", true, true, false, false);
        }
        String aVar2 = aVar.toString();
        try {
            return new URI(aVar2);
        } catch (URISyntaxException e2) {
            try {
                return URI.create(aVar2.replaceAll("[\\u0000-\\u001F\\u007F-\\u009F\\p{javaWhitespace}]", ""));
            } catch (Exception unused) {
                throw new RuntimeException(e2);
            }
        }
    }

    public String toString() {
        return this.j;
    }
}