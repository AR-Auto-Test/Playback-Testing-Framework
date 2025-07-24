package f;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/* compiled from: Headers.java */
/* loaded from: classes2.dex */
public final class q {

    /* renamed from: a  reason: collision with root package name */
    public final String[] f6084a;

    /* compiled from: Headers.java */
    /* loaded from: classes2.dex */
    public static final class a {

        /* renamed from: a  reason: collision with root package name */
        public final List<String> f6085a = new ArrayList(20);

        public a a(String str, String str2) {
            this.f6085a.add(str);
            this.f6085a.add(str2.trim());
            return this;
        }

        public final void b(String str, String str2) {
            Objects.requireNonNull(str, "name == null");
            if (!str.isEmpty()) {
                int length = str.length();
                for (int i = 0; i < length; i++) {
                    char charAt = str.charAt(i);
                    if (charAt <= ' ' || charAt >= 127) {
                        throw new IllegalArgumentException(f.g0.c.n("Unexpected char %#04x at %d in header name: %s", Integer.valueOf(charAt), Integer.valueOf(i), str));
                    }
                }
                if (str2 != null) {
                    int length2 = str2.length();
                    for (int i2 = 0; i2 < length2; i2++) {
                        char charAt2 = str2.charAt(i2);
                        if ((charAt2 <= 31 && charAt2 != '\t') || charAt2 >= 127) {
                            throw new IllegalArgumentException(f.g0.c.n("Unexpected char %#04x at %d in %s value: %s", Integer.valueOf(charAt2), Integer.valueOf(i2), str, str2));
                        }
                    }
                    return;
                }
                throw new NullPointerException(c.b.a.a.a.r("value for name ", str, " == null"));
            }
            throw new IllegalArgumentException("name is empty");
        }

        public a c(String str) {
            int i = 0;
            while (i < this.f6085a.size()) {
                if (str.equalsIgnoreCase(this.f6085a.get(i))) {
                    this.f6085a.remove(i);
                    this.f6085a.remove(i);
                    i -= 2;
                }
                i += 2;
            }
            return this;
        }
    }

    public q(a aVar) {
        List<String> list = aVar.f6085a;
        this.f6084a = (String[]) list.toArray(new String[list.size()]);
    }

    public String a(String str) {
        String[] strArr = this.f6084a;
        int length = strArr.length;
        do {
            length -= 2;
            if (length < 0) {
                return null;
            }
        } while (!str.equalsIgnoreCase(strArr[length]));
        return strArr[length + 1];
    }

    public String b(int i) {
        return this.f6084a[i * 2];
    }

    public a c() {
        a aVar = new a();
        Collections.addAll(aVar.f6085a, this.f6084a);
        return aVar;
    }

    public int d() {
        return this.f6084a.length / 2;
    }

    public String e(int i) {
        return this.f6084a[(i * 2) + 1];
    }

    public boolean equals(Object obj) {
        return (obj instanceof q) && Arrays.equals(((q) obj).f6084a, this.f6084a);
    }

    public int hashCode() {
        return Arrays.hashCode(this.f6084a);
    }

    public String toString() {
        StringBuilder sb = new StringBuilder();
        int d2 = d();
        for (int i = 0; i < d2; i++) {
            sb.append(b(i));
            sb.append(": ");
            sb.append(e(i));
            sb.append("\n");
        }
        return sb.toString();
    }

    public q(String[] strArr) {
        this.f6084a = strArr;
    }
}