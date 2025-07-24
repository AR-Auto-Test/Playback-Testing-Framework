package f;

import com.google.common.net.HttpHeaders;
import java.util.concurrent.TimeUnit;

/* compiled from: CacheControl.java */
/* loaded from: classes2.dex */
public final class c {

    /* renamed from: a  reason: collision with root package name */
    public final boolean f5739a;

    /* renamed from: b  reason: collision with root package name */
    public final boolean f5740b;

    /* renamed from: c  reason: collision with root package name */
    public final int f5741c;

    /* renamed from: d  reason: collision with root package name */
    public final int f5742d;

    /* renamed from: e  reason: collision with root package name */
    public final boolean f5743e;

    /* renamed from: f  reason: collision with root package name */
    public final boolean f5744f;

    /* renamed from: g  reason: collision with root package name */
    public final boolean f5745g;

    /* renamed from: h  reason: collision with root package name */
    public final int f5746h;
    public final int i;
    public final boolean j;
    public final boolean k;
    public final boolean l;
    public String m;

    static {
        TimeUnit.SECONDS.toSeconds(Integer.MAX_VALUE);
    }

    public c(boolean z, boolean z2, int i, int i2, boolean z3, boolean z4, boolean z5, int i3, int i4, boolean z6, boolean z7, boolean z8, String str) {
        this.f5739a = z;
        this.f5740b = z2;
        this.f5741c = i;
        this.f5742d = i2;
        this.f5743e = z3;
        this.f5744f = z4;
        this.f5745g = z5;
        this.f5746h = i3;
        this.i = i4;
        this.j = z6;
        this.k = z7;
        this.l = z8;
        this.m = str;
    }

    /* JADX WARN: Removed duplicated region for block: B:15:0x0041  */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public static c a(q qVar) {
        int i;
        int i2;
        String str;
        char charAt;
        q qVar2 = qVar;
        int d2 = qVar.d();
        int i3 = 0;
        boolean z = true;
        String str2 = null;
        boolean z2 = false;
        boolean z3 = false;
        int i4 = -1;
        int i5 = -1;
        boolean z4 = false;
        boolean z5 = false;
        boolean z6 = false;
        int i6 = -1;
        int i7 = -1;
        boolean z7 = false;
        boolean z8 = false;
        boolean z9 = false;
        while (i3 < d2) {
            String b2 = qVar2.b(i3);
            String e2 = qVar2.e(i3);
            if (b2.equalsIgnoreCase(HttpHeaders.CACHE_CONTROL)) {
                if (str2 == null) {
                    str2 = e2;
                    for (i = 0; i < e2.length(); i = i2) {
                        int e3 = f.g0.g.e.e(e2, i, "=,;");
                        String trim = e2.substring(i, e3).trim();
                        if (e3 == e2.length() || e2.charAt(e3) == ',' || e2.charAt(e3) == ';') {
                            i2 = e3 + 1;
                            str = null;
                        } else {
                            while (true) {
                                e3++;
                                if (e3 >= e2.length() || ((charAt = e2.charAt(e3)) != ' ' && charAt != '\t')) {
                                    break;
                                }
                            }
                            if (e3 < e2.length() && e2.charAt(e3) == '\"') {
                                int i8 = e3 + 1;
                                int e4 = f.g0.g.e.e(e2, i8, "\"");
                                str = e2.substring(i8, e4);
                                i2 = e4 + 1;
                            } else {
                                i2 = f.g0.g.e.e(e2, e3, ",;");
                                str = e2.substring(e3, i2).trim();
                            }
                        }
                        if ("no-cache".equalsIgnoreCase(trim)) {
                            z2 = true;
                        } else if ("no-store".equalsIgnoreCase(trim)) {
                            z3 = true;
                        } else if ("max-age".equalsIgnoreCase(trim)) {
                            i4 = f.g0.g.e.c(str, -1);
                        } else if ("s-maxage".equalsIgnoreCase(trim)) {
                            i5 = f.g0.g.e.c(str, -1);
                        } else if ("private".equalsIgnoreCase(trim)) {
                            z4 = true;
                        } else if ("public".equalsIgnoreCase(trim)) {
                            z5 = true;
                        } else if ("must-revalidate".equalsIgnoreCase(trim)) {
                            z6 = true;
                        } else if ("max-stale".equalsIgnoreCase(trim)) {
                            i6 = f.g0.g.e.c(str, Integer.MAX_VALUE);
                        } else if ("min-fresh".equalsIgnoreCase(trim)) {
                            i7 = f.g0.g.e.c(str, -1);
                        } else if ("only-if-cached".equalsIgnoreCase(trim)) {
                            z7 = true;
                        } else if ("no-transform".equalsIgnoreCase(trim)) {
                            z8 = true;
                        } else if ("immutable".equalsIgnoreCase(trim)) {
                            z9 = true;
                        }
                    }
                    i3++;
                    qVar2 = qVar;
                }
            } else if (!b2.equalsIgnoreCase(HttpHeaders.PRAGMA)) {
                i3++;
                qVar2 = qVar;
            }
            z = false;
            while (i < e2.length()) {
            }
            i3++;
            qVar2 = qVar;
        }
        return new c(z2, z3, i4, i5, z4, z5, z6, i6, i7, z7, z8, z9, !z ? null : str2);
    }

    public String toString() {
        String str = this.m;
        if (str == null) {
            StringBuilder sb = new StringBuilder();
            if (this.f5739a) {
                sb.append("no-cache, ");
            }
            if (this.f5740b) {
                sb.append("no-store, ");
            }
            if (this.f5741c != -1) {
                sb.append("max-age=");
                sb.append(this.f5741c);
                sb.append(", ");
            }
            if (this.f5742d != -1) {
                sb.append("s-maxage=");
                sb.append(this.f5742d);
                sb.append(", ");
            }
            if (this.f5743e) {
                sb.append("private, ");
            }
            if (this.f5744f) {
                sb.append("public, ");
            }
            if (this.f5745g) {
                sb.append("must-revalidate, ");
            }
            if (this.f5746h != -1) {
                sb.append("max-stale=");
                sb.append(this.f5746h);
                sb.append(", ");
            }
            if (this.i != -1) {
                sb.append("min-fresh=");
                sb.append(this.i);
                sb.append(", ");
            }
            if (this.j) {
                sb.append("only-if-cached, ");
            }
            if (this.k) {
                sb.append("no-transform, ");
            }
            if (this.l) {
                sb.append("immutable, ");
            }
            if (sb.length() == 0) {
                str = "";
            } else {
                sb.delete(sb.length() - 2, sb.length());
                str = sb.toString();
            }
            this.m = str;
        }
        return str;
    }
}