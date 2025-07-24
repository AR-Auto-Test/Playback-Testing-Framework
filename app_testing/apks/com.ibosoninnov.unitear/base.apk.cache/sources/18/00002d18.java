package f.g0.i;

import com.google.common.primitives.UnsignedBytes;
import com.google.firebase.analytics.FirebaseAnalytics;
import f.g0.i.r;
import g.x;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.logging.Logger;

/* compiled from: Hpack.java */
/* loaded from: classes2.dex */
public final class d {

    /* renamed from: a  reason: collision with root package name */
    public static final c[] f5881a;

    /* renamed from: b  reason: collision with root package name */
    public static final Map<g.h, Integer> f5882b;

    /* compiled from: Hpack.java */
    /* loaded from: classes2.dex */
    public static final class a {

        /* renamed from: b  reason: collision with root package name */
        public final g.g f5884b;

        /* renamed from: c  reason: collision with root package name */
        public final int f5885c;

        /* renamed from: d  reason: collision with root package name */
        public int f5886d;

        /* renamed from: a  reason: collision with root package name */
        public final List<c> f5883a = new ArrayList();

        /* renamed from: e  reason: collision with root package name */
        public c[] f5887e = new c[8];

        /* renamed from: f  reason: collision with root package name */
        public int f5888f = 7;

        /* renamed from: g  reason: collision with root package name */
        public int f5889g = 0;

        /* renamed from: h  reason: collision with root package name */
        public int f5890h = 0;

        public a(int i, x xVar) {
            this.f5885c = i;
            this.f5886d = i;
            Logger logger = g.o.f6197a;
            this.f5884b = new g.s(xVar);
        }

        public final void a() {
            Arrays.fill(this.f5887e, (Object) null);
            this.f5888f = this.f5887e.length - 1;
            this.f5889g = 0;
            this.f5890h = 0;
        }

        public final int b(int i) {
            return this.f5888f + 1 + i;
        }

        public final int c(int i) {
            int i2;
            int i3 = 0;
            if (i > 0) {
                int length = this.f5887e.length;
                while (true) {
                    length--;
                    i2 = this.f5888f;
                    if (length < i2 || i <= 0) {
                        break;
                    }
                    c[] cVarArr = this.f5887e;
                    i -= cVarArr[length].i;
                    this.f5890h -= cVarArr[length].i;
                    this.f5889g--;
                    i3++;
                }
                c[] cVarArr2 = this.f5887e;
                System.arraycopy(cVarArr2, i2 + 1, cVarArr2, i2 + 1 + i3, this.f5889g);
                this.f5888f += i3;
            }
            return i3;
        }

        public final g.h d(int i) {
            if (i >= 0 && i <= d.f5881a.length - 1) {
                return d.f5881a[i].f5879g;
            }
            int b2 = b(i - d.f5881a.length);
            if (b2 >= 0) {
                c[] cVarArr = this.f5887e;
                if (b2 < cVarArr.length) {
                    return cVarArr[b2].f5879g;
                }
            }
            StringBuilder x = c.b.a.a.a.x("Header index too large ");
            x.append(i + 1);
            throw new IOException(x.toString());
        }

        public final void e(int i, c cVar) {
            this.f5883a.add(cVar);
            int i2 = cVar.i;
            if (i != -1) {
                i2 -= this.f5887e[(this.f5888f + 1) + i].i;
            }
            int i3 = this.f5886d;
            if (i2 > i3) {
                a();
                return;
            }
            int c2 = c((this.f5890h + i2) - i3);
            if (i == -1) {
                int i4 = this.f5889g + 1;
                c[] cVarArr = this.f5887e;
                if (i4 > cVarArr.length) {
                    c[] cVarArr2 = new c[cVarArr.length * 2];
                    System.arraycopy(cVarArr, 0, cVarArr2, cVarArr.length, cVarArr.length);
                    this.f5888f = this.f5887e.length - 1;
                    this.f5887e = cVarArr2;
                }
                int i5 = this.f5888f;
                this.f5888f = i5 - 1;
                this.f5887e[i5] = cVar;
                this.f5889g++;
            } else {
                this.f5887e[this.f5888f + 1 + i + c2 + i] = cVar;
            }
            this.f5890h += i2;
        }

        public g.h f() {
            int readByte = this.f5884b.readByte() & UnsignedBytes.MAX_VALUE;
            boolean z = (readByte & 128) == 128;
            int g2 = g(readByte, 127);
            if (z) {
                r rVar = r.f5998c;
                byte[] r = this.f5884b.r(g2);
                Objects.requireNonNull(rVar);
                ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
                int i = 0;
                r.a aVar = rVar.f5999d;
                int i2 = 0;
                for (byte b2 : r) {
                    i2 = (i2 << 8) | (b2 & UnsignedBytes.MAX_VALUE);
                    i += 8;
                    while (i >= 8) {
                        int i3 = i - 8;
                        aVar = aVar.f6000a[(i2 >>> i3) & 255];
                        if (aVar.f6000a == null) {
                            byteArrayOutputStream.write(aVar.f6001b);
                            i -= aVar.f6002c;
                            aVar = rVar.f5999d;
                        } else {
                            i = i3;
                        }
                    }
                }
                while (i > 0) {
                    r.a aVar2 = aVar.f6000a[(i2 << (8 - i)) & 255];
                    if (aVar2.f6000a != null || aVar2.f6002c > i) {
                        break;
                    }
                    byteArrayOutputStream.write(aVar2.f6001b);
                    i -= aVar2.f6002c;
                    aVar = rVar.f5999d;
                }
                return g.h.i(byteArrayOutputStream.toByteArray());
            }
            return this.f5884b.d(g2);
        }

        public int g(int i, int i2) {
            int i3 = i & i2;
            if (i3 < i2) {
                return i3;
            }
            int i4 = 0;
            while (true) {
                int readByte = this.f5884b.readByte() & UnsignedBytes.MAX_VALUE;
                if ((readByte & 128) == 0) {
                    return i2 + (readByte << i4);
                }
                i2 += (readByte & 127) << i4;
                i4 += 7;
            }
        }
    }

    /* compiled from: Hpack.java */
    /* loaded from: classes2.dex */
    public static final class b {

        /* renamed from: a  reason: collision with root package name */
        public final g.e f5891a;

        /* renamed from: c  reason: collision with root package name */
        public boolean f5893c;

        /* renamed from: b  reason: collision with root package name */
        public int f5892b = Integer.MAX_VALUE;

        /* renamed from: e  reason: collision with root package name */
        public c[] f5895e = new c[8];

        /* renamed from: f  reason: collision with root package name */
        public int f5896f = 7;

        /* renamed from: g  reason: collision with root package name */
        public int f5897g = 0;

        /* renamed from: h  reason: collision with root package name */
        public int f5898h = 0;

        /* renamed from: d  reason: collision with root package name */
        public int f5894d = 4096;

        public b(g.e eVar) {
            this.f5891a = eVar;
        }

        public final void a() {
            Arrays.fill(this.f5895e, (Object) null);
            this.f5896f = this.f5895e.length - 1;
            this.f5897g = 0;
            this.f5898h = 0;
        }

        public final int b(int i) {
            int i2;
            int i3 = 0;
            if (i > 0) {
                int length = this.f5895e.length;
                while (true) {
                    length--;
                    i2 = this.f5896f;
                    if (length < i2 || i <= 0) {
                        break;
                    }
                    c[] cVarArr = this.f5895e;
                    i -= cVarArr[length].i;
                    this.f5898h -= cVarArr[length].i;
                    this.f5897g--;
                    i3++;
                }
                c[] cVarArr2 = this.f5895e;
                System.arraycopy(cVarArr2, i2 + 1, cVarArr2, i2 + 1 + i3, this.f5897g);
                c[] cVarArr3 = this.f5895e;
                int i4 = this.f5896f;
                Arrays.fill(cVarArr3, i4 + 1, i4 + 1 + i3, (Object) null);
                this.f5896f += i3;
            }
            return i3;
        }

        public final void c(c cVar) {
            int i = cVar.i;
            int i2 = this.f5894d;
            if (i > i2) {
                a();
                return;
            }
            b((this.f5898h + i) - i2);
            int i3 = this.f5897g + 1;
            c[] cVarArr = this.f5895e;
            if (i3 > cVarArr.length) {
                c[] cVarArr2 = new c[cVarArr.length * 2];
                System.arraycopy(cVarArr, 0, cVarArr2, cVarArr.length, cVarArr.length);
                this.f5896f = this.f5895e.length - 1;
                this.f5895e = cVarArr2;
            }
            int i4 = this.f5896f;
            this.f5896f = i4 - 1;
            this.f5895e[i4] = cVar;
            this.f5897g++;
            this.f5898h += i;
        }

        public void d(g.h hVar) {
            Objects.requireNonNull(r.f5998c);
            long j = 0;
            int i = 0;
            long j2 = 0;
            for (int i2 = 0; i2 < hVar.l(); i2++) {
                j2 += r.f5997b[hVar.f(i2) & UnsignedBytes.MAX_VALUE];
            }
            if (((int) ((j2 + 7) >> 3)) < hVar.l()) {
                g.e eVar = new g.e();
                Objects.requireNonNull(r.f5998c);
                byte b2 = 0;
                while (i < hVar.l()) {
                    int f2 = hVar.f(i) & UnsignedBytes.MAX_VALUE;
                    int i3 = r.f5996a[f2];
                    byte b3 = r.f5997b[f2];
                    j = (j << b3) | i3;
                    int i4 = b2 + b3;
                    while (i4 >= 8) {
                        i4 = (i4 == 1 ? 1 : 0) - 8;
                        eVar.writeByte((int) (j >> i4));
                    }
                    i++;
                    b2 = i4;
                }
                if (b2 > 0) {
                    eVar.writeByte((int) ((j << (8 - b2)) | (255 >>> b2)));
                }
                g.h H = eVar.H();
                f(H.f6180d.length, 127, 128);
                this.f5891a.P(H);
                return;
            }
            f(hVar.l(), 127, 0);
            this.f5891a.P(hVar);
        }

        /* JADX WARN: Removed duplicated region for block: B:26:0x006c  */
        /* JADX WARN: Removed duplicated region for block: B:37:0x00a4  */
        /* JADX WARN: Removed duplicated region for block: B:38:0x00ac  */
        /*
            Code decompiled incorrectly, please refer to instructions dump.
        */
        public void e(List<c> list) {
            int i;
            int i2;
            if (this.f5893c) {
                int i3 = this.f5892b;
                if (i3 < this.f5894d) {
                    f(i3, 31, 32);
                }
                this.f5893c = false;
                this.f5892b = Integer.MAX_VALUE;
                f(this.f5894d, 31, 32);
            }
            int size = list.size();
            for (int i4 = 0; i4 < size; i4++) {
                c cVar = list.get(i4);
                g.h n = cVar.f5879g.n();
                g.h hVar = cVar.f5880h;
                Integer num = d.f5882b.get(n);
                if (num != null) {
                    i = num.intValue() + 1;
                    if (i > 1 && i < 8) {
                        c[] cVarArr = d.f5881a;
                        if (!f.g0.c.m(cVarArr[i - 1].f5880h, hVar)) {
                            if (f.g0.c.m(cVarArr[i].f5880h, hVar)) {
                                i2 = i + 1;
                                if (i2 == -1) {
                                    int i5 = this.f5896f + 1;
                                    int length = this.f5895e.length;
                                    while (true) {
                                        if (i5 >= length) {
                                            break;
                                        }
                                        if (f.g0.c.m(this.f5895e[i5].f5879g, n)) {
                                            if (f.g0.c.m(this.f5895e[i5].f5880h, hVar)) {
                                                i2 = d.f5881a.length + (i5 - this.f5896f);
                                                break;
                                            } else if (i == -1) {
                                                i = (i5 - this.f5896f) + d.f5881a.length;
                                            }
                                        }
                                        i5++;
                                    }
                                }
                                if (i2 != -1) {
                                    f(i2, 127, 128);
                                } else if (i == -1) {
                                    this.f5891a.T(64);
                                    d(n);
                                    d(hVar);
                                    c(cVar);
                                } else {
                                    g.h hVar2 = c.f5873a;
                                    Objects.requireNonNull(n);
                                    if (n.j(0, hVar2, 0, hVar2.l()) && !c.f5878f.equals(n)) {
                                        f(i, 15, 0);
                                        d(hVar);
                                    } else {
                                        f(i, 63, 64);
                                        d(hVar);
                                        c(cVar);
                                    }
                                }
                            }
                        }
                    }
                    i2 = -1;
                    if (i2 == -1) {
                    }
                    if (i2 != -1) {
                    }
                } else {
                    i = -1;
                }
                i2 = i;
                if (i2 == -1) {
                }
                if (i2 != -1) {
                }
            }
        }

        public void f(int i, int i2, int i3) {
            if (i < i2) {
                this.f5891a.T(i | i3);
                return;
            }
            this.f5891a.T(i3 | i2);
            int i4 = i - i2;
            while (i4 >= 128) {
                this.f5891a.T(128 | (i4 & 127));
                i4 >>>= 7;
            }
            this.f5891a.T(i4);
        }
    }

    static {
        c cVar = new c(c.f5878f, "");
        int i = 0;
        g.h hVar = c.f5875c;
        g.h hVar2 = c.f5876d;
        g.h hVar3 = c.f5877e;
        g.h hVar4 = c.f5874b;
        c[] cVarArr = {cVar, new c(hVar, "GET"), new c(hVar, "POST"), new c(hVar2, "/"), new c(hVar2, "/index.html"), new c(hVar3, "http"), new c(hVar3, "https"), new c(hVar4, "200"), new c(hVar4, "204"), new c(hVar4, "206"), new c(hVar4, "304"), new c(hVar4, "400"), new c(hVar4, "404"), new c(hVar4, "500"), new c("accept-charset", ""), new c("accept-encoding", "gzip, deflate"), new c("accept-language", ""), new c("accept-ranges", ""), new c("accept", ""), new c("access-control-allow-origin", ""), new c("age", ""), new c("allow", ""), new c("authorization", ""), new c("cache-control", ""), new c("content-disposition", ""), new c("content-encoding", ""), new c("content-language", ""), new c("content-length", ""), new c("content-location", ""), new c("content-range", ""), new c("content-type", ""), new c("cookie", ""), new c("date", ""), new c("etag", ""), new c("expect", ""), new c("expires", ""), new c("from", ""), new c("host", ""), new c("if-match", ""), new c("if-modified-since", ""), new c("if-none-match", ""), new c("if-range", ""), new c("if-unmodified-since", ""), new c("last-modified", ""), new c("link", ""), new c(FirebaseAnalytics.Param.LOCATION, ""), new c("max-forwards", ""), new c("proxy-authenticate", ""), new c("proxy-authorization", ""), new c("range", ""), new c("referer", ""), new c("refresh", ""), new c("retry-after", ""), new c("server", ""), new c("set-cookie", ""), new c("strict-transport-security", ""), new c("transfer-encoding", ""), new c("user-agent", ""), new c("vary", ""), new c("via", ""), new c("www-authenticate", "")};
        f5881a = cVarArr;
        LinkedHashMap linkedHashMap = new LinkedHashMap(cVarArr.length);
        while (true) {
            c[] cVarArr2 = f5881a;
            if (i < cVarArr2.length) {
                if (!linkedHashMap.containsKey(cVarArr2[i].f5879g)) {
                    linkedHashMap.put(cVarArr2[i].f5879g, Integer.valueOf(i));
                }
                i++;
            } else {
                f5882b = Collections.unmodifiableMap(linkedHashMap);
                return;
            }
        }
    }

    public static g.h a(g.h hVar) {
        int l = hVar.l();
        for (int i = 0; i < l; i++) {
            byte f2 = hVar.f(i);
            if (f2 >= 65 && f2 <= 90) {
                StringBuilder x = c.b.a.a.a.x("PROTOCOL_ERROR response malformed: mixed case name: ");
                x.append(hVar.p());
                throw new IOException(x.toString());
            }
        }
        return hVar;
    }
}