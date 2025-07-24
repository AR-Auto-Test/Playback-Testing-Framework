package f;

import com.google.common.net.HttpHeaders;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.UUID;

/* compiled from: MultipartBody.java */
/* loaded from: classes2.dex */
public final class u extends a0 {

    /* renamed from: a  reason: collision with root package name */
    public static final t f6107a = t.a("multipart/mixed");

    /* renamed from: b  reason: collision with root package name */
    public static final t f6108b;

    /* renamed from: c  reason: collision with root package name */
    public static final byte[] f6109c;

    /* renamed from: d  reason: collision with root package name */
    public static final byte[] f6110d;

    /* renamed from: e  reason: collision with root package name */
    public static final byte[] f6111e;

    /* renamed from: f  reason: collision with root package name */
    public final g.h f6112f;

    /* renamed from: g  reason: collision with root package name */
    public final t f6113g;

    /* renamed from: h  reason: collision with root package name */
    public final List<b> f6114h;
    public long i = -1;

    /* compiled from: MultipartBody.java */
    /* loaded from: classes2.dex */
    public static final class a {

        /* renamed from: a  reason: collision with root package name */
        public final g.h f6115a;

        /* renamed from: b  reason: collision with root package name */
        public t f6116b;

        /* renamed from: c  reason: collision with root package name */
        public final List<b> f6117c;

        public a() {
            String uuid = UUID.randomUUID().toString();
            this.f6116b = u.f6107a;
            this.f6117c = new ArrayList();
            this.f6115a = g.h.e(uuid);
        }

        public a a(String str, String str2) {
            byte[] bytes = str2.getBytes(f.g0.c.i);
            int length = bytes.length;
            f.g0.c.e(bytes.length, 0, length);
            z zVar = new z(null, length, bytes, 0);
            StringBuilder sb = new StringBuilder("form-data; name=");
            u.d(sb, str);
            String[] strArr = (String[]) new String[]{HttpHeaders.CONTENT_DISPOSITION, sb.toString()}.clone();
            for (int i = 0; i < strArr.length; i++) {
                if (strArr[i] != null) {
                    strArr[i] = strArr[i].trim();
                } else {
                    throw new IllegalArgumentException("Headers cannot be null");
                }
            }
            for (int i2 = 0; i2 < strArr.length; i2 += 2) {
                String str3 = strArr[i2];
                String str4 = strArr[i2 + 1];
                if (str3.length() == 0 || str3.indexOf(0) != -1 || str4.indexOf(0) != -1) {
                    throw new IllegalArgumentException("Unexpected header: " + str3 + ": " + str4);
                }
            }
            q qVar = new q(strArr);
            if (qVar.a(HttpHeaders.CONTENT_TYPE) == null) {
                if (qVar.a(HttpHeaders.CONTENT_LENGTH) == null) {
                    this.f6117c.add(new b(qVar, zVar));
                    return this;
                }
                throw new IllegalArgumentException("Unexpected header: Content-Length");
            }
            throw new IllegalArgumentException("Unexpected header: Content-Type");
        }

        public u b() {
            if (!this.f6117c.isEmpty()) {
                return new u(this.f6115a, this.f6116b, this.f6117c);
            }
            throw new IllegalStateException("Multipart body must have at least one part.");
        }

        public a c(t tVar) {
            Objects.requireNonNull(tVar, "type == null");
            if (tVar.f6105d.equals("multipart")) {
                this.f6116b = tVar;
                return this;
            }
            throw new IllegalArgumentException("multipart != " + tVar);
        }
    }

    /* compiled from: MultipartBody.java */
    /* loaded from: classes2.dex */
    public static final class b {

        /* renamed from: a  reason: collision with root package name */
        public final q f6118a;

        /* renamed from: b  reason: collision with root package name */
        public final a0 f6119b;

        public b(q qVar, a0 a0Var) {
            this.f6118a = qVar;
            this.f6119b = a0Var;
        }
    }

    static {
        t.a("multipart/alternative");
        t.a("multipart/digest");
        t.a("multipart/parallel");
        f6108b = t.a("multipart/form-data");
        f6109c = new byte[]{58, 32};
        f6110d = new byte[]{13, 10};
        f6111e = new byte[]{45, 45};
    }

    public u(g.h hVar, t tVar, List<b> list) {
        this.f6112f = hVar;
        this.f6113g = t.a(tVar + "; boundary=" + hVar.p());
        this.f6114h = f.g0.c.p(list);
    }

    public static StringBuilder d(StringBuilder sb, String str) {
        sb.append('\"');
        int length = str.length();
        for (int i = 0; i < length; i++) {
            char charAt = str.charAt(i);
            if (charAt == '\n') {
                sb.append("%0A");
            } else if (charAt == '\r') {
                sb.append("%0D");
            } else if (charAt != '\"') {
                sb.append(charAt);
            } else {
                sb.append("%22");
            }
        }
        sb.append('\"');
        return sb;
    }

    @Override // f.a0
    public long a() {
        long j = this.i;
        if (j != -1) {
            return j;
        }
        long e2 = e(null, true);
        this.i = e2;
        return e2;
    }

    @Override // f.a0
    public t b() {
        return this.f6113g;
    }

    @Override // f.a0
    public void c(g.f fVar) {
        e(fVar, false);
    }

    /* JADX DEBUG: Multi-variable search result rejected for r0v0, resolved type: g.e */
    /* JADX DEBUG: Multi-variable search result rejected for r0v1, resolved type: g.e */
    /* JADX DEBUG: Multi-variable search result rejected for r0v2, resolved type: g.e */
    /* JADX WARN: Multi-variable type inference failed */
    public final long e(g.f fVar, boolean z) {
        g.e eVar;
        if (z) {
            fVar = new g.e();
            eVar = fVar;
        } else {
            eVar = 0;
        }
        int size = this.f6114h.size();
        long j = 0;
        for (int i = 0; i < size; i++) {
            b bVar = this.f6114h.get(i);
            q qVar = bVar.f6118a;
            a0 a0Var = bVar.f6119b;
            fVar.write(f6111e);
            fVar.s(this.f6112f);
            fVar.write(f6110d);
            if (qVar != null) {
                int d2 = qVar.d();
                for (int i2 = 0; i2 < d2; i2++) {
                    fVar.i(qVar.b(i2)).write(f6109c).i(qVar.e(i2)).write(f6110d);
                }
            }
            t b2 = a0Var.b();
            if (b2 != null) {
                fVar.i("Content-Type: ").i(b2.f6104c).write(f6110d);
            }
            long a2 = a0Var.a();
            if (a2 != -1) {
                fVar.i("Content-Length: ").w(a2).write(f6110d);
            } else if (z) {
                eVar.B();
                return -1L;
            }
            byte[] bArr = f6110d;
            fVar.write(bArr);
            if (z) {
                j += a2;
            } else {
                a0Var.c(fVar);
            }
            fVar.write(bArr);
        }
        byte[] bArr2 = f6111e;
        fVar.write(bArr2);
        fVar.s(this.f6112f);
        fVar.write(bArr2);
        fVar.write(f6110d);
        if (z) {
            long j2 = j + eVar.f6176d;
            eVar.B();
            return j2;
        }
        return j;
    }
}