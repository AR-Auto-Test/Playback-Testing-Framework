package f;

import f.q;
import f.r;
import java.util.Objects;

/* compiled from: Request.java */
/* loaded from: classes2.dex */
public final class y {

    /* renamed from: a  reason: collision with root package name */
    public final r f6150a;

    /* renamed from: b  reason: collision with root package name */
    public final String f6151b;

    /* renamed from: c  reason: collision with root package name */
    public final q f6152c;

    /* renamed from: d  reason: collision with root package name */
    public final a0 f6153d;

    /* renamed from: e  reason: collision with root package name */
    public final Object f6154e;

    /* renamed from: f  reason: collision with root package name */
    public volatile c f6155f;

    public y(a aVar) {
        this.f6150a = aVar.f6156a;
        this.f6151b = aVar.f6157b;
        this.f6152c = new q(aVar.f6158c);
        this.f6153d = aVar.f6159d;
        Object obj = aVar.f6160e;
        this.f6154e = obj == null ? this : obj;
    }

    public c a() {
        c cVar = this.f6155f;
        if (cVar != null) {
            return cVar;
        }
        c a2 = c.a(this.f6152c);
        this.f6155f = a2;
        return a2;
    }

    public String toString() {
        StringBuilder x = c.b.a.a.a.x("Request{method=");
        x.append(this.f6151b);
        x.append(", url=");
        x.append(this.f6150a);
        x.append(", tag=");
        Object obj = this.f6154e;
        if (obj == this) {
            obj = null;
        }
        x.append(obj);
        x.append('}');
        return x.toString();
    }

    /* compiled from: Request.java */
    /* loaded from: classes2.dex */
    public static class a {

        /* renamed from: a  reason: collision with root package name */
        public r f6156a;

        /* renamed from: b  reason: collision with root package name */
        public String f6157b;

        /* renamed from: c  reason: collision with root package name */
        public q.a f6158c;

        /* renamed from: d  reason: collision with root package name */
        public a0 f6159d;

        /* renamed from: e  reason: collision with root package name */
        public Object f6160e;

        public a() {
            this.f6157b = "GET";
            this.f6158c = new q.a();
        }

        public y a() {
            if (this.f6156a != null) {
                return new y(this);
            }
            throw new IllegalStateException("url == null");
        }

        public a b(String str, String str2) {
            q.a aVar = this.f6158c;
            aVar.b(str, str2);
            aVar.c(str);
            aVar.f6085a.add(str);
            aVar.f6085a.add(str2.trim());
            return this;
        }

        public a c(String str, a0 a0Var) {
            if (str.length() != 0) {
                if (a0Var != null && !b.v.u.c.w(str)) {
                    throw new IllegalArgumentException(c.b.a.a.a.r("method ", str, " must not have a request body."));
                }
                if (a0Var == null) {
                    if (str.equals("POST") || str.equals("PUT") || str.equals("PATCH") || str.equals("PROPPATCH") || str.equals("REPORT")) {
                        throw new IllegalArgumentException(c.b.a.a.a.r("method ", str, " must have a request body."));
                    }
                }
                this.f6157b = str;
                this.f6159d = a0Var;
                return this;
            }
            throw new IllegalArgumentException("method.length() == 0");
        }

        public a d(String str) {
            Objects.requireNonNull(str, "url == null");
            if (str.regionMatches(true, 0, "ws:", 0, 3)) {
                StringBuilder x = c.b.a.a.a.x("http:");
                x.append(str.substring(3));
                str = x.toString();
            } else if (str.regionMatches(true, 0, "wss:", 0, 4)) {
                StringBuilder x2 = c.b.a.a.a.x("https:");
                x2.append(str.substring(4));
                str = x2.toString();
            }
            r.a aVar = new r.a();
            r a2 = aVar.c(null, str) == 1 ? aVar.a() : null;
            if (a2 != null) {
                e(a2);
                return this;
            }
            throw new IllegalArgumentException(c.b.a.a.a.q("unexpected url: ", str));
        }

        public a e(r rVar) {
            Objects.requireNonNull(rVar, "url == null");
            this.f6156a = rVar;
            return this;
        }

        public a(y yVar) {
            this.f6156a = yVar.f6150a;
            this.f6157b = yVar.f6151b;
            this.f6159d = yVar.f6153d;
            this.f6160e = yVar.f6154e;
            this.f6158c = yVar.f6152c.c();
        }
    }
}