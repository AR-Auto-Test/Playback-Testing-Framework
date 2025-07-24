package f;

import f.q;
import java.io.Closeable;

/* compiled from: Response.java */
/* loaded from: classes2.dex */
public final class b0 implements Closeable {

    /* renamed from: b  reason: collision with root package name */
    public final y f5724b;

    /* renamed from: c  reason: collision with root package name */
    public final w f5725c;

    /* renamed from: d  reason: collision with root package name */
    public final int f5726d;

    /* renamed from: e  reason: collision with root package name */
    public final String f5727e;

    /* renamed from: f  reason: collision with root package name */
    public final p f5728f;

    /* renamed from: g  reason: collision with root package name */
    public final q f5729g;

    /* renamed from: h  reason: collision with root package name */
    public final d0 f5730h;
    public final b0 i;
    public final b0 j;
    public final b0 k;
    public final long l;
    public final long m;

    public b0(a aVar) {
        this.f5724b = aVar.f5731a;
        this.f5725c = aVar.f5732b;
        this.f5726d = aVar.f5733c;
        this.f5727e = aVar.f5734d;
        this.f5728f = aVar.f5735e;
        this.f5729g = new q(aVar.f5736f);
        this.f5730h = aVar.f5737g;
        this.i = aVar.f5738h;
        this.j = aVar.i;
        this.k = aVar.j;
        this.l = aVar.k;
        this.m = aVar.l;
    }

    public boolean B() {
        int i = this.f5726d;
        return i >= 200 && i < 300;
    }

    @Override // java.io.Closeable, java.lang.AutoCloseable
    public void close() {
        d0 d0Var = this.f5730h;
        if (d0Var != null) {
            d0Var.close();
            return;
        }
        throw new IllegalStateException("response is not eligible for a body and must not be closed");
    }

    public String toString() {
        StringBuilder x = c.b.a.a.a.x("Response{protocol=");
        x.append(this.f5725c);
        x.append(", code=");
        x.append(this.f5726d);
        x.append(", message=");
        x.append(this.f5727e);
        x.append(", url=");
        x.append(this.f5724b.f6150a);
        x.append('}');
        return x.toString();
    }

    /* compiled from: Response.java */
    /* loaded from: classes2.dex */
    public static class a {

        /* renamed from: a  reason: collision with root package name */
        public y f5731a;

        /* renamed from: b  reason: collision with root package name */
        public w f5732b;

        /* renamed from: c  reason: collision with root package name */
        public int f5733c;

        /* renamed from: d  reason: collision with root package name */
        public String f5734d;

        /* renamed from: e  reason: collision with root package name */
        public p f5735e;

        /* renamed from: f  reason: collision with root package name */
        public q.a f5736f;

        /* renamed from: g  reason: collision with root package name */
        public d0 f5737g;

        /* renamed from: h  reason: collision with root package name */
        public b0 f5738h;
        public b0 i;
        public b0 j;
        public long k;
        public long l;

        public a() {
            this.f5733c = -1;
            this.f5736f = new q.a();
        }

        public b0 a() {
            if (this.f5731a != null) {
                if (this.f5732b != null) {
                    if (this.f5733c >= 0) {
                        if (this.f5734d != null) {
                            return new b0(this);
                        }
                        throw new IllegalStateException("message == null");
                    }
                    StringBuilder x = c.b.a.a.a.x("code < 0: ");
                    x.append(this.f5733c);
                    throw new IllegalStateException(x.toString());
                }
                throw new IllegalStateException("protocol == null");
            }
            throw new IllegalStateException("request == null");
        }

        public a b(b0 b0Var) {
            if (b0Var != null) {
                c("cacheResponse", b0Var);
            }
            this.i = b0Var;
            return this;
        }

        public final void c(String str, b0 b0Var) {
            if (b0Var.f5730h == null) {
                if (b0Var.i == null) {
                    if (b0Var.j == null) {
                        if (b0Var.k != null) {
                            throw new IllegalArgumentException(c.b.a.a.a.q(str, ".priorResponse != null"));
                        }
                        return;
                    }
                    throw new IllegalArgumentException(c.b.a.a.a.q(str, ".cacheResponse != null"));
                }
                throw new IllegalArgumentException(c.b.a.a.a.q(str, ".networkResponse != null"));
            }
            throw new IllegalArgumentException(c.b.a.a.a.q(str, ".body != null"));
        }

        public a d(q qVar) {
            this.f5736f = qVar.c();
            return this;
        }

        public a(b0 b0Var) {
            this.f5733c = -1;
            this.f5731a = b0Var.f5724b;
            this.f5732b = b0Var.f5725c;
            this.f5733c = b0Var.f5726d;
            this.f5734d = b0Var.f5727e;
            this.f5735e = b0Var.f5728f;
            this.f5736f = b0Var.f5729g.c();
            this.f5737g = b0Var.f5730h;
            this.f5738h = b0Var.i;
            this.i = b0Var.j;
            this.j = b0Var.k;
            this.k = b0Var.l;
            this.l = b0Var.m;
        }
    }
}