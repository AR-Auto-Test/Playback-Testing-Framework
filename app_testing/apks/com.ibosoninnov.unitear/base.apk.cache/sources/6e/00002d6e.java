package g;

import java.io.IOException;

/* compiled from: AsyncTimeout.java */
/* loaded from: classes2.dex */
public class b implements x {

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ x f6167b;

    /* renamed from: c  reason: collision with root package name */
    public final /* synthetic */ c f6168c;

    public b(c cVar, x xVar) {
        this.f6168c = cVar;
        this.f6167b = xVar;
    }

    @Override // g.x
    public y b() {
        return this.f6168c;
    }

    @Override // g.x, java.io.Closeable, java.lang.AutoCloseable
    public void close() {
        this.f6168c.i();
        try {
            try {
                this.f6167b.close();
                this.f6168c.j(true);
            } catch (IOException e2) {
                c cVar = this.f6168c;
                if (!cVar.k()) {
                    throw e2;
                }
                throw cVar.l(e2);
            }
        } catch (Throwable th) {
            this.f6168c.j(false);
            throw th;
        }
    }

    public String toString() {
        StringBuilder x = c.b.a.a.a.x("AsyncTimeout.source(");
        x.append(this.f6167b);
        x.append(")");
        return x.toString();
    }

    @Override // g.x
    public long u(e eVar, long j) {
        this.f6168c.i();
        try {
            try {
                long u = this.f6167b.u(eVar, j);
                this.f6168c.j(true);
                return u;
            } catch (IOException e2) {
                c cVar = this.f6168c;
                if (!cVar.k()) {
                    throw e2;
                }
                throw cVar.l(e2);
            }
        } catch (Throwable th) {
            this.f6168c.j(false);
            throw th;
        }
    }
}