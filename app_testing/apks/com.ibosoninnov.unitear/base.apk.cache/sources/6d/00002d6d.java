package g;

import java.io.IOException;

/* compiled from: AsyncTimeout.java */
/* loaded from: classes2.dex */
public class a implements w {

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ w f6165b;

    /* renamed from: c  reason: collision with root package name */
    public final /* synthetic */ c f6166c;

    public a(c cVar, w wVar) {
        this.f6166c = cVar;
        this.f6165b = wVar;
    }

    @Override // g.w
    public y b() {
        return this.f6166c;
    }

    @Override // g.w, java.io.Closeable, java.lang.AutoCloseable
    public void close() {
        this.f6166c.i();
        try {
            try {
                this.f6165b.close();
                this.f6166c.j(true);
            } catch (IOException e2) {
                c cVar = this.f6166c;
                if (!cVar.k()) {
                    throw e2;
                }
                throw cVar.l(e2);
            }
        } catch (Throwable th) {
            this.f6166c.j(false);
            throw th;
        }
    }

    @Override // g.w, java.io.Flushable
    public void flush() {
        this.f6166c.i();
        try {
            try {
                this.f6165b.flush();
                this.f6166c.j(true);
            } catch (IOException e2) {
                c cVar = this.f6166c;
                if (!cVar.k()) {
                    throw e2;
                }
                throw cVar.l(e2);
            }
        } catch (Throwable th) {
            this.f6166c.j(false);
            throw th;
        }
    }

    @Override // g.w
    public void l(e eVar, long j) {
        z.b(eVar.f6176d, 0L, j);
        while (true) {
            long j2 = 0;
            if (j <= 0) {
                return;
            }
            t tVar = eVar.f6175c;
            while (true) {
                if (j2 >= 65536) {
                    break;
                }
                j2 += tVar.f6211c - tVar.f6210b;
                if (j2 >= j) {
                    j2 = j;
                    break;
                }
                tVar = tVar.f6214f;
            }
            this.f6166c.i();
            try {
                try {
                    this.f6165b.l(eVar, j2);
                    j -= j2;
                    this.f6166c.j(true);
                } catch (IOException e2) {
                    c cVar = this.f6166c;
                    if (!cVar.k()) {
                        throw e2;
                    }
                    throw cVar.l(e2);
                }
            } catch (Throwable th) {
                this.f6166c.j(false);
                throw th;
            }
        }
    }

    public String toString() {
        StringBuilder x = c.b.a.a.a.x("AsyncTimeout.sink(");
        x.append(this.f6165b);
        x.append(")");
        return x.toString();
    }
}