package g;

import java.io.OutputStream;

/* compiled from: Okio.java */
/* loaded from: classes2.dex */
public final class n implements w {

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ y f6195b;

    /* renamed from: c  reason: collision with root package name */
    public final /* synthetic */ OutputStream f6196c;

    public n(y yVar, OutputStream outputStream) {
        this.f6195b = yVar;
        this.f6196c = outputStream;
    }

    @Override // g.w
    public y b() {
        return this.f6195b;
    }

    @Override // g.w, java.io.Closeable, java.lang.AutoCloseable
    public void close() {
        this.f6196c.close();
    }

    @Override // g.w, java.io.Flushable
    public void flush() {
        this.f6196c.flush();
    }

    @Override // g.w
    public void l(e eVar, long j) {
        z.b(eVar.f6176d, 0L, j);
        while (j > 0) {
            this.f6195b.f();
            t tVar = eVar.f6175c;
            int min = (int) Math.min(j, tVar.f6211c - tVar.f6210b);
            this.f6196c.write(tVar.f6209a, tVar.f6210b, min);
            int i = tVar.f6210b + min;
            tVar.f6210b = i;
            long j2 = min;
            j -= j2;
            eVar.f6176d -= j2;
            if (i == tVar.f6211c) {
                eVar.f6175c = tVar.a();
                u.a(tVar);
            }
        }
    }

    public String toString() {
        StringBuilder x = c.b.a.a.a.x("sink(");
        x.append(this.f6196c);
        x.append(")");
        return x.toString();
    }
}