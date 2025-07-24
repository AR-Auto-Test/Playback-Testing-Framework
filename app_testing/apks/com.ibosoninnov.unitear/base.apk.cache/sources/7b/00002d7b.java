package g;

import java.io.EOFException;
import java.io.IOException;
import java.util.zip.DataFormatException;
import java.util.zip.Inflater;

/* compiled from: InflaterSource.java */
/* loaded from: classes2.dex */
public final class m implements x {

    /* renamed from: b  reason: collision with root package name */
    public final g f6191b;

    /* renamed from: c  reason: collision with root package name */
    public final Inflater f6192c;

    /* renamed from: d  reason: collision with root package name */
    public int f6193d;

    /* renamed from: e  reason: collision with root package name */
    public boolean f6194e;

    public m(g gVar, Inflater inflater) {
        this.f6191b = gVar;
        this.f6192c = inflater;
    }

    public final void B() {
        int i = this.f6193d;
        if (i == 0) {
            return;
        }
        int remaining = i - this.f6192c.getRemaining();
        this.f6193d -= remaining;
        this.f6191b.c(remaining);
    }

    @Override // g.x
    public y b() {
        return this.f6191b.b();
    }

    @Override // g.x, java.io.Closeable, java.lang.AutoCloseable
    public void close() {
        if (this.f6194e) {
            return;
        }
        this.f6192c.end();
        this.f6194e = true;
        this.f6191b.close();
    }

    /* JADX WARN: Code restructure failed: missing block: B:30:0x0085, code lost:
        B();
     */
    /* JADX WARN: Code restructure failed: missing block: B:31:0x008c, code lost:
        if (r0.f6210b != r0.f6211c) goto L37;
     */
    /* JADX WARN: Code restructure failed: missing block: B:32:0x008e, code lost:
        r7.f6175c = r0.a();
        g.u.a(r0);
     */
    /* JADX WARN: Code restructure failed: missing block: B:33:0x0097, code lost:
        return -1;
     */
    /* JADX WARN: Code restructure failed: missing block: B:50:?, code lost:
        return -1;
     */
    @Override // g.x
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public long u(e eVar, long j) {
        int i = (j > 0L ? 1 : (j == 0L ? 0 : -1));
        if (i >= 0) {
            if (this.f6194e) {
                throw new IllegalStateException("closed");
            }
            if (i == 0) {
                return 0L;
            }
            while (true) {
                boolean z = false;
                if (this.f6192c.needsInput()) {
                    B();
                    if (this.f6192c.getRemaining() == 0) {
                        if (this.f6191b.f()) {
                            z = true;
                        } else {
                            t tVar = this.f6191b.a().f6175c;
                            int i2 = tVar.f6211c;
                            int i3 = tVar.f6210b;
                            int i4 = i2 - i3;
                            this.f6193d = i4;
                            this.f6192c.setInput(tVar.f6209a, i3, i4);
                        }
                    } else {
                        throw new IllegalStateException("?");
                    }
                }
                try {
                    t O = eVar.O(1);
                    int inflate = this.f6192c.inflate(O.f6209a, O.f6211c, (int) Math.min(j, 8192 - O.f6211c));
                    if (inflate > 0) {
                        O.f6211c += inflate;
                        long j2 = inflate;
                        eVar.f6176d += j2;
                        return j2;
                    } else if (this.f6192c.finished() || this.f6192c.needsDictionary()) {
                        break;
                    } else if (z) {
                        throw new EOFException("source exhausted prematurely");
                    }
                } catch (DataFormatException e2) {
                    throw new IOException(e2);
                }
            }
        } else {
            throw new IllegalArgumentException(c.b.a.a.a.l("byteCount < 0: ", j));
        }
    }
}