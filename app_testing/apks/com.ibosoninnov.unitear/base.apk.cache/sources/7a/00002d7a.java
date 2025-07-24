package g;

import java.io.EOFException;
import java.io.IOException;
import java.util.logging.Logger;
import java.util.zip.CRC32;
import java.util.zip.Inflater;

/* compiled from: GzipSource.java */
/* loaded from: classes2.dex */
public final class l implements x {

    /* renamed from: c  reason: collision with root package name */
    public final g f6187c;

    /* renamed from: d  reason: collision with root package name */
    public final Inflater f6188d;

    /* renamed from: e  reason: collision with root package name */
    public final m f6189e;

    /* renamed from: b  reason: collision with root package name */
    public int f6186b = 0;

    /* renamed from: f  reason: collision with root package name */
    public final CRC32 f6190f = new CRC32();

    public l(x xVar) {
        if (xVar != null) {
            Inflater inflater = new Inflater(true);
            this.f6188d = inflater;
            Logger logger = o.f6197a;
            s sVar = new s(xVar);
            this.f6187c = sVar;
            this.f6189e = new m(sVar, inflater);
            return;
        }
        throw new IllegalArgumentException("source == null");
    }

    public final void B(String str, int i, int i2) {
        if (i2 != i) {
            throw new IOException(String.format("%s: actual 0x%08x != expected 0x%08x", str, Integer.valueOf(i2), Integer.valueOf(i)));
        }
    }

    public final void C(e eVar, long j, long j2) {
        int i;
        t tVar = eVar.f6175c;
        while (true) {
            int i2 = tVar.f6211c;
            int i3 = tVar.f6210b;
            if (j < i2 - i3) {
                break;
            }
            j -= i2 - i3;
            tVar = tVar.f6214f;
        }
        while (j2 > 0) {
            int min = (int) Math.min(tVar.f6211c - i, j2);
            this.f6190f.update(tVar.f6209a, (int) (tVar.f6210b + j), min);
            j2 -= min;
            tVar = tVar.f6214f;
            j = 0;
        }
    }

    @Override // g.x
    public y b() {
        return this.f6187c.b();
    }

    @Override // g.x, java.io.Closeable, java.lang.AutoCloseable
    public void close() {
        this.f6189e.close();
    }

    @Override // g.x
    public long u(e eVar, long j) {
        long j2;
        int i = (j > 0L ? 1 : (j == 0L ? 0 : -1));
        if (i >= 0) {
            if (i == 0) {
                return 0L;
            }
            if (this.f6186b == 0) {
                this.f6187c.v(10L);
                byte D = this.f6187c.a().D(3L);
                boolean z = ((D >> 1) & 1) == 1;
                if (z) {
                    C(this.f6187c.a(), 0L, 10L);
                }
                B("ID1ID2", 8075, this.f6187c.readShort());
                this.f6187c.c(8L);
                if (((D >> 2) & 1) == 1) {
                    this.f6187c.v(2L);
                    if (z) {
                        C(this.f6187c.a(), 0L, 2L);
                    }
                    long t = this.f6187c.a().t();
                    this.f6187c.v(t);
                    if (z) {
                        j2 = t;
                        C(this.f6187c.a(), 0L, t);
                    } else {
                        j2 = t;
                    }
                    this.f6187c.c(j2);
                }
                if (((D >> 3) & 1) == 1) {
                    long x = this.f6187c.x((byte) 0);
                    if (x != -1) {
                        if (z) {
                            C(this.f6187c.a(), 0L, x + 1);
                        }
                        this.f6187c.c(x + 1);
                    } else {
                        throw new EOFException();
                    }
                }
                if (((D >> 4) & 1) == 1) {
                    long x2 = this.f6187c.x((byte) 0);
                    if (x2 != -1) {
                        if (z) {
                            C(this.f6187c.a(), 0L, x2 + 1);
                        }
                        this.f6187c.c(x2 + 1);
                    } else {
                        throw new EOFException();
                    }
                }
                if (z) {
                    B("FHCRC", this.f6187c.t(), (short) this.f6190f.getValue());
                    this.f6190f.reset();
                }
                this.f6186b = 1;
            }
            if (this.f6186b == 1) {
                long j3 = eVar.f6176d;
                long u = this.f6189e.u(eVar, j);
                if (u != -1) {
                    C(eVar, j3, u);
                    return u;
                }
                this.f6186b = 2;
            }
            if (this.f6186b == 2) {
                B("CRC", this.f6187c.q(), (int) this.f6190f.getValue());
                B("ISIZE", this.f6187c.q(), (int) this.f6188d.getBytesWritten());
                this.f6186b = 3;
                if (!this.f6187c.f()) {
                    throw new IOException("gzip finished without exhausting source");
                }
            }
            return -1L;
        }
        throw new IllegalArgumentException(c.b.a.a.a.l("byteCount < 0: ", j));
    }
}