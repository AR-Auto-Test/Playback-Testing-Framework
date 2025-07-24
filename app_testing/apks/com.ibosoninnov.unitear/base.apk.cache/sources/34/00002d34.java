package f.g0.i;

import com.google.common.primitives.UnsignedBytes;
import f.g0.i.d;
import java.io.Closeable;
import java.io.IOException;
import java.util.List;
import java.util.Objects;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.opencv.calib3d.Calib3d;

/* compiled from: Http2Writer.java */
/* loaded from: classes2.dex */
public final class q implements Closeable {

    /* renamed from: b  reason: collision with root package name */
    public static final Logger f5989b = Logger.getLogger(e.class.getName());

    /* renamed from: c  reason: collision with root package name */
    public final g.f f5990c;

    /* renamed from: d  reason: collision with root package name */
    public final boolean f5991d;

    /* renamed from: e  reason: collision with root package name */
    public final g.e f5992e;

    /* renamed from: f  reason: collision with root package name */
    public int f5993f;

    /* renamed from: g  reason: collision with root package name */
    public boolean f5994g;

    /* renamed from: h  reason: collision with root package name */
    public final d.b f5995h;

    public q(g.f fVar, boolean z) {
        this.f5990c = fVar;
        this.f5991d = z;
        g.e eVar = new g.e();
        this.f5992e = eVar;
        this.f5995h = new d.b(eVar);
        this.f5993f = Calib3d.CALIB_RATIONAL_MODEL;
    }

    public synchronized void B(t tVar) {
        if (!this.f5994g) {
            int i = this.f5993f;
            int i2 = tVar.f6004a;
            if ((i2 & 32) != 0) {
                i = tVar.f6005b[5];
            }
            this.f5993f = i;
            int i3 = i2 & 2;
            if ((i3 != 0 ? tVar.f6005b[1] : -1) != -1) {
                d.b bVar = this.f5995h;
                int i4 = i3 != 0 ? tVar.f6005b[1] : -1;
                Objects.requireNonNull(bVar);
                int min = Math.min(i4, (int) Calib3d.CALIB_RATIONAL_MODEL);
                int i5 = bVar.f5894d;
                if (i5 != min) {
                    if (min < i5) {
                        bVar.f5892b = Math.min(bVar.f5892b, min);
                    }
                    bVar.f5893c = true;
                    bVar.f5894d = min;
                    int i6 = bVar.f5898h;
                    if (min < i6) {
                        if (min == 0) {
                            bVar.a();
                        } else {
                            bVar.b(i6 - min);
                        }
                    }
                }
            }
            D(0, 0, (byte) 4, (byte) 1);
            this.f5990c.flush();
        } else {
            throw new IOException("closed");
        }
    }

    public synchronized void C(boolean z, int i, g.e eVar, int i2) {
        if (!this.f5994g) {
            D(i, i2, (byte) 0, z ? (byte) 1 : (byte) 0);
            if (i2 > 0) {
                this.f5990c.l(eVar, i2);
            }
        } else {
            throw new IOException("closed");
        }
    }

    public void D(int i, int i2, byte b2, byte b3) {
        Logger logger = f5989b;
        if (logger.isLoggable(Level.FINE)) {
            logger.fine(e.a(false, i, i2, b2, b3));
        }
        int i3 = this.f5993f;
        if (i2 > i3) {
            e.b("FRAME_SIZE_ERROR length > %d: %d", Integer.valueOf(i3), Integer.valueOf(i2));
            throw null;
        } else if ((Integer.MIN_VALUE & i) != 0) {
            e.b("reserved bit set: %s", Integer.valueOf(i));
            throw null;
        } else {
            g.f fVar = this.f5990c;
            fVar.writeByte((i2 >>> 16) & 255);
            fVar.writeByte((i2 >>> 8) & 255);
            fVar.writeByte(i2 & 255);
            this.f5990c.writeByte(b2 & UnsignedBytes.MAX_VALUE);
            this.f5990c.writeByte(b3 & UnsignedBytes.MAX_VALUE);
            this.f5990c.writeInt(i & Integer.MAX_VALUE);
        }
    }

    public synchronized void E(int i, b bVar, byte[] bArr) {
        if (!this.f5994g) {
            if (bVar.n != -1) {
                D(0, bArr.length + 8, (byte) 7, (byte) 0);
                this.f5990c.writeInt(i);
                this.f5990c.writeInt(bVar.n);
                if (bArr.length > 0) {
                    this.f5990c.write(bArr);
                }
                this.f5990c.flush();
            } else {
                e.b("errorCode.httpCode == -1", new Object[0]);
                throw null;
            }
        } else {
            throw new IOException("closed");
        }
    }

    public void F(boolean z, int i, List<c> list) {
        if (!this.f5994g) {
            this.f5995h.e(list);
            long j = this.f5992e.f6176d;
            int min = (int) Math.min(this.f5993f, j);
            long j2 = min;
            int i2 = (j > j2 ? 1 : (j == j2 ? 0 : -1));
            byte b2 = i2 == 0 ? (byte) 4 : (byte) 0;
            if (z) {
                b2 = (byte) (b2 | 1);
            }
            D(i, min, (byte) 1, b2);
            this.f5990c.l(this.f5992e, j2);
            if (i2 > 0) {
                J(i, j - j2);
                return;
            }
            return;
        }
        throw new IOException("closed");
    }

    public synchronized void G(boolean z, int i, int i2) {
        if (!this.f5994g) {
            D(0, 8, (byte) 6, z ? (byte) 1 : (byte) 0);
            this.f5990c.writeInt(i);
            this.f5990c.writeInt(i2);
            this.f5990c.flush();
        } else {
            throw new IOException("closed");
        }
    }

    public synchronized void H(int i, b bVar) {
        if (!this.f5994g) {
            if (bVar.n != -1) {
                D(i, 4, (byte) 3, (byte) 0);
                this.f5990c.writeInt(bVar.n);
                this.f5990c.flush();
            } else {
                throw new IllegalArgumentException();
            }
        } else {
            throw new IOException("closed");
        }
    }

    public synchronized void I(int i, long j) {
        if (this.f5994g) {
            throw new IOException("closed");
        }
        if (j != 0 && j <= 2147483647L) {
            D(i, 4, (byte) 8, (byte) 0);
            this.f5990c.writeInt((int) j);
            this.f5990c.flush();
        } else {
            e.b("windowSizeIncrement == 0 || windowSizeIncrement > 0x7fffffffL: %s", Long.valueOf(j));
            throw null;
        }
    }

    public final void J(int i, long j) {
        while (j > 0) {
            int min = (int) Math.min(this.f5993f, j);
            long j2 = min;
            j -= j2;
            D(i, min, (byte) 9, j == 0 ? (byte) 4 : (byte) 0);
            this.f5990c.l(this.f5992e, j2);
        }
    }

    @Override // java.io.Closeable, java.lang.AutoCloseable
    public synchronized void close() {
        this.f5994g = true;
        this.f5990c.close();
    }

    public synchronized void flush() {
        if (!this.f5994g) {
            this.f5990c.flush();
        } else {
            throw new IOException("closed");
        }
    }
}