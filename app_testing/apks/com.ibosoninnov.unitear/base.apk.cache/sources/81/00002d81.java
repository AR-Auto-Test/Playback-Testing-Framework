package g;

import java.nio.ByteBuffer;
import java.nio.charset.Charset;
import java.util.Objects;

/* compiled from: RealBufferedSink.java */
/* loaded from: classes2.dex */
public final class r implements f {

    /* renamed from: b  reason: collision with root package name */
    public final e f6202b = new e();

    /* renamed from: c  reason: collision with root package name */
    public final w f6203c;

    /* renamed from: d  reason: collision with root package name */
    public boolean f6204d;

    public r(w wVar) {
        Objects.requireNonNull(wVar, "sink == null");
        this.f6203c = wVar;
    }

    public f B() {
        int i;
        if (!this.f6204d) {
            e eVar = this.f6202b;
            long j = eVar.f6176d;
            if (j == 0) {
                j = 0;
            } else {
                t tVar = eVar.f6175c.f6215g;
                if (tVar.f6211c < 8192 && tVar.f6213e) {
                    j -= i - tVar.f6210b;
                }
            }
            if (j > 0) {
                this.f6203c.l(eVar, j);
            }
            return this;
        }
        throw new IllegalStateException("closed");
    }

    @Override // g.f
    public e a() {
        return this.f6202b;
    }

    @Override // g.w
    public y b() {
        return this.f6203c.b();
    }

    @Override // g.w, java.io.Closeable, java.lang.AutoCloseable
    public void close() {
        if (this.f6204d) {
            return;
        }
        Throwable th = null;
        try {
            e eVar = this.f6202b;
            long j = eVar.f6176d;
            if (j > 0) {
                this.f6203c.l(eVar, j);
            }
        } catch (Throwable th2) {
            th = th2;
        }
        try {
            this.f6203c.close();
        } catch (Throwable th3) {
            if (th == null) {
                th = th3;
            }
        }
        this.f6204d = true;
        if (th == null) {
            return;
        }
        Charset charset = z.f6224a;
        throw th;
    }

    @Override // g.f, g.w, java.io.Flushable
    public void flush() {
        if (!this.f6204d) {
            e eVar = this.f6202b;
            long j = eVar.f6176d;
            if (j > 0) {
                this.f6203c.l(eVar, j);
            }
            this.f6203c.flush();
            return;
        }
        throw new IllegalStateException("closed");
    }

    @Override // g.f
    public f i(String str) {
        if (!this.f6204d) {
            this.f6202b.Y(str);
            B();
            return this;
        }
        throw new IllegalStateException("closed");
    }

    @Override // java.nio.channels.Channel
    public boolean isOpen() {
        return !this.f6204d;
    }

    @Override // g.w
    public void l(e eVar, long j) {
        if (!this.f6204d) {
            this.f6202b.l(eVar, j);
            B();
            return;
        }
        throw new IllegalStateException("closed");
    }

    @Override // g.f
    public f m(long j) {
        if (!this.f6204d) {
            this.f6202b.m(j);
            B();
            return this;
        }
        throw new IllegalStateException("closed");
    }

    @Override // g.f
    public f s(h hVar) {
        if (!this.f6204d) {
            this.f6202b.P(hVar);
            B();
            return this;
        }
        throw new IllegalStateException("closed");
    }

    public String toString() {
        StringBuilder x = c.b.a.a.a.x("buffer(");
        x.append(this.f6203c);
        x.append(")");
        return x.toString();
    }

    @Override // g.f
    public f w(long j) {
        if (!this.f6204d) {
            this.f6202b.w(j);
            B();
            return this;
        }
        throw new IllegalStateException("closed");
    }

    @Override // g.f
    public f write(byte[] bArr) {
        if (!this.f6204d) {
            this.f6202b.Q(bArr);
            B();
            return this;
        }
        throw new IllegalStateException("closed");
    }

    @Override // g.f
    public f writeByte(int i) {
        if (!this.f6204d) {
            this.f6202b.T(i);
            B();
            return this;
        }
        throw new IllegalStateException("closed");
    }

    @Override // g.f
    public f writeInt(int i) {
        if (!this.f6204d) {
            this.f6202b.W(i);
            return B();
        }
        throw new IllegalStateException("closed");
    }

    @Override // g.f
    public f writeShort(int i) {
        if (!this.f6204d) {
            this.f6202b.X(i);
            B();
            return this;
        }
        throw new IllegalStateException("closed");
    }

    @Override // g.f
    public f write(byte[] bArr, int i, int i2) {
        if (!this.f6204d) {
            this.f6202b.R(bArr, i, i2);
            B();
            return this;
        }
        throw new IllegalStateException("closed");
    }

    @Override // java.nio.channels.WritableByteChannel
    public int write(ByteBuffer byteBuffer) {
        if (!this.f6204d) {
            int write = this.f6202b.write(byteBuffer);
            B();
            return write;
        }
        throw new IllegalStateException("closed");
    }
}