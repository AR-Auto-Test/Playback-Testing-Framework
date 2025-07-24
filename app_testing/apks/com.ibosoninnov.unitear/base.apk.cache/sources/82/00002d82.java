package g;

import androidx.recyclerview.widget.RecyclerView;
import com.google.common.primitives.UnsignedBytes;
import java.io.EOFException;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.charset.Charset;
import java.util.Objects;

/* compiled from: RealBufferedSource.java */
/* loaded from: classes2.dex */
public final class s implements g {

    /* renamed from: b  reason: collision with root package name */
    public final e f6205b = new e();

    /* renamed from: c  reason: collision with root package name */
    public final x f6206c;

    /* renamed from: d  reason: collision with root package name */
    public boolean f6207d;

    public s(x xVar) {
        Objects.requireNonNull(xVar, "source == null");
        this.f6206c = xVar;
    }

    @Override // g.g
    public int A(q qVar) {
        if (!this.f6207d) {
            do {
                int N = this.f6205b.N(qVar, true);
                if (N == -1) {
                    return -1;
                }
                if (N != -2) {
                    this.f6205b.c(qVar.f6200b[N].l());
                    return N;
                }
            } while (this.f6206c.u(this.f6205b, 8192L) != -1);
            return -1;
        }
        throw new IllegalStateException("closed");
    }

    public long B(byte b2, long j, long j2) {
        if (this.f6207d) {
            throw new IllegalStateException("closed");
        }
        if (j < 0 || j2 < j) {
            throw new IllegalArgumentException(String.format("fromIndex=%s toIndex=%s", Long.valueOf(j), Long.valueOf(j2)));
        }
        while (j < j2) {
            long E = this.f6205b.E(b2, j, j2);
            if (E == -1) {
                e eVar = this.f6205b;
                long j3 = eVar.f6176d;
                if (j3 >= j2 || this.f6206c.u(eVar, 8192L) == -1) {
                    break;
                }
                j = Math.max(j, j3);
            } else {
                return E;
            }
        }
        return -1L;
    }

    public void C(byte[] bArr) {
        try {
            v(bArr.length);
            this.f6205b.I(bArr);
        } catch (EOFException e2) {
            int i = 0;
            while (true) {
                e eVar = this.f6205b;
                long j = eVar.f6176d;
                if (j > 0) {
                    int G = eVar.G(bArr, i, (int) j);
                    if (G == -1) {
                        throw new AssertionError();
                    }
                    i += G;
                } else {
                    throw e2;
                }
            }
        }
    }

    @Override // g.g, g.f
    public e a() {
        return this.f6205b;
    }

    @Override // g.x
    public y b() {
        return this.f6206c.b();
    }

    @Override // g.g
    public void c(long j) {
        if (this.f6207d) {
            throw new IllegalStateException("closed");
        }
        while (j > 0) {
            e eVar = this.f6205b;
            if (eVar.f6176d == 0 && this.f6206c.u(eVar, 8192L) == -1) {
                throw new EOFException();
            }
            long min = Math.min(j, this.f6205b.f6176d);
            this.f6205b.c(min);
            j -= min;
        }
    }

    @Override // g.x, java.io.Closeable, java.lang.AutoCloseable
    public void close() {
        if (this.f6207d) {
            return;
        }
        this.f6207d = true;
        this.f6206c.close();
        this.f6205b.B();
    }

    @Override // g.g
    public h d(long j) {
        if (o(j)) {
            return this.f6205b.d(j);
        }
        throw new EOFException();
    }

    @Override // g.g
    public e e() {
        return this.f6205b;
    }

    @Override // g.g
    public boolean f() {
        if (this.f6207d) {
            throw new IllegalStateException("closed");
        }
        return this.f6205b.f() && this.f6206c.u(this.f6205b, 8192L) == -1;
    }

    @Override // g.g
    public long g(h hVar) {
        if (this.f6207d) {
            throw new IllegalStateException("closed");
        }
        long j = 0;
        while (true) {
            long F = this.f6205b.F(hVar, j);
            if (F != -1) {
                return F;
            }
            e eVar = this.f6205b;
            long j2 = eVar.f6176d;
            if (this.f6206c.u(eVar, 8192L) == -1) {
                return -1L;
            }
            j = Math.max(j, j2);
        }
    }

    @Override // g.g
    public String h(long j) {
        if (j >= 0) {
            long j2 = j == RecyclerView.FOREVER_NS ? Long.MAX_VALUE : j + 1;
            long B = B((byte) 10, 0L, j2);
            if (B != -1) {
                return this.f6205b.M(B);
            }
            if (j2 < RecyclerView.FOREVER_NS && o(j2) && this.f6205b.D(j2 - 1) == 13 && o(1 + j2) && this.f6205b.D(j2) == 10) {
                return this.f6205b.M(j2);
            }
            e eVar = new e();
            e eVar2 = this.f6205b;
            eVar2.C(eVar, 0L, Math.min(32L, eVar2.f6176d));
            StringBuilder x = c.b.a.a.a.x("\\n not found: limit=");
            x.append(Math.min(this.f6205b.f6176d, j));
            x.append(" content=");
            x.append(eVar.H().g());
            x.append((char) 8230);
            throw new EOFException(x.toString());
        }
        throw new IllegalArgumentException(c.b.a.a.a.l("limit < 0: ", j));
    }

    @Override // java.nio.channels.Channel
    public boolean isOpen() {
        return !this.f6207d;
    }

    @Override // g.g
    public boolean j(long j, h hVar) {
        int l = hVar.l();
        if (!this.f6207d) {
            if (j < 0 || l < 0 || hVar.l() - 0 < l) {
                return false;
            }
            for (int i = 0; i < l; i++) {
                long j2 = i + j;
                if (!o(1 + j2) || this.f6205b.D(j2) != hVar.f(0 + i)) {
                    return false;
                }
            }
            return true;
        }
        throw new IllegalStateException("closed");
    }

    @Override // g.g
    public String k(Charset charset) {
        if (charset != null) {
            this.f6205b.S(this.f6206c);
            e eVar = this.f6205b;
            Objects.requireNonNull(eVar);
            try {
                return eVar.J(eVar.f6176d, charset);
            } catch (EOFException e2) {
                throw new AssertionError(e2);
            }
        }
        throw new IllegalArgumentException("charset == null");
    }

    @Override // g.g
    public boolean o(long j) {
        e eVar;
        if (j >= 0) {
            if (!this.f6207d) {
                do {
                    eVar = this.f6205b;
                    if (eVar.f6176d >= j) {
                        return true;
                    }
                } while (this.f6206c.u(eVar, 8192L) != -1);
                return false;
            }
            throw new IllegalStateException("closed");
        }
        throw new IllegalArgumentException(c.b.a.a.a.l("byteCount < 0: ", j));
    }

    @Override // g.g
    public String p() {
        return h(RecyclerView.FOREVER_NS);
    }

    @Override // g.g
    public int q() {
        v(4L);
        return this.f6205b.q();
    }

    @Override // g.g
    public byte[] r(long j) {
        if (o(j)) {
            return this.f6205b.r(j);
        }
        throw new EOFException();
    }

    @Override // java.nio.channels.ReadableByteChannel
    public int read(ByteBuffer byteBuffer) {
        e eVar = this.f6205b;
        if (eVar.f6176d == 0 && this.f6206c.u(eVar, 8192L) == -1) {
            return -1;
        }
        return this.f6205b.read(byteBuffer);
    }

    @Override // g.g
    public byte readByte() {
        v(1L);
        return this.f6205b.readByte();
    }

    @Override // g.g
    public int readInt() {
        v(4L);
        return this.f6205b.readInt();
    }

    @Override // g.g
    public short readShort() {
        v(2L);
        return this.f6205b.readShort();
    }

    @Override // g.g
    public short t() {
        v(2L);
        return this.f6205b.t();
    }

    public String toString() {
        StringBuilder x = c.b.a.a.a.x("buffer(");
        x.append(this.f6206c);
        x.append(")");
        return x.toString();
    }

    @Override // g.x
    public long u(e eVar, long j) {
        if (eVar != null) {
            if (j >= 0) {
                if (!this.f6207d) {
                    e eVar2 = this.f6205b;
                    if (eVar2.f6176d == 0 && this.f6206c.u(eVar2, 8192L) == -1) {
                        return -1L;
                    }
                    return this.f6205b.u(eVar, Math.min(j, this.f6205b.f6176d));
                }
                throw new IllegalStateException("closed");
            }
            throw new IllegalArgumentException(c.b.a.a.a.l("byteCount < 0: ", j));
        }
        throw new IllegalArgumentException("sink == null");
    }

    @Override // g.g
    public void v(long j) {
        if (!o(j)) {
            throw new EOFException();
        }
    }

    @Override // g.g
    public long x(byte b2) {
        return B(b2, 0L, RecyclerView.FOREVER_NS);
    }

    /* JADX WARN: Code restructure failed: missing block: B:19:0x0032, code lost:
        if (r1 == 0) goto L21;
     */
    /* JADX WARN: Code restructure failed: missing block: B:22:0x0049, code lost:
        throw new java.lang.NumberFormatException(java.lang.String.format("Expected leading [0-9a-fA-F] character but was %#x", java.lang.Byte.valueOf(r3)));
     */
    @Override // g.g
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public long y() {
        v(1L);
        int i = 0;
        while (true) {
            int i2 = i + 1;
            if (!o(i2)) {
                break;
            }
            byte D = this.f6205b.D(i);
            if ((D < 48 || D > 57) && ((D < 97 || D > 102) && (D < 65 || D > 70))) {
                break;
            }
            i = i2;
        }
        return this.f6205b.y();
    }

    @Override // g.g
    public InputStream z() {
        return new a();
    }

    /* compiled from: RealBufferedSource.java */
    /* loaded from: classes2.dex */
    public class a extends InputStream {
        public a() {
        }

        @Override // java.io.InputStream
        public int available() {
            s sVar = s.this;
            if (!sVar.f6207d) {
                return (int) Math.min(sVar.f6205b.f6176d, 2147483647L);
            }
            throw new IOException("closed");
        }

        @Override // java.io.InputStream, java.io.Closeable, java.lang.AutoCloseable
        public void close() {
            s.this.close();
        }

        @Override // java.io.InputStream
        public int read() {
            s sVar = s.this;
            if (!sVar.f6207d) {
                e eVar = sVar.f6205b;
                if (eVar.f6176d == 0 && sVar.f6206c.u(eVar, 8192L) == -1) {
                    return -1;
                }
                return s.this.f6205b.readByte() & UnsignedBytes.MAX_VALUE;
            }
            throw new IOException("closed");
        }

        public String toString() {
            return s.this + ".inputStream()";
        }

        @Override // java.io.InputStream
        public int read(byte[] bArr, int i, int i2) {
            if (!s.this.f6207d) {
                z.b(bArr.length, i, i2);
                s sVar = s.this;
                e eVar = sVar.f6205b;
                if (eVar.f6176d == 0 && sVar.f6206c.u(eVar, 8192L) == -1) {
                    return -1;
                }
                return s.this.f6205b.G(bArr, i, i2);
            }
            throw new IOException("closed");
        }
    }
}