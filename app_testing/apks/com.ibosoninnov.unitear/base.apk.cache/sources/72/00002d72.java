package g;

import androidx.recyclerview.widget.RecyclerView;
import com.google.common.primitives.UnsignedBytes;
import java.io.EOFException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.channels.ByteChannel;
import java.nio.charset.Charset;
import java.util.Objects;

/* compiled from: Buffer.java */
/* loaded from: classes2.dex */
public final class e implements g, f, Cloneable, ByteChannel {

    /* renamed from: b  reason: collision with root package name */
    public static final byte[] f6174b = {48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 97, 98, 99, 100, 101, 102};

    /* renamed from: c  reason: collision with root package name */
    public t f6175c;

    /* renamed from: d  reason: collision with root package name */
    public long f6176d;

    /* compiled from: Buffer.java */
    /* loaded from: classes2.dex */
    public class a extends InputStream {
        public a() {
        }

        @Override // java.io.InputStream
        public int available() {
            return (int) Math.min(e.this.f6176d, 2147483647L);
        }

        @Override // java.io.InputStream, java.io.Closeable, java.lang.AutoCloseable
        public void close() {
        }

        @Override // java.io.InputStream
        public int read() {
            e eVar = e.this;
            if (eVar.f6176d > 0) {
                return eVar.readByte() & UnsignedBytes.MAX_VALUE;
            }
            return -1;
        }

        public String toString() {
            return e.this + ".inputStream()";
        }

        @Override // java.io.InputStream
        public int read(byte[] bArr, int i, int i2) {
            return e.this.G(bArr, i, i2);
        }
    }

    @Override // g.g
    public int A(q qVar) {
        int N = N(qVar, false);
        if (N == -1) {
            return -1;
        }
        try {
            c(qVar.f6200b[N].l());
            return N;
        } catch (EOFException unused) {
            throw new AssertionError();
        }
    }

    public final void B() {
        try {
            c(this.f6176d);
        } catch (EOFException e2) {
            throw new AssertionError(e2);
        }
    }

    public final e C(e eVar, long j, long j2) {
        if (eVar != null) {
            z.b(this.f6176d, j, j2);
            if (j2 == 0) {
                return this;
            }
            eVar.f6176d += j2;
            t tVar = this.f6175c;
            while (true) {
                int i = tVar.f6211c;
                int i2 = tVar.f6210b;
                if (j < i - i2) {
                    break;
                }
                j -= i - i2;
                tVar = tVar.f6214f;
            }
            while (j2 > 0) {
                t c2 = tVar.c();
                int i3 = (int) (c2.f6210b + j);
                c2.f6210b = i3;
                c2.f6211c = Math.min(i3 + ((int) j2), c2.f6211c);
                t tVar2 = eVar.f6175c;
                if (tVar2 == null) {
                    c2.f6215g = c2;
                    c2.f6214f = c2;
                    eVar.f6175c = c2;
                } else {
                    tVar2.f6215g.b(c2);
                }
                j2 -= c2.f6211c - c2.f6210b;
                tVar = tVar.f6214f;
                j = 0;
            }
            return this;
        }
        throw new IllegalArgumentException("out == null");
    }

    public final byte D(long j) {
        int i;
        z.b(this.f6176d, j, 1L);
        long j2 = this.f6176d;
        if (j2 - j > j) {
            t tVar = this.f6175c;
            while (true) {
                int i2 = tVar.f6211c;
                int i3 = tVar.f6210b;
                long j3 = i2 - i3;
                if (j < j3) {
                    return tVar.f6209a[i3 + ((int) j)];
                }
                j -= j3;
                tVar = tVar.f6214f;
            }
        } else {
            long j4 = j - j2;
            t tVar2 = this.f6175c;
            do {
                tVar2 = tVar2.f6215g;
                int i4 = tVar2.f6211c;
                i = tVar2.f6210b;
                j4 += i4 - i;
            } while (j4 < 0);
            return tVar2.f6209a[i + ((int) j4)];
        }
    }

    public long E(byte b2, long j, long j2) {
        t tVar;
        long j3 = 0;
        if (j >= 0 && j2 >= j) {
            long j4 = this.f6176d;
            long j5 = j2 > j4 ? j4 : j2;
            if (j == j5 || (tVar = this.f6175c) == null) {
                return -1L;
            }
            if (j4 - j < j) {
                while (j4 > j) {
                    tVar = tVar.f6215g;
                    j4 -= tVar.f6211c - tVar.f6210b;
                }
            } else {
                while (true) {
                    long j6 = (tVar.f6211c - tVar.f6210b) + j3;
                    if (j6 >= j) {
                        break;
                    }
                    tVar = tVar.f6214f;
                    j3 = j6;
                }
                j4 = j3;
            }
            long j7 = j;
            while (j4 < j5) {
                byte[] bArr = tVar.f6209a;
                int min = (int) Math.min(tVar.f6211c, (tVar.f6210b + j5) - j4);
                for (int i = (int) ((tVar.f6210b + j7) - j4); i < min; i++) {
                    if (bArr[i] == b2) {
                        return (i - tVar.f6210b) + j4;
                    }
                }
                j4 += tVar.f6211c - tVar.f6210b;
                tVar = tVar.f6214f;
                j7 = j4;
            }
            return -1L;
        }
        throw new IllegalArgumentException(String.format("size=%s fromIndex=%s toIndex=%s", Long.valueOf(this.f6176d), Long.valueOf(j), Long.valueOf(j2)));
    }

    public long F(h hVar, long j) {
        int i;
        int i2;
        long j2 = 0;
        if (j >= 0) {
            t tVar = this.f6175c;
            if (tVar == null) {
                return -1L;
            }
            long j3 = this.f6176d;
            if (j3 - j < j) {
                while (j3 > j) {
                    tVar = tVar.f6215g;
                    j3 -= tVar.f6211c - tVar.f6210b;
                }
            } else {
                while (true) {
                    long j4 = (tVar.f6211c - tVar.f6210b) + j2;
                    if (j4 >= j) {
                        break;
                    }
                    tVar = tVar.f6214f;
                    j2 = j4;
                }
                j3 = j2;
            }
            if (hVar.l() == 2) {
                byte f2 = hVar.f(0);
                byte f3 = hVar.f(1);
                while (j3 < this.f6176d) {
                    byte[] bArr = tVar.f6209a;
                    i = (int) ((tVar.f6210b + j) - j3);
                    int i3 = tVar.f6211c;
                    while (i < i3) {
                        byte b2 = bArr[i];
                        if (b2 == f2 || b2 == f3) {
                            i2 = tVar.f6210b;
                            return (i - i2) + j3;
                        }
                        i++;
                    }
                    j3 += tVar.f6211c - tVar.f6210b;
                    tVar = tVar.f6214f;
                    j = j3;
                }
                return -1L;
            }
            byte[] h2 = hVar.h();
            while (j3 < this.f6176d) {
                byte[] bArr2 = tVar.f6209a;
                i = (int) ((tVar.f6210b + j) - j3);
                int i4 = tVar.f6211c;
                while (i < i4) {
                    byte b3 = bArr2[i];
                    for (byte b4 : h2) {
                        if (b3 == b4) {
                            i2 = tVar.f6210b;
                            return (i - i2) + j3;
                        }
                    }
                    i++;
                }
                j3 += tVar.f6211c - tVar.f6210b;
                tVar = tVar.f6214f;
                j = j3;
            }
            return -1L;
        }
        throw new IllegalArgumentException("fromIndex < 0");
    }

    public int G(byte[] bArr, int i, int i2) {
        z.b(bArr.length, i, i2);
        t tVar = this.f6175c;
        if (tVar == null) {
            return -1;
        }
        int min = Math.min(i2, tVar.f6211c - tVar.f6210b);
        System.arraycopy(tVar.f6209a, tVar.f6210b, bArr, i, min);
        int i3 = tVar.f6210b + min;
        tVar.f6210b = i3;
        this.f6176d -= min;
        if (i3 == tVar.f6211c) {
            this.f6175c = tVar.a();
            u.a(tVar);
        }
        return min;
    }

    public h H() {
        try {
            return new h(r(this.f6176d));
        } catch (EOFException e2) {
            throw new AssertionError(e2);
        }
    }

    public void I(byte[] bArr) {
        int i = 0;
        while (i < bArr.length) {
            int G = G(bArr, i, bArr.length - i);
            if (G == -1) {
                throw new EOFException();
            }
            i += G;
        }
    }

    public String J(long j, Charset charset) {
        z.b(this.f6176d, 0L, j);
        if (charset != null) {
            if (j <= 2147483647L) {
                if (j == 0) {
                    return "";
                }
                t tVar = this.f6175c;
                if (tVar.f6210b + j > tVar.f6211c) {
                    return new String(r(j), charset);
                }
                String str = new String(tVar.f6209a, tVar.f6210b, (int) j, charset);
                int i = (int) (tVar.f6210b + j);
                tVar.f6210b = i;
                this.f6176d -= j;
                if (i == tVar.f6211c) {
                    this.f6175c = tVar.a();
                    u.a(tVar);
                }
                return str;
            }
            throw new IllegalArgumentException(c.b.a.a.a.l("byteCount > Integer.MAX_VALUE: ", j));
        }
        throw new IllegalArgumentException("charset == null");
    }

    public String K() {
        try {
            return J(this.f6176d, z.f6224a);
        } catch (EOFException e2) {
            throw new AssertionError(e2);
        }
    }

    public String L(long j) {
        return J(j, z.f6224a);
    }

    public String M(long j) {
        if (j > 0) {
            long j2 = j - 1;
            if (D(j2) == 13) {
                String L = L(j2);
                c(2L);
                return L;
            }
        }
        String L2 = L(j);
        c(1L);
        return L2;
    }

    /* JADX WARN: Code restructure failed: missing block: B:27:0x0055, code lost:
        if (r19 == false) goto L36;
     */
    /* JADX WARN: Code restructure failed: missing block: B:28:0x0057, code lost:
        return r3;
     */
    /* JADX WARN: Code restructure failed: missing block: B:29:0x0058, code lost:
        return r11;
     */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public int N(q qVar, boolean z) {
        int i;
        int i2;
        int i3;
        int i4;
        t tVar;
        t tVar2 = this.f6175c;
        int i5 = -2;
        if (tVar2 != null) {
            byte[] bArr = tVar2.f6209a;
            int i6 = tVar2.f6210b;
            int i7 = tVar2.f6211c;
            int[] iArr = qVar.f6201c;
            t tVar3 = tVar2;
            int i8 = 0;
            int i9 = -1;
            loop0: while (true) {
                int i10 = i8 + 1;
                int i11 = iArr[i8];
                int i12 = i10 + 1;
                int i13 = iArr[i10];
                if (i13 != -1) {
                    i9 = i13;
                }
                if (tVar3 == null) {
                    break;
                }
                if (i11 >= 0) {
                    int i14 = i6 + 1;
                    int i15 = bArr[i6] & UnsignedBytes.MAX_VALUE;
                    int i16 = i12 + i11;
                    while (i12 != i16) {
                        if (i15 == iArr[i12]) {
                            i = iArr[i12 + i11];
                            if (i14 == i7) {
                                tVar3 = tVar3.f6214f;
                                i2 = tVar3.f6210b;
                                bArr = tVar3.f6209a;
                                i7 = tVar3.f6211c;
                                if (tVar3 == tVar2) {
                                    tVar3 = null;
                                }
                            } else {
                                i2 = i14;
                            }
                        } else {
                            i12++;
                        }
                    }
                    return i9;
                }
                int i17 = (i11 * (-1)) + i12;
                while (true) {
                    int i18 = i6 + 1;
                    int i19 = i12 + 1;
                    if ((bArr[i6] & UnsignedBytes.MAX_VALUE) != iArr[i12]) {
                        return i9;
                    }
                    boolean z2 = i19 == i17;
                    if (i18 == i7) {
                        t tVar4 = tVar3.f6214f;
                        i4 = tVar4.f6210b;
                        byte[] bArr2 = tVar4.f6209a;
                        i3 = tVar4.f6211c;
                        if (tVar4 != tVar2) {
                            tVar = tVar4;
                            bArr = bArr2;
                        } else if (!z2) {
                            break loop0;
                        } else {
                            bArr = bArr2;
                            tVar = null;
                        }
                    } else {
                        t tVar5 = tVar3;
                        i3 = i7;
                        i4 = i18;
                        tVar = tVar5;
                    }
                    if (z2) {
                        i = iArr[i19];
                        i2 = i4;
                        i7 = i3;
                        tVar3 = tVar;
                        break;
                    }
                    i6 = i4;
                    i7 = i3;
                    i12 = i19;
                    tVar3 = tVar;
                }
                if (i >= 0) {
                    return i;
                }
                i8 = -i;
                i6 = i2;
                i5 = -2;
            }
        } else if (z) {
            return -2;
        } else {
            return qVar.indexOf(h.f6179c);
        }
    }

    public t O(int i) {
        if (i >= 1 && i <= 8192) {
            t tVar = this.f6175c;
            if (tVar == null) {
                t b2 = u.b();
                this.f6175c = b2;
                b2.f6215g = b2;
                b2.f6214f = b2;
                return b2;
            }
            t tVar2 = tVar.f6215g;
            if (tVar2.f6211c + i > 8192 || !tVar2.f6213e) {
                t b3 = u.b();
                tVar2.b(b3);
                return b3;
            }
            return tVar2;
        }
        throw new IllegalArgumentException();
    }

    public e P(h hVar) {
        if (hVar != null) {
            hVar.q(this);
            return this;
        }
        throw new IllegalArgumentException("byteString == null");
    }

    public e Q(byte[] bArr) {
        if (bArr != null) {
            R(bArr, 0, bArr.length);
            return this;
        }
        throw new IllegalArgumentException("source == null");
    }

    public e R(byte[] bArr, int i, int i2) {
        if (bArr != null) {
            long j = i2;
            z.b(bArr.length, i, j);
            int i3 = i2 + i;
            while (i < i3) {
                t O = O(1);
                int min = Math.min(i3 - i, 8192 - O.f6211c);
                System.arraycopy(bArr, i, O.f6209a, O.f6211c, min);
                i += min;
                O.f6211c += min;
            }
            this.f6176d += j;
            return this;
        }
        throw new IllegalArgumentException("source == null");
    }

    public long S(x xVar) {
        if (xVar == null) {
            throw new IllegalArgumentException("source == null");
        }
        long j = 0;
        while (true) {
            long u = xVar.u(this, 8192L);
            if (u == -1) {
                return j;
            }
            j += u;
        }
    }

    public e T(int i) {
        t O = O(1);
        byte[] bArr = O.f6209a;
        int i2 = O.f6211c;
        O.f6211c = i2 + 1;
        bArr[i2] = (byte) i;
        this.f6176d++;
        return this;
    }

    /* JADX DEBUG: Method merged with bridge method */
    @Override // g.f
    /* renamed from: U */
    public e w(long j) {
        int i = (j > 0L ? 1 : (j == 0L ? 0 : -1));
        if (i == 0) {
            T(48);
            return this;
        }
        boolean z = false;
        int i2 = 1;
        if (i < 0) {
            j = -j;
            if (j < 0) {
                Y("-9223372036854775808");
                return this;
            }
            z = true;
        }
        if (j >= 100000000) {
            i2 = j < 1000000000000L ? j < 10000000000L ? j < 1000000000 ? 9 : 10 : j < 100000000000L ? 11 : 12 : j < 1000000000000000L ? j < 10000000000000L ? 13 : j < 100000000000000L ? 14 : 15 : j < 100000000000000000L ? j < 10000000000000000L ? 16 : 17 : j < 1000000000000000000L ? 18 : 19;
        } else if (j >= 10000) {
            i2 = j < 1000000 ? j < 100000 ? 5 : 6 : j < 10000000 ? 7 : 8;
        } else if (j >= 100) {
            i2 = j < 1000 ? 3 : 4;
        } else if (j >= 10) {
            i2 = 2;
        }
        if (z) {
            i2++;
        }
        t O = O(i2);
        byte[] bArr = O.f6209a;
        int i3 = O.f6211c + i2;
        while (j != 0) {
            i3--;
            bArr[i3] = f6174b[(int) (j % 10)];
            j /= 10;
        }
        if (z) {
            bArr[i3 - 1] = 45;
        }
        O.f6211c += i2;
        this.f6176d += i2;
        return this;
    }

    /* JADX DEBUG: Method merged with bridge method */
    @Override // g.f
    /* renamed from: V */
    public e m(long j) {
        if (j == 0) {
            T(48);
            return this;
        }
        int numberOfTrailingZeros = (Long.numberOfTrailingZeros(Long.highestOneBit(j)) / 4) + 1;
        t O = O(numberOfTrailingZeros);
        byte[] bArr = O.f6209a;
        int i = O.f6211c;
        for (int i2 = (i + numberOfTrailingZeros) - 1; i2 >= i; i2--) {
            bArr[i2] = f6174b[(int) (15 & j)];
            j >>>= 4;
        }
        O.f6211c += numberOfTrailingZeros;
        this.f6176d += numberOfTrailingZeros;
        return this;
    }

    public e W(int i) {
        t O = O(4);
        byte[] bArr = O.f6209a;
        int i2 = O.f6211c;
        int i3 = i2 + 1;
        bArr[i2] = (byte) ((i >>> 24) & 255);
        int i4 = i3 + 1;
        bArr[i3] = (byte) ((i >>> 16) & 255);
        int i5 = i4 + 1;
        bArr[i4] = (byte) ((i >>> 8) & 255);
        bArr[i5] = (byte) (i & 255);
        O.f6211c = i5 + 1;
        this.f6176d += 4;
        return this;
    }

    public e X(int i) {
        t O = O(2);
        byte[] bArr = O.f6209a;
        int i2 = O.f6211c;
        int i3 = i2 + 1;
        bArr[i2] = (byte) ((i >>> 8) & 255);
        bArr[i3] = (byte) (i & 255);
        O.f6211c = i3 + 1;
        this.f6176d += 2;
        return this;
    }

    public e Y(String str) {
        Z(str, 0, str.length());
        return this;
    }

    public e Z(String str, int i, int i2) {
        char charAt;
        if (str != null) {
            if (i >= 0) {
                if (i2 >= i) {
                    if (i2 > str.length()) {
                        StringBuilder y = c.b.a.a.a.y("endIndex > string.length: ", i2, " > ");
                        y.append(str.length());
                        throw new IllegalArgumentException(y.toString());
                    }
                    while (i < i2) {
                        char charAt2 = str.charAt(i);
                        if (charAt2 < 128) {
                            t O = O(1);
                            byte[] bArr = O.f6209a;
                            int i3 = O.f6211c - i;
                            int min = Math.min(i2, 8192 - i3);
                            int i4 = i + 1;
                            bArr[i + i3] = (byte) charAt2;
                            while (true) {
                                i = i4;
                                if (i >= min || (charAt = str.charAt(i)) >= 128) {
                                    break;
                                }
                                i4 = i + 1;
                                bArr[i + i3] = (byte) charAt;
                            }
                            int i5 = O.f6211c;
                            int i6 = (i3 + i) - i5;
                            O.f6211c = i5 + i6;
                            this.f6176d += i6;
                        } else {
                            if (charAt2 < 2048) {
                                T((charAt2 >> 6) | 192);
                                T((charAt2 & '?') | 128);
                            } else if (charAt2 >= 55296 && charAt2 <= 57343) {
                                int i7 = i + 1;
                                char charAt3 = i7 < i2 ? str.charAt(i7) : (char) 0;
                                if (charAt2 <= 56319 && charAt3 >= 56320 && charAt3 <= 57343) {
                                    int i8 = (((charAt2 & 10239) << 10) | (9215 & charAt3)) + 65536;
                                    T((i8 >> 18) | 240);
                                    T(((i8 >> 12) & 63) | 128);
                                    T(((i8 >> 6) & 63) | 128);
                                    T((i8 & 63) | 128);
                                    i += 2;
                                } else {
                                    T(63);
                                    i = i7;
                                }
                            } else {
                                T((charAt2 >> '\f') | 224);
                                T(((charAt2 >> 6) & 63) | 128);
                                T((charAt2 & '?') | 128);
                            }
                            i++;
                        }
                    }
                    return this;
                }
                throw new IllegalArgumentException(c.b.a.a.a.k("endIndex < beginIndex: ", i2, " < ", i));
            }
            throw new IllegalArgumentException(c.b.a.a.a.j("beginIndex < 0: ", i));
        }
        throw new IllegalArgumentException("string == null");
    }

    @Override // g.g, g.f
    public e a() {
        return this;
    }

    public e a0(int i) {
        if (i < 128) {
            T(i);
        } else if (i < 2048) {
            T((i >> 6) | 192);
            T((i & 63) | 128);
        } else if (i < 65536) {
            if (i >= 55296 && i <= 57343) {
                T(63);
            } else {
                T((i >> 12) | 224);
                T(((i >> 6) & 63) | 128);
                T((i & 63) | 128);
            }
        } else if (i <= 1114111) {
            T((i >> 18) | 240);
            T(((i >> 12) & 63) | 128);
            T(((i >> 6) & 63) | 128);
            T((i & 63) | 128);
        } else {
            StringBuilder x = c.b.a.a.a.x("Unexpected code point: ");
            x.append(Integer.toHexString(i));
            throw new IllegalArgumentException(x.toString());
        }
        return this;
    }

    @Override // g.x
    public y b() {
        return y.f6220a;
    }

    @Override // g.g
    public void c(long j) {
        t tVar;
        while (j > 0) {
            if (this.f6175c != null) {
                int min = (int) Math.min(j, tVar.f6211c - tVar.f6210b);
                long j2 = min;
                this.f6176d -= j2;
                j -= j2;
                t tVar2 = this.f6175c;
                int i = tVar2.f6210b + min;
                tVar2.f6210b = i;
                if (i == tVar2.f6211c) {
                    this.f6175c = tVar2.a();
                    u.a(tVar2);
                }
            } else {
                throw new EOFException();
            }
        }
    }

    public Object clone() {
        e eVar = new e();
        if (this.f6176d != 0) {
            t c2 = this.f6175c.c();
            eVar.f6175c = c2;
            c2.f6215g = c2;
            c2.f6214f = c2;
            t tVar = this.f6175c;
            while (true) {
                tVar = tVar.f6214f;
                if (tVar == this.f6175c) {
                    break;
                }
                eVar.f6175c.f6215g.b(tVar.c());
            }
            eVar.f6176d = this.f6176d;
        }
        return eVar;
    }

    @Override // g.x, java.io.Closeable, java.lang.AutoCloseable
    public void close() {
    }

    @Override // g.g
    public h d(long j) {
        return new h(r(j));
    }

    @Override // g.g
    public e e() {
        return this;
    }

    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }
        if (obj instanceof e) {
            e eVar = (e) obj;
            long j = this.f6176d;
            if (j != eVar.f6176d) {
                return false;
            }
            long j2 = 0;
            if (j == 0) {
                return true;
            }
            t tVar = this.f6175c;
            t tVar2 = eVar.f6175c;
            int i = tVar.f6210b;
            int i2 = tVar2.f6210b;
            while (j2 < this.f6176d) {
                long min = Math.min(tVar.f6211c - i, tVar2.f6211c - i2);
                int i3 = 0;
                while (i3 < min) {
                    int i4 = i + 1;
                    int i5 = i2 + 1;
                    if (tVar.f6209a[i] != tVar2.f6209a[i2]) {
                        return false;
                    }
                    i3++;
                    i = i4;
                    i2 = i5;
                }
                if (i == tVar.f6211c) {
                    tVar = tVar.f6214f;
                    i = tVar.f6210b;
                }
                if (i2 == tVar2.f6211c) {
                    tVar2 = tVar2.f6214f;
                    i2 = tVar2.f6210b;
                }
                j2 += min;
            }
            return true;
        }
        return false;
    }

    @Override // g.g
    public boolean f() {
        return this.f6176d == 0;
    }

    @Override // g.f, g.w, java.io.Flushable
    public void flush() {
    }

    @Override // g.g
    public long g(h hVar) {
        return F(hVar, 0L);
    }

    @Override // g.g
    public String h(long j) {
        if (j >= 0) {
            long j2 = RecyclerView.FOREVER_NS;
            if (j != RecyclerView.FOREVER_NS) {
                j2 = j + 1;
            }
            long E = E((byte) 10, 0L, j2);
            if (E != -1) {
                return M(E);
            }
            if (j2 < this.f6176d && D(j2 - 1) == 13 && D(j2) == 10) {
                return M(j2);
            }
            e eVar = new e();
            C(eVar, 0L, Math.min(32L, this.f6176d));
            StringBuilder x = c.b.a.a.a.x("\\n not found: limit=");
            x.append(Math.min(this.f6176d, j));
            x.append(" content=");
            x.append(eVar.H().g());
            x.append((char) 8230);
            throw new EOFException(x.toString());
        }
        throw new IllegalArgumentException(c.b.a.a.a.l("limit < 0: ", j));
    }

    public int hashCode() {
        t tVar = this.f6175c;
        if (tVar == null) {
            return 0;
        }
        int i = 1;
        do {
            int i2 = tVar.f6211c;
            for (int i3 = tVar.f6210b; i3 < i2; i3++) {
                i = (i * 31) + tVar.f6209a[i3];
            }
            tVar = tVar.f6214f;
        } while (tVar != this.f6175c);
        return i;
    }

    @Override // g.f
    public /* bridge */ /* synthetic */ f i(String str) {
        Y(str);
        return this;
    }

    @Override // java.nio.channels.Channel
    public boolean isOpen() {
        return true;
    }

    @Override // g.g
    public boolean j(long j, h hVar) {
        int l = hVar.l();
        if (j < 0 || l < 0 || this.f6176d - j < l || hVar.l() - 0 < l) {
            return false;
        }
        for (int i = 0; i < l; i++) {
            if (D(i + j) != hVar.f(0 + i)) {
                return false;
            }
        }
        return true;
    }

    @Override // g.g
    public String k(Charset charset) {
        try {
            return J(this.f6176d, charset);
        } catch (EOFException e2) {
            throw new AssertionError(e2);
        }
    }

    @Override // g.w
    public void l(e eVar, long j) {
        t b2;
        if (eVar == null) {
            throw new IllegalArgumentException("source == null");
        }
        if (eVar != this) {
            z.b(eVar.f6176d, 0L, j);
            while (j > 0) {
                t tVar = eVar.f6175c;
                if (j < tVar.f6211c - tVar.f6210b) {
                    t tVar2 = this.f6175c;
                    t tVar3 = tVar2 != null ? tVar2.f6215g : null;
                    if (tVar3 != null && tVar3.f6213e) {
                        if ((tVar3.f6211c + j) - (tVar3.f6212d ? 0 : tVar3.f6210b) <= 8192) {
                            tVar.d(tVar3, (int) j);
                            eVar.f6176d -= j;
                            this.f6176d += j;
                            return;
                        }
                    }
                    int i = (int) j;
                    Objects.requireNonNull(tVar);
                    if (i > 0 && i <= tVar.f6211c - tVar.f6210b) {
                        if (i >= 1024) {
                            b2 = tVar.c();
                        } else {
                            b2 = u.b();
                            System.arraycopy(tVar.f6209a, tVar.f6210b, b2.f6209a, 0, i);
                        }
                        b2.f6211c = b2.f6210b + i;
                        tVar.f6210b += i;
                        tVar.f6215g.b(b2);
                        eVar.f6175c = b2;
                    } else {
                        throw new IllegalArgumentException();
                    }
                }
                t tVar4 = eVar.f6175c;
                long j2 = tVar4.f6211c - tVar4.f6210b;
                eVar.f6175c = tVar4.a();
                t tVar5 = this.f6175c;
                if (tVar5 == null) {
                    this.f6175c = tVar4;
                    tVar4.f6215g = tVar4;
                    tVar4.f6214f = tVar4;
                } else {
                    tVar5.f6215g.b(tVar4);
                    t tVar6 = tVar4.f6215g;
                    if (tVar6 != tVar4) {
                        if (tVar6.f6213e) {
                            int i2 = tVar4.f6211c - tVar4.f6210b;
                            if (i2 <= (8192 - tVar6.f6211c) + (tVar6.f6212d ? 0 : tVar6.f6210b)) {
                                tVar4.d(tVar6, i2);
                                tVar4.a();
                                u.a(tVar4);
                            }
                        }
                    } else {
                        throw new IllegalStateException();
                    }
                }
                eVar.f6176d -= j2;
                this.f6176d += j2;
                j -= j2;
            }
            return;
        }
        throw new IllegalArgumentException("source == this");
    }

    @Override // g.g
    public boolean o(long j) {
        return this.f6176d >= j;
    }

    @Override // g.g
    public String p() {
        return h(RecyclerView.FOREVER_NS);
    }

    @Override // g.g
    public int q() {
        int readInt = readInt();
        Charset charset = z.f6224a;
        return ((readInt & 255) << 24) | (((-16777216) & readInt) >>> 24) | ((16711680 & readInt) >>> 8) | ((65280 & readInt) << 8);
    }

    @Override // g.g
    public byte[] r(long j) {
        z.b(this.f6176d, 0L, j);
        if (j <= 2147483647L) {
            byte[] bArr = new byte[(int) j];
            I(bArr);
            return bArr;
        }
        throw new IllegalArgumentException(c.b.a.a.a.l("byteCount > Integer.MAX_VALUE: ", j));
    }

    @Override // java.nio.channels.ReadableByteChannel
    public int read(ByteBuffer byteBuffer) {
        t tVar = this.f6175c;
        if (tVar == null) {
            return -1;
        }
        int min = Math.min(byteBuffer.remaining(), tVar.f6211c - tVar.f6210b);
        byteBuffer.put(tVar.f6209a, tVar.f6210b, min);
        int i = tVar.f6210b + min;
        tVar.f6210b = i;
        this.f6176d -= min;
        if (i == tVar.f6211c) {
            this.f6175c = tVar.a();
            u.a(tVar);
        }
        return min;
    }

    @Override // g.g
    public byte readByte() {
        long j = this.f6176d;
        if (j != 0) {
            t tVar = this.f6175c;
            int i = tVar.f6210b;
            int i2 = tVar.f6211c;
            int i3 = i + 1;
            byte b2 = tVar.f6209a[i];
            this.f6176d = j - 1;
            if (i3 == i2) {
                this.f6175c = tVar.a();
                u.a(tVar);
            } else {
                tVar.f6210b = i3;
            }
            return b2;
        }
        throw new IllegalStateException("size == 0");
    }

    @Override // g.g
    public int readInt() {
        long j = this.f6176d;
        if (j >= 4) {
            t tVar = this.f6175c;
            int i = tVar.f6210b;
            int i2 = tVar.f6211c;
            if (i2 - i < 4) {
                return ((readByte() & UnsignedBytes.MAX_VALUE) << 24) | ((readByte() & UnsignedBytes.MAX_VALUE) << 16) | ((readByte() & UnsignedBytes.MAX_VALUE) << 8) | (readByte() & UnsignedBytes.MAX_VALUE);
            }
            byte[] bArr = tVar.f6209a;
            int i3 = i + 1;
            int i4 = i3 + 1;
            int i5 = ((bArr[i] & UnsignedBytes.MAX_VALUE) << 24) | ((bArr[i3] & UnsignedBytes.MAX_VALUE) << 16);
            int i6 = i4 + 1;
            int i7 = i5 | ((bArr[i4] & UnsignedBytes.MAX_VALUE) << 8);
            int i8 = i6 + 1;
            int i9 = i7 | (bArr[i6] & UnsignedBytes.MAX_VALUE);
            this.f6176d = j - 4;
            if (i8 == i2) {
                this.f6175c = tVar.a();
                u.a(tVar);
            } else {
                tVar.f6210b = i8;
            }
            return i9;
        }
        StringBuilder x = c.b.a.a.a.x("size < 4: ");
        x.append(this.f6176d);
        throw new IllegalStateException(x.toString());
    }

    @Override // g.g
    public short readShort() {
        long j = this.f6176d;
        if (j >= 2) {
            t tVar = this.f6175c;
            int i = tVar.f6210b;
            int i2 = tVar.f6211c;
            if (i2 - i < 2) {
                return (short) (((readByte() & UnsignedBytes.MAX_VALUE) << 8) | (readByte() & UnsignedBytes.MAX_VALUE));
            }
            byte[] bArr = tVar.f6209a;
            int i3 = i + 1;
            int i4 = i3 + 1;
            int i5 = ((bArr[i] & UnsignedBytes.MAX_VALUE) << 8) | (bArr[i3] & UnsignedBytes.MAX_VALUE);
            this.f6176d = j - 2;
            if (i4 == i2) {
                this.f6175c = tVar.a();
                u.a(tVar);
            } else {
                tVar.f6210b = i4;
            }
            return (short) i5;
        }
        StringBuilder x = c.b.a.a.a.x("size < 2: ");
        x.append(this.f6176d);
        throw new IllegalStateException(x.toString());
    }

    @Override // g.f
    public /* bridge */ /* synthetic */ f s(h hVar) {
        P(hVar);
        return this;
    }

    @Override // g.g
    public short t() {
        short readShort = readShort();
        Charset charset = z.f6224a;
        int i = readShort & 65535;
        return (short) (((i & 255) << 8) | ((65280 & i) >>> 8));
    }

    public String toString() {
        h vVar;
        long j = this.f6176d;
        if (j <= 2147483647L) {
            int i = (int) j;
            if (i == 0) {
                vVar = h.f6179c;
            } else {
                vVar = new v(this, i);
            }
            return vVar.toString();
        }
        StringBuilder x = c.b.a.a.a.x("size > Integer.MAX_VALUE: ");
        x.append(this.f6176d);
        throw new IllegalArgumentException(x.toString());
    }

    @Override // g.x
    public long u(e eVar, long j) {
        if (eVar != null) {
            if (j >= 0) {
                long j2 = this.f6176d;
                if (j2 == 0) {
                    return -1L;
                }
                if (j > j2) {
                    j = j2;
                }
                eVar.l(this, j);
                return j;
            }
            throw new IllegalArgumentException(c.b.a.a.a.l("byteCount < 0: ", j));
        }
        throw new IllegalArgumentException("sink == null");
    }

    @Override // g.g
    public void v(long j) {
        if (this.f6176d < j) {
            throw new EOFException();
        }
    }

    @Override // g.f
    public /* bridge */ /* synthetic */ f write(byte[] bArr) {
        Q(bArr);
        return this;
    }

    @Override // g.f
    public /* bridge */ /* synthetic */ f writeByte(int i) {
        T(i);
        return this;
    }

    @Override // g.f
    public /* bridge */ /* synthetic */ f writeInt(int i) {
        W(i);
        return this;
    }

    @Override // g.f
    public /* bridge */ /* synthetic */ f writeShort(int i) {
        X(i);
        return this;
    }

    @Override // g.g
    public long x(byte b2) {
        return E(b2, 0L, RecyclerView.FOREVER_NS);
    }

    /* JADX WARN: Removed duplicated region for block: B:33:0x0089  */
    /* JADX WARN: Removed duplicated region for block: B:34:0x0093  */
    /* JADX WARN: Removed duplicated region for block: B:36:0x0097  */
    /* JADX WARN: Removed duplicated region for block: B:42:0x009b A[EDGE_INSN: B:42:0x009b->B:38:0x009b ?: BREAK  , SYNTHETIC] */
    @Override // g.g
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public long y() {
        int i;
        int i2;
        if (this.f6176d != 0) {
            int i3 = 0;
            boolean z = false;
            long j = 0;
            do {
                t tVar = this.f6175c;
                byte[] bArr = tVar.f6209a;
                int i4 = tVar.f6210b;
                int i5 = tVar.f6211c;
                while (i4 < i5) {
                    byte b2 = bArr[i4];
                    if (b2 < 48 || b2 > 57) {
                        if (b2 >= 97 && b2 <= 102) {
                            i = b2 - 97;
                        } else if (b2 >= 65 && b2 <= 70) {
                            i = b2 - 65;
                        } else if (i3 == 0) {
                            StringBuilder x = c.b.a.a.a.x("Expected leading [0-9a-fA-F] character but was 0x");
                            x.append(Integer.toHexString(b2));
                            throw new NumberFormatException(x.toString());
                        } else {
                            z = true;
                            if (i4 != i5) {
                                this.f6175c = tVar.a();
                                u.a(tVar);
                            } else {
                                tVar.f6210b = i4;
                            }
                            if (!z) {
                                break;
                            }
                        }
                        i2 = i + 10;
                    } else {
                        i2 = b2 - 48;
                    }
                    if (((-1152921504606846976L) & j) != 0) {
                        e m = new e().m(j);
                        m.T(b2);
                        StringBuilder x2 = c.b.a.a.a.x("Number too large: ");
                        x2.append(m.K());
                        throw new NumberFormatException(x2.toString());
                    }
                    j = (j << 4) | i2;
                    i4++;
                    i3++;
                }
                if (i4 != i5) {
                }
                if (!z) {
                }
            } while (this.f6175c != null);
            this.f6176d -= i3;
            return j;
        }
        throw new IllegalStateException("size == 0");
    }

    @Override // g.g
    public InputStream z() {
        return new a();
    }

    @Override // g.f
    public /* bridge */ /* synthetic */ f write(byte[] bArr, int i, int i2) {
        R(bArr, i, i2);
        return this;
    }

    @Override // java.nio.channels.WritableByteChannel
    public int write(ByteBuffer byteBuffer) {
        if (byteBuffer != null) {
            int remaining = byteBuffer.remaining();
            int i = remaining;
            while (i > 0) {
                t O = O(1);
                int min = Math.min(i, 8192 - O.f6211c);
                byteBuffer.get(O.f6209a, O.f6211c, min);
                i -= min;
                O.f6211c += min;
            }
            this.f6176d += remaining;
            return remaining;
        }
        throw new IllegalArgumentException("source == null");
    }
}