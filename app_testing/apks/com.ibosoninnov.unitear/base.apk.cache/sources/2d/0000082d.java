package c.c.a.m.x.c;

import com.google.common.primitives.UnsignedBytes;
import java.io.FilterInputStream;
import java.io.IOException;
import java.io.InputStream;

/* compiled from: RecyclableBufferedInputStream.java */
/* loaded from: classes.dex */
public class w extends FilterInputStream {

    /* renamed from: b  reason: collision with root package name */
    public volatile byte[] f4006b;

    /* renamed from: c  reason: collision with root package name */
    public int f4007c;

    /* renamed from: d  reason: collision with root package name */
    public int f4008d;

    /* renamed from: e  reason: collision with root package name */
    public int f4009e;

    /* renamed from: f  reason: collision with root package name */
    public int f4010f;

    /* renamed from: g  reason: collision with root package name */
    public final c.c.a.m.v.c0.b f4011g;

    /* compiled from: RecyclableBufferedInputStream.java */
    /* loaded from: classes.dex */
    public static class a extends IOException {
        public a(String str) {
            super(str);
        }
    }

    public w(InputStream inputStream, c.c.a.m.v.c0.b bVar) {
        super(inputStream);
        this.f4009e = -1;
        this.f4011g = bVar;
        this.f4006b = (byte[]) bVar.d(65536, byte[].class);
    }

    public static IOException C() {
        throw new IOException("BufferedInputStream is closed");
    }

    public final int B(InputStream inputStream, byte[] bArr) {
        int i = this.f4009e;
        if (i != -1) {
            int i2 = this.f4010f - i;
            int i3 = this.f4008d;
            if (i2 < i3) {
                if (i == 0 && i3 > bArr.length && this.f4007c == bArr.length) {
                    int length = bArr.length * 2;
                    if (length <= i3) {
                        i3 = length;
                    }
                    byte[] bArr2 = (byte[]) this.f4011g.d(i3, byte[].class);
                    System.arraycopy(bArr, 0, bArr2, 0, bArr.length);
                    this.f4006b = bArr2;
                    this.f4011g.put(bArr);
                    bArr = bArr2;
                } else if (i > 0) {
                    System.arraycopy(bArr, i, bArr, 0, bArr.length - i);
                }
                int i4 = this.f4010f - this.f4009e;
                this.f4010f = i4;
                this.f4009e = 0;
                this.f4007c = 0;
                int read = inputStream.read(bArr, i4, bArr.length - i4);
                int i5 = this.f4010f;
                if (read > 0) {
                    i5 += read;
                }
                this.f4007c = i5;
                return read;
            }
        }
        int read2 = inputStream.read(bArr);
        if (read2 > 0) {
            this.f4009e = -1;
            this.f4010f = 0;
            this.f4007c = read2;
        }
        return read2;
    }

    @Override // java.io.FilterInputStream, java.io.InputStream
    public synchronized int available() {
        InputStream inputStream;
        inputStream = ((FilterInputStream) this).in;
        if (this.f4006b != null && inputStream != null) {
        } else {
            C();
            throw null;
        }
        return (this.f4007c - this.f4010f) + inputStream.available();
    }

    @Override // java.io.FilterInputStream, java.io.InputStream, java.io.Closeable, java.lang.AutoCloseable
    public void close() {
        if (this.f4006b != null) {
            this.f4011g.put(this.f4006b);
            this.f4006b = null;
        }
        InputStream inputStream = ((FilterInputStream) this).in;
        ((FilterInputStream) this).in = null;
        if (inputStream != null) {
            inputStream.close();
        }
    }

    @Override // java.io.FilterInputStream, java.io.InputStream
    public synchronized void mark(int i) {
        this.f4008d = Math.max(this.f4008d, i);
        this.f4009e = this.f4010f;
    }

    @Override // java.io.FilterInputStream, java.io.InputStream
    public boolean markSupported() {
        return true;
    }

    @Override // java.io.FilterInputStream, java.io.InputStream
    public synchronized int read() {
        byte[] bArr = this.f4006b;
        InputStream inputStream = ((FilterInputStream) this).in;
        if (bArr != null && inputStream != null) {
            if (this.f4010f < this.f4007c || B(inputStream, bArr) != -1) {
                if (bArr != this.f4006b && (bArr = this.f4006b) == null) {
                    C();
                    throw null;
                }
                int i = this.f4007c;
                int i2 = this.f4010f;
                if (i - i2 > 0) {
                    this.f4010f = i2 + 1;
                    return bArr[i2] & UnsignedBytes.MAX_VALUE;
                }
                return -1;
            }
            return -1;
        }
        C();
        throw null;
    }

    public synchronized void release() {
        if (this.f4006b != null) {
            this.f4011g.put(this.f4006b);
            this.f4006b = null;
        }
    }

    @Override // java.io.FilterInputStream, java.io.InputStream
    public synchronized void reset() {
        if (this.f4006b != null) {
            int i = this.f4009e;
            if (-1 != i) {
                this.f4010f = i;
            } else {
                throw new a("Mark has been invalidated, pos: " + this.f4010f + " markLimit: " + this.f4008d);
            }
        } else {
            throw new IOException("Stream is closed");
        }
    }

    @Override // java.io.FilterInputStream, java.io.InputStream
    public synchronized long skip(long j) {
        if (j < 1) {
            return 0L;
        }
        byte[] bArr = this.f4006b;
        if (bArr != null) {
            InputStream inputStream = ((FilterInputStream) this).in;
            if (inputStream != null) {
                int i = this.f4007c;
                int i2 = this.f4010f;
                if (i - i2 >= j) {
                    this.f4010f = (int) (i2 + j);
                    return j;
                }
                long j2 = i - i2;
                this.f4010f = i;
                if (this.f4009e != -1 && j <= this.f4008d) {
                    if (B(inputStream, bArr) == -1) {
                        return j2;
                    }
                    int i3 = this.f4007c;
                    int i4 = this.f4010f;
                    if (i3 - i4 >= j - j2) {
                        this.f4010f = (int) ((i4 + j) - j2);
                        return j;
                    }
                    long j3 = (j2 + i3) - i4;
                    this.f4010f = i3;
                    return j3;
                }
                long skip = inputStream.skip(j - j2);
                if (skip > 0) {
                    this.f4009e = -1;
                }
                return j2 + skip;
            }
            C();
            throw null;
        }
        C();
        throw null;
    }

    @Override // java.io.FilterInputStream, java.io.InputStream
    public synchronized int read(byte[] bArr, int i, int i2) {
        int i3;
        int i4;
        byte[] bArr2 = this.f4006b;
        if (bArr2 == null) {
            C();
            throw null;
        } else if (i2 == 0) {
            return 0;
        } else {
            InputStream inputStream = ((FilterInputStream) this).in;
            if (inputStream != null) {
                int i5 = this.f4010f;
                int i6 = this.f4007c;
                if (i5 < i6) {
                    int i7 = i6 - i5 >= i2 ? i2 : i6 - i5;
                    System.arraycopy(bArr2, i5, bArr, i, i7);
                    this.f4010f += i7;
                    if (i7 == i2 || inputStream.available() == 0) {
                        return i7;
                    }
                    i += i7;
                    i3 = i2 - i7;
                } else {
                    i3 = i2;
                }
                while (true) {
                    if (this.f4009e == -1 && i3 >= bArr2.length) {
                        i4 = inputStream.read(bArr, i, i3);
                        if (i4 == -1) {
                            return i3 != i2 ? i2 - i3 : -1;
                        }
                    } else if (B(inputStream, bArr2) == -1) {
                        return i3 != i2 ? i2 - i3 : -1;
                    } else {
                        if (bArr2 != this.f4006b && (bArr2 = this.f4006b) == null) {
                            C();
                            throw null;
                        }
                        int i8 = this.f4007c;
                        int i9 = this.f4010f;
                        i4 = i8 - i9 >= i3 ? i3 : i8 - i9;
                        System.arraycopy(bArr2, i9, bArr, i, i4);
                        this.f4010f += i4;
                    }
                    i3 -= i4;
                    if (i3 == 0) {
                        return i2;
                    }
                    if (inputStream.available() == 0) {
                        return i2 - i3;
                    }
                    i += i4;
                }
            } else {
                C();
                throw null;
            }
        }
    }
}