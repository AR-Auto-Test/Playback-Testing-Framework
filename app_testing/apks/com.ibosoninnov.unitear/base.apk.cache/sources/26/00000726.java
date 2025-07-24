package c.c.a.m.u;

import com.google.common.base.Ascii;
import com.google.common.primitives.UnsignedBytes;
import java.io.FilterInputStream;
import java.io.InputStream;

/* compiled from: ExifOrientationStream.java */
/* loaded from: classes.dex */
public final class g extends FilterInputStream {

    /* renamed from: b  reason: collision with root package name */
    public static final byte[] f3558b;

    /* renamed from: c  reason: collision with root package name */
    public static final int f3559c;

    /* renamed from: d  reason: collision with root package name */
    public static final int f3560d;

    /* renamed from: e  reason: collision with root package name */
    public final byte f3561e;

    /* renamed from: f  reason: collision with root package name */
    public int f3562f;

    static {
        byte[] bArr = {-1, -31, 0, Ascii.FS, 69, 120, 105, 102, 0, 0, 77, 77, 0, 0, 0, 0, 0, 8, 0, 1, 1, 18, 0, 2, 0, 0, 0, 1, 0};
        f3558b = bArr;
        int length = bArr.length;
        f3559c = length;
        f3560d = length + 2;
    }

    public g(InputStream inputStream, int i) {
        super(inputStream);
        if (i >= -1 && i <= 8) {
            this.f3561e = (byte) i;
            return;
        }
        throw new IllegalArgumentException(c.b.a.a.a.j("Cannot add invalid orientation: ", i));
    }

    @Override // java.io.FilterInputStream, java.io.InputStream
    public void mark(int i) {
        throw new UnsupportedOperationException();
    }

    @Override // java.io.FilterInputStream, java.io.InputStream
    public boolean markSupported() {
        return false;
    }

    @Override // java.io.FilterInputStream, java.io.InputStream
    public int read() {
        int read;
        int i;
        int i2 = this.f3562f;
        if (i2 < 2 || i2 > (i = f3560d)) {
            read = super.read();
        } else if (i2 == i) {
            read = this.f3561e;
        } else {
            read = f3558b[i2 - 2] & UnsignedBytes.MAX_VALUE;
        }
        if (read != -1) {
            this.f3562f++;
        }
        return read;
    }

    @Override // java.io.FilterInputStream, java.io.InputStream
    public void reset() {
        throw new UnsupportedOperationException();
    }

    @Override // java.io.FilterInputStream, java.io.InputStream
    public long skip(long j) {
        long skip = super.skip(j);
        if (skip > 0) {
            this.f3562f = (int) (this.f3562f + skip);
        }
        return skip;
    }

    @Override // java.io.FilterInputStream, java.io.InputStream
    public int read(byte[] bArr, int i, int i2) {
        int i3;
        int i4 = this.f3562f;
        int i5 = f3560d;
        if (i4 > i5) {
            i3 = super.read(bArr, i, i2);
        } else if (i4 == i5) {
            bArr[i] = this.f3561e;
            i3 = 1;
        } else if (i4 < 2) {
            i3 = super.read(bArr, i, 2 - i4);
        } else {
            int min = Math.min(i5 - i4, i2);
            System.arraycopy(f3558b, this.f3562f - 2, bArr, i, min);
            i3 = min;
        }
        if (i3 > 0) {
            this.f3562f += i3;
        }
        return i3;
    }
}