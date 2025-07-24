package c.c.a.s;

import java.io.FilterInputStream;
import java.io.InputStream;

/* compiled from: MarkEnforcingInputStream.java */
/* loaded from: classes.dex */
public class h extends FilterInputStream {

    /* renamed from: b  reason: collision with root package name */
    public int f4193b;

    public h(InputStream inputStream) {
        super(inputStream);
        this.f4193b = Integer.MIN_VALUE;
    }

    public final long B(long j) {
        int i = this.f4193b;
        if (i == 0) {
            return -1L;
        }
        return (i == Integer.MIN_VALUE || j <= ((long) i)) ? j : i;
    }

    public final void C(long j) {
        int i = this.f4193b;
        if (i == Integer.MIN_VALUE || j == -1) {
            return;
        }
        this.f4193b = (int) (i - j);
    }

    @Override // java.io.FilterInputStream, java.io.InputStream
    public int available() {
        int i = this.f4193b;
        if (i == Integer.MIN_VALUE) {
            return super.available();
        }
        return Math.min(i, super.available());
    }

    @Override // java.io.FilterInputStream, java.io.InputStream
    public synchronized void mark(int i) {
        super.mark(i);
        this.f4193b = i;
    }

    @Override // java.io.FilterInputStream, java.io.InputStream
    public int read() {
        if (B(1L) == -1) {
            return -1;
        }
        int read = super.read();
        C(1L);
        return read;
    }

    @Override // java.io.FilterInputStream, java.io.InputStream
    public synchronized void reset() {
        super.reset();
        this.f4193b = Integer.MIN_VALUE;
    }

    @Override // java.io.FilterInputStream, java.io.InputStream
    public long skip(long j) {
        long B = B(j);
        if (B == -1) {
            return 0L;
        }
        long skip = super.skip(B);
        C(skip);
        return skip;
    }

    @Override // java.io.FilterInputStream, java.io.InputStream
    public int read(byte[] bArr, int i, int i2) {
        int B = (int) B(i2);
        if (B == -1) {
            return -1;
        }
        int read = super.read(bArr, i, B);
        C(read);
        return read;
    }
}