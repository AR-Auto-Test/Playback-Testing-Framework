package c.c.a.s;

import java.io.FilterInputStream;
import java.io.IOException;
import java.io.InputStream;

/* compiled from: ContentLengthInputStream.java */
/* loaded from: classes.dex */
public final class c extends FilterInputStream {

    /* renamed from: b  reason: collision with root package name */
    public final long f4179b;

    /* renamed from: c  reason: collision with root package name */
    public int f4180c;

    public c(InputStream inputStream, long j) {
        super(inputStream);
        this.f4179b = j;
    }

    public final int B(int i) {
        if (i >= 0) {
            this.f4180c += i;
        } else if (this.f4179b - this.f4180c > 0) {
            StringBuilder x = c.b.a.a.a.x("Failed to read all expected data, expected: ");
            x.append(this.f4179b);
            x.append(", but read: ");
            x.append(this.f4180c);
            throw new IOException(x.toString());
        }
        return i;
    }

    @Override // java.io.FilterInputStream, java.io.InputStream
    public synchronized int available() {
        return (int) Math.max(this.f4179b - this.f4180c, ((FilterInputStream) this).in.available());
    }

    @Override // java.io.FilterInputStream, java.io.InputStream
    public synchronized int read() {
        int read;
        read = super.read();
        B(read >= 0 ? 1 : -1);
        return read;
    }

    @Override // java.io.FilterInputStream, java.io.InputStream
    public int read(byte[] bArr) {
        return read(bArr, 0, bArr.length);
    }

    @Override // java.io.FilterInputStream, java.io.InputStream
    public synchronized int read(byte[] bArr, int i, int i2) {
        int read;
        read = super.read(bArr, i, i2);
        B(read);
        return read;
    }
}