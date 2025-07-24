package c.c.a.m.u;

import java.io.OutputStream;

/* compiled from: BufferedOutputStream.java */
/* loaded from: classes.dex */
public final class c extends OutputStream {

    /* renamed from: b  reason: collision with root package name */
    public final OutputStream f3551b;

    /* renamed from: c  reason: collision with root package name */
    public byte[] f3552c;

    /* renamed from: d  reason: collision with root package name */
    public c.c.a.m.v.c0.b f3553d;

    /* renamed from: e  reason: collision with root package name */
    public int f3554e;

    public c(OutputStream outputStream, c.c.a.m.v.c0.b bVar) {
        this.f3551b = outputStream;
        this.f3553d = bVar;
        this.f3552c = (byte[]) bVar.d(65536, byte[].class);
    }

    @Override // java.io.OutputStream, java.io.Closeable, java.lang.AutoCloseable
    public void close() {
        try {
            flush();
            this.f3551b.close();
            byte[] bArr = this.f3552c;
            if (bArr != null) {
                this.f3553d.put(bArr);
                this.f3552c = null;
            }
        } catch (Throwable th) {
            this.f3551b.close();
            throw th;
        }
    }

    @Override // java.io.OutputStream, java.io.Flushable
    public void flush() {
        int i = this.f3554e;
        if (i > 0) {
            this.f3551b.write(this.f3552c, 0, i);
            this.f3554e = 0;
        }
        this.f3551b.flush();
    }

    @Override // java.io.OutputStream
    public void write(int i) {
        byte[] bArr = this.f3552c;
        int i2 = this.f3554e;
        int i3 = i2 + 1;
        this.f3554e = i3;
        bArr[i2] = (byte) i;
        if (i3 != bArr.length || i3 <= 0) {
            return;
        }
        this.f3551b.write(bArr, 0, i3);
        this.f3554e = 0;
    }

    @Override // java.io.OutputStream
    public void write(byte[] bArr) {
        write(bArr, 0, bArr.length);
    }

    @Override // java.io.OutputStream
    public void write(byte[] bArr, int i, int i2) {
        int i3 = 0;
        do {
            int i4 = i2 - i3;
            int i5 = i + i3;
            int i6 = this.f3554e;
            if (i6 == 0 && i4 >= this.f3552c.length) {
                this.f3551b.write(bArr, i5, i4);
                return;
            }
            int min = Math.min(i4, this.f3552c.length - i6);
            System.arraycopy(bArr, i5, this.f3552c, this.f3554e, min);
            int i7 = this.f3554e + min;
            this.f3554e = i7;
            i3 += min;
            byte[] bArr2 = this.f3552c;
            if (i7 == bArr2.length && i7 > 0) {
                this.f3551b.write(bArr2, 0, i7);
                this.f3554e = 0;
                continue;
            }
        } while (i3 < i2);
    }
}