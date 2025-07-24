package c.c.a.k;

import java.io.ByteArrayOutputStream;
import java.io.Closeable;
import java.io.EOFException;
import java.io.IOException;
import java.io.InputStream;
import java.io.UnsupportedEncodingException;
import java.nio.charset.Charset;

/* compiled from: StrictLineReader.java */
/* loaded from: classes.dex */
public class b implements Closeable {

    /* renamed from: b  reason: collision with root package name */
    public final InputStream f3481b;

    /* renamed from: c  reason: collision with root package name */
    public final Charset f3482c;

    /* renamed from: d  reason: collision with root package name */
    public byte[] f3483d;

    /* renamed from: e  reason: collision with root package name */
    public int f3484e;

    /* renamed from: f  reason: collision with root package name */
    public int f3485f;

    /* compiled from: StrictLineReader.java */
    /* loaded from: classes.dex */
    public class a extends ByteArrayOutputStream {
        public a(int i) {
            super(i);
        }

        @Override // java.io.ByteArrayOutputStream
        public String toString() {
            int i = ((ByteArrayOutputStream) this).count;
            if (i > 0 && ((ByteArrayOutputStream) this).buf[i - 1] == 13) {
                i--;
            }
            try {
                return new String(((ByteArrayOutputStream) this).buf, 0, i, b.this.f3482c.name());
            } catch (UnsupportedEncodingException e2) {
                throw new AssertionError(e2);
            }
        }
    }

    public b(InputStream inputStream, Charset charset) {
        if (charset != null) {
            if (charset.equals(c.f3487a)) {
                this.f3481b = inputStream;
                this.f3482c = charset;
                this.f3483d = new byte[8192];
                return;
            }
            throw new IllegalArgumentException("Unsupported encoding");
        }
        throw null;
    }

    public final void B() {
        InputStream inputStream = this.f3481b;
        byte[] bArr = this.f3483d;
        int read = inputStream.read(bArr, 0, bArr.length);
        if (read != -1) {
            this.f3484e = 0;
            this.f3485f = read;
            return;
        }
        throw new EOFException();
    }

    public String C() {
        int i;
        byte[] bArr;
        int i2;
        synchronized (this.f3481b) {
            if (this.f3483d != null) {
                if (this.f3484e >= this.f3485f) {
                    B();
                }
                for (int i3 = this.f3484e; i3 != this.f3485f; i3++) {
                    byte[] bArr2 = this.f3483d;
                    if (bArr2[i3] == 10) {
                        if (i3 != this.f3484e) {
                            i2 = i3 - 1;
                            if (bArr2[i2] == 13) {
                                byte[] bArr3 = this.f3483d;
                                int i4 = this.f3484e;
                                String str = new String(bArr3, i4, i2 - i4, this.f3482c.name());
                                this.f3484e = i3 + 1;
                                return str;
                            }
                        }
                        i2 = i3;
                        byte[] bArr32 = this.f3483d;
                        int i42 = this.f3484e;
                        String str2 = new String(bArr32, i42, i2 - i42, this.f3482c.name());
                        this.f3484e = i3 + 1;
                        return str2;
                    }
                }
                a aVar = new a((this.f3485f - this.f3484e) + 80);
                loop1: while (true) {
                    byte[] bArr4 = this.f3483d;
                    int i5 = this.f3484e;
                    aVar.write(bArr4, i5, this.f3485f - i5);
                    this.f3485f = -1;
                    B();
                    i = this.f3484e;
                    while (i != this.f3485f) {
                        bArr = this.f3483d;
                        if (bArr[i] == 10) {
                            break loop1;
                        }
                        i++;
                    }
                }
                int i6 = this.f3484e;
                if (i != i6) {
                    aVar.write(bArr, i6, i - i6);
                }
                this.f3484e = i + 1;
                return aVar.toString();
            }
            throw new IOException("LineReader is closed");
        }
    }

    @Override // java.io.Closeable, java.lang.AutoCloseable
    public void close() {
        synchronized (this.f3481b) {
            if (this.f3483d != null) {
                this.f3483d = null;
                this.f3481b.close();
            }
        }
    }
}