package c.c.a.s;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayDeque;
import java.util.Queue;

/* compiled from: ExceptionPassthroughInputStream.java */
/* loaded from: classes.dex */
public final class d extends InputStream {

    /* renamed from: b  reason: collision with root package name */
    public static final Queue<d> f4181b;

    /* renamed from: c  reason: collision with root package name */
    public InputStream f4182c;

    /* renamed from: d  reason: collision with root package name */
    public IOException f4183d;

    static {
        char[] cArr = j.f4197a;
        f4181b = new ArrayDeque(0);
    }

    @Override // java.io.InputStream
    public int available() {
        return this.f4182c.available();
    }

    @Override // java.io.InputStream, java.io.Closeable, java.lang.AutoCloseable
    public void close() {
        this.f4182c.close();
    }

    @Override // java.io.InputStream
    public void mark(int i) {
        this.f4182c.mark(i);
    }

    @Override // java.io.InputStream
    public boolean markSupported() {
        return this.f4182c.markSupported();
    }

    @Override // java.io.InputStream
    public int read() {
        try {
            return this.f4182c.read();
        } catch (IOException e2) {
            this.f4183d = e2;
            throw e2;
        }
    }

    public void release() {
        this.f4183d = null;
        this.f4182c = null;
        Queue<d> queue = f4181b;
        synchronized (queue) {
            queue.offer(this);
        }
    }

    @Override // java.io.InputStream
    public synchronized void reset() {
        this.f4182c.reset();
    }

    @Override // java.io.InputStream
    public long skip(long j) {
        try {
            return this.f4182c.skip(j);
        } catch (IOException e2) {
            this.f4183d = e2;
            throw e2;
        }
    }

    @Override // java.io.InputStream
    public int read(byte[] bArr) {
        try {
            return this.f4182c.read(bArr);
        } catch (IOException e2) {
            this.f4183d = e2;
            throw e2;
        }
    }

    @Override // java.io.InputStream
    public int read(byte[] bArr, int i, int i2) {
        try {
            return this.f4182c.read(bArr, i, i2);
        } catch (IOException e2) {
            this.f4183d = e2;
            throw e2;
        }
    }
}