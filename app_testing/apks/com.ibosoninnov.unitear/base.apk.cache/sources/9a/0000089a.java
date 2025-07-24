package c.c.a.s;

import com.google.common.primitives.UnsignedBytes;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.concurrent.atomic.AtomicReference;

/* compiled from: ByteBufferUtil.java */
/* loaded from: classes.dex */
public final class a {

    /* renamed from: a  reason: collision with root package name */
    public static final AtomicReference<byte[]> f4173a = new AtomicReference<>();

    /* compiled from: ByteBufferUtil.java */
    /* loaded from: classes.dex */
    public static final class b {

        /* renamed from: a  reason: collision with root package name */
        public final int f4176a;

        /* renamed from: b  reason: collision with root package name */
        public final int f4177b;

        /* renamed from: c  reason: collision with root package name */
        public final byte[] f4178c;

        public b(byte[] bArr, int i, int i2) {
            this.f4178c = bArr;
            this.f4176a = i;
            this.f4177b = i2;
        }
    }

    public static ByteBuffer a(File file) {
        RandomAccessFile randomAccessFile;
        FileChannel fileChannel = null;
        try {
            long length = file.length();
            if (length <= 2147483647L) {
                if (length != 0) {
                    randomAccessFile = new RandomAccessFile(file, "r");
                    try {
                        fileChannel = randomAccessFile.getChannel();
                        MappedByteBuffer load = fileChannel.map(FileChannel.MapMode.READ_ONLY, 0L, length).load();
                        try {
                            fileChannel.close();
                        } catch (IOException unused) {
                        }
                        try {
                            randomAccessFile.close();
                        } catch (IOException unused2) {
                        }
                        return load;
                    } catch (Throwable th) {
                        th = th;
                        if (fileChannel != null) {
                            try {
                                fileChannel.close();
                            } catch (IOException unused3) {
                            }
                        }
                        if (randomAccessFile != null) {
                            try {
                                randomAccessFile.close();
                            } catch (IOException unused4) {
                            }
                        }
                        throw th;
                    }
                }
                throw new IOException("File unsuitable for memory mapping");
            }
            throw new IOException("File too large to map into memory");
        } catch (Throwable th2) {
            th = th2;
            randomAccessFile = null;
        }
    }

    public static void b(ByteBuffer byteBuffer, File file) {
        RandomAccessFile randomAccessFile;
        byteBuffer.position(0);
        FileChannel fileChannel = null;
        try {
            randomAccessFile = new RandomAccessFile(file, "rw");
        } catch (Throwable th) {
            th = th;
            randomAccessFile = null;
        }
        try {
            fileChannel = randomAccessFile.getChannel();
            fileChannel.write(byteBuffer);
            fileChannel.force(false);
            fileChannel.close();
            randomAccessFile.close();
            try {
                fileChannel.close();
            } catch (IOException unused) {
            }
            try {
                randomAccessFile.close();
            } catch (IOException unused2) {
            }
        } catch (Throwable th2) {
            th = th2;
            if (fileChannel != null) {
                try {
                    fileChannel.close();
                } catch (IOException unused3) {
                }
            }
            if (randomAccessFile != null) {
                try {
                    randomAccessFile.close();
                } catch (IOException unused4) {
                }
            }
            throw th;
        }
    }

    /* compiled from: ByteBufferUtil.java */
    /* renamed from: c.c.a.s.a$a  reason: collision with other inner class name */
    /* loaded from: classes.dex */
    public static class C0086a extends InputStream {

        /* renamed from: b  reason: collision with root package name */
        public final ByteBuffer f4174b;

        /* renamed from: c  reason: collision with root package name */
        public int f4175c = -1;

        public C0086a(ByteBuffer byteBuffer) {
            this.f4174b = byteBuffer;
        }

        @Override // java.io.InputStream
        public int available() {
            return this.f4174b.remaining();
        }

        @Override // java.io.InputStream
        public synchronized void mark(int i) {
            this.f4175c = this.f4174b.position();
        }

        @Override // java.io.InputStream
        public boolean markSupported() {
            return true;
        }

        @Override // java.io.InputStream
        public int read() {
            if (this.f4174b.hasRemaining()) {
                return this.f4174b.get() & UnsignedBytes.MAX_VALUE;
            }
            return -1;
        }

        @Override // java.io.InputStream
        public synchronized void reset() {
            int i = this.f4175c;
            if (i != -1) {
                this.f4174b.position(i);
            } else {
                throw new IOException("Cannot reset to unset mark position");
            }
        }

        @Override // java.io.InputStream
        public long skip(long j) {
            if (this.f4174b.hasRemaining()) {
                long min = Math.min(j, available());
                ByteBuffer byteBuffer = this.f4174b;
                byteBuffer.position((int) (byteBuffer.position() + min));
                return min;
            }
            return -1L;
        }

        @Override // java.io.InputStream
        public int read(byte[] bArr, int i, int i2) {
            if (this.f4174b.hasRemaining()) {
                int min = Math.min(i2, available());
                this.f4174b.get(bArr, i, min);
                return min;
            }
            return -1;
        }
    }
}