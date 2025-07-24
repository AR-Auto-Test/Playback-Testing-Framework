package c.c.a.m.x.c;

import android.media.MediaDataSource;
import c.c.a.m.x.c.c0;
import java.nio.ByteBuffer;

/* compiled from: VideoDecoder.java */
/* loaded from: classes.dex */
public class d0 extends MediaDataSource {

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ ByteBuffer f3957b;

    public d0(c0.d dVar, ByteBuffer byteBuffer) {
        this.f3957b = byteBuffer;
    }

    @Override // java.io.Closeable, java.lang.AutoCloseable
    public void close() {
    }

    @Override // android.media.MediaDataSource
    public long getSize() {
        return this.f3957b.limit();
    }

    @Override // android.media.MediaDataSource
    public int readAt(long j, byte[] bArr, int i, int i2) {
        if (j >= this.f3957b.limit()) {
            return -1;
        }
        this.f3957b.position((int) j);
        int min = Math.min(i2, this.f3957b.remaining());
        this.f3957b.get(bArr, i, min);
        return min;
    }
}