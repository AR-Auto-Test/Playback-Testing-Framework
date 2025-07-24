package g;

import java.nio.channels.WritableByteChannel;

/* compiled from: BufferedSink.java */
/* loaded from: classes2.dex */
public interface f extends w, WritableByteChannel {
    e a();

    @Override // g.w, java.io.Flushable
    void flush();

    f i(String str);

    f m(long j);

    f s(h hVar);

    f w(long j);

    f write(byte[] bArr);

    f write(byte[] bArr, int i, int i2);

    f writeByte(int i);

    f writeInt(int i);

    f writeShort(int i);
}