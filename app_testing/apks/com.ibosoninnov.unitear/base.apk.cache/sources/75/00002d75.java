package g;

import java.io.InputStream;
import java.nio.channels.ReadableByteChannel;
import java.nio.charset.Charset;

/* compiled from: BufferedSource.java */
/* loaded from: classes2.dex */
public interface g extends x, ReadableByteChannel {
    int A(q qVar);

    @Deprecated
    e a();

    void c(long j);

    h d(long j);

    e e();

    boolean f();

    long g(h hVar);

    String h(long j);

    boolean j(long j, h hVar);

    String k(Charset charset);

    boolean o(long j);

    String p();

    int q();

    byte[] r(long j);

    byte readByte();

    int readInt();

    short readShort();

    short t();

    void v(long j);

    long x(byte b2);

    long y();

    InputStream z();
}