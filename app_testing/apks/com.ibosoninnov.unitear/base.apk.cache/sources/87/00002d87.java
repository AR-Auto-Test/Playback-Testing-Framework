package g;

import java.io.Closeable;
import java.io.Flushable;

/* compiled from: Sink.java */
/* loaded from: classes2.dex */
public interface w extends Closeable, Flushable {
    y b();

    @Override // java.io.Closeable, java.lang.AutoCloseable
    void close();

    void flush();

    void l(e eVar, long j);
}