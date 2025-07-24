package b.j.i;

import android.util.Log;
import java.io.Writer;

/* compiled from: LogWriter.java */
@Deprecated
/* loaded from: classes.dex */
public class b extends Writer {

    /* renamed from: b  reason: collision with root package name */
    public final String f2190b;

    /* renamed from: c  reason: collision with root package name */
    public StringBuilder f2191c = new StringBuilder(128);

    public b(String str) {
        this.f2190b = str;
    }

    public final void B() {
        if (this.f2191c.length() > 0) {
            Log.d(this.f2190b, this.f2191c.toString());
            StringBuilder sb = this.f2191c;
            sb.delete(0, sb.length());
        }
    }

    @Override // java.io.Writer, java.io.Closeable, java.lang.AutoCloseable
    public void close() {
        B();
    }

    @Override // java.io.Writer, java.io.Flushable
    public void flush() {
        B();
    }

    @Override // java.io.Writer
    public void write(char[] cArr, int i, int i2) {
        for (int i3 = 0; i3 < i2; i3++) {
            char c2 = cArr[i + i3];
            if (c2 == '\n') {
                B();
            } else {
                this.f2191c.append(c2);
            }
        }
    }
}