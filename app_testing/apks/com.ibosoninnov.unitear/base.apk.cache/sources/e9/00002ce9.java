package f;

import java.io.Closeable;
import java.io.InputStream;
import java.nio.charset.Charset;

/* compiled from: ResponseBody.java */
/* loaded from: classes2.dex */
public abstract class d0 implements Closeable {
    public final InputStream B() {
        return E().z();
    }

    public abstract long C();

    public abstract t D();

    public abstract g.g E();

    public final String F() {
        g.g E = E();
        try {
            t D = D();
            Charset charset = f.g0.c.i;
            if (D != null) {
                try {
                    String str = D.f6106e;
                    if (str != null) {
                        charset = Charset.forName(str);
                    }
                } catch (IllegalArgumentException unused) {
                }
            }
            return E.k(f.g0.c.b(E, charset));
        } finally {
            f.g0.c.f(E);
        }
    }

    @Override // java.io.Closeable, java.lang.AutoCloseable
    public void close() {
        f.g0.c.f(E());
    }
}