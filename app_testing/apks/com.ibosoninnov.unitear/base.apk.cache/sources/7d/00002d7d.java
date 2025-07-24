package g;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.Socket;
import java.util.logging.Logger;

/* compiled from: Okio.java */
/* loaded from: classes2.dex */
public final class o {

    /* renamed from: a  reason: collision with root package name */
    public static final Logger f6197a = Logger.getLogger(o.class.getName());

    /* compiled from: Okio.java */
    /* loaded from: classes2.dex */
    public class a implements x {

        /* renamed from: b  reason: collision with root package name */
        public final /* synthetic */ y f6198b;

        /* renamed from: c  reason: collision with root package name */
        public final /* synthetic */ InputStream f6199c;

        public a(y yVar, InputStream inputStream) {
            this.f6198b = yVar;
            this.f6199c = inputStream;
        }

        @Override // g.x
        public y b() {
            return this.f6198b;
        }

        @Override // g.x, java.io.Closeable, java.lang.AutoCloseable
        public void close() {
            this.f6199c.close();
        }

        public String toString() {
            StringBuilder x = c.b.a.a.a.x("source(");
            x.append(this.f6199c);
            x.append(")");
            return x.toString();
        }

        @Override // g.x
        public long u(e eVar, long j) {
            int i = (j > 0L ? 1 : (j == 0L ? 0 : -1));
            if (i >= 0) {
                if (i == 0) {
                    return 0L;
                }
                try {
                    this.f6198b.f();
                    t O = eVar.O(1);
                    int read = this.f6199c.read(O.f6209a, O.f6211c, (int) Math.min(j, 8192 - O.f6211c));
                    if (read == -1) {
                        return -1L;
                    }
                    O.f6211c += read;
                    long j2 = read;
                    eVar.f6176d += j2;
                    return j2;
                } catch (AssertionError e2) {
                    if (o.a(e2)) {
                        throw new IOException(e2);
                    }
                    throw e2;
                }
            }
            throw new IllegalArgumentException(c.b.a.a.a.l("byteCount < 0: ", j));
        }
    }

    public static boolean a(AssertionError assertionError) {
        return (assertionError.getCause() == null || assertionError.getMessage() == null || !assertionError.getMessage().contains("getsockname failed")) ? false : true;
    }

    public static w b(Socket socket) {
        if (socket != null) {
            if (socket.getOutputStream() != null) {
                p pVar = new p(socket);
                OutputStream outputStream = socket.getOutputStream();
                if (outputStream != null) {
                    return new g.a(pVar, new n(pVar, outputStream));
                }
                throw new IllegalArgumentException("out == null");
            }
            throw new IOException("socket's output stream == null");
        }
        throw new IllegalArgumentException("socket == null");
    }

    public static x c(InputStream inputStream) {
        return d(inputStream, new y());
    }

    public static x d(InputStream inputStream, y yVar) {
        if (inputStream != null) {
            return new a(yVar, inputStream);
        }
        throw new IllegalArgumentException("in == null");
    }

    public static x e(Socket socket) {
        if (socket != null) {
            if (socket.getInputStream() != null) {
                p pVar = new p(socket);
                return new b(pVar, d(socket.getInputStream(), pVar));
            }
            throw new IOException("socket's input stream == null");
        }
        throw new IllegalArgumentException("socket == null");
    }
}