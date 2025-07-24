package g;

import java.io.IOException;
import java.net.Socket;
import java.net.SocketTimeoutException;
import java.util.logging.Level;
import java.util.logging.Logger;

/* compiled from: Okio.java */
/* loaded from: classes2.dex */
public final class p extends c {
    public final /* synthetic */ Socket k;

    public p(Socket socket) {
        this.k = socket;
    }

    @Override // g.c
    public IOException l(IOException iOException) {
        SocketTimeoutException socketTimeoutException = new SocketTimeoutException("timeout");
        if (iOException != null) {
            socketTimeoutException.initCause(iOException);
        }
        return socketTimeoutException;
    }

    @Override // g.c
    public void m() {
        try {
            this.k.close();
        } catch (AssertionError e2) {
            if (o.a(e2)) {
                Logger logger = o.f6197a;
                Level level = Level.WARNING;
                StringBuilder x = c.b.a.a.a.x("Failed to close timed out socket ");
                x.append(this.k);
                logger.log(level, x.toString(), (Throwable) e2);
                return;
            }
            throw e2;
        } catch (Exception e3) {
            Logger logger2 = o.f6197a;
            Level level2 = Level.WARNING;
            StringBuilder x2 = c.b.a.a.a.x("Failed to close timed out socket ");
            x2.append(this.k);
            logger2.log(level2, x2.toString(), (Throwable) e3);
        }
    }
}