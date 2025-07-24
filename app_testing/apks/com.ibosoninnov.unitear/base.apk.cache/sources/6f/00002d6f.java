package g;

import java.io.IOException;
import java.io.InterruptedIOException;
import java.util.concurrent.TimeUnit;

/* compiled from: AsyncTimeout.java */
/* loaded from: classes2.dex */
public class c extends y {

    /* renamed from: e  reason: collision with root package name */
    public static final long f6169e;

    /* renamed from: f  reason: collision with root package name */
    public static final long f6170f;

    /* renamed from: g  reason: collision with root package name */
    public static c f6171g;

    /* renamed from: h  reason: collision with root package name */
    public boolean f6172h;
    public c i;
    public long j;

    /* compiled from: AsyncTimeout.java */
    /* loaded from: classes2.dex */
    public static final class a extends Thread {
        public a() {
            super("Okio Watchdog");
            setDaemon(true);
        }

        /* JADX WARN: Code restructure failed: missing block: B:14:0x0015, code lost:
            r1.m();
         */
        @Override // java.lang.Thread, java.lang.Runnable
        /*
            Code decompiled incorrectly, please refer to instructions dump.
        */
        public void run() {
            while (true) {
                synchronized (c.class) {
                    c h2 = c.h();
                    if (h2 != null) {
                        if (h2 == c.f6171g) {
                            c.f6171g = null;
                            return;
                        }
                    }
                }
            }
        }
    }

    static {
        long millis = TimeUnit.SECONDS.toMillis(60L);
        f6169e = millis;
        f6170f = TimeUnit.MILLISECONDS.toNanos(millis);
    }

    public static c h() {
        c cVar = f6171g.i;
        if (cVar == null) {
            long nanoTime = System.nanoTime();
            c.class.wait(f6169e);
            if (f6171g.i != null || System.nanoTime() - nanoTime < f6170f) {
                return null;
            }
            return f6171g;
        }
        long nanoTime2 = cVar.j - System.nanoTime();
        if (nanoTime2 > 0) {
            long j = nanoTime2 / 1000000;
            c.class.wait(j, (int) (nanoTime2 - (1000000 * j)));
            return null;
        }
        f6171g.i = cVar.i;
        cVar.i = null;
        return cVar;
    }

    public final void i() {
        c cVar;
        if (!this.f6172h) {
            long j = this.f6223d;
            boolean z = this.f6221b;
            int i = (j > 0L ? 1 : (j == 0L ? 0 : -1));
            if (i != 0 || z) {
                this.f6172h = true;
                synchronized (c.class) {
                    if (f6171g == null) {
                        f6171g = new c();
                        new a().start();
                    }
                    long nanoTime = System.nanoTime();
                    if (i != 0 && z) {
                        this.j = Math.min(j, c() - nanoTime) + nanoTime;
                    } else if (i != 0) {
                        this.j = j + nanoTime;
                    } else if (z) {
                        this.j = c();
                    } else {
                        throw new AssertionError();
                    }
                    long j2 = this.j - nanoTime;
                    c cVar2 = f6171g;
                    while (true) {
                        cVar = cVar2.i;
                        if (cVar == null || j2 < cVar.j - nanoTime) {
                            break;
                        }
                        cVar2 = cVar;
                    }
                    this.i = cVar;
                    cVar2.i = this;
                    if (cVar2 == f6171g) {
                        c.class.notify();
                    }
                }
                return;
            }
            return;
        }
        throw new IllegalStateException("Unbalanced enter/exit");
    }

    public final void j(boolean z) {
        if (k() && z) {
            throw l(null);
        }
    }

    /* JADX WARN: Code restructure failed: missing block: B:11:0x0013, code lost:
        r2.i = r4.i;
        r4.i = null;
     */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public final boolean k() {
        boolean z = false;
        if (this.f6172h) {
            this.f6172h = false;
            synchronized (c.class) {
                c cVar = f6171g;
                while (true) {
                    if (cVar == null) {
                        z = true;
                        break;
                    }
                    c cVar2 = cVar.i;
                    if (cVar2 == this) {
                        break;
                    }
                    cVar = cVar2;
                }
            }
            return z;
        }
        return false;
    }

    public IOException l(IOException iOException) {
        InterruptedIOException interruptedIOException = new InterruptedIOException("timeout");
        if (iOException != null) {
            interruptedIOException.initCause(iOException);
        }
        return interruptedIOException;
    }

    public void m() {
    }
}