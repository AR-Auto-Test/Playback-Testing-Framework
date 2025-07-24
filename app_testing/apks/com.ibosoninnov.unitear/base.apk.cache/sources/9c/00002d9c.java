package h.a.a;

import java.lang.Thread;

/* compiled from: SafeRunnable.java */
/* loaded from: classes2.dex */
public abstract class k implements Runnable {

    /* renamed from: b  reason: collision with root package name */
    public final c f6252b;

    public k(c cVar) {
        this.f6252b = cVar;
    }

    public abstract void a();

    @Override // java.lang.Runnable
    public final void run() {
        try {
            if (this.f6252b.f6232h.d()) {
                return;
            }
            a();
        } catch (Throwable th) {
            Thread.UncaughtExceptionHandler defaultUncaughtExceptionHandler = Thread.getDefaultUncaughtExceptionHandler();
            if (defaultUncaughtExceptionHandler != null) {
                defaultUncaughtExceptionHandler.uncaughtException(Thread.currentThread(), th);
            }
            throw th;
        }
    }
}