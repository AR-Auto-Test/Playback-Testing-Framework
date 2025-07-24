package b.d.b.d1.k1.b;

import java.util.concurrent.Executor;

/* compiled from: DirectExecutor.java */
/* loaded from: classes.dex */
public final class a implements Executor {

    /* renamed from: b  reason: collision with root package name */
    public static volatile a f1516b;

    @Override // java.util.concurrent.Executor
    public void execute(Runnable runnable) {
        runnable.run();
    }
}