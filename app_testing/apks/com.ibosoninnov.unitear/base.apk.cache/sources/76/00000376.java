package b.d.b;

import java.util.Locale;
import java.util.Objects;
import java.util.concurrent.Executor;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

/* compiled from: CameraExecutor.java */
/* loaded from: classes.dex */
public class g0 implements Executor {

    /* renamed from: b  reason: collision with root package name */
    public static final ThreadFactory f1616b = new a();

    /* renamed from: c  reason: collision with root package name */
    public final Object f1617c = new Object();

    /* renamed from: d  reason: collision with root package name */
    public ThreadPoolExecutor f1618d = a();

    /* compiled from: CameraExecutor.java */
    /* loaded from: classes.dex */
    public class a implements ThreadFactory {

        /* renamed from: a  reason: collision with root package name */
        public final AtomicInteger f1619a = new AtomicInteger(0);

        @Override // java.util.concurrent.ThreadFactory
        public Thread newThread(Runnable runnable) {
            Thread thread = new Thread(runnable);
            thread.setName(String.format(Locale.US, "CameraX-core_camera_%d", Integer.valueOf(this.f1619a.getAndIncrement())));
            return thread;
        }
    }

    public static ThreadPoolExecutor a() {
        return new ThreadPoolExecutor(1, 1, 0L, TimeUnit.MILLISECONDS, new LinkedBlockingQueue(), f1616b);
    }

    public void b(b.d.b.d1.y yVar) {
        ThreadPoolExecutor threadPoolExecutor;
        Objects.requireNonNull(yVar);
        synchronized (this.f1617c) {
            if (this.f1618d.isShutdown()) {
                this.f1618d = a();
            }
            threadPoolExecutor = this.f1618d;
        }
        int max = Math.max(1, ((b.d.a.e.p0) yVar).a().size());
        threadPoolExecutor.setMaximumPoolSize(max);
        threadPoolExecutor.setCorePoolSize(max);
    }

    @Override // java.util.concurrent.Executor
    public void execute(Runnable runnable) {
        Objects.requireNonNull(runnable);
        synchronized (this.f1617c) {
            this.f1618d.execute(runnable);
        }
    }
}