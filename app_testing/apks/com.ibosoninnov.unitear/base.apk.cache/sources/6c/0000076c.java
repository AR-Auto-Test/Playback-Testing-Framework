package c.c.a.m.v.e0;

import android.os.Process;
import android.os.StrictMode;
import android.util.Log;
import java.util.Collection;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.TimeUnit;

/* compiled from: GlideExecutor.java */
/* loaded from: classes.dex */
public final class a implements ExecutorService {

    /* renamed from: b  reason: collision with root package name */
    public static final long f3680b = TimeUnit.SECONDS.toMillis(10);

    /* renamed from: c  reason: collision with root package name */
    public static volatile int f3681c;

    /* renamed from: d  reason: collision with root package name */
    public final ExecutorService f3682d;

    /* compiled from: GlideExecutor.java */
    /* renamed from: c.c.a.m.v.e0.a$a  reason: collision with other inner class name */
    /* loaded from: classes.dex */
    public static final class ThreadFactoryC0069a implements ThreadFactory {

        /* renamed from: a  reason: collision with root package name */
        public final String f3683a;

        /* renamed from: b  reason: collision with root package name */
        public final boolean f3684b;

        /* renamed from: c  reason: collision with root package name */
        public int f3685c;

        /* compiled from: GlideExecutor.java */
        /* renamed from: c.c.a.m.v.e0.a$a$a  reason: collision with other inner class name */
        /* loaded from: classes.dex */
        public class C0070a extends Thread {
            public C0070a(Runnable runnable, String str) {
                super(runnable, str);
            }

            @Override // java.lang.Thread, java.lang.Runnable
            public void run() {
                Process.setThreadPriority(9);
                if (ThreadFactoryC0069a.this.f3684b) {
                    StrictMode.setThreadPolicy(new StrictMode.ThreadPolicy.Builder().detectNetwork().penaltyDeath().build());
                }
                try {
                    super.run();
                } catch (Throwable th) {
                    Objects.requireNonNull(ThreadFactoryC0069a.this);
                    ((b.C0071a) b.f3688b).a(th);
                }
            }
        }

        public ThreadFactoryC0069a(String str, b bVar, boolean z) {
            this.f3683a = str;
            this.f3684b = z;
        }

        @Override // java.util.concurrent.ThreadFactory
        public synchronized Thread newThread(Runnable runnable) {
            C0070a c0070a;
            c0070a = new C0070a(runnable, "glide-" + this.f3683a + "-thread-" + this.f3685c);
            this.f3685c = this.f3685c + 1;
            return c0070a;
        }
    }

    /* compiled from: GlideExecutor.java */
    /* loaded from: classes.dex */
    public interface b {

        /* renamed from: a  reason: collision with root package name */
        public static final b f3687a;

        /* renamed from: b  reason: collision with root package name */
        public static final b f3688b;

        /* compiled from: GlideExecutor.java */
        /* renamed from: c.c.a.m.v.e0.a$b$a  reason: collision with other inner class name */
        /* loaded from: classes.dex */
        public class C0071a implements b {
            public void a(Throwable th) {
                if (Log.isLoggable("GlideExecutor", 6)) {
                    Log.e("GlideExecutor", "Request threw uncaught throwable", th);
                }
            }
        }

        static {
            C0071a c0071a = new C0071a();
            f3687a = c0071a;
            f3688b = c0071a;
        }
    }

    public a(ExecutorService executorService) {
        this.f3682d = executorService;
    }

    public static int a() {
        if (f3681c == 0) {
            f3681c = Math.min(4, Runtime.getRuntime().availableProcessors());
        }
        return f3681c;
    }

    @Override // java.util.concurrent.ExecutorService
    public boolean awaitTermination(long j, TimeUnit timeUnit) {
        return this.f3682d.awaitTermination(j, timeUnit);
    }

    @Override // java.util.concurrent.Executor
    public void execute(Runnable runnable) {
        this.f3682d.execute(runnable);
    }

    @Override // java.util.concurrent.ExecutorService
    public <T> List<Future<T>> invokeAll(Collection<? extends Callable<T>> collection) {
        return this.f3682d.invokeAll(collection);
    }

    @Override // java.util.concurrent.ExecutorService
    public <T> T invokeAny(Collection<? extends Callable<T>> collection) {
        return (T) this.f3682d.invokeAny(collection);
    }

    @Override // java.util.concurrent.ExecutorService
    public boolean isShutdown() {
        return this.f3682d.isShutdown();
    }

    @Override // java.util.concurrent.ExecutorService
    public boolean isTerminated() {
        return this.f3682d.isTerminated();
    }

    @Override // java.util.concurrent.ExecutorService
    public void shutdown() {
        this.f3682d.shutdown();
    }

    @Override // java.util.concurrent.ExecutorService
    public List<Runnable> shutdownNow() {
        return this.f3682d.shutdownNow();
    }

    @Override // java.util.concurrent.ExecutorService
    public Future<?> submit(Runnable runnable) {
        return this.f3682d.submit(runnable);
    }

    public String toString() {
        return this.f3682d.toString();
    }

    @Override // java.util.concurrent.ExecutorService
    public <T> List<Future<T>> invokeAll(Collection<? extends Callable<T>> collection, long j, TimeUnit timeUnit) {
        return this.f3682d.invokeAll(collection, j, timeUnit);
    }

    @Override // java.util.concurrent.ExecutorService
    public <T> T invokeAny(Collection<? extends Callable<T>> collection, long j, TimeUnit timeUnit) {
        return (T) this.f3682d.invokeAny(collection, j, timeUnit);
    }

    @Override // java.util.concurrent.ExecutorService
    public <T> Future<T> submit(Runnable runnable, T t) {
        return this.f3682d.submit(runnable, t);
    }

    @Override // java.util.concurrent.ExecutorService
    public <T> Future<T> submit(Callable<T> callable) {
        return this.f3682d.submit(callable);
    }
}