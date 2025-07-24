package b.d.b.d1.k1.b;

import android.os.Handler;
import android.os.Looper;
import android.os.SystemClock;
import b.d.b.d1.k1.c.h;
import b.g.a.f;
import com.google.common.util.concurrent.ListenableFuture;
import java.util.List;
import java.util.concurrent.AbstractExecutorService;
import java.util.concurrent.Callable;
import java.util.concurrent.Delayed;
import java.util.concurrent.Executor;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.RunnableScheduledFuture;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

/* compiled from: HandlerScheduledExecutorService.java */
/* loaded from: classes.dex */
public final class b extends AbstractExecutorService implements ScheduledExecutorService {

    /* renamed from: b  reason: collision with root package name */
    public final Handler f1517b;

    /* compiled from: HandlerScheduledExecutorService.java */
    /* loaded from: classes.dex */
    public class a extends ThreadLocal<ScheduledExecutorService> {
        /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
        @Override // java.lang.ThreadLocal
        public ScheduledExecutorService initialValue() {
            if (Looper.myLooper() == Looper.getMainLooper()) {
                return b.b.a.l();
            }
            if (Looper.myLooper() != null) {
                return new b(new Handler(Looper.myLooper()));
            }
            return null;
        }
    }

    /* compiled from: HandlerScheduledExecutorService.java */
    /* renamed from: b.d.b.d1.k1.b.b$b  reason: collision with other inner class name */
    /* loaded from: classes.dex */
    public class CallableC0020b implements Callable<Void> {

        /* renamed from: b  reason: collision with root package name */
        public final /* synthetic */ Runnable f1518b;

        public CallableC0020b(b bVar, Runnable runnable) {
            this.f1518b = runnable;
        }

        /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
        @Override // java.util.concurrent.Callable
        public Void call() {
            this.f1518b.run();
            return null;
        }
    }

    /* compiled from: HandlerScheduledExecutorService.java */
    /* loaded from: classes.dex */
    public static class c<V> implements RunnableScheduledFuture<V> {

        /* renamed from: b  reason: collision with root package name */
        public final AtomicReference<b.g.a.b<V>> f1519b = new AtomicReference<>(null);

        /* renamed from: c  reason: collision with root package name */
        public final long f1520c;

        /* renamed from: d  reason: collision with root package name */
        public final Callable<V> f1521d;

        /* renamed from: e  reason: collision with root package name */
        public final ListenableFuture<V> f1522e;

        /* compiled from: HandlerScheduledExecutorService.java */
        /* loaded from: classes.dex */
        public class a implements b.g.a.d<V> {

            /* renamed from: a  reason: collision with root package name */
            public final /* synthetic */ Handler f1523a;

            /* renamed from: b  reason: collision with root package name */
            public final /* synthetic */ Callable f1524b;

            /* compiled from: HandlerScheduledExecutorService.java */
            /* renamed from: b.d.b.d1.k1.b.b$c$a$a  reason: collision with other inner class name */
            /* loaded from: classes.dex */
            public class RunnableC0021a implements Runnable {
                public RunnableC0021a() {
                }

                @Override // java.lang.Runnable
                public void run() {
                    if (c.this.f1519b.getAndSet(null) != null) {
                        a aVar = a.this;
                        aVar.f1523a.removeCallbacks(c.this);
                    }
                }
            }

            public a(Handler handler, Callable callable) {
                this.f1523a = handler;
                this.f1524b = callable;
            }

            @Override // b.g.a.d
            public Object a(b.g.a.b<V> bVar) {
                RunnableC0021a runnableC0021a = new RunnableC0021a();
                Executor f2 = b.b.a.f();
                f<Void> fVar = bVar.f1807c;
                if (fVar != null) {
                    fVar.addListener(runnableC0021a, f2);
                }
                c.this.f1519b.set(bVar);
                return "HandlerScheduledFuture-" + this.f1524b.toString();
            }
        }

        public c(Handler handler, long j, Callable<V> callable) {
            this.f1520c = j;
            this.f1521d = callable;
            this.f1522e = b.e.a.d(new a(handler, callable));
        }

        @Override // java.util.concurrent.Future
        public boolean cancel(boolean z) {
            return this.f1522e.cancel(z);
        }

        /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object] */
        @Override // java.lang.Comparable
        public int compareTo(Delayed delayed) {
            TimeUnit timeUnit = TimeUnit.MILLISECONDS;
            return Long.compare(getDelay(timeUnit), delayed.getDelay(timeUnit));
        }

        @Override // java.util.concurrent.Future
        public V get() {
            return this.f1522e.get();
        }

        @Override // java.util.concurrent.Delayed
        public long getDelay(TimeUnit timeUnit) {
            return timeUnit.convert(this.f1520c - System.currentTimeMillis(), TimeUnit.MILLISECONDS);
        }

        @Override // java.util.concurrent.Future
        public boolean isCancelled() {
            return this.f1522e.isCancelled();
        }

        @Override // java.util.concurrent.Future
        public boolean isDone() {
            return this.f1522e.isDone();
        }

        @Override // java.util.concurrent.RunnableScheduledFuture
        public boolean isPeriodic() {
            return false;
        }

        @Override // java.util.concurrent.RunnableFuture, java.lang.Runnable
        public void run() {
            b.g.a.b<V> andSet = this.f1519b.getAndSet(null);
            if (andSet != null) {
                try {
                    andSet.a(this.f1521d.call());
                } catch (Exception e2) {
                    andSet.c(e2);
                }
            }
        }

        @Override // java.util.concurrent.Future
        public V get(long j, TimeUnit timeUnit) {
            return this.f1522e.get(j, timeUnit);
        }
    }

    static {
        new a();
    }

    public b(Handler handler) {
        this.f1517b = handler;
    }

    public final RejectedExecutionException a() {
        return new RejectedExecutionException(this.f1517b + " is shutting down");
    }

    @Override // java.util.concurrent.ExecutorService
    public boolean awaitTermination(long j, TimeUnit timeUnit) {
        throw new UnsupportedOperationException(b.class.getSimpleName() + " cannot be shut down. Use Looper.quitSafely().");
    }

    @Override // java.util.concurrent.Executor
    public void execute(Runnable runnable) {
        if (!this.f1517b.post(runnable)) {
            throw a();
        }
    }

    @Override // java.util.concurrent.ExecutorService
    public boolean isShutdown() {
        return false;
    }

    @Override // java.util.concurrent.ExecutorService
    public boolean isTerminated() {
        return false;
    }

    @Override // java.util.concurrent.ScheduledExecutorService
    public ScheduledFuture<?> schedule(Runnable runnable, long j, TimeUnit timeUnit) {
        return schedule(new CallableC0020b(this, runnable), j, timeUnit);
    }

    @Override // java.util.concurrent.ScheduledExecutorService
    public ScheduledFuture<?> scheduleAtFixedRate(Runnable runnable, long j, long j2, TimeUnit timeUnit) {
        throw new UnsupportedOperationException(b.class.getSimpleName() + " does not yet support fixed-rate scheduling.");
    }

    @Override // java.util.concurrent.ScheduledExecutorService
    public ScheduledFuture<?> scheduleWithFixedDelay(Runnable runnable, long j, long j2, TimeUnit timeUnit) {
        throw new UnsupportedOperationException(b.class.getSimpleName() + " does not yet support fixed-delay scheduling.");
    }

    @Override // java.util.concurrent.ExecutorService
    public void shutdown() {
        throw new UnsupportedOperationException(b.class.getSimpleName() + " cannot be shut down. Use Looper.quitSafely().");
    }

    @Override // java.util.concurrent.ExecutorService
    public List<Runnable> shutdownNow() {
        throw new UnsupportedOperationException(b.class.getSimpleName() + " cannot be shut down. Use Looper.quitSafely().");
    }

    @Override // java.util.concurrent.ScheduledExecutorService
    public <V> ScheduledFuture<V> schedule(Callable<V> callable, long j, TimeUnit timeUnit) {
        long convert = TimeUnit.MILLISECONDS.convert(j, timeUnit) + SystemClock.uptimeMillis();
        c cVar = new c(this.f1517b, convert, callable);
        return this.f1517b.postAtTime(cVar, convert) ? cVar : new h.b(a());
    }
}