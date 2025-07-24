package b.d.b.d1.k1.c;

import b.d.b.u0;
import com.google.common.util.concurrent.ListenableFuture;
import java.util.Objects;
import java.util.concurrent.Delayed;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;

/* compiled from: ImmediateFuture.java */
/* loaded from: classes.dex */
public abstract class h<V> implements ListenableFuture<V> {

    /* compiled from: ImmediateFuture.java */
    /* loaded from: classes.dex */
    public static class a<V> extends h<V> {

        /* renamed from: b  reason: collision with root package name */
        public final Throwable f1553b;

        public a(Throwable th) {
            this.f1553b = th;
        }

        @Override // b.d.b.d1.k1.c.h, java.util.concurrent.Future
        public V get() {
            throw new ExecutionException(this.f1553b);
        }

        public String toString() {
            return super.toString() + "[status=FAILURE, cause=[" + this.f1553b + "]]";
        }
    }

    /* compiled from: ImmediateFuture.java */
    /* loaded from: classes.dex */
    public static final class b<V> extends a<V> implements ScheduledFuture<V> {
        public b(Throwable th) {
            super(th);
        }

        /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object] */
        @Override // java.lang.Comparable
        public /* bridge */ /* synthetic */ int compareTo(Delayed delayed) {
            return -1;
        }

        @Override // java.util.concurrent.Delayed
        public long getDelay(TimeUnit timeUnit) {
            return 0L;
        }
    }

    /* compiled from: ImmediateFuture.java */
    /* loaded from: classes.dex */
    public static final class c<V> extends h<V> {

        /* renamed from: b  reason: collision with root package name */
        public static final h<Object> f1554b = new c(null);

        /* renamed from: c  reason: collision with root package name */
        public final V f1555c;

        public c(V v) {
            this.f1555c = v;
        }

        @Override // b.d.b.d1.k1.c.h, java.util.concurrent.Future
        public V get() {
            return this.f1555c;
        }

        public String toString() {
            StringBuilder sb = new StringBuilder();
            sb.append(super.toString());
            sb.append("[status=SUCCESS, result=[");
            return c.b.a.a.a.u(sb, this.f1555c, "]]");
        }
    }

    @Override // com.google.common.util.concurrent.ListenableFuture
    public void addListener(Runnable runnable, Executor executor) {
        Objects.requireNonNull(runnable);
        Objects.requireNonNull(executor);
        try {
            executor.execute(runnable);
        } catch (RuntimeException e2) {
            u0.b("ImmediateFuture", "Experienced RuntimeException while attempting to notify " + runnable + " on Executor " + executor, e2);
        }
    }

    @Override // java.util.concurrent.Future
    public boolean cancel(boolean z) {
        return false;
    }

    @Override // java.util.concurrent.Future
    public abstract V get();

    @Override // java.util.concurrent.Future
    public V get(long j, TimeUnit timeUnit) {
        Objects.requireNonNull(timeUnit);
        return get();
    }

    @Override // java.util.concurrent.Future
    public boolean isCancelled() {
        return false;
    }

    @Override // java.util.concurrent.Future
    public boolean isDone() {
        return true;
    }
}