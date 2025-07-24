package b.d.b.d1.k1.c;

import com.google.common.util.concurrent.ListenableFuture;
import java.lang.reflect.UndeclaredThrowableException;
import java.util.Objects;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.CancellationException;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

/* compiled from: ChainingListenableFuture.java */
/* loaded from: classes.dex */
public class c<I, O> extends e<O> implements Runnable {

    /* renamed from: d  reason: collision with root package name */
    public b<? super I, ? extends O> f1536d;

    /* renamed from: e  reason: collision with root package name */
    public final BlockingQueue<Boolean> f1537e = new LinkedBlockingQueue(1);

    /* renamed from: f  reason: collision with root package name */
    public final CountDownLatch f1538f = new CountDownLatch(1);

    /* renamed from: g  reason: collision with root package name */
    public ListenableFuture<? extends I> f1539g;

    /* renamed from: h  reason: collision with root package name */
    public volatile ListenableFuture<? extends O> f1540h;

    /* compiled from: ChainingListenableFuture.java */
    /* loaded from: classes.dex */
    public class a implements Runnable {

        /* renamed from: b  reason: collision with root package name */
        public final /* synthetic */ ListenableFuture f1541b;

        public a(ListenableFuture listenableFuture) {
            this.f1541b = listenableFuture;
        }

        @Override // java.lang.Runnable
        public void run() {
            try {
                try {
                    c cVar = c.this;
                    Object b2 = g.b(this.f1541b);
                    b.g.a.b<V> bVar = cVar.f1544c;
                    if (bVar != 0) {
                        bVar.a(b2);
                    }
                } catch (CancellationException unused) {
                    c.this.cancel(false);
                    c.this.f1540h = null;
                    return;
                } catch (ExecutionException e2) {
                    c.this.b(e2.getCause());
                }
                c.this.f1540h = null;
            } catch (Throwable th) {
                c.this.f1540h = null;
                throw th;
            }
        }
    }

    public c(b<? super I, ? extends O> bVar, ListenableFuture<? extends I> listenableFuture) {
        Objects.requireNonNull(bVar);
        this.f1536d = bVar;
        Objects.requireNonNull(listenableFuture);
        this.f1539g = listenableFuture;
    }

    @Override // b.d.b.d1.k1.c.e, java.util.concurrent.Future
    public boolean cancel(boolean z) {
        boolean z2 = false;
        if (this.f1543b.cancel(z)) {
            while (true) {
                try {
                    this.f1537e.put(Boolean.valueOf(z));
                    break;
                } catch (InterruptedException unused) {
                    z2 = true;
                } catch (Throwable th) {
                    if (z2) {
                        Thread.currentThread().interrupt();
                    }
                    throw th;
                }
            }
            if (z2) {
                Thread.currentThread().interrupt();
            }
            ListenableFuture<? extends I> listenableFuture = this.f1539g;
            if (listenableFuture != null) {
                listenableFuture.cancel(z);
            }
            ListenableFuture<? extends O> listenableFuture2 = this.f1540h;
            if (listenableFuture2 != null) {
                listenableFuture2.cancel(z);
            }
            return true;
        }
        return false;
    }

    public final <E> E d(BlockingQueue<E> blockingQueue) {
        E take;
        boolean z = false;
        while (true) {
            try {
                take = blockingQueue.take();
                break;
            } catch (InterruptedException unused) {
                z = true;
            } catch (Throwable th) {
                if (z) {
                    Thread.currentThread().interrupt();
                }
                throw th;
            }
        }
        if (z) {
            Thread.currentThread().interrupt();
        }
        return take;
    }

    @Override // b.d.b.d1.k1.c.e, java.util.concurrent.Future
    public O get() {
        if (!isDone()) {
            ListenableFuture<? extends I> listenableFuture = this.f1539g;
            if (listenableFuture != null) {
                listenableFuture.get();
            }
            this.f1538f.await();
            ListenableFuture<? extends O> listenableFuture2 = this.f1540h;
            if (listenableFuture2 != null) {
                listenableFuture2.get();
            }
        }
        return (O) super.get();
    }

    /* JADX DEBUG: Another duplicated slice has different insns count: {[GOTO]}, finally: {[GOTO, IPUT, IPUT, IGET, INVOKE] complete} */
    /* JADX DEBUG: Type inference failed for r2v5. Raw type applied. Possible types: java.util.concurrent.BlockingQueue<java.lang.Boolean>, java.util.concurrent.BlockingQueue<E> */
    @Override // java.lang.Runnable
    public void run() {
        ListenableFuture<? extends O> apply;
        try {
        } catch (Exception e2) {
            b.g.a.b<V> bVar = this.f1544c;
            if (bVar != 0) {
                bVar.c(e2);
            }
        }
        try {
            try {
                try {
                    apply = this.f1536d.apply(g.b(this.f1539g));
                    this.f1540h = apply;
                } catch (Error e3) {
                    b.g.a.b<V> bVar2 = this.f1544c;
                    if (bVar2 != 0) {
                        bVar2.c(e3);
                    }
                } catch (UndeclaredThrowableException e4) {
                    b(e4.getCause());
                }
            } catch (CancellationException unused) {
                cancel(false);
            } catch (ExecutionException e5) {
                b(e5.getCause());
            }
            if (isCancelled()) {
                apply.cancel(((Boolean) d(this.f1537e)).booleanValue());
                this.f1540h = null;
                this.f1536d = null;
                this.f1539g = null;
                this.f1538f.countDown();
                return;
            }
            apply.addListener(new a(apply), b.b.a.f());
            this.f1536d = null;
            this.f1539g = null;
            this.f1538f.countDown();
        } catch (Throwable th) {
            this.f1536d = null;
            this.f1539g = null;
            this.f1538f.countDown();
            throw th;
        }
    }

    @Override // b.d.b.d1.k1.c.e, java.util.concurrent.Future
    public O get(long j, TimeUnit timeUnit) {
        if (!isDone()) {
            TimeUnit timeUnit2 = TimeUnit.NANOSECONDS;
            if (timeUnit != timeUnit2) {
                j = timeUnit2.convert(j, timeUnit);
                timeUnit = timeUnit2;
            }
            ListenableFuture<? extends I> listenableFuture = this.f1539g;
            if (listenableFuture != null) {
                long nanoTime = System.nanoTime();
                listenableFuture.get(j, timeUnit);
                j -= Math.max(0L, System.nanoTime() - nanoTime);
            }
            long nanoTime2 = System.nanoTime();
            if (this.f1538f.await(j, timeUnit)) {
                j -= Math.max(0L, System.nanoTime() - nanoTime2);
                ListenableFuture<? extends O> listenableFuture2 = this.f1540h;
                if (listenableFuture2 != null) {
                    listenableFuture2.get(j, timeUnit);
                }
            } else {
                throw new TimeoutException();
            }
        }
        return (O) super.get(j, timeUnit);
    }
}