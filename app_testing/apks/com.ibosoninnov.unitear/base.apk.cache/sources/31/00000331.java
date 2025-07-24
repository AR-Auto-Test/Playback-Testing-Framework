package b.d.b.d1.k1.c;

import b.d.b.d1.k1.c.h;
import com.google.common.util.concurrent.ListenableFuture;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.Future;

/* compiled from: Futures.java */
/* loaded from: classes.dex */
public final class g {

    /* renamed from: a  reason: collision with root package name */
    public static final b.c.a.c.a<?, ?> f1547a = new a();

    /* compiled from: Futures.java */
    /* loaded from: classes.dex */
    public class a implements b.c.a.c.a<Object, Object> {
        @Override // b.c.a.c.a
        public Object apply(Object obj) {
            return obj;
        }
    }

    /* compiled from: Futures.java */
    /* loaded from: classes.dex */
    public class b implements b.d.b.d1.k1.c.d<I> {

        /* renamed from: a  reason: collision with root package name */
        public final /* synthetic */ b.g.a.b f1548a;

        /* renamed from: b  reason: collision with root package name */
        public final /* synthetic */ b.c.a.c.a f1549b;

        public b(b.g.a.b bVar, b.c.a.c.a aVar) {
            this.f1548a = bVar;
            this.f1549b = aVar;
        }

        @Override // b.d.b.d1.k1.c.d
        public void onFailure(Throwable th) {
            this.f1548a.c(th);
        }

        @Override // b.d.b.d1.k1.c.d
        public void onSuccess(I i) {
            try {
                this.f1548a.a(this.f1549b.apply(i));
            } catch (Throwable th) {
                this.f1548a.c(th);
            }
        }
    }

    /* compiled from: Futures.java */
    /* loaded from: classes.dex */
    public class c implements Runnable {

        /* renamed from: b  reason: collision with root package name */
        public final /* synthetic */ ListenableFuture f1550b;

        public c(ListenableFuture listenableFuture) {
            this.f1550b = listenableFuture;
        }

        @Override // java.lang.Runnable
        public void run() {
            this.f1550b.cancel(true);
        }
    }

    /* compiled from: Futures.java */
    /* loaded from: classes.dex */
    public static final class d<V> implements Runnable {

        /* renamed from: b  reason: collision with root package name */
        public final Future<V> f1551b;

        /* renamed from: c  reason: collision with root package name */
        public final b.d.b.d1.k1.c.d<? super V> f1552c;

        public d(Future<V> future, b.d.b.d1.k1.c.d<? super V> dVar) {
            this.f1551b = future;
            this.f1552c = dVar;
        }

        @Override // java.lang.Runnable
        public void run() {
            try {
                this.f1552c.onSuccess(g.a(this.f1551b));
            } catch (Error e2) {
                e = e2;
                this.f1552c.onFailure(e);
            } catch (RuntimeException e3) {
                e = e3;
                this.f1552c.onFailure(e);
            } catch (ExecutionException e4) {
                this.f1552c.onFailure(e4.getCause());
            }
        }

        public String toString() {
            return d.class.getSimpleName() + "," + this.f1552c;
        }
    }

    public static <V> V a(Future<V> future) {
        boolean isDone = future.isDone();
        b.j.b.d.k(isDone, "Future was expected to be done, " + future);
        return (V) b(future);
    }

    public static <V> V b(Future<V> future) {
        V v;
        boolean z = false;
        while (true) {
            try {
                v = future.get();
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
        return v;
    }

    public static <V> ListenableFuture<V> c(V v) {
        if (v == null) {
            return h.c.f1554b;
        }
        return new h.c(v);
    }

    public static <V> ListenableFuture<V> d(final ListenableFuture<V> listenableFuture) {
        Objects.requireNonNull(listenableFuture);
        return listenableFuture.isDone() ? listenableFuture : b.e.a.d(new b.g.a.d() { // from class: b.d.b.d1.k1.c.a
            @Override // b.g.a.d
            public final Object a(b.g.a.b bVar) {
                ListenableFuture listenableFuture2 = ListenableFuture.this;
                g.f(false, listenableFuture2, g.f1547a, bVar, b.b.a.f());
                return "nonCancellationPropagating[" + listenableFuture2 + "]";
            }
        });
    }

    public static <V> void e(ListenableFuture<V> listenableFuture, b.g.a.b<V> bVar) {
        f(true, listenableFuture, f1547a, bVar, b.b.a.f());
    }

    public static <I, O> void f(boolean z, ListenableFuture<I> listenableFuture, b.c.a.c.a<? super I, ? extends O> aVar, b.g.a.b<O> bVar, Executor executor) {
        Objects.requireNonNull(listenableFuture);
        Objects.requireNonNull(aVar);
        Objects.requireNonNull(bVar);
        Objects.requireNonNull(executor);
        listenableFuture.addListener(new d(listenableFuture, new b(bVar, aVar)), executor);
        if (z) {
            c cVar = new c(listenableFuture);
            Executor f2 = b.b.a.f();
            b.g.a.f<Void> fVar = bVar.f1807c;
            if (fVar != null) {
                fVar.addListener(cVar, f2);
            }
        }
    }

    public static <V> ListenableFuture<List<V>> g(Collection<? extends ListenableFuture<? extends V>> collection) {
        return new i(new ArrayList(collection), false, b.b.a.f());
    }
}