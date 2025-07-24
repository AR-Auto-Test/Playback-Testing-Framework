package b.d.b.d1.k1.c;

import com.google.common.util.concurrent.ListenableFuture;
import java.util.Objects;
import java.util.concurrent.Executor;
import java.util.concurrent.TimeUnit;

/* compiled from: FutureChain.java */
/* loaded from: classes.dex */
public class e<V> implements ListenableFuture<V> {

    /* renamed from: b  reason: collision with root package name */
    public final ListenableFuture<V> f1543b;

    /* renamed from: c  reason: collision with root package name */
    public b.g.a.b<V> f1544c;

    /* compiled from: FutureChain.java */
    /* loaded from: classes.dex */
    public class a implements b.g.a.d<V> {
        public a() {
        }

        @Override // b.g.a.d
        public Object a(b.g.a.b<V> bVar) {
            b.j.b.d.k(e.this.f1544c == null, "The result can only set once!");
            e.this.f1544c = bVar;
            StringBuilder x = c.b.a.a.a.x("FutureChain[");
            x.append(e.this);
            x.append("]");
            return x.toString();
        }
    }

    public e(ListenableFuture<V> listenableFuture) {
        Objects.requireNonNull(listenableFuture);
        this.f1543b = listenableFuture;
    }

    public static <V> e<V> a(ListenableFuture<V> listenableFuture) {
        return listenableFuture instanceof e ? (e) listenableFuture : new e<>(listenableFuture);
    }

    @Override // com.google.common.util.concurrent.ListenableFuture
    public void addListener(Runnable runnable, Executor executor) {
        this.f1543b.addListener(runnable, executor);
    }

    public boolean b(Throwable th) {
        b.g.a.b<V> bVar = this.f1544c;
        if (bVar != null) {
            return bVar.c(th);
        }
        return false;
    }

    public final <T> e<T> c(b<? super V, T> bVar, Executor executor) {
        c cVar = new c(bVar, this);
        this.f1543b.addListener(cVar, executor);
        return cVar;
    }

    @Override // java.util.concurrent.Future
    public boolean cancel(boolean z) {
        return this.f1543b.cancel(z);
    }

    @Override // java.util.concurrent.Future
    public V get() {
        return this.f1543b.get();
    }

    @Override // java.util.concurrent.Future
    public boolean isCancelled() {
        return this.f1543b.isCancelled();
    }

    @Override // java.util.concurrent.Future
    public boolean isDone() {
        return this.f1543b.isDone();
    }

    @Override // java.util.concurrent.Future
    public V get(long j, TimeUnit timeUnit) {
        return this.f1543b.get(j, timeUnit);
    }

    public e() {
        this.f1543b = b.e.a.d(new a());
    }
}