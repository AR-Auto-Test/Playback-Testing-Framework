package b.d.b.d1.k1.c;

import com.google.common.util.concurrent.ListenableFuture;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Executor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

/* compiled from: ListFuture.java */
/* loaded from: classes.dex */
public class i<V> implements ListenableFuture<List<V>> {

    /* renamed from: b  reason: collision with root package name */
    public List<? extends ListenableFuture<? extends V>> f1556b;

    /* renamed from: c  reason: collision with root package name */
    public List<V> f1557c;

    /* renamed from: d  reason: collision with root package name */
    public final boolean f1558d;

    /* renamed from: e  reason: collision with root package name */
    public final AtomicInteger f1559e;

    /* renamed from: f  reason: collision with root package name */
    public final ListenableFuture<List<V>> f1560f;

    /* renamed from: g  reason: collision with root package name */
    public b.g.a.b<List<V>> f1561g;

    /* compiled from: ListFuture.java */
    /* loaded from: classes.dex */
    public class a implements b.g.a.d<List<V>> {
        public a() {
        }

        @Override // b.g.a.d
        public Object a(b.g.a.b<List<V>> bVar) {
            b.j.b.d.k(i.this.f1561g == null, "The result can only set once!");
            i.this.f1561g = bVar;
            return "ListFuture[" + this + "]";
        }
    }

    public i(List<? extends ListenableFuture<? extends V>> list, boolean z, Executor executor) {
        this.f1556b = list;
        this.f1557c = new ArrayList(list.size());
        this.f1558d = z;
        this.f1559e = new AtomicInteger(list.size());
        ListenableFuture<List<V>> d2 = b.e.a.d(new a());
        this.f1560f = d2;
        ((b.g.a.e) d2).f1810c.addListener(new j(this), b.b.a.f());
        if (this.f1556b.isEmpty()) {
            this.f1561g.a(new ArrayList(this.f1557c));
            return;
        }
        for (int i = 0; i < this.f1556b.size(); i++) {
            this.f1557c.add(null);
        }
        List<? extends ListenableFuture<? extends V>> list2 = this.f1556b;
        for (int i2 = 0; i2 < list2.size(); i2++) {
            ListenableFuture<? extends V> listenableFuture = list2.get(i2);
            listenableFuture.addListener(new k(this, i2, listenableFuture), executor);
        }
    }

    @Override // com.google.common.util.concurrent.ListenableFuture
    public void addListener(Runnable runnable, Executor executor) {
        this.f1560f.addListener(runnable, executor);
    }

    @Override // java.util.concurrent.Future
    public boolean cancel(boolean z) {
        List<? extends ListenableFuture<? extends V>> list = this.f1556b;
        if (list != null) {
            for (ListenableFuture<? extends V> listenableFuture : list) {
                listenableFuture.cancel(z);
            }
        }
        return this.f1560f.cancel(z);
    }

    @Override // java.util.concurrent.Future
    public Object get(long j, TimeUnit timeUnit) {
        return this.f1560f.get(j, timeUnit);
    }

    @Override // java.util.concurrent.Future
    public boolean isCancelled() {
        return this.f1560f.isCancelled();
    }

    @Override // java.util.concurrent.Future
    public boolean isDone() {
        return this.f1560f.isDone();
    }

    @Override // java.util.concurrent.Future
    public Object get() {
        List<? extends ListenableFuture<? extends V>> list = this.f1556b;
        if (list != null && !isDone()) {
            loop0: for (ListenableFuture<? extends V> listenableFuture : list) {
                while (!listenableFuture.isDone()) {
                    try {
                        listenableFuture.get();
                    } catch (Error e2) {
                        throw e2;
                    } catch (InterruptedException e3) {
                        throw e3;
                    } catch (Throwable unused) {
                        if (this.f1558d) {
                            break loop0;
                        }
                    }
                }
            }
        }
        return this.f1560f.get();
    }
}