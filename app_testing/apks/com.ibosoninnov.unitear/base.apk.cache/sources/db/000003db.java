package b.g.a;

import b.g.a.a;
import com.google.common.util.concurrent.ListenableFuture;
import java.lang.ref.WeakReference;
import java.util.concurrent.Executor;
import java.util.concurrent.TimeUnit;

/* compiled from: CallbackToFutureAdapter.java */
/* loaded from: classes.dex */
public final class e<T> implements ListenableFuture<T> {

    /* renamed from: b  reason: collision with root package name */
    public final WeakReference<b<T>> f1809b;

    /* renamed from: c  reason: collision with root package name */
    public final b.g.a.a<T> f1810c = new a();

    /* compiled from: CallbackToFutureAdapter.java */
    /* loaded from: classes.dex */
    public class a extends b.g.a.a<T> {
        public a() {
        }

        @Override // b.g.a.a
        public String f() {
            b<T> bVar = e.this.f1809b.get();
            return bVar == null ? "Completer object has been garbage collected, future will fail soon" : c.b.a.a.a.u(c.b.a.a.a.x("tag=["), bVar.f1805a, "]");
        }
    }

    public e(b<T> bVar) {
        this.f1809b = new WeakReference<>(bVar);
    }

    @Override // com.google.common.util.concurrent.ListenableFuture
    public void addListener(Runnable runnable, Executor executor) {
        this.f1810c.addListener(runnable, executor);
    }

    @Override // java.util.concurrent.Future
    public boolean cancel(boolean z) {
        b<T> bVar = this.f1809b.get();
        boolean cancel = this.f1810c.cancel(z);
        if (cancel && bVar != null) {
            bVar.f1805a = null;
            bVar.f1806b = null;
            bVar.f1807c.h(null);
        }
        return cancel;
    }

    @Override // java.util.concurrent.Future
    public T get() {
        return this.f1810c.get();
    }

    @Override // java.util.concurrent.Future
    public boolean isCancelled() {
        return this.f1810c.f1785f instanceof a.c;
    }

    @Override // java.util.concurrent.Future
    public boolean isDone() {
        return this.f1810c.isDone();
    }

    public String toString() {
        return this.f1810c.toString();
    }

    @Override // java.util.concurrent.Future
    public T get(long j, TimeUnit timeUnit) {
        return this.f1810c.get(j, timeUnit);
    }
}