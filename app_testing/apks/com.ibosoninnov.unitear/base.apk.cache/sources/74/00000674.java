package c.a.a;

import android.os.Handler;
import android.os.Looper;
import java.util.LinkedHashSet;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;
import java.util.concurrent.FutureTask;

/* compiled from: LottieTask.java */
/* loaded from: classes.dex */
public class r<T> {

    /* renamed from: a  reason: collision with root package name */
    public static Executor f3125a = Executors.newCachedThreadPool();

    /* renamed from: b  reason: collision with root package name */
    public final Set<l<T>> f3126b = new LinkedHashSet(1);

    /* renamed from: c  reason: collision with root package name */
    public final Set<l<Throwable>> f3127c = new LinkedHashSet(1);

    /* renamed from: d  reason: collision with root package name */
    public final Handler f3128d = new Handler(Looper.getMainLooper());

    /* renamed from: e  reason: collision with root package name */
    public volatile p<T> f3129e = null;

    /* compiled from: LottieTask.java */
    /* loaded from: classes.dex */
    public class a extends FutureTask<p<T>> {
        public a(Callable<p<T>> callable) {
            super(callable);
        }

        @Override // java.util.concurrent.FutureTask
        public void done() {
            if (isCancelled()) {
                return;
            }
            try {
                r.this.c(get());
            } catch (InterruptedException | ExecutionException e2) {
                r.this.c(new p<>(e2));
            }
        }
    }

    public r(Callable<p<T>> callable) {
        f3125a.execute(new a(callable));
    }

    public synchronized r<T> a(l<Throwable> lVar) {
        if (this.f3129e != null && this.f3129e.f3123b != null) {
            lVar.a(this.f3129e.f3123b);
        }
        this.f3127c.add(lVar);
        return this;
    }

    public synchronized r<T> b(l<T> lVar) {
        if (this.f3129e != null && this.f3129e.f3122a != null) {
            lVar.a(this.f3129e.f3122a);
        }
        this.f3126b.add(lVar);
        return this;
    }

    public final void c(p<T> pVar) {
        if (this.f3129e == null) {
            this.f3129e = pVar;
            this.f3128d.post(new q(this));
            return;
        }
        throw new IllegalStateException("A task may only be set once.");
    }
}