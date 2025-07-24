package c.c.a.m.v;

import android.os.Process;
import c.c.a.m.v.q;
import java.lang.ref.ReferenceQueue;
import java.lang.ref.WeakReference;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.Executor;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadFactory;

/* compiled from: ActiveResources.java */
/* loaded from: classes.dex */
public final class a {

    /* renamed from: a  reason: collision with root package name */
    public final boolean f3587a;

    /* renamed from: b  reason: collision with root package name */
    public final Executor f3588b;

    /* renamed from: c  reason: collision with root package name */
    public final Map<c.c.a.m.m, b> f3589c;

    /* renamed from: d  reason: collision with root package name */
    public final ReferenceQueue<q<?>> f3590d;

    /* renamed from: e  reason: collision with root package name */
    public q.a f3591e;

    /* compiled from: ActiveResources.java */
    /* renamed from: c.c.a.m.v.a$a  reason: collision with other inner class name */
    /* loaded from: classes.dex */
    public class ThreadFactoryC0066a implements ThreadFactory {

        /* compiled from: ActiveResources.java */
        /* renamed from: c.c.a.m.v.a$a$a  reason: collision with other inner class name */
        /* loaded from: classes.dex */
        public class RunnableC0067a implements Runnable {

            /* renamed from: b  reason: collision with root package name */
            public final /* synthetic */ Runnable f3592b;

            public RunnableC0067a(ThreadFactoryC0066a threadFactoryC0066a, Runnable runnable) {
                this.f3592b = runnable;
            }

            @Override // java.lang.Runnable
            public void run() {
                Process.setThreadPriority(10);
                this.f3592b.run();
            }
        }

        @Override // java.util.concurrent.ThreadFactory
        public Thread newThread(Runnable runnable) {
            return new Thread(new RunnableC0067a(this, runnable), "glide-active-resources");
        }
    }

    /* compiled from: ActiveResources.java */
    /* loaded from: classes.dex */
    public static final class b extends WeakReference<q<?>> {

        /* renamed from: a  reason: collision with root package name */
        public final c.c.a.m.m f3593a;

        /* renamed from: b  reason: collision with root package name */
        public final boolean f3594b;

        /* renamed from: c  reason: collision with root package name */
        public w<?> f3595c;

        public b(c.c.a.m.m mVar, q<?> qVar, ReferenceQueue<? super q<?>> referenceQueue, boolean z) {
            super(qVar, referenceQueue);
            w<?> wVar;
            Objects.requireNonNull(mVar, "Argument must not be null");
            this.f3593a = mVar;
            if (qVar.f3780b && z) {
                wVar = qVar.f3782d;
                Objects.requireNonNull(wVar, "Argument must not be null");
            } else {
                wVar = null;
            }
            this.f3595c = wVar;
            this.f3594b = qVar.f3780b;
        }
    }

    public a(boolean z) {
        ExecutorService newSingleThreadExecutor = Executors.newSingleThreadExecutor(new ThreadFactoryC0066a());
        this.f3589c = new HashMap();
        this.f3590d = new ReferenceQueue<>();
        this.f3587a = z;
        this.f3588b = newSingleThreadExecutor;
        newSingleThreadExecutor.execute(new c.c.a.m.v.b(this));
    }

    public synchronized void a(c.c.a.m.m mVar, q<?> qVar) {
        b put = this.f3589c.put(mVar, new b(mVar, qVar, this.f3590d, this.f3587a));
        if (put != null) {
            put.f3595c = null;
            put.clear();
        }
    }

    public void b(b bVar) {
        w<?> wVar;
        synchronized (this) {
            this.f3589c.remove(bVar.f3593a);
            if (bVar.f3594b && (wVar = bVar.f3595c) != null) {
                this.f3591e.a(bVar.f3593a, new q<>(wVar, true, false, bVar.f3593a, this.f3591e));
            }
        }
    }
}