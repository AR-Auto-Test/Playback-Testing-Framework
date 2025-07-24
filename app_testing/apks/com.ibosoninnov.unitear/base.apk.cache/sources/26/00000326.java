package b.d.b.d1.k1.b;

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.Objects;
import java.util.concurrent.Executor;
import java.util.concurrent.RejectedExecutionException;

/* compiled from: SequentialExecutor.java */
/* loaded from: classes.dex */
public final class d implements Executor {

    /* renamed from: c  reason: collision with root package name */
    public final Executor f1529c;

    /* renamed from: b  reason: collision with root package name */
    public final Deque<Runnable> f1528b = new ArrayDeque();

    /* renamed from: d  reason: collision with root package name */
    public final b f1530d = new b();

    /* renamed from: e  reason: collision with root package name */
    public int f1531e = 1;

    /* renamed from: f  reason: collision with root package name */
    public long f1532f = 0;

    /* compiled from: SequentialExecutor.java */
    /* loaded from: classes.dex */
    public class a implements Runnable {

        /* renamed from: b  reason: collision with root package name */
        public final /* synthetic */ Runnable f1533b;

        public a(d dVar, Runnable runnable) {
            this.f1533b = runnable;
        }

        @Override // java.lang.Runnable
        public void run() {
            this.f1533b.run();
        }
    }

    /* compiled from: SequentialExecutor.java */
    /* loaded from: classes.dex */
    public final class b implements Runnable {
        public b() {
        }

        /* JADX DEBUG: Another duplicated slice has different insns count: {[IF]}, finally: {[IF, INVOKE, INVOKE] complete} */
        /* JADX WARN: Code restructure failed: missing block: B:18:0x0037, code lost:
            if (r1 == false) goto L36;
         */
        /* JADX WARN: Code restructure failed: missing block: B:19:0x0039, code lost:
            java.lang.Thread.currentThread().interrupt();
         */
        /* JADX WARN: Code restructure failed: missing block: B:20:0x0040, code lost:
            return;
         */
        /* JADX WARN: Code restructure failed: missing block: B:23:0x0046, code lost:
            r1 = r1 | java.lang.Thread.interrupted();
         */
        /* JADX WARN: Code restructure failed: missing block: B:24:0x0047, code lost:
            r4.run();
         */
        /* JADX WARN: Code restructure failed: missing block: B:26:0x004b, code lost:
            r2 = move-exception;
         */
        /* JADX WARN: Code restructure failed: missing block: B:27:0x004c, code lost:
            b.d.b.u0.b("SequentialExecutor", "Exception while executing runnable " + r4, r2);
         */
        /* JADX WARN: Code restructure failed: missing block: B:45:?, code lost:
            return;
         */
        /*
            Code decompiled incorrectly, please refer to instructions dump.
        */
        public final void a() {
            boolean z = false;
            boolean z2 = false;
            while (true) {
                try {
                    synchronized (d.this.f1528b) {
                        if (!z) {
                            d dVar = d.this;
                            if (dVar.f1531e != 4) {
                                dVar.f1532f++;
                                dVar.f1531e = 4;
                                z = true;
                            }
                        }
                        Runnable poll = d.this.f1528b.poll();
                        if (poll == null) {
                            d.this.f1531e = 1;
                        }
                    }
                    if (z2) {
                        return;
                    }
                    return;
                } finally {
                    if (z2) {
                        Thread.currentThread().interrupt();
                    }
                }
            }
        }

        @Override // java.lang.Runnable
        public void run() {
            try {
                a();
            } catch (Error e2) {
                synchronized (d.this.f1528b) {
                    d.this.f1531e = 1;
                    throw e2;
                }
            }
        }
    }

    public d(Executor executor) {
        Objects.requireNonNull(executor);
        this.f1529c = executor;
    }

    @Override // java.util.concurrent.Executor
    public void execute(Runnable runnable) {
        Objects.requireNonNull(runnable);
        synchronized (this.f1528b) {
            int i = this.f1531e;
            if (i != 4 && i != 3) {
                long j = this.f1532f;
                a aVar = new a(this, runnable);
                this.f1528b.add(aVar);
                this.f1531e = 2;
                try {
                    this.f1529c.execute(this.f1530d);
                    if (this.f1531e != 2) {
                        return;
                    }
                    synchronized (this.f1528b) {
                        if (this.f1532f == j && this.f1531e == 2) {
                            this.f1531e = 3;
                        }
                    }
                    return;
                } catch (Error | RuntimeException e2) {
                    synchronized (this.f1528b) {
                        int i2 = this.f1531e;
                        if ((i2 == 1 || i2 == 2) && this.f1528b.removeLastOccurrence(aVar)) {
                            r0 = true;
                        }
                        if (!(e2 instanceof RejectedExecutionException) || r0) {
                            throw e2;
                        }
                    }
                    return;
                }
            }
            this.f1528b.add(runnable);
        }
    }
}