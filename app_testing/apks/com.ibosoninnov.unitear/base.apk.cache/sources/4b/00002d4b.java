package f;

import f.g0.f.g;
import java.lang.ref.Reference;
import java.util.ArrayDeque;
import java.util.Deque;
import java.util.List;
import java.util.concurrent.Executor;
import java.util.concurrent.SynchronousQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

/* compiled from: ConnectionPool.java */
/* loaded from: classes2.dex */
public final class h {

    /* renamed from: a  reason: collision with root package name */
    public static final Executor f6045a;

    /* renamed from: b  reason: collision with root package name */
    public final int f6046b;

    /* renamed from: c  reason: collision with root package name */
    public final long f6047c;

    /* renamed from: d  reason: collision with root package name */
    public final Runnable f6048d;

    /* renamed from: e  reason: collision with root package name */
    public final Deque<f.g0.f.c> f6049e;

    /* renamed from: f  reason: collision with root package name */
    public final f.g0.f.d f6050f;

    /* renamed from: g  reason: collision with root package name */
    public boolean f6051g;

    /* compiled from: ConnectionPool.java */
    /* loaded from: classes2.dex */
    public class a implements Runnable {
        public a() {
        }

        @Override // java.lang.Runnable
        public void run() {
            long j;
            while (true) {
                h hVar = h.this;
                long nanoTime = System.nanoTime();
                synchronized (hVar) {
                    f.g0.f.c cVar = null;
                    long j2 = Long.MIN_VALUE;
                    int i = 0;
                    int i2 = 0;
                    for (f.g0.f.c cVar2 : hVar.f6049e) {
                        if (hVar.a(cVar2, nanoTime) > 0) {
                            i2++;
                        } else {
                            i++;
                            long j3 = nanoTime - cVar2.o;
                            if (j3 > j2) {
                                cVar = cVar2;
                                j2 = j3;
                            }
                        }
                    }
                    j = hVar.f6047c;
                    if (j2 < j && i <= hVar.f6046b) {
                        if (i > 0) {
                            j -= j2;
                        } else if (i2 <= 0) {
                            hVar.f6051g = false;
                            j = -1;
                        }
                    }
                    hVar.f6049e.remove(cVar);
                    f.g0.c.g(cVar.f5793e);
                    j = 0;
                }
                if (j == -1) {
                    return;
                }
                if (j > 0) {
                    long j4 = j / 1000000;
                    long j5 = j - (1000000 * j4);
                    synchronized (h.this) {
                        try {
                            h.this.wait(j4, (int) j5);
                        } catch (InterruptedException unused) {
                        }
                    }
                }
            }
        }
    }

    static {
        TimeUnit timeUnit = TimeUnit.SECONDS;
        SynchronousQueue synchronousQueue = new SynchronousQueue();
        byte[] bArr = f.g0.c.f5773a;
        f6045a = new ThreadPoolExecutor(0, Integer.MAX_VALUE, 60L, timeUnit, synchronousQueue, new f.g0.d("OkHttp ConnectionPool", true));
    }

    public h() {
        TimeUnit timeUnit = TimeUnit.MINUTES;
        this.f6048d = new a();
        this.f6049e = new ArrayDeque();
        this.f6050f = new f.g0.f.d();
        this.f6046b = 5;
        this.f6047c = timeUnit.toNanos(5L);
    }

    public final int a(f.g0.f.c cVar, long j) {
        List<Reference<f.g0.f.g>> list = cVar.n;
        int i = 0;
        while (i < list.size()) {
            Reference<f.g0.f.g> reference = list.get(i);
            if (reference.get() != null) {
                i++;
            } else {
                StringBuilder x = c.b.a.a.a.x("A connection to ");
                x.append(cVar.f5791c.f5750a.f5715a);
                x.append(" was leaked. Did you forget to close a response body?");
                f.g0.j.f.f6032a.l(x.toString(), ((g.a) reference).f5818a);
                list.remove(i);
                cVar.k = true;
                if (list.isEmpty()) {
                    cVar.o = j - this.f6047c;
                    return 0;
                }
            }
        }
        return list.size();
    }
}