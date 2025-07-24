package f;

import f.x;
import java.util.ArrayDeque;
import java.util.Deque;
import java.util.Iterator;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.SynchronousQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

/* compiled from: Dispatcher.java */
/* loaded from: classes2.dex */
public final class l {

    /* renamed from: a  reason: collision with root package name */
    public ExecutorService f6073a;

    /* renamed from: b  reason: collision with root package name */
    public final Deque<x.a> f6074b = new ArrayDeque();

    /* renamed from: c  reason: collision with root package name */
    public final Deque<x.a> f6075c = new ArrayDeque();

    /* renamed from: d  reason: collision with root package name */
    public final Deque<x> f6076d = new ArrayDeque();

    public synchronized void a() {
        for (x.a aVar : this.f6074b) {
            x.this.a();
        }
        for (x.a aVar2 : this.f6075c) {
            x.this.a();
        }
        for (x xVar : this.f6076d) {
            xVar.a();
        }
    }

    public synchronized ExecutorService b() {
        if (this.f6073a == null) {
            TimeUnit timeUnit = TimeUnit.SECONDS;
            SynchronousQueue synchronousQueue = new SynchronousQueue();
            byte[] bArr = f.g0.c.f5773a;
            this.f6073a = new ThreadPoolExecutor(0, Integer.MAX_VALUE, 60L, timeUnit, synchronousQueue, new f.g0.d("OkHttp Dispatcher", false));
        }
        return this.f6073a;
    }

    public void c(x.a aVar) {
        Deque<x.a> deque = this.f6075c;
        synchronized (this) {
            if (deque.remove(aVar)) {
                d();
                synchronized (this) {
                    this.f6075c.size();
                    this.f6076d.size();
                }
            }
            throw new AssertionError("Call wasn't in-flight!");
        }
    }

    public final void d() {
        if (this.f6075c.size() < 64 && !this.f6074b.isEmpty()) {
            Iterator<x.a> it = this.f6074b.iterator();
            while (it.hasNext()) {
                x.a next = it.next();
                if (e(next) < 5) {
                    it.remove();
                    this.f6075c.add(next);
                    b().execute(next);
                }
                if (this.f6075c.size() >= 64) {
                    return;
                }
            }
        }
    }

    public final int e(x.a aVar) {
        int i = 0;
        for (x.a aVar2 : this.f6075c) {
            x xVar = x.this;
            if (!xVar.f6146f && xVar.f6145e.f6150a.f6090e.equals(x.this.f6145e.f6150a.f6090e)) {
                i++;
            }
        }
        return i;
    }
}