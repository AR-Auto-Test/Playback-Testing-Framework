package c.c.a.m.v.d0;

import java.util.ArrayDeque;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Queue;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

/* compiled from: DiskCacheWriteLocker.java */
/* loaded from: classes.dex */
public final class c {

    /* renamed from: a  reason: collision with root package name */
    public final Map<String, a> f3649a = new HashMap();

    /* renamed from: b  reason: collision with root package name */
    public final b f3650b = new b();

    /* compiled from: DiskCacheWriteLocker.java */
    /* loaded from: classes.dex */
    public static class a {

        /* renamed from: a  reason: collision with root package name */
        public final Lock f3651a = new ReentrantLock();

        /* renamed from: b  reason: collision with root package name */
        public int f3652b;
    }

    /* compiled from: DiskCacheWriteLocker.java */
    /* loaded from: classes.dex */
    public static class b {

        /* renamed from: a  reason: collision with root package name */
        public final Queue<a> f3653a = new ArrayDeque();
    }

    public void a(String str) {
        a aVar;
        synchronized (this) {
            a aVar2 = this.f3649a.get(str);
            Objects.requireNonNull(aVar2, "Argument must not be null");
            aVar = aVar2;
            int i = aVar.f3652b;
            if (i >= 1) {
                int i2 = i - 1;
                aVar.f3652b = i2;
                if (i2 == 0) {
                    a remove = this.f3649a.remove(str);
                    if (remove.equals(aVar)) {
                        b bVar = this.f3650b;
                        synchronized (bVar.f3653a) {
                            if (bVar.f3653a.size() < 10) {
                                bVar.f3653a.offer(remove);
                            }
                        }
                    } else {
                        throw new IllegalStateException("Removed the wrong lock, expected to remove: " + aVar + ", but actually removed: " + remove + ", safeKey: " + str);
                    }
                }
            } else {
                throw new IllegalStateException("Cannot release a lock that is not held, safeKey: " + str + ", interestedThreads: " + aVar.f3652b);
            }
        }
        aVar.f3651a.unlock();
    }
}