package c.c.a.s;

import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.Map;

/* compiled from: LruCache.java */
/* loaded from: classes.dex */
public class g<T, Y> {

    /* renamed from: a  reason: collision with root package name */
    public final Map<T, a<Y>> f4188a = new LinkedHashMap(100, 0.75f, true);

    /* renamed from: b  reason: collision with root package name */
    public long f4189b;

    /* renamed from: c  reason: collision with root package name */
    public long f4190c;

    /* compiled from: LruCache.java */
    /* loaded from: classes.dex */
    public static final class a<Y> {

        /* renamed from: a  reason: collision with root package name */
        public final Y f4191a;

        /* renamed from: b  reason: collision with root package name */
        public final int f4192b;

        public a(Y y, int i) {
            this.f4191a = y;
            this.f4192b = i;
        }
    }

    public g(long j) {
        this.f4189b = j;
    }

    public synchronized Y a(T t) {
        a<Y> aVar;
        aVar = this.f4188a.get(t);
        return aVar != null ? aVar.f4191a : null;
    }

    public int b(Y y) {
        return 1;
    }

    public void c(T t, Y y) {
    }

    public synchronized Y d(T t, Y y) {
        int b2 = b(y);
        long j = b2;
        if (j >= this.f4189b) {
            c(t, y);
            return null;
        }
        if (y != null) {
            this.f4190c += j;
        }
        a<Y> put = this.f4188a.put(t, y == null ? null : new a<>(y, b2));
        if (put != null) {
            this.f4190c -= put.f4192b;
            if (!put.f4191a.equals(y)) {
                c(t, put.f4191a);
            }
        }
        e(this.f4189b);
        return put != null ? put.f4191a : null;
    }

    public synchronized void e(long j) {
        while (this.f4190c > j) {
            Iterator<Map.Entry<T, a<Y>>> it = this.f4188a.entrySet().iterator();
            Map.Entry<T, a<Y>> next = it.next();
            a<Y> value = next.getValue();
            this.f4190c -= value.f4192b;
            T key = next.getKey();
            it.remove();
            c(key, value.f4191a);
        }
    }
}