package c.c.a.m.w;

import java.util.ArrayDeque;
import java.util.Objects;
import java.util.Queue;

/* compiled from: ModelCache.java */
/* loaded from: classes.dex */
public class m<A, B> {

    /* renamed from: a  reason: collision with root package name */
    public final c.c.a.s.g<b<A>, B> f3858a;

    /* compiled from: ModelCache.java */
    /* loaded from: classes.dex */
    public class a extends c.c.a.s.g<b<A>, B> {
        public a(m mVar, long j) {
            super(j);
        }

        @Override // c.c.a.s.g
        public void c(Object obj, Object obj2) {
            b<?> bVar = (b) obj;
            Objects.requireNonNull(bVar);
            Queue<b<?>> queue = b.f3859a;
            synchronized (queue) {
                queue.offer(bVar);
            }
        }
    }

    /* compiled from: ModelCache.java */
    /* loaded from: classes.dex */
    public static final class b<A> {

        /* renamed from: a  reason: collision with root package name */
        public static final Queue<b<?>> f3859a;

        /* renamed from: b  reason: collision with root package name */
        public int f3860b;

        /* renamed from: c  reason: collision with root package name */
        public int f3861c;

        /* renamed from: d  reason: collision with root package name */
        public A f3862d;

        static {
            char[] cArr = c.c.a.s.j.f4197a;
            f3859a = new ArrayDeque(0);
        }

        public static <A> b<A> a(A a2, int i, int i2) {
            b<A> bVar;
            Queue<b<?>> queue = f3859a;
            synchronized (queue) {
                bVar = (b<A>) queue.poll();
            }
            if (bVar == null) {
                bVar = new b<>();
            }
            bVar.f3862d = a2;
            bVar.f3861c = i;
            bVar.f3860b = i2;
            return bVar;
        }

        public boolean equals(Object obj) {
            if (obj instanceof b) {
                b bVar = (b) obj;
                return this.f3861c == bVar.f3861c && this.f3860b == bVar.f3860b && this.f3862d.equals(bVar.f3862d);
            }
            return false;
        }

        public int hashCode() {
            return this.f3862d.hashCode() + (((this.f3860b * 31) + this.f3861c) * 31);
        }
    }

    public m(long j) {
        this.f3858a = new a(this, j);
    }
}