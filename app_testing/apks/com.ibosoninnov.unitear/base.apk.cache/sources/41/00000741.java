package c.c.a.m.v.c0;

import c.c.a.m.v.c0.l;
import java.util.ArrayDeque;
import java.util.Queue;

/* compiled from: BaseKeyPool.java */
/* loaded from: classes.dex */
public abstract class c<T extends l> {

    /* renamed from: a  reason: collision with root package name */
    public final Queue<T> f3606a;

    public c() {
        char[] cArr = c.c.a.s.j.f4197a;
        this.f3606a = new ArrayDeque(20);
    }

    public abstract T a();

    public T b() {
        T poll = this.f3606a.poll();
        return poll == null ? a() : poll;
    }

    public void c(T t) {
        if (this.f3606a.size() < 20) {
            this.f3606a.offer(t);
        }
    }
}