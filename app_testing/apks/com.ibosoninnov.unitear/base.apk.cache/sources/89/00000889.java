package c.c.a.q.j;

import android.graphics.drawable.Drawable;
import c.c.a.s.j;

/* compiled from: CustomTarget.java */
/* loaded from: classes.dex */
public abstract class c<T> implements h<T> {

    /* renamed from: b  reason: collision with root package name */
    public final int f4154b;

    /* renamed from: c  reason: collision with root package name */
    public final int f4155c;

    /* renamed from: d  reason: collision with root package name */
    public c.c.a.q.c f4156d;

    public c() {
        if (j.j(Integer.MIN_VALUE, Integer.MIN_VALUE)) {
            this.f4154b = Integer.MIN_VALUE;
            this.f4155c = Integer.MIN_VALUE;
            return;
        }
        throw new IllegalArgumentException(c.b.a.a.a.k("Width and height must both be > 0 or Target#SIZE_ORIGINAL, but given width: ", Integer.MIN_VALUE, " and height: ", Integer.MIN_VALUE));
    }

    @Override // c.c.a.q.j.h
    public final void a(g gVar) {
    }

    @Override // c.c.a.q.j.h
    public final void c(c.c.a.q.c cVar) {
        this.f4156d = cVar;
    }

    @Override // c.c.a.q.j.h
    public void d(Drawable drawable) {
    }

    @Override // c.c.a.q.j.h
    public void e(Drawable drawable) {
    }

    @Override // c.c.a.q.j.h
    public final c.c.a.q.c f() {
        return this.f4156d;
    }

    @Override // c.c.a.q.j.h
    public final void h(g gVar) {
        ((c.c.a.q.h) gVar).b(this.f4154b, this.f4155c);
    }

    @Override // c.c.a.n.m
    public void onDestroy() {
    }

    @Override // c.c.a.n.m
    public void onStart() {
    }

    @Override // c.c.a.n.m
    public void onStop() {
    }
}