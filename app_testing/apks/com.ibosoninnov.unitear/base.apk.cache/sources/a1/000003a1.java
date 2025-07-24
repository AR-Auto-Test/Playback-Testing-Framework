package b.d.b;

import android.os.Handler;
import android.util.Size;
import android.view.Surface;
import b.d.b.d1.k1.c.g;
import b.d.b.d1.o0;
import com.google.common.util.concurrent.ListenableFuture;
import java.util.Objects;

/* compiled from: ProcessingSurface.java */
/* loaded from: classes.dex */
public final class x0 extends b.d.b.d1.j0 {
    public final Object i = new Object();
    public final o0.a j;
    public boolean k;
    public final Size l;
    public final v0 m;
    public final Surface n;
    public final Handler o;
    public final b.d.b.d1.h0 p;
    public final b.d.b.d1.g0 q;
    public final b.d.b.d1.q r;
    public final b.d.b.d1.j0 s;
    public String t;

    /* compiled from: ProcessingSurface.java */
    /* loaded from: classes.dex */
    public class a implements b.d.b.d1.k1.c.d<Surface> {
        public a() {
        }

        @Override // b.d.b.d1.k1.c.d
        public void onFailure(Throwable th) {
            u0.b("ProcessingSurfaceTextur", "Failed to extract Listenable<Surface>.", th);
        }

        /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object] */
        @Override // b.d.b.d1.k1.c.d
        public void onSuccess(Surface surface) {
            Surface surface2 = surface;
            synchronized (x0.this.i) {
                x0.this.q.a(surface2, 1);
            }
        }
    }

    public x0(int i, int i2, int i3, Handler handler, b.d.b.d1.h0 h0Var, b.d.b.d1.g0 g0Var, b.d.b.d1.j0 j0Var, String str) {
        Surface a2;
        o0.a aVar = new o0.a() { // from class: b.d.b.s
            @Override // b.d.b.d1.o0.a
            public final void a(b.d.b.d1.o0 o0Var) {
                x0 x0Var = x0.this;
                synchronized (x0Var.i) {
                    x0Var.h(o0Var);
                }
            }
        };
        this.j = aVar;
        this.k = false;
        Size size = new Size(i, i2);
        this.l = size;
        this.o = handler;
        b.d.b.d1.k1.b.b bVar = new b.d.b.d1.k1.b.b(handler);
        v0 v0Var = new v0(i, i2, i3, 2);
        this.m = v0Var;
        v0Var.e(aVar, bVar);
        synchronized (v0Var.f1674a) {
            a2 = v0Var.f1678e.a();
        }
        this.n = a2;
        this.r = v0Var.f1675b;
        this.q = g0Var;
        g0Var.b(size);
        this.p = h0Var;
        this.s = j0Var;
        this.t = str;
        ListenableFuture<Surface> c2 = j0Var.c();
        a aVar2 = new a();
        c2.addListener(new g.d(c2, aVar2), b.b.a.f());
        d().addListener(new Runnable() { // from class: b.d.b.r
            @Override // java.lang.Runnable
            public final void run() {
                x0 x0Var = x0.this;
                synchronized (x0Var.i) {
                    if (x0Var.k) {
                        return;
                    }
                    x0Var.m.close();
                    x0Var.n.release();
                    x0Var.s.a();
                    x0Var.k = true;
                }
            }
        }, b.b.a.f());
    }

    @Override // b.d.b.d1.j0
    public ListenableFuture<Surface> g() {
        ListenableFuture<Surface> c2;
        synchronized (this.i) {
            c2 = b.d.b.d1.k1.c.g.c(this.n);
        }
        return c2;
    }

    public void h(b.d.b.d1.o0 o0Var) {
        r0 r0Var;
        if (this.k) {
            return;
        }
        try {
            r0Var = o0Var.d();
        } catch (IllegalStateException e2) {
            u0.b("ProcessingSurfaceTextur", "Failed to acquire next image.", e2);
            r0Var = null;
        }
        if (r0Var == null) {
            return;
        }
        q0 n = r0Var.n();
        if (n == null) {
            r0Var.close();
            return;
        }
        Integer a2 = n.a().a(this.t);
        if (a2 == null) {
            r0Var.close();
            return;
        }
        Objects.requireNonNull(this.p);
        if (a2.intValue() != 0) {
            u0.d("ProcessingSurfaceTextur", "ImageProxyBundle does not contain this id: " + a2, null);
            r0Var.close();
            return;
        }
        b.d.b.d1.c1 c1Var = new b.d.b.d1.c1(r0Var, this.t);
        this.q.c(c1Var);
        c1Var.f1441a.close();
    }
}