package b.d.b.d1;

import android.util.Log;
import android.view.Surface;
import b.d.b.d1.k1.c.h;
import com.google.common.util.concurrent.ListenableFuture;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicInteger;

/* compiled from: DeferrableSurface.java */
/* loaded from: classes.dex */
public abstract class j0 {

    /* renamed from: a  reason: collision with root package name */
    public static final boolean f1497a = b.d.b.u0.c("DeferrableSurface");

    /* renamed from: b  reason: collision with root package name */
    public static final AtomicInteger f1498b = new AtomicInteger(0);

    /* renamed from: c  reason: collision with root package name */
    public static final AtomicInteger f1499c = new AtomicInteger(0);

    /* renamed from: d  reason: collision with root package name */
    public final Object f1500d = new Object();

    /* renamed from: e  reason: collision with root package name */
    public int f1501e = 0;

    /* renamed from: f  reason: collision with root package name */
    public boolean f1502f = false;

    /* renamed from: g  reason: collision with root package name */
    public b.g.a.b<Void> f1503g;

    /* renamed from: h  reason: collision with root package name */
    public final ListenableFuture<Void> f1504h;

    /* compiled from: DeferrableSurface.java */
    /* loaded from: classes.dex */
    public static final class a extends Exception {

        /* renamed from: b  reason: collision with root package name */
        public j0 f1505b;

        public a(String str, j0 j0Var) {
            super(str);
            this.f1505b = j0Var;
        }
    }

    public j0() {
        ListenableFuture<Void> d2 = b.e.a.d(new b.g.a.d() { // from class: b.d.b.d1.d
            @Override // b.g.a.d
            public final Object a(b.g.a.b bVar) {
                j0 j0Var = j0.this;
                synchronized (j0Var.f1500d) {
                    j0Var.f1503g = bVar;
                }
                return "DeferrableSurface-termination(" + j0Var + ")";
            }
        });
        this.f1504h = d2;
        if (b.d.b.u0.c("DeferrableSurface")) {
            f("Surface created", f1499c.incrementAndGet(), f1498b.get());
            final String stackTraceString = Log.getStackTraceString(new Exception());
            d2.addListener(new Runnable() { // from class: b.d.b.d1.c
                @Override // java.lang.Runnable
                public final void run() {
                    j0 j0Var = j0.this;
                    String str = stackTraceString;
                    Objects.requireNonNull(j0Var);
                    try {
                        j0Var.f1504h.get();
                        j0Var.f("Surface terminated", j0.f1499c.decrementAndGet(), j0.f1498b.get());
                    } catch (Exception e2) {
                        b.d.b.u0.b("DeferrableSurface", "Unexpected surface termination for " + j0Var + "\nStack Trace:\n" + str, null);
                        synchronized (j0Var.f1500d) {
                            throw new IllegalArgumentException(String.format("DeferrableSurface %s [closed: %b, use_count: %s] terminated with unexpected exception.", j0Var, Boolean.valueOf(j0Var.f1502f), Integer.valueOf(j0Var.f1501e)), e2);
                        }
                    }
                }
            }, b.b.a.f());
        }
    }

    public final void a() {
        b.g.a.b<Void> bVar;
        synchronized (this.f1500d) {
            if (this.f1502f) {
                bVar = null;
            } else {
                this.f1502f = true;
                if (this.f1501e == 0) {
                    bVar = this.f1503g;
                    this.f1503g = null;
                } else {
                    bVar = null;
                }
                if (b.d.b.u0.c("DeferrableSurface")) {
                    b.d.b.u0.a("DeferrableSurface", "surface closed,  useCount=" + this.f1501e + " closed=true " + this, null);
                }
            }
        }
        if (bVar != null) {
            bVar.a(null);
        }
    }

    public void b() {
        b.g.a.b<Void> bVar;
        synchronized (this.f1500d) {
            int i = this.f1501e;
            if (i != 0) {
                int i2 = i - 1;
                this.f1501e = i2;
                if (i2 == 0 && this.f1502f) {
                    bVar = this.f1503g;
                    this.f1503g = null;
                } else {
                    bVar = null;
                }
                if (b.d.b.u0.c("DeferrableSurface")) {
                    b.d.b.u0.a("DeferrableSurface", "use count-1,  useCount=" + this.f1501e + " closed=" + this.f1502f + " " + this, null);
                    if (this.f1501e == 0) {
                        f("Surface no longer in use", f1499c.get(), f1498b.decrementAndGet());
                    }
                }
            } else {
                throw new IllegalStateException("Decrementing use count occurs more times than incrementing");
            }
        }
        if (bVar != null) {
            bVar.a(null);
        }
    }

    public final ListenableFuture<Surface> c() {
        synchronized (this.f1500d) {
            if (this.f1502f) {
                return new h.a(new a("DeferrableSurface already closed.", this));
            }
            return g();
        }
    }

    public ListenableFuture<Void> d() {
        return b.d.b.d1.k1.c.g.d(this.f1504h);
    }

    public void e() {
        synchronized (this.f1500d) {
            int i = this.f1501e;
            if (i == 0 && this.f1502f) {
                throw new a("Cannot begin use on a closed surface.", this);
            }
            this.f1501e = i + 1;
            if (b.d.b.u0.c("DeferrableSurface")) {
                if (this.f1501e == 1) {
                    f("New surface in use", f1499c.get(), f1498b.incrementAndGet());
                }
                b.d.b.u0.a("DeferrableSurface", "use count+1, useCount=" + this.f1501e + " " + this, null);
            }
        }
    }

    public final void f(String str, int i, int i2) {
        if (!f1497a && b.d.b.u0.c("DeferrableSurface")) {
            b.d.b.u0.a("DeferrableSurface", "DeferrableSurface usage statistics may be inaccurate since debug logging was not enabled at static initialization time. App restart may be required to enable accurate usage statistics.", null);
        }
        b.d.b.u0.a("DeferrableSurface", str + "[total_surfaces=" + i + ", used_surfaces=" + i2 + "](" + this + "}", null);
    }

    public abstract ListenableFuture<Surface> g();
}