package b.d.b;

import android.graphics.Rect;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.Size;
import b.d.b.a1;
import b.d.b.d1.b1;
import b.d.b.d1.h0;
import b.d.b.d1.i0;
import b.d.b.d1.i1;
import b.d.b.d1.j1;
import b.d.b.w0;
import java.util.Objects;
import java.util.UUID;
import java.util.concurrent.Executor;

/* compiled from: Preview.java */
/* loaded from: classes.dex */
public final class w0 extends a1 {
    public static final c l = new c();
    public static final Executor m = b.b.a.l();
    public d n;
    public Executor o;
    public b.d.b.d1.j0 p;
    public z0 q;
    public boolean r;
    public Size s;

    /* compiled from: Preview.java */
    /* loaded from: classes.dex */
    public class a extends b.d.b.d1.q {

        /* renamed from: a  reason: collision with root package name */
        public final /* synthetic */ b.d.b.d1.l0 f1685a;

        public a(b.d.b.d1.l0 l0Var) {
            this.f1685a = l0Var;
        }

        @Override // b.d.b.d1.q
        public void b(b.d.b.d1.t tVar) {
            if (this.f1685a.a(new b.d.b.e1.b(tVar))) {
                w0 w0Var = w0.this;
                for (a1.b bVar : w0Var.f1379a) {
                    bVar.e(w0Var);
                }
            }
        }
    }

    /* compiled from: Preview.java */
    /* loaded from: classes.dex */
    public static final class b implements i1.a<w0, b.d.b.d1.x0, b> {

        /* renamed from: a  reason: collision with root package name */
        public final b.d.b.d1.u0 f1687a;

        public b() {
            this(b.d.b.d1.u0.y());
        }

        public w0 a() {
            if (this.f1687a.f(b.d.b.d1.n0.f1574b, null) != null && this.f1687a.f(b.d.b.d1.n0.f1576d, null) != null) {
                throw new IllegalArgumentException("Cannot use both setTargetResolution and setTargetAspectRatio on the same config.");
            }
            return new w0(b());
        }

        public b.d.b.d1.x0 b() {
            return new b.d.b.d1.x0(b.d.b.d1.w0.x(this.f1687a));
        }

        public b(b.d.b.d1.u0 u0Var) {
            this.f1687a = u0Var;
            i0.a<Class<?>> aVar = b.d.b.e1.e.o;
            Class cls = (Class) u0Var.f(aVar, null);
            if (cls != null && !cls.equals(w0.class)) {
                throw new IllegalArgumentException("Invalid target class configuration for " + this + ": " + cls);
            }
            i0.c cVar = i0.c.OPTIONAL;
            u0Var.A(aVar, cVar, w0.class);
            i0.a<String> aVar2 = b.d.b.e1.e.n;
            if (u0Var.f(aVar2, null) == null) {
                u0Var.A(aVar2, cVar, w0.class.getCanonicalName() + "-" + UUID.randomUUID());
            }
        }
    }

    /* compiled from: Preview.java */
    /* loaded from: classes.dex */
    public static final class c {

        /* renamed from: a  reason: collision with root package name */
        public static final b.d.b.d1.x0 f1688a;

        static {
            b bVar = new b();
            b.d.b.d1.u0 u0Var = bVar.f1687a;
            i0.a<Integer> aVar = i1.l;
            i0.c cVar = i0.c.OPTIONAL;
            u0Var.A(aVar, cVar, 2);
            bVar.f1687a.A(b.d.b.d1.n0.f1574b, cVar, 0);
            f1688a = bVar.b();
        }
    }

    /* compiled from: Preview.java */
    /* loaded from: classes.dex */
    public interface d {
        void a(z0 z0Var);
    }

    public w0(b.d.b.d1.x0 x0Var) {
        super(x0Var);
        this.o = m;
        this.r = false;
    }

    @Override // b.d.b.a1
    public i1<?> c(boolean z, j1 j1Var) {
        b.d.b.d1.u0 y;
        b.d.b.d1.w0 x;
        b.d.b.d1.i0 a2 = j1Var.a(j1.a.PREVIEW);
        if (z) {
            Objects.requireNonNull(l);
            b.d.b.d1.x0 x0Var = c.f1688a;
            if (a2 == null && x0Var == null) {
                x = b.d.b.d1.w0.q;
            } else {
                if (x0Var != null) {
                    y = b.d.b.d1.u0.z(x0Var);
                } else {
                    y = b.d.b.d1.u0.y();
                }
                if (a2 != null) {
                    for (i0.a<?> aVar : a2.e()) {
                        y.A(aVar, a2.g(aVar), a2.a(aVar));
                    }
                }
                x = b.d.b.d1.w0.x(y);
            }
            a2 = x;
        }
        if (a2 == null) {
            return null;
        }
        return new b(b.d.b.d1.u0.z(a2)).b();
    }

    @Override // b.d.b.a1
    public i1.a<?, ?, ?> e(b.d.b.d1.i0 i0Var) {
        return new b(b.d.b.d1.u0.z(i0Var));
    }

    @Override // b.d.b.a1
    public void k() {
        b.d.b.d1.j0 j0Var = this.p;
        if (j0Var != null) {
            j0Var.a();
        }
        this.q = null;
    }

    @Override // b.d.b.a1
    public i1<?> l(b.d.b.d1.z zVar, i1.a<?, ?, ?> aVar) {
        i0.c cVar = i0.c.OPTIONAL;
        if (((b) aVar).f1687a.f(b.d.b.d1.x0.r, null) != null) {
            ((b) aVar).f1687a.A(b.d.b.d1.m0.f1570a, cVar, 35);
        } else {
            ((b) aVar).f1687a.A(b.d.b.d1.m0.f1570a, cVar, 34);
        }
        return ((b) aVar).b();
    }

    @Override // b.d.b.a1
    public Size m(Size size) {
        this.s = size;
        this.k = n(b(), (b.d.b.d1.x0) this.f1384f, this.s).d();
        return size;
    }

    public b1.b n(final String str, final b.d.b.d1.x0 x0Var, final Size size) {
        b.d.b.d1.q qVar;
        b.b.a.c();
        b1.b e2 = b1.b.e(x0Var);
        b.d.b.d1.g0 g0Var = (b.d.b.d1.g0) x0Var.f(b.d.b.d1.x0.r, null);
        b.d.b.d1.j0 j0Var = this.p;
        if (j0Var != null) {
            j0Var.a();
        }
        z0 z0Var = new z0(size, a(), g0Var != null);
        this.q = z0Var;
        if (o()) {
            p();
        } else {
            this.r = true;
        }
        if (g0Var != null) {
            h0.a aVar = new h0.a();
            final HandlerThread handlerThread = new HandlerThread("CameraX-preview_processing");
            handlerThread.start();
            String num = Integer.toString(aVar.hashCode());
            x0 x0Var2 = new x0(size.getWidth(), size.getHeight(), x0Var.l(), new Handler(handlerThread.getLooper()), aVar, g0Var, z0Var.f1706g, num);
            synchronized (x0Var2.i) {
                if (!x0Var2.k) {
                    qVar = x0Var2.r;
                } else {
                    throw new IllegalStateException("ProcessingSurface already released!");
                }
            }
            e2.a(qVar);
            x0Var2.d().addListener(new Runnable() { // from class: b.d.b.a
                @Override // java.lang.Runnable
                public final void run() {
                    handlerThread.quitSafely();
                }
            }, b.b.a.f());
            this.p = x0Var2;
            e2.f1421b.f1473f.f1480b.put(num, 0);
        } else {
            b.d.b.d1.l0 l0Var = (b.d.b.d1.l0) x0Var.f(b.d.b.d1.x0.q, null);
            if (l0Var != null) {
                a aVar2 = new a(l0Var);
                e2.f1421b.b(aVar2);
                e2.f1425f.add(aVar2);
            }
            this.p = z0Var.f1706g;
        }
        b.d.b.d1.j0 j0Var2 = this.p;
        e2.f1420a.add(j0Var2);
        e2.f1421b.f1468a.add(j0Var2);
        e2.f1424e.add(new b1.c() { // from class: b.d.b.p
            @Override // b.d.b.d1.b1.c
            public final void a(b.d.b.d1.b1 b1Var, b1.e eVar) {
                w0 w0Var = w0.this;
                String str2 = str;
                b.d.b.d1.x0 x0Var3 = x0Var;
                Size size2 = size;
                if (w0Var.a() == null ? false : Objects.equals(str2, w0Var.b())) {
                    w0Var.k = w0Var.n(str2, x0Var3, size2).d();
                    w0Var.g();
                }
            }
        });
        return e2;
    }

    public final boolean o() {
        final z0 z0Var = this.q;
        final d dVar = this.n;
        if (dVar == null || z0Var == null) {
            return false;
        }
        this.o.execute(new Runnable() { // from class: b.d.b.q
            @Override // java.lang.Runnable
            public final void run() {
                w0.d.this.a(z0Var);
            }
        });
        return true;
    }

    public final void p() {
        b.d.b.d1.a0 a2 = a();
        d dVar = this.n;
        Size size = this.s;
        Rect rect = this.i;
        if (rect == null) {
            rect = size != null ? new Rect(0, 0, size.getWidth(), size.getHeight()) : null;
        }
        z0 z0Var = this.q;
        if (a2 == null || dVar == null || rect == null) {
            return;
        }
        a2.j().d(((b.d.b.d1.n0) this.f1384f).w(0));
        ((b.d.b.d1.n0) this.f1384f).w(0);
        Objects.requireNonNull(rect, "Null cropRect");
        Objects.requireNonNull(z0Var);
    }

    public void q(Executor executor, d dVar) {
        b.b.a.c();
        this.n = dVar;
        this.o = executor;
        this.f1381c = 1;
        h();
        if (this.r) {
            if (o()) {
                p();
                this.r = false;
            }
        } else if (this.f1385g != null) {
            this.k = n(b(), (b.d.b.d1.x0) this.f1384f, this.f1385g).d();
            g();
        }
    }

    public String toString() {
        StringBuilder x = c.b.a.a.a.x("Preview:");
        x.append(d());
        return x.toString();
    }
}