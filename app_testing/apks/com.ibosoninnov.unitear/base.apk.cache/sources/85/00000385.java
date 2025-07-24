package b.d.b;

import android.app.Application;
import android.content.Context;
import android.content.ContextWrapper;
import android.content.res.Resources;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.SystemClock;
import b.d.b.d1.j1;
import b.d.b.d1.k1.c.g;
import b.d.b.d1.k1.c.h;
import b.d.b.o0;
import com.google.common.util.concurrent.ListenableFuture;
import com.ibosoninnov.unitear.R;
import java.lang.reflect.InvocationTargetException;
import java.util.Objects;
import java.util.concurrent.Executor;

/* compiled from: CameraX.java */
/* loaded from: classes.dex */
public final class n0 {

    /* renamed from: b  reason: collision with root package name */
    public static n0 f1648b;

    /* renamed from: c  reason: collision with root package name */
    public static o0.b f1649c;

    /* renamed from: h  reason: collision with root package name */
    public final o0 f1654h;
    public final Executor i;
    public final Handler j;
    public final HandlerThread k;
    public b.d.b.d1.y l;
    public b.d.b.d1.x m;
    public j1 n;
    public Context o;

    /* renamed from: a  reason: collision with root package name */
    public static final Object f1647a = new Object();

    /* renamed from: d  reason: collision with root package name */
    public static ListenableFuture<Void> f1650d = new h.a(new IllegalStateException("CameraX is not initialized."));

    /* renamed from: e  reason: collision with root package name */
    public static ListenableFuture<Void> f1651e = b.d.b.d1.k1.c.g.c(null);

    /* renamed from: f  reason: collision with root package name */
    public final b.d.b.d1.b0 f1652f = new b.d.b.d1.b0();

    /* renamed from: g  reason: collision with root package name */
    public final Object f1653g = new Object();
    public int p = 1;
    public ListenableFuture<Void> q = b.d.b.d1.k1.c.g.c(null);

    public n0(o0 o0Var) {
        Objects.requireNonNull(o0Var);
        this.f1654h = o0Var;
        Executor executor = (Executor) o0Var.x.f(o0.t, null);
        Handler handler = (Handler) o0Var.x.f(o0.u, null);
        this.i = executor == null ? new g0() : executor;
        if (handler == null) {
            HandlerThread handlerThread = new HandlerThread("CameraX-scheduler", 10);
            this.k = handlerThread;
            handlerThread.start();
            this.j = b.j.b.d.q(handlerThread.getLooper());
            return;
        }
        this.k = null;
        this.j = handler;
    }

    public static Application a(Context context) {
        for (Context applicationContext = context.getApplicationContext(); applicationContext instanceof ContextWrapper; applicationContext = ((ContextWrapper) applicationContext).getBaseContext()) {
            if (applicationContext instanceof Application) {
                return (Application) applicationContext;
            }
        }
        return null;
    }

    public static o0.b b(Context context) {
        Application a2 = a(context);
        if (a2 instanceof o0.b) {
            return (o0.b) a2;
        }
        try {
            return (o0.b) Class.forName(context.getApplicationContext().getResources().getString(R.string.androidx_camera_default_config_provider)).getDeclaredConstructor(new Class[0]).newInstance(new Object[0]);
        } catch (Resources.NotFoundException | ClassNotFoundException | IllegalAccessException | InstantiationException | NoSuchMethodException | NullPointerException | InvocationTargetException e2) {
            u0.b("CameraX", "Failed to retrieve default CameraXConfig.Provider from resources", e2);
            return null;
        }
    }

    public static ListenableFuture<n0> c() {
        n0 n0Var = f1648b;
        if (n0Var == null) {
            return new h.a(new IllegalStateException("Must call CameraX.initialize() first"));
        }
        ListenableFuture<Void> listenableFuture = f1650d;
        b.c.a.c.a aVar = new b.c.a.c.a() { // from class: b.d.b.e
            @Override // b.c.a.c.a
            public final Object apply(Object obj) {
                Void r2 = (Void) obj;
                return n0.this;
            }
        };
        Executor f2 = b.b.a.f();
        b.d.b.d1.k1.c.c cVar = new b.d.b.d1.k1.c.c(new b.d.b.d1.k1.c.f(aVar), listenableFuture);
        listenableFuture.addListener(cVar, f2);
        return cVar;
    }

    public static void d(final Context context) {
        b.j.b.d.k(f1648b == null, "CameraX already initialized.");
        Objects.requireNonNull(f1649c);
        final n0 n0Var = new n0(f1649c.getCameraXConfig());
        f1648b = n0Var;
        f1650d = b.e.a.d(new b.g.a.d() { // from class: b.d.b.f
            @Override // b.g.a.d
            public final Object a(b.g.a.b bVar) {
                final n0 n0Var2 = n0.this;
                final Context context2 = context;
                synchronized (n0.f1647a) {
                    b.d.b.d1.k1.c.e c2 = b.d.b.d1.k1.c.e.a(n0.f1651e).c(new b.d.b.d1.k1.c.b() { // from class: b.d.b.h
                        @Override // b.d.b.d1.k1.c.b
                        public final ListenableFuture apply(Object obj) {
                            ListenableFuture d2;
                            final n0 n0Var3 = n0.this;
                            final Context context3 = context2;
                            Void r5 = (Void) obj;
                            synchronized (n0Var3.f1653g) {
                                boolean z = true;
                                if (n0Var3.p != 1) {
                                    z = false;
                                }
                                b.j.b.d.k(z, "CameraX.initInternal() should only be called once per instance");
                                n0Var3.p = 2;
                                d2 = b.e.a.d(new b.g.a.d() { // from class: b.d.b.d
                                    @Override // b.g.a.d
                                    public final Object a(b.g.a.b bVar2) {
                                        n0 n0Var4 = n0.this;
                                        Context context4 = context3;
                                        Executor executor = n0Var4.i;
                                        executor.execute(new j(n0Var4, context4, executor, bVar2, SystemClock.elapsedRealtime()));
                                        return "CameraX initInternal";
                                    }
                                });
                            }
                            return d2;
                        }
                    }, b.b.a.f());
                    l0 l0Var = new l0(bVar, n0Var2);
                    c2.addListener(new g.d(c2, l0Var), b.b.a.f());
                }
                return "CameraX-initialize";
            }
        });
    }

    public static ListenableFuture<Void> f() {
        n0 n0Var = f1648b;
        if (n0Var == null) {
            return f1651e;
        }
        f1648b = null;
        ListenableFuture<Void> d2 = b.e.a.d(new b.g.a.d() { // from class: b.d.b.l
            @Override // b.g.a.d
            public final Object a(final b.g.a.b bVar) {
                final n0 n0Var2 = n0.this;
                synchronized (n0.f1647a) {
                    n0.f1650d.addListener(new Runnable() { // from class: b.d.b.k
                        @Override // java.lang.Runnable
                        public final void run() {
                            ListenableFuture<Void> c2;
                            final n0 n0Var3 = n0.this;
                            b.g.a.b bVar2 = bVar;
                            synchronized (n0Var3.f1653g) {
                                n0Var3.j.removeCallbacksAndMessages("retry_token");
                                int f2 = m0.f(n0Var3.p);
                                if (f2 == 0) {
                                    n0Var3.p = 4;
                                    c2 = b.d.b.d1.k1.c.g.c(null);
                                } else if (f2 != 1) {
                                    if (f2 == 2) {
                                        n0Var3.p = 4;
                                        n0Var3.q = b.e.a.d(new b.g.a.d() { // from class: b.d.b.m
                                            @Override // b.g.a.d
                                            public final Object a(final b.g.a.b bVar3) {
                                                ListenableFuture<Void> listenableFuture;
                                                final n0 n0Var4 = n0.this;
                                                final b.d.b.d1.b0 b0Var = n0Var4.f1652f;
                                                synchronized (b0Var.f1409a) {
                                                    if (b0Var.f1410b.isEmpty()) {
                                                        listenableFuture = b0Var.f1412d;
                                                        if (listenableFuture == null) {
                                                            listenableFuture = b.d.b.d1.k1.c.g.c(null);
                                                        }
                                                    } else {
                                                        ListenableFuture<Void> listenableFuture2 = b0Var.f1412d;
                                                        if (listenableFuture2 == null) {
                                                            listenableFuture2 = b.e.a.d(new b.g.a.d() { // from class: b.d.b.d1.a
                                                                @Override // b.g.a.d
                                                                public final Object a(b.g.a.b bVar4) {
                                                                    b0 b0Var2 = b0.this;
                                                                    synchronized (b0Var2.f1409a) {
                                                                        b0Var2.f1413e = bVar4;
                                                                    }
                                                                    return "CameraRepository-deinit";
                                                                }
                                                            });
                                                            b0Var.f1412d = listenableFuture2;
                                                        }
                                                        b0Var.f1411c.addAll(b0Var.f1410b.values());
                                                        for (final b.d.b.d1.a0 a0Var : b0Var.f1410b.values()) {
                                                            a0Var.release().addListener(new Runnable() { // from class: b.d.b.d1.b
                                                                @Override // java.lang.Runnable
                                                                public final void run() {
                                                                    b0 b0Var2 = b0.this;
                                                                    a0 a0Var2 = a0Var;
                                                                    synchronized (b0Var2.f1409a) {
                                                                        b0Var2.f1411c.remove(a0Var2);
                                                                        if (b0Var2.f1411c.isEmpty()) {
                                                                            Objects.requireNonNull(b0Var2.f1413e);
                                                                            b0Var2.f1413e.a(null);
                                                                            b0Var2.f1413e = null;
                                                                            b0Var2.f1412d = null;
                                                                        }
                                                                    }
                                                                }
                                                            }, b.b.a.f());
                                                        }
                                                        b0Var.f1410b.clear();
                                                        listenableFuture = listenableFuture2;
                                                    }
                                                }
                                                listenableFuture.addListener(new Runnable() { // from class: b.d.b.i
                                                    @Override // java.lang.Runnable
                                                    public final void run() {
                                                        n0 n0Var5 = n0.this;
                                                        b.g.a.b bVar4 = bVar3;
                                                        if (n0Var5.k != null) {
                                                            Executor executor = n0Var5.i;
                                                            if (executor instanceof g0) {
                                                                g0 g0Var = (g0) executor;
                                                                synchronized (g0Var.f1617c) {
                                                                    if (!g0Var.f1618d.isShutdown()) {
                                                                        g0Var.f1618d.shutdown();
                                                                    }
                                                                }
                                                            }
                                                            n0Var5.k.quit();
                                                            bVar4.a(null);
                                                        }
                                                    }
                                                }, n0Var4.i);
                                                return "CameraX shutdownInternal";
                                            }
                                        });
                                    }
                                    c2 = n0Var3.q;
                                } else {
                                    throw new IllegalStateException("CameraX could not be shutdown when it is initializing.");
                                }
                            }
                            b.d.b.d1.k1.c.g.e(c2, bVar2);
                        }
                    }, b.b.a.f());
                }
                return "CameraX shutdown";
            }
        });
        f1651e = d2;
        return d2;
    }

    public final void e() {
        synchronized (this.f1653g) {
            this.p = 3;
        }
    }
}