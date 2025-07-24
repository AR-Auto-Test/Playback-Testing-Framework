package b.d.b;

import android.app.Application;
import android.content.Context;
import android.os.Build;
import android.os.Handler;
import android.os.Message;
import android.os.SystemClock;
import b.d.b.d1.j1;
import b.d.b.d1.x;
import b.d.b.d1.y;
import java.util.Objects;
import java.util.concurrent.Executor;

/* compiled from: lambda */
/* loaded from: classes.dex */
public final /* synthetic */ class j implements Runnable {

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ n0 f1624b;

    /* renamed from: c  reason: collision with root package name */
    public final /* synthetic */ Context f1625c;

    /* renamed from: d  reason: collision with root package name */
    public final /* synthetic */ Executor f1626d;

    /* renamed from: e  reason: collision with root package name */
    public final /* synthetic */ b.g.a.b f1627e;

    /* renamed from: f  reason: collision with root package name */
    public final /* synthetic */ long f1628f;

    public /* synthetic */ j(n0 n0Var, Context context, Executor executor, b.g.a.b bVar, long j) {
        this.f1624b = n0Var;
        this.f1625c = context;
        this.f1626d = executor;
        this.f1627e = bVar;
        this.f1628f = j;
    }

    @Override // java.lang.Runnable
    public final void run() {
        final n0 n0Var = this.f1624b;
        Context context = this.f1625c;
        final Executor executor = this.f1626d;
        final b.g.a.b bVar = this.f1627e;
        final long j = this.f1628f;
        Objects.requireNonNull(n0Var);
        try {
            Application a2 = n0.a(context);
            n0Var.o = a2;
            if (a2 == null) {
                n0Var.o = context.getApplicationContext();
            }
            y.a aVar = (y.a) n0Var.f1654h.x.f(o0.q, null);
            if (aVar != null) {
                n0Var.l = aVar.a(n0Var.o, new b.d.b.d1.m(n0Var.i, n0Var.j), (j0) n0Var.f1654h.x.f(o0.w, null));
                x.a aVar2 = (x.a) n0Var.f1654h.x.f(o0.r, null);
                if (aVar2 != null) {
                    Context context2 = n0Var.o;
                    b.d.b.d1.y yVar = n0Var.l;
                    n0Var.m = aVar2.a(context2, ((b.d.a.e.p0) yVar).f1136c, ((b.d.a.e.p0) yVar).a());
                    j1.b bVar2 = (j1.b) n0Var.f1654h.x.f(o0.s, null);
                    if (bVar2 != null) {
                        n0Var.n = bVar2.a(n0Var.o);
                        if (executor instanceof g0) {
                            ((g0) executor).b(n0Var.l);
                        }
                        n0Var.f1652f.b(n0Var.l);
                        if (b.d.b.e1.h.a.a.f1608a.a(b.d.b.e1.h.a.c.class) != null) {
                            b.b.a.p(n0Var.o, n0Var.f1652f);
                        }
                        n0Var.e();
                        bVar.a(null);
                        return;
                    }
                    throw new t0(new IllegalArgumentException("Invalid app configuration provided. Missing UseCaseConfigFactory."));
                }
                throw new t0(new IllegalArgumentException("Invalid app configuration provided. Missing CameraDeviceSurfaceManager."));
            }
            throw new t0(new IllegalArgumentException("Invalid app configuration provided. Missing CameraFactory."));
        } catch (b.d.b.d1.e0 | t0 | RuntimeException e2) {
            if (SystemClock.elapsedRealtime() - j < 2500) {
                u0.d("CameraX", "Retry init. Start time " + j + " current time " + SystemClock.elapsedRealtime(), e2);
                Handler handler = n0Var.j;
                Runnable runnable = new Runnable() { // from class: b.d.b.g
                    @Override // java.lang.Runnable
                    public final void run() {
                        n0 n0Var2 = n0.this;
                        Executor executor2 = executor;
                        long j2 = j;
                        executor2.execute(new j(n0Var2, n0Var2.o, executor2, bVar, j2));
                    }
                };
                if (Build.VERSION.SDK_INT >= 28) {
                    handler.postDelayed(runnable, "retry_token", 500L);
                    return;
                }
                Message obtain = Message.obtain(handler, runnable);
                obtain.obj = "retry_token";
                handler.sendMessageDelayed(obtain, 500L);
                return;
            }
            synchronized (n0Var.f1653g) {
                n0Var.p = 3;
            }
            if (e2 instanceof b.d.b.d1.e0) {
                u0.b("CameraX", "The device might underreport the amount of the cameras. Finish the initialize task since we are already reaching the maximum number of retries.", null);
                bVar.a(null);
            } else if (e2 instanceof t0) {
                bVar.c(e2);
            } else {
                bVar.c(new t0(e2));
            }
        }
    }
}