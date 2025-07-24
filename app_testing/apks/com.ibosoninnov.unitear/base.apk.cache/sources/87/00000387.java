package b.d.b;

import android.os.Handler;
import b.d.b.d1.i0;
import b.d.b.d1.j1;
import b.d.b.d1.x;
import b.d.b.d1.y;
import java.util.UUID;
import java.util.concurrent.Executor;

/* compiled from: CameraXConfig.java */
/* loaded from: classes.dex */
public final class o0 implements b.d.b.e1.e<n0> {
    public static final i0.a<y.a> q = new b.d.b.d1.n("camerax.core.appConfig.cameraFactoryProvider", y.a.class, null);
    public static final i0.a<x.a> r = new b.d.b.d1.n("camerax.core.appConfig.deviceSurfaceManagerProvider", x.a.class, null);
    public static final i0.a<j1.b> s = new b.d.b.d1.n("camerax.core.appConfig.useCaseConfigFactoryProvider", j1.b.class, null);
    public static final i0.a<Executor> t = new b.d.b.d1.n("camerax.core.appConfig.cameraExecutor", Executor.class, null);
    public static final i0.a<Handler> u = new b.d.b.d1.n("camerax.core.appConfig.schedulerHandler", Handler.class, null);
    public static final i0.a<Integer> v = new b.d.b.d1.n("camerax.core.appConfig.minimumLoggingLevel", Integer.TYPE, null);
    public static final i0.a<j0> w = new b.d.b.d1.n("camerax.core.appConfig.availableCamerasLimiter", j0.class, null);
    public final b.d.b.d1.w0 x;

    /* compiled from: CameraXConfig.java */
    /* loaded from: classes.dex */
    public static final class a {

        /* renamed from: a  reason: collision with root package name */
        public final b.d.b.d1.u0 f1657a;

        public a() {
            b.d.b.d1.u0 y = b.d.b.d1.u0.y();
            this.f1657a = y;
            i0.a<Class<?>> aVar = b.d.b.e1.e.o;
            Class cls = (Class) y.f(aVar, null);
            if (cls != null && !cls.equals(n0.class)) {
                throw new IllegalArgumentException("Invalid target class configuration for " + this + ": " + cls);
            }
            i0.c cVar = i0.c.OPTIONAL;
            y.A(aVar, cVar, n0.class);
            i0.a<String> aVar2 = b.d.b.e1.e.n;
            if (y.f(aVar2, null) == null) {
                y.A(aVar2, cVar, n0.class.getCanonicalName() + "-" + UUID.randomUUID());
            }
        }
    }

    /* compiled from: CameraXConfig.java */
    /* loaded from: classes.dex */
    public interface b {
        o0 getCameraXConfig();
    }

    public o0(b.d.b.d1.w0 w0Var) {
        this.x = w0Var;
    }

    @Override // b.d.b.d1.a1
    public b.d.b.d1.i0 k() {
        return this.x;
    }
}