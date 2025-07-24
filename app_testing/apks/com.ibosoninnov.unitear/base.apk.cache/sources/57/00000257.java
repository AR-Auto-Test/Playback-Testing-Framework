package b.d.a.e;

import android.annotation.SuppressLint;
import android.graphics.SurfaceTexture;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.params.MeteringRectangle;
import android.os.Build;
import android.os.Handler;
import android.os.SystemClock;
import android.text.TextUtils;
import android.util.ArrayMap;
import android.util.Rational;
import android.util.Size;
import android.view.Surface;
import b.d.a.d.c;
import b.d.a.e.g1;
import b.d.a.e.q0;
import b.d.a.e.t1;
import b.d.b.d1.a0;
import b.d.b.d1.b1;
import b.d.b.d1.c0;
import b.d.b.d1.h1;
import b.d.b.d1.j0;
import b.d.b.d1.k1.c.g;
import b.d.b.d1.r0;
import b.d.b.d1.w;
import com.google.common.util.concurrent.ListenableFuture;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.CancellationException;
import java.util.concurrent.Executor;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicInteger;

/* compiled from: Camera2CameraImpl.java */
/* loaded from: classes.dex */
public final class q0 implements b.d.b.d1.a0 {

    /* renamed from: a  reason: collision with root package name */
    public final b.d.b.d1.h1 f1141a;

    /* renamed from: b  reason: collision with root package name */
    public final b.d.a.e.y1.k f1142b;

    /* renamed from: c  reason: collision with root package name */
    public final Executor f1143c;

    /* renamed from: d  reason: collision with root package name */
    public volatile e f1144d = e.INITIALIZED;

    /* renamed from: e  reason: collision with root package name */
    public final b.d.b.d1.r0<a0.a> f1145e;

    /* renamed from: f  reason: collision with root package name */
    public final o0 f1146f;

    /* renamed from: g  reason: collision with root package name */
    public final f f1147g;

    /* renamed from: h  reason: collision with root package name */
    public final r0 f1148h;
    public CameraDevice i;
    public int j;
    public g1 k;
    public b.d.b.d1.b1 l;
    public final AtomicInteger m;
    public ListenableFuture<Void> n;
    public b.g.a.b<Void> o;
    public final Map<g1, ListenableFuture<Void>> p;
    public final c q;
    public final b.d.b.d1.c0 r;
    public final Set<g1> s;
    public n1 t;
    public final h1 u;
    public final t1.a v;
    public final Set<String> w;

    /* compiled from: Camera2CameraImpl.java */
    /* loaded from: classes.dex */
    public class a implements b.d.b.d1.k1.c.d<Void> {

        /* renamed from: a  reason: collision with root package name */
        public final /* synthetic */ g1 f1149a;

        public a(g1 g1Var) {
            this.f1149a = g1Var;
        }

        @Override // b.d.b.d1.k1.c.d
        public void onFailure(Throwable th) {
        }

        /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object] */
        @Override // b.d.b.d1.k1.c.d
        public void onSuccess(Void r2) {
            CameraDevice cameraDevice;
            q0.this.p.remove(this.f1149a);
            int ordinal = q0.this.f1144d.ordinal();
            if (ordinal != 4) {
                if (ordinal != 5) {
                    if (ordinal != 6) {
                        return;
                    }
                } else if (q0.this.j == 0) {
                    return;
                }
            }
            if (!q0.this.q() || (cameraDevice = q0.this.i) == null) {
                return;
            }
            cameraDevice.close();
            q0.this.i = null;
        }
    }

    /* compiled from: Camera2CameraImpl.java */
    /* loaded from: classes.dex */
    public class b implements b.d.b.d1.k1.c.d<Void> {
        public b() {
        }

        @Override // b.d.b.d1.k1.c.d
        public void onFailure(Throwable th) {
            final b.d.b.d1.b1 b1Var = null;
            if (th instanceof CameraAccessException) {
                q0 q0Var = q0.this;
                StringBuilder x = c.b.a.a.a.x("Unable to configure camera due to ");
                x.append(th.getMessage());
                q0Var.n(x.toString(), null);
            } else if (th instanceof CancellationException) {
                q0.this.n("Unable to configure camera cancelled", null);
            } else if (th instanceof j0.a) {
                q0 q0Var2 = q0.this;
                b.d.b.d1.j0 j0Var = ((j0.a) th).f1505b;
                Iterator<b.d.b.d1.b1> it = q0Var2.f1141a.b().iterator();
                while (true) {
                    if (!it.hasNext()) {
                        break;
                    }
                    b.d.b.d1.b1 next = it.next();
                    if (next.b().contains(j0Var)) {
                        b1Var = next;
                        break;
                    }
                }
                if (b1Var != null) {
                    q0 q0Var3 = q0.this;
                    Objects.requireNonNull(q0Var3);
                    ScheduledExecutorService l = b.b.a.l();
                    List<b1.c> list = b1Var.f1418e;
                    if (list.isEmpty()) {
                        return;
                    }
                    final b1.c cVar = list.get(0);
                    q0Var3.n("Posting surface closed", new Throwable());
                    l.execute(new Runnable() { // from class: b.d.a.e.k
                        @Override // java.lang.Runnable
                        public final void run() {
                            b1.c.this.a(b1Var, b1.e.SESSION_ERROR_SURFACE_NEEDS_RESET);
                        }
                    });
                }
            } else if (th instanceof TimeoutException) {
                StringBuilder x2 = c.b.a.a.a.x("Unable to configure camera ");
                x2.append(q0.this.f1148h.f1176a);
                x2.append(", timeout!");
                b.d.b.u0.b("Camera2CameraImpl", x2.toString(), null);
            } else {
                throw new RuntimeException(th);
            }
        }

        /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object] */
        @Override // b.d.b.d1.k1.c.d
        public void onSuccess(Void r1) {
        }
    }

    /* compiled from: Camera2CameraImpl.java */
    /* loaded from: classes.dex */
    public final class c extends CameraManager.AvailabilityCallback implements c0.b {

        /* renamed from: a  reason: collision with root package name */
        public final String f1152a;

        /* renamed from: b  reason: collision with root package name */
        public boolean f1153b = true;

        public c(String str) {
            this.f1152a = str;
        }

        @Override // android.hardware.camera2.CameraManager.AvailabilityCallback
        public void onCameraAvailable(String str) {
            if (this.f1152a.equals(str)) {
                this.f1153b = true;
                if (q0.this.f1144d == e.PENDING_OPEN) {
                    q0.this.r(false);
                }
            }
        }

        @Override // android.hardware.camera2.CameraManager.AvailabilityCallback
        public void onCameraUnavailable(String str) {
            if (this.f1152a.equals(str)) {
                this.f1153b = false;
            }
        }
    }

    /* compiled from: Camera2CameraImpl.java */
    /* loaded from: classes.dex */
    public final class d implements w.a {
        public d() {
        }
    }

    /* compiled from: Camera2CameraImpl.java */
    /* loaded from: classes.dex */
    public enum e {
        INITIALIZED,
        PENDING_OPEN,
        OPENING,
        OPENED,
        CLOSING,
        REOPENING,
        RELEASING,
        RELEASED
    }

    /* compiled from: Camera2CameraImpl.java */
    /* loaded from: classes.dex */
    public final class f extends CameraDevice.StateCallback {

        /* renamed from: a  reason: collision with root package name */
        public final Executor f1163a;

        /* renamed from: b  reason: collision with root package name */
        public final ScheduledExecutorService f1164b;

        /* renamed from: c  reason: collision with root package name */
        public b f1165c;

        /* renamed from: d  reason: collision with root package name */
        public ScheduledFuture<?> f1166d;

        /* renamed from: e  reason: collision with root package name */
        public final a f1167e = new a(this);

        /* compiled from: Camera2CameraImpl.java */
        /* loaded from: classes.dex */
        public class a {

            /* renamed from: a  reason: collision with root package name */
            public long f1169a = -1;

            public a(f fVar) {
            }
        }

        /* compiled from: Camera2CameraImpl.java */
        /* loaded from: classes.dex */
        public class b implements Runnable {

            /* renamed from: b  reason: collision with root package name */
            public Executor f1170b;

            /* renamed from: c  reason: collision with root package name */
            public boolean f1171c = false;

            public b(Executor executor) {
                this.f1170b = executor;
            }

            @Override // java.lang.Runnable
            public void run() {
                this.f1170b.execute(new Runnable() { // from class: b.d.a.e.m
                    @Override // java.lang.Runnable
                    public final void run() {
                        q0.f.b bVar = q0.f.b.this;
                        if (bVar.f1171c) {
                            return;
                        }
                        b.j.b.d.k(q0.this.f1144d == q0.e.REOPENING, null);
                        q0.this.r(true);
                    }
                });
            }
        }

        public f(Executor executor, ScheduledExecutorService scheduledExecutorService) {
            this.f1163a = executor;
            this.f1164b = scheduledExecutorService;
        }

        public boolean a() {
            if (this.f1166d != null) {
                q0 q0Var = q0.this;
                StringBuilder x = c.b.a.a.a.x("Cancelling scheduled re-open: ");
                x.append(this.f1165c);
                q0Var.n(x.toString(), null);
                this.f1165c.f1171c = true;
                this.f1165c = null;
                this.f1166d.cancel(false);
                this.f1166d = null;
                return true;
            }
            return false;
        }

        public void b() {
            boolean z = true;
            b.j.b.d.k(this.f1165c == null, null);
            b.j.b.d.k(this.f1166d == null, null);
            a aVar = this.f1167e;
            Objects.requireNonNull(aVar);
            long uptimeMillis = SystemClock.uptimeMillis();
            long j = aVar.f1169a;
            if (j == -1) {
                aVar.f1169a = uptimeMillis;
            } else {
                if (uptimeMillis - j >= 10000) {
                    aVar.f1169a = -1L;
                    z = false;
                }
            }
            if (z) {
                this.f1165c = new b(this.f1163a);
                q0 q0Var = q0.this;
                StringBuilder x = c.b.a.a.a.x("Attempting camera re-open in 700ms: ");
                x.append(this.f1165c);
                q0Var.n(x.toString(), null);
                this.f1166d = this.f1164b.schedule(this.f1165c, 700L, TimeUnit.MILLISECONDS);
                return;
            }
            b.d.b.u0.b("Camera2CameraImpl", "Camera reopening attempted for 10000ms without success.", null);
            q0.this.w(e.INITIALIZED);
        }

        @Override // android.hardware.camera2.CameraDevice.StateCallback
        public void onClosed(CameraDevice cameraDevice) {
            q0.this.n("CameraDevice.onClosed()", null);
            boolean z = q0.this.i == null;
            b.j.b.d.k(z, "Unexpected onClose callback on camera device: " + cameraDevice);
            int ordinal = q0.this.f1144d.ordinal();
            if (ordinal != 4) {
                if (ordinal == 5) {
                    q0 q0Var = q0.this;
                    if (q0Var.j != 0) {
                        StringBuilder x = c.b.a.a.a.x("Camera closed due to error: ");
                        x.append(q0.p(q0.this.j));
                        q0Var.n(x.toString(), null);
                        b();
                        return;
                    }
                    q0Var.r(false);
                    return;
                } else if (ordinal != 6) {
                    StringBuilder x2 = c.b.a.a.a.x("Camera closed while in state: ");
                    x2.append(q0.this.f1144d);
                    throw new IllegalStateException(x2.toString());
                }
            }
            b.j.b.d.k(q0.this.q(), null);
            q0.this.o();
        }

        @Override // android.hardware.camera2.CameraDevice.StateCallback
        public void onDisconnected(CameraDevice cameraDevice) {
            q0.this.n("CameraDevice.onDisconnected()", null);
            onError(cameraDevice, 1);
        }

        @Override // android.hardware.camera2.CameraDevice.StateCallback
        public void onError(CameraDevice cameraDevice, int i) {
            q0 q0Var = q0.this;
            q0Var.i = cameraDevice;
            q0Var.j = i;
            int ordinal = q0Var.f1144d.ordinal();
            if (ordinal != 2 && ordinal != 3) {
                if (ordinal != 4) {
                    if (ordinal != 5) {
                        if (ordinal != 6) {
                            StringBuilder x = c.b.a.a.a.x("onError() should not be possible from state: ");
                            x.append(q0.this.f1144d);
                            throw new IllegalStateException(x.toString());
                        }
                    }
                }
                b.d.b.u0.b("Camera2CameraImpl", String.format("CameraDevice.onError(): %s failed with %s while in %s state. Will finish closing camera.", cameraDevice.getId(), q0.p(i), q0.this.f1144d.name()), null);
                q0.this.l(false);
                return;
            }
            b.d.b.u0.a("Camera2CameraImpl", String.format("CameraDevice.onError(): %s failed with %s while in %s state. Will attempt recovering from error.", cameraDevice.getId(), q0.p(i), q0.this.f1144d.name()), null);
            e eVar = e.REOPENING;
            boolean z = q0.this.f1144d == e.OPENING || q0.this.f1144d == e.OPENED || q0.this.f1144d == eVar;
            StringBuilder x2 = c.b.a.a.a.x("Attempt to handle open error from non open state: ");
            x2.append(q0.this.f1144d);
            b.j.b.d.k(z, x2.toString());
            if (i != 1 && i != 2 && i != 4) {
                StringBuilder x3 = c.b.a.a.a.x("Error observed on open (or opening) camera device ");
                x3.append(cameraDevice.getId());
                x3.append(": ");
                x3.append(q0.p(i));
                x3.append(" closing camera.");
                b.d.b.u0.b("Camera2CameraImpl", x3.toString(), null);
                q0.this.w(e.CLOSING);
                q0.this.l(false);
                return;
            }
            b.d.b.u0.a("Camera2CameraImpl", String.format("Attempt to reopen camera[%s] after error[%s]", cameraDevice.getId(), q0.p(i)), null);
            b.j.b.d.k(q0.this.j != 0, "Can only reopen camera device after error if the camera device is actually in an error state.");
            q0.this.w(eVar);
            q0.this.l(false);
        }

        @Override // android.hardware.camera2.CameraDevice.StateCallback
        public void onOpened(CameraDevice cameraDevice) {
            q0.this.n("CameraDevice.onOpened()", null);
            q0 q0Var = q0.this;
            q0Var.i = cameraDevice;
            Objects.requireNonNull(q0Var);
            try {
                Objects.requireNonNull(q0Var.f1146f);
                CaptureRequest.Builder createCaptureRequest = cameraDevice.createCaptureRequest(1);
                l1 l1Var = q0Var.f1146f.f1117h;
                Objects.requireNonNull(l1Var);
                l1Var.i = (MeteringRectangle[]) createCaptureRequest.get(CaptureRequest.CONTROL_AF_REGIONS);
                l1Var.j = (MeteringRectangle[]) createCaptureRequest.get(CaptureRequest.CONTROL_AE_REGIONS);
                l1Var.k = (MeteringRectangle[]) createCaptureRequest.get(CaptureRequest.CONTROL_AWB_REGIONS);
            } catch (CameraAccessException e2) {
                b.d.b.u0.b("Camera2CameraImpl", "fail to create capture request.", e2);
            }
            q0 q0Var2 = q0.this;
            q0Var2.j = 0;
            int ordinal = q0Var2.f1144d.ordinal();
            if (ordinal != 2) {
                if (ordinal != 4) {
                    if (ordinal != 5) {
                        if (ordinal != 6) {
                            StringBuilder x = c.b.a.a.a.x("onOpened() should not be possible from state: ");
                            x.append(q0.this.f1144d);
                            throw new IllegalStateException(x.toString());
                        }
                    }
                }
                b.j.b.d.k(q0.this.q(), null);
                q0.this.i.close();
                q0.this.i = null;
                return;
            }
            q0.this.w(e.OPENED);
            q0.this.s();
        }
    }

    public q0(b.d.a.e.y1.k kVar, String str, r0 r0Var, b.d.b.d1.c0 c0Var, Executor executor, Handler handler) {
        b.d.b.d1.r0<a0.a> r0Var2 = new b.d.b.d1.r0<>();
        this.f1145e = r0Var2;
        this.j = 0;
        this.l = b.d.b.d1.b1.a();
        this.m = new AtomicInteger(0);
        this.p = new LinkedHashMap();
        this.s = new HashSet();
        this.w = new HashSet();
        this.f1142b = kVar;
        this.r = c0Var;
        b.d.b.d1.k1.b.b bVar = new b.d.b.d1.k1.b.b(handler);
        b.d.b.d1.k1.b.d dVar = new b.d.b.d1.k1.b.d(executor);
        this.f1143c = dVar;
        this.f1147g = new f(dVar, bVar);
        this.f1141a = new b.d.b.d1.h1(str);
        r0Var2.f1586a.i(new r0.a<>(a0.a.CLOSED, null));
        h1 h1Var = new h1(dVar);
        this.u = h1Var;
        this.k = new g1();
        try {
            o0 o0Var = new o0(kVar.b(str), bVar, dVar, new d(), r0Var.f1180e);
            this.f1146f = o0Var;
            this.f1148h = r0Var;
            r0Var.g(o0Var);
            this.v = new t1.a(dVar, bVar, handler, h1Var, r0Var.f());
            c cVar = new c(str);
            this.q = cVar;
            synchronized (c0Var.f1434b) {
                boolean z = !c0Var.f1436d.containsKey(this);
                b.j.b.d.k(z, "Camera is already registered: " + this);
                c0Var.f1436d.put(this, new c0.a(null, dVar, cVar));
            }
            kVar.f1313a.a(dVar, cVar);
        } catch (b.d.a.e.y1.a e2) {
            throw b.b.a.d(e2);
        }
    }

    public static String p(int i) {
        return i != 0 ? i != 1 ? i != 2 ? i != 3 ? i != 4 ? i != 5 ? "UNKNOWN ERROR" : "ERROR_CAMERA_SERVICE" : "ERROR_CAMERA_DEVICE" : "ERROR_CAMERA_DISABLED" : "ERROR_MAX_CAMERAS_IN_USE" : "ERROR_CAMERA_IN_USE" : "ERROR_NONE";
    }

    @Override // b.d.b.a1.b
    public void c(final b.d.b.a1 a1Var) {
        this.f1143c.execute(new Runnable() { // from class: b.d.a.e.l
            @Override // java.lang.Runnable
            public final void run() {
                q0 q0Var = q0.this;
                b.d.b.a1 a1Var2 = a1Var;
                Objects.requireNonNull(q0Var);
                q0Var.n("Use case " + a1Var2 + " ACTIVE", null);
                try {
                    b.d.b.d1.h1 h1Var = q0Var.f1141a;
                    h1Var.e(a1Var2.d() + a1Var2.hashCode(), a1Var2.k);
                    b.d.b.d1.h1 h1Var2 = q0Var.f1141a;
                    h1Var2.h(a1Var2.d() + a1Var2.hashCode(), a1Var2.k);
                    q0Var.y();
                } catch (NullPointerException unused) {
                    q0Var.n("Failed to set already detached use case active", null);
                }
            }
        });
    }

    @Override // b.d.b.a1.b
    public void d(final b.d.b.a1 a1Var) {
        this.f1143c.execute(new Runnable() { // from class: b.d.a.e.u
            @Override // java.lang.Runnable
            public final void run() {
                q0 q0Var = q0.this;
                b.d.b.a1 a1Var2 = a1Var;
                Objects.requireNonNull(q0Var);
                q0Var.n("Use case " + a1Var2 + " RESET", null);
                b.d.b.d1.h1 h1Var = q0Var.f1141a;
                h1Var.h(a1Var2.d() + a1Var2.hashCode(), a1Var2.k);
                q0Var.v(false);
                q0Var.y();
                if (q0Var.f1144d == q0.e.OPENED) {
                    q0Var.s();
                }
            }
        });
    }

    @Override // b.d.b.a1.b
    public void e(final b.d.b.a1 a1Var) {
        this.f1143c.execute(new Runnable() { // from class: b.d.a.e.p
            @Override // java.lang.Runnable
            public final void run() {
                q0 q0Var = q0.this;
                b.d.b.a1 a1Var2 = a1Var;
                Objects.requireNonNull(q0Var);
                q0Var.n("Use case " + a1Var2 + " UPDATED", null);
                b.d.b.d1.h1 h1Var = q0Var.f1141a;
                h1Var.h(a1Var2.d() + a1Var2.hashCode(), a1Var2.k);
                q0Var.y();
            }
        });
    }

    @Override // b.d.b.a1.b
    public void f(final b.d.b.a1 a1Var) {
        this.f1143c.execute(new Runnable() { // from class: b.d.a.e.t
            @Override // java.lang.Runnable
            public final void run() {
                q0 q0Var = q0.this;
                b.d.b.a1 a1Var2 = a1Var;
                Objects.requireNonNull(q0Var);
                q0Var.n("Use case " + a1Var2 + " INACTIVE", null);
                b.d.b.d1.h1 h1Var = q0Var.f1141a;
                h1Var.g(a1Var2.d() + a1Var2.hashCode());
                q0Var.y();
            }
        });
    }

    @Override // b.d.b.d1.a0
    public b.d.b.d1.w g() {
        return this.f1146f;
    }

    @Override // b.d.b.d1.a0
    public void h(final Collection<b.d.b.a1> collection) {
        if (collection.isEmpty()) {
            return;
        }
        o0 o0Var = this.f1146f;
        synchronized (o0Var.f1113d) {
            o0Var.n++;
        }
        Iterator it = new ArrayList(collection).iterator();
        while (it.hasNext()) {
            b.d.b.a1 a1Var = (b.d.b.a1) it.next();
            if (!this.w.contains(a1Var.d() + a1Var.hashCode())) {
                this.w.add(a1Var.d() + a1Var.hashCode());
            }
        }
        try {
            this.f1143c.execute(new Runnable() { // from class: b.d.a.e.r
                @Override // java.lang.Runnable
                public final void run() {
                    q0 q0Var = q0.this;
                    try {
                        q0Var.x(collection);
                    } finally {
                        q0Var.f1146f.f();
                    }
                }
            });
        } catch (RejectedExecutionException e2) {
            n("Unable to attach use cases.", e2);
            this.f1146f.f();
        }
    }

    @Override // b.d.b.d1.a0
    public void i(final Collection<b.d.b.a1> collection) {
        if (collection.isEmpty()) {
            return;
        }
        Iterator it = new ArrayList(collection).iterator();
        while (it.hasNext()) {
            b.d.b.a1 a1Var = (b.d.b.a1) it.next();
            Set<String> set = this.w;
            if (set.contains(a1Var.d() + a1Var.hashCode())) {
                Set<String> set2 = this.w;
                set2.remove(a1Var.d() + a1Var.hashCode());
            }
        }
        this.f1143c.execute(new Runnable() { // from class: b.d.a.e.i
            @Override // java.lang.Runnable
            public final void run() {
                q0 q0Var = q0.this;
                Collection<b.d.b.a1> collection2 = collection;
                Objects.requireNonNull(q0Var);
                ArrayList arrayList = new ArrayList();
                for (b.d.b.a1 a1Var2 : collection2) {
                    b.d.b.d1.h1 h1Var = q0Var.f1141a;
                    if (h1Var.d(a1Var2.d() + a1Var2.hashCode())) {
                        b.d.b.d1.h1 h1Var2 = q0Var.f1141a;
                        h1Var2.f1486b.remove(a1Var2.d() + a1Var2.hashCode());
                        arrayList.add(a1Var2);
                    }
                }
                if (arrayList.isEmpty()) {
                    return;
                }
                StringBuilder x = c.b.a.a.a.x("Use cases [");
                x.append(TextUtils.join(", ", arrayList));
                x.append("] now DETACHED for camera");
                q0Var.n(x.toString(), null);
                Iterator it2 = arrayList.iterator();
                while (true) {
                    if (it2.hasNext()) {
                        if (((b.d.b.a1) it2.next()) instanceof b.d.b.w0) {
                            Objects.requireNonNull(q0Var.f1146f);
                            break;
                        }
                    } else {
                        break;
                    }
                }
                q0Var.k();
                if (q0Var.f1141a.b().isEmpty()) {
                    q0Var.f1146f.f();
                    q0Var.v(false);
                    q0Var.f1146f.l(false);
                    q0Var.k = new g1();
                    q0.e eVar = q0.e.CLOSING;
                    q0Var.n("Closing camera.", null);
                    int ordinal = q0Var.f1144d.ordinal();
                    if (ordinal != 1) {
                        if (ordinal != 2) {
                            if (ordinal == 3) {
                                q0Var.w(eVar);
                                q0Var.l(false);
                                return;
                            } else if (ordinal != 5) {
                                StringBuilder x2 = c.b.a.a.a.x("close() ignored due to being in state: ");
                                x2.append(q0Var.f1144d);
                                q0Var.n(x2.toString(), null);
                                return;
                            }
                        }
                        boolean a2 = q0Var.f1147g.a();
                        q0Var.w(eVar);
                        if (a2) {
                            b.j.b.d.k(q0Var.q(), null);
                            q0Var.o();
                            return;
                        }
                        return;
                    }
                    b.j.b.d.k(q0Var.i == null, null);
                    q0Var.w(q0.e.INITIALIZED);
                    return;
                }
                q0Var.y();
                q0Var.v(false);
                if (q0Var.f1144d == q0.e.OPENED) {
                    q0Var.s();
                }
            }
        });
    }

    @Override // b.d.b.d1.a0
    public b.d.b.d1.z j() {
        return this.f1148h;
    }

    public final void k() {
        b.d.b.d1.b1 b2 = this.f1141a.a().b();
        b.d.b.d1.f0 f0Var = b2.f1419f;
        int size = f0Var.a().size();
        int size2 = b2.b().size();
        if (b2.b().isEmpty()) {
            return;
        }
        if (!f0Var.a().isEmpty()) {
            if (size2 == 1 && size == 1) {
                u();
                return;
            } else if (size >= 2) {
                u();
                return;
            } else {
                b.d.b.u0.a("Camera2CameraImpl", c.b.a.a.a.k("mMeteringRepeating is ATTACHED, SessionConfig Surfaces: ", size2, ", CaptureConfig Surfaces: ", size), null);
                return;
            }
        }
        if (this.t == null) {
            this.t = new n1(this.f1148h.f1177b);
        }
        if (this.t != null) {
            b.d.b.d1.h1 h1Var = this.f1141a;
            StringBuilder sb = new StringBuilder();
            Objects.requireNonNull(this.t);
            sb.append("MeteringRepeating");
            sb.append(this.t.hashCode());
            h1Var.f(sb.toString(), this.t.f1106b);
            b.d.b.d1.h1 h1Var2 = this.f1141a;
            StringBuilder sb2 = new StringBuilder();
            Objects.requireNonNull(this.t);
            sb2.append("MeteringRepeating");
            sb2.append(this.t.hashCode());
            h1Var2.e(sb2.toString(), this.t.f1106b);
        }
    }

    /* JADX WARN: Removed duplicated region for block: B:30:0x012a  */
    /* JADX WARN: Removed duplicated region for block: B:42:? A[RETURN, SYNTHETIC] */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public void l(boolean z) {
        g1 g1Var;
        boolean z2 = this.f1144d == e.CLOSING || this.f1144d == e.RELEASING || (this.f1144d == e.REOPENING && this.j != 0);
        StringBuilder x = c.b.a.a.a.x("closeCamera should only be called in a CLOSING, RELEASING or REOPENING (with error) state. Current state: ");
        x.append(this.f1144d);
        x.append(" (error: ");
        x.append(p(this.j));
        x.append(")");
        b.j.b.d.k(z2, x.toString());
        if (Build.VERSION.SDK_INT < 29) {
            if ((this.f1148h.f() == 2) && this.j == 0) {
                final g1 g1Var2 = new g1();
                this.s.add(g1Var2);
                v(z);
                final SurfaceTexture surfaceTexture = new SurfaceTexture(0);
                surfaceTexture.setDefaultBufferSize(640, 480);
                final Surface surface = new Surface(surfaceTexture);
                final Runnable runnable = new Runnable() { // from class: b.d.a.e.q
                    @Override // java.lang.Runnable
                    public final void run() {
                        Surface surface2 = surface;
                        SurfaceTexture surfaceTexture2 = surfaceTexture;
                        surface2.release();
                        surfaceTexture2.release();
                    }
                };
                HashSet hashSet = new HashSet();
                HashSet hashSet2 = new HashSet();
                b.d.b.d1.u0 y = b.d.b.d1.u0.y();
                ArrayList arrayList = new ArrayList();
                b.d.b.d1.v0 v0Var = new b.d.b.d1.v0(new ArrayMap());
                ArrayList arrayList2 = new ArrayList();
                ArrayList arrayList3 = new ArrayList();
                ArrayList arrayList4 = new ArrayList();
                ArrayList arrayList5 = new ArrayList();
                hashSet.add(new b.d.b.d1.p0(surface));
                n("Start configAndClose.", null);
                ArrayList arrayList6 = new ArrayList(hashSet);
                ArrayList arrayList7 = new ArrayList(hashSet2);
                b.d.b.d1.w0 x2 = b.d.b.d1.w0.x(y);
                b.d.b.d1.g1 g1Var3 = b.d.b.d1.g1.f1479a;
                ArrayMap arrayMap = new ArrayMap();
                for (String str : v0Var.f1480b.keySet()) {
                    arrayMap.put(str, v0Var.a(str));
                }
                b.d.b.d1.b1 b1Var = new b.d.b.d1.b1(arrayList6, arrayList2, arrayList3, arrayList5, arrayList4, new b.d.b.d1.f0(arrayList7, x2, 1, arrayList, false, new b.d.b.d1.g1(arrayMap)));
                CameraDevice cameraDevice = this.i;
                Objects.requireNonNull(cameraDevice);
                g1Var2.h(b1Var, cameraDevice, this.v.a()).addListener(new Runnable() { // from class: b.d.a.e.s
                    @Override // java.lang.Runnable
                    public final void run() {
                        q0 q0Var = q0.this;
                        g1 g1Var4 = g1Var2;
                        Runnable runnable2 = runnable;
                        q0Var.s.remove(g1Var4);
                        q0Var.t(g1Var4, false).addListener(runnable2, b.b.a.f());
                    }
                }, this.f1143c);
                g1Var = this.k;
                if (g1Var.f1040b.isEmpty()) {
                    for (b.d.b.d1.f0 f0Var : g1Var.f1040b) {
                        for (b.d.b.d1.q qVar : f0Var.f1465f) {
                            qVar.a();
                        }
                    }
                    g1Var.f1040b.clear();
                    return;
                }
                return;
            }
        }
        v(z);
        g1Var = this.k;
        if (g1Var.f1040b.isEmpty()) {
        }
    }

    public final CameraDevice.StateCallback m() {
        ArrayList arrayList = new ArrayList(this.f1141a.a().b().f1415b);
        arrayList.add(this.f1147g);
        arrayList.add(this.u.f1066g);
        if (arrayList.isEmpty()) {
            return new c1();
        }
        if (arrayList.size() == 1) {
            return (CameraDevice.StateCallback) arrayList.get(0);
        }
        return new b1(arrayList);
    }

    public final void n(String str, Throwable th) {
        b.d.b.u0.a("Camera2CameraImpl", String.format("{%s} %s", toString(), str), th);
    }

    public void o() {
        e eVar = e.CLOSING;
        b.j.b.d.k(this.f1144d == e.RELEASING || this.f1144d == eVar, null);
        b.j.b.d.k(this.p.isEmpty(), null);
        this.i = null;
        if (this.f1144d == eVar) {
            w(e.INITIALIZED);
            return;
        }
        this.f1142b.f1313a.b(this.q);
        w(e.RELEASED);
        b.g.a.b<Void> bVar = this.o;
        if (bVar != null) {
            bVar.a(null);
            this.o = null;
        }
    }

    public boolean q() {
        return this.p.isEmpty() && this.s.isEmpty();
    }

    /* JADX WARN: Removed duplicated region for block: B:20:0x007f A[Catch: all -> 0x0109, TryCatch #2 {, blocks: (B:8:0x001b, B:10:0x0032, B:11:0x0063, B:13:0x0067, B:18:0x0077, B:20:0x007f, B:24:0x008e, B:26:0x00a4, B:27:0x00a7, B:17:0x0072), top: B:46:0x001b }] */
    /* JADX WARN: Removed duplicated region for block: B:26:0x00a4 A[Catch: all -> 0x0109, TryCatch #2 {, blocks: (B:8:0x001b, B:10:0x0032, B:11:0x0063, B:13:0x0067, B:18:0x0077, B:20:0x007f, B:24:0x008e, B:26:0x00a4, B:27:0x00a7, B:17:0x0072), top: B:46:0x001b }] */
    @SuppressLint({"MissingPermission"})
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public void r(boolean z) {
        boolean z2;
        if (!z) {
            this.f1147g.f1167e.f1169a = -1L;
        }
        this.f1147g.a();
        if (this.q.f1153b) {
            b.d.b.d1.c0 c0Var = this.r;
            synchronized (c0Var.f1434b) {
                c0.a aVar = c0Var.f1436d.get(this);
                b.j.b.d.h(aVar, "Camera must first be registered with registerCamera()");
                if (b.d.b.u0.c("CameraStateRegistry")) {
                    c0Var.f1433a.setLength(0);
                    c0Var.f1433a.append(String.format(Locale.US, "tryOpenCamera(%s) [Available Cameras: %d, Already Open: %b (Previous state: %s)]", this, Integer.valueOf(c0Var.f1437e), Boolean.valueOf(b.d.b.d1.c0.a(aVar.f1438a)), aVar.f1438a));
                }
                if (c0Var.f1437e <= 0 && !b.d.b.d1.c0.a(aVar.f1438a)) {
                    z2 = false;
                    if (b.d.b.u0.c("CameraStateRegistry")) {
                        StringBuilder sb = c0Var.f1433a;
                        Locale locale = Locale.US;
                        Object[] objArr = new Object[1];
                        objArr[0] = z2 ? "SUCCESS" : "FAIL";
                        sb.append(String.format(locale, " --> %s", objArr));
                        b.d.b.u0.a("CameraStateRegistry", c0Var.f1433a.toString(), null);
                    }
                    if (z2) {
                        c0Var.b();
                    }
                }
                aVar.f1438a = a0.a.OPENING;
                z2 = true;
                if (b.d.b.u0.c("CameraStateRegistry")) {
                }
                if (z2) {
                }
            }
            if (z2) {
                w(e.OPENING);
                n("Opening camera.", null);
                try {
                    this.f1142b.f1313a.d(this.f1148h.f1176a, this.f1143c, m());
                    return;
                } catch (b.d.a.e.y1.a e2) {
                    StringBuilder x = c.b.a.a.a.x("Unable to open camera due to ");
                    x.append(e2.getMessage());
                    n(x.toString(), null);
                    if (e2.f1243c != 10001) {
                        return;
                    }
                    w(e.INITIALIZED);
                    return;
                } catch (SecurityException e3) {
                    StringBuilder x2 = c.b.a.a.a.x("Unable to open camera due to ");
                    x2.append(e3.getMessage());
                    n(x2.toString(), null);
                    w(e.REOPENING);
                    this.f1147g.b();
                    return;
                }
            }
        }
        n("No cameras available. Waiting for available camera before opening camera.", null);
        w(e.PENDING_OPEN);
    }

    @Override // b.d.b.d1.a0
    public ListenableFuture<Void> release() {
        return b.e.a.d(new b.g.a.d() { // from class: b.d.a.e.o
            @Override // b.g.a.d
            public final Object a(final b.g.a.b bVar) {
                final q0 q0Var = q0.this;
                q0Var.f1143c.execute(new Runnable() { // from class: b.d.a.e.j
                    @Override // java.lang.Runnable
                    public final void run() {
                        final q0 q0Var2 = q0.this;
                        b.g.a.b bVar2 = bVar;
                        q0.e eVar = q0.e.RELEASING;
                        if (q0Var2.n == null) {
                            if (q0Var2.f1144d != q0.e.RELEASED) {
                                q0Var2.n = b.e.a.d(new b.g.a.d() { // from class: b.d.a.e.n
                                    @Override // b.g.a.d
                                    public final Object a(b.g.a.b bVar3) {
                                        q0 q0Var3 = q0.this;
                                        b.j.b.d.k(q0Var3.o == null, "Camera can only be released once, so release completer should be null on creation.");
                                        q0Var3.o = bVar3;
                                        return "Release[camera=" + q0Var3 + "]";
                                    }
                                });
                            } else {
                                q0Var2.n = b.d.b.d1.k1.c.g.c(null);
                            }
                        }
                        ListenableFuture<Void> listenableFuture = q0Var2.n;
                        switch (q0Var2.f1144d.ordinal()) {
                            case 0:
                            case 1:
                                b.j.b.d.k(q0Var2.i == null, null);
                                q0Var2.w(eVar);
                                b.j.b.d.k(q0Var2.q(), null);
                                q0Var2.o();
                                break;
                            case 2:
                            case 4:
                            case 5:
                            case 6:
                                boolean a2 = q0Var2.f1147g.a();
                                q0Var2.w(eVar);
                                if (a2) {
                                    b.j.b.d.k(q0Var2.q(), null);
                                    q0Var2.o();
                                    break;
                                }
                                break;
                            case 3:
                                q0Var2.w(eVar);
                                q0Var2.l(false);
                                break;
                            default:
                                StringBuilder x = c.b.a.a.a.x("release() ignored due to being in state: ");
                                x.append(q0Var2.f1144d);
                                q0Var2.n(x.toString(), null);
                                break;
                        }
                        b.d.b.d1.k1.c.g.e(listenableFuture, bVar2);
                    }
                });
                return "Release[request=" + q0Var.m.getAndIncrement() + "]";
            }
        });
    }

    public void s() {
        boolean z = true;
        b.j.b.d.k(this.f1144d == e.OPENED, null);
        b1.f a2 = this.f1141a.a();
        if (!a2.f1430h || !a2.f1429g) {
            z = false;
        }
        if (!z) {
            n("Unable to create capture session due to conflicting configurations", null);
            return;
        }
        g1 g1Var = this.k;
        b.d.b.d1.b1 b2 = a2.b();
        CameraDevice cameraDevice = this.i;
        Objects.requireNonNull(cameraDevice);
        ListenableFuture<Void> h2 = g1Var.h(b2, cameraDevice, this.v.a());
        h2.addListener(new g.d(h2, new b()), this.f1143c);
    }

    /* JADX WARN: Can't fix incorrect switch cases order, some code will duplicate */
    /* JADX WARN: Removed duplicated region for block: B:46:0x00e4 A[Catch: all -> 0x0166, TryCatch #3 {, blocks: (B:33:0x009e, B:34:0x00a4, B:57:0x0134, B:36:0x00a9, B:39:0x00af, B:43:0x00bb, B:42:0x00b4, B:44:0x00c0, B:46:0x00e4, B:47:0x00e8, B:49:0x00ec, B:50:0x00f7, B:51:0x00f9, B:53:0x00fb, B:54:0x0118, B:55:0x011b, B:56:0x0133), top: B:74:0x009e, inners: #2 }] */
    /* JADX WARN: Removed duplicated region for block: B:49:0x00ec A[Catch: all -> 0x0166, TryCatch #3 {, blocks: (B:33:0x009e, B:34:0x00a4, B:57:0x0134, B:36:0x00a9, B:39:0x00af, B:43:0x00bb, B:42:0x00b4, B:44:0x00c0, B:46:0x00e4, B:47:0x00e8, B:49:0x00ec, B:50:0x00f7, B:51:0x00f9, B:53:0x00fb, B:54:0x0118, B:55:0x011b, B:56:0x0133), top: B:74:0x009e, inners: #2 }] */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public ListenableFuture<Void> t(final g1 g1Var, boolean z) {
        ListenableFuture<Void> listenableFuture;
        g1.c cVar = g1.c.RELEASED;
        synchronized (g1Var.f1039a) {
            int ordinal = g1Var.l.ordinal();
            if (ordinal != 0) {
                if (ordinal != 1) {
                    if (ordinal != 2) {
                        if (ordinal != 3) {
                            if (ordinal == 4) {
                                if (g1Var.f1045g != null) {
                                    c.a c2 = g1Var.i.c();
                                    ArrayList arrayList = new ArrayList();
                                    for (b.d.a.d.b bVar : c2.f1012a) {
                                        Objects.requireNonNull(bVar);
                                    }
                                    if (!arrayList.isEmpty()) {
                                        try {
                                            g1Var.d(g1Var.j(arrayList));
                                        } catch (IllegalStateException e2) {
                                            b.d.b.u0.b("CaptureSession", "Unable to issue the request before close the capture session", e2);
                                        }
                                    }
                                }
                            }
                        }
                        t1 t1Var = g1Var.f1043e;
                        b.j.b.d.h(t1Var, "The Opener shouldn't null in state:" + g1Var.l);
                        g1Var.f1043e.a();
                        g1Var.l = g1.c.CLOSED;
                        g1Var.f1045g = null;
                    } else {
                        t1 t1Var2 = g1Var.f1043e;
                        b.j.b.d.h(t1Var2, "The Opener shouldn't null in state:" + g1Var.l);
                        g1Var.f1043e.a();
                    }
                }
                g1Var.l = cVar;
            } else {
                throw new IllegalStateException("close() should not be possible in state: " + g1Var.l);
            }
        }
        synchronized (g1Var.f1039a) {
            switch (g1Var.l.ordinal()) {
                case 0:
                    throw new IllegalStateException("release() should not be possible in state: " + g1Var.l);
                case 1:
                    g1Var.l = cVar;
                    listenableFuture = b.d.b.d1.k1.c.g.c(null);
                    break;
                case 2:
                    t1 t1Var3 = g1Var.f1043e;
                    b.j.b.d.h(t1Var3, "The Opener shouldn't null in state:" + g1Var.l);
                    g1Var.f1043e.a();
                    g1Var.l = cVar;
                    listenableFuture = b.d.b.d1.k1.c.g.c(null);
                    break;
                case 3:
                    g1Var.l = g1.c.RELEASING;
                    t1 t1Var4 = g1Var.f1043e;
                    b.j.b.d.h(t1Var4, "The Opener shouldn't null in state:" + g1Var.l);
                    if (g1Var.f1043e.a()) {
                        g1Var.b();
                        listenableFuture = b.d.b.d1.k1.c.g.c(null);
                        break;
                    }
                    if (g1Var.m == null) {
                        g1Var.m = b.e.a.d(new b.g.a.d() { // from class: b.d.a.e.v
                            @Override // b.g.a.d
                            public final Object a(b.g.a.b bVar2) {
                                String str;
                                g1 g1Var2 = g1.this;
                                synchronized (g1Var2.f1039a) {
                                    b.j.b.d.k(g1Var2.n == null, "Release completer expected to be null");
                                    g1Var2.n = bVar2;
                                    str = "Release[session=" + g1Var2 + "]";
                                }
                                return str;
                            }
                        });
                    }
                    listenableFuture = g1Var.m;
                    break;
                case 4:
                case 5:
                    p1 p1Var = g1Var.f1044f;
                    if (p1Var != null) {
                        if (z) {
                            try {
                                p1Var.e();
                            } catch (CameraAccessException e3) {
                                b.d.b.u0.b("CaptureSession", "Unable to abort captures.", e3);
                            }
                        }
                        g1Var.f1044f.close();
                    }
                    g1Var.l = g1.c.RELEASING;
                    t1 t1Var42 = g1Var.f1043e;
                    b.j.b.d.h(t1Var42, "The Opener shouldn't null in state:" + g1Var.l);
                    if (g1Var.f1043e.a()) {
                    }
                    if (g1Var.m == null) {
                    }
                    listenableFuture = g1Var.m;
                    break;
                case 6:
                    if (g1Var.m == null) {
                    }
                    listenableFuture = g1Var.m;
                    break;
                default:
                    listenableFuture = b.d.b.d1.k1.c.g.c(null);
                    break;
            }
        }
        StringBuilder x = c.b.a.a.a.x("Releasing session in state ");
        x.append(this.f1144d.name());
        n(x.toString(), null);
        this.p.put(g1Var, listenableFuture);
        a aVar = new a(g1Var);
        listenableFuture.addListener(new g.d(listenableFuture, aVar), b.b.a.f());
        return listenableFuture;
    }

    public String toString() {
        return String.format(Locale.US, "Camera@%x[id=%s]", Integer.valueOf(hashCode()), this.f1148h.f1176a);
    }

    public final void u() {
        if (this.t != null) {
            b.d.b.d1.h1 h1Var = this.f1141a;
            StringBuilder sb = new StringBuilder();
            Objects.requireNonNull(this.t);
            sb.append("MeteringRepeating");
            sb.append(this.t.hashCode());
            String sb2 = sb.toString();
            if (h1Var.f1486b.containsKey(sb2)) {
                h1.b bVar = h1Var.f1486b.get(sb2);
                bVar.f1488b = false;
                if (!bVar.f1489c) {
                    h1Var.f1486b.remove(sb2);
                }
            }
            b.d.b.d1.h1 h1Var2 = this.f1141a;
            StringBuilder sb3 = new StringBuilder();
            Objects.requireNonNull(this.t);
            sb3.append("MeteringRepeating");
            sb3.append(this.t.hashCode());
            h1Var2.g(sb3.toString());
            n1 n1Var = this.t;
            Objects.requireNonNull(n1Var);
            b.d.b.u0.a("MeteringRepeating", "MeteringRepeating clear!", null);
            b.d.b.d1.j0 j0Var = n1Var.f1105a;
            if (j0Var != null) {
                j0Var.a();
            }
            n1Var.f1105a = null;
            this.t = null;
        }
    }

    public void v(boolean z) {
        b.d.b.d1.b1 b1Var;
        List<b.d.b.d1.f0> unmodifiableList;
        b.j.b.d.k(this.k != null, null);
        n("Resetting Capture Session", null);
        g1 g1Var = this.k;
        synchronized (g1Var.f1039a) {
            b1Var = g1Var.f1045g;
        }
        synchronized (g1Var.f1039a) {
            unmodifiableList = Collections.unmodifiableList(g1Var.f1040b);
        }
        g1 g1Var2 = new g1();
        this.k = g1Var2;
        g1Var2.i(b1Var);
        this.k.d(unmodifiableList);
        t(g1Var, z);
    }

    /* JADX DEBUG: Failed to insert an additional move for type inference into block B:69:0x00e6 */
    /* JADX WARN: Multi-variable type inference failed */
    /* JADX WARN: Type inference failed for: r0v10 */
    /* JADX WARN: Type inference failed for: r0v15, types: [java.util.List] */
    /* JADX WARN: Type inference failed for: r0v16, types: [java.util.List] */
    /* JADX WARN: Type inference failed for: r0v19, types: [java.util.ArrayList] */
    public void w(e eVar) {
        a0.a aVar;
        a0.a aVar2;
        boolean z;
        ?? singletonList;
        a0.a aVar3 = a0.a.RELEASED;
        a0.a aVar4 = a0.a.PENDING_OPEN;
        a0.a aVar5 = a0.a.OPENING;
        StringBuilder x = c.b.a.a.a.x("Transitioning camera internal state: ");
        x.append(this.f1144d);
        x.append(" --> ");
        x.append(eVar);
        n(x.toString(), null);
        this.f1144d = eVar;
        switch (eVar.ordinal()) {
            case 0:
                aVar = a0.a.CLOSED;
                break;
            case 1:
                aVar = aVar4;
                break;
            case 2:
            case 5:
                aVar = aVar5;
                break;
            case 3:
                aVar = a0.a.OPEN;
                break;
            case 4:
                aVar = a0.a.CLOSING;
                break;
            case 6:
                aVar = a0.a.RELEASING;
                break;
            case 7:
                aVar = aVar3;
                break;
            default:
                throw new IllegalStateException("Unknown state: " + eVar);
        }
        b.d.b.d1.c0 c0Var = this.r;
        synchronized (c0Var.f1434b) {
            int i = c0Var.f1437e;
            if (aVar == aVar3) {
                c0.a remove = c0Var.f1436d.remove(this);
                if (remove != null) {
                    c0Var.b();
                    aVar2 = remove.f1438a;
                } else {
                    aVar2 = null;
                }
            } else {
                c0.a aVar6 = c0Var.f1436d.get(this);
                b.j.b.d.h(aVar6, "Cannot update state of camera which has not yet been registered. Register with CameraAvailabilityRegistry.registerCamera()");
                a0.a aVar7 = aVar6.f1438a;
                aVar6.f1438a = aVar;
                if (aVar == aVar5) {
                    if (!b.d.b.d1.c0.a(aVar) && aVar7 != aVar5) {
                        z = false;
                        b.j.b.d.k(z, "Cannot mark camera as opening until camera was successful at calling CameraAvailabilityRegistry.tryOpen()");
                    }
                    z = true;
                    b.j.b.d.k(z, "Cannot mark camera as opening until camera was successful at calling CameraAvailabilityRegistry.tryOpen()");
                }
                if (aVar7 != aVar) {
                    c0Var.b();
                }
                aVar2 = aVar7;
            }
            if (aVar2 != aVar) {
                if (i < 1 && c0Var.f1437e > 0) {
                    singletonList = new ArrayList();
                    for (Map.Entry<b.d.b.e0, c0.a> entry : c0Var.f1436d.entrySet()) {
                        if (entry.getValue().f1438a == aVar4) {
                            singletonList.add(entry.getValue());
                        }
                    }
                } else {
                    singletonList = (aVar != aVar4 || c0Var.f1437e <= 0) ? 0 : Collections.singletonList(c0Var.f1436d.get(this));
                }
                if (singletonList != 0) {
                    for (c0.a aVar8 : singletonList) {
                        Objects.requireNonNull(aVar8);
                        try {
                            Executor executor = aVar8.f1439b;
                            final c0.b bVar = aVar8.f1440c;
                            Objects.requireNonNull(bVar);
                            executor.execute(new Runnable() { // from class: b.d.b.d1.l
                                @Override // java.lang.Runnable
                                public final void run() {
                                    q0.c cVar = (q0.c) c0.b.this;
                                    if (b.d.a.e.q0.this.f1144d == q0.e.PENDING_OPEN) {
                                        b.d.a.e.q0.this.r(false);
                                    }
                                }
                            });
                        } catch (RejectedExecutionException e2) {
                            b.d.b.u0.b("CameraStateRegistry", "Unable to notify camera.", e2);
                        }
                    }
                }
            }
        }
        this.f1145e.f1586a.i(new r0.a<>(aVar, null));
    }

    public final void x(Collection<b.d.b.a1> collection) {
        boolean isEmpty = this.f1141a.b().isEmpty();
        ArrayList arrayList = new ArrayList();
        for (b.d.b.a1 a1Var : collection) {
            if (!this.f1141a.d(a1Var.d() + a1Var.hashCode())) {
                try {
                    this.f1141a.f(a1Var.d() + a1Var.hashCode(), a1Var.k);
                    arrayList.add(a1Var);
                } catch (NullPointerException unused) {
                    n("Failed to attach a detached use case", null);
                }
            }
        }
        if (arrayList.isEmpty()) {
            return;
        }
        StringBuilder x = c.b.a.a.a.x("Use cases [");
        x.append(TextUtils.join(", ", arrayList));
        x.append("] now ATTACHED");
        n(x.toString(), null);
        if (isEmpty) {
            this.f1146f.l(true);
            o0 o0Var = this.f1146f;
            synchronized (o0Var.f1113d) {
                o0Var.n++;
            }
        }
        k();
        y();
        v(false);
        e eVar = this.f1144d;
        e eVar2 = e.OPENED;
        if (eVar == eVar2) {
            s();
        } else {
            int ordinal = this.f1144d.ordinal();
            if (ordinal == 0) {
                r(false);
            } else if (ordinal != 4) {
                StringBuilder x2 = c.b.a.a.a.x("open() ignored due to being in state: ");
                x2.append(this.f1144d);
                n(x2.toString(), null);
            } else {
                w(e.REOPENING);
                if (!q() && this.j == 0) {
                    b.j.b.d.k(this.i != null, "Camera Device should be open if session close is not complete");
                    w(eVar2);
                    s();
                }
            }
        }
        Iterator it = arrayList.iterator();
        while (it.hasNext()) {
            b.d.b.a1 a1Var2 = (b.d.b.a1) it.next();
            if (a1Var2 instanceof b.d.b.w0) {
                Size size = a1Var2.f1385g;
                if (size != null) {
                    new Rational(size.getWidth(), size.getHeight());
                    Objects.requireNonNull(this.f1146f);
                    return;
                }
                return;
            }
        }
    }

    public void y() {
        b.d.b.d1.h1 h1Var = this.f1141a;
        Objects.requireNonNull(h1Var);
        b1.f fVar = new b1.f();
        ArrayList arrayList = new ArrayList();
        for (Map.Entry<String, h1.b> entry : h1Var.f1486b.entrySet()) {
            h1.b value = entry.getValue();
            if (value.f1489c && value.f1488b) {
                fVar.a(value.f1487a);
                arrayList.add(entry.getKey());
            }
        }
        b.d.b.u0.a("UseCaseAttachState", "Active and attached use case: " + arrayList + " for camera: " + h1Var.f1485a, null);
        if (fVar.f1430h && fVar.f1429g) {
            fVar.a(this.l);
            this.k.i(fVar.b());
            return;
        }
        this.k.i(this.l);
    }
}