package b.d.a.e;

import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CaptureRequest;
import android.os.Handler;
import android.view.Surface;
import b.d.a.e.p1;
import b.d.a.e.t1;
import b.d.b.d1.j0;
import b.d.b.d1.k1.c.g;
import b.d.b.d1.k1.c.h;
import com.google.common.util.concurrent.ListenableFuture;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.CancellationException;
import java.util.concurrent.Executor;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

/* compiled from: SynchronizedCaptureSessionBaseImpl.java */
/* loaded from: classes.dex */
public class r1 extends p1.a implements p1, t1.b {

    /* renamed from: b  reason: collision with root package name */
    public final h1 f1182b;

    /* renamed from: c  reason: collision with root package name */
    public final Handler f1183c;

    /* renamed from: d  reason: collision with root package name */
    public final Executor f1184d;

    /* renamed from: e  reason: collision with root package name */
    public final ScheduledExecutorService f1185e;

    /* renamed from: f  reason: collision with root package name */
    public p1.a f1186f;

    /* renamed from: g  reason: collision with root package name */
    public b.d.a.e.y1.b f1187g;

    /* renamed from: h  reason: collision with root package name */
    public ListenableFuture<Void> f1188h;
    public b.g.a.b<Void> i;
    public ListenableFuture<List<Surface>> j;

    /* renamed from: a  reason: collision with root package name */
    public final Object f1181a = new Object();
    public boolean k = false;
    public boolean l = false;

    public r1(h1 h1Var, Executor executor, ScheduledExecutorService scheduledExecutorService, Handler handler) {
        this.f1182b = h1Var;
        this.f1183c = handler;
        this.f1184d = executor;
        this.f1185e = scheduledExecutorService;
    }

    @Override // b.d.a.e.t1.b
    public ListenableFuture<List<Surface>> a(final List<b.d.b.d1.j0> list, final long j) {
        synchronized (this.f1181a) {
            if (this.l) {
                return new h.a(new CancellationException("Opener is disabled"));
            }
            final Executor executor = this.f1184d;
            final ScheduledExecutorService scheduledExecutorService = this.f1185e;
            final ArrayList arrayList = new ArrayList();
            for (b.d.b.d1.j0 j0Var : list) {
                arrayList.add(j0Var.c());
            }
            b.d.b.d1.k1.c.e c2 = b.d.b.d1.k1.c.e.a(b.e.a.d(new b.g.a.d() { // from class: b.d.b.d1.g
                @Override // b.g.a.d
                public final Object a(final b.g.a.b bVar) {
                    List list2 = arrayList;
                    ScheduledExecutorService scheduledExecutorService2 = scheduledExecutorService;
                    final Executor executor2 = executor;
                    final long j2 = j;
                    boolean z = r6;
                    final ListenableFuture g2 = b.d.b.d1.k1.c.g.g(list2);
                    ScheduledFuture<?> schedule = scheduledExecutorService2.schedule(new Runnable() { // from class: b.d.b.d1.h
                        @Override // java.lang.Runnable
                        public final void run() {
                            Executor executor3 = executor2;
                            final ListenableFuture listenableFuture = g2;
                            final b.g.a.b bVar2 = bVar;
                            final long j3 = j2;
                            executor3.execute(new Runnable() { // from class: b.d.b.d1.f
                                @Override // java.lang.Runnable
                                public final void run() {
                                    ListenableFuture listenableFuture2 = ListenableFuture.this;
                                    b.g.a.b bVar3 = bVar2;
                                    long j4 = j3;
                                    if (listenableFuture2.isDone()) {
                                        return;
                                    }
                                    bVar3.c(new TimeoutException(c.b.a.a.a.l("Cannot complete surfaceList within ", j4)));
                                    listenableFuture2.cancel(true);
                                }
                            });
                        }
                    }, j2, TimeUnit.MILLISECONDS);
                    Runnable runnable = new Runnable() { // from class: b.d.b.d1.e
                        @Override // java.lang.Runnable
                        public final void run() {
                            ListenableFuture.this.cancel(true);
                        }
                    };
                    b.g.a.f<Void> fVar = bVar.f1807c;
                    if (fVar != null) {
                        fVar.addListener(runnable, executor2);
                    }
                    ((b.d.b.d1.k1.c.i) g2).addListener(new g.d(g2, new k0(z, bVar, schedule)), executor2);
                    return "surfaceList";
                }
            })).c(new b.d.b.d1.k1.c.b() { // from class: b.d.a.e.c0
                @Override // b.d.b.d1.k1.c.b
                public final ListenableFuture apply(Object obj) {
                    r1 r1Var = r1.this;
                    List list2 = list;
                    List list3 = (List) obj;
                    Objects.requireNonNull(r1Var);
                    b.d.b.u0.a("SyncCaptureSessionBase", "[" + r1Var + "] getSurface...done", null);
                    if (list3.contains(null)) {
                        return new h.a(new j0.a("Surface closed", (b.d.b.d1.j0) list2.get(list3.indexOf(null))));
                    }
                    if (list3.isEmpty()) {
                        return new h.a(new IllegalArgumentException("Unable to open capture session without surfaces"));
                    }
                    return b.d.b.d1.k1.c.g.c(list3);
                }
            }, this.f1184d);
            this.j = c2;
            return b.d.b.d1.k1.c.g.d(c2);
        }
    }

    @Override // b.d.a.e.p1
    public p1.a b() {
        return this;
    }

    @Override // b.d.a.e.p1
    public int c(List<CaptureRequest> list, CameraCaptureSession.CaptureCallback captureCallback) {
        b.j.b.d.h(this.f1187g, "Need to call openCaptureSession before using this API.");
        b.d.a.e.y1.b bVar = this.f1187g;
        return bVar.f1244a.b(list, this.f1184d, captureCallback);
    }

    @Override // b.d.a.e.p1
    public void close() {
        b.j.b.d.h(this.f1187g, "Need to call openCaptureSession before using this API.");
        h1 h1Var = this.f1182b;
        synchronized (h1Var.f1061b) {
            h1Var.f1063d.add(this);
        }
        this.f1187g.a().close();
    }

    @Override // b.d.a.e.p1
    public b.d.a.e.y1.b d() {
        Objects.requireNonNull(this.f1187g);
        return this.f1187g;
    }

    @Override // b.d.a.e.p1
    public void e() {
        b.j.b.d.h(this.f1187g, "Need to call openCaptureSession before using this API.");
        this.f1187g.a().abortCaptures();
    }

    @Override // b.d.a.e.p1
    public CameraDevice f() {
        Objects.requireNonNull(this.f1187g);
        return this.f1187g.a().getDevice();
    }

    @Override // b.d.a.e.p1
    public int g(CaptureRequest captureRequest, CameraCaptureSession.CaptureCallback captureCallback) {
        b.j.b.d.h(this.f1187g, "Need to call openCaptureSession before using this API.");
        b.d.a.e.y1.b bVar = this.f1187g;
        return bVar.f1244a.a(captureRequest, this.f1184d, captureCallback);
    }

    @Override // b.d.a.e.p1
    public void h() {
        b.j.b.d.h(this.f1187g, "Need to call openCaptureSession before using this API.");
        this.f1187g.a().stopRepeating();
    }

    @Override // b.d.a.e.t1.b
    public ListenableFuture<Void> i(CameraDevice cameraDevice, final b.d.a.e.y1.o.g gVar) {
        synchronized (this.f1181a) {
            if (this.l) {
                return new h.a(new CancellationException("Opener is disabled"));
            }
            h1 h1Var = this.f1182b;
            synchronized (h1Var.f1061b) {
                h1Var.f1064e.add(this);
            }
            final b.d.a.e.y1.f fVar = new b.d.a.e.y1.f(cameraDevice, this.f1183c);
            ListenableFuture<Void> d2 = b.e.a.d(new b.g.a.d() { // from class: b.d.a.e.b0
                @Override // b.g.a.d
                public final Object a(b.g.a.b bVar) {
                    String str;
                    r1 r1Var = r1.this;
                    b.d.a.e.y1.f fVar2 = fVar;
                    b.d.a.e.y1.o.g gVar2 = gVar;
                    synchronized (r1Var.f1181a) {
                        b.j.b.d.k(r1Var.i == null, "The openCaptureSessionCompleter can only set once!");
                        r1Var.i = bVar;
                        fVar2.f1298a.a(gVar2);
                        str = "openCaptureSession[session=" + r1Var + "]";
                    }
                    return str;
                }
            });
            this.f1188h = d2;
            return b.d.b.d1.k1.c.g.d(d2);
        }
    }

    @Override // b.d.a.e.p1
    public ListenableFuture<Void> j(String str) {
        return b.d.b.d1.k1.c.g.c(null);
    }

    @Override // b.d.a.e.p1.a
    public void k(p1 p1Var) {
        this.f1186f.k(p1Var);
    }

    @Override // b.d.a.e.p1.a
    public void l(p1 p1Var) {
        this.f1186f.l(p1Var);
    }

    @Override // b.d.a.e.p1.a
    public void m(final p1 p1Var) {
        ListenableFuture<Void> listenableFuture;
        synchronized (this.f1181a) {
            if (this.k) {
                listenableFuture = null;
            } else {
                this.k = true;
                b.j.b.d.h(this.f1188h, "Need to call openCaptureSession before using this API.");
                listenableFuture = this.f1188h;
            }
        }
        if (listenableFuture != null) {
            listenableFuture.addListener(new Runnable() { // from class: b.d.a.e.d0
                @Override // java.lang.Runnable
                public final void run() {
                    r1 r1Var = r1.this;
                    p1 p1Var2 = p1Var;
                    h1 h1Var = r1Var.f1182b;
                    synchronized (h1Var.f1061b) {
                        h1Var.f1062c.remove(r1Var);
                        h1Var.f1063d.remove(r1Var);
                    }
                    r1Var.f1186f.m(p1Var2);
                }
            }, b.b.a.f());
        }
    }

    @Override // b.d.a.e.p1.a
    public void n(p1 p1Var) {
        h1 h1Var = this.f1182b;
        synchronized (h1Var.f1061b) {
            h1Var.f1064e.remove(this);
        }
        this.f1186f.n(p1Var);
    }

    @Override // b.d.a.e.p1.a
    public void o(p1 p1Var) {
        h1 h1Var = this.f1182b;
        synchronized (h1Var.f1061b) {
            h1Var.f1062c.add(this);
            h1Var.f1064e.remove(this);
        }
        this.f1186f.o(p1Var);
    }

    @Override // b.d.a.e.p1.a
    public void p(p1 p1Var) {
        this.f1186f.p(p1Var);
    }

    @Override // b.d.a.e.p1.a
    public void q(p1 p1Var, Surface surface) {
        this.f1186f.q(p1Var, surface);
    }

    public boolean r() {
        boolean z;
        synchronized (this.f1181a) {
            z = this.f1188h != null;
        }
        return z;
    }

    @Override // b.d.a.e.t1.b
    public boolean stop() {
        boolean z;
        try {
            synchronized (this.f1181a) {
                if (!this.l) {
                    ListenableFuture<List<Surface>> listenableFuture = this.j;
                    r1 = listenableFuture != null ? listenableFuture : null;
                    this.l = true;
                }
                z = !r();
            }
            return z;
        } finally {
            if (r1 != null) {
                r1.cancel(true);
            }
        }
    }
}