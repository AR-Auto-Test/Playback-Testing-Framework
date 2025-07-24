package b.d.a.e;

import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CaptureRequest;
import android.os.Handler;
import android.view.Surface;
import com.google.common.util.concurrent.ListenableFuture;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.Executor;
import java.util.concurrent.ScheduledExecutorService;

/* compiled from: SynchronizedCaptureSessionImpl.java */
/* loaded from: classes.dex */
public class s1 extends r1 {
    public final Object m;
    public final Set<String> n;
    public final ListenableFuture<Void> o;
    public b.g.a.b<Void> p;
    public final ListenableFuture<Void> q;
    public b.g.a.b<Void> r;
    public List<b.d.b.d1.j0> s;
    public ListenableFuture<Void> t;
    public ListenableFuture<List<Surface>> u;
    public boolean v;
    public final CameraCaptureSession.CaptureCallback w;

    /* compiled from: SynchronizedCaptureSessionImpl.java */
    /* loaded from: classes.dex */
    public class a extends CameraCaptureSession.CaptureCallback {
        public a() {
        }

        @Override // android.hardware.camera2.CameraCaptureSession.CaptureCallback
        public void onCaptureSequenceAborted(CameraCaptureSession cameraCaptureSession, int i) {
            b.g.a.b<Void> bVar = s1.this.p;
            if (bVar != null) {
                boolean z = true;
                bVar.f1808d = true;
                b.g.a.e<Void> eVar = bVar.f1806b;
                if ((eVar == null || !eVar.f1810c.cancel(true)) ? false : false) {
                    bVar.b();
                }
                s1.this.p = null;
            }
        }

        @Override // android.hardware.camera2.CameraCaptureSession.CaptureCallback
        public void onCaptureStarted(CameraCaptureSession cameraCaptureSession, CaptureRequest captureRequest, long j, long j2) {
            b.g.a.b<Void> bVar = s1.this.p;
            if (bVar != null) {
                bVar.a(null);
                s1.this.p = null;
            }
        }
    }

    public s1(Set<String> set, h1 h1Var, Executor executor, ScheduledExecutorService scheduledExecutorService, Handler handler) {
        super(h1Var, executor, scheduledExecutorService, handler);
        this.m = new Object();
        this.w = new a();
        this.n = set;
        if (set.contains("wait_for_request")) {
            this.o = b.e.a.d(new b.g.a.d() { // from class: b.d.a.e.i0
                @Override // b.g.a.d
                public final Object a(b.g.a.b bVar) {
                    s1 s1Var = s1.this;
                    s1Var.p = bVar;
                    return "StartStreamingFuture[session=" + s1Var + "]";
                }
            });
        } else {
            this.o = b.d.b.d1.k1.c.g.c(null);
        }
        if (set.contains("deferrableSurface_close")) {
            this.q = b.e.a.d(new b.g.a.d() { // from class: b.d.a.e.e0
                @Override // b.g.a.d
                public final Object a(b.g.a.b bVar) {
                    s1 s1Var = s1.this;
                    s1Var.r = bVar;
                    return "ClosingDeferrableSurfaceFuture[session=" + s1Var + "]";
                }
            });
        } else {
            this.q = b.d.b.d1.k1.c.g.c(null);
        }
    }

    @Override // b.d.a.e.r1, b.d.a.e.t1.b
    public ListenableFuture<List<Surface>> a(final List<b.d.b.d1.j0> list, final long j) {
        ListenableFuture<List<Surface>> d2;
        HashMap hashMap;
        synchronized (this.m) {
            this.s = list;
            List<ListenableFuture<Void>> emptyList = Collections.emptyList();
            if (this.n.contains("force_close")) {
                h1 h1Var = this.f1182b;
                synchronized (h1Var.f1061b) {
                    h1Var.f1065f.put(this, list);
                    hashMap = new HashMap(h1Var.f1065f);
                }
                ArrayList arrayList = new ArrayList();
                for (Map.Entry entry : hashMap.entrySet()) {
                    if (entry.getKey() != this && !Collections.disjoint((Collection) entry.getValue(), this.s)) {
                        arrayList.add((p1) entry.getKey());
                    }
                }
                emptyList = u("deferrableSurface_close", arrayList);
            }
            b.d.b.d1.k1.c.e c2 = b.d.b.d1.k1.c.e.a(b.d.b.d1.k1.c.g.g(emptyList)).c(new b.d.b.d1.k1.c.b() { // from class: b.d.a.e.g0
                @Override // b.d.b.d1.k1.c.b
                public final ListenableFuture apply(Object obj) {
                    return s1.this.x(list, j, (List) obj);
                }
            }, this.f1184d);
            this.u = c2;
            d2 = b.d.b.d1.k1.c.g.d(c2);
        }
        return d2;
    }

    @Override // b.d.a.e.r1, b.d.a.e.p1
    public void close() {
        t("Session call close()");
        if (this.n.contains("wait_for_request")) {
            synchronized (this.m) {
                if (!this.v) {
                    this.o.cancel(true);
                }
            }
        }
        this.o.addListener(new Runnable() { // from class: b.d.a.e.f0
            @Override // java.lang.Runnable
            public final void run() {
                s1.this.v();
            }
        }, this.f1184d);
    }

    @Override // b.d.a.e.r1, b.d.a.e.p1
    public int g(CaptureRequest captureRequest, CameraCaptureSession.CaptureCallback captureCallback) {
        int g2;
        if (this.n.contains("wait_for_request")) {
            synchronized (this.m) {
                this.v = true;
                g2 = super.g(captureRequest, new s0(Arrays.asList(this.w, captureCallback)));
            }
            return g2;
        }
        return super.g(captureRequest, captureCallback);
    }

    @Override // b.d.a.e.r1, b.d.a.e.t1.b
    public ListenableFuture<Void> i(final CameraDevice cameraDevice, final b.d.a.e.y1.o.g gVar) {
        ArrayList arrayList;
        ListenableFuture<Void> d2;
        synchronized (this.m) {
            h1 h1Var = this.f1182b;
            synchronized (h1Var.f1061b) {
                arrayList = new ArrayList(h1Var.f1063d);
            }
            b.d.b.d1.k1.c.e c2 = b.d.b.d1.k1.c.e.a(b.d.b.d1.k1.c.g.g(u("wait_for_request", arrayList))).c(new b.d.b.d1.k1.c.b() { // from class: b.d.a.e.h0
                @Override // b.d.b.d1.k1.c.b
                public final ListenableFuture apply(Object obj) {
                    return s1.this.w(cameraDevice, gVar, (List) obj);
                }
            }, b.b.a.f());
            this.t = c2;
            d2 = b.d.b.d1.k1.c.g.d(c2);
        }
        return d2;
    }

    @Override // b.d.a.e.r1, b.d.a.e.p1
    public ListenableFuture<Void> j(String str) {
        str.hashCode();
        if (str.equals("wait_for_request")) {
            return b.d.b.d1.k1.c.g.d(this.o);
        }
        if (!str.equals("deferrableSurface_close")) {
            return b.d.b.d1.k1.c.g.c(null);
        }
        return b.d.b.d1.k1.c.g.d(this.q);
    }

    @Override // b.d.a.e.r1, b.d.a.e.p1.a
    public void m(p1 p1Var) {
        s();
        t("onClosed()");
        super.m(p1Var);
    }

    @Override // b.d.a.e.r1, b.d.a.e.p1.a
    public void o(p1 p1Var) {
        ArrayList arrayList;
        p1 p1Var2;
        ArrayList arrayList2;
        p1 p1Var3;
        t("Session onConfigured()");
        if (this.n.contains("force_close")) {
            LinkedHashSet<p1> linkedHashSet = new LinkedHashSet();
            h1 h1Var = this.f1182b;
            synchronized (h1Var.f1061b) {
                arrayList2 = new ArrayList(h1Var.f1064e);
            }
            Iterator it = arrayList2.iterator();
            while (it.hasNext() && (p1Var3 = (p1) it.next()) != p1Var) {
                linkedHashSet.add(p1Var3);
            }
            for (p1 p1Var4 : linkedHashSet) {
                p1Var4.b().n(p1Var4);
            }
        }
        super.o(p1Var);
        if (this.n.contains("force_close")) {
            LinkedHashSet<p1> linkedHashSet2 = new LinkedHashSet();
            h1 h1Var2 = this.f1182b;
            synchronized (h1Var2.f1061b) {
                arrayList = new ArrayList(h1Var2.f1062c);
            }
            Iterator it2 = arrayList.iterator();
            while (it2.hasNext() && (p1Var2 = (p1) it2.next()) != p1Var) {
                linkedHashSet2.add(p1Var2);
            }
            for (p1 p1Var5 : linkedHashSet2) {
                p1Var5.b().m(p1Var5);
            }
        }
    }

    public void s() {
        synchronized (this.m) {
            if (this.s == null) {
                t("deferrableSurface == null, maybe forceClose, skip close");
                return;
            }
            if (this.n.contains("deferrableSurface_close")) {
                for (b.d.b.d1.j0 j0Var : this.s) {
                    j0Var.a();
                }
                t("deferrableSurface closed");
                y();
            }
        }
    }

    @Override // b.d.a.e.r1, b.d.a.e.t1.b
    public boolean stop() {
        boolean stop;
        synchronized (this.m) {
            if (r()) {
                s();
            } else {
                ListenableFuture<Void> listenableFuture = this.t;
                if (listenableFuture != null) {
                    listenableFuture.cancel(true);
                }
                ListenableFuture<List<Surface>> listenableFuture2 = this.u;
                if (listenableFuture2 != null) {
                    listenableFuture2.cancel(true);
                }
                y();
            }
            stop = super.stop();
        }
        return stop;
    }

    public void t(String str) {
        b.d.b.u0.a("SyncCaptureSessionImpl", "[" + this + "] " + str, null);
    }

    public final List<ListenableFuture<Void>> u(String str, List<p1> list) {
        ArrayList arrayList = new ArrayList();
        for (p1 p1Var : list) {
            arrayList.add(p1Var.j(str));
        }
        return arrayList;
    }

    public /* synthetic */ void v() {
        t("Session call super.close()");
        super.close();
    }

    public /* synthetic */ ListenableFuture w(CameraDevice cameraDevice, b.d.a.e.y1.o.g gVar, List list) {
        return super.i(cameraDevice, gVar);
    }

    public /* synthetic */ ListenableFuture x(List list, long j, List list2) {
        return super.a(list, j);
    }

    public void y() {
        if (this.n.contains("deferrableSurface_close")) {
            h1 h1Var = this.f1182b;
            synchronized (h1Var.f1061b) {
                h1Var.f1065f.remove(this);
            }
            b.g.a.b<Void> bVar = this.r;
            if (bVar != null) {
                bVar.a(null);
            }
        }
    }
}