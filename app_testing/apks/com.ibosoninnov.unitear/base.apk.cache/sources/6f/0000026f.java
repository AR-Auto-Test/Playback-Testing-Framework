package b.d.a.e;

import android.hardware.camera2.CameraCaptureSession;
import android.view.Surface;
import b.d.a.e.p1;
import java.util.ArrayList;
import java.util.List;

/* compiled from: SynchronizedCaptureSessionStateCallbacks.java */
/* loaded from: classes.dex */
public final class u1 extends p1.a {

    /* renamed from: a  reason: collision with root package name */
    public final List<p1.a> f1206a;

    /* compiled from: SynchronizedCaptureSessionStateCallbacks.java */
    /* loaded from: classes.dex */
    public static class a extends p1.a {

        /* renamed from: a  reason: collision with root package name */
        public final CameraCaptureSession.StateCallback f1207a;

        public a(List<CameraCaptureSession.StateCallback> list) {
            CameraCaptureSession.StateCallback z0Var;
            if (list.isEmpty()) {
                z0Var = new a1();
            } else if (list.size() == 1) {
                z0Var = list.get(0);
            } else {
                z0Var = new z0(list);
            }
            this.f1207a = z0Var;
        }

        @Override // b.d.a.e.p1.a
        public void k(p1 p1Var) {
            this.f1207a.onActive(p1Var.d().a());
        }

        @Override // b.d.a.e.p1.a
        public void l(p1 p1Var) {
            this.f1207a.onCaptureQueueEmpty(p1Var.d().a());
        }

        @Override // b.d.a.e.p1.a
        public void m(p1 p1Var) {
            this.f1207a.onClosed(p1Var.d().a());
        }

        @Override // b.d.a.e.p1.a
        public void n(p1 p1Var) {
            this.f1207a.onConfigureFailed(p1Var.d().a());
        }

        @Override // b.d.a.e.p1.a
        public void o(p1 p1Var) {
            this.f1207a.onConfigured(p1Var.d().a());
        }

        @Override // b.d.a.e.p1.a
        public void p(p1 p1Var) {
            this.f1207a.onReady(p1Var.d().a());
        }

        @Override // b.d.a.e.p1.a
        public void q(p1 p1Var, Surface surface) {
            this.f1207a.onSurfacePrepared(p1Var.d().a(), surface);
        }
    }

    public u1(List<p1.a> list) {
        ArrayList arrayList = new ArrayList();
        this.f1206a = arrayList;
        arrayList.addAll(list);
    }

    @Override // b.d.a.e.p1.a
    public void k(p1 p1Var) {
        for (p1.a aVar : this.f1206a) {
            aVar.k(p1Var);
        }
    }

    @Override // b.d.a.e.p1.a
    public void l(p1 p1Var) {
        for (p1.a aVar : this.f1206a) {
            aVar.l(p1Var);
        }
    }

    @Override // b.d.a.e.p1.a
    public void m(p1 p1Var) {
        for (p1.a aVar : this.f1206a) {
            aVar.m(p1Var);
        }
    }

    @Override // b.d.a.e.p1.a
    public void n(p1 p1Var) {
        for (p1.a aVar : this.f1206a) {
            aVar.n(p1Var);
        }
    }

    @Override // b.d.a.e.p1.a
    public void o(p1 p1Var) {
        for (p1.a aVar : this.f1206a) {
            aVar.o(p1Var);
        }
    }

    @Override // b.d.a.e.p1.a
    public void p(p1 p1Var) {
        for (p1.a aVar : this.f1206a) {
            aVar.p(p1Var);
        }
    }

    @Override // b.d.a.e.p1.a
    public void q(p1 p1Var, Surface surface) {
        for (p1.a aVar : this.f1206a) {
            aVar.q(p1Var, surface);
        }
    }
}