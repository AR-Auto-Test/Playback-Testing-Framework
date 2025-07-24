package b.d.a.e;

import android.hardware.camera2.CameraCaptureSession;
import android.view.Surface;

/* compiled from: SynchronizedCaptureSessionBaseImpl.java */
/* loaded from: classes.dex */
public class q1 extends CameraCaptureSession.StateCallback {

    /* renamed from: a  reason: collision with root package name */
    public final /* synthetic */ r1 f1173a;

    public q1(r1 r1Var) {
        this.f1173a = r1Var;
    }

    @Override // android.hardware.camera2.CameraCaptureSession.StateCallback
    public void onActive(CameraCaptureSession cameraCaptureSession) {
        r1 r1Var = this.f1173a;
        if (r1Var.f1187g == null) {
            r1Var.f1187g = new b.d.a.e.y1.b(cameraCaptureSession, r1Var.f1183c);
        }
        r1 r1Var2 = this.f1173a;
        r1Var2.f1186f.k(r1Var2);
    }

    @Override // android.hardware.camera2.CameraCaptureSession.StateCallback
    public void onCaptureQueueEmpty(CameraCaptureSession cameraCaptureSession) {
        r1 r1Var = this.f1173a;
        if (r1Var.f1187g == null) {
            r1Var.f1187g = new b.d.a.e.y1.b(cameraCaptureSession, r1Var.f1183c);
        }
        r1 r1Var2 = this.f1173a;
        r1Var2.f1186f.l(r1Var2);
    }

    @Override // android.hardware.camera2.CameraCaptureSession.StateCallback
    public void onClosed(CameraCaptureSession cameraCaptureSession) {
        r1 r1Var = this.f1173a;
        if (r1Var.f1187g == null) {
            r1Var.f1187g = new b.d.a.e.y1.b(cameraCaptureSession, r1Var.f1183c);
        }
        r1 r1Var2 = this.f1173a;
        r1Var2.m(r1Var2);
    }

    /* JADX DEBUG: Finally have unexpected throw blocks count: 2, expect 1 */
    @Override // android.hardware.camera2.CameraCaptureSession.StateCallback
    public void onConfigureFailed(CameraCaptureSession cameraCaptureSession) {
        b.g.a.b<Void> bVar;
        try {
            r1 r1Var = this.f1173a;
            if (r1Var.f1187g == null) {
                r1Var.f1187g = new b.d.a.e.y1.b(cameraCaptureSession, r1Var.f1183c);
            }
            r1 r1Var2 = this.f1173a;
            r1Var2.n(r1Var2);
            synchronized (this.f1173a.f1181a) {
                b.j.b.d.h(this.f1173a.i, "OpenCaptureSession completer should not null");
                r1 r1Var3 = this.f1173a;
                bVar = r1Var3.i;
                r1Var3.i = null;
            }
            bVar.c(new IllegalStateException("onConfigureFailed"));
        } catch (Throwable th) {
            synchronized (this.f1173a.f1181a) {
                b.j.b.d.h(this.f1173a.i, "OpenCaptureSession completer should not null");
                r1 r1Var4 = this.f1173a;
                b.g.a.b<Void> bVar2 = r1Var4.i;
                r1Var4.i = null;
                bVar2.c(new IllegalStateException("onConfigureFailed"));
                throw th;
            }
        }
    }

    /* JADX DEBUG: Finally have unexpected throw blocks count: 2, expect 1 */
    @Override // android.hardware.camera2.CameraCaptureSession.StateCallback
    public void onConfigured(CameraCaptureSession cameraCaptureSession) {
        b.g.a.b<Void> bVar;
        try {
            r1 r1Var = this.f1173a;
            if (r1Var.f1187g == null) {
                r1Var.f1187g = new b.d.a.e.y1.b(cameraCaptureSession, r1Var.f1183c);
            }
            r1 r1Var2 = this.f1173a;
            r1Var2.o(r1Var2);
            synchronized (this.f1173a.f1181a) {
                b.j.b.d.h(this.f1173a.i, "OpenCaptureSession completer should not null");
                r1 r1Var3 = this.f1173a;
                bVar = r1Var3.i;
                r1Var3.i = null;
            }
            bVar.a(null);
        } catch (Throwable th) {
            synchronized (this.f1173a.f1181a) {
                b.j.b.d.h(this.f1173a.i, "OpenCaptureSession completer should not null");
                r1 r1Var4 = this.f1173a;
                b.g.a.b<Void> bVar2 = r1Var4.i;
                r1Var4.i = null;
                bVar2.a(null);
                throw th;
            }
        }
    }

    @Override // android.hardware.camera2.CameraCaptureSession.StateCallback
    public void onReady(CameraCaptureSession cameraCaptureSession) {
        r1 r1Var = this.f1173a;
        if (r1Var.f1187g == null) {
            r1Var.f1187g = new b.d.a.e.y1.b(cameraCaptureSession, r1Var.f1183c);
        }
        r1 r1Var2 = this.f1173a;
        r1Var2.f1186f.p(r1Var2);
    }

    @Override // android.hardware.camera2.CameraCaptureSession.StateCallback
    public void onSurfacePrepared(CameraCaptureSession cameraCaptureSession, Surface surface) {
        r1 r1Var = this.f1173a;
        if (r1Var.f1187g == null) {
            r1Var.f1187g = new b.d.a.e.y1.b(cameraCaptureSession, r1Var.f1183c);
        }
        r1 r1Var2 = this.f1173a;
        r1Var2.f1186f.q(r1Var2, surface);
    }
}