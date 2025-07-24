package b.d.a.e;

import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CaptureRequest;
import android.view.Surface;
import com.google.common.util.concurrent.ListenableFuture;
import java.util.List;

/* compiled from: SynchronizedCaptureSession.java */
/* loaded from: classes.dex */
public interface p1 {

    /* compiled from: SynchronizedCaptureSession.java */
    /* loaded from: classes.dex */
    public static abstract class a {
        public void k(p1 p1Var) {
        }

        public void l(p1 p1Var) {
        }

        public void m(p1 p1Var) {
        }

        public void n(p1 p1Var) {
        }

        public void o(p1 p1Var) {
        }

        public void p(p1 p1Var) {
        }

        public void q(p1 p1Var, Surface surface) {
        }
    }

    a b();

    int c(List<CaptureRequest> list, CameraCaptureSession.CaptureCallback captureCallback);

    void close();

    b.d.a.e.y1.b d();

    void e();

    CameraDevice f();

    int g(CaptureRequest captureRequest, CameraCaptureSession.CaptureCallback captureCallback);

    void h();

    ListenableFuture<Void> j(String str);
}