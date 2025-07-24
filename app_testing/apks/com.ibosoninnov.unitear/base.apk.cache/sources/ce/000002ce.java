package b.d.a.e;

import android.hardware.camera2.CameraCaptureSession;
import android.view.Surface;
import java.util.ArrayList;
import java.util.List;

/* compiled from: CameraCaptureSessionStateCallbacks.java */
/* loaded from: classes.dex */
public final class z0 extends CameraCaptureSession.StateCallback {

    /* renamed from: a  reason: collision with root package name */
    public final List<CameraCaptureSession.StateCallback> f1356a = new ArrayList();

    public z0(List<CameraCaptureSession.StateCallback> list) {
        for (CameraCaptureSession.StateCallback stateCallback : list) {
            if (!(stateCallback instanceof a1)) {
                this.f1356a.add(stateCallback);
            }
        }
    }

    @Override // android.hardware.camera2.CameraCaptureSession.StateCallback
    public void onActive(CameraCaptureSession cameraCaptureSession) {
        for (CameraCaptureSession.StateCallback stateCallback : this.f1356a) {
            stateCallback.onActive(cameraCaptureSession);
        }
    }

    @Override // android.hardware.camera2.CameraCaptureSession.StateCallback
    public void onCaptureQueueEmpty(CameraCaptureSession cameraCaptureSession) {
        for (CameraCaptureSession.StateCallback stateCallback : this.f1356a) {
            stateCallback.onCaptureQueueEmpty(cameraCaptureSession);
        }
    }

    @Override // android.hardware.camera2.CameraCaptureSession.StateCallback
    public void onClosed(CameraCaptureSession cameraCaptureSession) {
        for (CameraCaptureSession.StateCallback stateCallback : this.f1356a) {
            stateCallback.onClosed(cameraCaptureSession);
        }
    }

    @Override // android.hardware.camera2.CameraCaptureSession.StateCallback
    public void onConfigureFailed(CameraCaptureSession cameraCaptureSession) {
        for (CameraCaptureSession.StateCallback stateCallback : this.f1356a) {
            stateCallback.onConfigureFailed(cameraCaptureSession);
        }
    }

    @Override // android.hardware.camera2.CameraCaptureSession.StateCallback
    public void onConfigured(CameraCaptureSession cameraCaptureSession) {
        for (CameraCaptureSession.StateCallback stateCallback : this.f1356a) {
            stateCallback.onConfigured(cameraCaptureSession);
        }
    }

    @Override // android.hardware.camera2.CameraCaptureSession.StateCallback
    public void onReady(CameraCaptureSession cameraCaptureSession) {
        for (CameraCaptureSession.StateCallback stateCallback : this.f1356a) {
            stateCallback.onReady(cameraCaptureSession);
        }
    }

    @Override // android.hardware.camera2.CameraCaptureSession.StateCallback
    public void onSurfacePrepared(CameraCaptureSession cameraCaptureSession, Surface surface) {
        for (CameraCaptureSession.StateCallback stateCallback : this.f1356a) {
            stateCallback.onSurfacePrepared(cameraCaptureSession, surface);
        }
    }
}