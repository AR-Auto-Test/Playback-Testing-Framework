package b.d.a.e.y1;

import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CaptureRequest;
import java.util.List;
import java.util.concurrent.Executor;

/* compiled from: CameraCaptureSessionCompatApi28Impl.java */
/* loaded from: classes.dex */
public class c extends d {
    public c(CameraCaptureSession cameraCaptureSession) {
        super(cameraCaptureSession, null);
    }

    @Override // b.d.a.e.y1.d, b.d.a.e.y1.b.a
    public int a(CaptureRequest captureRequest, Executor executor, CameraCaptureSession.CaptureCallback captureCallback) {
        return this.f1293a.setSingleRepeatingRequest(captureRequest, executor, captureCallback);
    }

    @Override // b.d.a.e.y1.d, b.d.a.e.y1.b.a
    public int b(List<CaptureRequest> list, Executor executor, CameraCaptureSession.CaptureCallback captureCallback) {
        return this.f1293a.captureBurstRequests(list, executor, captureCallback);
    }
}