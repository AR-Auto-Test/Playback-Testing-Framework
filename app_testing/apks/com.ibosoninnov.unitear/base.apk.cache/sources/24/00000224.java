package b.d.a.e;

import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CaptureFailure;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.TotalCaptureResult;
import java.util.Objects;

/* compiled from: CaptureCallbackAdapter.java */
/* loaded from: classes.dex */
public final class d1 extends CameraCaptureSession.CaptureCallback {

    /* renamed from: a  reason: collision with root package name */
    public final b.d.b.d1.q f1028a;

    public d1(b.d.b.d1.q qVar) {
        Objects.requireNonNull(qVar, "cameraCaptureCallback is null");
        this.f1028a = qVar;
    }

    @Override // android.hardware.camera2.CameraCaptureSession.CaptureCallback
    public void onCaptureCompleted(CameraCaptureSession cameraCaptureSession, CaptureRequest captureRequest, TotalCaptureResult totalCaptureResult) {
        b.d.b.d1.g1 g1Var;
        super.onCaptureCompleted(cameraCaptureSession, captureRequest, totalCaptureResult);
        Object tag = captureRequest.getTag();
        if (tag != null) {
            b.j.b.d.e(tag instanceof b.d.b.d1.g1, "The tagBundle object from the CaptureResult is not a TagBundle object.");
            g1Var = (b.d.b.d1.g1) tag;
        } else {
            g1Var = b.d.b.d1.g1.f1479a;
        }
        this.f1028a.b(new n0(g1Var, totalCaptureResult));
    }

    @Override // android.hardware.camera2.CameraCaptureSession.CaptureCallback
    public void onCaptureFailed(CameraCaptureSession cameraCaptureSession, CaptureRequest captureRequest, CaptureFailure captureFailure) {
        super.onCaptureFailed(cameraCaptureSession, captureRequest, captureFailure);
        this.f1028a.c(new b.d.b.d1.s(1));
    }
}