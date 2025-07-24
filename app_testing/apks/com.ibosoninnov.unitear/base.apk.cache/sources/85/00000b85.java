package c.e.b;

import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CaptureRequest;
import android.util.Log;
import android.util.Range;

/* compiled from: CamPreviewHelper.java */
/* loaded from: classes2.dex */
public class zb extends CameraCaptureSession.StateCallback {

    /* renamed from: a  reason: collision with root package name */
    public final /* synthetic */ yb f5512a;

    public zb(yb ybVar) {
        this.f5512a = ybVar;
    }

    @Override // android.hardware.camera2.CameraCaptureSession.StateCallback
    public void onConfigureFailed(CameraCaptureSession cameraCaptureSession) {
        Log.d("CCV2WithPreview", "Configuration Failed");
    }

    @Override // android.hardware.camera2.CameraCaptureSession.StateCallback
    public void onConfigured(CameraCaptureSession cameraCaptureSession) {
        yb ybVar = this.f5512a;
        if (ybVar.l == null) {
            return;
        }
        ybVar.k = cameraCaptureSession;
        try {
            ybVar.o.set(CaptureRequest.CONTROL_AF_MODE, 4);
            this.f5512a.o.set(CaptureRequest.CONTROL_AE_EXPOSURE_COMPENSATION, 1);
            yb ybVar2 = this.f5512a;
            Range<Integer> range = ybVar2.r;
            if (range != null) {
                ybVar2.o.set(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, range);
            }
            this.f5512a.o.set(CaptureRequest.CONTROL_CAPTURE_INTENT, 3);
            yb ybVar3 = this.f5512a;
            ybVar3.p = ybVar3.o.build();
            yb ybVar4 = this.f5512a;
            ybVar4.k.setRepeatingRequest(ybVar4.p, null, ybVar4.u);
        } catch (CameraAccessException e2) {
            e2.printStackTrace();
        }
    }
}