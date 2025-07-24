package b.d.a.e;

import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.TotalCaptureResult;
import b.d.a.d.a;
import b.d.a.e.w1;

/* compiled from: CropRegionZoomImpl.java */
/* loaded from: classes.dex */
public final class i1 implements w1.b {

    /* renamed from: a  reason: collision with root package name */
    public final b.d.a.e.y1.e f1071a;

    public i1(b.d.a.e.y1.e eVar) {
        this.f1071a = eVar;
    }

    @Override // b.d.a.e.w1.b
    public void a(TotalCaptureResult totalCaptureResult) {
    }

    @Override // b.d.a.e.w1.b
    public void b(a.C0012a c0012a) {
    }

    @Override // b.d.a.e.w1.b
    public float c() {
        Float f2 = (Float) this.f1071a.a(CameraCharacteristics.SCALER_AVAILABLE_MAX_DIGITAL_ZOOM);
        if (f2 == null) {
            return 1.0f;
        }
        return f2.floatValue();
    }

    @Override // b.d.a.e.w1.b
    public float d() {
        return 1.0f;
    }

    @Override // b.d.a.e.w1.b
    public void e() {
    }
}