package b.d.a.e;

import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.TotalCaptureResult;
import android.util.Range;
import b.d.a.d.a;
import b.d.a.e.w1;

/* compiled from: AndroidRZoomImpl.java */
/* loaded from: classes.dex */
public final class l0 implements w1.b {

    /* renamed from: a  reason: collision with root package name */
    public final b.d.a.e.y1.e f1089a;

    /* renamed from: b  reason: collision with root package name */
    public final Range<Float> f1090b;

    /* renamed from: c  reason: collision with root package name */
    public float f1091c = 1.0f;

    public l0(b.d.a.e.y1.e eVar) {
        this.f1089a = eVar;
        this.f1090b = (Range) eVar.a(CameraCharacteristics.CONTROL_ZOOM_RATIO_RANGE);
    }

    @Override // b.d.a.e.w1.b
    public void a(TotalCaptureResult totalCaptureResult) {
    }

    @Override // b.d.a.e.w1.b
    public void b(a.C0012a c0012a) {
        c0012a.b(CaptureRequest.CONTROL_ZOOM_RATIO, Float.valueOf(this.f1091c));
    }

    @Override // b.d.a.e.w1.b
    public float c() {
        return this.f1090b.getUpper().floatValue();
    }

    @Override // b.d.a.e.w1.b
    public float d() {
        return this.f1090b.getLower().floatValue();
    }

    @Override // b.d.a.e.w1.b
    public void e() {
        this.f1091c = 1.0f;
    }
}