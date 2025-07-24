package b.d.a.e;

import android.hardware.camera2.CaptureResult;

/* compiled from: Camera2CameraCaptureResult.java */
/* loaded from: classes.dex */
public class n0 implements b.d.b.d1.t {

    /* renamed from: a  reason: collision with root package name */
    public final b.d.b.d1.g1 f1103a;

    /* renamed from: b  reason: collision with root package name */
    public final CaptureResult f1104b;

    public n0(b.d.b.d1.g1 g1Var, CaptureResult captureResult) {
        this.f1103a = g1Var;
        this.f1104b = captureResult;
    }

    public long a() {
        Long l = (Long) this.f1104b.get(CaptureResult.SENSOR_TIMESTAMP);
        if (l == null) {
            return -1L;
        }
        return l.longValue();
    }
}