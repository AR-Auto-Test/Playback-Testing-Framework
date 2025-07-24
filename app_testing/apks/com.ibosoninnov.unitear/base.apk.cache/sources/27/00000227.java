package b.d.a.e;

import android.hardware.camera2.CameraCaptureSession;
import java.util.Objects;

/* compiled from: CaptureCallbackContainer.java */
/* loaded from: classes.dex */
public final class e1 extends b.d.b.d1.q {

    /* renamed from: a  reason: collision with root package name */
    public final CameraCaptureSession.CaptureCallback f1032a;

    public e1(CameraCaptureSession.CaptureCallback captureCallback) {
        Objects.requireNonNull(captureCallback, "captureCallback is null");
        this.f1032a = captureCallback;
    }
}