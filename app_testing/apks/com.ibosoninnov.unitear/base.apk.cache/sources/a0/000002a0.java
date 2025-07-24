package b.d.a.e.y1;

import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.params.SessionConfiguration;
import java.util.Objects;

/* compiled from: CameraDeviceCompatApi28Impl.java */
/* loaded from: classes.dex */
public class i extends h {
    /* JADX WARN: 'super' call moved to the top of the method (can break code semantics) */
    public i(CameraDevice cameraDevice) {
        super(cameraDevice, null);
        Objects.requireNonNull(cameraDevice);
    }

    @Override // b.d.a.e.y1.h, b.d.a.e.y1.f.a
    public void a(b.d.a.e.y1.o.g gVar) {
        SessionConfiguration sessionConfiguration = (SessionConfiguration) gVar.f1338a.c();
        Objects.requireNonNull(sessionConfiguration);
        this.f1310a.createCaptureSession(sessionConfiguration);
    }
}