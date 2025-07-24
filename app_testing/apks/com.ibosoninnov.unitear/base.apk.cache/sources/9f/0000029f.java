package b.d.a.e.y1;

import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.params.InputConfiguration;
import android.os.Handler;
import b.d.a.e.y1.b;
import b.d.a.e.y1.j;
import java.util.List;
import java.util.Objects;

/* compiled from: CameraDeviceCompatApi24Impl.java */
/* loaded from: classes.dex */
public class h extends g {
    public h(CameraDevice cameraDevice, Object obj) {
        super(cameraDevice, obj);
    }

    @Override // b.d.a.e.y1.f.a
    public void a(b.d.a.e.y1.o.g gVar) {
        j.b(this.f1310a, gVar);
        b.c cVar = new b.c(gVar.a(), gVar.e());
        List<b.d.a.e.y1.o.b> c2 = gVar.c();
        j.a aVar = (j.a) this.f1311b;
        Objects.requireNonNull(aVar);
        Handler handler = aVar.f1312a;
        b.d.a.e.y1.o.a b2 = gVar.b();
        if (b2 != null) {
            InputConfiguration inputConfiguration = (InputConfiguration) b2.f1328a.a();
            Objects.requireNonNull(inputConfiguration);
            this.f1310a.createReprocessableCaptureSessionByConfigurations(inputConfiguration, b.d.a.e.y1.o.g.f(c2), cVar, handler);
        } else if (gVar.d() == 1) {
            this.f1310a.createConstrainedHighSpeedCaptureSession(j.c(c2), cVar, handler);
        } else {
            this.f1310a.createCaptureSessionByOutputConfigurations(b.d.a.e.y1.o.g.f(c2), cVar, handler);
        }
    }
}