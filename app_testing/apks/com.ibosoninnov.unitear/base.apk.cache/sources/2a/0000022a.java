package b.d.a.e;

import android.hardware.camera2.CameraCaptureSession;
import java.util.List;
import java.util.Objects;

/* compiled from: CaptureCallbackConverter.java */
/* loaded from: classes.dex */
public final class f1 {
    public static void a(b.d.b.d1.q qVar, List<CameraCaptureSession.CaptureCallback> list) {
        if (!(qVar instanceof b.d.b.d1.r)) {
            if (qVar instanceof e1) {
                list.add(((e1) qVar).f1032a);
                return;
            } else {
                list.add(new d1(qVar));
                return;
            }
        }
        Objects.requireNonNull((b.d.b.d1.r) qVar);
        throw null;
    }
}