package b.d.a.e.y1;

import android.content.Context;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import java.util.Set;
import java.util.concurrent.Executor;

/* compiled from: CameraManagerCompatApi29Impl.java */
/* loaded from: classes.dex */
public class m extends l {
    public m(Context context) {
        super(context);
    }

    @Override // b.d.a.e.y1.l, b.d.a.e.y1.n, b.d.a.e.y1.k.b
    public CameraCharacteristics c(String str) {
        try {
            return this.f1324a.getCameraCharacteristics(str);
        } catch (CameraAccessException e2) {
            Set<Integer> set = a.f1242b;
            throw new a(e2);
        }
    }

    @Override // b.d.a.e.y1.l, b.d.a.e.y1.n, b.d.a.e.y1.k.b
    public void d(String str, Executor executor, CameraDevice.StateCallback stateCallback) {
        try {
            this.f1324a.openCamera(str, executor, stateCallback);
        } catch (CameraAccessException e2) {
            Set<Integer> set = a.f1242b;
            throw new a(e2);
        }
    }
}