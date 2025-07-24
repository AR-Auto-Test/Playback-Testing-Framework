package b.d.a.e.y1;

import android.content.Context;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.os.Build;
import java.util.Set;
import java.util.concurrent.Executor;

/* compiled from: CameraManagerCompatApi28Impl.java */
/* loaded from: classes.dex */
public class l extends n {
    public l(Context context) {
        super(context, null);
    }

    @Override // b.d.a.e.y1.n, b.d.a.e.y1.k.b
    public void a(Executor executor, CameraManager.AvailabilityCallback availabilityCallback) {
        this.f1324a.registerAvailabilityCallback(executor, availabilityCallback);
    }

    @Override // b.d.a.e.y1.n, b.d.a.e.y1.k.b
    public void b(CameraManager.AvailabilityCallback availabilityCallback) {
        this.f1324a.unregisterAvailabilityCallback(availabilityCallback);
    }

    @Override // b.d.a.e.y1.n, b.d.a.e.y1.k.b
    public CameraCharacteristics c(String str) {
        try {
            try {
                return this.f1324a.getCameraCharacteristics(str);
            } catch (CameraAccessException e2) {
                Set<Integer> set = a.f1242b;
                throw new a(e2);
            }
        } catch (RuntimeException e3) {
            if (e(e3)) {
                throw new a(10001, e3);
            }
            throw e3;
        }
    }

    @Override // b.d.a.e.y1.n, b.d.a.e.y1.k.b
    public void d(String str, Executor executor, CameraDevice.StateCallback stateCallback) {
        try {
            this.f1324a.openCamera(str, executor, stateCallback);
        } catch (CameraAccessException e2) {
            Set<Integer> set = a.f1242b;
            throw new a(e2);
        } catch (IllegalArgumentException e3) {
            throw e3;
        } catch (SecurityException e4) {
        } catch (RuntimeException e5) {
            if (e(e5)) {
                throw new a(10001, e5);
            }
            throw e5;
        }
    }

    public final boolean e(Throwable th) {
        StackTraceElement[] stackTrace;
        if (Build.VERSION.SDK_INT == 28) {
            return (!th.getClass().equals(RuntimeException.class) || (stackTrace = th.getStackTrace()) == null || stackTrace.length < 0) ? false : "_enableShutterSound".equals(stackTrace[0].getMethodName());
        }
        return false;
    }
}