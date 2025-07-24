package b.d.a.e;

import android.hardware.camera2.CameraDevice;
import java.util.ArrayList;
import java.util.List;

/* compiled from: CameraDeviceStateCallbacks.java */
/* loaded from: classes.dex */
public final class b1 extends CameraDevice.StateCallback {

    /* renamed from: a  reason: collision with root package name */
    public final List<CameraDevice.StateCallback> f1020a = new ArrayList();

    public b1(List<CameraDevice.StateCallback> list) {
        for (CameraDevice.StateCallback stateCallback : list) {
            if (!(stateCallback instanceof c1)) {
                this.f1020a.add(stateCallback);
            }
        }
    }

    @Override // android.hardware.camera2.CameraDevice.StateCallback
    public void onClosed(CameraDevice cameraDevice) {
        for (CameraDevice.StateCallback stateCallback : this.f1020a) {
            stateCallback.onClosed(cameraDevice);
        }
    }

    @Override // android.hardware.camera2.CameraDevice.StateCallback
    public void onDisconnected(CameraDevice cameraDevice) {
        for (CameraDevice.StateCallback stateCallback : this.f1020a) {
            stateCallback.onDisconnected(cameraDevice);
        }
    }

    @Override // android.hardware.camera2.CameraDevice.StateCallback
    public void onError(CameraDevice cameraDevice, int i) {
        for (CameraDevice.StateCallback stateCallback : this.f1020a) {
            stateCallback.onError(cameraDevice, i);
        }
    }

    @Override // android.hardware.camera2.CameraDevice.StateCallback
    public void onOpened(CameraDevice cameraDevice) {
        for (CameraDevice.StateCallback stateCallback : this.f1020a) {
            stateCallback.onOpened(cameraDevice);
        }
    }
}