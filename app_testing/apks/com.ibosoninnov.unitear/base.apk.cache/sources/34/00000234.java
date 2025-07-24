package b.d.a.e;

import android.hardware.camera2.CameraDevice;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.Executor;

/* compiled from: CaptureSessionRepository.java */
/* loaded from: classes.dex */
public class h1 {

    /* renamed from: a  reason: collision with root package name */
    public final Executor f1060a;

    /* renamed from: b  reason: collision with root package name */
    public final Object f1061b = new Object();

    /* renamed from: c  reason: collision with root package name */
    public final Set<p1> f1062c = new LinkedHashSet();

    /* renamed from: d  reason: collision with root package name */
    public final Set<p1> f1063d = new LinkedHashSet();

    /* renamed from: e  reason: collision with root package name */
    public final Set<p1> f1064e = new LinkedHashSet();

    /* renamed from: f  reason: collision with root package name */
    public final Map<p1, List<b.d.b.d1.j0>> f1065f = new HashMap();

    /* renamed from: g  reason: collision with root package name */
    public final CameraDevice.StateCallback f1066g = new a();

    /* compiled from: CaptureSessionRepository.java */
    /* loaded from: classes.dex */
    public class a extends CameraDevice.StateCallback {
        public a() {
        }

        @Override // android.hardware.camera2.CameraDevice.StateCallback
        public void onDisconnected(CameraDevice cameraDevice) {
            h1.this.f1060a.execute(new y(this));
        }

        @Override // android.hardware.camera2.CameraDevice.StateCallback
        public void onError(CameraDevice cameraDevice, int i) {
            h1.this.f1060a.execute(new y(this));
        }

        @Override // android.hardware.camera2.CameraDevice.StateCallback
        public void onOpened(CameraDevice cameraDevice) {
        }
    }

    public h1(Executor executor) {
        this.f1060a = executor;
    }
}