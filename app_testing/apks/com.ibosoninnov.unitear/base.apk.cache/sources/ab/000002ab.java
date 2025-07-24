package b.d.a.e.y1;

import android.content.Context;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.os.Handler;
import b.d.a.e.y1.f;
import b.d.a.e.y1.k;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.Executor;

/* compiled from: CameraManagerCompatBaseImpl.java */
/* loaded from: classes.dex */
public class n implements k.b {

    /* renamed from: a  reason: collision with root package name */
    public final CameraManager f1324a;

    /* renamed from: b  reason: collision with root package name */
    public final Object f1325b;

    /* compiled from: CameraManagerCompatBaseImpl.java */
    /* loaded from: classes.dex */
    public static final class a {

        /* renamed from: a  reason: collision with root package name */
        public final Map<CameraManager.AvailabilityCallback, k.a> f1326a = new HashMap();

        /* renamed from: b  reason: collision with root package name */
        public final Handler f1327b;

        public a(Handler handler) {
            this.f1327b = handler;
        }
    }

    public n(Context context, Object obj) {
        this.f1324a = (CameraManager) context.getSystemService("camera");
        this.f1325b = obj;
    }

    @Override // b.d.a.e.y1.k.b
    public void a(Executor executor, CameraManager.AvailabilityCallback availabilityCallback) {
        if (executor != null) {
            k.a aVar = null;
            a aVar2 = (a) this.f1325b;
            if (availabilityCallback != null) {
                synchronized (aVar2.f1326a) {
                    aVar = aVar2.f1326a.get(availabilityCallback);
                    if (aVar == null) {
                        aVar = new k.a(executor, availabilityCallback);
                        aVar2.f1326a.put(availabilityCallback, aVar);
                    }
                }
            }
            this.f1324a.registerAvailabilityCallback(aVar, aVar2.f1327b);
            return;
        }
        throw new IllegalArgumentException("executor was null");
    }

    @Override // b.d.a.e.y1.k.b
    public void b(CameraManager.AvailabilityCallback availabilityCallback) {
        k.a aVar;
        if (availabilityCallback != null) {
            a aVar2 = (a) this.f1325b;
            synchronized (aVar2.f1326a) {
                aVar = aVar2.f1326a.remove(availabilityCallback);
            }
        } else {
            aVar = null;
        }
        if (aVar != null) {
            synchronized (aVar.f1317c) {
                aVar.f1318d = true;
            }
        }
        this.f1324a.unregisterAvailabilityCallback(aVar);
    }

    @Override // b.d.a.e.y1.k.b
    public CameraCharacteristics c(String str) {
        try {
            return this.f1324a.getCameraCharacteristics(str);
        } catch (CameraAccessException e2) {
            Set<Integer> set = b.d.a.e.y1.a.f1242b;
            throw new b.d.a.e.y1.a(e2);
        }
    }

    @Override // b.d.a.e.y1.k.b
    public void d(String str, Executor executor, CameraDevice.StateCallback stateCallback) {
        Objects.requireNonNull(executor);
        Objects.requireNonNull(stateCallback);
        try {
            this.f1324a.openCamera(str, new f.b(executor, stateCallback), ((a) this.f1325b).f1327b);
        } catch (CameraAccessException e2) {
            Set<Integer> set = b.d.a.e.y1.a.f1242b;
            throw new b.d.a.e.y1.a(e2);
        }
    }
}