package b.d.a.e.y1;

import android.hardware.camera2.CameraCharacteristics;
import java.util.HashMap;
import java.util.Map;

/* compiled from: CameraCharacteristicsCompat.java */
/* loaded from: classes.dex */
public class e {

    /* renamed from: a  reason: collision with root package name */
    public final Map<CameraCharacteristics.Key<?>, Object> f1296a = new HashMap();

    /* renamed from: b  reason: collision with root package name */
    public final CameraCharacteristics f1297b;

    public e(CameraCharacteristics cameraCharacteristics) {
        this.f1297b = cameraCharacteristics;
    }

    public <T> T a(CameraCharacteristics.Key<T> key) {
        synchronized (this) {
            T t = (T) this.f1296a.get(key);
            if (t != null) {
                return t;
            }
            T t2 = (T) this.f1297b.get(key);
            if (t2 != null) {
                this.f1296a.put(key, t2);
            }
            return t2;
        }
    }
}