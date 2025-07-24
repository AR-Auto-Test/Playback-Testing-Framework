package b.d.a.e.y1;

import android.hardware.camera2.CameraDevice;
import android.os.Handler;
import android.view.Surface;
import b.d.a.e.y1.f;
import b.d.b.u0;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/* compiled from: CameraDeviceCompatBaseImpl.java */
/* loaded from: classes.dex */
public class j implements f.a {

    /* renamed from: a  reason: collision with root package name */
    public final CameraDevice f1310a;

    /* renamed from: b  reason: collision with root package name */
    public final Object f1311b;

    /* compiled from: CameraDeviceCompatBaseImpl.java */
    /* loaded from: classes.dex */
    public static class a {

        /* renamed from: a  reason: collision with root package name */
        public final Handler f1312a;

        public a(Handler handler) {
            this.f1312a = handler;
        }
    }

    public j(CameraDevice cameraDevice, Object obj) {
        Objects.requireNonNull(cameraDevice);
        this.f1310a = cameraDevice;
        this.f1311b = obj;
    }

    public static void b(CameraDevice cameraDevice, b.d.a.e.y1.o.g gVar) {
        Objects.requireNonNull(cameraDevice);
        Objects.requireNonNull(gVar);
        Objects.requireNonNull(gVar.e());
        List<b.d.a.e.y1.o.b> c2 = gVar.c();
        if (c2 != null) {
            if (gVar.a() != null) {
                String id = cameraDevice.getId();
                for (b.d.a.e.y1.o.b bVar : c2) {
                    String b2 = bVar.f1330a.b();
                    if (b2 != null && !b2.isEmpty()) {
                        u0.d("CameraDeviceCompat", "Camera " + id + ": Camera doesn't support physicalCameraId " + b2 + ". Ignoring.", null);
                    }
                }
                return;
            }
            throw new IllegalArgumentException("Invalid executor");
        }
        throw new IllegalArgumentException("Invalid output configurations");
    }

    public static List<Surface> c(List<b.d.a.e.y1.o.b> list) {
        ArrayList arrayList = new ArrayList(list.size());
        for (b.d.a.e.y1.o.b bVar : list) {
            arrayList.add(bVar.f1330a.a());
        }
        return arrayList;
    }
}