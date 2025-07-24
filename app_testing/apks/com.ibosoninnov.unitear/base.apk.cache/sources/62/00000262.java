package b.d.a.e;

import android.hardware.camera2.CameraCharacteristics;
import android.util.Log;
import java.util.Objects;

/* compiled from: Camera2CameraInfoImpl.java */
/* loaded from: classes.dex */
public final class r0 implements b.d.b.d1.z {

    /* renamed from: a  reason: collision with root package name */
    public final String f1176a;

    /* renamed from: b  reason: collision with root package name */
    public final b.d.a.e.y1.e f1177b;

    /* renamed from: c  reason: collision with root package name */
    public final Object f1178c = new Object();

    /* renamed from: d  reason: collision with root package name */
    public o0 f1179d;

    /* renamed from: e  reason: collision with root package name */
    public final b.d.b.d1.z0 f1180e;

    public r0(String str, b.d.a.e.y1.e eVar) {
        Objects.requireNonNull(str);
        this.f1176a = str;
        this.f1177b = eVar;
        this.f1180e = b.b.a.h(eVar);
    }

    @Override // b.d.b.i0
    public int a() {
        return d(0);
    }

    @Override // b.d.b.d1.z
    public String b() {
        return this.f1176a;
    }

    @Override // b.d.b.d1.z
    public Integer c() {
        Integer num = (Integer) this.f1177b.a(CameraCharacteristics.LENS_FACING);
        Objects.requireNonNull(num);
        int intValue = num.intValue();
        if (intValue != 0) {
            return intValue != 1 ? null : 1;
        }
        return 0;
    }

    @Override // b.d.b.i0
    public int d(int i) {
        Integer num = (Integer) this.f1177b.a(CameraCharacteristics.SENSOR_ORIENTATION);
        Objects.requireNonNull(num);
        Integer valueOf = Integer.valueOf(num.intValue());
        int o = b.b.a.o(i);
        Integer c2 = c();
        boolean z = true;
        return b.b.a.i(o, valueOf.intValue(), (c2 == null || 1 != c2.intValue()) ? false : false);
    }

    @Override // b.d.b.i0
    public boolean e() {
        Boolean bool = (Boolean) this.f1177b.a(CameraCharacteristics.FLASH_INFO_AVAILABLE);
        Objects.requireNonNull(bool);
        return bool.booleanValue();
    }

    public int f() {
        Integer num = (Integer) this.f1177b.a(CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL);
        Objects.requireNonNull(num);
        return num.intValue();
    }

    public void g(o0 o0Var) {
        String str;
        synchronized (this.f1178c) {
            this.f1179d = o0Var;
        }
        int f2 = f();
        boolean z = true;
        if (f2 == 0) {
            str = "INFO_SUPPORTED_HARDWARE_LEVEL_LIMITED";
        } else if (f2 == 1) {
            str = "INFO_SUPPORTED_HARDWARE_LEVEL_FULL";
        } else if (f2 == 2) {
            str = "INFO_SUPPORTED_HARDWARE_LEVEL_LEGACY";
        } else if (f2 != 3) {
            str = f2 != 4 ? c.b.a.a.a.j("Unknown value: ", f2) : "INFO_SUPPORTED_HARDWARE_LEVEL_EXTERNAL";
        } else {
            str = "INFO_SUPPORTED_HARDWARE_LEVEL_3";
        }
        String q = c.b.a.a.a.q("Device Level: ", str);
        if (b.d.b.u0.f1672a > 4) {
            "Camera2CameraInfo".length();
            if (!Log.isLoggable("Camera2CameraInfo", 4)) {
                z = false;
            }
        }
        if (z) {
            "Camera2CameraInfo".length();
            Log.i("Camera2CameraInfo", q, null);
        }
    }
}