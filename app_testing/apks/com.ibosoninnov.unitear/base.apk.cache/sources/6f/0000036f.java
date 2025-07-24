package b.d.b.e1.h.a;

import android.os.Build;
import b.d.b.d1.z0;
import java.util.ArrayList;
import java.util.Locale;

/* compiled from: DeviceQuirks.java */
/* loaded from: classes.dex */
public class a {

    /* renamed from: a  reason: collision with root package name */
    public static final z0 f1608a;

    static {
        ArrayList arrayList = new ArrayList();
        if (c.f1609a.contains(Build.DEVICE.toLowerCase(Locale.getDefault()))) {
            arrayList.add(new c());
        }
        String str = Build.BRAND;
        boolean z = true;
        if (!("HUAWEI".equalsIgnoreCase(str) && "SNE-LX1".equalsIgnoreCase(Build.MODEL))) {
            if (!("HONOR".equalsIgnoreCase(str) && "STK-LX1".equalsIgnoreCase(Build.MODEL))) {
                z = false;
            }
        }
        if (z) {
            arrayList.add(new b());
        }
        f1608a = new z0(arrayList);
    }
}