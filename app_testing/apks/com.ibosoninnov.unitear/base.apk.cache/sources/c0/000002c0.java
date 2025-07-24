package b.d.a.e.y1.p;

import android.os.Build;
import b.d.b.d1.y0;
import b.d.b.d1.z0;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/* compiled from: DeviceQuirks.java */
/* loaded from: classes.dex */
public class d {

    /* renamed from: a  reason: collision with root package name */
    public static final z0 f1346a;

    /* JADX WARN: Code restructure failed: missing block: B:15:0x0035, code lost:
        if (("Google".equals(r2) && "Pixel 3".equals(android.os.Build.MODEL)) != false) goto L59;
     */
    /* JADX WARN: Removed duplicated region for block: B:21:0x0042  */
    /* JADX WARN: Removed duplicated region for block: B:29:0x006d  */
    /* JADX WARN: Removed duplicated region for block: B:38:0x0092  */
    /* JADX WARN: Removed duplicated region for block: B:46:0x00b7  */
    /* JADX WARN: Removed duplicated region for block: B:54:0x00de  */
    /* JADX WARN: Removed duplicated region for block: B:61:0x0103  */
    static {
        boolean z;
        String str;
        Locale locale;
        ArrayList arrayList = new ArrayList();
        String str2 = Build.MANUFACTURER;
        boolean z2 = false;
        if (!("Google".equals(str2) && "Pixel 2".equals(Build.MODEL))) {
        }
        if (Build.VERSION.SDK_INT >= 26) {
            z = true;
            if (z) {
                arrayList.add(new f());
            }
            List<String> list = j.f1351a;
            str = Build.BRAND;
            if (!"SAMSUNG".equals(str.toUpperCase()) && j.f1351a.contains(Build.MODEL.toUpperCase())) {
                arrayList.add(new j());
            }
            List<String> list2 = h.f1349a;
            "GOOGLE".equals(str.toUpperCase());
            if (!e.a() || e.b()) {
                arrayList.add(new e());
            }
            if (!"SAMSUNG".equals(str2.toUpperCase()) && Build.MODEL.toUpperCase().startsWith("SM-A300")) {
                arrayList.add(new c());
            }
            List<String> list3 = i.f1350a;
            if (!"Google".equals(str2) && i.f1350a.contains(Build.DEVICE.toLowerCase(Locale.getDefault()))) {
                arrayList.add(new i());
            }
            locale = Locale.US;
            if ("SAMSUNG".equals(str2.toUpperCase(locale)) && Build.MODEL.toUpperCase(locale).startsWith("SM-A716")) {
                z2 = true;
            }
            if (z2) {
                arrayList.add(new k());
            }
            f1346a = new z0(arrayList);
        }
        z = false;
        if (z) {
        }
        List<String> list4 = j.f1351a;
        str = Build.BRAND;
        if (!"SAMSUNG".equals(str.toUpperCase()) && j.f1351a.contains(Build.MODEL.toUpperCase())) {
        }
        List<String> list22 = h.f1349a;
        "GOOGLE".equals(str.toUpperCase());
        if (!e.a() || e.b()) {
        }
        if (!"SAMSUNG".equals(str2.toUpperCase()) && Build.MODEL.toUpperCase().startsWith("SM-A300")) {
        }
        List<String> list32 = i.f1350a;
        if (!"Google".equals(str2) && i.f1350a.contains(Build.DEVICE.toLowerCase(Locale.getDefault()))) {
        }
        locale = Locale.US;
        if ("SAMSUNG".equals(str2.toUpperCase(locale))) {
            z2 = true;
        }
        if (z2) {
        }
        f1346a = new z0(arrayList);
    }

    public static <T extends y0> T a(Class<T> cls) {
        return (T) f1346a.a(cls);
    }
}