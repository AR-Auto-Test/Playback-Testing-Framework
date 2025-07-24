package b.d.a.e.y1.p;

import android.os.Build;
import b.d.b.d1.y0;

/* compiled from: ExcludedSupportedSizesQuirk.java */
/* loaded from: classes.dex */
public class e implements y0 {
    public static boolean a() {
        return "OnePlus".equalsIgnoreCase(Build.BRAND) && "OnePlus6".equalsIgnoreCase(Build.DEVICE);
    }

    public static boolean b() {
        return "OnePlus".equalsIgnoreCase(Build.BRAND) && "OnePlus6T".equalsIgnoreCase(Build.DEVICE);
    }
}