package b.z;

import android.annotation.SuppressLint;
import android.view.View;

/* compiled from: ViewUtilsApi22.java */
/* loaded from: classes.dex */
public class v extends u {

    /* renamed from: f  reason: collision with root package name */
    public static boolean f2926f = true;

    @Override // b.z.y
    @SuppressLint({"NewApi"})
    public void d(View view, int i, int i2, int i3, int i4) {
        if (f2926f) {
            try {
                view.setLeftTopRightBottom(i, i2, i3, i4);
            } catch (NoSuchMethodError unused) {
                f2926f = false;
            }
        }
    }
}