package b.z;

import android.annotation.SuppressLint;
import android.os.Build;
import android.view.View;

/* compiled from: ViewUtilsApi23.java */
/* loaded from: classes.dex */
public class w extends v {

    /* renamed from: g  reason: collision with root package name */
    public static boolean f2927g = true;

    @Override // b.z.y
    @SuppressLint({"NewApi"})
    public void f(View view, int i) {
        if (Build.VERSION.SDK_INT == 28) {
            super.f(view, i);
        } else if (f2927g) {
            try {
                view.setTransitionVisibility(i);
            } catch (NoSuchMethodError unused) {
                f2927g = false;
            }
        }
    }
}