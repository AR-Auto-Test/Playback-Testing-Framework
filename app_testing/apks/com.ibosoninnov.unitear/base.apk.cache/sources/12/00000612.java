package b.z;

import android.annotation.SuppressLint;
import android.view.View;

/* compiled from: ViewUtilsApi19.java */
/* loaded from: classes.dex */
public class t extends y {

    /* renamed from: c  reason: collision with root package name */
    public static boolean f2923c = true;

    @Override // b.z.y
    public void a(View view) {
    }

    @Override // b.z.y
    @SuppressLint({"NewApi"})
    public float b(View view) {
        if (f2923c) {
            try {
                return view.getTransitionAlpha();
            } catch (NoSuchMethodError unused) {
                f2923c = false;
            }
        }
        return view.getAlpha();
    }

    @Override // b.z.y
    public void c(View view) {
    }

    @Override // b.z.y
    @SuppressLint({"NewApi"})
    public void e(View view, float f2) {
        if (f2923c) {
            try {
                view.setTransitionAlpha(f2);
                return;
            } catch (NoSuchMethodError unused) {
                f2923c = false;
            }
        }
        view.setAlpha(f2);
    }
}