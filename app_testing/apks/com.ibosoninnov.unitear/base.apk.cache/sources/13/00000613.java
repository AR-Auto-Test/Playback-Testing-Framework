package b.z;

import android.annotation.SuppressLint;
import android.graphics.Matrix;
import android.view.View;

/* compiled from: ViewUtilsApi21.java */
/* loaded from: classes.dex */
public class u extends t {

    /* renamed from: d  reason: collision with root package name */
    public static boolean f2924d = true;

    /* renamed from: e  reason: collision with root package name */
    public static boolean f2925e = true;

    @Override // b.z.y
    @SuppressLint({"NewApi"})
    public void g(View view, Matrix matrix) {
        if (f2924d) {
            try {
                view.transformMatrixToGlobal(matrix);
            } catch (NoSuchMethodError unused) {
                f2924d = false;
            }
        }
    }

    @Override // b.z.y
    @SuppressLint({"NewApi"})
    public void h(View view, Matrix matrix) {
        if (f2925e) {
            try {
                view.transformMatrixToLocal(matrix);
            } catch (NoSuchMethodError unused) {
                f2925e = false;
            }
        }
    }
}