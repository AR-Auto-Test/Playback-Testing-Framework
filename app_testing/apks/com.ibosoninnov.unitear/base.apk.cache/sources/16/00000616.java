package b.z;

import android.graphics.Matrix;
import android.view.View;

/* compiled from: ViewUtilsApi29.java */
/* loaded from: classes.dex */
public class x extends w {
    @Override // b.z.t, b.z.y
    public float b(View view) {
        return view.getTransitionAlpha();
    }

    @Override // b.z.v, b.z.y
    public void d(View view, int i, int i2, int i3, int i4) {
        view.setLeftTopRightBottom(i, i2, i3, i4);
    }

    @Override // b.z.t, b.z.y
    public void e(View view, float f2) {
        view.setTransitionAlpha(f2);
    }

    @Override // b.z.w, b.z.y
    public void f(View view, int i) {
        view.setTransitionVisibility(i);
    }

    @Override // b.z.u, b.z.y
    public void g(View view, Matrix matrix) {
        view.transformMatrixToGlobal(matrix);
    }

    @Override // b.z.u, b.z.y
    public void h(View view, Matrix matrix) {
        view.transformMatrixToLocal(matrix);
    }
}