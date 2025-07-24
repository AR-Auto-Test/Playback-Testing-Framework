package b.z;

import android.graphics.Matrix;
import android.util.Log;
import android.view.View;
import java.lang.reflect.Field;

/* compiled from: ViewUtilsBase.java */
/* loaded from: classes.dex */
public class y {

    /* renamed from: a  reason: collision with root package name */
    public static Field f2928a;

    /* renamed from: b  reason: collision with root package name */
    public static boolean f2929b;

    public void a(View view) {
        throw null;
    }

    public float b(View view) {
        throw null;
    }

    public void c(View view) {
        throw null;
    }

    public void d(View view, int i, int i2, int i3, int i4) {
        throw null;
    }

    public void e(View view, float f2) {
        throw null;
    }

    public void f(View view, int i) {
        if (!f2929b) {
            try {
                Field declaredField = View.class.getDeclaredField("mViewFlags");
                f2928a = declaredField;
                declaredField.setAccessible(true);
            } catch (NoSuchFieldException unused) {
                Log.i("ViewUtilsBase", "fetchViewFlagsField: ");
            }
            f2929b = true;
        }
        Field field = f2928a;
        if (field != null) {
            try {
                f2928a.setInt(view, i | (field.getInt(view) & (-13)));
            } catch (IllegalAccessException unused2) {
            }
        }
    }

    public void g(View view, Matrix matrix) {
        throw null;
    }

    public void h(View view, Matrix matrix) {
        throw null;
    }
}