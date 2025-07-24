package b.b.h;

import android.graphics.Rect;
import android.util.Log;
import android.view.View;
import java.lang.reflect.Method;
import java.util.concurrent.atomic.AtomicInteger;

/* compiled from: ViewUtils.java */
/* loaded from: classes.dex */
public class e1 {

    /* renamed from: a  reason: collision with root package name */
    public static Method f833a;

    static {
        try {
            Method declaredMethod = View.class.getDeclaredMethod("computeFitSystemWindows", Rect.class, Rect.class);
            f833a = declaredMethod;
            if (declaredMethod.isAccessible()) {
                return;
            }
            f833a.setAccessible(true);
        } catch (NoSuchMethodException unused) {
            Log.d("ViewUtils", "Could not find method computeFitSystemWindows. Oh well.");
        }
    }

    public static void a(View view, Rect rect, Rect rect2) {
        Method method = f833a;
        if (method != null) {
            try {
                method.invoke(view, rect, rect2);
            } catch (Exception e2) {
                Log.d("ViewUtils", "Could not invoke computeFitSystemWindows", e2);
            }
        }
    }

    public static boolean b(View view) {
        AtomicInteger atomicInteger = b.j.j.q.f2214a;
        return view.getLayoutDirection() == 1;
    }
}