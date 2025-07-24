package androidx.activity;

import android.app.Activity;
import android.view.View;
import android.view.inputmethod.InputMethodManager;
import b.t.e;
import b.t.f;
import b.t.h;
import java.lang.reflect.Field;

/* loaded from: classes.dex */
public final class ImmLeaksCleaner implements f {

    /* renamed from: a  reason: collision with root package name */
    public static int f46a;

    /* renamed from: b  reason: collision with root package name */
    public static Field f47b;

    /* renamed from: c  reason: collision with root package name */
    public static Field f48c;

    /* renamed from: d  reason: collision with root package name */
    public static Field f49d;

    /* renamed from: e  reason: collision with root package name */
    public Activity f50e;

    @Override // b.t.f
    public void e(h hVar, e.a aVar) {
        if (aVar != e.a.ON_DESTROY) {
            return;
        }
        if (f46a == 0) {
            try {
                f46a = 2;
                Field declaredField = InputMethodManager.class.getDeclaredField("mServedView");
                f48c = declaredField;
                declaredField.setAccessible(true);
                Field declaredField2 = InputMethodManager.class.getDeclaredField("mNextServedView");
                f49d = declaredField2;
                declaredField2.setAccessible(true);
                Field declaredField3 = InputMethodManager.class.getDeclaredField("mH");
                f47b = declaredField3;
                declaredField3.setAccessible(true);
                f46a = 1;
            } catch (NoSuchFieldException unused) {
            }
        }
        if (f46a == 1) {
            InputMethodManager inputMethodManager = (InputMethodManager) this.f50e.getSystemService("input_method");
            try {
                Object obj = f47b.get(inputMethodManager);
                if (obj == null) {
                    return;
                }
                synchronized (obj) {
                    try {
                        try {
                            View view = (View) f48c.get(inputMethodManager);
                            if (view == null) {
                                return;
                            }
                            if (view.isAttachedToWindow()) {
                                return;
                            }
                            try {
                                f49d.set(inputMethodManager, null);
                                inputMethodManager.isActive();
                            } catch (IllegalAccessException unused2) {
                            }
                        } catch (ClassCastException unused3) {
                        } catch (IllegalAccessException unused4) {
                        }
                    } catch (Throwable th) {
                        throw th;
                    }
                }
            } catch (IllegalAccessException unused5) {
            }
        }
    }
}