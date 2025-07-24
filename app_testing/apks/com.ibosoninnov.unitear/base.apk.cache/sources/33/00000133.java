package b.b;

import android.content.Context;
import android.content.pm.PackageManager;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CaptureRequest;
import android.os.Build;
import android.os.Handler;
import android.os.Looper;
import android.text.TextUtils;
import android.util.Log;
import android.util.LongSparseArray;
import android.view.Surface;
import android.view.View;
import android.view.ViewParent;
import android.view.inputmethod.EditorInfo;
import android.view.inputmethod.InputConnection;
import b.b.h.b1;
import b.b.h.f1;
import b.d.a.e.y1.e;
import b.d.a.e.y1.k;
import b.d.a.e.y1.p.g;
import b.d.a.f.i;
import b.d.b.d1.b0;
import b.d.b.d1.e0;
import b.d.b.d1.f0;
import b.d.b.d1.i0;
import b.d.b.d1.j0;
import b.d.b.d1.k1.b.c;
import b.d.b.d1.z0;
import b.d.b.k0;
import b.d.b.u0;
import b.j.b.d;
import com.google.android.material.snackbar.BaseTransientBottomBar;
import com.google.firebase.crashlytics.internal.common.CrashlyticsReportDataCapture;
import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.Executor;
import java.util.concurrent.ScheduledExecutorService;

/* loaded from: classes.dex */
public final class a {

    /* renamed from: a  reason: collision with root package name */
    public static Field f533a;

    /* renamed from: b  reason: collision with root package name */
    public static boolean f534b;

    /* renamed from: c  reason: collision with root package name */
    public static Class<?> f535c;

    /* renamed from: d  reason: collision with root package name */
    public static boolean f536d;

    /* renamed from: e  reason: collision with root package name */
    public static Field f537e;

    /* renamed from: f  reason: collision with root package name */
    public static boolean f538f;

    /* renamed from: g  reason: collision with root package name */
    public static Field f539g;

    /* renamed from: h  reason: collision with root package name */
    public static boolean f540h;

    public static void a(CaptureRequest.Builder builder, i0 i0Var) {
        i a2 = i.a.b(i0Var).a();
        for (i0.a<?> aVar : a2.e()) {
            CaptureRequest.Key key = (CaptureRequest.Key) aVar.b();
            try {
                builder.set(key, a2.k().a(aVar));
            } catch (IllegalArgumentException unused) {
                u0.b("CaptureRequestBuilder", "CaptureRequest.Key is not supported: " + key, null);
            }
        }
    }

    public static CaptureRequest b(f0 f0Var, CameraDevice cameraDevice, Map<j0, Surface> map) {
        if (cameraDevice == null) {
            return null;
        }
        List<j0> a2 = f0Var.a();
        ArrayList arrayList = new ArrayList();
        for (j0 j0Var : a2) {
            Surface surface = map.get(j0Var);
            if (surface != null) {
                arrayList.add(surface);
            } else {
                throw new IllegalArgumentException("DeferrableSurface not in configuredSurfaceMap");
            }
        }
        if (arrayList.isEmpty()) {
            return null;
        }
        CaptureRequest.Builder createCaptureRequest = cameraDevice.createCaptureRequest(f0Var.f1464e);
        a(createCaptureRequest, f0Var.f1463d);
        i0 i0Var = f0Var.f1463d;
        i0.a<Integer> aVar = f0.f1460a;
        if (i0Var.b(aVar)) {
            createCaptureRequest.set(CaptureRequest.JPEG_ORIENTATION, (Integer) f0Var.f1463d.a(aVar));
        }
        i0 i0Var2 = f0Var.f1463d;
        i0.a<Integer> aVar2 = f0.f1461b;
        if (i0Var2.b(aVar2)) {
            createCaptureRequest.set(CaptureRequest.JPEG_QUALITY, Byte.valueOf(((Integer) f0Var.f1463d.a(aVar2)).byteValue()));
        }
        Iterator it = arrayList.iterator();
        while (it.hasNext()) {
            createCaptureRequest.addTarget((Surface) it.next());
        }
        createCaptureRequest.setTag(f0Var.f1467h);
        return createCaptureRequest.build();
    }

    public static void c() {
        d.k(k(), "Not in application's main thread");
    }

    public static k0 d(b.d.a.e.y1.a aVar) {
        int i = aVar.f1243c;
        int i2 = 5;
        if (i == 1) {
            i2 = 1;
        } else if (i == 2) {
            i2 = 2;
        } else if (i == 3) {
            i2 = 3;
        } else if (i == 4) {
            i2 = 4;
        } else if (i != 5) {
            i2 = i != 10001 ? 0 : 6;
        }
        return new k0(i2, aVar);
    }

    public static String e(k kVar, Integer num, List<String> list) {
        if (num != null && list.contains(CrashlyticsReportDataCapture.SIGNAL_DEFAULT) && list.contains("1")) {
            if (num.intValue() == 1) {
                if (((Integer) kVar.b(CrashlyticsReportDataCapture.SIGNAL_DEFAULT).a(CameraCharacteristics.LENS_FACING)).intValue() == 1) {
                    return "1";
                }
                return null;
            } else if (num.intValue() == 0 && ((Integer) kVar.b("1").a(CameraCharacteristics.LENS_FACING)).intValue() == 0) {
                return CrashlyticsReportDataCapture.SIGNAL_DEFAULT;
            } else {
                return null;
            }
        }
        return null;
    }

    public static Executor f() {
        if (b.d.b.d1.k1.b.a.f1516b != null) {
            return b.d.b.d1.k1.b.a.f1516b;
        }
        synchronized (b.d.b.d1.k1.b.a.class) {
            if (b.d.b.d1.k1.b.a.f1516b == null) {
                b.d.b.d1.k1.b.a.f1516b = new b.d.b.d1.k1.b.a();
            }
        }
        return b.d.b.d1.k1.b.a.f1516b;
    }

    public static void g(Object obj) {
        if (!f536d) {
            try {
                f535c = Class.forName("android.content.res.ThemedResourceCache");
            } catch (ClassNotFoundException e2) {
                Log.e("ResourcesFlusher", "Could not find ThemedResourceCache class", e2);
            }
            f536d = true;
        }
        Class<?> cls = f535c;
        if (cls == null) {
            return;
        }
        if (!f538f) {
            try {
                Field declaredField = cls.getDeclaredField("mUnthemedEntries");
                f537e = declaredField;
                declaredField.setAccessible(true);
            } catch (NoSuchFieldException e3) {
                Log.e("ResourcesFlusher", "Could not retrieve ThemedResourceCache#mUnthemedEntries field", e3);
            }
            f538f = true;
        }
        Field field = f537e;
        if (field == null) {
            return;
        }
        LongSparseArray longSparseArray = null;
        try {
            longSparseArray = (LongSparseArray) field.get(obj);
        } catch (IllegalAccessException e4) {
            Log.e("ResourcesFlusher", "Could not retrieve value from ThemedResourceCache#mUnthemedEntries", e4);
        }
        if (longSparseArray != null) {
            longSparseArray.clear();
        }
    }

    public static z0 h(e eVar) {
        ArrayList arrayList = new ArrayList();
        Integer num = (Integer) eVar.a(CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL);
        boolean z = true;
        if (num != null && num.intValue() == 2) {
            arrayList.add(new b.d.a.e.y1.p.a(eVar));
        }
        Integer num2 = (Integer) eVar.a(CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL);
        if (num2 != null) {
            num2.intValue();
        }
        Set<String> set = g.f1347a;
        Integer num3 = (Integer) eVar.a(CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL);
        Objects.requireNonNull(num3);
        int intValue = num3.intValue();
        if (!g.f1347a.contains(Build.DEVICE.toLowerCase(Locale.US)) || !g.f1348b.contains(Integer.valueOf(intValue))) {
            z = false;
        }
        if (z) {
            arrayList.add(new g());
        }
        return new z0(arrayList);
    }

    public static int i(int i, int i2, boolean z) {
        int i3;
        if (z) {
            i3 = ((i2 - i) + 360) % 360;
        } else {
            i3 = (i2 + i) % 360;
        }
        if (u0.c("CameraOrientationUtil")) {
            u0.a("CameraOrientationUtil", String.format("getRelativeImageRotation: destRotationDegrees=%s, sourceRotationDegrees=%s, isOppositeFacing=%s, result=%s", Integer.valueOf(i), Integer.valueOf(i2), Boolean.valueOf(z), Integer.valueOf(i3)), null);
        }
        return i3;
    }

    public static void j(List<j0> list) {
        if (list.isEmpty()) {
            return;
        }
        int i = 0;
        do {
            try {
                list.get(i).e();
                i++;
            } catch (j0.a e2) {
                for (int i2 = i - 1; i2 >= 0; i2--) {
                    list.get(i2).b();
                }
                throw e2;
            }
        } while (i < list.size());
    }

    public static boolean k() {
        return Looper.getMainLooper().getThread() == Thread.currentThread();
    }

    public static ScheduledExecutorService l() {
        if (c.f1527a != null) {
            return c.f1527a;
        }
        synchronized (c.class) {
            if (c.f1527a == null) {
                c.f1527a = new b.d.b.d1.k1.b.b(new Handler(Looper.getMainLooper()));
            }
        }
        return c.f1527a;
    }

    public static InputConnection m(InputConnection inputConnection, EditorInfo editorInfo, View view) {
        if (inputConnection != null && editorInfo.hintText == null) {
            ViewParent parent = view.getParent();
            while (true) {
                if (!(parent instanceof View)) {
                    break;
                } else if (parent instanceof f1) {
                    editorInfo.hintText = ((f1) parent).a();
                    break;
                } else {
                    parent = parent.getParent();
                }
            }
        }
        return inputConnection;
    }

    public static void n(View view, CharSequence charSequence) {
        if (Build.VERSION.SDK_INT >= 26) {
            view.setTooltipText(charSequence);
            return;
        }
        b1 b1Var = b1.f800b;
        if (b1Var != null && b1Var.f802d == view) {
            b1.c(null);
        }
        if (TextUtils.isEmpty(charSequence)) {
            b1 b1Var2 = b1.f801c;
            if (b1Var2 != null && b1Var2.f802d == view) {
                b1Var2.b();
            }
            view.setOnLongClickListener(null);
            view.setLongClickable(false);
            view.setOnHoverListener(null);
            return;
        }
        new b1(view, charSequence);
    }

    public static int o(int i) {
        if (i != 0) {
            if (i != 1) {
                if (i != 2) {
                    if (i == 3) {
                        return 270;
                    }
                    throw new IllegalArgumentException(c.b.a.a.a.j("Unsupported surface rotation: ", i));
                }
                return BaseTransientBottomBar.ANIMATION_FADE_DURATION;
            }
            return 90;
        }
        return 0;
    }

    public static void p(Context context, b0 b0Var) {
        PackageManager packageManager = context.getPackageManager();
        StringBuilder x = c.b.a.a.a.x("Verifying camera lens facing on ");
        x.append(Build.DEVICE);
        u0.a("CameraValidator", x.toString(), null);
        try {
            if (packageManager.hasSystemFeature("android.hardware.camera")) {
                b.d.b.j0.f1630b.a(b0Var.a()).iterator().next();
            }
            if (packageManager.hasSystemFeature("android.hardware.camera.front")) {
                b.d.b.j0.f1629a.a(b0Var.a()).iterator().next();
            }
        } catch (IllegalArgumentException e2) {
            StringBuilder x2 = c.b.a.a.a.x("Camera LensFacing verification failed, existing cameras: ");
            x2.append(b0Var.a());
            u0.b("CameraValidator", x2.toString(), null);
            throw new e0("Expected camera missing from device.", e2);
        }
    }
}