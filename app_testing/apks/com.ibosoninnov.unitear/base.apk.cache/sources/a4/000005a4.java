package b.v.u;

import android.animation.ObjectAnimator;
import android.animation.TypeConverter;
import android.annotation.SuppressLint;
import android.content.Context;
import android.graphics.Path;
import android.graphics.PointF;
import android.net.Uri;
import android.os.Build;
import android.os.VibrationEffect;
import android.os.Vibrator;
import android.util.Property;
import android.view.View;
import androidx.recyclerview.widget.RecyclerView;
import b.w.b.s;
import c.a.a.b0.e;
import c.a.a.b0.g0;
import c.a.a.b0.h;
import c.a.a.b0.n;
import c.a.a.b0.q;
import c.a.a.b0.v;
import c.a.a.c0.g;
import c.c.a.m.f;
import c.c.a.m.i;
import c.c.a.m.k;
import c.c.a.m.l;
import c.c.a.m.x.c.w;
import com.bumptech.glide.load.ImageHeaderParser;
import com.google.firebase.analytics.FirebaseAnalytics;
import f.r;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

/* loaded from: classes.dex */
public final class c {
    @SuppressLint({"StaticFieldLeak"})

    /* renamed from: a  reason: collision with root package name */
    public static Context f2697a;

    public static float a(float f2) {
        return f2 <= 0.04045f ? f2 / 12.92f : (float) Math.pow((f2 + 0.055f) / 1.055f, 2.4000000953674316d);
    }

    public static float b(float f2) {
        return f2 <= 0.0031308f ? f2 * 12.92f : (float) ((Math.pow(f2, 0.4166666567325592d) * 1.0549999475479126d) - 0.054999999701976776d);
    }

    public static <T> ArrayList<T> c(ArrayList<T> arrayList, T t) {
        if (arrayList == null) {
            arrayList = new ArrayList<>();
        }
        if (!arrayList.contains(t)) {
            arrayList.add(t);
        }
        return arrayList;
    }

    public static void d(boolean z, String str) {
        if (!z) {
            throw new IllegalArgumentException(str);
        }
    }

    public static int e(RecyclerView.a0 a0Var, s sVar, View view, View view2, RecyclerView.o oVar, boolean z) {
        if (oVar.getChildCount() == 0 || a0Var.b() == 0 || view == null || view2 == null) {
            return 0;
        }
        if (!z) {
            return Math.abs(oVar.getPosition(view) - oVar.getPosition(view2)) + 1;
        }
        return Math.min(sVar.l(), sVar.b(view2) - sVar.e(view));
    }

    public static int f(RecyclerView.a0 a0Var, s sVar, View view, View view2, RecyclerView.o oVar, boolean z, boolean z2) {
        int max;
        if (oVar.getChildCount() == 0 || a0Var.b() == 0 || view == null || view2 == null) {
            return 0;
        }
        int min = Math.min(oVar.getPosition(view), oVar.getPosition(view2));
        int max2 = Math.max(oVar.getPosition(view), oVar.getPosition(view2));
        if (z2) {
            max = Math.max(0, (a0Var.b() - max2) - 1);
        } else {
            max = Math.max(0, min);
        }
        if (z) {
            return Math.round((max * (Math.abs(sVar.b(view2) - sVar.e(view)) / (Math.abs(oVar.getPosition(view) - oVar.getPosition(view2)) + 1))) + (sVar.k() - sVar.e(view)));
        }
        return max;
    }

    public static int g(RecyclerView.a0 a0Var, s sVar, View view, View view2, RecyclerView.o oVar, boolean z) {
        if (oVar.getChildCount() == 0 || a0Var.b() == 0 || view == null || view2 == null) {
            return 0;
        }
        if (!z) {
            return a0Var.b();
        }
        return (int) (((sVar.b(view2) - sVar.e(view)) / (Math.abs(oVar.getPosition(view) - oVar.getPosition(view2)) + 1)) * a0Var.b());
    }

    public static int h(float f2, int i, int i2) {
        if (i == i2) {
            return i;
        }
        float f3 = ((i >> 24) & 255) / 255.0f;
        float a2 = a(((i >> 16) & 255) / 255.0f);
        float a3 = a(((i >> 8) & 255) / 255.0f);
        float a4 = a((i & 255) / 255.0f);
        float a5 = a(((i2 >> 16) & 255) / 255.0f);
        float a6 = a(((i2 >> 8) & 255) / 255.0f);
        float a7 = a((i2 & 255) / 255.0f);
        float a8 = c.b.a.a.a.a(((i2 >> 24) & 255) / 255.0f, f3, f2, f3);
        float a9 = c.b.a.a.a.a(a5, a2, f2, a2);
        float a10 = c.b.a.a.a.a(a6, a3, f2, a3);
        float a11 = c.b.a.a.a.a(a7, a4, f2, a4);
        int round = Math.round(b(a9) * 255.0f) << 16;
        return Math.round(b(a11) * 255.0f) | round | (Math.round(a8 * 255.0f) << 24) | (Math.round(b(a10) * 255.0f) << 8);
    }

    public static int i(List<ImageHeaderParser> list, InputStream inputStream, c.c.a.m.v.c0.b bVar) {
        if (inputStream == null) {
            return -1;
        }
        if (!inputStream.markSupported()) {
            inputStream = new w(inputStream, bVar);
        }
        inputStream.mark(5242880);
        return j(list, new i(inputStream, bVar));
    }

    public static int j(List<ImageHeaderParser> list, k kVar) {
        int size = list.size();
        for (int i = 0; i < size; i++) {
            int a2 = kVar.a(list.get(i));
            if (a2 != -1) {
                return a2;
            }
        }
        return -1;
    }

    public static ImageHeaderParser.ImageType k(List<ImageHeaderParser> list, InputStream inputStream, c.c.a.m.v.c0.b bVar) {
        if (inputStream == null) {
            return ImageHeaderParser.ImageType.UNKNOWN;
        }
        if (!inputStream.markSupported()) {
            inputStream = new w(inputStream, bVar);
        }
        inputStream.mark(5242880);
        return l(list, new f(inputStream));
    }

    public static ImageHeaderParser.ImageType l(List<ImageHeaderParser> list, l lVar) {
        int size = list.size();
        for (int i = 0; i < size; i++) {
            ImageHeaderParser.ImageType a2 = lVar.a(list.get(i));
            if (a2 != ImageHeaderParser.ImageType.UNKNOWN) {
                return a2;
            }
        }
        return ImageHeaderParser.ImageType.UNKNOWN;
    }

    public static boolean m(CharSequence charSequence) {
        return charSequence == null || charSequence.length() == 0;
    }

    public static boolean n(Uri uri) {
        return uri != null && FirebaseAnalytics.Param.CONTENT.equals(uri.getScheme()) && "media".equals(uri.getAuthority());
    }

    public static boolean o(int i, int i2) {
        return i != Integer.MIN_VALUE && i2 != Integer.MIN_VALUE && i <= 512 && i2 <= 384;
    }

    public static <T> ObjectAnimator p(T t, Property<T, PointF> property, Path path) {
        return ObjectAnimator.ofObject(t, property, (TypeConverter) null, path);
    }

    public static <T> List<c.a.a.d0.a<T>> q(c.a.a.b0.h0.c cVar, c.a.a.d dVar, g0<T> g0Var) {
        return q.a(cVar, dVar, 1.0f, g0Var);
    }

    public static c.a.a.z.j.a r(c.a.a.b0.h0.c cVar, c.a.a.d dVar) {
        return new c.a.a.z.j.a(q(cVar, dVar, e.f2965a));
    }

    public static c.a.a.z.j.b s(c.a.a.b0.h0.c cVar, c.a.a.d dVar) {
        return t(cVar, dVar, true);
    }

    public static c.a.a.z.j.b t(c.a.a.b0.h0.c cVar, c.a.a.d dVar, boolean z) {
        return new c.a.a.z.j.b(q.a(cVar, dVar, z ? g.c() : 1.0f, h.f2972a));
    }

    public static c.a.a.z.j.d u(c.a.a.b0.h0.c cVar, c.a.a.d dVar) {
        return new c.a.a.z.j.d(q(cVar, dVar, n.f2998a));
    }

    public static c.a.a.z.j.f v(c.a.a.b0.h0.c cVar, c.a.a.d dVar) {
        return new c.a.a.z.j.f(q.a(cVar, dVar, g.c(), v.f3013a));
    }

    public static boolean w(String str) {
        return (str.equals("GET") || str.equals("HEAD")) ? false : true;
    }

    public static <T> ArrayList<T> x(ArrayList<T> arrayList, T t) {
        if (arrayList != null) {
            arrayList.remove(t);
            if (arrayList.isEmpty()) {
                return null;
            }
            return arrayList;
        }
        return arrayList;
    }

    public static String y(r rVar) {
        String e2 = rVar.e();
        String g2 = rVar.g();
        if (g2 != null) {
            return e2 + '?' + g2;
        }
        return e2;
    }

    public static void z(Context context) {
        Vibrator vibrator = (Vibrator) context.getSystemService("vibrator");
        if (Build.VERSION.SDK_INT >= 26) {
            vibrator.vibrate(VibrationEffect.createOneShot(10L, -1));
        } else {
            vibrator.vibrate(10L);
        }
    }
}