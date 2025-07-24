package c.c.a.m.v.d0;

import android.app.ActivityManager;
import android.content.Context;
import android.os.Build;
import android.text.format.Formatter;
import android.util.DisplayMetrics;
import android.util.Log;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import org.opencv.calib3d.Calib3d;

/* compiled from: MemorySizeCalculator.java */
/* loaded from: classes.dex */
public final class j {

    /* renamed from: a  reason: collision with root package name */
    public final int f3664a;

    /* renamed from: b  reason: collision with root package name */
    public final int f3665b;

    /* renamed from: c  reason: collision with root package name */
    public final Context f3666c;

    /* renamed from: d  reason: collision with root package name */
    public final int f3667d;

    /* compiled from: MemorySizeCalculator.java */
    /* loaded from: classes.dex */
    public static final class a {

        /* renamed from: a  reason: collision with root package name */
        public static final int f3668a;

        /* renamed from: b  reason: collision with root package name */
        public final Context f3669b;

        /* renamed from: c  reason: collision with root package name */
        public ActivityManager f3670c;

        /* renamed from: d  reason: collision with root package name */
        public c f3671d;

        /* renamed from: e  reason: collision with root package name */
        public float f3672e;

        static {
            f3668a = Build.VERSION.SDK_INT < 26 ? 4 : 1;
        }

        public a(Context context) {
            this.f3672e = f3668a;
            this.f3669b = context;
            this.f3670c = (ActivityManager) context.getSystemService("activity");
            this.f3671d = new b(context.getResources().getDisplayMetrics());
            if (Build.VERSION.SDK_INT < 26 || !this.f3670c.isLowRamDevice()) {
                return;
            }
            this.f3672e = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        }
    }

    /* compiled from: MemorySizeCalculator.java */
    /* loaded from: classes.dex */
    public static final class b implements c {

        /* renamed from: a  reason: collision with root package name */
        public final DisplayMetrics f3673a;

        public b(DisplayMetrics displayMetrics) {
            this.f3673a = displayMetrics;
        }
    }

    /* compiled from: MemorySizeCalculator.java */
    /* loaded from: classes.dex */
    public interface c {
    }

    public j(a aVar) {
        ActivityManager activityManager;
        this.f3666c = aVar.f3669b;
        int i = aVar.f3670c.isLowRamDevice() ? Calib3d.CALIB_FIX_TANGENT_DIST : Calib3d.CALIB_USE_EXTRINSIC_GUESS;
        this.f3667d = i;
        int round = Math.round(activityManager.getMemoryClass() * 1024 * 1024 * (aVar.f3670c.isLowRamDevice() ? 0.33f : 0.4f));
        DisplayMetrics displayMetrics = ((b) aVar.f3671d).f3673a;
        float f2 = displayMetrics.widthPixels * displayMetrics.heightPixels * 4;
        int round2 = Math.round(aVar.f3672e * f2);
        int round3 = Math.round(f2 * 2.0f);
        int i2 = round - i;
        int i3 = round3 + round2;
        if (i3 <= i2) {
            this.f3665b = round3;
            this.f3664a = round2;
        } else {
            float f3 = i2 / (aVar.f3672e + 2.0f);
            this.f3665b = Math.round(2.0f * f3);
            this.f3664a = Math.round(f3 * aVar.f3672e);
        }
        if (Log.isLoggable("MemorySizeCalculator", 3)) {
            StringBuilder x = c.b.a.a.a.x("Calculation complete, Calculated memory cache size: ");
            x.append(a(this.f3665b));
            x.append(", pool size: ");
            x.append(a(this.f3664a));
            x.append(", byte array size: ");
            x.append(a(i));
            x.append(", memory class limited? ");
            x.append(i3 > round);
            x.append(", max size: ");
            x.append(a(round));
            x.append(", memoryClass: ");
            x.append(aVar.f3670c.getMemoryClass());
            x.append(", isLowMemoryDevice: ");
            x.append(aVar.f3670c.isLowRamDevice());
            Log.d("MemorySizeCalculator", x.toString());
        }
    }

    public final String a(int i) {
        return Formatter.formatFileSize(this.f3666c, i);
    }
}