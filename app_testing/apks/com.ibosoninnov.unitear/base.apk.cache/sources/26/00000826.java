package c.c.a.m.x.c;

import android.os.Build;
import android.util.Log;
import java.io.File;
import java.util.Arrays;
import java.util.concurrent.atomic.AtomicBoolean;

/* compiled from: HardwareConfigState.java */
/* loaded from: classes.dex */
public final class r {

    /* renamed from: a  reason: collision with root package name */
    public static final boolean f3988a;

    /* renamed from: b  reason: collision with root package name */
    public static final boolean f3989b;

    /* renamed from: c  reason: collision with root package name */
    public static final File f3990c;

    /* renamed from: d  reason: collision with root package name */
    public static volatile r f3991d;

    /* renamed from: e  reason: collision with root package name */
    public static volatile int f3992e;

    /* renamed from: f  reason: collision with root package name */
    public final boolean f3993f;

    /* renamed from: g  reason: collision with root package name */
    public final int f3994g;

    /* renamed from: h  reason: collision with root package name */
    public final int f3995h;
    public int i;
    public boolean j = true;
    public final AtomicBoolean k = new AtomicBoolean(false);

    static {
        int i = Build.VERSION.SDK_INT;
        f3988a = i < 29;
        f3989b = i >= 26;
        f3990c = new File("/proc/self/fd");
        f3992e = -1;
    }

    /* JADX WARN: Code restructure failed: missing block: B:17:0x009e, code lost:
        if ((android.os.Build.VERSION.SDK_INT != 27 ? false : java.util.Arrays.asList("LG-M250", "LG-M320", "LG-Q710AL", "LG-Q710PL", "LGM-K121K", "LGM-K121L", "LGM-K121S", "LGM-X320K", "LGM-X320L", "LGM-X320S", "LGM-X401L", "LGM-X401S", "LM-Q610.FG", "LM-Q610.FGN", "LM-Q617.FG", "LM-Q617.FGN", "LM-Q710.FG", "LM-Q710.FGN", "LM-X220PM", "LM-X220QMA", "LM-X410PM").contains(android.os.Build.MODEL)) == false) goto L15;
     */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public r() {
        boolean z;
        boolean z2 = true;
        if (Build.VERSION.SDK_INT == 26) {
            for (String str : Arrays.asList("SC-04J", "SM-N935", "SM-J720", "SM-G570F", "SM-G570M", "SM-G960", "SM-G965", "SM-G935", "SM-G930", "SM-A520", "SM-A720F", "moto e5", "moto e5 play", "moto e5 plus", "moto e5 cruise", "moto g(6) forge", "moto g(6) play")) {
                if (Build.MODEL.startsWith(str)) {
                    z = true;
                    break;
                }
            }
        }
        z = false;
        if (!z) {
        }
        z2 = false;
        this.f3993f = z2;
        if (Build.VERSION.SDK_INT >= 28) {
            this.f3994g = 20000;
            this.f3995h = 0;
            return;
        }
        this.f3994g = 700;
        this.f3995h = 128;
    }

    public static r a() {
        if (f3991d == null) {
            synchronized (r.class) {
                if (f3991d == null) {
                    f3991d = new r();
                }
            }
        }
        return f3991d;
    }

    public boolean b(int i, int i2, boolean z, boolean z2) {
        boolean z3;
        int i3;
        if (!z) {
            if (Log.isLoggable("HardwareConfig", 2)) {
                Log.v("HardwareConfig", "Hardware config disallowed by caller");
            }
            return false;
        } else if (!this.f3993f) {
            if (Log.isLoggable("HardwareConfig", 2)) {
                Log.v("HardwareConfig", "Hardware config disallowed by device model");
            }
            return false;
        } else if (!f3989b) {
            if (Log.isLoggable("HardwareConfig", 2)) {
                Log.v("HardwareConfig", "Hardware config disallowed by sdk");
            }
            return false;
        } else {
            if (f3988a && !this.k.get()) {
                if (Log.isLoggable("HardwareConfig", 2)) {
                    Log.v("HardwareConfig", "Hardware config disallowed by app state");
                }
                return false;
            } else if (z2) {
                if (Log.isLoggable("HardwareConfig", 2)) {
                    Log.v("HardwareConfig", "Hardware config disallowed because exif orientation is required");
                }
                return false;
            } else {
                int i4 = this.f3995h;
                if (i < i4) {
                    if (Log.isLoggable("HardwareConfig", 2)) {
                        Log.v("HardwareConfig", "Hardware config disallowed because width is too small");
                    }
                    return false;
                } else if (i2 < i4) {
                    if (Log.isLoggable("HardwareConfig", 2)) {
                        Log.v("HardwareConfig", "Hardware config disallowed because height is too small");
                    }
                    return false;
                } else {
                    synchronized (this) {
                        int i5 = this.i + 1;
                        this.i = i5;
                        if (i5 >= 50) {
                            this.i = 0;
                            int length = f3990c.list().length;
                            if (f3992e != -1) {
                                i3 = f3992e;
                            } else {
                                i3 = this.f3994g;
                            }
                            long j = i3;
                            boolean z4 = ((long) length) < j;
                            this.j = z4;
                            if (!z4 && Log.isLoggable("Downsampler", 5)) {
                                Log.w("Downsampler", "Excluding HARDWARE bitmap config because we're over the file descriptor limit, file descriptors " + length + ", limit " + j);
                            }
                        }
                        z3 = this.j;
                    }
                    if (z3) {
                        return true;
                    }
                    if (Log.isLoggable("HardwareConfig", 2)) {
                        Log.v("HardwareConfig", "Hardware config disallowed because there are insufficient FDs");
                    }
                    return false;
                }
            }
        }
    }
}