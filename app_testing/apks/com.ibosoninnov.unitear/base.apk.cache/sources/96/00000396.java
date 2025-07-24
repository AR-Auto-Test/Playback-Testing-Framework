package b.d.b;

import android.util.Log;

/* compiled from: Logger.java */
/* loaded from: classes.dex */
public final class u0 {

    /* renamed from: a  reason: collision with root package name */
    public static int f1672a = 3;

    public static void a(String str, String str2, Throwable th) {
        if (c(str)) {
            str.length();
            Log.d(str, str2, th);
        }
    }

    /* JADX WARN: Removed duplicated region for block: B:10:0x0014  */
    /* JADX WARN: Removed duplicated region for block: B:12:? A[RETURN, SYNTHETIC] */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public static void b(String str, String str2, Throwable th) {
        boolean z;
        if (f1672a > 6) {
            str.length();
            if (!Log.isLoggable(str, 6)) {
                z = false;
                if (z) {
                    return;
                }
                str.length();
                Log.e(str, str2, th);
                return;
            }
        }
        z = true;
        if (z) {
        }
    }

    public static boolean c(String str) {
        if (f1672a > 3) {
            str.length();
            if (!Log.isLoggable(str, 3)) {
                return false;
            }
        }
        return true;
    }

    /* JADX WARN: Removed duplicated region for block: B:10:0x0014  */
    /* JADX WARN: Removed duplicated region for block: B:12:? A[RETURN, SYNTHETIC] */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public static void d(String str, String str2, Throwable th) {
        boolean z;
        if (f1672a > 5) {
            str.length();
            if (!Log.isLoggable(str, 5)) {
                z = false;
                if (z) {
                    return;
                }
                str.length();
                Log.w(str, str2, th);
                return;
            }
        }
        z = true;
        if (z) {
        }
    }
}