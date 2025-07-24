package c.a.a.c0;

import android.util.Log;
import c.a.a.m;
import java.util.Objects;
import java.util.Set;

/* compiled from: Logger.java */
/* loaded from: classes.dex */
public class c {

    /* renamed from: a  reason: collision with root package name */
    public static m f3022a = new b();

    public static void a(String str) {
        Objects.requireNonNull((b) f3022a);
    }

    public static void b(String str) {
        Objects.requireNonNull((b) f3022a);
        Set<String> set = b.f3021a;
        if (set.contains(str)) {
            return;
        }
        Log.w("LOTTIE", str, null);
        set.add(str);
    }

    public static void c(String str, Throwable th) {
        Objects.requireNonNull((b) f3022a);
        Set<String> set = b.f3021a;
        if (set.contains(str)) {
            return;
        }
        Log.w("LOTTIE", str, th);
        set.add(str);
    }
}