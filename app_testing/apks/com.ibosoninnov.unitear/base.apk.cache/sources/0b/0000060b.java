package b.z;

import android.os.Build;

/* compiled from: TransitionUtils.java */
/* loaded from: classes.dex */
public class o {

    /* renamed from: a  reason: collision with root package name */
    public static final boolean f2910a;

    /* renamed from: b  reason: collision with root package name */
    public static final boolean f2911b;

    /* renamed from: c  reason: collision with root package name */
    public static final boolean f2912c;

    static {
        int i = Build.VERSION.SDK_INT;
        f2910a = true;
        f2911b = true;
        f2912c = i >= 28;
    }
}