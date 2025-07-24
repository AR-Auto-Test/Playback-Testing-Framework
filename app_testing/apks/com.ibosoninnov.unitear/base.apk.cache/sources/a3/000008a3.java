package c.c.a.s;

import android.os.SystemClock;

/* compiled from: LogTime.java */
/* loaded from: classes.dex */
public final class f {

    /* renamed from: a  reason: collision with root package name */
    public static final double f4186a = 1.0d / Math.pow(10.0d, 6.0d);

    /* renamed from: b  reason: collision with root package name */
    public static final /* synthetic */ int f4187b = 0;

    public static double a(long j) {
        return (SystemClock.elapsedRealtimeNanos() - j) * f4186a;
    }
}