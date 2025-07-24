package com.google.android.material.animation;

import android.animation.TimeInterpolator;
import android.view.animation.DecelerateInterpolator;
import android.view.animation.LinearInterpolator;
import b.r.a.a.a;
import b.r.a.a.b;
import b.r.a.a.c;

/* loaded from: classes.dex */
public class AnimationUtils {
    public static final TimeInterpolator LINEAR_INTERPOLATOR = new LinearInterpolator();
    public static final TimeInterpolator FAST_OUT_SLOW_IN_INTERPOLATOR = new b();
    public static final TimeInterpolator FAST_OUT_LINEAR_IN_INTERPOLATOR = new a();
    public static final TimeInterpolator LINEAR_OUT_SLOW_IN_INTERPOLATOR = new c();
    public static final TimeInterpolator DECELERATE_INTERPOLATOR = new DecelerateInterpolator();

    public static float lerp(float f2, float f3, float f4) {
        return c.b.a.a.a.a(f3, f2, f4, f2);
    }

    public static int lerp(int i, int i2, float f2) {
        return Math.round(f2 * (i2 - i)) + i;
    }

    public static float lerp(float f2, float f3, float f4, float f5, float f6) {
        return f6 < f4 ? f2 : f6 > f5 ? f3 : lerp(f2, f3, (f6 - f4) / (f5 - f4));
    }
}