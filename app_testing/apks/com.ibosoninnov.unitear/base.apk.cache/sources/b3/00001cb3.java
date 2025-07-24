package com.google.ar.sceneform.math;

import c.b.a.a.a;
import com.google.android.material.internal.StaticLayoutBuilderCompat;

/* loaded from: classes.dex */
public class MathHelper {
    public static final float FLT_EPSILON = 1.1920929E-7f;
    public static final float MAX_DELTA = 1.0E-10f;

    public static boolean almostEqualRelativeAndAbs(float f2, float f3) {
        float abs = Math.abs(f2 - f3);
        return abs <= 1.0E-10f || abs <= Math.max(Math.abs(f2), Math.abs(f3)) * 1.1920929E-7f;
    }

    public static float clamp(float f2, float f3, float f4) {
        return Math.min(f4, Math.max(f3, f2));
    }

    public static float clamp01(float f2) {
        return clamp(f2, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 1.0f);
    }

    public static float lerp(float f2, float f3, float f4) {
        return a.a(f3, f2, f4, f2);
    }
}