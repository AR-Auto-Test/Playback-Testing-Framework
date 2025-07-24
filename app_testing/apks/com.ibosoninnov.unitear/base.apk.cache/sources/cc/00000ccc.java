package com.google.android.filament;

/* loaded from: classes.dex */
public final class MathUtils {
    private MathUtils() {
    }

    private static native void nPackTangentFrame(float f2, float f3, float f4, float f5, float f6, float f7, float f8, float f9, float f10, float[] fArr, int i);

    public static void packTangentFrame(float f2, float f3, float f4, float f5, float f6, float f7, float f8, float f9, float f10, float[] fArr) {
        nPackTangentFrame(f2, f3, f4, f5, f6, f7, f8, f9, f10, fArr, 0);
    }

    public static void packTangentFrame(float f2, float f3, float f4, float f5, float f6, float f7, float f8, float f9, float f10, float[] fArr, int i) {
        nPackTangentFrame(f2, f3, f4, f5, f6, f7, f8, f9, f10, fArr, i);
    }
}