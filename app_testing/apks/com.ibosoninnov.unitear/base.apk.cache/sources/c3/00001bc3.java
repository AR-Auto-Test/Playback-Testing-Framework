package com.google.ar.core;

import com.google.android.material.internal.StaticLayoutBuilderCompat;
import com.google.ar.core.annotations.UsedByNative;

@UsedByNative("session_jni_wrapper.cc")
/* loaded from: classes.dex */
public class Quaternion {

    /* renamed from: a  reason: collision with root package name */
    public static final Quaternion f5539a = new Quaternion();
    @UsedByNative("session_jni_wrapper.cc")
    private float x = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
    @UsedByNative("session_jni_wrapper.cc")
    private float y = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
    @UsedByNative("session_jni_wrapper.cc")
    private float z = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
    @UsedByNative("session_jni_wrapper.cc")
    private float w = 1.0f;

    public Quaternion() {
        j(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 1.0f);
    }

    public static Quaternion g(Quaternion quaternion, Quaternion quaternion2, float f2) {
        float f3;
        float f4;
        float f5;
        Quaternion quaternion3 = new Quaternion();
        float f6 = (quaternion.w * quaternion2.w) + (quaternion.z * quaternion2.z) + (quaternion.y * quaternion2.y) + (quaternion.x * quaternion2.x);
        if (f6 < StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
            Quaternion quaternion4 = new Quaternion(quaternion2);
            f6 = -f6;
            quaternion4.x = -quaternion4.x;
            quaternion4.y = -quaternion4.y;
            quaternion4.z = -quaternion4.z;
            quaternion4.w = -quaternion4.w;
            quaternion2 = quaternion4;
        }
        float acos = (float) Math.acos(f6);
        float sqrt = (float) Math.sqrt(1.0f - (f6 * f6));
        if (Math.abs(sqrt) > 0.001d) {
            float f7 = 1.0f / sqrt;
            f3 = ((float) Math.sin((1.0f - f2) * acos)) * f7;
            f2 = ((float) Math.sin(f2 * acos)) * f7;
        } else {
            f3 = 1.0f - f2;
        }
        quaternion3.x = (quaternion2.x * f2) + (quaternion.x * f3);
        quaternion3.y = (quaternion2.y * f2) + (quaternion.y * f3);
        float f8 = (quaternion2.z * f2) + (quaternion.z * f3);
        quaternion3.z = f8;
        float f9 = (f2 * quaternion2.w) + (f3 * quaternion.w);
        quaternion3.w = f9;
        float f10 = f8 * f8;
        float f11 = f9 * f9;
        float sqrt2 = (float) (1.0d / Math.sqrt(f11 + (f10 + ((f5 * f5) + (f4 * f4)))));
        quaternion3.x *= sqrt2;
        quaternion3.y *= sqrt2;
        quaternion3.z *= sqrt2;
        quaternion3.w *= sqrt2;
        return quaternion3;
    }

    public static void i(Quaternion quaternion, float[] fArr, int i, float[] fArr2, int i2) {
        float f2 = fArr[i];
        float f3 = fArr[i + 1];
        float f4 = fArr[i + 2];
        float f5 = quaternion.x;
        float f6 = quaternion.y;
        float f7 = quaternion.z;
        float f8 = quaternion.w;
        float f9 = ((f6 * f4) + (f8 * f2)) - (f7 * f3);
        float f10 = ((f7 * f2) + (f8 * f3)) - (f5 * f4);
        float f11 = ((f5 * f3) + (f8 * f4)) - (f6 * f2);
        float f12 = -f5;
        float f13 = ((f2 * f12) - (f3 * f6)) - (f4 * f7);
        float f14 = -f7;
        float f15 = -f6;
        fArr2[i2] = ((f10 * f14) + ((f13 * f12) + (f9 * f8))) - (f11 * f15);
        fArr2[i2 + 1] = ((f11 * f12) + ((f13 * f15) + (f10 * f8))) - (f9 * f14);
        float f16 = f9 * f15;
        fArr2[i2 + 2] = (f16 + ((f13 * f14) + (f11 * f8))) - (f10 * f12);
    }

    public final float a() {
        return this.w;
    }

    public final float b() {
        return this.x;
    }

    public final float c() {
        return this.y;
    }

    public final float d() {
        return this.z;
    }

    public final Quaternion e(Quaternion quaternion) {
        Quaternion quaternion2 = new Quaternion();
        float f2 = this.x;
        float f3 = quaternion.w;
        float f4 = this.y;
        float f5 = quaternion.z;
        float f6 = this.z;
        float f7 = quaternion.y;
        float f8 = this.w;
        quaternion2.x = (quaternion.x * f8) + (((f4 * f5) + (f2 * f3)) - (f6 * f7));
        float f9 = this.x;
        float f10 = -f9;
        float f11 = quaternion.x;
        float f12 = f7 * f8;
        quaternion2.y = f12 + (f6 * f11) + (f4 * f3) + (f10 * f5);
        float f13 = quaternion.y;
        float f14 = this.y;
        float f15 = f5 * f8;
        quaternion2.z = f15 + (f6 * f3) + ((f9 * f13) - (f14 * f11));
        quaternion2.w = (f8 * f3) + (((f10 * f11) - (f14 * f13)) - (this.z * quaternion.z));
        return quaternion2;
    }

    public final Quaternion f() {
        return new Quaternion(-this.x, -this.y, -this.z, this.w);
    }

    public final void h(float[] fArr, int i) {
        fArr[i] = this.x;
        fArr[i + 1] = this.y;
        fArr[i + 2] = this.z;
        fArr[i + 3] = this.w;
    }

    public final void j(float f2, float f3, float f4, float f5) {
        this.x = f2;
        this.y = f3;
        this.z = f4;
        this.w = f5;
    }

    public final void k(float[] fArr, int i) {
        float f2 = this.x;
        float f3 = this.y;
        float f4 = this.z;
        float f5 = this.w;
        float f6 = (f5 * f5) + (f4 * f4) + (f3 * f3) + (f2 * f2);
        float f7 = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        if (f6 > StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
            f7 = 2.0f / f6;
        }
        float f8 = f2 * f7;
        float f9 = f3 * f7;
        float f10 = f7 * f4;
        float f11 = f5 * f8;
        float f12 = f5 * f9;
        float f13 = f5 * f10;
        float f14 = f8 * f2;
        float f15 = f2 * f9;
        float f16 = f2 * f10;
        float f17 = f9 * f3;
        float f18 = f3 * f10;
        float f19 = f4 * f10;
        fArr[i] = 1.0f - (f17 + f19);
        fArr[i + 4] = f15 - f13;
        fArr[i + 8] = f16 + f12;
        fArr[i + 1] = f15 + f13;
        fArr[i + 5] = 1.0f - (f19 + f14);
        fArr[i + 9] = f18 - f11;
        fArr[i + 2] = f16 - f12;
        fArr[i + 6] = f18 + f11;
        fArr[i + 10] = 1.0f - (f14 + f17);
    }

    public final String toString() {
        return String.format("[%.3f, %.3f, %.3f, %.3f]", Float.valueOf(this.x), Float.valueOf(this.y), Float.valueOf(this.z), Float.valueOf(this.w));
    }

    @UsedByNative("session_jni_wrapper.cc")
    public Quaternion(float f2, float f3, float f4, float f5) {
        j(f2, f3, f4, f5);
    }

    public Quaternion(Quaternion quaternion) {
        j(quaternion.x, quaternion.y, quaternion.z, quaternion.w);
    }
}