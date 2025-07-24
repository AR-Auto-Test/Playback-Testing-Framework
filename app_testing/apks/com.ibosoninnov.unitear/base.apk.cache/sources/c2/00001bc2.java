package com.google.ar.core;

import com.google.android.material.internal.StaticLayoutBuilderCompat;
import com.google.ar.core.annotations.UsedByNative;
import java.util.Locale;

@UsedByNative("session_jni_wrapper.cc")
/* loaded from: classes.dex */
public class Pose {
    public static final Pose IDENTITY = new Pose(new float[]{StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD}, Quaternion.f5539a);
    @UsedByNative("session_jni_wrapper.cc")
    private final Quaternion quaternion;
    @UsedByNative("session_jni_wrapper.cc")
    private final float[] translation;

    private Pose(float f2, float f3, float f4, float f5, float f6, float f7, float f8) {
        this.quaternion = new Quaternion(f5, f6, f7, f8);
        this.translation = new float[]{f2, f3, f4};
    }

    @UsedByNative("session_jni_wrapper.cc")
    private Pose(float[] fArr, Quaternion quaternion) {
        this.translation = fArr;
        this.quaternion = quaternion;
    }

    public static Pose makeInterpolated(Pose pose, Pose pose2, float f2) {
        if (f2 == StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
            return pose;
        }
        if (f2 == 1.0f) {
            return pose2;
        }
        float[] fArr = new float[3];
        for (int i = 0; i < 3; i++) {
            fArr[i] = (pose2.translation[i] * f2) + ((1.0f - f2) * pose.translation[i]);
        }
        return new Pose(fArr, Quaternion.g(pose.quaternion, pose2.quaternion, f2));
    }

    public static Pose makeRotation(float f2, float f3, float f4, float f5) {
        return new Pose(IDENTITY.translation, new Quaternion(f2, f3, f4, f5));
    }

    public static Pose makeTranslation(float f2, float f3, float f4) {
        return new Pose(new float[]{f2, f3, f4}, IDENTITY.quaternion);
    }

    public Pose compose(Pose pose) {
        Quaternion.i(this.quaternion, pose.translation, 0, r0, 0);
        float f2 = r0[0];
        float[] fArr = this.translation;
        float[] fArr2 = {f2 + fArr[0], fArr2[1] + fArr[1], fArr2[2] + fArr[2]};
        return new Pose(fArr2, this.quaternion.e(pose.quaternion));
    }

    public Pose extractRotation() {
        return new Pose(IDENTITY.translation, this.quaternion);
    }

    public Pose extractTranslation() {
        return new Pose(this.translation, IDENTITY.quaternion);
    }

    public Quaternion getQuaternion() {
        return this.quaternion;
    }

    public float[] getRotationQuaternion() {
        float[] fArr = new float[4];
        getRotationQuaternion(fArr, 0);
        return fArr;
    }

    public float[] getTransformedAxis(int i, float f2) {
        float[] fArr = new float[3];
        getTransformedAxis(i, f2, fArr, 0);
        return fArr;
    }

    public float[] getTranslation() {
        float[] fArr = new float[3];
        getTranslation(fArr, 0);
        return fArr;
    }

    public float[] getXAxis() {
        return getTransformedAxis(0, 1.0f);
    }

    public float[] getYAxis() {
        return getTransformedAxis(1, 1.0f);
    }

    public float[] getZAxis() {
        return getTransformedAxis(2, 1.0f);
    }

    public Pose inverse() {
        Quaternion f2 = this.quaternion.f();
        Quaternion.i(f2, this.translation, 0, r0, 0);
        float[] fArr = {-fArr[0], -fArr[1], -fArr[2]};
        return new Pose(fArr, f2);
    }

    public float qw() {
        return this.quaternion.a();
    }

    public float qx() {
        return this.quaternion.b();
    }

    public float qy() {
        return this.quaternion.c();
    }

    public float qz() {
        return this.quaternion.d();
    }

    public float[] rotateVector(float[] fArr) {
        float[] fArr2 = new float[3];
        rotateVector(fArr, 0, fArr2, 0);
        return fArr2;
    }

    public void toMatrix(float[] fArr, int i) {
        this.quaternion.k(fArr, i);
        float[] fArr2 = this.translation;
        fArr[i + 12] = fArr2[0];
        fArr[i + 13] = fArr2[1];
        fArr[i + 14] = fArr2[2];
        fArr[i + 3] = 0.0f;
        fArr[i + 7] = 0.0f;
        fArr[i + 11] = 0.0f;
        fArr[i + 15] = 1.0f;
    }

    public String toString() {
        return String.format(Locale.ENGLISH, "t:[x:%.3f, y:%.3f, z:%.3f], q:[x:%.2f, y:%.2f, z:%.2f, w:%.2f]", Float.valueOf(this.translation[0]), Float.valueOf(this.translation[1]), Float.valueOf(this.translation[2]), Float.valueOf(this.quaternion.b()), Float.valueOf(this.quaternion.c()), Float.valueOf(this.quaternion.d()), Float.valueOf(this.quaternion.a()));
    }

    public float[] transformPoint(float[] fArr) {
        float[] fArr2 = new float[3];
        transformPoint(fArr, 0, fArr2, 0);
        return fArr2;
    }

    public float tx() {
        return this.translation[0];
    }

    public float ty() {
        return this.translation[1];
    }

    public float tz() {
        return this.translation[2];
    }

    public Pose(float[] fArr, float[] fArr2) {
        this(fArr[0], fArr[1], fArr[2], fArr2[0], fArr2[1], fArr2[2], fArr2[3]);
    }

    public static Pose makeRotation(float[] fArr) {
        return makeRotation(fArr[0], fArr[1], fArr[2], fArr[3]);
    }

    public static Pose makeTranslation(float[] fArr) {
        return makeTranslation(fArr[0], fArr[1], fArr[2]);
    }

    public void getRotationQuaternion(float[] fArr, int i) {
        this.quaternion.h(fArr, i);
    }

    public void getTransformedAxis(int i, float f2, float[] fArr, int i2) {
        Quaternion quaternion = this.quaternion;
        float[] fArr2 = {StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD};
        fArr2[i] = f2;
        Quaternion.i(quaternion, fArr2, 0, fArr, i2);
    }

    public void getTranslation(float[] fArr, int i) {
        System.arraycopy(this.translation, 0, fArr, i, 3);
    }

    public void rotateVector(float[] fArr, int i, float[] fArr2, int i2) {
        Quaternion.i(this.quaternion, fArr, i, fArr2, i2);
    }

    public void transformPoint(float[] fArr, int i, float[] fArr2, int i2) {
        rotateVector(fArr, i, fArr2, i2);
        for (int i3 = 0; i3 < 3; i3++) {
            int i4 = i3 + i2;
            fArr2[i4] = fArr2[i4] + this.translation[i3];
        }
    }
}