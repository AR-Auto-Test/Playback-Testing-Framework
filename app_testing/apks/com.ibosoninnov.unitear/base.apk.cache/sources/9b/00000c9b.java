package com.google.android.filament;

/* loaded from: classes.dex */
public class Camera {
    private long mNativeObject;

    /* loaded from: classes.dex */
    public enum Fov {
        VERTICAL,
        HORIZONTAL
    }

    /* loaded from: classes.dex */
    public enum Projection {
        PERSPECTIVE,
        ORTHO
    }

    public Camera(long j) {
        this.mNativeObject = j;
    }

    private static native float nGetAperture(long j);

    private static native float nGetCullingFar(long j);

    private static native void nGetForwardVector(long j, float[] fArr);

    private static native void nGetLeftVector(long j, float[] fArr);

    private static native void nGetModelMatrix(long j, float[] fArr);

    private static native float nGetNear(long j);

    private static native void nGetPosition(long j, float[] fArr);

    private static native void nGetProjectionMatrix(long j, double[] dArr);

    private static native float nGetSensitivity(long j);

    private static native float nGetShutterSpeed(long j);

    private static native void nGetUpVector(long j, float[] fArr);

    private static native void nGetViewMatrix(long j, float[] fArr);

    private static native void nLookAt(long j, double d2, double d3, double d4, double d5, double d6, double d7, double d8, double d9, double d10);

    private static native void nSetCustomProjection(long j, double[] dArr, double d2, double d3);

    private static native void nSetExposure(long j, float f2, float f3, float f4);

    private static native void nSetLensProjection(long j, double d2, double d3, double d4, double d5);

    private static native void nSetModelMatrix(long j, float[] fArr);

    private static native void nSetProjection(long j, int i, double d2, double d3, double d4, double d5, double d6, double d7);

    private static native void nSetProjectionFov(long j, double d2, double d3, double d4, double d5, int i);

    public void clearNativeObject() {
        this.mNativeObject = 0L;
    }

    public float getAperture() {
        return nGetAperture(getNativeObject());
    }

    public float getCullingFar() {
        return nGetCullingFar(getNativeObject());
    }

    public float[] getForwardVector(float[] fArr) {
        float[] assertFloat3 = Asserts.assertFloat3(fArr);
        nGetForwardVector(getNativeObject(), assertFloat3);
        return assertFloat3;
    }

    public float[] getLeftVector(float[] fArr) {
        float[] assertFloat3 = Asserts.assertFloat3(fArr);
        nGetLeftVector(getNativeObject(), assertFloat3);
        return assertFloat3;
    }

    public float[] getModelMatrix(float[] fArr) {
        float[] assertMat4f = Asserts.assertMat4f(fArr);
        nGetModelMatrix(getNativeObject(), assertMat4f);
        return assertMat4f;
    }

    public long getNativeObject() {
        long j = this.mNativeObject;
        if (j != 0) {
            return j;
        }
        throw new IllegalStateException("Calling method on destroyed Camera");
    }

    public float getNear() {
        return nGetNear(getNativeObject());
    }

    public float[] getPosition(float[] fArr) {
        float[] assertFloat3 = Asserts.assertFloat3(fArr);
        nGetPosition(getNativeObject(), assertFloat3);
        return assertFloat3;
    }

    public double[] getProjectionMatrix(double[] dArr) {
        double[] assertMat4d = Asserts.assertMat4d(dArr);
        nGetProjectionMatrix(getNativeObject(), assertMat4d);
        return assertMat4d;
    }

    public float getSensitivity() {
        return nGetSensitivity(getNativeObject());
    }

    public float getShutterSpeed() {
        return nGetShutterSpeed(getNativeObject());
    }

    public float[] getUpVector(float[] fArr) {
        float[] assertFloat3 = Asserts.assertFloat3(fArr);
        nGetUpVector(getNativeObject(), assertFloat3);
        return assertFloat3;
    }

    public float[] getViewMatrix(float[] fArr) {
        float[] assertMat4f = Asserts.assertMat4f(fArr);
        nGetViewMatrix(getNativeObject(), assertMat4f);
        return assertMat4f;
    }

    public void lookAt(double d2, double d3, double d4, double d5, double d6, double d7, double d8, double d9, double d10) {
        nLookAt(getNativeObject(), d2, d3, d4, d5, d6, d7, d8, d9, d10);
    }

    public void setCustomProjection(double[] dArr, double d2, double d3) {
        Asserts.assertMat4dIn(dArr);
        nSetCustomProjection(getNativeObject(), dArr, d2, d3);
    }

    public void setExposure(float f2, float f3, float f4) {
        nSetExposure(getNativeObject(), f2, f3, f4);
    }

    public void setLensProjection(double d2, double d3, double d4, double d5) {
        nSetLensProjection(getNativeObject(), d2, d3, d4, d5);
    }

    public void setModelMatrix(float[] fArr) {
        Asserts.assertMat4fIn(fArr);
        nSetModelMatrix(getNativeObject(), fArr);
    }

    public void setProjection(Projection projection, double d2, double d3, double d4, double d5, double d6, double d7) {
        nSetProjection(getNativeObject(), projection.ordinal(), d2, d3, d4, d5, d6, d7);
    }

    public void setExposure(float f2) {
        setExposure(1.0f, 1.2f, (1.0f / f2) * 100.0f);
    }

    public void setProjection(double d2, double d3, double d4, double d5, Fov fov) {
        nSetProjectionFov(getNativeObject(), d2, d3, d4, d5, fov.ordinal());
    }
}