package com.google.android.filament;

/* loaded from: classes.dex */
public class TransformManager {
    private long mNativeObject;

    public TransformManager(long j) {
        this.mNativeObject = j;
    }

    private static native void nCommitLocalTransformTransaction(long j);

    private static native int nCreate(long j, int i);

    private static native int nCreateArray(long j, int i, int i2, float[] fArr);

    private static native void nDestroy(long j, int i);

    private static native int nGetInstance(long j, int i);

    private static native void nGetTransform(long j, int i, float[] fArr);

    private static native void nGetWorldTransform(long j, int i, float[] fArr);

    private static native boolean nHasComponent(long j, int i);

    private static native void nOpenLocalTransformTransaction(long j);

    private static native void nSetParent(long j, int i, int i2);

    private static native void nSetTransform(long j, int i, float[] fArr);

    public void commitLocalTransformTransaction() {
        nCommitLocalTransformTransaction(this.mNativeObject);
    }

    @EntityInstance
    public int create(@Entity int i) {
        return nCreate(this.mNativeObject, i);
    }

    public void destroy(@Entity int i) {
        nDestroy(this.mNativeObject, i);
    }

    @EntityInstance
    public int getInstance(@Entity int i) {
        return nGetInstance(this.mNativeObject, i);
    }

    public long getNativeObject() {
        return this.mNativeObject;
    }

    public float[] getTransform(@EntityInstance int i, float[] fArr) {
        float[] assertMat4f = Asserts.assertMat4f(fArr);
        nGetTransform(this.mNativeObject, i, assertMat4f);
        return assertMat4f;
    }

    public float[] getWorldTransform(@EntityInstance int i, float[] fArr) {
        float[] assertMat4f = Asserts.assertMat4f(fArr);
        nGetWorldTransform(this.mNativeObject, i, assertMat4f);
        return assertMat4f;
    }

    public boolean hasComponent(@Entity int i) {
        return nHasComponent(this.mNativeObject, i);
    }

    public void openLocalTransformTransaction() {
        nOpenLocalTransformTransaction(this.mNativeObject);
    }

    public void setParent(@EntityInstance int i, @EntityInstance int i2) {
        nSetParent(this.mNativeObject, i, i2);
    }

    public void setTransform(@EntityInstance int i, float[] fArr) {
        Asserts.assertMat4fIn(fArr);
        nSetTransform(this.mNativeObject, i, fArr);
    }

    @EntityInstance
    public int create(@Entity int i, @EntityInstance int i2, float[] fArr) {
        return nCreateArray(this.mNativeObject, i, i2, fArr);
    }
}