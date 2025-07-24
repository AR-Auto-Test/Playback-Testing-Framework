package com.google.android.filament.gltfio;

/* loaded from: classes.dex */
public class Animator {
    private long mNativeObject;

    public Animator(long j) {
        this.mNativeObject = j;
    }

    private static native void nApplyAnimation(long j, int i, float f2);

    private static native int nGetAnimationCount(long j);

    private static native float nGetAnimationDuration(long j, int i);

    private static native String nGetAnimationName(long j, int i);

    private static native void nUpdateBoneMatrices(long j);

    public void applyAnimation(int i, float f2) {
        nApplyAnimation(this.mNativeObject, i, f2);
    }

    public int getAnimationCount() {
        return nGetAnimationCount(this.mNativeObject);
    }

    public float getAnimationDuration(int i) {
        return nGetAnimationDuration(this.mNativeObject, i);
    }

    public String getAnimationName(int i) {
        return nGetAnimationName(this.mNativeObject, i);
    }

    public void updateBoneMatrices() {
        nUpdateBoneMatrices(this.mNativeObject);
    }
}