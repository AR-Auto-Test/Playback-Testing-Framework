package com.google.android.filament.gltfio;

import com.google.android.filament.Box;
import com.google.android.filament.Engine;
import com.google.android.filament.Entity;
import com.google.android.filament.MaterialInstance;

/* loaded from: classes.dex */
public class FilamentAsset {
    private Animator mAnimator = null;
    private Engine mEngine;
    private long mNativeObject;

    public FilamentAsset(Engine engine, long j) {
        this.mEngine = engine;
        this.mNativeObject = j;
    }

    private static native long nGetAnimator(long j);

    private static native void nGetBoundingBox(long j, float[] fArr);

    private static native void nGetEntities(long j, int[] iArr);

    private static native int nGetEntityCount(long j);

    private static native int nGetMaterialInstanceCount(long j);

    private static native void nGetMaterialInstances(long j, long[] jArr);

    private static native String nGetName(long j, int i);

    private static native int nGetResourceUriCount(long j);

    private static native void nGetResourceUris(long j, String[] strArr);

    private static native int nGetRoot(long j);

    private static native int nPopRenderable(long j);

    private static native int nPopRenderables(long j, int[] iArr);

    private static native void nReleaseSourceData(long j);

    public void clearNativeObject() {
        this.mNativeObject = 0L;
    }

    public Animator getAnimator() {
        Animator animator = this.mAnimator;
        if (animator != null) {
            return animator;
        }
        Animator animator2 = new Animator(nGetAnimator(getNativeObject()));
        this.mAnimator = animator2;
        return animator2;
    }

    public Box getBoundingBox() {
        float[] fArr = new float[6];
        nGetBoundingBox(this.mNativeObject, fArr);
        return new Box(fArr[0], fArr[1], fArr[2], fArr[3], fArr[4], fArr[5]);
    }

    @Entity
    public int[] getEntities() {
        int[] iArr = new int[nGetEntityCount(this.mNativeObject)];
        nGetEntities(this.mNativeObject, iArr);
        return iArr;
    }

    public MaterialInstance[] getMaterialInstances() {
        int nGetMaterialInstanceCount = nGetMaterialInstanceCount(this.mNativeObject);
        MaterialInstance[] materialInstanceArr = new MaterialInstance[nGetMaterialInstanceCount];
        long[] jArr = new long[nGetMaterialInstanceCount];
        nGetMaterialInstances(this.mNativeObject, jArr);
        for (int i = 0; i < nGetMaterialInstanceCount; i++) {
            materialInstanceArr[i] = new MaterialInstance(this.mEngine, jArr[i]);
        }
        return materialInstanceArr;
    }

    public String getName(@Entity int i) {
        return nGetName(getNativeObject(), i);
    }

    public long getNativeObject() {
        return this.mNativeObject;
    }

    public String[] getResourceUris() {
        String[] strArr = new String[nGetResourceUriCount(this.mNativeObject)];
        nGetResourceUris(this.mNativeObject, strArr);
        return strArr;
    }

    @Entity
    public int getRoot() {
        return nGetRoot(this.mNativeObject);
    }

    @Entity
    public int popRenderable() {
        return nPopRenderable(this.mNativeObject);
    }

    public int popRenderables(@Entity int[] iArr) {
        return nPopRenderables(this.mNativeObject, iArr);
    }

    public void releaseSourceData() {
        nReleaseSourceData(this.mNativeObject);
    }
}