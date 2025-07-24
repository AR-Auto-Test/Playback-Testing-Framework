package com.google.android.filament;

import com.google.android.filament.proguard.UsedByReflection;

/* loaded from: classes.dex */
public class EntityManager {
    private long mNativeObject;

    /* loaded from: classes.dex */
    public static class Holder {
        public static final EntityManager INSTANCE = new EntityManager();

        private Holder() {
        }
    }

    public static EntityManager get() {
        return Holder.INSTANCE;
    }

    private static native int nCreate(long j);

    private static native void nCreateArray(long j, int i, int[] iArr);

    private static native void nDestroy(long j, int i);

    private static native void nDestroyArray(long j, int i, int[] iArr);

    private static native long nGetEntityManager();

    private static native boolean nIsAlive(long j, int i);

    @Entity
    public int create() {
        return nCreate(this.mNativeObject);
    }

    public void destroy(@Entity int i) {
        nDestroy(this.mNativeObject, i);
    }

    @UsedByReflection("AssetLoader.java")
    public long getNativeObject() {
        return this.mNativeObject;
    }

    public boolean isAlive(@Entity int i) {
        return nIsAlive(this.mNativeObject, i);
    }

    private EntityManager() {
        this.mNativeObject = nGetEntityManager();
    }

    @Entity
    public int[] create(int i) {
        if (i >= 1) {
            int[] iArr = new int[i];
            nCreateArray(this.mNativeObject, i, iArr);
            return iArr;
        }
        throw new ArrayIndexOutOfBoundsException("n must be at least 1");
    }

    public void destroy(@Entity int[] iArr) {
        nDestroyArray(this.mNativeObject, iArr.length, iArr);
    }

    public int[] create(@Entity int[] iArr) {
        nCreateArray(this.mNativeObject, iArr.length, iArr);
        return iArr;
    }
}