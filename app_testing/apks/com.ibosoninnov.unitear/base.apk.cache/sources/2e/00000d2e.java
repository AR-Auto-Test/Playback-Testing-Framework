package com.google.android.filament.gltfio;

import com.google.android.filament.Engine;

/* loaded from: classes.dex */
public class MaterialProvider {
    private long mNativeObject;

    public MaterialProvider(Engine engine) {
        this.mNativeObject = nCreateMaterialProvider(engine.getNativeObject());
    }

    private static native long nCreateMaterialProvider(long j);

    private static native void nDestroyMaterialProvider(long j);

    public void destroy() {
        nDestroyMaterialProvider(this.mNativeObject);
        this.mNativeObject = 0L;
    }

    public long getNativeObject() {
        return this.mNativeObject;
    }
}