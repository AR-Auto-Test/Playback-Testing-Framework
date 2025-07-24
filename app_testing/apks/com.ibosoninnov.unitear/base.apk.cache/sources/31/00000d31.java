package com.google.android.filament.gltfio;

import com.google.android.filament.Engine;
import java.nio.Buffer;

/* loaded from: classes.dex */
public class ResourceLoader {
    private final long mNativeObject;

    public ResourceLoader(Engine engine) {
        this.mNativeObject = nCreateResourceLoader(engine.getNativeObject());
    }

    private static native void nAddResourceData(long j, String str, Buffer buffer, int i);

    private static native boolean nAsyncBeginLoad(long j, long j2);

    private static native float nAsyncGetLoadProgress(long j);

    private static native void nAsyncUpdateLoad(long j);

    private static native long nCreateResourceLoader(long j);

    private static native void nDestroyResourceLoader(long j);

    private static native boolean nHasResourceData(long j, String str);

    private static native void nLoadResources(long j, long j2);

    public ResourceLoader addResourceData(String str, Buffer buffer) {
        nAddResourceData(this.mNativeObject, str, buffer, buffer.remaining());
        return this;
    }

    public boolean asyncBeginLoad(FilamentAsset filamentAsset) {
        return nAsyncBeginLoad(this.mNativeObject, filamentAsset.getNativeObject());
    }

    public float asyncGetLoadProgress() {
        return nAsyncGetLoadProgress(this.mNativeObject);
    }

    public void asyncUpdateLoad() {
        nAsyncUpdateLoad(this.mNativeObject);
    }

    public void destroy() {
        nDestroyResourceLoader(this.mNativeObject);
    }

    public boolean hasResourceData(String str) {
        return nHasResourceData(this.mNativeObject, str);
    }

    public ResourceLoader loadResources(FilamentAsset filamentAsset) {
        nLoadResources(this.mNativeObject, filamentAsset.getNativeObject());
        return this;
    }
}