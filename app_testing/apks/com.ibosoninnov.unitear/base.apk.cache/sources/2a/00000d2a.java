package com.google.android.filament.gltfio;

import com.google.android.filament.Engine;
import com.google.android.filament.EntityManager;
import java.nio.Buffer;

/* loaded from: classes.dex */
public class AssetLoader {
    private Engine mEngine;
    private long mNativeObject;

    public AssetLoader(Engine engine, MaterialProvider materialProvider, EntityManager entityManager) {
        long nCreateAssetLoader = nCreateAssetLoader(engine.getNativeObject(), materialProvider.getNativeObject(), entityManager.getNativeObject());
        this.mNativeObject = nCreateAssetLoader;
        if (nCreateAssetLoader != 0) {
            this.mEngine = engine;
            return;
        }
        throw new IllegalStateException("Unable to parse glTF asset.");
    }

    private static native long nCreateAssetFromBinary(long j, Buffer buffer, int i);

    private static native long nCreateAssetFromJson(long j, Buffer buffer, int i);

    private static native long nCreateAssetLoader(long j, long j2, long j3);

    private static native void nDestroyAsset(long j, long j2);

    private static native void nDestroyAssetLoader(long j);

    private static native void nEnableDiagnostics(long j, boolean z);

    public FilamentAsset createAssetFromBinary(Buffer buffer) {
        long nCreateAssetFromBinary = nCreateAssetFromBinary(this.mNativeObject, buffer, buffer.remaining());
        if (nCreateAssetFromBinary != 0) {
            return new FilamentAsset(this.mEngine, nCreateAssetFromBinary);
        }
        return null;
    }

    public FilamentAsset createAssetFromJson(Buffer buffer) {
        long nCreateAssetFromJson = nCreateAssetFromJson(this.mNativeObject, buffer, buffer.remaining());
        if (nCreateAssetFromJson != 0) {
            return new FilamentAsset(this.mEngine, nCreateAssetFromJson);
        }
        return null;
    }

    public void destroy() {
        nDestroyAssetLoader(this.mNativeObject);
        this.mNativeObject = 0L;
    }

    public void destroyAsset(FilamentAsset filamentAsset) {
        nDestroyAsset(this.mNativeObject, filamentAsset.getNativeObject());
        filamentAsset.clearNativeObject();
    }

    public void enableDiagnostics(boolean z) {
        nEnableDiagnostics(this.mNativeObject, z);
    }
}