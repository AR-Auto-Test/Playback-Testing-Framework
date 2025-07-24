package com.google.mediapipe.framework;

import android.content.Context;
import android.content.res.AssetManager;
import com.google.common.io.ByteStreams;
import java.io.IOException;
import java.io.InputStream;

/* JADX WARN: Classes with same name are omitted:
  classes2.dex
 */
/* loaded from: base.apk:classes.jar:com/google/mediapipe/framework/AndroidAssetUtil.class */
public final class AndroidAssetUtil {
    private static native boolean nativeInitializeAssetManager(Context androidContext, String cacheDirPath);

    public static byte[] getAssetBytes(AssetManager assets, String assetName) {
        try {
            InputStream stream = assets.open(assetName);
            byte[] assetData = ByteStreams.toByteArray(stream);
            stream.close();
            return assetData;
        } catch (IOException e2) {
            throw new RuntimeException(e2);
        }
    }

    public static synchronized boolean initializeNativeAssetManager(Context androidContext) {
        return nativeInitializeAssetManager(androidContext, androidContext.getCacheDir().getAbsolutePath());
    }

    private AndroidAssetUtil() {
    }
}