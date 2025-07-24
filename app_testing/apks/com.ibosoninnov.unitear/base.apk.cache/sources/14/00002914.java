package com.google.mediapipe.framework;

import android.content.Context;

/* JADX WARN: Classes with same name are omitted:
  classes2.dex
 */
/* loaded from: base.apk:classes.jar:com/google/mediapipe/framework/MediaPipeRunner.class */
public abstract class MediaPipeRunner extends Graph {
    protected Context context;

    public abstract void start();

    public abstract void pause();

    public abstract void resume();

    public abstract void release();

    public MediaPipeRunner(Context context) {
        AssetCache.create(context);
        this.context = context;
    }

    public void loadBinaryGraphFromAsset(String assetPath) {
        try {
            loadBinaryGraph(AssetCache.getAssetCache().getAbsolutePathFromAsset(assetPath));
        } catch (MediaPipeException e2) {
        }
    }

    public void release(long timeoutMillis) {
        release();
    }
}