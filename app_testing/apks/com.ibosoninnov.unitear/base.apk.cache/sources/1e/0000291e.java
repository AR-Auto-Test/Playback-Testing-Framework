package com.google.mediapipe.framework;

/* JADX WARN: Classes with same name are omitted:
  classes2.dex
 */
/* loaded from: base.apk:classes.jar:com/google/mediapipe/framework/TextureFrame.class */
public interface TextureFrame extends TextureReleaseCallback {
    int getTextureName();

    int getWidth();

    int getHeight();

    long getTimestamp();

    void release();

    @Override // com.google.mediapipe.framework.TextureReleaseCallback
    void release(GlSyncToken syncToken);
}