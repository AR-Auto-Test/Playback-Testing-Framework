package com.google.mediapipe.framework;

/* JADX WARN: Classes with same name are omitted:
  classes2.dex
 */
/* loaded from: base.apk:classes.jar:com/google/mediapipe/framework/GraphTextureFrame.class */
public class GraphTextureFrame implements TextureFrame {
    private long nativeBufferHandle;
    private int textureName;
    private int width;
    private int height;
    private long timestamp;

    private native void nativeReleaseBuffer(long nativeHandle);

    private native int nativeGetTextureName(long nativeHandle);

    private native int nativeGetWidth(long nativeHandle);

    private native int nativeGetHeight(long nativeHandle);

    /* JADX INFO: Access modifiers changed from: package-private */
    public GraphTextureFrame(long nativeHandle, long timestamp) {
        this.timestamp = Long.MIN_VALUE;
        this.nativeBufferHandle = nativeHandle;
        this.textureName = nativeGetTextureName(this.nativeBufferHandle);
        this.width = nativeGetWidth(this.nativeBufferHandle);
        this.height = nativeGetHeight(this.nativeBufferHandle);
        this.timestamp = timestamp;
    }

    @Override // com.google.mediapipe.framework.TextureFrame
    public int getTextureName() {
        return this.textureName;
    }

    @Override // com.google.mediapipe.framework.TextureFrame
    public int getWidth() {
        return this.width;
    }

    @Override // com.google.mediapipe.framework.TextureFrame
    public int getHeight() {
        return this.height;
    }

    @Override // com.google.mediapipe.framework.TextureFrame
    public long getTimestamp() {
        return this.timestamp;
    }

    @Override // com.google.mediapipe.framework.TextureFrame
    public void release() {
        if (this.nativeBufferHandle != 0) {
            nativeReleaseBuffer(this.nativeBufferHandle);
            this.nativeBufferHandle = 0L;
        }
    }

    @Override // com.google.mediapipe.framework.TextureFrame, com.google.mediapipe.framework.TextureReleaseCallback
    public void release(GlSyncToken syncToken) {
        syncToken.release();
        release();
    }
}