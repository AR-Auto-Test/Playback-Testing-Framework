package com.google.mediapipe.framework;

/* JADX WARN: Classes with same name are omitted:
  classes2.dex
 */
/* loaded from: base.apk:classes.jar:com/google/mediapipe/framework/AppTextureFrame.class */
public class AppTextureFrame implements TextureFrame {
    private int textureName;
    private int width;
    private int height;
    private long timestamp = Long.MIN_VALUE;
    private boolean inUse = false;
    private boolean legacyInUse = false;
    private GlSyncToken releaseSyncToken = null;

    public AppTextureFrame(int textureName, int width, int height) {
        this.textureName = textureName;
        this.width = width;
        this.height = height;
    }

    public void setTimestamp(long timestamp) {
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

    public void waitUntilReleased() throws InterruptedException {
        synchronized (this) {
            while (this.inUse && this.releaseSyncToken == null) {
                wait();
            }
            if (this.releaseSyncToken != null) {
                this.releaseSyncToken.waitOnCpu();
                this.releaseSyncToken.release();
                this.inUse = false;
                this.releaseSyncToken = null;
            }
        }
    }

    @Deprecated
    public boolean getInUse() {
        boolean z;
        synchronized (this) {
            z = this.legacyInUse;
        }
        return z;
    }

    public void setInUse() {
        synchronized (this) {
            if (this.releaseSyncToken != null) {
                this.releaseSyncToken.release();
                this.releaseSyncToken = null;
            }
            this.inUse = true;
            this.legacyInUse = true;
        }
    }

    @Override // com.google.mediapipe.framework.TextureFrame
    public void release() {
        synchronized (this) {
            this.inUse = false;
            this.legacyInUse = false;
            notifyAll();
        }
    }

    @Override // com.google.mediapipe.framework.TextureFrame, com.google.mediapipe.framework.TextureReleaseCallback
    public void release(GlSyncToken syncToken) {
        synchronized (this) {
            if (this.releaseSyncToken != null) {
                this.releaseSyncToken.release();
                this.releaseSyncToken = null;
            }
            this.releaseSyncToken = syncToken;
            this.legacyInUse = false;
            notifyAll();
        }
    }

    public void finalize() {
        if (this.releaseSyncToken != null) {
            this.releaseSyncToken.release();
            this.releaseSyncToken = null;
        }
    }
}