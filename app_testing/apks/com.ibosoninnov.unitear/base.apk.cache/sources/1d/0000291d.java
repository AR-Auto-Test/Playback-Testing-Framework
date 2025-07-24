package com.google.mediapipe.framework;

import javax.annotation.Nullable;

/* JADX WARN: Classes with same name are omitted:
  classes2.dex
 */
/* loaded from: base.apk:classes.jar:com/google/mediapipe/framework/SurfaceOutput.class */
public class SurfaceOutput {
    private Packet surfaceHolderPacket;
    private Graph mediapipeGraph;

    private native void nativeSetFlipY(long nativePacket, boolean flip);

    private native void nativeSetSurface(long nativeContext, long nativePacket, Object surface);

    private native void nativeSetEglSurface(long nativeContext, long nativePacket, long nativeEglSurface);

    /* JADX INFO: Access modifiers changed from: package-private */
    public SurfaceOutput(Graph context, Packet holderPacket) {
        this.mediapipeGraph = context;
        this.surfaceHolderPacket = holderPacket;
    }

    public void setFlipY(boolean flip) {
        nativeSetFlipY(this.surfaceHolderPacket.getNativeHandle(), flip);
    }

    public void setSurface(@Nullable Object surface) {
        nativeSetSurface(this.mediapipeGraph.getNativeHandle(), this.surfaceHolderPacket.getNativeHandle(), surface);
    }

    public void setEglSurface(long nativeEglSurface) {
        nativeSetEglSurface(this.mediapipeGraph.getNativeHandle(), this.surfaceHolderPacket.getNativeHandle(), nativeEglSurface);
    }
}