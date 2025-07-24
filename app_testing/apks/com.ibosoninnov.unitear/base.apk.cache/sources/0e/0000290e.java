package com.google.mediapipe.framework;

/* JADX WARN: Classes with same name are omitted:
  classes2.dex
 */
/* loaded from: base.apk:classes.jar:com/google/mediapipe/framework/GraphGlSyncToken.class */
final class GraphGlSyncToken implements GlSyncToken {
    private long token;

    private static native void nativeWaitOnCpu(long token);

    private static native void nativeWaitOnGpu(long token);

    private static native void nativeRelease(long token);

    @Override // com.google.mediapipe.framework.GlSyncToken
    public void waitOnCpu() {
        if (this.token != 0) {
            nativeWaitOnCpu(this.token);
        }
    }

    @Override // com.google.mediapipe.framework.GlSyncToken
    public void waitOnGpu() {
        if (this.token != 0) {
            nativeWaitOnGpu(this.token);
        }
    }

    @Override // com.google.mediapipe.framework.GlSyncToken
    public void release() {
        if (this.token != 0) {
            nativeRelease(this.token);
            this.token = 0L;
        }
    }

    /* JADX INFO: Access modifiers changed from: package-private */
    public GraphGlSyncToken(long token) {
        this.token = token;
    }
}