package com.google.mediapipe.framework;

/* JADX WARN: Classes with same name are omitted:
  classes2.dex
 */
/* loaded from: base.apk:classes.jar:com/google/mediapipe/framework/GlSyncToken.class */
public interface GlSyncToken {
    void waitOnCpu();

    void waitOnGpu();

    void release();
}