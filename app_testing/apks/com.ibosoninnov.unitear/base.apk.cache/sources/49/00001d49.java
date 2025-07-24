package com.google.ar.sceneform.utilities;

/* loaded from: classes.dex */
public class TimeAccumulator {
    private long elapsedTimeMs;
    private long startSampleTimeMs;

    public void beginSample() {
        this.startSampleTimeMs = System.currentTimeMillis();
    }

    public void endSample() {
        this.elapsedTimeMs += System.currentTimeMillis() - this.startSampleTimeMs;
    }

    public long getElapsedTimeMs() {
        return this.elapsedTimeMs;
    }
}