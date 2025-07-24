package com.google.ar.sceneform;

import java.util.concurrent.TimeUnit;

/* loaded from: classes.dex */
public class FrameTime {
    private static final float NANOSECONDS_TO_SECONDS = 1.0E-9f;
    private long lastNanoTime = 0;
    private long deltaNanoseconds = 0;

    public float getDeltaSeconds() {
        return ((float) this.deltaNanoseconds) * NANOSECONDS_TO_SECONDS;
    }

    public long getDeltaTime(TimeUnit timeUnit) {
        return timeUnit.convert(this.deltaNanoseconds, TimeUnit.NANOSECONDS);
    }

    public float getStartSeconds() {
        return ((float) this.lastNanoTime) * NANOSECONDS_TO_SECONDS;
    }

    public long getStartTime(TimeUnit timeUnit) {
        return timeUnit.convert(this.lastNanoTime, TimeUnit.NANOSECONDS);
    }

    public void update(long j) {
        long j2 = this.lastNanoTime;
        this.deltaNanoseconds = j2 != 0 ? j - j2 : 0L;
        this.lastNanoTime = j;
    }
}