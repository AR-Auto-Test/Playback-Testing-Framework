package com.google.ar.sceneform.utilities;

import com.google.android.material.shadow.ShadowDrawableWrapper;

/* loaded from: classes.dex */
public class MovingAverageMillisecondsTracker {
    private static final double NANOSECONDS_TO_MILLISECONDS = 1.0E-6d;
    private long beginSampleTimestampNano;
    private final Clock clock;
    private MovingAverage movingAverage;
    private final double weight;

    /* loaded from: classes.dex */
    public interface Clock {
        long getNanoseconds();
    }

    /* loaded from: classes.dex */
    public static class DefaultClock implements Clock {
        private DefaultClock() {
        }

        @Override // com.google.ar.sceneform.utilities.MovingAverageMillisecondsTracker.Clock
        public long getNanoseconds() {
            return System.nanoTime();
        }
    }

    public MovingAverageMillisecondsTracker() {
        this(0.8999999761581421d);
    }

    public void beginSample() {
        this.beginSampleTimestampNano = this.clock.getNanoseconds();
    }

    public void endSample() {
        double nanoseconds = (this.clock.getNanoseconds() - this.beginSampleTimestampNano) * NANOSECONDS_TO_MILLISECONDS;
        MovingAverage movingAverage = this.movingAverage;
        if (movingAverage == null) {
            this.movingAverage = new MovingAverage(nanoseconds, this.weight);
        } else {
            movingAverage.addSample(nanoseconds);
        }
    }

    public double getAverage() {
        MovingAverage movingAverage = this.movingAverage;
        return movingAverage != null ? movingAverage.getAverage() : ShadowDrawableWrapper.COS_45;
    }

    public MovingAverageMillisecondsTracker(double d2) {
        this.weight = d2;
        this.clock = new DefaultClock();
    }

    public MovingAverageMillisecondsTracker(Clock clock) {
        this(clock, 0.8999999761581421d);
    }

    public MovingAverageMillisecondsTracker(Clock clock, double d2) {
        this.weight = d2;
        this.clock = clock;
    }
}