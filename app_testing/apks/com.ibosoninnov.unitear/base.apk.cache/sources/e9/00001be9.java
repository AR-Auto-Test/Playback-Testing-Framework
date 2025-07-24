package com.google.ar.core;

import com.google.ar.core.exceptions.FatalException;

/* loaded from: classes.dex */
public enum TrackingState {
    TRACKING(0),
    PAUSED(1),
    STOPPED(2);
    
    public final int nativeCode;

    TrackingState(int i) {
        this.nativeCode = i;
    }

    public static TrackingState forNumber(int i) {
        TrackingState[] values = values();
        for (int i2 = 0; i2 < 3; i2++) {
            TrackingState trackingState = values[i2];
            if (trackingState.nativeCode == i) {
                return trackingState;
            }
        }
        throw new FatalException(c.b.a.a.a.g(60, "Unexpected value for native TrackingState, value=", i));
    }
}