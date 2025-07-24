package com.google.ar.core;

import com.google.ar.core.exceptions.FatalException;

/* loaded from: classes.dex */
public enum TrackingFailureReason {
    NONE(0),
    BAD_STATE(1),
    INSUFFICIENT_LIGHT(2),
    EXCESSIVE_MOTION(3),
    INSUFFICIENT_FEATURES(4),
    CAMERA_UNAVAILABLE(5);
    
    public final int nativeCode;

    TrackingFailureReason(int i) {
        this.nativeCode = i;
    }

    public static TrackingFailureReason forNumber(int i) {
        TrackingFailureReason[] values = values();
        for (int i2 = 0; i2 < 6; i2++) {
            TrackingFailureReason trackingFailureReason = values[i2];
            if (trackingFailureReason.nativeCode == i) {
                return trackingFailureReason;
            }
        }
        throw new FatalException(c.b.a.a.a.g(68, "Unexpected value for native TrackingFailureReason, value=", i));
    }
}