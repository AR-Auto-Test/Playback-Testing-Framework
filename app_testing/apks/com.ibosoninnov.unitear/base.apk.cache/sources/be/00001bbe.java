package com.google.ar.core;

import com.google.ar.core.exceptions.FatalException;

/* loaded from: classes.dex */
public enum PlaybackStatus {
    NONE(0),
    OK(1),
    IO_ERROR(2),
    FINISHED(3);
    
    public final int nativeCode;

    PlaybackStatus(int i) {
        this.nativeCode = i;
    }

    public static PlaybackStatus forNumber(int i) {
        PlaybackStatus[] values = values();
        for (int i2 = 0; i2 < 4; i2++) {
            PlaybackStatus playbackStatus = values[i2];
            if (playbackStatus.nativeCode == i) {
                return playbackStatus;
            }
        }
        throw new FatalException(c.b.a.a.a.g(61, "Unexpected value for native PlaybackStatus, value=", i));
    }
}