package com.google.ar.core;

import com.google.ar.core.exceptions.FatalException;

/* loaded from: classes.dex */
public enum RecordingStatus {
    NONE(0),
    OK(1),
    IO_ERROR(2);
    
    public final int nativeCode;

    RecordingStatus(int i) {
        this.nativeCode = i;
    }

    public static RecordingStatus forNumber(int i) {
        RecordingStatus[] values = values();
        for (int i2 = 0; i2 < 3; i2++) {
            RecordingStatus recordingStatus = values[i2];
            if (recordingStatus.nativeCode == i) {
                return recordingStatus;
            }
        }
        throw new FatalException(c.b.a.a.a.g(62, "Unexpected value for native RecordingStatus, value=", i));
    }
}