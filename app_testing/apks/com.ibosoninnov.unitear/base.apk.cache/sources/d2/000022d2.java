package com.google.common.flogger.backend.system;

import java.util.concurrent.TimeUnit;

/* loaded from: classes.dex */
public final class SystemClock extends Clock {
    private static final SystemClock INSTANCE = new SystemClock();

    private SystemClock() {
    }

    public static SystemClock getInstance() {
        return INSTANCE;
    }

    @Override // com.google.common.flogger.backend.system.Clock
    public long getCurrentTimeNanos() {
        return TimeUnit.MILLISECONDS.toNanos(System.currentTimeMillis());
    }

    public String toString() {
        return "Default millisecond precision clock";
    }
}