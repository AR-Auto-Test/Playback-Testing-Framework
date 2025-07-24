package com.google.common.flogger.backend.system;

import c.b.a.a.a;
import com.google.common.flogger.AbstractLogger;
import com.google.common.flogger.LogSite;
import com.google.common.flogger.backend.Platform;
import com.google.common.flogger.util.CallerFinder;
import com.google.common.flogger.util.StackBasedLogSite;

/* loaded from: classes.dex */
public final class StackBasedCallerFinder extends Platform.LogCallerFinder {
    private static final Platform.LogCallerFinder INSTANCE = new StackBasedCallerFinder();

    private StackBasedCallerFinder() {
    }

    public static Platform.LogCallerFinder getInstance() {
        return INSTANCE;
    }

    @Override // com.google.common.flogger.backend.Platform.LogCallerFinder
    public LogSite findLogSite(Class<?> cls, int i) {
        StackTraceElement findCallerOf = CallerFinder.findCallerOf(cls, new Throwable(), i + 1);
        return findCallerOf != null ? new StackBasedLogSite(findCallerOf) : LogSite.INVALID;
    }

    @Override // com.google.common.flogger.backend.Platform.LogCallerFinder
    public String findLoggingClass(Class<? extends AbstractLogger<?>> cls) {
        StackTraceElement findCallerOf = CallerFinder.findCallerOf(cls, new Throwable(), 1);
        if (findCallerOf != null) {
            return findCallerOf.getClassName();
        }
        StringBuilder x = a.x("no caller found on the stack for: ");
        x.append(cls.getName());
        throw new IllegalStateException(x.toString());
    }

    public String toString() {
        return "Default stack-based caller finder";
    }
}