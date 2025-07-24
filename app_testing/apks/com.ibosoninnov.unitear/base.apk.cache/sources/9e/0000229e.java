package com.google.common.flogger;

/* loaded from: classes.dex */
public final class LogSiteStackTrace extends Exception {
    public LogSiteStackTrace(Throwable th, StackSize stackSize, StackTraceElement[] stackTraceElementArr) {
        super(stackSize.toString(), th);
        setStackTrace(stackTraceElementArr);
    }

    @Override // java.lang.Throwable
    public Throwable fillInStackTrace() {
        return this;
    }
}