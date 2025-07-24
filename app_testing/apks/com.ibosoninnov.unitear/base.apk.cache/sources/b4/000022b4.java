package com.google.common.flogger.backend;

/* loaded from: classes.dex */
public class LoggingException extends RuntimeException {
    public LoggingException(String str) {
        super(str);
    }

    public LoggingException(String str, Throwable th) {
        super(str, th);
    }
}