package com.google.common.flogger.backend;

import java.util.logging.Level;

/* loaded from: classes.dex */
public abstract class LoggerBackend {
    public abstract String getLoggerName();

    public abstract void handleError(RuntimeException runtimeException, LogData logData);

    public abstract boolean isLoggable(Level level);

    public abstract void log(LogData logData);
}