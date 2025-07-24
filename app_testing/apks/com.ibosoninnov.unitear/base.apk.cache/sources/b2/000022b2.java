package com.google.common.flogger.backend;

import com.google.common.flogger.LogSite;
import java.util.logging.Level;

/* loaded from: classes.dex */
public interface LogData {
    Object[] getArguments();

    Level getLevel();

    Object getLiteralArgument();

    LogSite getLogSite();

    String getLoggerName();

    Metadata getMetadata();

    TemplateContext getTemplateContext();

    @Deprecated
    long getTimestampMicros();

    long getTimestampNanos();

    boolean wasForced();
}