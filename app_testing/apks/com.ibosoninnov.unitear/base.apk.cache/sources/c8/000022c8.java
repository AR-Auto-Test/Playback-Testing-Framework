package com.google.common.flogger.backend.system;

import com.google.common.flogger.LogSite;
import com.google.common.flogger.backend.LogData;
import com.google.common.flogger.backend.Metadata;
import com.google.common.flogger.backend.SimpleMessageFormatter;
import java.util.Arrays;
import java.util.concurrent.TimeUnit;
import java.util.logging.Level;
import java.util.logging.LogRecord;

/* loaded from: classes.dex */
public abstract class AbstractLogRecord extends LogRecord {
    private final LogData data;

    public AbstractLogRecord(LogData logData) {
        super(logData.getLevel(), null);
        this.data = logData;
        LogSite logSite = logData.getLogSite();
        setSourceClassName(logSite.getClassName());
        setSourceMethodName(logSite.getMethodName());
        setLoggerName(logData.getLoggerName());
        setMillis(TimeUnit.NANOSECONDS.toMillis(logData.getTimestampNanos()));
    }

    private static void safeAppend(LogData logData, StringBuilder sb) {
        Object[] arguments;
        sb.append("  original message: ");
        if (logData.getTemplateContext() == null) {
            sb.append(logData.getLiteralArgument());
        } else {
            sb.append(logData.getTemplateContext().getMessage());
            sb.append("\n  original arguments:");
            for (Object obj : logData.getArguments()) {
                sb.append("\n    ");
                sb.append(SimpleMessageFormatter.safeToString(obj));
            }
        }
        Metadata metadata = logData.getMetadata();
        if (metadata.size() > 0) {
            sb.append("\n  metadata:");
            for (int i = 0; i < metadata.size(); i++) {
                sb.append("\n    ");
                sb.append(metadata.getKey(i).getLabel());
                sb.append(": ");
                sb.append(metadata.getValue(i));
            }
        }
        sb.append("\n  level: ");
        sb.append(logData.getLevel());
        sb.append("\n  timestamp (nanos): ");
        sb.append(logData.getTimestampNanos());
        sb.append("\n  class: ");
        sb.append(logData.getLogSite().getClassName());
        sb.append("\n  method: ");
        sb.append(logData.getLogSite().getMethodName());
        sb.append("\n  line number: ");
        sb.append(logData.getLogSite().getLineNumber());
    }

    public final LogData getLogData() {
        return this.data;
    }

    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append(getClass().getSimpleName());
        sb.append(" {\n  message: ");
        sb.append(getMessage());
        sb.append("\n  arguments: ");
        sb.append(getParameters() != null ? Arrays.asList(getParameters()) : "<none>");
        sb.append('\n');
        safeAppend(getLogData(), sb);
        sb.append("\n}");
        return sb.toString();
    }

    public AbstractLogRecord(RuntimeException runtimeException, LogData logData) {
        this(logData);
        int intValue = logData.getLevel().intValue();
        Level level = Level.WARNING;
        setLevel(intValue >= level.intValue() ? logData.getLevel() : level);
        setThrown(runtimeException);
        StringBuilder sb = new StringBuilder("LOGGING ERROR: ");
        sb.append(runtimeException.getMessage());
        sb.append('\n');
        safeAppend(logData, sb);
        setMessage(sb.toString());
    }
}