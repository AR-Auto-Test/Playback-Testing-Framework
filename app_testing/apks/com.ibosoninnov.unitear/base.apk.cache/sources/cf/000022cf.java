package com.google.common.flogger.backend.system;

import com.google.common.flogger.backend.LogData;
import com.google.common.flogger.backend.SimpleMessageFormatter;
import java.util.logging.Level;

/* loaded from: classes.dex */
public final class SimpleLogRecord extends AbstractLogRecord implements SimpleMessageFormatter.SimpleLogHandler {
    private SimpleLogRecord(LogData logData) {
        super(logData);
        SimpleMessageFormatter.format(logData, this);
    }

    public static SimpleLogRecord create(LogData logData) {
        return new SimpleLogRecord(logData);
    }

    public static SimpleLogRecord error(RuntimeException runtimeException, LogData logData) {
        return new SimpleLogRecord(runtimeException, logData);
    }

    @Override // com.google.common.flogger.backend.SimpleMessageFormatter.SimpleLogHandler
    public void handleFormattedLogMessage(Level level, String str, Throwable th) {
        setMessage(str);
        setThrown(th);
    }

    private SimpleLogRecord(RuntimeException runtimeException, LogData logData) {
        super(runtimeException, logData);
    }
}