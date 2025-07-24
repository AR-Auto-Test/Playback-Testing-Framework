package com.google.common.flogger.backend.system;

import com.google.common.flogger.backend.LogData;
import java.util.logging.Logger;

/* loaded from: classes.dex */
public class SimpleLoggerBackend extends AbstractBackend {
    public SimpleLoggerBackend(Logger logger) {
        super(logger);
    }

    @Override // com.google.common.flogger.backend.LoggerBackend
    public void handleError(RuntimeException runtimeException, LogData logData) {
        log(SimpleLogRecord.error(runtimeException, logData), logData.wasForced());
    }

    @Override // com.google.common.flogger.backend.LoggerBackend
    public void log(LogData logData) {
        log(SimpleLogRecord.create(logData), logData.wasForced());
    }
}