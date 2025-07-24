package com.google.common.flogger.backend.system;

import com.google.common.flogger.backend.LoggerBackend;
import java.util.logging.Filter;
import java.util.logging.Handler;
import java.util.logging.Level;
import java.util.logging.LogRecord;
import java.util.logging.Logger;

/* loaded from: classes.dex */
public abstract class AbstractBackend extends LoggerBackend {
    private static volatile boolean cannotUseForcingLogger = false;
    private final Logger logger;

    public AbstractBackend(Logger logger) {
        this.logger = logger;
    }

    private static void publish(Logger logger, LogRecord logRecord) {
        Logger parent;
        for (Handler handler : logger.getHandlers()) {
            handler.publish(logRecord);
        }
        if (!logger.getUseParentHandlers() || (parent = logger.getParent()) == null) {
            return;
        }
        publish(parent, logRecord);
    }

    public void forceLoggingViaChildLogger(LogRecord logRecord) {
        Logger forcingLogger = getForcingLogger(this.logger);
        try {
            forcingLogger.setLevel(Level.ALL);
            forcingLogger.log(logRecord);
        } catch (SecurityException unused) {
            cannotUseForcingLogger = true;
            Logger.getLogger("").log(Level.SEVERE, "Forcing log statements with Flogger has been partially disabled.\nThe Flogger library cannot modify logger log levels, which is necessary to force log statements. This is likely due to an installed SecurityManager.\nForced log statements will still be published directly to log handlers, but will not be visible to the 'log(LogRecord)' method of Logger sub-classes.\n");
            publish(this.logger, logRecord);
        }
    }

    public Logger getForcingLogger(Logger logger) {
        return Logger.getLogger(logger.getName() + ".__forced__");
    }

    @Override // com.google.common.flogger.backend.LoggerBackend
    public final String getLoggerName() {
        return this.logger.getName();
    }

    @Override // com.google.common.flogger.backend.LoggerBackend
    public final boolean isLoggable(Level level) {
        return this.logger.isLoggable(level);
    }

    public void log(LogRecord logRecord, boolean z) {
        if (z && !this.logger.isLoggable(logRecord.getLevel())) {
            Filter filter = this.logger.getFilter();
            if (filter != null) {
                filter.isLoggable(logRecord);
            }
            if (this.logger.getClass() != Logger.class && !cannotUseForcingLogger) {
                forceLoggingViaChildLogger(logRecord);
                return;
            } else {
                publish(this.logger, logRecord);
                return;
            }
        }
        this.logger.log(logRecord);
    }
}