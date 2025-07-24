package com.google.common.flogger;

import c.b.a.a.a;
import com.google.common.flogger.LoggingApi;
import com.google.common.flogger.backend.LogData;
import com.google.common.flogger.backend.LoggerBackend;
import com.google.common.flogger.backend.LoggingException;
import com.google.common.flogger.util.Checks;
import com.google.errorprone.annotations.CheckReturnValue;
import java.io.PrintStream;
import java.util.logging.Level;

@CheckReturnValue
/* loaded from: classes.dex */
public abstract class AbstractLogger<API extends LoggingApi<API>> {
    private final LoggerBackend backend;

    public AbstractLogger(LoggerBackend loggerBackend) {
        this.backend = (LoggerBackend) Checks.checkNotNull(loggerBackend, "backend");
    }

    public abstract API at(Level level);

    public final API atConfig() {
        return at(Level.CONFIG);
    }

    public final API atFine() {
        return at(Level.FINE);
    }

    public final API atFiner() {
        return at(Level.FINER);
    }

    public final API atFinest() {
        return at(Level.FINEST);
    }

    public final API atInfo() {
        return at(Level.INFO);
    }

    public final API atSevere() {
        return at(Level.SEVERE);
    }

    public final API atWarning() {
        return at(Level.WARNING);
    }

    public final LoggerBackend getBackend() {
        return this.backend;
    }

    public String getName() {
        return this.backend.getLoggerName();
    }

    public final boolean isLoggable(Level level) {
        return this.backend.isLoggable(level);
    }

    public final void write(LogData logData) {
        Checks.checkNotNull(logData, "data");
        try {
            this.backend.log(logData);
        } catch (RuntimeException e2) {
            try {
                this.backend.handleError(e2, logData);
            } catch (LoggingException e3) {
                throw e3;
            } catch (RuntimeException e4) {
                PrintStream printStream = System.err;
                StringBuilder x = a.x("logging error: ");
                x.append(e4.getMessage());
                printStream.println(x.toString());
                e4.printStackTrace(System.err);
            }
        }
    }
}