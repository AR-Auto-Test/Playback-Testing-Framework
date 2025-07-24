package com.google.common.flogger;

import com.google.common.flogger.LoggingApi;
import com.google.common.flogger.backend.LoggerBackend;
import com.google.common.flogger.backend.Platform;
import com.google.common.flogger.parser.DefaultPrintfMessageParser;
import com.google.common.flogger.parser.MessageParser;
import com.google.errorprone.annotations.CheckReturnValue;
import java.util.logging.Level;

@CheckReturnValue
/* loaded from: classes.dex */
public final class FluentLogger extends AbstractLogger<Api> {
    public static final NoOp NO_OP = new NoOp();

    /* loaded from: classes.dex */
    public interface Api extends LoggingApi<Api> {
    }

    /* loaded from: classes.dex */
    public final class Context extends LogContext<FluentLogger, Api> implements Api {
        /* JADX DEBUG: Method merged with bridge method */
        /* JADX WARN: Can't rename method to resolve collision */
        @Override // com.google.common.flogger.LogContext
        public Api api() {
            return this;
        }

        @Override // com.google.common.flogger.LogContext
        public MessageParser getMessageParser() {
            return DefaultPrintfMessageParser.getInstance();
        }

        private Context(Level level, boolean z) {
            super(level, z);
        }

        /* JADX DEBUG: Method merged with bridge method */
        @Override // com.google.common.flogger.LogContext
        public FluentLogger getLogger() {
            return FluentLogger.this;
        }

        /* JADX DEBUG: Method merged with bridge method */
        /* JADX WARN: Can't rename method to resolve collision */
        @Override // com.google.common.flogger.LogContext
        public Api noOp() {
            return FluentLogger.NO_OP;
        }
    }

    /* loaded from: classes.dex */
    public static final class NoOp extends LoggingApi.NoOp<Api> implements Api {
        private NoOp() {
        }
    }

    public FluentLogger(LoggerBackend loggerBackend) {
        super(loggerBackend);
    }

    public static FluentLogger forEnclosingClass() {
        return new FluentLogger(Platform.getBackend(Platform.getCallerFinder().findLoggingClass(FluentLogger.class)));
    }

    /* JADX DEBUG: Method merged with bridge method */
    /* JADX WARN: Can't rename method to resolve collision */
    @Override // com.google.common.flogger.AbstractLogger
    public Api at(Level level) {
        boolean isLoggable = isLoggable(level);
        boolean shouldForceLogging = Platform.shouldForceLogging(getName(), level, isLoggable);
        return (isLoggable || shouldForceLogging) ? new Context(level, shouldForceLogging) : NO_OP;
    }
}