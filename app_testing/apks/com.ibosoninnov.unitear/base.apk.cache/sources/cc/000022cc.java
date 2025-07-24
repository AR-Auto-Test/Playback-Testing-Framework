package com.google.common.flogger.backend.system;

import com.google.common.flogger.backend.Tags;
import java.util.logging.Level;

/* loaded from: classes.dex */
public final class EmptyLoggingContext extends LoggingContext {
    private static final LoggingContext INSTANCE = new EmptyLoggingContext();

    private EmptyLoggingContext() {
    }

    public static LoggingContext getInstance() {
        return INSTANCE;
    }

    @Override // com.google.common.flogger.backend.system.LoggingContext
    public Tags getTags() {
        return Tags.empty();
    }

    @Override // com.google.common.flogger.backend.system.LoggingContext
    public boolean shouldForceLogging(String str, Level level, boolean z) {
        return false;
    }

    public String toString() {
        return "Empty logging context";
    }
}