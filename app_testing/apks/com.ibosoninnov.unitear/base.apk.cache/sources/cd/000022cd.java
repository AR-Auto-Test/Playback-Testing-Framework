package com.google.common.flogger.backend.system;

import com.google.common.flogger.backend.Tags;
import java.util.logging.Level;

/* loaded from: classes.dex */
public abstract class LoggingContext {
    public abstract Tags getTags();

    public abstract boolean shouldForceLogging(String str, Level level, boolean z);
}