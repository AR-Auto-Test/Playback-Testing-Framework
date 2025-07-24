package com.google.common.flogger.backend.system;

import com.google.common.flogger.backend.LoggerBackend;
import java.util.logging.Logger;

/* loaded from: classes.dex */
public final class SimpleBackendFactory extends BackendFactory {
    private static final BackendFactory INSTANCE = new SimpleBackendFactory();

    private SimpleBackendFactory() {
    }

    public static BackendFactory getInstance() {
        return INSTANCE;
    }

    @Override // com.google.common.flogger.backend.system.BackendFactory
    public LoggerBackend create(String str) {
        return new SimpleLoggerBackend(Logger.getLogger(str.replace('$', '.')));
    }

    public String toString() {
        return "Default logger backend factory";
    }
}