package com.google.common.flogger;

import com.google.common.flogger.util.Checks;
import com.google.errorprone.annotations.CheckReturnValue;
import java.util.Map;
import java.util.ResourceBundle;
import java.util.concurrent.ConcurrentHashMap;
import java.util.logging.Filter;
import java.util.logging.Handler;
import java.util.logging.Level;
import java.util.logging.Logger;

@CheckReturnValue
/* loaded from: classes.dex */
public final class LoggerConfig {
    private static final Map<String, LoggerConfig> strongRefMap = new ConcurrentHashMap();
    private final Logger logger;

    private LoggerConfig(String str) {
        this.logger = (Logger) Checks.checkNotNull(Logger.getLogger(str), "logger");
    }

    public static LoggerConfig getConfig(Class<?> cls) {
        return getConfig(cls.getName());
    }

    public static LoggerConfig getPackageConfig(Class<?> cls) {
        return getConfig(cls.getPackage().getName());
    }

    public static LoggerConfig of(AbstractLogger<?> abstractLogger) {
        Checks.checkArgument(abstractLogger.getName() != null, "cannot obtain configuration for an anonymous logger");
        return getConfig(abstractLogger.getName());
    }

    public void addHandler(Handler handler) {
        Checks.checkNotNull(handler, "handler");
        this.logger.addHandler(handler);
    }

    public Filter getFilter() {
        return this.logger.getFilter();
    }

    public Handler[] getHandlers() {
        return this.logger.getHandlers();
    }

    public Level getLevel() {
        return this.logger.getLevel();
    }

    public String getName() {
        return this.logger.getName();
    }

    public Logger getParent() {
        return this.logger.getParent();
    }

    public ResourceBundle getResourceBundle() {
        return this.logger.getResourceBundle();
    }

    public String getResourceBundleName() {
        return this.logger.getResourceBundleName();
    }

    public boolean getUseParentHandlers() {
        return this.logger.getUseParentHandlers();
    }

    public void removeHandler(Handler handler) {
        Checks.checkNotNull(handler, "handler");
        this.logger.removeHandler(handler);
    }

    public void setFilter(Filter filter) {
        this.logger.setFilter(filter);
    }

    public void setLevel(Level level) {
        this.logger.setLevel(level);
    }

    public void setUseParentHandlers(boolean z) {
        this.logger.setUseParentHandlers(z);
    }

    public static LoggerConfig getConfig(String str) {
        Map<String, LoggerConfig> map = strongRefMap;
        LoggerConfig loggerConfig = map.get(Checks.checkNotNull(str, "logger name"));
        if (loggerConfig == null) {
            LoggerConfig loggerConfig2 = new LoggerConfig(str);
            map.put(str, loggerConfig2);
            return loggerConfig2;
        }
        return loggerConfig;
    }
}