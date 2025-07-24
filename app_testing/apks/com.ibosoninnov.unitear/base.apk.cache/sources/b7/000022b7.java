package com.google.common.flogger.backend;

import com.google.common.flogger.AbstractLogger;
import com.google.common.flogger.LogSite;
import java.lang.reflect.InvocationTargetException;
import java.util.concurrent.TimeUnit;
import java.util.logging.Level;

/* loaded from: classes.dex */
public abstract class Platform {
    private static String ANDROID_PLATFORM = "com.google.common.flogger.backend.android.AndroidPlatform";
    private static final String[] AVAILABLE_PLATFORMS = {"com.google.common.flogger.backend.system.DefaultPlatform"};
    private static String DEFAULT_PLATFORM = "com.google.common.flogger.backend.system.DefaultPlatform";

    /* loaded from: classes.dex */
    public static final class LazyHolder {
        private static final Platform INSTANCE = loadFirstAvailablePlatform(Platform.AVAILABLE_PLATFORMS);

        private LazyHolder() {
        }

        private static Platform loadFirstAvailablePlatform(String[] strArr) {
            Platform platform;
            StringBuilder sb = new StringBuilder();
            try {
                platform = PlatformProvider.getPlatform();
            } catch (NoClassDefFoundError unused) {
                platform = null;
            }
            if (platform != null) {
                return platform;
            }
            for (String str : strArr) {
                try {
                    return (Platform) Class.forName(str).getConstructor(new Class[0]).newInstance(new Object[0]);
                } catch (Throwable th) {
                    th = th;
                    if (th instanceof InvocationTargetException) {
                        th = th.getCause();
                    }
                    sb.append('\n');
                    sb.append(str);
                    sb.append(": ");
                    sb.append(th);
                }
            }
            throw new IllegalStateException(sb.insert(0, "No logging platforms found:").toString());
        }
    }

    /* loaded from: classes.dex */
    public static abstract class LogCallerFinder {
        public abstract LogSite findLogSite(Class<?> cls, int i);

        public abstract String findLoggingClass(Class<? extends AbstractLogger<?>> cls);
    }

    public static LoggerBackend getBackend(String str) {
        return LazyHolder.INSTANCE.getBackendImpl(str);
    }

    public static LogCallerFinder getCallerFinder() {
        return LazyHolder.INSTANCE.getCallerFinderImpl();
    }

    public static String getConfigInfo() {
        return LazyHolder.INSTANCE.getConfigInfoImpl();
    }

    public static long getCurrentTimeNanos() {
        return LazyHolder.INSTANCE.getCurrentTimeNanosImpl();
    }

    public static Tags getInjectedTags() {
        return LazyHolder.INSTANCE.getInjectedTagsImpl();
    }

    public static boolean shouldForceLogging(String str, Level level, boolean z) {
        return LazyHolder.INSTANCE.shouldForceLoggingImpl(str, level, z);
    }

    public abstract LoggerBackend getBackendImpl(String str);

    public abstract LogCallerFinder getCallerFinderImpl();

    public abstract String getConfigInfoImpl();

    public long getCurrentTimeNanosImpl() {
        return TimeUnit.MILLISECONDS.toNanos(System.currentTimeMillis());
    }

    public Tags getInjectedTagsImpl() {
        return Tags.empty();
    }

    public boolean shouldForceLoggingImpl(String str, Level level, boolean z) {
        return false;
    }
}