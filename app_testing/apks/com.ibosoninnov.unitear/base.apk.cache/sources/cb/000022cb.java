package com.google.common.flogger.backend.system;

import c.b.a.a.a;
import com.google.common.flogger.backend.LoggerBackend;
import com.google.common.flogger.backend.Platform;
import com.google.common.flogger.backend.Tags;
import com.google.common.flogger.util.Checks;
import java.io.PrintStream;
import java.util.logging.Level;

/* loaded from: classes.dex */
public class DefaultPlatform extends Platform {
    private static final String BACKEND_FACTORY = "backend_factory";
    private static final String CLOCK = "clock";
    private static final String LOGGING_CONTEXT = "logging_context";
    private final BackendFactory backendFactory;
    private final Platform.LogCallerFinder callerFinder;
    private final Clock clock;
    private final LoggingContext context;

    public DefaultPlatform() {
        BackendFactory backendFactory = (BackendFactory) resolveAttribute(BACKEND_FACTORY, BackendFactory.class);
        this.backendFactory = backendFactory == null ? SimpleBackendFactory.getInstance() : backendFactory;
        LoggingContext loggingContext = (LoggingContext) resolveAttribute(LOGGING_CONTEXT, LoggingContext.class);
        this.context = loggingContext == null ? EmptyLoggingContext.getInstance() : loggingContext;
        Clock clock = (Clock) resolveAttribute(CLOCK, Clock.class);
        this.clock = clock == null ? SystemClock.getInstance() : clock;
        this.callerFinder = StackBasedCallerFinder.getInstance();
    }

    private static <T> T callStaticMethod(String str, String str2, Class<T> cls) {
        try {
            return cls.cast(Class.forName(str).getMethod(str2, new Class[0]).invoke(null, new Object[0]));
        } catch (ClassCastException e2) {
            error("cannot cast result of calling '%s#%s' to '%s': %s\n", str, str2, cls.getName(), e2);
            return null;
        } catch (ClassNotFoundException unused) {
            return null;
        } catch (Exception e3) {
            error("cannot call expected no-argument static method '%s#%s': %s\n", str, str2, e3);
            return null;
        }
    }

    private static void error(String str, Object... objArr) {
        PrintStream printStream = System.err;
        printStream.println(DefaultPlatform.class + ": " + String.format(str, objArr));
    }

    private static String readProperty(String str) {
        Checks.checkNotNull(str, "attribute name");
        String str2 = "flogger." + str;
        try {
            return System.getProperty(str2);
        } catch (SecurityException e2) {
            error("cannot read property name %s: %s", str2, e2);
            return null;
        }
    }

    private static <T> T resolveAttribute(String str, Class<T> cls) {
        String readProperty = readProperty(str);
        if (readProperty == null) {
            return null;
        }
        int indexOf = readProperty.indexOf(35);
        if (indexOf > 0 && indexOf != readProperty.length() - 1) {
            return (T) callStaticMethod(readProperty.substring(0, indexOf), readProperty.substring(indexOf + 1), cls);
        }
        error("invalid getter (expected <class>#<method>): %s\n", readProperty);
        return null;
    }

    @Override // com.google.common.flogger.backend.Platform
    public LoggerBackend getBackendImpl(String str) {
        return this.backendFactory.create(str);
    }

    @Override // com.google.common.flogger.backend.Platform
    public Platform.LogCallerFinder getCallerFinderImpl() {
        return this.callerFinder;
    }

    @Override // com.google.common.flogger.backend.Platform
    public String getConfigInfoImpl() {
        StringBuilder x = a.x("Platform: ");
        x.append(getClass().getName());
        x.append("\nBackendFactory: ");
        x.append(this.backendFactory);
        x.append("\nClock: ");
        x.append(this.clock);
        x.append("\nLoggingContext: ");
        x.append(this.context);
        x.append("\nLogCallerFinder: ");
        x.append(this.callerFinder);
        x.append("\n");
        return x.toString();
    }

    @Override // com.google.common.flogger.backend.Platform
    public long getCurrentTimeNanosImpl() {
        return this.clock.getCurrentTimeNanos();
    }

    @Override // com.google.common.flogger.backend.Platform
    public Tags getInjectedTagsImpl() {
        return this.context.getTags();
    }

    @Override // com.google.common.flogger.backend.Platform
    public boolean shouldForceLoggingImpl(String str, Level level, boolean z) {
        return this.context.shouldForceLogging(str, level, z);
    }

    public DefaultPlatform(BackendFactory backendFactory, LoggingContext loggingContext, Clock clock, Platform.LogCallerFinder logCallerFinder) {
        this.backendFactory = backendFactory;
        this.context = loggingContext;
        this.clock = clock;
        this.callerFinder = logCallerFinder;
    }
}