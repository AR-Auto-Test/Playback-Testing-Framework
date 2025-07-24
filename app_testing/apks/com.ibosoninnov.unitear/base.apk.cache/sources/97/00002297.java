package com.google.common.flogger;

import com.google.common.flogger.AbstractLogger;
import com.google.common.flogger.LogSiteStats;
import com.google.common.flogger.LoggingApi;
import com.google.common.flogger.backend.LogData;
import com.google.common.flogger.backend.Metadata;
import com.google.common.flogger.backend.Platform;
import com.google.common.flogger.backend.Tags;
import com.google.common.flogger.backend.TemplateContext;
import com.google.common.flogger.parser.MessageParser;
import com.google.common.flogger.util.CallerFinder;
import com.google.common.flogger.util.Checks;
import com.google.errorprone.annotations.CheckReturnValue;
import com.google.firebase.analytics.FirebaseAnalytics;
import java.util.Arrays;
import java.util.concurrent.TimeUnit;
import java.util.logging.Level;

@CheckReturnValue
/* loaded from: classes.dex */
public abstract class LogContext<LOGGER extends AbstractLogger<API>, API extends LoggingApi<API>> implements LoggingApi<API>, LogData {
    private static final String LITERAL_VALUE_MESSAGE = new String();
    private Object[] args;
    private final Level level;
    private LogSite logSite;
    private MutableMetadata metadata;
    private TemplateContext templateContext;
    private final long timestampNanos;

    /* loaded from: classes.dex */
    public static final class Key {
        public static final MetadataKey<Throwable> LOG_CAUSE = MetadataKey.single("cause", Throwable.class);
        public static final MetadataKey<Integer> LOG_EVERY_N = MetadataKey.single("ratelimit_count", Integer.class);
        public static final MetadataKey<LogSiteStats.RateLimitPeriod> LOG_AT_MOST_EVERY = MetadataKey.single("ratelimit_period", LogSiteStats.RateLimitPeriod.class);
        public static final MetadataKey<Boolean> WAS_FORCED = MetadataKey.single("forced", Boolean.class);
        public static final MetadataKey<Tags> TAGS = MetadataKey.single("tags", Tags.class);
        public static final MetadataKey<StackSize> CONTEXT_STACK_SIZE = MetadataKey.single("stack_size", StackSize.class);

        private Key() {
        }
    }

    /* loaded from: classes.dex */
    public static final class MutableMetadata extends Metadata {
        private static final int INITIAL_KEY_VALUE_CAPACITY = 4;
        private Object[] keyValuePairs = new Object[8];
        private int keyValueCount = 0;

        private int indexOf(MetadataKey<?> metadataKey) {
            for (int i = 0; i < this.keyValueCount; i++) {
                if (this.keyValuePairs[i * 2].equals(metadataKey)) {
                    return i;
                }
            }
            return -1;
        }

        public <T> void addValue(MetadataKey<T> metadataKey, T t) {
            int indexOf;
            if (!metadataKey.canRepeat() && (indexOf = indexOf(metadataKey)) != -1) {
                this.keyValuePairs[(indexOf * 2) + 1] = Checks.checkNotNull(t, "metadata value");
                return;
            }
            int i = (this.keyValueCount + 1) * 2;
            Object[] objArr = this.keyValuePairs;
            if (i > objArr.length) {
                this.keyValuePairs = Arrays.copyOf(objArr, objArr.length * 2);
            }
            this.keyValuePairs[this.keyValueCount * 2] = Checks.checkNotNull(metadataKey, "metadata key");
            this.keyValuePairs[(this.keyValueCount * 2) + 1] = Checks.checkNotNull(t, "metadata value");
            this.keyValueCount++;
        }

        @Override // com.google.common.flogger.backend.Metadata
        public <T> T findValue(MetadataKey<T> metadataKey) {
            int indexOf = indexOf(metadataKey);
            if (indexOf != -1) {
                return metadataKey.cast(this.keyValuePairs[(indexOf * 2) + 1]);
            }
            return null;
        }

        @Override // com.google.common.flogger.backend.Metadata
        public MetadataKey<?> getKey(int i) {
            if (i < this.keyValueCount) {
                return (MetadataKey) this.keyValuePairs[i * 2];
            }
            throw new IndexOutOfBoundsException();
        }

        @Override // com.google.common.flogger.backend.Metadata
        public Object getValue(int i) {
            if (i < this.keyValueCount) {
                return this.keyValuePairs[(i * 2) + 1];
            }
            throw new IndexOutOfBoundsException();
        }

        public void removeAllValues(MetadataKey<?> metadataKey) {
            int i;
            int indexOf = indexOf(metadataKey);
            if (indexOf >= 0) {
                int i2 = indexOf * 2;
                int i3 = i2 + 2;
                while (true) {
                    i = this.keyValueCount;
                    if (i3 >= i * 2) {
                        break;
                    }
                    Object obj = this.keyValuePairs[i3];
                    if (!obj.equals(metadataKey)) {
                        Object[] objArr = this.keyValuePairs;
                        objArr[i2] = obj;
                        objArr[i2 + 1] = objArr[i3 + 1];
                        i2 += 2;
                    }
                    i3 += 2;
                }
                this.keyValueCount = i - ((i3 - i2) >> 1);
                while (i2 < i3) {
                    this.keyValuePairs[i2] = null;
                    i2++;
                }
            }
        }

        @Override // com.google.common.flogger.backend.Metadata
        public int size() {
            return this.keyValueCount;
        }

        public String toString() {
            StringBuilder sb = new StringBuilder("Metadata{");
            for (int i = 0; i < size(); i++) {
                sb.append(" '");
                sb.append(getKey(i));
                sb.append("': ");
                sb.append(getValue(i));
            }
            sb.append(" }");
            return sb.toString();
        }
    }

    public LogContext(Level level, boolean z) {
        this(level, z, Platform.getCurrentTimeNanos());
    }

    private void logImpl(String str, Object... objArr) {
        this.args = objArr;
        for (int i = 0; i < objArr.length; i++) {
            if (objArr[i] instanceof LazyArg) {
                objArr[i] = ((LazyArg) objArr[i]).evaluate();
            }
        }
        if (str != LITERAL_VALUE_MESSAGE) {
            this.templateContext = new TemplateContext(getMessageParser(), str);
        }
        getLogger().write(this);
    }

    /* JADX DEBUG: Type inference failed for r2v2. Raw type applied. Possible types: com.google.common.flogger.MetadataKey<com.google.common.flogger.backend.Tags>, com.google.common.flogger.MetadataKey<T> */
    private boolean shouldLog() {
        if (this.logSite == null) {
            this.logSite = (LogSite) Checks.checkNotNull(Platform.getCallerFinder().findLogSite(LogContext.class, 1), "logger backend must not return a null LogSite");
        }
        LogSite logSite = this.logSite;
        if (postProcess(logSite != LogSite.INVALID ? logSite : null)) {
            Tags injectedTags = Platform.getInjectedTags();
            if (!injectedTags.isEmpty()) {
                addMetadata(Key.TAGS, injectedTags);
            }
            return true;
        }
        return false;
    }

    public final <T> void addMetadata(MetadataKey<T> metadataKey, T t) {
        if (this.metadata == null) {
            this.metadata = new MutableMetadata();
        }
        this.metadata.addValue(metadataKey, t);
    }

    public abstract API api();

    /* JADX DEBUG: Type inference failed for r0v1. Raw type applied. Possible types: com.google.common.flogger.MetadataKey<com.google.common.flogger.LogSiteStats$RateLimitPeriod>, com.google.common.flogger.MetadataKey<T> */
    @Override // com.google.common.flogger.LoggingApi
    public final API atMostEvery(int i, TimeUnit timeUnit) {
        if (wasForced()) {
            return api();
        }
        if (i >= 0) {
            if (i > 0) {
                addMetadata(Key.LOG_AT_MOST_EVERY, LogSiteStats.newRateLimitPeriod(i, timeUnit));
            }
            return api();
        }
        throw new IllegalArgumentException("rate limit period cannot be negative");
    }

    /* JADX DEBUG: Type inference failed for r0v3. Raw type applied. Possible types: com.google.common.flogger.MetadataKey<java.lang.Integer>, com.google.common.flogger.MetadataKey<T> */
    @Override // com.google.common.flogger.LoggingApi
    public final API every(int i) {
        if (wasForced()) {
            return api();
        }
        if (i > 0) {
            if (i > 1) {
                addMetadata(Key.LOG_EVERY_N, Integer.valueOf(i));
            }
            return api();
        }
        throw new IllegalArgumentException("rate limit count must be positive");
    }

    @Override // com.google.common.flogger.backend.LogData
    public final Object[] getArguments() {
        if (this.templateContext != null) {
            return this.args;
        }
        throw new IllegalStateException("cannot get arguments unless a template context exists");
    }

    @Override // com.google.common.flogger.backend.LogData
    public final Level getLevel() {
        return this.level;
    }

    @Override // com.google.common.flogger.backend.LogData
    public final Object getLiteralArgument() {
        if (this.templateContext == null) {
            return this.args[0];
        }
        throw new IllegalStateException("cannot get literal argument if a template context exists");
    }

    @Override // com.google.common.flogger.backend.LogData
    public final LogSite getLogSite() {
        LogSite logSite = this.logSite;
        if (logSite != null) {
            return logSite;
        }
        throw new IllegalStateException("cannot request log site information prior to postProcess()");
    }

    public abstract LOGGER getLogger();

    @Override // com.google.common.flogger.backend.LogData
    public final String getLoggerName() {
        return getLogger().getBackend().getLoggerName();
    }

    public abstract MessageParser getMessageParser();

    @Override // com.google.common.flogger.backend.LogData
    public final Metadata getMetadata() {
        MutableMetadata mutableMetadata = this.metadata;
        return mutableMetadata != null ? mutableMetadata : Metadata.empty();
    }

    @Override // com.google.common.flogger.backend.LogData
    public final TemplateContext getTemplateContext() {
        return this.templateContext;
    }

    @Override // com.google.common.flogger.backend.LogData
    @Deprecated
    public final long getTimestampMicros() {
        return TimeUnit.NANOSECONDS.toMicros(this.timestampNanos);
    }

    @Override // com.google.common.flogger.backend.LogData
    public final long getTimestampNanos() {
        return this.timestampNanos;
    }

    @Override // com.google.common.flogger.LoggingApi
    public final boolean isEnabled() {
        return wasForced() || getLogger().isLoggable(this.level);
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log() {
        if (shouldLog()) {
            logImpl(LITERAL_VALUE_MESSAGE, "");
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void logVarargs(String str, Object[] objArr) {
        if (shouldLog()) {
            logImpl(str, Arrays.copyOf(objArr, objArr.length));
        }
    }

    public abstract API noOp();

    /* JADX DEBUG: Type inference failed for r2v0. Raw type applied. Possible types: com.google.common.flogger.MetadataKey<java.lang.Throwable>, com.google.common.flogger.MetadataKey<T> */
    public boolean postProcess(LogSiteKey logSiteKey) {
        MutableMetadata mutableMetadata = this.metadata;
        if (mutableMetadata != null && logSiteKey != null) {
            Integer num = (Integer) mutableMetadata.findValue(Key.LOG_EVERY_N);
            LogSiteStats.RateLimitPeriod rateLimitPeriod = (LogSiteStats.RateLimitPeriod) this.metadata.findValue(Key.LOG_AT_MOST_EVERY);
            LogSiteStats statsForKey = LogSiteStats.getStatsForKey(logSiteKey);
            if (num != null && !statsForKey.incrementAndCheckInvocationCount(num.intValue())) {
                return false;
            }
            if (rateLimitPeriod != null && !statsForKey.checkLastTimestamp(getTimestampNanos(), rateLimitPeriod)) {
                return false;
            }
        }
        Metadata metadata = getMetadata();
        MetadataKey<StackSize> metadataKey = Key.CONTEXT_STACK_SIZE;
        StackSize stackSize = (StackSize) metadata.findValue(metadataKey);
        if (stackSize != null) {
            removeMetadata(metadataKey);
            Metadata metadata2 = getMetadata();
            MetadataKey metadataKey2 = Key.LOG_CAUSE;
            addMetadata(metadataKey2, new LogSiteStackTrace((Throwable) metadata2.findValue(metadataKey2), stackSize, CallerFinder.getStackForCallerOf(LogContext.class, new Throwable(), stackSize.getMaxDepth())));
            return true;
        }
        return true;
    }

    public final void removeMetadata(MetadataKey<?> metadataKey) {
        MutableMetadata mutableMetadata = this.metadata;
        if (mutableMetadata != null) {
            mutableMetadata.removeAllValues(metadataKey);
        }
    }

    @Override // com.google.common.flogger.backend.LogData
    public final boolean wasForced() {
        MutableMetadata mutableMetadata = this.metadata;
        return mutableMetadata != null && Boolean.TRUE.equals(mutableMetadata.findValue(Key.WAS_FORCED));
    }

    /* JADX DEBUG: Type inference failed for r0v0. Raw type applied. Possible types: com.google.common.flogger.MetadataKey<java.lang.Throwable>, com.google.common.flogger.MetadataKey<T> */
    @Override // com.google.common.flogger.LoggingApi
    public final API withCause(Throwable th) {
        if (th != null) {
            addMetadata(Key.LOG_CAUSE, th);
        }
        return api();
    }

    @Override // com.google.common.flogger.LoggingApi
    public final API withInjectedLogSite(LogSite logSite) {
        if (this.logSite == null) {
            this.logSite = (LogSite) Checks.checkNotNull(logSite, "log site");
        }
        return api();
    }

    /* JADX DEBUG: Type inference failed for r0v2. Raw type applied. Possible types: com.google.common.flogger.MetadataKey<com.google.common.flogger.StackSize>, com.google.common.flogger.MetadataKey<T> */
    @Override // com.google.common.flogger.LoggingApi
    public API withStackTrace(StackSize stackSize) {
        if (Checks.checkNotNull(stackSize, "stack size") != StackSize.NONE) {
            addMetadata(Key.CONTEXT_STACK_SIZE, stackSize);
        }
        return api();
    }

    /* JADX DEBUG: Type inference failed for r2v3. Raw type applied. Possible types: com.google.common.flogger.MetadataKey<java.lang.Boolean>, com.google.common.flogger.MetadataKey<T> */
    public LogContext(Level level, boolean z, long j) {
        this.metadata = null;
        this.logSite = null;
        this.templateContext = null;
        this.args = null;
        this.level = (Level) Checks.checkNotNull(level, FirebaseAnalytics.Param.LEVEL);
        this.timestampNanos = j;
        if (z) {
            addMetadata(Key.WAS_FORCED, Boolean.TRUE);
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str) {
        if (shouldLog()) {
            logImpl(LITERAL_VALUE_MESSAGE, str);
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, Object obj) {
        if (shouldLog()) {
            logImpl(str, obj);
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, Object obj, Object obj2) {
        if (shouldLog()) {
            logImpl(str, obj, obj2);
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final API withInjectedLogSite(String str, String str2, int i, String str3) {
        return withInjectedLogSite(LogSite.injectedLogSite(str, str2, i, str3));
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, Object obj, Object obj2, Object obj3) {
        if (shouldLog()) {
            logImpl(str, obj, obj2, obj3);
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, Object obj, Object obj2, Object obj3, Object obj4) {
        if (shouldLog()) {
            logImpl(str, obj, obj2, obj3, obj4);
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, Object obj, Object obj2, Object obj3, Object obj4, Object obj5) {
        if (shouldLog()) {
            logImpl(str, obj, obj2, obj3, obj4, obj5);
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, Object obj, Object obj2, Object obj3, Object obj4, Object obj5, Object obj6) {
        if (shouldLog()) {
            logImpl(str, obj, obj2, obj3, obj4, obj5, obj6);
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, Object obj, Object obj2, Object obj3, Object obj4, Object obj5, Object obj6, Object obj7) {
        if (shouldLog()) {
            logImpl(str, obj, obj2, obj3, obj4, obj5, obj6, obj7);
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, Object obj, Object obj2, Object obj3, Object obj4, Object obj5, Object obj6, Object obj7, Object obj8) {
        if (shouldLog()) {
            logImpl(str, obj, obj2, obj3, obj4, obj5, obj6, obj7, obj8);
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, Object obj, Object obj2, Object obj3, Object obj4, Object obj5, Object obj6, Object obj7, Object obj8, Object obj9) {
        if (shouldLog()) {
            logImpl(str, obj, obj2, obj3, obj4, obj5, obj6, obj7, obj8, obj9);
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, Object obj, Object obj2, Object obj3, Object obj4, Object obj5, Object obj6, Object obj7, Object obj8, Object obj9, Object obj10) {
        if (shouldLog()) {
            logImpl(str, obj, obj2, obj3, obj4, obj5, obj6, obj7, obj8, obj9, obj10);
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, Object obj, Object obj2, Object obj3, Object obj4, Object obj5, Object obj6, Object obj7, Object obj8, Object obj9, Object obj10, Object... objArr) {
        if (shouldLog()) {
            Object[] objArr2 = new Object[objArr.length + 10];
            objArr2[0] = obj;
            objArr2[1] = obj2;
            objArr2[2] = obj3;
            objArr2[3] = obj4;
            objArr2[4] = obj5;
            objArr2[5] = obj6;
            objArr2[6] = obj7;
            objArr2[7] = obj8;
            objArr2[8] = obj9;
            objArr2[9] = obj10;
            System.arraycopy(objArr, 0, objArr2, 10, objArr.length);
            logImpl(str, objArr2);
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, char c2) {
        if (shouldLog()) {
            logImpl(str, Character.valueOf(c2));
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, byte b2) {
        if (shouldLog()) {
            logImpl(str, Byte.valueOf(b2));
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, short s) {
        if (shouldLog()) {
            logImpl(str, Short.valueOf(s));
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, int i) {
        if (shouldLog()) {
            logImpl(str, Integer.valueOf(i));
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, long j) {
        if (shouldLog()) {
            logImpl(str, Long.valueOf(j));
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, Object obj, boolean z) {
        if (shouldLog()) {
            logImpl(str, obj, Boolean.valueOf(z));
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, Object obj, char c2) {
        if (shouldLog()) {
            logImpl(str, obj, Character.valueOf(c2));
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, Object obj, byte b2) {
        if (shouldLog()) {
            logImpl(str, obj, Byte.valueOf(b2));
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, Object obj, short s) {
        if (shouldLog()) {
            logImpl(str, obj, Short.valueOf(s));
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, Object obj, int i) {
        if (shouldLog()) {
            logImpl(str, obj, Integer.valueOf(i));
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, Object obj, long j) {
        if (shouldLog()) {
            logImpl(str, obj, Long.valueOf(j));
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, Object obj, float f2) {
        if (shouldLog()) {
            logImpl(str, obj, Float.valueOf(f2));
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, Object obj, double d2) {
        if (shouldLog()) {
            logImpl(str, obj, Double.valueOf(d2));
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, boolean z, Object obj) {
        if (shouldLog()) {
            logImpl(str, Boolean.valueOf(z), obj);
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, char c2, Object obj) {
        if (shouldLog()) {
            logImpl(str, Character.valueOf(c2), obj);
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, byte b2, Object obj) {
        if (shouldLog()) {
            logImpl(str, Byte.valueOf(b2), obj);
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, short s, Object obj) {
        if (shouldLog()) {
            logImpl(str, Short.valueOf(s), obj);
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, int i, Object obj) {
        if (shouldLog()) {
            logImpl(str, Integer.valueOf(i), obj);
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, long j, Object obj) {
        if (shouldLog()) {
            logImpl(str, Long.valueOf(j), obj);
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, float f2, Object obj) {
        if (shouldLog()) {
            logImpl(str, Float.valueOf(f2), obj);
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, double d2, Object obj) {
        if (shouldLog()) {
            logImpl(str, Double.valueOf(d2), obj);
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, boolean z, boolean z2) {
        if (shouldLog()) {
            logImpl(str, Boolean.valueOf(z), Boolean.valueOf(z2));
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, char c2, boolean z) {
        if (shouldLog()) {
            logImpl(str, Character.valueOf(c2), Boolean.valueOf(z));
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, byte b2, boolean z) {
        if (shouldLog()) {
            logImpl(str, Byte.valueOf(b2), Boolean.valueOf(z));
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, short s, boolean z) {
        if (shouldLog()) {
            logImpl(str, Short.valueOf(s), Boolean.valueOf(z));
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, int i, boolean z) {
        if (shouldLog()) {
            logImpl(str, Integer.valueOf(i), Boolean.valueOf(z));
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, long j, boolean z) {
        if (shouldLog()) {
            logImpl(str, Long.valueOf(j), Boolean.valueOf(z));
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, float f2, boolean z) {
        if (shouldLog()) {
            logImpl(str, Float.valueOf(f2), Boolean.valueOf(z));
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, double d2, boolean z) {
        if (shouldLog()) {
            logImpl(str, Double.valueOf(d2), Boolean.valueOf(z));
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, boolean z, char c2) {
        if (shouldLog()) {
            logImpl(str, Boolean.valueOf(z), Character.valueOf(c2));
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, char c2, char c3) {
        if (shouldLog()) {
            logImpl(str, Character.valueOf(c2), Character.valueOf(c3));
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, byte b2, char c2) {
        if (shouldLog()) {
            logImpl(str, Byte.valueOf(b2), Character.valueOf(c2));
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, short s, char c2) {
        if (shouldLog()) {
            logImpl(str, Short.valueOf(s), Character.valueOf(c2));
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, int i, char c2) {
        if (shouldLog()) {
            logImpl(str, Integer.valueOf(i), Character.valueOf(c2));
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, long j, char c2) {
        if (shouldLog()) {
            logImpl(str, Long.valueOf(j), Character.valueOf(c2));
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, float f2, char c2) {
        if (shouldLog()) {
            logImpl(str, Float.valueOf(f2), Character.valueOf(c2));
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, double d2, char c2) {
        if (shouldLog()) {
            logImpl(str, Double.valueOf(d2), Character.valueOf(c2));
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, boolean z, byte b2) {
        if (shouldLog()) {
            logImpl(str, Boolean.valueOf(z), Byte.valueOf(b2));
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, char c2, byte b2) {
        if (shouldLog()) {
            logImpl(str, Character.valueOf(c2), Byte.valueOf(b2));
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, byte b2, byte b3) {
        if (shouldLog()) {
            logImpl(str, Byte.valueOf(b2), Byte.valueOf(b3));
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, short s, byte b2) {
        if (shouldLog()) {
            logImpl(str, Short.valueOf(s), Byte.valueOf(b2));
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, int i, byte b2) {
        if (shouldLog()) {
            logImpl(str, Integer.valueOf(i), Byte.valueOf(b2));
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, long j, byte b2) {
        if (shouldLog()) {
            logImpl(str, Long.valueOf(j), Byte.valueOf(b2));
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, float f2, byte b2) {
        if (shouldLog()) {
            logImpl(str, Float.valueOf(f2), Byte.valueOf(b2));
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, double d2, byte b2) {
        if (shouldLog()) {
            logImpl(str, Double.valueOf(d2), Byte.valueOf(b2));
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, boolean z, short s) {
        if (shouldLog()) {
            logImpl(str, Boolean.valueOf(z), Short.valueOf(s));
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, char c2, short s) {
        if (shouldLog()) {
            logImpl(str, Character.valueOf(c2), Short.valueOf(s));
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, byte b2, short s) {
        if (shouldLog()) {
            logImpl(str, Byte.valueOf(b2), Short.valueOf(s));
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, short s, short s2) {
        if (shouldLog()) {
            logImpl(str, Short.valueOf(s), Short.valueOf(s2));
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, int i, short s) {
        if (shouldLog()) {
            logImpl(str, Integer.valueOf(i), Short.valueOf(s));
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, long j, short s) {
        if (shouldLog()) {
            logImpl(str, Long.valueOf(j), Short.valueOf(s));
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, float f2, short s) {
        if (shouldLog()) {
            logImpl(str, Float.valueOf(f2), Short.valueOf(s));
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, double d2, short s) {
        if (shouldLog()) {
            logImpl(str, Double.valueOf(d2), Short.valueOf(s));
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, boolean z, int i) {
        if (shouldLog()) {
            logImpl(str, Boolean.valueOf(z), Integer.valueOf(i));
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, char c2, int i) {
        if (shouldLog()) {
            logImpl(str, Character.valueOf(c2), Integer.valueOf(i));
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, byte b2, int i) {
        if (shouldLog()) {
            logImpl(str, Byte.valueOf(b2), Integer.valueOf(i));
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, short s, int i) {
        if (shouldLog()) {
            logImpl(str, Short.valueOf(s), Integer.valueOf(i));
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, int i, int i2) {
        if (shouldLog()) {
            logImpl(str, Integer.valueOf(i), Integer.valueOf(i2));
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, long j, int i) {
        if (shouldLog()) {
            logImpl(str, Long.valueOf(j), Integer.valueOf(i));
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, float f2, int i) {
        if (shouldLog()) {
            logImpl(str, Float.valueOf(f2), Integer.valueOf(i));
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, double d2, int i) {
        if (shouldLog()) {
            logImpl(str, Double.valueOf(d2), Integer.valueOf(i));
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, boolean z, long j) {
        if (shouldLog()) {
            logImpl(str, Boolean.valueOf(z), Long.valueOf(j));
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, char c2, long j) {
        if (shouldLog()) {
            logImpl(str, Character.valueOf(c2), Long.valueOf(j));
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, byte b2, long j) {
        if (shouldLog()) {
            logImpl(str, Byte.valueOf(b2), Long.valueOf(j));
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, short s, long j) {
        if (shouldLog()) {
            logImpl(str, Short.valueOf(s), Long.valueOf(j));
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, int i, long j) {
        if (shouldLog()) {
            logImpl(str, Integer.valueOf(i), Long.valueOf(j));
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, long j, long j2) {
        if (shouldLog()) {
            logImpl(str, Long.valueOf(j), Long.valueOf(j2));
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, float f2, long j) {
        if (shouldLog()) {
            logImpl(str, Float.valueOf(f2), Long.valueOf(j));
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, double d2, long j) {
        if (shouldLog()) {
            logImpl(str, Double.valueOf(d2), Long.valueOf(j));
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, boolean z, float f2) {
        if (shouldLog()) {
            logImpl(str, Boolean.valueOf(z), Float.valueOf(f2));
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, char c2, float f2) {
        if (shouldLog()) {
            logImpl(str, Character.valueOf(c2), Float.valueOf(f2));
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, byte b2, float f2) {
        if (shouldLog()) {
            logImpl(str, Byte.valueOf(b2), Float.valueOf(f2));
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, short s, float f2) {
        if (shouldLog()) {
            logImpl(str, Short.valueOf(s), Float.valueOf(f2));
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, int i, float f2) {
        if (shouldLog()) {
            logImpl(str, Integer.valueOf(i), Float.valueOf(f2));
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, long j, float f2) {
        if (shouldLog()) {
            logImpl(str, Long.valueOf(j), Float.valueOf(f2));
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, float f2, float f3) {
        if (shouldLog()) {
            logImpl(str, Float.valueOf(f2), Float.valueOf(f3));
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, double d2, float f2) {
        if (shouldLog()) {
            logImpl(str, Double.valueOf(d2), Float.valueOf(f2));
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, boolean z, double d2) {
        if (shouldLog()) {
            logImpl(str, Boolean.valueOf(z), Double.valueOf(d2));
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, char c2, double d2) {
        if (shouldLog()) {
            logImpl(str, Character.valueOf(c2), Double.valueOf(d2));
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, byte b2, double d2) {
        if (shouldLog()) {
            logImpl(str, Byte.valueOf(b2), Double.valueOf(d2));
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, short s, double d2) {
        if (shouldLog()) {
            logImpl(str, Short.valueOf(s), Double.valueOf(d2));
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, int i, double d2) {
        if (shouldLog()) {
            logImpl(str, Integer.valueOf(i), Double.valueOf(d2));
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, long j, double d2) {
        if (shouldLog()) {
            logImpl(str, Long.valueOf(j), Double.valueOf(d2));
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, float f2, double d2) {
        if (shouldLog()) {
            logImpl(str, Float.valueOf(f2), Double.valueOf(d2));
        }
    }

    @Override // com.google.common.flogger.LoggingApi
    public final void log(String str, double d2, double d3) {
        if (shouldLog()) {
            logImpl(str, Double.valueOf(d2), Double.valueOf(d3));
        }
    }
}