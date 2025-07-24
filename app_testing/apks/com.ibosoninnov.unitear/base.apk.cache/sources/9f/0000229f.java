package com.google.common.flogger;

import c.b.a.a.a;
import com.google.common.flogger.util.Checks;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/* loaded from: classes.dex */
public final class LogSiteStats {
    private static final StatsMap map = new StatsMap();
    private final AtomicLong invocationCount = new AtomicLong();
    private final AtomicLong lastTimestampNanos = new AtomicLong();
    private final AtomicInteger skippedLogStatements = new AtomicInteger();

    /* loaded from: classes.dex */
    public static final class RateLimitPeriod {
        private final int n;
        private int skipCount;
        private final TimeUnit unit;

        /* JADX INFO: Access modifiers changed from: private */
        public void setSkipCount(int i) {
            this.skipCount = i;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public long toNanos() {
            return this.unit.toNanos(this.n);
        }

        public boolean equals(Object obj) {
            if (obj instanceof RateLimitPeriod) {
                RateLimitPeriod rateLimitPeriod = (RateLimitPeriod) obj;
                return this.n == rateLimitPeriod.n && this.unit == rateLimitPeriod.unit;
            }
            return false;
        }

        public int hashCode() {
            return (this.n * 37) ^ this.unit.hashCode();
        }

        public String toString() {
            StringBuilder sb = new StringBuilder();
            sb.append(this.n);
            sb.append(' ');
            sb.append(this.unit);
            if (this.skipCount > 0) {
                sb.append(" [skipped: ");
                sb.append(this.skipCount);
                sb.append(']');
            }
            return sb.toString();
        }

        private RateLimitPeriod(int i, TimeUnit timeUnit) {
            this.skipCount = -1;
            if (i > 0) {
                this.n = i;
                this.unit = (TimeUnit) Checks.checkNotNull(timeUnit, "time unit");
                return;
            }
            throw new IllegalArgumentException(a.j("time period must be positive: ", i));
        }
    }

    /* loaded from: classes.dex */
    public static final class StatsMap {
        private final ConcurrentMap<Object, LogSiteStats> statsMap = new ConcurrentHashMap();

        public LogSiteStats getStatsForKey(Object obj) {
            LogSiteStats logSiteStats = this.statsMap.get(obj);
            if (logSiteStats == null) {
                LogSiteStats logSiteStats2 = new LogSiteStats();
                LogSiteStats putIfAbsent = this.statsMap.putIfAbsent(obj, logSiteStats2);
                return putIfAbsent != null ? putIfAbsent : logSiteStats2;
            }
            return logSiteStats;
        }
    }

    public static LogSiteStats getStatsForKey(Object obj) {
        return map.getStatsForKey(obj);
    }

    public static RateLimitPeriod newRateLimitPeriod(int i, TimeUnit timeUnit) {
        return new RateLimitPeriod(i, timeUnit);
    }

    public boolean checkLastTimestamp(long j, RateLimitPeriod rateLimitPeriod) {
        long j2 = this.lastTimestampNanos.get();
        long nanos = rateLimitPeriod.toNanos() + j2;
        if (nanos >= 0 && ((j >= nanos || j2 == 0) && this.lastTimestampNanos.compareAndSet(j2, j))) {
            rateLimitPeriod.setSkipCount(this.skippedLogStatements.getAndSet(0));
            return true;
        }
        this.skippedLogStatements.incrementAndGet();
        return false;
    }

    public boolean incrementAndCheckInvocationCount(int i) {
        return this.invocationCount.getAndIncrement() % ((long) i) == 0;
    }
}