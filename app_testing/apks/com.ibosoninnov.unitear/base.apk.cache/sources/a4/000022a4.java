package com.google.common.flogger;

import com.google.common.flogger.LoggingApi;
import com.google.common.flogger.util.Checks;
import com.google.errorprone.annotations.CheckReturnValue;
import java.util.concurrent.TimeUnit;

@CheckReturnValue
/* loaded from: classes.dex */
public interface LoggingApi<API extends LoggingApi<API>> {
    API atMostEvery(int i, TimeUnit timeUnit);

    API every(int i);

    boolean isEnabled();

    void log();

    void log(String str);

    void log(String str, byte b2);

    void log(String str, byte b2, byte b3);

    void log(String str, byte b2, char c2);

    void log(String str, byte b2, double d2);

    void log(String str, byte b2, float f2);

    void log(String str, byte b2, int i);

    void log(String str, byte b2, long j);

    void log(String str, byte b2, Object obj);

    void log(String str, byte b2, short s);

    void log(String str, byte b2, boolean z);

    void log(String str, char c2);

    void log(String str, char c2, byte b2);

    void log(String str, char c2, char c3);

    void log(String str, char c2, double d2);

    void log(String str, char c2, float f2);

    void log(String str, char c2, int i);

    void log(String str, char c2, long j);

    void log(String str, char c2, Object obj);

    void log(String str, char c2, short s);

    void log(String str, char c2, boolean z);

    void log(String str, double d2, byte b2);

    void log(String str, double d2, char c2);

    void log(String str, double d2, double d3);

    void log(String str, double d2, float f2);

    void log(String str, double d2, int i);

    void log(String str, double d2, long j);

    void log(String str, double d2, Object obj);

    void log(String str, double d2, short s);

    void log(String str, double d2, boolean z);

    void log(String str, float f2, byte b2);

    void log(String str, float f2, char c2);

    void log(String str, float f2, double d2);

    void log(String str, float f2, float f3);

    void log(String str, float f2, int i);

    void log(String str, float f2, long j);

    void log(String str, float f2, Object obj);

    void log(String str, float f2, short s);

    void log(String str, float f2, boolean z);

    void log(String str, int i);

    void log(String str, int i, byte b2);

    void log(String str, int i, char c2);

    void log(String str, int i, double d2);

    void log(String str, int i, float f2);

    void log(String str, int i, int i2);

    void log(String str, int i, long j);

    void log(String str, int i, Object obj);

    void log(String str, int i, short s);

    void log(String str, int i, boolean z);

    void log(String str, long j);

    void log(String str, long j, byte b2);

    void log(String str, long j, char c2);

    void log(String str, long j, double d2);

    void log(String str, long j, float f2);

    void log(String str, long j, int i);

    void log(String str, long j, long j2);

    void log(String str, long j, Object obj);

    void log(String str, long j, short s);

    void log(String str, long j, boolean z);

    void log(String str, Object obj);

    void log(String str, Object obj, byte b2);

    void log(String str, Object obj, char c2);

    void log(String str, Object obj, double d2);

    void log(String str, Object obj, float f2);

    void log(String str, Object obj, int i);

    void log(String str, Object obj, long j);

    void log(String str, Object obj, Object obj2);

    void log(String str, Object obj, Object obj2, Object obj3);

    void log(String str, Object obj, Object obj2, Object obj3, Object obj4);

    void log(String str, Object obj, Object obj2, Object obj3, Object obj4, Object obj5);

    void log(String str, Object obj, Object obj2, Object obj3, Object obj4, Object obj5, Object obj6);

    void log(String str, Object obj, Object obj2, Object obj3, Object obj4, Object obj5, Object obj6, Object obj7);

    void log(String str, Object obj, Object obj2, Object obj3, Object obj4, Object obj5, Object obj6, Object obj7, Object obj8);

    void log(String str, Object obj, Object obj2, Object obj3, Object obj4, Object obj5, Object obj6, Object obj7, Object obj8, Object obj9);

    void log(String str, Object obj, Object obj2, Object obj3, Object obj4, Object obj5, Object obj6, Object obj7, Object obj8, Object obj9, Object obj10);

    void log(String str, Object obj, Object obj2, Object obj3, Object obj4, Object obj5, Object obj6, Object obj7, Object obj8, Object obj9, Object obj10, Object... objArr);

    void log(String str, Object obj, short s);

    void log(String str, Object obj, boolean z);

    void log(String str, short s);

    void log(String str, short s, byte b2);

    void log(String str, short s, char c2);

    void log(String str, short s, double d2);

    void log(String str, short s, float f2);

    void log(String str, short s, int i);

    void log(String str, short s, long j);

    void log(String str, short s, Object obj);

    void log(String str, short s, short s2);

    void log(String str, short s, boolean z);

    void log(String str, boolean z, byte b2);

    void log(String str, boolean z, char c2);

    void log(String str, boolean z, double d2);

    void log(String str, boolean z, float f2);

    void log(String str, boolean z, int i);

    void log(String str, boolean z, long j);

    void log(String str, boolean z, Object obj);

    void log(String str, boolean z, short s);

    void log(String str, boolean z, boolean z2);

    void logVarargs(String str, Object[] objArr);

    API withCause(Throwable th);

    API withInjectedLogSite(LogSite logSite);

    API withInjectedLogSite(String str, String str2, int i, String str3);

    API withStackTrace(StackSize stackSize);

    /* loaded from: classes.dex */
    public static class NoOp<API extends LoggingApi<API>> implements LoggingApi<API> {
        @Override // com.google.common.flogger.LoggingApi
        public final API atMostEvery(int i, TimeUnit timeUnit) {
            Checks.checkNotNull(timeUnit, "time unit");
            return noOp();
        }

        @Override // com.google.common.flogger.LoggingApi
        public final API every(int i) {
            return noOp();
        }

        @Override // com.google.common.flogger.LoggingApi
        public final boolean isEnabled() {
            return false;
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log() {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, byte b2) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, byte b2, byte b3) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, byte b2, char c2) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, byte b2, double d2) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, byte b2, float f2) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, byte b2, int i) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, byte b2, long j) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, byte b2, Object obj) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, byte b2, short s) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, byte b2, boolean z) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, char c2) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, char c2, byte b2) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, char c2, char c3) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, char c2, double d2) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, char c2, float f2) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, char c2, int i) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, char c2, long j) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, char c2, Object obj) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, char c2, short s) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, char c2, boolean z) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, double d2, byte b2) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, double d2, char c2) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, double d2, double d3) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, double d2, float f2) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, double d2, int i) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, double d2, long j) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, double d2, Object obj) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, double d2, short s) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, double d2, boolean z) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, float f2, byte b2) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, float f2, char c2) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, float f2, double d2) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, float f2, float f3) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, float f2, int i) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, float f2, long j) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, float f2, Object obj) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, float f2, short s) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, float f2, boolean z) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, int i) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, int i, byte b2) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, int i, char c2) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, int i, double d2) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, int i, float f2) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, int i, int i2) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, int i, long j) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, int i, Object obj) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, int i, short s) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, int i, boolean z) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, long j) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, long j, byte b2) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, long j, char c2) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, long j, double d2) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, long j, float f2) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, long j, int i) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, long j, long j2) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, long j, Object obj) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, long j, short s) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, long j, boolean z) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, Object obj) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, Object obj, byte b2) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, Object obj, char c2) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, Object obj, double d2) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, Object obj, float f2) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, Object obj, int i) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, Object obj, long j) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, Object obj, Object obj2) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, Object obj, Object obj2, Object obj3) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, Object obj, Object obj2, Object obj3, Object obj4) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, Object obj, Object obj2, Object obj3, Object obj4, Object obj5) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, Object obj, Object obj2, Object obj3, Object obj4, Object obj5, Object obj6) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, Object obj, Object obj2, Object obj3, Object obj4, Object obj5, Object obj6, Object obj7) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, Object obj, Object obj2, Object obj3, Object obj4, Object obj5, Object obj6, Object obj7, Object obj8) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, Object obj, Object obj2, Object obj3, Object obj4, Object obj5, Object obj6, Object obj7, Object obj8, Object obj9) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, Object obj, Object obj2, Object obj3, Object obj4, Object obj5, Object obj6, Object obj7, Object obj8, Object obj9, Object obj10) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, Object obj, Object obj2, Object obj3, Object obj4, Object obj5, Object obj6, Object obj7, Object obj8, Object obj9, Object obj10, Object... objArr) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, Object obj, short s) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, Object obj, boolean z) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, short s) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, short s, byte b2) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, short s, char c2) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, short s, double d2) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, short s, float f2) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, short s, int i) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, short s, long j) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, short s, Object obj) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, short s, short s2) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, short s, boolean z) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, boolean z, byte b2) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, boolean z, char c2) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, boolean z, double d2) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, boolean z, float f2) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, boolean z, int i) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, boolean z, long j) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, boolean z, Object obj) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, boolean z, short s) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void log(String str, boolean z, boolean z2) {
        }

        @Override // com.google.common.flogger.LoggingApi
        public final void logVarargs(String str, Object[] objArr) {
        }

        public final API noOp() {
            return this;
        }

        @Override // com.google.common.flogger.LoggingApi
        public final API withCause(Throwable th) {
            return noOp();
        }

        @Override // com.google.common.flogger.LoggingApi
        public API withInjectedLogSite(LogSite logSite) {
            Checks.checkNotNull(logSite, "log site");
            return noOp();
        }

        @Override // com.google.common.flogger.LoggingApi
        public API withStackTrace(StackSize stackSize) {
            Checks.checkNotNull(stackSize, "stack size");
            return noOp();
        }

        @Override // com.google.common.flogger.LoggingApi
        public API withInjectedLogSite(String str, String str2, int i, String str3) {
            return noOp();
        }
    }
}