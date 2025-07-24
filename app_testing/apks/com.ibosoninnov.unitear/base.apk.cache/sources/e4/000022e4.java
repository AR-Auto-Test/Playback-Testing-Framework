package com.google.common.flogger.util;

import c.b.a.a.a;
import com.google.errorprone.annotations.CheckReturnValue;

@CheckReturnValue
/* loaded from: classes.dex */
public final class CallerFinder {
    private static final FastStackGetter stackGetter = FastStackGetter.createIfSupported();

    public static StackTraceElement findCallerOf(Class<?> cls, Throwable th, int i) {
        StackTraceElement stackTraceElement;
        Checks.checkNotNull(cls, "target");
        Checks.checkNotNull(th, "throwable");
        if (i >= 0) {
            StackTraceElement[] stackTrace = stackGetter != null ? null : th.getStackTrace();
            boolean z = false;
            while (true) {
                try {
                    FastStackGetter fastStackGetter = stackGetter;
                    if (fastStackGetter != null) {
                        stackTraceElement = fastStackGetter.getStackTraceElement(th, i);
                    } else {
                        stackTraceElement = stackTrace[i];
                    }
                    if (cls.getName().equals(stackTraceElement.getClassName())) {
                        z = true;
                    } else if (z) {
                        return stackTraceElement;
                    }
                    i++;
                } catch (Exception unused) {
                    return null;
                }
            }
        } else {
            throw new IllegalArgumentException(a.j("skip count cannot be negative: ", i));
        }
    }

    public static StackTraceElement[] getStackForCallerOf(Class<?> cls, Throwable th, int i) {
        StackTraceElement[] stackTrace;
        int length;
        StackTraceElement stackTraceElement;
        Checks.checkNotNull(cls, "target");
        Checks.checkNotNull(th, "throwable");
        if (i <= 0 && i != -1) {
            throw new IllegalArgumentException(a.j("invalid maximum depth: ", i));
        }
        FastStackGetter fastStackGetter = stackGetter;
        if (fastStackGetter != null) {
            stackTrace = null;
            length = fastStackGetter.getStackTraceDepth(th);
        } else {
            stackTrace = th.getStackTrace();
            length = stackTrace.length;
        }
        boolean z = false;
        for (int i2 = 0; i2 < length; i2++) {
            FastStackGetter fastStackGetter2 = stackGetter;
            StackTraceElement stackTraceElement2 = fastStackGetter2 != null ? fastStackGetter2.getStackTraceElement(th, i2) : stackTrace[i2];
            if (cls.getName().equals(stackTraceElement2.getClassName())) {
                z = true;
            } else if (z) {
                int i3 = length - i2;
                if (i <= 0 || i >= i3) {
                    i = i3;
                }
                StackTraceElement[] stackTraceElementArr = new StackTraceElement[i];
                stackTraceElementArr[0] = stackTraceElement2;
                for (int i4 = 1; i4 < i; i4++) {
                    FastStackGetter fastStackGetter3 = stackGetter;
                    if (fastStackGetter3 != null) {
                        stackTraceElement = fastStackGetter3.getStackTraceElement(th, i2 + i4);
                    } else {
                        stackTraceElement = stackTrace[i2 + i4];
                    }
                    stackTraceElementArr[i4] = stackTraceElement;
                }
                return stackTraceElementArr;
            }
        }
        return new StackTraceElement[0];
    }
}