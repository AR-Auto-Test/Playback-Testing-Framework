package com.google.common.flogger.util;

import com.google.common.flogger.LogSite;
import com.google.errorprone.annotations.CheckReturnValue;

@CheckReturnValue
/* loaded from: classes.dex */
public final class StackBasedLogSite extends LogSite {
    private final StackTraceElement stackElement;

    public StackBasedLogSite(StackTraceElement stackTraceElement) {
        this.stackElement = (StackTraceElement) Checks.checkNotNull(stackTraceElement, "stack element");
    }

    public boolean equals(Object obj) {
        return (obj instanceof StackBasedLogSite) && this.stackElement.equals(((StackBasedLogSite) obj).stackElement);
    }

    @Override // com.google.common.flogger.LogSite
    public String getClassName() {
        return this.stackElement.getClassName();
    }

    @Override // com.google.common.flogger.LogSite
    public String getFileName() {
        return this.stackElement.getFileName();
    }

    @Override // com.google.common.flogger.LogSite
    public int getLineNumber() {
        return Math.max(this.stackElement.getLineNumber(), 0);
    }

    @Override // com.google.common.flogger.LogSite
    public String getMethodName() {
        return this.stackElement.getMethodName();
    }

    public int hashCode() {
        return this.stackElement.hashCode();
    }
}