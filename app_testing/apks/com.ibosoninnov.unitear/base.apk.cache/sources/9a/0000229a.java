package com.google.common.flogger;

import c.b.a.a.a;
import com.google.common.flogger.util.Checks;
import com.google.errorprone.annotations.CheckReturnValue;

@CheckReturnValue
/* loaded from: classes.dex */
public abstract class LogSite implements LogSiteKey {
    public static final LogSite INVALID = new LogSite() { // from class: com.google.common.flogger.LogSite.1
        @Override // com.google.common.flogger.LogSite
        public String getClassName() {
            return "<unknown class>";
        }

        @Override // com.google.common.flogger.LogSite
        public String getFileName() {
            return null;
        }

        @Override // com.google.common.flogger.LogSite
        public int getLineNumber() {
            return 0;
        }

        @Override // com.google.common.flogger.LogSite
        public String getMethodName() {
            return "<unknown method>";
        }
    };
    public static final int UNKNOWN_LINE = 0;

    /* loaded from: classes.dex */
    public static final class InjectedLogSite extends LogSite {
        private final int encodedLineNumber;
        private int hashcode;
        private final String internalClassName;
        private final String methodName;
        private final String sourceFileName;

        public boolean equals(Object obj) {
            if (obj instanceof InjectedLogSite) {
                InjectedLogSite injectedLogSite = (InjectedLogSite) obj;
                return this.internalClassName.equals(injectedLogSite.internalClassName) && this.methodName.equals(injectedLogSite.methodName) && this.encodedLineNumber == injectedLogSite.encodedLineNumber;
            }
            return false;
        }

        @Override // com.google.common.flogger.LogSite
        public String getClassName() {
            return this.internalClassName.replace('/', '.');
        }

        @Override // com.google.common.flogger.LogSite
        public String getFileName() {
            return this.sourceFileName;
        }

        @Override // com.google.common.flogger.LogSite
        public int getLineNumber() {
            return this.encodedLineNumber & 65535;
        }

        @Override // com.google.common.flogger.LogSite
        public String getMethodName() {
            return this.methodName;
        }

        public int hashCode() {
            if (this.hashcode == 0) {
                this.hashcode = ((this.methodName.hashCode() + ((this.internalClassName.hashCode() + 4867) * 31)) * 31) + this.encodedLineNumber;
            }
            return this.hashcode;
        }

        private InjectedLogSite(String str, String str2, int i, String str3) {
            this.hashcode = 0;
            this.internalClassName = (String) Checks.checkNotNull(str, "class name");
            this.methodName = (String) Checks.checkNotNull(str2, "method name");
            this.encodedLineNumber = i;
            this.sourceFileName = str3;
        }
    }

    @Deprecated
    public static LogSite injectedLogSite(String str, String str2, int i, String str3) {
        return new InjectedLogSite(str, str2, i, str3);
    }

    public abstract String getClassName();

    public abstract String getFileName();

    public abstract int getLineNumber();

    public abstract String getMethodName();

    public final String toString() {
        StringBuilder x = a.x("LogSite{ class=");
        x.append(getClassName());
        x.append(", method=");
        x.append(getMethodName());
        x.append(", line=");
        x.append(getLineNumber());
        if (getFileName() != null) {
            x.append(", file=");
            x.append(getFileName());
        }
        x.append(" }");
        return x.toString();
    }
}