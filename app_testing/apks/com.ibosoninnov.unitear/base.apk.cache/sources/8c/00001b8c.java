package com.google.ar.core;

import android.app.Activity;
import android.content.Context;
import com.google.ar.core.exceptions.FatalException;

/* loaded from: classes.dex */
public class ArCoreApk {

    /* JADX WARN: Failed to restore enum class, 'enum' modifier and super class removed */
    /* JADX WARN: Unknown enum class pattern. Please report as an issue! */
    /* loaded from: classes.dex */
    public static class Availability {
        public final int nativeCode;
        public static final Availability UNKNOWN_ERROR = new a();
        public static final Availability UNKNOWN_CHECKING = new b();
        public static final Availability UNKNOWN_TIMED_OUT = new c();
        public static final Availability UNSUPPORTED_DEVICE_NOT_CAPABLE = new d();
        public static final Availability SUPPORTED_NOT_INSTALLED = new e();
        public static final Availability SUPPORTED_APK_TOO_OLD = new f();
        public static final Availability SUPPORTED_INSTALLED = new g();
        private static final /* synthetic */ Availability[] $VALUES = $values();

        private static /* synthetic */ Availability[] $values() {
            return new Availability[]{UNKNOWN_ERROR, UNKNOWN_CHECKING, UNKNOWN_TIMED_OUT, UNSUPPORTED_DEVICE_NOT_CAPABLE, SUPPORTED_NOT_INSTALLED, SUPPORTED_APK_TOO_OLD, SUPPORTED_INSTALLED};
        }

        private Availability(String str, int i, int i2) {
            this.nativeCode = i2;
        }

        public static Availability forNumber(int i) {
            Availability[] values;
            for (Availability availability : values()) {
                if (availability.nativeCode == i) {
                    return availability;
                }
            }
            throw new FatalException(c.b.a.a.a.g(59, "Unexpected value for native Availability, value=", i));
        }

        public static Availability valueOf(String str) {
            return (Availability) Enum.valueOf(Availability.class, str);
        }

        public static Availability[] values() {
            return (Availability[]) $VALUES.clone();
        }

        public boolean isSupported() {
            return false;
        }

        public boolean isTransient() {
            return false;
        }

        public boolean isUnknown() {
            return false;
        }

        public boolean isUnsupported() {
            return false;
        }
    }

    /* loaded from: classes.dex */
    public enum InstallBehavior {
        REQUIRED(0),
        OPTIONAL(1);
        
        public final int nativeCode;

        InstallBehavior(int i) {
            this.nativeCode = i;
        }

        public static InstallBehavior forNumber(int i) {
            InstallBehavior[] values = values();
            for (int i2 = 0; i2 < 2; i2++) {
                InstallBehavior installBehavior = values[i2];
                if (installBehavior.nativeCode == i) {
                    return installBehavior;
                }
            }
            throw new FatalException(c.b.a.a.a.g(62, "Unexpected value for native InstallBehavior, value=", i));
        }
    }

    /* loaded from: classes.dex */
    public enum InstallStatus {
        INSTALLED(0),
        INSTALL_REQUESTED(1);
        
        public final int nativeCode;

        InstallStatus(int i) {
            this.nativeCode = i;
        }

        public static InstallStatus forNumber(int i) {
            InstallStatus[] values = values();
            for (int i2 = 0; i2 < 2; i2++) {
                InstallStatus installStatus = values[i2];
                if (installStatus.nativeCode == i) {
                    return installStatus;
                }
            }
            throw new FatalException(c.b.a.a.a.g(60, "Unexpected value for native InstallStatus, value=", i));
        }
    }

    /* loaded from: classes.dex */
    public enum UserMessageType {
        APPLICATION(0),
        FEATURE(1),
        USER_ALREADY_INFORMED(2);
        
        public final int nativeCode;

        UserMessageType(int i) {
            this.nativeCode = i;
        }

        public static UserMessageType forNumber(int i) {
            UserMessageType[] values = values();
            for (int i2 = 0; i2 < 3; i2++) {
                UserMessageType userMessageType = values[i2];
                if (userMessageType.nativeCode == i) {
                    return userMessageType;
                }
            }
            throw new FatalException(c.b.a.a.a.g(62, "Unexpected value for native UserMessageType, value=", i));
        }
    }

    public static ArCoreApk getInstance() {
        return j.a();
    }

    public Availability checkAvailability(Context context) {
        throw new UnsupportedOperationException("Stub");
    }

    public InstallStatus requestInstall(Activity activity, boolean z) {
        throw new UnsupportedOperationException("Stub");
    }

    public InstallStatus requestInstall(Activity activity, boolean z, InstallBehavior installBehavior, UserMessageType userMessageType) {
        throw new UnsupportedOperationException("Stub");
    }
}