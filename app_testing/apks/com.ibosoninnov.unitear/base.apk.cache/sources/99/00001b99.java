package com.google.ar.core;

import android.app.Activity;
import android.content.Context;
import android.util.Log;
import com.google.ar.core.ArCoreApk;
import com.google.ar.core.annotations.UsedByNative;
import com.google.ar.core.exceptions.ResourceExhaustedException;
import com.google.ar.core.exceptions.UnavailableApkTooOldException;
import com.google.ar.core.exceptions.UnavailableArcoreNotInstalledException;
import com.google.ar.core.exceptions.UnavailableDeviceNotCompatibleException;
import com.google.ar.core.exceptions.UnavailableSdkTooOldException;
import com.google.ar.core.exceptions.UnavailableUserDeclinedInstallationException;
import java.util.HashMap;
import java.util.Map;

@UsedByNative("arcoreapk.cc")
/* loaded from: classes.dex */
public final class ArCoreApkJniAdapter {

    /* renamed from: a  reason: collision with root package name */
    private static final Map<Class<? extends Throwable>, Integer> f5538a;

    static {
        HashMap hashMap = new HashMap();
        f5538a = hashMap;
        hashMap.put(IllegalArgumentException.class, Integer.valueOf(ae.ERROR_INVALID_ARGUMENT.E));
        hashMap.put(ResourceExhaustedException.class, Integer.valueOf(ae.ERROR_RESOURCE_EXHAUSTED.E));
        hashMap.put(UnavailableArcoreNotInstalledException.class, Integer.valueOf(ae.UNAVAILABLE_ARCORE_NOT_INSTALLED.E));
        hashMap.put(UnavailableDeviceNotCompatibleException.class, Integer.valueOf(ae.UNAVAILABLE_DEVICE_NOT_COMPATIBLE.E));
        hashMap.put(UnavailableApkTooOldException.class, Integer.valueOf(ae.UNAVAILABLE_APK_TOO_OLD.E));
        hashMap.put(UnavailableSdkTooOldException.class, Integer.valueOf(ae.UNAVAILABLE_SDK_TOO_OLD.E));
        hashMap.put(UnavailableUserDeclinedInstallationException.class, Integer.valueOf(ae.UNAVAILABLE_USER_DECLINED_INSTALLATION.E));
    }

    private ArCoreApkJniAdapter() {
    }

    private static int a(Throwable th) {
        Log.e("ARCore-ArCoreApkJniAdapter", "Exception details:", th);
        Class<?> cls = th.getClass();
        Map<Class<? extends Throwable>, Integer> map = f5538a;
        if (map.containsKey(cls)) {
            return map.get(cls).intValue();
        }
        return ae.ERROR_FATAL.E;
    }

    @UsedByNative("arcoreapk.cc")
    public static int checkAvailability(Context context) {
        try {
            return ArCoreApk.getInstance().checkAvailability(context).nativeCode;
        } catch (Throwable th) {
            a(th);
            return ArCoreApk.Availability.UNKNOWN_ERROR.nativeCode;
        }
    }

    @UsedByNative("arcoreapk.cc")
    public static int requestInstall(Activity activity, boolean z, int[] iArr) {
        try {
            iArr[0] = ArCoreApk.getInstance().requestInstall(activity, z).nativeCode;
            return ae.SUCCESS.E;
        } catch (Throwable th) {
            return a(th);
        }
    }

    @UsedByNative("arcoreapk.cc")
    public static int requestInstallCustom(Activity activity, boolean z, int i, int i2, int[] iArr) {
        try {
            iArr[0] = ArCoreApk.getInstance().requestInstall(activity, z, ArCoreApk.InstallBehavior.forNumber(i), ArCoreApk.UserMessageType.forNumber(i2)).nativeCode;
            return ae.SUCCESS.E;
        } catch (Throwable th) {
            return a(th);
        }
    }
}