package com.google.ar.sceneform.utilities;

import android.os.Looper;

/* loaded from: classes.dex */
public class AndroidPreconditions {
    private static final boolean IS_ANDROID_API_AVAILABLE = checkAndroidApiAvailable();
    private static final boolean IS_MIN_ANDROID_API_LEVEL = isMinAndroidApiLevelImpl();
    private static boolean isUnderTesting = false;

    private static boolean checkAndroidApiAvailable() {
        try {
            Class.forName("android.app.Activity");
            return true;
        } catch (ClassNotFoundException unused) {
            return false;
        }
    }

    public static void checkMinAndroidApiLevel() {
        Preconditions.checkState(isMinAndroidApiLevel(), "Sceneform requires Android N or later");
    }

    public static void checkUiThread() {
        if (!isAndroidApiAvailable() || isUnderTesting()) {
            return;
        }
        Preconditions.checkState(Looper.getMainLooper().getThread() == Thread.currentThread(), "Must be called from the UI thread.");
    }

    public static boolean isAndroidApiAvailable() {
        return IS_ANDROID_API_AVAILABLE;
    }

    public static boolean isMinAndroidApiLevel() {
        return isUnderTesting() || IS_MIN_ANDROID_API_LEVEL;
    }

    private static boolean isMinAndroidApiLevelImpl() {
        isAndroidApiAvailable();
        return true;
    }

    public static boolean isUnderTesting() {
        return isUnderTesting;
    }

    public static void setUnderTesting(boolean z) {
        isUnderTesting = z;
    }
}