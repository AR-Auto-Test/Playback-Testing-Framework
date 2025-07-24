package com.google.android.play.core.install.model;

import c.b.a.a.a;
import java.util.HashMap;
import java.util.Map;

/* compiled from: com.google.android.play:core@@1.10.3 */
/* loaded from: classes.dex */
public final class zza {
    private static final Map zza;
    private static final Map zzb;

    static {
        HashMap hashMap = new HashMap();
        zza = hashMap;
        HashMap hashMap2 = new HashMap();
        zzb = hashMap2;
        Integer d2 = a.d(hashMap, -2, "An unknown error occurred.", -3, "The API is not available on this device.");
        Integer d3 = a.d(hashMap, -4, "The request that was sent by the app is malformed.", -5, "The install is unavailable to this user or device.");
        Integer d4 = a.d(hashMap, -6, "The download/install is not allowed, due to the current device state (e.g. low battery, low disk space, ...).", -7, "The install/update has not been (fully) downloaded yet.");
        Integer d5 = a.d(hashMap, -8, "The install is already in progress and there is no UI flow to resume.", -9, "The Play Store app is either not installed or not the official version.");
        Integer d6 = a.d(hashMap, -10, "The app is not owned by any user on this device. An app is \"owned\" if it has been acquired from Play.", -100, "An internal error happened in the Play Store.");
        hashMap2.put(-2, "ERROR_UNKNOWN");
        hashMap2.put(d2, "ERROR_API_NOT_AVAILABLE");
        hashMap2.put(-4, "ERROR_INVALID_REQUEST");
        hashMap2.put(d3, "ERROR_INSTALL_UNAVAILABLE");
        hashMap2.put(-6, "ERROR_INSTALL_NOT_ALLOWED");
        hashMap2.put(d4, "ERROR_DOWNLOAD_NOT_PRESENT");
        hashMap2.put(-8, "ERROR_INSTALL_IN_PROGRESS");
        hashMap2.put(d6, "ERROR_INTERNAL_ERROR");
        hashMap2.put(d5, "ERROR_PLAY_STORE_NOT_FOUND");
        hashMap2.put(-10, "ERROR_APP_NOT_OWNED");
        hashMap2.put(d6, "ERROR_INTERNAL_ERROR");
    }

    public static String zza(@InstallErrorCode int i) {
        Map map = zza;
        Integer valueOf = Integer.valueOf(i);
        if (map.containsKey(valueOf)) {
            Map map2 = zzb;
            if (map2.containsKey(valueOf)) {
                String str = (String) map.get(valueOf);
                String str2 = (String) map2.get(valueOf);
                StringBuilder sb = new StringBuilder(String.valueOf(str).length() + 103 + String.valueOf(str2).length());
                sb.append(str);
                sb.append(" (https://developer.android.com/reference/com/google/android/play/core/install/model/InstallErrorCode#");
                sb.append(str2);
                sb.append(")");
                return sb.toString();
            }
            return "";
        }
        return "";
    }
}