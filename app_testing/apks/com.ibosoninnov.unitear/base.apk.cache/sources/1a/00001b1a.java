package com.google.android.play.core.splitinstall.model;

import c.b.a.a.a;
import java.util.HashMap;
import java.util.Map;

/* compiled from: com.google.android.play:core@@1.10.3 */
/* loaded from: classes.dex */
public final class zza {
    private static final Map zza;
    private static final Map zzb;
    private static final Map zzc;

    static {
        HashMap hashMap = new HashMap();
        zza = hashMap;
        HashMap hashMap2 = new HashMap();
        zzb = hashMap2;
        Integer d2 = a.d(hashMap, -1, "Too many sessions are running for current app, existing sessions must be resolved first.", -2, "A requested module is not available (to this user/device, for the installed apk).");
        Integer d3 = a.d(hashMap, -3, "Request is otherwise invalid.", -4, "Requested session is not found.");
        Integer d4 = a.d(hashMap, -5, "Split Install API is not available.", -6, "Network error: unable to obtain split details.");
        Integer d5 = a.d(hashMap, -7, "Download not permitted under current device circumstances (e.g. in background).", -8, "Requested session contains modules from an existing active session and also new modules.");
        Integer d6 = a.d(hashMap, -9, "Service handling split install has died.", -10, "Install failed due to insufficient storage.");
        Integer d7 = a.d(hashMap, -11, "Signature verification error when invoking SplitCompat.", -12, "Error in SplitCompat emulation.");
        Integer d8 = a.d(hashMap, -13, "Error in copying files for SplitCompat.", -14, "The Play Store app is either not installed or not the official version.");
        Integer d9 = a.d(hashMap, -15, "The app is not owned by any user on this device. An app is \"owned\" if it has been acquired from Play.", -100, "Unknown error processing split install.");
        hashMap2.put(-1, "ACTIVE_SESSIONS_LIMIT_EXCEEDED");
        hashMap2.put(d2, "MODULE_UNAVAILABLE");
        hashMap2.put(-3, "INVALID_REQUEST");
        hashMap2.put(d3, "DOWNLOAD_NOT_FOUND");
        hashMap2.put(-5, "API_NOT_AVAILABLE");
        hashMap2.put(d4, "NETWORK_ERROR");
        hashMap2.put(-7, "ACCESS_DENIED");
        hashMap2.put(d5, "INCOMPATIBLE_WITH_EXISTING_SESSION");
        hashMap2.put(-9, "SERVICE_DIED");
        hashMap2.put(d6, "INSUFFICIENT_STORAGE");
        hashMap2.put(-11, "SPLITCOMPAT_VERIFICATION_ERROR");
        hashMap2.put(d7, "SPLITCOMPAT_EMULATION_ERROR");
        hashMap2.put(-13, "SPLITCOMPAT_COPY_ERROR");
        hashMap2.put(d8, "PLAY_STORE_NOT_FOUND");
        hashMap2.put(-15, "APP_NOT_OWNED");
        hashMap2.put(d9, "INTERNAL_ERROR");
        zzc = new HashMap();
        for (Map.Entry entry : hashMap2.entrySet()) {
            zzc.put((String) entry.getValue(), (Integer) entry.getKey());
        }
    }

    @SplitInstallErrorCode
    public static int zza(String str) {
        Integer num = (Integer) zzc.get(str);
        if (num != null) {
            return num.intValue();
        }
        throw new IllegalArgumentException(String.valueOf(str).concat(" is unknown error."));
    }

    public static String zzb(@SplitInstallErrorCode int i) {
        Map map = zza;
        Integer valueOf = Integer.valueOf(i);
        if (map.containsKey(valueOf)) {
            Map map2 = zzb;
            if (map2.containsKey(valueOf)) {
                String str = (String) map.get(valueOf);
                String str2 = (String) map2.get(valueOf);
                StringBuilder sb = new StringBuilder(String.valueOf(str).length() + 118 + String.valueOf(str2).length());
                sb.append(str);
                sb.append(" (https://developer.android.com/reference/com/google/android/play/core/splitinstall/model/SplitInstallErrorCode.html#");
                sb.append(str2);
                sb.append(")");
                return sb.toString();
            }
            return "";
        }
        return "";
    }
}