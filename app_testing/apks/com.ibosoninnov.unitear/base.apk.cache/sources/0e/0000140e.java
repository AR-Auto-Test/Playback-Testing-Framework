package com.google.android.gms.internal.vision;

/* compiled from: com.google.android.gms:play-services-vision-common@@19.1.3 */
/* loaded from: classes.dex */
public final class zzkc<K, V> {
    public static <K, V> void zza(zzii zziiVar, zzkf<K, V> zzkfVar, K k, V v) {
        zziu.zza(zziiVar, zzkfVar.zza, 1, k);
        zziu.zza(zziiVar, zzkfVar.zzc, 2, v);
    }

    public static <K, V> int zza(zzkf<K, V> zzkfVar, K k, V v) {
        return zziu.zza(zzkfVar.zzc, 2, v) + zziu.zza(zzkfVar.zza, 1, k);
    }
}