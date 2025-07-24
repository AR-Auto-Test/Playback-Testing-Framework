package com.google.android.gms.measurement.internal;

/* compiled from: com.google.android.gms:play-services-measurement@@21.2.0 */
/* loaded from: classes.dex */
public final class zzks {
    public final String zza;
    public long zzb;

    private zzks(zzkt zzktVar, String str) {
        this.zza = str;
        this.zzb = zzktVar.zzav().elapsedRealtime();
    }
}