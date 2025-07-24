package com.google.android.play.core.assetpacks;

/* compiled from: com.google.android.play:core@@1.10.3 */
/* loaded from: classes.dex */
public abstract class zzet {
    public abstract int zza();

    public abstract long zzb();

    public abstract String zzc();

    public abstract boolean zzd();

    public abstract boolean zze();

    public abstract byte[] zzf();

    public final boolean zzg() {
        if (zzc() == null) {
            return false;
        }
        return zzc().endsWith("/");
    }

    public final boolean zzh() {
        return zza() == 0;
    }
}