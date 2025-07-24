package com.google.android.gms.measurement.internal;

/* compiled from: com.google.android.gms:play-services-measurement-impl@@21.2.0 */
/* loaded from: classes.dex */
public final class zzha implements Runnable {
    public final /* synthetic */ long zza;
    public final /* synthetic */ zzhx zzb;

    public zzha(zzhx zzhxVar, long j) {
        this.zzb = zzhxVar;
        this.zza = j;
    }

    @Override // java.lang.Runnable
    public final void run() {
        this.zzb.zzt.zzm().zzf.zzb(this.zza);
        this.zzb.zzt.zzay().zzc().zzb("Session timeout duration set", Long.valueOf(this.zza));
    }
}