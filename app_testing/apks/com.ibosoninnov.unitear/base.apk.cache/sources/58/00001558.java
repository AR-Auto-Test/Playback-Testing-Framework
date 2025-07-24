package com.google.android.gms.measurement.internal;

import c.b.a.a.a;
import com.google.android.gms.common.internal.Preconditions;

/* compiled from: com.google.android.gms:play-services-measurement-impl@@21.2.0 */
/* loaded from: classes.dex */
public final class zzio implements Runnable {
    public final /* synthetic */ zzq zza;
    public final /* synthetic */ boolean zzb;
    public final /* synthetic */ zzkw zzc;
    public final /* synthetic */ zzjm zzd;

    public zzio(zzjm zzjmVar, zzq zzqVar, boolean z, zzkw zzkwVar) {
        this.zzd = zzjmVar;
        this.zza = zzqVar;
        this.zzb = z;
        this.zzc = zzkwVar;
    }

    @Override // java.lang.Runnable
    public final void run() {
        zzdx zzdxVar;
        zzjm zzjmVar = this.zzd;
        zzdxVar = zzjmVar.zzb;
        if (zzdxVar == null) {
            a.E(zzjmVar.zzt, "Discarding data. Failed to set user property");
            return;
        }
        Preconditions.checkNotNull(this.zza);
        this.zzd.zzD(zzdxVar, this.zzb ? null : this.zzc, this.zza);
        this.zzd.zzQ();
    }
}