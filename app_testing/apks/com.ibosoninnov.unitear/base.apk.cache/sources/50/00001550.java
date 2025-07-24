package com.google.android.gms.measurement.internal;

import android.os.Bundle;

/* compiled from: com.google.android.gms:play-services-measurement-impl@@21.2.0 */
/* loaded from: classes.dex */
public final class zzig implements Runnable {
    public final /* synthetic */ Bundle zza;
    public final /* synthetic */ zzie zzb;
    public final /* synthetic */ zzie zzc;
    public final /* synthetic */ long zzd;
    public final /* synthetic */ zzim zze;

    public zzig(zzim zzimVar, Bundle bundle, zzie zzieVar, zzie zzieVar2, long j) {
        this.zze = zzimVar;
        this.zza = bundle;
        this.zzb = zzieVar;
        this.zzc = zzieVar2;
        this.zzd = j;
    }

    @Override // java.lang.Runnable
    public final void run() {
        zzim.zzp(this.zze, this.zza, this.zzb, this.zzc, this.zzd);
    }
}