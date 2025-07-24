package com.google.android.gms.measurement.internal;

import com.google.android.gms.common.internal.Preconditions;
import java.util.concurrent.Callable;

/* compiled from: com.google.android.gms:play-services-measurement@@21.2.0 */
/* loaded from: classes.dex */
public final class zzkm implements Callable {
    public final /* synthetic */ zzq zza;
    public final /* synthetic */ zzkt zzb;

    public zzkm(zzkt zzktVar, zzq zzqVar) {
        this.zzb = zzktVar;
        this.zza = zzqVar;
    }

    @Override // java.util.concurrent.Callable
    public final /* bridge */ /* synthetic */ Object call() {
        zzai zzh = this.zzb.zzh((String) Preconditions.checkNotNull(this.zza.zza));
        zzah zzahVar = zzah.ANALYTICS_STORAGE;
        if (zzh.zzi(zzahVar) && zzai.zzb(this.zza.zzv).zzi(zzahVar)) {
            return this.zzb.zzd(this.zza).zzu();
        }
        this.zzb.zzay().zzj().zza("Analytics storage consent denied. Returning null app instance id");
        return null;
    }
}