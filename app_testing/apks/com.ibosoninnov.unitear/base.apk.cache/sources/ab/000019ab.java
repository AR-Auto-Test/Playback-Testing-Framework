package com.google.android.play.core.appupdate;

import com.google.android.play.core.internal.zzcq;
import com.google.android.play.core.internal.zzcs;

/* compiled from: com.google.android.play:core@@1.10.3 */
/* loaded from: classes.dex */
public final class zzaa {
    private final zzaa zza = this;
    private final zzcs zzb;
    private final zzcs zzc;
    private final zzcs zzd;
    private final zzcs zze;
    private final zzcs zzf;
    private final zzcs zzg;

    public /* synthetic */ zzaa(zzh zzhVar, zzz zzzVar) {
        zzj zzjVar = new zzj(zzhVar);
        this.zzb = zzjVar;
        zzcs zzc = zzcq.zzc(new zzt(zzjVar));
        this.zzc = zzc;
        zzcs zzc2 = zzcq.zzc(new zzr(zzjVar, zzc));
        this.zzd = zzc2;
        zzcs zzc3 = zzcq.zzc(new zzc(zzjVar));
        this.zze = zzc3;
        zzcs zzc4 = zzcq.zzc(new zzg(zzc2, zzc3, zzjVar));
        this.zzf = zzc4;
        this.zzg = zzcq.zzc(new zzi(zzc4));
    }

    public final AppUpdateManager zza() {
        return (AppUpdateManager) this.zzg.zza();
    }
}