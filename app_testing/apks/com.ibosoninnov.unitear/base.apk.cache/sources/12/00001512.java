package com.google.android.gms.measurement.internal;

/* compiled from: com.google.android.gms:play-services-measurement@@21.2.0 */
/* loaded from: classes.dex */
public final class zzfz implements Runnable {
    public final /* synthetic */ zzq zza;
    public final /* synthetic */ zzgj zzb;

    public zzfz(zzgj zzgjVar, zzq zzqVar) {
        this.zzb = zzgjVar;
        this.zza = zzqVar;
    }

    @Override // java.lang.Runnable
    public final void run() {
        zzkt zzktVar;
        zzkt zzktVar2;
        zzktVar = this.zzb.zza;
        zzktVar.zzA();
        zzktVar2 = this.zzb.zza;
        zzktVar2.zzQ(this.zza);
    }
}