package com.google.android.gms.measurement.internal;

import java.util.concurrent.Callable;

/* compiled from: com.google.android.gms:play-services-measurement@@21.2.0 */
/* loaded from: classes.dex */
public final class zzfy implements Callable {
    public final /* synthetic */ String zza;
    public final /* synthetic */ String zzb;
    public final /* synthetic */ String zzc;
    public final /* synthetic */ zzgj zzd;

    public zzfy(zzgj zzgjVar, String str, String str2, String str3) {
        this.zzd = zzgjVar;
        this.zza = str;
        this.zzb = str2;
        this.zzc = str3;
    }

    @Override // java.util.concurrent.Callable
    public final /* bridge */ /* synthetic */ Object call() {
        zzkt zzktVar;
        zzkt zzktVar2;
        zzktVar = this.zzd.zza;
        zzktVar.zzA();
        zzktVar2 = this.zzd.zza;
        return zzktVar2.zzi().zzs(this.zza, this.zzb, this.zzc);
    }
}