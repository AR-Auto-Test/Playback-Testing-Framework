package com.google.android.play.core.assetpacks;

import android.os.Bundle;
import android.os.RemoteException;

/* compiled from: com.google.android.play:core@@1.10.3 */
/* loaded from: classes.dex */
public final class zzah extends com.google.android.play.core.internal.zzah {
    public final /* synthetic */ int zza;
    public final /* synthetic */ String zzb;
    public final /* synthetic */ com.google.android.play.core.tasks.zzi zzc;
    public final /* synthetic */ int zzd;
    public final /* synthetic */ zzaw zze;

    /* JADX WARN: 'super' call moved to the top of the method (can break code semantics) */
    public zzah(zzaw zzawVar, com.google.android.play.core.tasks.zzi zziVar, int i, String str, com.google.android.play.core.tasks.zzi zziVar2, int i2) {
        super(zziVar);
        this.zze = zzawVar;
        this.zza = i;
        this.zzb = str;
        this.zzc = zziVar2;
        this.zzd = i2;
    }

    @Override // com.google.android.play.core.internal.zzah
    public final void zza() {
        com.google.android.play.core.internal.zzag zzagVar;
        com.google.android.play.core.internal.zzas zzasVar;
        String str;
        Bundle zzz;
        Bundle zzA;
        try {
            zzasVar = this.zze.zzf;
            str = this.zze.zzc;
            zzz = zzaw.zzz(this.zza, this.zzb);
            zzA = zzaw.zzA();
            ((com.google.android.play.core.internal.zzu) zzasVar.zze()).zzh(str, zzz, zzA, new zzar(this.zze, this.zzc, this.zza, this.zzb, this.zzd));
        } catch (RemoteException e2) {
            zzagVar = zzaw.zza;
            zzagVar.zzc(e2, "notifyModuleCompleted", new Object[0]);
        }
    }
}