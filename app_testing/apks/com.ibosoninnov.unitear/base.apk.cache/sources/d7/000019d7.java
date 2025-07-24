package com.google.android.play.core.assetpacks;

import android.os.Bundle;
import android.os.RemoteException;

/* compiled from: com.google.android.play:core@@1.10.3 */
/* loaded from: classes.dex */
public final class zzab extends com.google.android.play.core.internal.zzah {
    public final /* synthetic */ String zza;
    public final /* synthetic */ com.google.android.play.core.tasks.zzi zzb;
    public final /* synthetic */ zzaw zzc;

    /* JADX WARN: 'super' call moved to the top of the method (can break code semantics) */
    public zzab(zzaw zzawVar, com.google.android.play.core.tasks.zzi zziVar, String str, com.google.android.play.core.tasks.zzi zziVar2) {
        super(zziVar);
        this.zzc = zzawVar;
        this.zza = str;
        this.zzb = zziVar2;
    }

    @Override // com.google.android.play.core.internal.zzah
    public final void zza() {
        com.google.android.play.core.internal.zzag zzagVar;
        com.google.android.play.core.internal.zzas zzasVar;
        String str;
        Bundle zzz;
        Bundle zzA;
        try {
            zzasVar = this.zzc.zzf;
            str = this.zzc.zzc;
            zzz = zzaw.zzz(0, this.zza);
            zzA = zzaw.zzA();
            ((com.google.android.play.core.internal.zzu) zzasVar.zze()).zzj(str, zzz, zzA, new zzat(this.zzc, this.zzb));
        } catch (RemoteException e2) {
            zzagVar = zzaw.zza;
            zzagVar.zzc(e2, "removePack(%s)", this.zza);
        }
    }
}