package com.google.android.play.core.assetpacks;

import android.os.Bundle;
import android.os.RemoteException;

/* compiled from: com.google.android.play:core@@1.10.3 */
/* loaded from: classes.dex */
public final class zzag extends com.google.android.play.core.internal.zzah {
    public final /* synthetic */ int zza;
    public final /* synthetic */ String zzb;
    public final /* synthetic */ String zzc;
    public final /* synthetic */ int zzd;
    public final /* synthetic */ com.google.android.play.core.tasks.zzi zze;
    public final /* synthetic */ zzaw zzf;

    /* JADX WARN: 'super' call moved to the top of the method (can break code semantics) */
    public zzag(zzaw zzawVar, com.google.android.play.core.tasks.zzi zziVar, int i, String str, String str2, int i2, com.google.android.play.core.tasks.zzi zziVar2) {
        super(zziVar);
        this.zzf = zzawVar;
        this.zza = i;
        this.zzb = str;
        this.zzc = str2;
        this.zzd = i2;
        this.zze = zziVar2;
    }

    @Override // com.google.android.play.core.internal.zzah
    public final void zza() {
        com.google.android.play.core.internal.zzag zzagVar;
        com.google.android.play.core.internal.zzas zzasVar;
        String str;
        Bundle zzA;
        try {
            zzasVar = this.zzf.zzf;
            str = this.zzf.zzc;
            Bundle zzk = zzaw.zzk(this.zza, this.zzb, this.zzc, this.zzd);
            zzA = zzaw.zzA();
            ((com.google.android.play.core.internal.zzu) zzasVar.zze()).zzg(str, zzk, zzA, new zzaq(this.zzf, this.zze));
        } catch (RemoteException e2) {
            zzagVar = zzaw.zza;
            zzagVar.zzc(e2, "notifyChunkTransferred", new Object[0]);
        }
    }
}