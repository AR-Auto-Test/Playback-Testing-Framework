package com.google.android.play.core.assetpacks;

import android.os.Bundle;
import android.os.RemoteException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/* compiled from: com.google.android.play:core@@1.10.3 */
/* loaded from: classes.dex */
public final class zzaf extends com.google.android.play.core.internal.zzah {
    public final /* synthetic */ List zza;
    public final /* synthetic */ Map zzb;
    public final /* synthetic */ com.google.android.play.core.tasks.zzi zzc;
    public final /* synthetic */ zzbe zzd;
    public final /* synthetic */ zzaw zze;

    /* JADX WARN: 'super' call moved to the top of the method (can break code semantics) */
    public zzaf(zzaw zzawVar, com.google.android.play.core.tasks.zzi zziVar, List list, Map map, com.google.android.play.core.tasks.zzi zziVar2, zzbe zzbeVar) {
        super(zziVar);
        this.zze = zzawVar;
        this.zza = list;
        this.zzb = map;
        this.zzc = zziVar2;
        this.zzd = zzbeVar;
    }

    @Override // com.google.android.play.core.internal.zzah
    public final void zza() {
        com.google.android.play.core.internal.zzag zzagVar;
        com.google.android.play.core.internal.zzas zzasVar;
        String str;
        zzco zzcoVar;
        zzeb zzebVar;
        ArrayList zzv = zzaw.zzv(this.zza);
        try {
            zzasVar = this.zze.zzf;
            str = this.zze.zzc;
            Bundle zzn = zzaw.zzn(this.zzb);
            zzaw zzawVar = this.zze;
            com.google.android.play.core.tasks.zzi zziVar = this.zzc;
            zzcoVar = zzawVar.zzd;
            zzebVar = zzawVar.zze;
            ((com.google.android.play.core.internal.zzu) zzasVar.zze()).zzk(str, zzv, zzn, new zzau(zzawVar, zziVar, zzcoVar, zzebVar, this.zzd));
        } catch (RemoteException e2) {
            zzagVar = zzaw.zza;
            zzagVar.zzc(e2, "getPackStates(%s)", this.zza);
            this.zzc.zzd(new RuntimeException(e2));
        }
    }
}