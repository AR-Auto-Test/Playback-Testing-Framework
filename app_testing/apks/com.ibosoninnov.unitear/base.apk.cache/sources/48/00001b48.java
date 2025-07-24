package com.google.android.play.core.splitinstall;

import android.os.RemoteException;
import com.google.android.play.core.internal.zzca;

/* compiled from: com.google.android.play:core@@1.10.3 */
/* loaded from: classes.dex */
public final class zzar extends com.google.android.play.core.internal.zzah {
    public final /* synthetic */ com.google.android.play.core.tasks.zzi zza;
    public final /* synthetic */ zzbc zzb;

    /* JADX WARN: 'super' call moved to the top of the method (can break code semantics) */
    public zzar(zzbc zzbcVar, com.google.android.play.core.tasks.zzi zziVar, com.google.android.play.core.tasks.zzi zziVar2) {
        super(zziVar);
        this.zzb = zzbcVar;
        this.zza = zziVar2;
    }

    @Override // com.google.android.play.core.internal.zzah
    public final void zza() {
        com.google.android.play.core.internal.zzag zzagVar;
        String str;
        try {
            zzbc zzbcVar = this.zzb;
            str = zzbcVar.zzd;
            ((zzca) this.zzb.zza.zze()).zzi(str, new zzaz(zzbcVar, this.zza));
        } catch (RemoteException e2) {
            zzagVar = zzbc.zzb;
            zzagVar.zzc(e2, "getSessionStates", new Object[0]);
            this.zza.zzd(new RuntimeException(e2));
        }
    }
}