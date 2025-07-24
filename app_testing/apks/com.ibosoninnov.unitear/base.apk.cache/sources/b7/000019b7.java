package com.google.android.play.core.appupdate;

import android.os.Bundle;
import android.os.RemoteException;
import com.google.android.play.core.internal.zzag;
import com.google.android.play.core.internal.zzah;

/* compiled from: com.google.android.play:core@@1.10.3 */
/* loaded from: classes.dex */
public final class zzm extends zzah {
    public final /* synthetic */ com.google.android.play.core.tasks.zzi zza;
    public final /* synthetic */ String zzb;
    public final /* synthetic */ zzq zzc;

    /* JADX WARN: 'super' call moved to the top of the method (can break code semantics) */
    public zzm(zzq zzqVar, com.google.android.play.core.tasks.zzi zziVar, com.google.android.play.core.tasks.zzi zziVar2, String str) {
        super(zziVar);
        this.zzc = zzqVar;
        this.zza = zziVar2;
        this.zzb = str;
    }

    @Override // com.google.android.play.core.internal.zzah
    public final void zza() {
        zzag zzagVar;
        String str;
        Bundle zzi;
        try {
            str = this.zzc.zzd;
            zzi = zzq.zzi();
            ((com.google.android.play.core.internal.zzp) this.zzc.zza.zze()).zzc(str, zzi, new zzo(this.zzc, this.zza));
        } catch (RemoteException e2) {
            zzagVar = zzq.zzb;
            zzagVar.zzc(e2, "completeUpdate(%s)", this.zzb);
            this.zza.zzd(new RuntimeException(e2));
        }
    }
}