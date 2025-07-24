package com.google.android.play.core.splitinstall;

import android.os.RemoteException;
import com.google.android.play.core.internal.zzca;
import java.util.List;

/* compiled from: com.google.android.play:core@@1.10.3 */
/* loaded from: classes.dex */
public final class zzao extends com.google.android.play.core.internal.zzah {
    public final /* synthetic */ List zza;
    public final /* synthetic */ com.google.android.play.core.tasks.zzi zzb;
    public final /* synthetic */ zzbc zzc;

    /* JADX WARN: 'super' call moved to the top of the method (can break code semantics) */
    public zzao(zzbc zzbcVar, com.google.android.play.core.tasks.zzi zziVar, List list, com.google.android.play.core.tasks.zzi zziVar2) {
        super(zziVar);
        this.zzc = zzbcVar;
        this.zza = list;
        this.zzb = zziVar2;
    }

    @Override // com.google.android.play.core.internal.zzah
    public final void zza() {
        com.google.android.play.core.internal.zzag zzagVar;
        String str;
        try {
            str = this.zzc.zzd;
            ((zzca) this.zzc.zza.zze()).zze(str, zzbc.zzl(this.zza), zzbc.zza(), new zzav(this.zzc, this.zzb));
        } catch (RemoteException e2) {
            zzagVar = zzbc.zzb;
            zzagVar.zzc(e2, "deferredLanguageInstall(%s)", this.zza);
            this.zzb.zzd(new RuntimeException(e2));
        }
    }
}