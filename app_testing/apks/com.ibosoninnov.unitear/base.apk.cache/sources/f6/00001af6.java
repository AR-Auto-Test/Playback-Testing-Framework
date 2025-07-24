package com.google.android.play.core.review;

import android.os.Bundle;
import android.os.RemoteException;
import com.google.android.play.core.common.PlayCoreVersion;
import com.google.android.play.core.internal.zzac;
import com.google.android.play.core.internal.zzag;
import com.google.android.play.core.internal.zzah;

/* compiled from: com.google.android.play:core@@1.10.3 */
/* loaded from: classes.dex */
public final class zzf extends zzah {
    public final /* synthetic */ com.google.android.play.core.tasks.zzi zza;
    public final /* synthetic */ zzi zzb;

    /* JADX WARN: 'super' call moved to the top of the method (can break code semantics) */
    public zzf(zzi zziVar, com.google.android.play.core.tasks.zzi zziVar2, com.google.android.play.core.tasks.zzi zziVar3) {
        super(zziVar2);
        this.zzb = zziVar;
        this.zza = zziVar3;
    }

    @Override // com.google.android.play.core.internal.zzah
    public final void zza() {
        zzag zzagVar;
        String str;
        String str2;
        String str3;
        try {
            str2 = this.zzb.zzc;
            Bundle zza = PlayCoreVersion.zza("review");
            zzi zziVar = this.zzb;
            com.google.android.play.core.tasks.zzi zziVar2 = this.zza;
            str3 = zziVar.zzc;
            ((zzac) this.zzb.zza.zze()).zzc(str2, zza, new zzh(zziVar, zziVar2, str3));
        } catch (RemoteException e2) {
            zzagVar = zzi.zzb;
            str = this.zzb.zzc;
            zzagVar.zzc(e2, "error requesting in-app review for %s", str);
            this.zza.zzd(new RuntimeException(e2));
        }
    }
}