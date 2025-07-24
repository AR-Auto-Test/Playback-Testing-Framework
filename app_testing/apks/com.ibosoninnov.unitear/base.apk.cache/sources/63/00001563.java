package com.google.android.gms.measurement.internal;

import android.os.RemoteException;
import c.b.a.a.a;
import com.google.android.gms.common.internal.Preconditions;

/* compiled from: com.google.android.gms:play-services-measurement-impl@@21.2.0 */
/* loaded from: classes.dex */
public final class zziz implements Runnable {
    public final /* synthetic */ zzq zza;
    public final /* synthetic */ zzjm zzb;

    public zziz(zzjm zzjmVar, zzq zzqVar) {
        this.zzb = zzjmVar;
        this.zza = zzqVar;
    }

    @Override // java.lang.Runnable
    public final void run() {
        zzdx zzdxVar;
        zzjm zzjmVar = this.zzb;
        zzdxVar = zzjmVar.zzb;
        if (zzdxVar == null) {
            a.E(zzjmVar.zzt, "Failed to send measurementEnabled to service");
            return;
        }
        try {
            Preconditions.checkNotNull(this.zza);
            zzdxVar.zzs(this.zza);
            this.zzb.zzQ();
        } catch (RemoteException e2) {
            this.zzb.zzt.zzay().zzd().zzb("Failed to send measurementEnabled to the service", e2);
        }
    }
}