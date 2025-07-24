package com.google.android.gms.measurement.internal;

import android.os.RemoteException;
import c.b.a.a.a;

/* compiled from: com.google.android.gms:play-services-measurement-impl@@21.2.0 */
/* loaded from: classes.dex */
public final class zziu implements Runnable {
    public final /* synthetic */ zzie zza;
    public final /* synthetic */ zzjm zzb;

    public zziu(zzjm zzjmVar, zzie zzieVar) {
        this.zzb = zzjmVar;
        this.zza = zzieVar;
    }

    @Override // java.lang.Runnable
    public final void run() {
        zzdx zzdxVar;
        zzjm zzjmVar = this.zzb;
        zzdxVar = zzjmVar.zzb;
        if (zzdxVar == null) {
            a.E(zzjmVar.zzt, "Failed to send current screen to service");
            return;
        }
        try {
            zzie zzieVar = this.zza;
            if (zzieVar == null) {
                zzdxVar.zzq(0L, null, null, zzjmVar.zzt.zzau().getPackageName());
            } else {
                zzdxVar.zzq(zzieVar.zzc, zzieVar.zza, zzieVar.zzb, zzjmVar.zzt.zzau().getPackageName());
            }
            this.zzb.zzQ();
        } catch (RemoteException e2) {
            this.zzb.zzt.zzay().zzd().zzb("Failed to send current screen to the service", e2);
        }
    }
}