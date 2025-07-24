package com.google.android.gms.measurement.internal;

import android.os.RemoteException;
import com.google.android.gms.common.internal.Preconditions;
import java.util.ArrayList;

/* compiled from: com.google.android.gms:play-services-measurement-impl@@21.2.0 */
/* loaded from: classes.dex */
public final class zzje implements Runnable {
    public final /* synthetic */ String zza;
    public final /* synthetic */ String zzb;
    public final /* synthetic */ zzq zzc;
    public final /* synthetic */ com.google.android.gms.internal.measurement.zzcf zzd;
    public final /* synthetic */ zzjm zze;

    public zzje(zzjm zzjmVar, String str, String str2, zzq zzqVar, com.google.android.gms.internal.measurement.zzcf zzcfVar) {
        this.zze = zzjmVar;
        this.zza = str;
        this.zzb = str2;
        this.zzc = zzqVar;
        this.zzd = zzcfVar;
    }

    @Override // java.lang.Runnable
    public final void run() {
        zzfr zzfrVar;
        zzdx zzdxVar;
        ArrayList arrayList = new ArrayList();
        try {
            try {
                zzjm zzjmVar = this.zze;
                zzdxVar = zzjmVar.zzb;
                if (zzdxVar == null) {
                    zzjmVar.zzt.zzay().zzd().zzc("Failed to get conditional properties; not connected to service", this.zza, this.zzb);
                    zzfrVar = this.zze.zzt;
                } else {
                    Preconditions.checkNotNull(this.zzc);
                    arrayList = zzlb.zzH(zzdxVar.zzf(this.zza, this.zzb, this.zzc));
                    this.zze.zzQ();
                    zzfrVar = this.zze.zzt;
                }
            } catch (RemoteException e2) {
                this.zze.zzt.zzay().zzd().zzd("Failed to get conditional properties; remote exception", this.zza, this.zzb, e2);
                zzfrVar = this.zze.zzt;
            }
            zzfrVar.zzv().zzQ(this.zzd, arrayList);
        } catch (Throwable th) {
            this.zze.zzt.zzv().zzQ(this.zzd, arrayList);
            throw th;
        }
    }
}