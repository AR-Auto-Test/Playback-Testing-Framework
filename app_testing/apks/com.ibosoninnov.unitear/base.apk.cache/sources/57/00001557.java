package com.google.android.gms.measurement.internal;

import android.os.Bundle;
import android.os.RemoteException;
import com.google.android.gms.common.internal.Preconditions;
import java.util.List;

/* compiled from: com.google.android.gms:play-services-measurement-impl@@21.2.0 */
/* loaded from: classes.dex */
public final class zzin implements Runnable {
    public final /* synthetic */ String zza;
    public final /* synthetic */ String zzb;
    public final /* synthetic */ zzq zzc;
    public final /* synthetic */ boolean zzd;
    public final /* synthetic */ com.google.android.gms.internal.measurement.zzcf zze;
    public final /* synthetic */ zzjm zzf;

    public zzin(zzjm zzjmVar, String str, String str2, zzq zzqVar, boolean z, com.google.android.gms.internal.measurement.zzcf zzcfVar) {
        this.zzf = zzjmVar;
        this.zza = str;
        this.zzb = str2;
        this.zzc = zzqVar;
        this.zzd = z;
        this.zze = zzcfVar;
    }

    @Override // java.lang.Runnable
    public final void run() {
        Bundle bundle;
        RemoteException e2;
        zzdx zzdxVar;
        Bundle bundle2 = new Bundle();
        try {
            zzjm zzjmVar = this.zzf;
            zzdxVar = zzjmVar.zzb;
            if (zzdxVar == null) {
                zzjmVar.zzt.zzay().zzd().zzc("Failed to get user properties; not connected to service", this.zza, this.zzb);
                this.zzf.zzt.zzv().zzR(this.zze, bundle2);
                return;
            }
            Preconditions.checkNotNull(this.zzc);
            List<zzkw> zzh = zzdxVar.zzh(this.zza, this.zzb, this.zzd, this.zzc);
            bundle = new Bundle();
            if (zzh != null) {
                for (zzkw zzkwVar : zzh) {
                    String str = zzkwVar.zze;
                    if (str != null) {
                        bundle.putString(zzkwVar.zzb, str);
                    } else {
                        Long l = zzkwVar.zzd;
                        if (l != null) {
                            bundle.putLong(zzkwVar.zzb, l.longValue());
                        } else {
                            Double d2 = zzkwVar.zzg;
                            if (d2 != null) {
                                bundle.putDouble(zzkwVar.zzb, d2.doubleValue());
                            }
                        }
                    }
                }
            }
            try {
                try {
                    this.zzf.zzQ();
                    this.zzf.zzt.zzv().zzR(this.zze, bundle);
                } catch (Throwable th) {
                    th = th;
                    bundle2 = bundle;
                    this.zzf.zzt.zzv().zzR(this.zze, bundle2);
                    throw th;
                }
            } catch (RemoteException e3) {
                e2 = e3;
                this.zzf.zzt.zzay().zzd().zzc("Failed to get user properties; remote exception", this.zza, e2);
                this.zzf.zzt.zzv().zzR(this.zze, bundle);
            }
        } catch (RemoteException e4) {
            bundle = bundle2;
            e2 = e4;
        } catch (Throwable th2) {
            th = th2;
            this.zzf.zzt.zzv().zzR(this.zze, bundle2);
            throw th;
        }
    }
}