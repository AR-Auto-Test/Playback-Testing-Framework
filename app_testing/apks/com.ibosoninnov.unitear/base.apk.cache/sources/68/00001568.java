package com.google.android.gms.measurement.internal;

import android.os.RemoteException;
import android.text.TextUtils;
import com.google.android.gms.common.internal.Preconditions;
import java.util.Collections;
import java.util.concurrent.atomic.AtomicReference;

/* compiled from: com.google.android.gms:play-services-measurement-impl@@21.2.0 */
/* loaded from: classes.dex */
public final class zzjd implements Runnable {
    public final /* synthetic */ AtomicReference zza;
    public final /* synthetic */ String zzb;
    public final /* synthetic */ String zzc;
    public final /* synthetic */ zzq zzd;
    public final /* synthetic */ zzjm zze;

    public zzjd(zzjm zzjmVar, AtomicReference atomicReference, String str, String str2, String str3, zzq zzqVar) {
        this.zze = zzjmVar;
        this.zza = atomicReference;
        this.zzb = str2;
        this.zzc = str3;
        this.zzd = zzqVar;
    }

    @Override // java.lang.Runnable
    public final void run() {
        AtomicReference atomicReference;
        zzjm zzjmVar;
        zzdx zzdxVar;
        synchronized (this.zza) {
            try {
                zzjmVar = this.zze;
                zzdxVar = zzjmVar.zzb;
            } catch (RemoteException e2) {
                this.zze.zzt.zzay().zzd().zzd("(legacy) Failed to get conditional properties; remote exception", null, this.zzb, e2);
                this.zza.set(Collections.emptyList());
                atomicReference = this.zza;
            }
            if (zzdxVar == null) {
                zzjmVar.zzt.zzay().zzd().zzd("(legacy) Failed to get conditional properties; not connected to service", null, this.zzb, this.zzc);
                this.zza.set(Collections.emptyList());
                this.zza.notify();
                return;
            }
            if (TextUtils.isEmpty(null)) {
                Preconditions.checkNotNull(this.zzd);
                this.zza.set(zzdxVar.zzf(this.zzb, this.zzc, this.zzd));
            } else {
                this.zza.set(zzdxVar.zzg(null, this.zzb, this.zzc));
            }
            this.zze.zzQ();
            atomicReference = this.zza;
            atomicReference.notify();
        }
    }
}