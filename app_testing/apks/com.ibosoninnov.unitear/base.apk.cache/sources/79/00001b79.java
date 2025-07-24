package com.google.android.play.core.tasks;

/* compiled from: com.google.android.play:core@@1.10.3 */
/* loaded from: classes.dex */
public final class zza implements Runnable {
    public final /* synthetic */ Task zza;
    public final /* synthetic */ zzb zzb;

    public zza(zzb zzbVar, Task task) {
        this.zzb = zzbVar;
        this.zza = task;
    }

    @Override // java.lang.Runnable
    public final void run() {
        Object obj;
        OnCompleteListener onCompleteListener;
        OnCompleteListener onCompleteListener2;
        obj = this.zzb.zzb;
        synchronized (obj) {
            zzb zzbVar = this.zzb;
            onCompleteListener = zzbVar.zzc;
            if (onCompleteListener != null) {
                onCompleteListener2 = zzbVar.zzc;
                onCompleteListener2.onComplete(this.zza);
            }
        }
    }
}