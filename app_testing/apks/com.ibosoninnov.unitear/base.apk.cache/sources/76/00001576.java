package com.google.android.gms.measurement.internal;

/* compiled from: com.google.android.gms:play-services-measurement@@21.2.0 */
/* loaded from: classes.dex */
public final class zzjr implements Runnable {
    public final /* synthetic */ zzkt zza;
    public final /* synthetic */ Runnable zzb;

    public zzjr(zzjt zzjtVar, zzkt zzktVar, Runnable runnable) {
        this.zza = zzktVar;
        this.zzb = runnable;
    }

    @Override // java.lang.Runnable
    public final void run() {
        this.zza.zzA();
        this.zza.zzz(this.zzb);
        this.zza.zzX();
    }
}