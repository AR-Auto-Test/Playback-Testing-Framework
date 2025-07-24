package com.google.android.gms.measurement.internal;

/* compiled from: com.google.android.gms:play-services-measurement@@21.2.0 */
/* loaded from: classes.dex */
public final class zzkj implements Runnable {
    public final /* synthetic */ zzku zza;
    public final /* synthetic */ zzkt zzb;

    public zzkj(zzkt zzktVar, zzku zzkuVar) {
        this.zzb = zzktVar;
        this.zza = zzkuVar;
    }

    @Override // java.lang.Runnable
    public final void run() {
        zzkt.zzy(this.zzb, this.zza);
        this.zzb.zzS();
    }
}