package com.google.android.gms.measurement.internal;

/* compiled from: com.google.android.gms:play-services-measurement@@21.2.0 */
/* loaded from: classes.dex */
public final class zzeo implements Runnable {
    public final /* synthetic */ boolean zza;
    public final /* synthetic */ zzep zzb;

    public zzeo(zzep zzepVar, boolean z) {
        this.zzb = zzepVar;
        this.zza = z;
    }

    @Override // java.lang.Runnable
    public final void run() {
        zzkt zzktVar;
        zzktVar = this.zzb.zzb;
        zzktVar.zzJ(this.zza);
    }
}