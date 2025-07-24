package com.google.android.gms.measurement.internal;

/* compiled from: com.google.android.gms:play-services-measurement-impl@@21.2.0 */
/* loaded from: classes.dex */
public final class zzii implements Runnable {
    public final /* synthetic */ zzim zza;

    public zzii(zzim zzimVar) {
        this.zza = zzimVar;
    }

    @Override // java.lang.Runnable
    public final void run() {
        zzie zzieVar;
        zzim zzimVar = this.zza;
        zzieVar = zzimVar.zzh;
        zzimVar.zza = zzieVar;
    }
}