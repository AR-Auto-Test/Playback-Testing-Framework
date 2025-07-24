package com.google.android.gms.measurement.internal;

import c.b.a.a.a;

/* compiled from: com.google.android.gms:play-services-measurement@@21.2.0 */
/* loaded from: classes.dex */
public final class zzke extends zzap {
    public final /* synthetic */ zzkf zza;

    /* JADX WARN: 'super' call moved to the top of the method (can break code semantics) */
    public zzke(zzkf zzkfVar, zzgm zzgmVar) {
        super(zzgmVar);
        this.zza = zzkfVar;
    }

    @Override // com.google.android.gms.measurement.internal.zzap
    public final void zzc() {
        this.zza.zza();
        a.F(this.zza.zzt, "Starting upload from DelayedRunnable");
        this.zza.zzf.zzX();
    }
}