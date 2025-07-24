package com.google.android.gms.internal.clearcut;

/* loaded from: classes.dex */
public final class zzez extends zzex<zzey, zzey> {
    private static void zza(Object obj, zzey zzeyVar) {
        ((zzcg) obj).zzjp = zzeyVar;
    }

    /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object, int, long] */
    @Override // com.google.android.gms.internal.clearcut.zzex
    public final /* synthetic */ void zza(zzey zzeyVar, int i, long j) {
        zzeyVar.zzb(i << 3, Long.valueOf(j));
    }

    /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object, int, com.google.android.gms.internal.clearcut.zzbb] */
    @Override // com.google.android.gms.internal.clearcut.zzex
    public final /* synthetic */ void zza(zzey zzeyVar, int i, zzbb zzbbVar) {
        zzeyVar.zzb((i << 3) | 2, zzbbVar);
    }

    /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object, com.google.android.gms.internal.clearcut.zzfr] */
    @Override // com.google.android.gms.internal.clearcut.zzex
    public final /* synthetic */ void zza(zzey zzeyVar, zzfr zzfrVar) {
        zzeyVar.zzb(zzfrVar);
    }

    @Override // com.google.android.gms.internal.clearcut.zzex
    public final void zzc(Object obj) {
        ((zzcg) obj).zzjp.zzv();
    }

    /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object, com.google.android.gms.internal.clearcut.zzfr] */
    @Override // com.google.android.gms.internal.clearcut.zzex
    public final /* synthetic */ void zzc(zzey zzeyVar, zzfr zzfrVar) {
        zzeyVar.zza(zzfrVar);
    }

    /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
    @Override // com.google.android.gms.internal.clearcut.zzex
    public final /* synthetic */ zzey zzdz() {
        return zzey.zzeb();
    }

    /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object, java.lang.Object] */
    @Override // com.google.android.gms.internal.clearcut.zzex
    public final /* synthetic */ void zze(Object obj, zzey zzeyVar) {
        zza(obj, zzeyVar);
    }

    /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object, java.lang.Object] */
    @Override // com.google.android.gms.internal.clearcut.zzex
    public final /* synthetic */ void zzf(Object obj, zzey zzeyVar) {
        zza(obj, zzeyVar);
    }

    /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object, java.lang.Object] */
    /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
    @Override // com.google.android.gms.internal.clearcut.zzex
    public final /* synthetic */ zzey zzg(zzey zzeyVar, zzey zzeyVar2) {
        zzey zzeyVar3 = zzeyVar;
        zzey zzeyVar4 = zzeyVar2;
        return zzeyVar4.equals(zzey.zzea()) ? zzeyVar3 : zzey.zza(zzeyVar3, zzeyVar4);
    }

    /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object] */
    @Override // com.google.android.gms.internal.clearcut.zzex
    public final /* synthetic */ int zzm(zzey zzeyVar) {
        return zzeyVar.zzas();
    }

    /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
    @Override // com.google.android.gms.internal.clearcut.zzex
    public final /* synthetic */ zzey zzq(Object obj) {
        return ((zzcg) obj).zzjp;
    }

    /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object] */
    @Override // com.google.android.gms.internal.clearcut.zzex
    public final /* synthetic */ int zzr(zzey zzeyVar) {
        return zzeyVar.zzec();
    }
}