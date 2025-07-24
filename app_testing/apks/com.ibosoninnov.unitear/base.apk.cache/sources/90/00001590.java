package com.google.android.gms.measurement.internal;

import com.google.android.gms.common.internal.Preconditions;
import java.util.ArrayList;
import java.util.List;

/* compiled from: com.google.android.gms:play-services-measurement@@21.2.0 */
/* loaded from: classes.dex */
public final class zzkq {
    public com.google.android.gms.internal.measurement.zzgd zza;
    public List zzb;
    public List zzc;
    public long zzd;
    public final /* synthetic */ zzkt zze;

    /* JADX DEBUG: Marked for inline */
    /* JADX DEBUG: Method not inlined, still used in: [com.google.android.gms.measurement.internal.zzkt.zzah(java.lang.String, long):boolean] */
    public /* synthetic */ zzkq(zzkt zzktVar, zzkp zzkpVar) {
        this.zze = zzktVar;
    }

    private static final long zzb(com.google.android.gms.internal.measurement.zzft zzftVar) {
        return ((zzftVar.zzd() / 1000) / 60) / 60;
    }

    public final boolean zza(long j, com.google.android.gms.internal.measurement.zzft zzftVar) {
        Preconditions.checkNotNull(zzftVar);
        if (this.zzc == null) {
            this.zzc = new ArrayList();
        }
        if (this.zzb == null) {
            this.zzb = new ArrayList();
        }
        if (this.zzc.isEmpty() || zzb((com.google.android.gms.internal.measurement.zzft) this.zzc.get(0)) == zzb(zzftVar)) {
            long zzbw = this.zzd + zzftVar.zzbw();
            this.zze.zzg();
            if (zzbw >= Math.max(0, ((Integer) zzdu.zzh.zza(null)).intValue())) {
                return false;
            }
            this.zzd = zzbw;
            this.zzc.add(zzftVar);
            this.zzb.add(Long.valueOf(j));
            int size = this.zzc.size();
            this.zze.zzg();
            return size < Math.max(1, ((Integer) zzdu.zzi.zza(null)).intValue());
        }
        return false;
    }
}