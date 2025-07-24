package com.google.android.play.core.assetpacks;

import java.util.concurrent.atomic.AtomicBoolean;

/* compiled from: com.google.android.play:core@@1.10.3 */
/* loaded from: classes.dex */
public final class zzcl {
    private static final com.google.android.play.core.internal.zzag zza = new com.google.android.play.core.internal.zzag("ExtractorLooper");
    private final zzde zzb;
    private final zzcf zzc;
    private final zzer zzd;
    private final zzdu zze;
    private final zzdz zzf;
    private final zzeg zzg;
    private final zzek zzh;
    private final com.google.android.play.core.internal.zzco zzi;
    private final zzdh zzj;
    private final AtomicBoolean zzk = new AtomicBoolean(false);

    public zzcl(zzde zzdeVar, com.google.android.play.core.internal.zzco zzcoVar, zzcf zzcfVar, zzer zzerVar, zzdu zzduVar, zzdz zzdzVar, zzeg zzegVar, zzek zzekVar, zzdh zzdhVar) {
        this.zzb = zzdeVar;
        this.zzi = zzcoVar;
        this.zzc = zzcfVar;
        this.zzd = zzerVar;
        this.zze = zzduVar;
        this.zzf = zzdzVar;
        this.zzg = zzegVar;
        this.zzh = zzekVar;
        this.zzj = zzdhVar;
    }

    private final void zzb(int i, Exception exc) {
        try {
            this.zzb.zzm(i, 5);
            this.zzb.zzn(i);
        } catch (zzck unused) {
            zza.zzb("Error during error handling: %s", exc.getMessage());
        }
    }

    public final void zza() {
        com.google.android.play.core.internal.zzag zzagVar = zza;
        zzagVar.zza("Run extractor loop", new Object[0]);
        if (!this.zzk.compareAndSet(false, true)) {
            zzagVar.zze("runLoop already looping; return", new Object[0]);
            return;
        }
        while (true) {
            zzdg zzdgVar = null;
            try {
                zzdgVar = this.zzj.zza();
            } catch (zzck e2) {
                zza.zzb("Error while getting next extraction task: %s", e2.getMessage());
                if (e2.zza >= 0) {
                    ((zzy) this.zzi.zza()).zzi(e2.zza);
                    zzb(e2.zza, e2);
                }
            }
            if (zzdgVar != null) {
                try {
                    if (zzdgVar instanceof zzce) {
                        this.zzc.zza((zzce) zzdgVar);
                    } else if (zzdgVar instanceof zzeq) {
                        this.zzd.zza((zzeq) zzdgVar);
                    } else if (zzdgVar instanceof zzdt) {
                        this.zze.zza((zzdt) zzdgVar);
                    } else if (zzdgVar instanceof zzdw) {
                        this.zzf.zza((zzdw) zzdgVar);
                    } else if (zzdgVar instanceof zzef) {
                        this.zzg.zza((zzef) zzdgVar);
                    } else if (zzdgVar instanceof zzei) {
                        this.zzh.zza((zzei) zzdgVar);
                    } else {
                        zza.zzb("Unknown task type: %s", zzdgVar.getClass().getName());
                    }
                } catch (Exception e3) {
                    zza.zzb("Error during extraction task: %s", e3.getMessage());
                    ((zzy) this.zzi.zza()).zzi(zzdgVar.zzk);
                    zzb(zzdgVar.zzk, e3);
                }
            } else {
                this.zzk.set(false);
                return;
            }
        }
    }
}