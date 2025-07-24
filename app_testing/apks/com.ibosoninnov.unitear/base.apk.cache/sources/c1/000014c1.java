package com.google.android.gms.measurement.internal;

import android.os.Bundle;
import b.f.a;
import com.google.android.gms.common.internal.Preconditions;
import java.util.Map;

/* compiled from: com.google.android.gms:play-services-measurement-impl@@21.2.0 */
/* loaded from: classes.dex */
public final class zzd extends zze {
    private final Map zza;
    private final Map zzb;
    private long zzc;

    public zzd(zzfr zzfrVar) {
        super(zzfrVar);
        this.zzb = new a();
        this.zza = new a();
    }

    public static /* synthetic */ void zza(zzd zzdVar, String str, long j) {
        zzdVar.zzg();
        Preconditions.checkNotEmpty(str);
        if (zzdVar.zzb.isEmpty()) {
            zzdVar.zzc = j;
        }
        Integer num = (Integer) zzdVar.zzb.get(str);
        if (num != null) {
            zzdVar.zzb.put(str, Integer.valueOf(num.intValue() + 1));
        } else if (zzdVar.zzb.size() >= 100) {
            c.b.a.a.a.G(zzdVar.zzt, "Too many ads visible");
        } else {
            zzdVar.zzb.put(str, 1);
            zzdVar.zza.put(str, Long.valueOf(j));
        }
    }

    public static /* synthetic */ void zzb(zzd zzdVar, String str, long j) {
        zzdVar.zzg();
        Preconditions.checkNotEmpty(str);
        Integer num = (Integer) zzdVar.zzb.get(str);
        if (num != null) {
            zzie zzj = zzdVar.zzt.zzs().zzj(false);
            int intValue = num.intValue() - 1;
            if (intValue == 0) {
                zzdVar.zzb.remove(str);
                Long l = (Long) zzdVar.zza.get(str);
                if (l == null) {
                    c.b.a.a.a.E(zzdVar.zzt, "First ad unit exposure time was never set");
                } else {
                    long longValue = l.longValue();
                    zzdVar.zza.remove(str);
                    zzdVar.zzi(str, j - longValue, zzj);
                }
                if (zzdVar.zzb.isEmpty()) {
                    long j2 = zzdVar.zzc;
                    if (j2 == 0) {
                        c.b.a.a.a.E(zzdVar.zzt, "First ad exposure time was never set");
                        return;
                    }
                    zzdVar.zzh(j - j2, zzj);
                    zzdVar.zzc = 0L;
                    return;
                }
                return;
            }
            zzdVar.zzb.put(str, Integer.valueOf(intValue));
            return;
        }
        zzdVar.zzt.zzay().zzd().zzb("Call to endAdUnitExposure for unknown ad unit id", str);
    }

    private final void zzh(long j, zzie zzieVar) {
        if (zzieVar == null) {
            c.b.a.a.a.F(this.zzt, "Not logging ad exposure. No active activity");
        } else if (j < 1000) {
            this.zzt.zzay().zzj().zzb("Not logging ad exposure. Less than 1000 ms. exposure", Long.valueOf(j));
        } else {
            Bundle bundle = new Bundle();
            bundle.putLong("_xt", j);
            zzlb.zzK(zzieVar, bundle, true);
            this.zzt.zzq().zzG("am", "_xa", bundle);
        }
    }

    private final void zzi(String str, long j, zzie zzieVar) {
        if (zzieVar == null) {
            c.b.a.a.a.F(this.zzt, "Not logging ad unit exposure. No active activity");
        } else if (j < 1000) {
            this.zzt.zzay().zzj().zzb("Not logging ad unit exposure. Less than 1000 ms. exposure", Long.valueOf(j));
        } else {
            Bundle bundle = new Bundle();
            bundle.putString("_ai", str);
            bundle.putLong("_xt", j);
            zzlb.zzK(zzieVar, bundle, true);
            this.zzt.zzq().zzG("am", "_xu", bundle);
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public final void zzj(long j) {
        for (String str : this.zza.keySet()) {
            this.zza.put(str, Long.valueOf(j));
        }
        if (this.zza.isEmpty()) {
            return;
        }
        this.zzc = j;
    }

    public final void zzd(String str, long j) {
        if (str != null && str.length() != 0) {
            this.zzt.zzaz().zzp(new zza(this, str, j));
        } else {
            c.b.a.a.a.E(this.zzt, "Ad unit id must be a non-empty string");
        }
    }

    public final void zze(String str, long j) {
        if (str != null && str.length() != 0) {
            this.zzt.zzaz().zzp(new zzb(this, str, j));
        } else {
            c.b.a.a.a.E(this.zzt, "Ad unit id must be a non-empty string");
        }
    }

    public final void zzf(long j) {
        zzie zzj = this.zzt.zzs().zzj(false);
        for (String str : this.zza.keySet()) {
            zzi(str, j - ((Long) this.zza.get(str)).longValue(), zzj);
        }
        if (!this.zza.isEmpty()) {
            zzh(j - this.zzc, zzj);
        }
        zzj(j);
    }
}