package com.google.android.gms.internal.measurement;

import c.b.a.a.a;
import java.util.List;

/* compiled from: com.google.android.gms:play-services-measurement@@21.2.0 */
/* loaded from: classes.dex */
public final class zzbi extends zzaw {
    public zzbi() {
        this.zza.add(zzbl.ADD);
        this.zza.add(zzbl.DIVIDE);
        this.zza.add(zzbl.MODULUS);
        this.zza.add(zzbl.MULTIPLY);
        this.zza.add(zzbl.NEGATE);
        this.zza.add(zzbl.POST_DECREMENT);
        this.zza.add(zzbl.POST_INCREMENT);
        this.zza.add(zzbl.PRE_DECREMENT);
        this.zza.add(zzbl.PRE_INCREMENT);
        this.zza.add(zzbl.SUBTRACT);
    }

    @Override // com.google.android.gms.internal.measurement.zzaw
    public final zzap zza(String str, zzg zzgVar, List list) {
        zzbl zzblVar = zzbl.ADD;
        int ordinal = zzh.zze(str).ordinal();
        if (ordinal == 0) {
            zzap zzb = zzgVar.zzb((zzap) a.f(zzblVar, 2, list, 0));
            zzap zzb2 = zzgVar.zzb((zzap) list.get(1));
            if (!(zzb instanceof zzal) && !(zzb instanceof zzat) && !(zzb2 instanceof zzal) && !(zzb2 instanceof zzat)) {
                return new zzah(Double.valueOf(zzb2.zzh().doubleValue() + zzb.zzh().doubleValue()));
            }
            return new zzat(String.valueOf(zzb.zzi()).concat(String.valueOf(zzb2.zzi())));
        } else if (ordinal != 21) {
            if (ordinal == 59) {
                zzap zzb3 = zzgVar.zzb((zzap) a.f(zzbl.SUBTRACT, 2, list, 0));
                zzah zzahVar = new zzah(Double.valueOf(-zzgVar.zzb((zzap) list.get(1)).zzh().doubleValue()));
                return new zzah(Double.valueOf(zzahVar.zzh().doubleValue() + zzb3.zzh().doubleValue()));
            } else if (ordinal == 52 || ordinal == 53) {
                zzh.zzh(str, 2, list);
                zzap zzb4 = zzgVar.zzb((zzap) list.get(0));
                zzgVar.zzb((zzap) list.get(1));
                return zzb4;
            } else if (ordinal != 55 && ordinal != 56) {
                switch (ordinal) {
                    case 44:
                        return new zzah(Double.valueOf(zzgVar.zzb((zzap) a.f(zzbl.MODULUS, 2, list, 0)).zzh().doubleValue() % zzgVar.zzb((zzap) list.get(1)).zzh().doubleValue()));
                    case 45:
                        return new zzah(Double.valueOf(zzgVar.zzb((zzap) a.f(zzbl.MULTIPLY, 2, list, 0)).zzh().doubleValue() * zzgVar.zzb((zzap) list.get(1)).zzh().doubleValue()));
                    case 46:
                        return new zzah(Double.valueOf(-zzgVar.zzb((zzap) a.f(zzbl.NEGATE, 1, list, 0)).zzh().doubleValue()));
                    default:
                        return super.zzb(str);
                }
            } else {
                zzh.zzh(str, 1, list);
                return zzgVar.zzb((zzap) list.get(0));
            }
        } else {
            return new zzah(Double.valueOf(zzgVar.zzb((zzap) a.f(zzbl.DIVIDE, 2, list, 0)).zzh().doubleValue() / zzgVar.zzb((zzap) list.get(1)).zzh().doubleValue()));
        }
    }
}