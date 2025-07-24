package com.google.android.gms.internal.measurement;

import c.b.a.a.a;
import java.util.Iterator;
import java.util.List;

/* compiled from: com.google.android.gms:play-services-measurement@@21.2.0 */
/* loaded from: classes.dex */
public final class zzbk extends zzaw {
    public zzbk() {
        this.zza.add(zzbl.ASSIGN);
        this.zza.add(zzbl.CONST);
        this.zza.add(zzbl.CREATE_ARRAY);
        this.zza.add(zzbl.CREATE_OBJECT);
        this.zza.add(zzbl.EXPRESSION_LIST);
        this.zza.add(zzbl.GET);
        this.zza.add(zzbl.GET_INDEX);
        this.zza.add(zzbl.GET_PROPERTY);
        this.zza.add(zzbl.NULL);
        this.zza.add(zzbl.SET_PROPERTY);
        this.zza.add(zzbl.TYPEOF);
        this.zza.add(zzbl.UNDEFINED);
        this.zza.add(zzbl.VAR);
    }

    @Override // com.google.android.gms.internal.measurement.zzaw
    public final zzap zza(String str, zzg zzgVar, List list) {
        String str2;
        zzbl zzblVar = zzbl.ADD;
        int ordinal = zzh.zze(str).ordinal();
        int i = 0;
        if (ordinal == 3) {
            zzap zzb = zzgVar.zzb((zzap) a.f(zzbl.ASSIGN, 2, list, 0));
            if (zzb instanceof zzat) {
                if (zzgVar.zzh(zzb.zzi())) {
                    zzap zzb2 = zzgVar.zzb((zzap) list.get(1));
                    zzgVar.zzg(zzb.zzi(), zzb2);
                    return zzb2;
                }
                throw new IllegalArgumentException(String.format("Attempting to assign undefined value %s", zzb.zzi()));
            }
            throw new IllegalArgumentException(String.format("Expected string for assign var. got %s", zzb.getClass().getCanonicalName()));
        } else if (ordinal == 14) {
            zzh.zzi(zzbl.CONST.name(), 2, list);
            if (list.size() % 2 == 0) {
                for (int i2 = 0; i2 < list.size() - 1; i2 += 2) {
                    zzap zzb3 = zzgVar.zzb((zzap) list.get(i2));
                    if (zzb3 instanceof zzat) {
                        zzgVar.zzf(zzb3.zzi(), zzgVar.zzb((zzap) list.get(i2 + 1)));
                    } else {
                        throw new IllegalArgumentException(String.format("Expected string for const name. got %s", zzb3.getClass().getCanonicalName()));
                    }
                }
                return zzap.zzf;
            }
            throw new IllegalArgumentException(String.format("CONST requires an even number of arguments, found %s", Integer.valueOf(list.size())));
        } else if (ordinal == 24) {
            zzh.zzi(zzbl.EXPRESSION_LIST.name(), 1, list);
            zzap zzapVar = zzap.zzf;
            while (i < list.size()) {
                zzapVar = zzgVar.zzb((zzap) list.get(i));
                if (zzapVar instanceof zzag) {
                    throw new IllegalStateException("ControlValue cannot be in an expression list");
                }
                i++;
            }
            return zzapVar;
        } else if (ordinal == 33) {
            zzap zzb4 = zzgVar.zzb((zzap) a.f(zzbl.GET, 1, list, 0));
            if (zzb4 instanceof zzat) {
                return zzgVar.zzd(zzb4.zzi());
            }
            throw new IllegalArgumentException(String.format("Expected string for get var. got %s", zzb4.getClass().getCanonicalName()));
        } else if (ordinal == 49) {
            zzh.zzh(zzbl.NULL.name(), 0, list);
            return zzap.zzg;
        } else if (ordinal == 58) {
            zzap zzb5 = zzgVar.zzb((zzap) a.f(zzbl.SET_PROPERTY, 3, list, 0));
            zzap zzb6 = zzgVar.zzb((zzap) list.get(1));
            zzap zzb7 = zzgVar.zzb((zzap) list.get(2));
            if (zzb5 != zzap.zzf && zzb5 != zzap.zzg) {
                if ((zzb5 instanceof zzae) && (zzb6 instanceof zzah)) {
                    ((zzae) zzb5).zzq(zzb6.zzh().intValue(), zzb7);
                } else if (zzb5 instanceof zzal) {
                    ((zzal) zzb5).zzr(zzb6.zzi(), zzb7);
                }
                return zzb7;
            }
            throw new IllegalStateException(String.format("Can't set property %s of %s", zzb6.zzi(), zzb5.zzi()));
        } else if (ordinal == 17) {
            if (list.isEmpty()) {
                return new zzae();
            }
            zzae zzaeVar = new zzae();
            Iterator it = list.iterator();
            while (it.hasNext()) {
                zzap zzb8 = zzgVar.zzb((zzap) it.next());
                if (!(zzb8 instanceof zzag)) {
                    zzaeVar.zzq(i, zzb8);
                    i++;
                } else {
                    throw new IllegalStateException("Failed to evaluate array element");
                }
            }
            return zzaeVar;
        } else if (ordinal == 18) {
            if (list.isEmpty()) {
                return new zzam();
            }
            if (list.size() % 2 == 0) {
                zzam zzamVar = new zzam();
                while (i < list.size() - 1) {
                    zzap zzb9 = zzgVar.zzb((zzap) list.get(i));
                    zzap zzb10 = zzgVar.zzb((zzap) list.get(i + 1));
                    if (!(zzb9 instanceof zzag) && !(zzb10 instanceof zzag)) {
                        zzamVar.zzr(zzb9.zzi(), zzb10);
                        i += 2;
                    } else {
                        throw new IllegalStateException("Failed to evaluate map entry");
                    }
                }
                return zzamVar;
            }
            throw new IllegalArgumentException(String.format("CREATE_OBJECT requires an even number of arguments, found %s", Integer.valueOf(list.size())));
        } else if (ordinal != 35 && ordinal != 36) {
            switch (ordinal) {
                case 62:
                    zzap zzb11 = zzgVar.zzb((zzap) a.f(zzbl.TYPEOF, 1, list, 0));
                    if (zzb11 instanceof zzau) {
                        str2 = "undefined";
                    } else if (zzb11 instanceof zzaf) {
                        str2 = "boolean";
                    } else if (zzb11 instanceof zzah) {
                        str2 = "number";
                    } else if (zzb11 instanceof zzat) {
                        str2 = "string";
                    } else if (zzb11 instanceof zzao) {
                        str2 = "function";
                    } else if ((zzb11 instanceof zzaq) || (zzb11 instanceof zzag)) {
                        throw new IllegalArgumentException(String.format("Unsupported value type %s in typeof", zzb11));
                    } else {
                        str2 = "object";
                    }
                    return new zzat(str2);
                case 63:
                    zzh.zzh(zzbl.UNDEFINED.name(), 0, list);
                    return zzap.zzf;
                case 64:
                    zzh.zzi(zzbl.VAR.name(), 1, list);
                    Iterator it2 = list.iterator();
                    while (it2.hasNext()) {
                        zzap zzb12 = zzgVar.zzb((zzap) it2.next());
                        if (zzb12 instanceof zzat) {
                            zzgVar.zze(zzb12.zzi(), zzap.zzf);
                        } else {
                            throw new IllegalArgumentException(String.format("Expected string for var name. got %s", zzb12.getClass().getCanonicalName()));
                        }
                    }
                    return zzap.zzf;
                default:
                    return super.zzb(str);
            }
        } else {
            zzap zzb13 = zzgVar.zzb((zzap) a.f(zzbl.GET_PROPERTY, 2, list, 0));
            zzap zzb14 = zzgVar.zzb((zzap) list.get(1));
            if ((zzb13 instanceof zzae) && zzh.zzk(zzb14)) {
                return ((zzae) zzb13).zze(zzb14.zzh().intValue());
            }
            if (zzb13 instanceof zzal) {
                return ((zzal) zzb13).zzf(zzb14.zzi());
            }
            if (zzb13 instanceof zzat) {
                if ("length".equals(zzb14.zzi())) {
                    return new zzah(Double.valueOf(zzb13.zzi().length()));
                }
                if (zzh.zzk(zzb14) && zzb14.zzh().doubleValue() < zzb13.zzi().length()) {
                    return new zzat(String.valueOf(zzb13.zzi().charAt(zzb14.zzh().intValue())));
                }
            }
            return zzap.zzf;
        }
    }
}