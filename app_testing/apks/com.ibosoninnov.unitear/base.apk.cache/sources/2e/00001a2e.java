package com.google.android.play.core.assetpacks;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.Map;

/* compiled from: com.google.android.play:core@@1.10.3 */
/* loaded from: classes.dex */
public final class zzdh {
    private static final com.google.android.play.core.internal.zzag zza = new com.google.android.play.core.internal.zzag("ExtractorTaskFinder");
    private final zzde zzb;
    private final zzbh zzc;
    private final zzbu zzd;
    private final com.google.android.play.core.common.zza zze;

    public zzdh(zzde zzdeVar, zzbh zzbhVar, zzbu zzbuVar, com.google.android.play.core.common.zza zzaVar) {
        this.zzb = zzdeVar;
        this.zzc = zzbhVar;
        this.zzd = zzbuVar;
        this.zze = zzaVar;
    }

    private final boolean zzb(zzdb zzdbVar, zzdc zzdcVar) {
        zzbh zzbhVar = this.zzc;
        zzda zzdaVar = zzdbVar.zzc;
        return new zzen(zzbhVar, zzdaVar.zza, zzdbVar.zzb, zzdaVar.zzb, zzdcVar.zza).zzm();
    }

    private static boolean zzc(zzdc zzdcVar) {
        int i = zzdcVar.zzf;
        return i == 1 || i == 2;
    }

    /* JADX WARN: Code restructure failed: missing block: B:27:0x00ae, code lost:
        if (r0 == null) goto L42;
     */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public final zzdg zza() {
        zzdg zzdgVar;
        zzde zzdeVar;
        zzde zzdeVar2;
        int i;
        try {
            this.zzb.zzj();
            ArrayList arrayList = new ArrayList();
            for (zzdb zzdbVar : this.zzb.zzg().values()) {
                if (zzbg.zzb(zzdbVar.zzc.zzd)) {
                    arrayList.add(zzdbVar);
                }
            }
            zzce zzceVar = null;
            if (!arrayList.isEmpty()) {
                if (this.zze.zza("assetOnlyUpdates")) {
                    Map zzt = this.zzc.zzt();
                    Iterator it = arrayList.iterator();
                    while (true) {
                        if (!it.hasNext()) {
                            zzdgVar = null;
                            break;
                        }
                        zzdb zzdbVar2 = (zzdb) it.next();
                        Long l = (Long) zzt.get(zzdbVar2.zzc.zza);
                        if (l != null && zzdbVar2.zzc.zzb == l.longValue()) {
                            zza.zza("Found promote pack task for session %s with pack %s.", Integer.valueOf(zzdbVar2.zza), zzdbVar2.zzc.zza);
                            int i2 = zzdbVar2.zza;
                            String str = zzdbVar2.zzc.zza;
                            zzdgVar = new zzei(i2, str, this.zzc.zza(str), zzdbVar2.zzb, zzdbVar2.zzc.zzb);
                            break;
                        }
                    }
                }
                Iterator it2 = arrayList.iterator();
                while (true) {
                    if (!it2.hasNext()) {
                        zzdgVar = null;
                        break;
                    }
                    zzdb zzdbVar3 = (zzdb) it2.next();
                    try {
                        zzbh zzbhVar = this.zzc;
                        zzda zzdaVar = zzdbVar3.zzc;
                        if (zzbhVar.zzb(zzdaVar.zza, zzdbVar3.zzb, zzdaVar.zzb) == zzdbVar3.zzc.zzf.size()) {
                            zza.zza("Found final move task for session %s with pack %s.", Integer.valueOf(zzdbVar3.zza), zzdbVar3.zzc.zza);
                            int i3 = zzdbVar3.zza;
                            zzda zzdaVar2 = zzdbVar3.zzc;
                            zzdgVar = new zzdw(i3, zzdaVar2.zza, zzdbVar3.zzb, zzdaVar2.zzb, zzdaVar2.zzc);
                            break;
                        }
                    } catch (IOException e2) {
                        throw new zzck(String.format("Failed to check number of completed merges for session %s, pack %s", Integer.valueOf(zzdbVar3.zza), zzdbVar3.zzc.zza), e2, zzdbVar3.zza);
                    }
                }
                if (zzdgVar == null) {
                    Iterator it3 = arrayList.iterator();
                    loop3: while (true) {
                        if (!it3.hasNext()) {
                            zzdgVar = null;
                            break;
                        }
                        zzdb zzdbVar4 = (zzdb) it3.next();
                        zzda zzdaVar3 = zzdbVar4.zzc;
                        if (zzbg.zzb(zzdaVar3.zzd)) {
                            for (zzdc zzdcVar : zzdaVar3.zzf) {
                                zzbh zzbhVar2 = this.zzc;
                                zzda zzdaVar4 = zzdbVar4.zzc;
                                if (zzbhVar2.zzq(zzdaVar4.zza, zzdbVar4.zzb, zzdaVar4.zzb, zzdcVar.zza).exists()) {
                                    zza.zza("Found merge task for session %s with pack %s and slice %s.", Integer.valueOf(zzdbVar4.zza), zzdbVar4.zzc.zza, zzdcVar.zza);
                                    int i4 = zzdbVar4.zza;
                                    zzda zzdaVar5 = zzdbVar4.zzc;
                                    zzdgVar = new zzdt(i4, zzdaVar5.zza, zzdbVar4.zzb, zzdaVar5.zzb, zzdcVar.zza);
                                    break loop3;
                                }
                            }
                            continue;
                        }
                    }
                    if (zzdgVar == null) {
                        Iterator it4 = arrayList.iterator();
                        loop5: while (true) {
                            if (!it4.hasNext()) {
                                zzdgVar = null;
                                break;
                            }
                            zzdb zzdbVar5 = (zzdb) it4.next();
                            zzda zzdaVar6 = zzdbVar5.zzc;
                            if (zzbg.zzb(zzdaVar6.zzd)) {
                                for (zzdc zzdcVar2 : zzdaVar6.zzf) {
                                    if (zzb(zzdbVar5, zzdcVar2)) {
                                        zzbh zzbhVar3 = this.zzc;
                                        zzda zzdaVar7 = zzdbVar5.zzc;
                                        if (zzbhVar3.zzp(zzdaVar7.zza, zzdbVar5.zzb, zzdaVar7.zzb, zzdcVar2.zza).exists()) {
                                            zza.zza("Found verify task for session %s with pack %s and slice %s.", Integer.valueOf(zzdbVar5.zza), zzdbVar5.zzc.zza, zzdcVar2.zza);
                                            int i5 = zzdbVar5.zza;
                                            zzda zzdaVar8 = zzdbVar5.zzc;
                                            zzdgVar = new zzeq(i5, zzdaVar8.zza, zzdbVar5.zzb, zzdaVar8.zzb, zzdcVar2.zza, zzdcVar2.zzb, zzdcVar2.zzc);
                                            break loop5;
                                        }
                                    }
                                }
                                continue;
                            }
                        }
                        if (zzdgVar == null) {
                            Iterator it5 = arrayList.iterator();
                            loop7: while (true) {
                                if (!it5.hasNext()) {
                                    zzceVar = null;
                                    break;
                                }
                                zzdb zzdbVar6 = (zzdb) it5.next();
                                zzda zzdaVar9 = zzdbVar6.zzc;
                                if (zzbg.zzb(zzdaVar9.zzd)) {
                                    Iterator it6 = zzdaVar9.zzf.iterator();
                                    while (it6.hasNext()) {
                                        zzdc zzdcVar3 = (zzdc) it6.next();
                                        if (!zzc(zzdcVar3)) {
                                            zzbh zzbhVar4 = this.zzc;
                                            zzda zzdaVar10 = zzdbVar6.zzc;
                                            Iterator it7 = it6;
                                            try {
                                                i = new zzen(zzbhVar4, zzdaVar10.zza, zzdbVar6.zzb, zzdaVar10.zzb, zzdcVar3.zza).zza();
                                            } catch (IOException e3) {
                                                zza.zzb("Slice checkpoint corrupt, restarting extraction. %s", e3);
                                                i = 0;
                                            }
                                            if (i != -1 && ((zzcz) zzdcVar3.zzd.get(i)).zza) {
                                                zza.zza("Found extraction task using compression format %s for session %s, pack %s, slice %s, chunk %s.", Integer.valueOf(zzdcVar3.zze), Integer.valueOf(zzdbVar6.zza), zzdbVar6.zzc.zza, zzdcVar3.zza, Integer.valueOf(i));
                                                InputStream zza2 = this.zzd.zza(zzdbVar6.zza, zzdbVar6.zzc.zza, zzdcVar3.zza, i);
                                                int i6 = zzdbVar6.zza;
                                                zzda zzdaVar11 = zzdbVar6.zzc;
                                                String str2 = zzdaVar11.zza;
                                                int i7 = zzdbVar6.zzb;
                                                long j = zzdaVar11.zzb;
                                                String str3 = zzdaVar11.zzc;
                                                String str4 = zzdcVar3.zza;
                                                int i8 = zzdcVar3.zze;
                                                int size = zzdcVar3.zzd.size();
                                                zzda zzdaVar12 = zzdbVar6.zzc;
                                                zzceVar = new zzce(i6, str2, i7, j, str3, str4, i8, i, size, zzdaVar12.zze, zzdaVar12.zzd, zza2);
                                                break loop7;
                                            }
                                            it6 = it7;
                                        }
                                    }
                                    continue;
                                }
                            }
                            if (zzceVar == null) {
                                Iterator it8 = arrayList.iterator();
                                loop9: while (true) {
                                    if (!it8.hasNext()) {
                                        zzdgVar = null;
                                        break;
                                    }
                                    zzdb zzdbVar7 = (zzdb) it8.next();
                                    zzda zzdaVar13 = zzdbVar7.zzc;
                                    if (zzbg.zzb(zzdaVar13.zzd)) {
                                        for (zzdc zzdcVar4 : zzdaVar13.zzf) {
                                            if (zzc(zzdcVar4) && ((zzcz) zzdcVar4.zzd.get(0)).zza && !zzb(zzdbVar7, zzdcVar4)) {
                                                zza.zza("Found patch slice task using patch format %s for session %s, pack %s, slice %s.", Integer.valueOf(zzdcVar4.zzf), Integer.valueOf(zzdbVar7.zza), zzdbVar7.zzc.zza, zzdcVar4.zza);
                                                InputStream zza3 = this.zzd.zza(zzdbVar7.zza, zzdbVar7.zzc.zza, zzdcVar4.zza, 0);
                                                int i9 = zzdbVar7.zza;
                                                String str5 = zzdbVar7.zzc.zza;
                                                zzdgVar = new zzef(i9, str5, this.zzc.zza(str5), this.zzc.zzc(zzdbVar7.zzc.zza), zzdbVar7.zzb, zzdbVar7.zzc.zzb, zzdcVar4.zzf, zzdcVar4.zza, zzdcVar4.zzc, zza3);
                                                break loop9;
                                            }
                                        }
                                        continue;
                                    }
                                }
                                if (zzdgVar == null) {
                                    this.zzb.zzl();
                                    return null;
                                }
                                zzdeVar2 = this.zzb;
                                zzdeVar2.zzl();
                                return zzdgVar;
                            }
                            zzdeVar = this.zzb;
                        }
                    }
                }
                zzdeVar2 = this.zzb;
                zzdeVar2.zzl();
                return zzdgVar;
            }
            zzdeVar = this.zzb;
            zzdeVar.zzl();
            return zzceVar;
        } catch (Throwable th) {
            this.zzb.zzl();
            throw th;
        }
    }
}