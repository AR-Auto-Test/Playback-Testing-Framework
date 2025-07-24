package com.google.android.gms.measurement.internal;

import android.util.Log;
import b.f.a;
import com.google.android.gms.internal.measurement.zznz;
import java.util.HashSet;
import java.util.Iterator;

/* compiled from: com.google.android.gms:play-services-measurement@@21.2.0 */
/* loaded from: classes.dex */
public final class zzx extends zzy {
    public final /* synthetic */ zzaa zza;
    private final com.google.android.gms.internal.measurement.zzek zzh;

    /* JADX WARN: 'super' call moved to the top of the method (can break code semantics) */
    public zzx(zzaa zzaaVar, String str, int i, com.google.android.gms.internal.measurement.zzek zzekVar) {
        super(str, i);
        this.zza = zzaaVar;
        this.zzh = zzekVar;
    }

    @Override // com.google.android.gms.measurement.internal.zzy
    public final int zza() {
        return this.zzh.zzb();
    }

    @Override // com.google.android.gms.measurement.internal.zzy
    public final boolean zzb() {
        return this.zzh.zzo();
    }

    @Override // com.google.android.gms.measurement.internal.zzy
    public final boolean zzc() {
        return false;
    }

    /* JADX WARN: Removed duplicated region for block: B:127:0x03ef  */
    /* JADX WARN: Removed duplicated region for block: B:128:0x03f2  */
    /* JADX WARN: Removed duplicated region for block: B:131:0x03fa A[RETURN] */
    /* JADX WARN: Removed duplicated region for block: B:132:0x03fb  */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public final boolean zzd(Long l, Long l2, com.google.android.gms.internal.measurement.zzft zzftVar, long j, zzas zzasVar, boolean z) {
        Boolean zzi;
        zznz.zzc();
        boolean zzs = this.zza.zzt.zzf().zzs(this.zzb, zzdu.zzW);
        long j2 = this.zzh.zzn() ? zzasVar.zze : j;
        r5 = null;
        r5 = null;
        r5 = null;
        r5 = null;
        r5 = null;
        r5 = null;
        r5 = null;
        r5 = null;
        r5 = null;
        r5 = null;
        r5 = null;
        Boolean bool = null;
        if (Log.isLoggable(this.zza.zzt.zzay().zzq(), 2)) {
            this.zza.zzt.zzay().zzj().zzd("Evaluating filter. audience, filter, event", Integer.valueOf(this.zzc), this.zzh.zzp() ? Integer.valueOf(this.zzh.zzb()) : null, this.zza.zzt.zzj().zzd(this.zzh.zzg()));
            this.zza.zzt.zzay().zzj().zzb("Filter definition", this.zza.zzf.zzu().zzo(this.zzh));
        }
        if (!this.zzh.zzp() || this.zzh.zzb() > 256) {
            this.zza.zzt.zzay().zzk().zzc("Invalid event filter ID. appId, id", zzeh.zzn(this.zzb), String.valueOf(this.zzh.zzp() ? Integer.valueOf(this.zzh.zzb()) : null));
            return false;
        }
        byte b2 = (this.zzh.zzk() || this.zzh.zzm() || this.zzh.zzn()) ? (byte) 1 : (byte) 0;
        if (z && b2 == 0) {
            this.zza.zzt.zzay().zzj().zzc("Event filter already evaluated true and it is not associated with an enhanced audience. audience ID, filter ID", Integer.valueOf(this.zzc), this.zzh.zzp() ? Integer.valueOf(this.zzh.zzb()) : null);
            return true;
        }
        com.google.android.gms.internal.measurement.zzek zzekVar = this.zzh;
        String zzh = zzftVar.zzh();
        if (zzekVar.zzo()) {
            Boolean zzh2 = zzy.zzh(j2, zzekVar.zzf());
            if (zzh2 != null) {
                if (!zzh2.booleanValue()) {
                    bool = Boolean.FALSE;
                }
            }
            this.zza.zzt.zzay().zzj().zzb("Event filter result", bool != null ? "null" : bool);
            if (bool != null) {
                return false;
            }
            Boolean bool2 = Boolean.TRUE;
            this.zzd = bool2;
            if (bool.booleanValue()) {
                this.zze = bool2;
                if (b2 != 0 && zzftVar.zzu()) {
                    Long valueOf = Long.valueOf(zzftVar.zzd());
                    if (this.zzh.zzm()) {
                        if (zzs && this.zzh.zzo()) {
                            valueOf = l;
                        }
                        this.zzg = valueOf;
                    } else {
                        if (zzs && this.zzh.zzo()) {
                            valueOf = l2;
                        }
                        this.zzf = valueOf;
                    }
                }
                return true;
            }
            return true;
        }
        HashSet hashSet = new HashSet();
        Iterator it = zzekVar.zzh().iterator();
        while (true) {
            if (it.hasNext()) {
                com.google.android.gms.internal.measurement.zzem zzemVar = (com.google.android.gms.internal.measurement.zzem) it.next();
                if (zzemVar.zze().isEmpty()) {
                    this.zza.zzt.zzay().zzk().zzb("null or empty param name in filter. event", this.zza.zzt.zzj().zzd(zzh));
                    break;
                }
                hashSet.add(zzemVar.zze());
            } else {
                a aVar = new a();
                Iterator it2 = zzftVar.zzi().iterator();
                while (true) {
                    if (it2.hasNext()) {
                        com.google.android.gms.internal.measurement.zzfx zzfxVar = (com.google.android.gms.internal.measurement.zzfx) it2.next();
                        if (hashSet.contains(zzfxVar.zzg())) {
                            if (zzfxVar.zzw()) {
                                aVar.put(zzfxVar.zzg(), zzfxVar.zzw() ? Long.valueOf(zzfxVar.zzd()) : null);
                            } else if (zzfxVar.zzu()) {
                                aVar.put(zzfxVar.zzg(), zzfxVar.zzu() ? Double.valueOf(zzfxVar.zza()) : null);
                            } else if (zzfxVar.zzy()) {
                                aVar.put(zzfxVar.zzg(), zzfxVar.zzh());
                            } else {
                                this.zza.zzt.zzay().zzk().zzc("Unknown value for param. event, param", this.zza.zzt.zzj().zzd(zzh), this.zza.zzt.zzj().zze(zzfxVar.zzg()));
                                break;
                            }
                        }
                    } else {
                        Iterator it3 = zzekVar.zzh().iterator();
                        while (true) {
                            if (it3.hasNext()) {
                                com.google.android.gms.internal.measurement.zzem zzemVar2 = (com.google.android.gms.internal.measurement.zzem) it3.next();
                                boolean z2 = zzemVar2.zzh() && zzemVar2.zzg();
                                String zze = zzemVar2.zze();
                                if (zze.isEmpty()) {
                                    this.zza.zzt.zzay().zzk().zzb("Event has empty param name. event", this.zza.zzt.zzj().zzd(zzh));
                                    break;
                                }
                                Object obj = aVar.get(zze);
                                if (obj instanceof Long) {
                                    if (!zzemVar2.zzi()) {
                                        this.zza.zzt.zzay().zzk().zzc("No number filter for long param. event, param", this.zza.zzt.zzj().zzd(zzh), this.zza.zzt.zzj().zze(zze));
                                        break;
                                    }
                                    Boolean zzh3 = zzy.zzh(((Long) obj).longValue(), zzemVar2.zzc());
                                    if (zzh3 == null) {
                                        break;
                                    } else if (zzh3.booleanValue() == z2) {
                                        bool = Boolean.FALSE;
                                        break;
                                    }
                                } else if (obj instanceof Double) {
                                    if (!zzemVar2.zzi()) {
                                        this.zza.zzt.zzay().zzk().zzc("No number filter for double param. event, param", this.zza.zzt.zzj().zzd(zzh), this.zza.zzt.zzj().zze(zze));
                                        break;
                                    }
                                    Boolean zzg = zzy.zzg(((Double) obj).doubleValue(), zzemVar2.zzc());
                                    if (zzg == null) {
                                        break;
                                    } else if (zzg.booleanValue() == z2) {
                                        bool = Boolean.FALSE;
                                        break;
                                    }
                                } else if (obj instanceof String) {
                                    if (zzemVar2.zzk()) {
                                        zzi = zzy.zzf((String) obj, zzemVar2.zzd(), this.zza.zzt.zzay());
                                    } else if (zzemVar2.zzi()) {
                                        String str = (String) obj;
                                        if (zzkv.zzx(str)) {
                                            zzi = zzy.zzi(str, zzemVar2.zzc());
                                        } else {
                                            this.zza.zzt.zzay().zzk().zzc("Invalid param value for number filter. event, param", this.zza.zzt.zzj().zzd(zzh), this.zza.zzt.zzj().zze(zze));
                                            break;
                                        }
                                    } else {
                                        this.zza.zzt.zzay().zzk().zzc("No filter for String param. event, param", this.zza.zzt.zzj().zzd(zzh), this.zza.zzt.zzj().zze(zze));
                                        break;
                                    }
                                    if (zzi == null) {
                                        break;
                                    } else if (zzi.booleanValue() == z2) {
                                        bool = Boolean.FALSE;
                                        break;
                                    }
                                } else if (obj == null) {
                                    this.zza.zzt.zzay().zzj().zzc("Missing param for filter. event, param", this.zza.zzt.zzj().zzd(zzh), this.zza.zzt.zzj().zze(zze));
                                    bool = Boolean.FALSE;
                                } else {
                                    this.zza.zzt.zzay().zzk().zzc("Unknown param type. event, param", this.zza.zzt.zzj().zzd(zzh), this.zza.zzt.zzj().zze(zze));
                                }
                            } else {
                                bool = Boolean.TRUE;
                                break;
                            }
                        }
                    }
                }
            }
        }
        this.zza.zzt.zzay().zzj().zzb("Event filter result", bool != null ? "null" : bool);
        if (bool != null) {
        }
    }
}