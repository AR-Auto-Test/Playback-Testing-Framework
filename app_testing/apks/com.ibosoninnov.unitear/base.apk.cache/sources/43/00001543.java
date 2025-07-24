package com.google.android.gms.measurement.internal;

import android.net.Uri;
import android.os.Bundle;
import android.text.TextUtils;

/* compiled from: com.google.android.gms:play-services-measurement-impl@@21.2.0 */
/* loaded from: classes.dex */
public final class zzhu implements Runnable {
    public final /* synthetic */ boolean zza;
    public final /* synthetic */ Uri zzb;
    public final /* synthetic */ String zzc;
    public final /* synthetic */ String zzd;
    public final /* synthetic */ zzhw zze;

    public zzhu(zzhw zzhwVar, boolean z, Uri uri, String str, String str2) {
        this.zze = zzhwVar;
        this.zza = z;
        this.zzb = uri;
        this.zzc = str;
        this.zzd = str2;
    }

    /* JADX WARN: Removed duplicated region for block: B:38:0x00cd  */
    /* JADX WARN: Removed duplicated region for block: B:39:0x00cf A[Catch: RuntimeException -> 0x0160, TRY_LEAVE, TryCatch #0 {RuntimeException -> 0x0160, blocks: (B:3:0x0011, B:27:0x0086, B:29:0x0094, B:32:0x00a1, B:34:0x00a7, B:35:0x00bb, B:36:0x00c7, B:39:0x00cf, B:43:0x00f6, B:45:0x0114, B:44:0x0103, B:47:0x011b, B:49:0x0121, B:51:0x0127, B:53:0x012d, B:55:0x0133, B:57:0x013b, B:59:0x0143, B:61:0x0149, B:63:0x0150, B:7:0x002e, B:9:0x0034, B:11:0x003a, B:13:0x0040, B:15:0x0046, B:17:0x004e, B:19:0x0056, B:21:0x005e, B:22:0x006c, B:24:0x007c), top: B:68:0x0011 }] */
    @Override // java.lang.Runnable
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public final void run() {
        Bundle zzs;
        Bundle zzs2;
        zzhw zzhwVar = this.zze;
        boolean z = this.zza;
        Uri uri = this.zzb;
        String str = this.zzc;
        String str2 = this.zzd;
        zzhwVar.zza.zzg();
        try {
            zzlb zzv = zzhwVar.zza.zzt.zzv();
            if (!TextUtils.isEmpty(str2)) {
                if (!str2.contains("gclid") && !str2.contains("utm_campaign") && !str2.contains("utm_source") && !str2.contains("utm_medium") && !str2.contains("utm_id") && !str2.contains("dclid") && !str2.contains("srsltid")) {
                    zzv.zzt.zzay().zzc().zza("Activity created with data 'referrer' without required params");
                } else {
                    zzs = zzv.zzs(Uri.parse("https://google.com/search?".concat(str2)));
                    if (zzs != null) {
                        zzs.putString("_cis", "referrer");
                    }
                    if (z && (zzs2 = zzhwVar.zza.zzt.zzv().zzs(uri)) != null) {
                        zzs2.putString("_cis", "intent");
                        if (!zzs2.containsKey("gclid") && zzs != null && zzs.containsKey("gclid")) {
                            zzs2.putString("_cer", String.format("gclid=%s", zzs.getString("gclid")));
                        }
                        zzhwVar.zza.zzG(str, "_cmp", zzs2);
                        zzhwVar.zza.zzb.zza(str, zzs2);
                    }
                    if (TextUtils.isEmpty(str2)) {
                        zzhwVar.zza.zzt.zzay().zzc().zzb("Activity created with referrer", str2);
                        if (zzhwVar.zza.zzt.zzf().zzs(null, zzdu.zzY)) {
                            if (zzs != null) {
                                zzhwVar.zza.zzG(str, "_cmp", zzs);
                                zzhwVar.zza.zzb.zza(str, zzs);
                            } else {
                                zzhwVar.zza.zzt.zzay().zzc().zzb("Referrer does not contain valid parameters", str2);
                            }
                            zzhwVar.zza.zzW("auto", "_ldl", null, true);
                            return;
                        } else if (str2.contains("gclid") && (str2.contains("utm_campaign") || str2.contains("utm_source") || str2.contains("utm_medium") || str2.contains("utm_term") || str2.contains("utm_content"))) {
                            if (TextUtils.isEmpty(str2)) {
                                return;
                            }
                            zzhwVar.zza.zzW("auto", "_ldl", str2, true);
                            return;
                        } else {
                            zzhwVar.zza.zzt.zzay().zzc().zza("Activity created with data 'referrer' without required params");
                            return;
                        }
                    }
                    return;
                }
            }
            zzs = null;
            if (z) {
                zzs2.putString("_cis", "intent");
                if (!zzs2.containsKey("gclid")) {
                    zzs2.putString("_cer", String.format("gclid=%s", zzs.getString("gclid")));
                }
                zzhwVar.zza.zzG(str, "_cmp", zzs2);
                zzhwVar.zza.zzb.zza(str, zzs2);
            }
            if (TextUtils.isEmpty(str2)) {
            }
        } catch (RuntimeException e2) {
            zzhwVar.zza.zzt.zzay().zzd().zzb("Throwable caught in handleReferrerForOnActivityCreated", e2);
        }
    }
}