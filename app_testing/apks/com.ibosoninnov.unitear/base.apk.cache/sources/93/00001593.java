package com.google.android.gms.measurement.internal;

import android.content.ComponentName;
import android.content.ContentValues;
import android.content.Context;
import android.content.Intent;
import android.content.pm.ApplicationInfo;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager;
import android.content.pm.ResolveInfo;
import android.content.pm.ServiceInfo;
import android.database.Cursor;
import android.database.sqlite.SQLiteDatabase;
import android.database.sqlite.SQLiteException;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Parcelable;
import android.text.TextUtils;
import android.util.Log;
import android.util.Pair;
import androidx.recyclerview.widget.RecyclerView;
import b.f.a;
import com.google.android.gms.common.internal.Preconditions;
import com.google.android.gms.common.stats.ConnectionTracker;
import com.google.android.gms.common.util.Clock;
import com.google.android.gms.common.util.VisibleForTesting;
import com.google.android.gms.common.wrappers.Wrappers;
import com.google.android.gms.internal.measurement.zznt;
import com.google.android.gms.internal.measurement.zzoi;
import com.google.android.gms.internal.measurement.zzox;
import com.google.android.gms.internal.measurement.zzpd;
import com.google.common.flogger.parser.MessageParser;
import com.google.common.net.HttpHeaders;
import com.google.firebase.analytics.FirebaseAnalytics;
import com.google.firebase.crashlytics.CrashlyticsAnalyticsListener;
import com.google.firebase.crashlytics.internal.common.CrashlyticsReportDataCapture;
import com.google.firebase.crashlytics.internal.settings.DefaultSettingsSpiCall;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.math.BigInteger;
import java.net.MalformedURLException;
import java.net.URL;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.channels.FileLock;
import java.nio.channels.OverlappingFileLockException;
import java.security.SecureRandom;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.zip.GZIPInputStream;

/* compiled from: com.google.android.gms:play-services-measurement@@21.2.0 */
/* loaded from: classes.dex */
public final class zzkt implements zzgm {
    private static volatile zzkt zzb;
    private long zzA;
    private final Map zzB;
    private final Map zzC;
    private zzie zzD;
    private String zzE;
    @VisibleForTesting
    public long zza;
    private final zzfi zzc;
    private final zzen zzd;
    private zzam zze;
    private zzep zzf;
    private zzkf zzg;
    private zzaa zzh;
    private final zzkv zzi;
    private zzic zzj;
    private zzjo zzk;
    private final zzki zzl;
    private zzez zzm;
    private final zzfr zzn;
    private boolean zzp;
    private List zzq;
    private int zzr;
    private int zzs;
    private boolean zzt;
    private boolean zzu;
    private boolean zzv;
    private FileLock zzw;
    private FileChannel zzx;
    private List zzy;
    private List zzz;
    private boolean zzo = false;
    private final zzla zzF = new zzko(this);

    public zzkt(zzku zzkuVar, zzfr zzfrVar) {
        Preconditions.checkNotNull(zzkuVar);
        this.zzn = zzfr.zzp(zzkuVar.zza, null, null);
        this.zzA = -1L;
        this.zzl = new zzki(this);
        zzkv zzkvVar = new zzkv(this);
        zzkvVar.zzX();
        this.zzi = zzkvVar;
        zzen zzenVar = new zzen(this);
        zzenVar.zzX();
        this.zzd = zzenVar;
        zzfi zzfiVar = new zzfi(this);
        zzfiVar.zzX();
        this.zzc = zzfiVar;
        this.zzB = new HashMap();
        this.zzC = new HashMap();
        zzaz().zzp(new zzkj(this, zzkuVar));
    }

    @VisibleForTesting
    public static final void zzaa(com.google.android.gms.internal.measurement.zzfs zzfsVar, int i, String str) {
        List zzp = zzfsVar.zzp();
        for (int i2 = 0; i2 < zzp.size(); i2++) {
            if ("_err".equals(((com.google.android.gms.internal.measurement.zzfx) zzp.get(i2)).zzg())) {
                return;
            }
        }
        com.google.android.gms.internal.measurement.zzfw zze = com.google.android.gms.internal.measurement.zzfx.zze();
        zze.zzj("_err");
        zze.zzi(Long.valueOf(i).longValue());
        com.google.android.gms.internal.measurement.zzfw zze2 = com.google.android.gms.internal.measurement.zzfx.zze();
        zze2.zzj("_ev");
        zze2.zzk(str);
        zzfsVar.zzf((com.google.android.gms.internal.measurement.zzfx) zze.zzaC());
        zzfsVar.zzf((com.google.android.gms.internal.measurement.zzfx) zze2.zzaC());
    }

    @VisibleForTesting
    public static final void zzab(com.google.android.gms.internal.measurement.zzfs zzfsVar, String str) {
        List zzp = zzfsVar.zzp();
        for (int i = 0; i < zzp.size(); i++) {
            if (str.equals(((com.google.android.gms.internal.measurement.zzfx) zzp.get(i)).zzg())) {
                zzfsVar.zzh(i);
                return;
            }
        }
    }

    private final zzq zzac(String str) {
        zzam zzamVar = this.zze;
        zzal(zzamVar);
        zzh zzj = zzamVar.zzj(str);
        if (zzj != null && !TextUtils.isEmpty(zzj.zzw())) {
            Boolean zzad = zzad(zzj);
            if (zzad != null && !zzad.booleanValue()) {
                zzay().zzd().zzb("App version does not match; dropping. appId", zzeh.zzn(str));
                return null;
            }
            String zzy = zzj.zzy();
            String zzw = zzj.zzw();
            long zzb2 = zzj.zzb();
            String zzv = zzj.zzv();
            long zzm = zzj.zzm();
            long zzj2 = zzj.zzj();
            boolean zzai = zzj.zzai();
            String zzx = zzj.zzx();
            zzj.zza();
            return new zzq(str, zzy, zzw, zzb2, zzv, zzm, zzj2, (String) null, zzai, false, zzx, 0L, 0L, 0, zzj.zzah(), false, zzj.zzr(), zzj.zzq(), zzj.zzk(), zzj.zzC(), (String) null, zzh(str).zzh(), "", (String) null);
        }
        zzay().zzc().zzb("No app data available; dropping", str);
        return null;
    }

    private final Boolean zzad(zzh zzhVar) {
        try {
            if (zzhVar.zzb() != -2147483648L) {
                if (zzhVar.zzb() == Wrappers.packageManager(this.zzn.zzau()).getPackageInfo(zzhVar.zzt(), 0).versionCode) {
                    return Boolean.TRUE;
                }
            } else {
                String str = Wrappers.packageManager(this.zzn.zzau()).getPackageInfo(zzhVar.zzt(), 0).versionName;
                String zzw = zzhVar.zzw();
                if (zzw != null && zzw.equals(str)) {
                    return Boolean.TRUE;
                }
            }
            return Boolean.FALSE;
        } catch (PackageManager.NameNotFoundException unused) {
            return null;
        }
    }

    private final void zzae() {
        zzaz().zzg();
        if (!this.zzt && !this.zzu && !this.zzv) {
            zzay().zzj().zza("Stopping uploading service(s)");
            List<Runnable> list = this.zzq;
            if (list == null) {
                return;
            }
            for (Runnable runnable : list) {
                runnable.run();
            }
            ((List) Preconditions.checkNotNull(this.zzq)).clear();
            return;
        }
        zzay().zzj().zzd("Not stopping services. fetch, network, upload", Boolean.valueOf(this.zzt), Boolean.valueOf(this.zzu), Boolean.valueOf(this.zzv));
    }

    @VisibleForTesting
    private final void zzaf(com.google.android.gms.internal.measurement.zzgc zzgcVar, long j, boolean z) {
        zzky zzkyVar;
        String str = true != z ? "_lte" : "_se";
        zzam zzamVar = this.zze;
        zzal(zzamVar);
        zzky zzp = zzamVar.zzp(zzgcVar.zzap(), str);
        if (zzp != null && zzp.zze != null) {
            zzkyVar = new zzky(zzgcVar.zzap(), "auto", str, zzav().currentTimeMillis(), Long.valueOf(((Long) zzp.zze).longValue() + j));
        } else {
            zzkyVar = new zzky(zzgcVar.zzap(), "auto", str, zzav().currentTimeMillis(), Long.valueOf(j));
        }
        com.google.android.gms.internal.measurement.zzgl zzd = com.google.android.gms.internal.measurement.zzgm.zzd();
        zzd.zzf(str);
        zzd.zzg(zzav().currentTimeMillis());
        zzd.zze(((Long) zzkyVar.zze).longValue());
        com.google.android.gms.internal.measurement.zzgm zzgmVar = (com.google.android.gms.internal.measurement.zzgm) zzd.zzaC();
        int zza = zzkv.zza(zzgcVar, str);
        if (zza >= 0) {
            zzgcVar.zzam(zza, zzgmVar);
        } else {
            zzgcVar.zzm(zzgmVar);
        }
        if (j > 0) {
            zzam zzamVar2 = this.zze;
            zzal(zzamVar2);
            zzamVar2.zzL(zzkyVar);
            zzay().zzj().zzc("Updated engagement user property. scope, value", true != z ? "lifetime" : "session-scoped", zzkyVar.zze);
        }
    }

    /* JADX WARN: Removed duplicated region for block: B:51:0x0192  */
    /* JADX WARN: Removed duplicated region for block: B:63:0x0237  */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    private final void zzag() {
        long max;
        long j;
        zzaz().zzg();
        zzB();
        if (this.zza > 0) {
            long abs = 3600000 - Math.abs(zzav().elapsedRealtime() - this.zza);
            if (abs > 0) {
                zzay().zzj().zzb("Upload has been suspended. Will update scheduling later in approximately ms", Long.valueOf(abs));
                zzm().zzc();
                zzkf zzkfVar = this.zzg;
                zzal(zzkfVar);
                zzkfVar.zza();
                return;
            }
            this.zza = 0L;
        }
        if (this.zzn.zzM() && zzai()) {
            long currentTimeMillis = zzav().currentTimeMillis();
            zzg();
            long max2 = Math.max(0L, ((Long) zzdu.zzz.zza(null)).longValue());
            zzam zzamVar = this.zze;
            zzal(zzamVar);
            boolean z = true;
            if (!zzamVar.zzH()) {
                zzam zzamVar2 = this.zze;
                zzal(zzamVar2);
                if (!zzamVar2.zzG()) {
                    z = false;
                }
            }
            if (z) {
                String zzl = zzg().zzl();
                if (!TextUtils.isEmpty(zzl) && !".none.".equals(zzl)) {
                    zzg();
                    max = Math.max(0L, ((Long) zzdu.zzu.zza(null)).longValue());
                } else {
                    zzg();
                    max = Math.max(0L, ((Long) zzdu.zzt.zza(null)).longValue());
                }
            } else {
                zzg();
                max = Math.max(0L, ((Long) zzdu.zzs.zza(null)).longValue());
            }
            long zza = this.zzk.zzc.zza();
            long zza2 = this.zzk.zzd.zza();
            zzam zzamVar3 = this.zze;
            zzal(zzamVar3);
            boolean z2 = z;
            long zzd = zzamVar3.zzd();
            zzam zzamVar4 = this.zze;
            zzal(zzamVar4);
            long max3 = Math.max(zzd, zzamVar4.zze());
            if (max3 != 0) {
                long abs2 = currentTimeMillis - Math.abs(max3 - currentTimeMillis);
                long abs3 = Math.abs(zza - currentTimeMillis);
                long abs4 = currentTimeMillis - Math.abs(zza2 - currentTimeMillis);
                long max4 = Math.max(currentTimeMillis - abs3, abs4);
                j = abs2 + max2;
                if (z2 && max4 > 0) {
                    j = Math.min(abs2, max4) + max;
                }
                zzkv zzkvVar = this.zzi;
                zzal(zzkvVar);
                if (!zzkvVar.zzw(max4, max)) {
                    j = max4 + max;
                }
                if (abs4 != 0 && abs4 >= abs2) {
                    int i = 0;
                    while (true) {
                        zzg();
                        if (i >= Math.min(20, Math.max(0, ((Integer) zzdu.zzB.zza(null)).intValue()))) {
                            break;
                        }
                        zzg();
                        j += Math.max(0L, ((Long) zzdu.zzA.zza(null)).longValue()) * (1 << i);
                        if (j > abs4) {
                            break;
                        }
                        i++;
                    }
                }
                if (j == 0) {
                    zzen zzenVar = this.zzd;
                    zzal(zzenVar);
                    if (zzenVar.zza()) {
                        long zza3 = this.zzk.zzb.zza();
                        zzg();
                        long max5 = Math.max(0L, ((Long) zzdu.zzq.zza(null)).longValue());
                        zzkv zzkvVar2 = this.zzi;
                        zzal(zzkvVar2);
                        if (!zzkvVar2.zzw(zza3, max5)) {
                            j = Math.max(j, zza3 + max5);
                        }
                        zzm().zzc();
                        long currentTimeMillis2 = j - zzav().currentTimeMillis();
                        if (currentTimeMillis2 <= 0) {
                            zzg();
                            currentTimeMillis2 = Math.max(0L, ((Long) zzdu.zzv.zza(null)).longValue());
                            this.zzk.zzc.zzb(zzav().currentTimeMillis());
                        }
                        zzay().zzj().zzb("Upload scheduled in approximately ms", Long.valueOf(currentTimeMillis2));
                        zzkf zzkfVar2 = this.zzg;
                        zzal(zzkfVar2);
                        zzkfVar2.zzd(currentTimeMillis2);
                        return;
                    }
                    zzay().zzj().zza("No network");
                    zzm().zzb();
                    zzkf zzkfVar3 = this.zzg;
                    zzal(zzkfVar3);
                    zzkfVar3.zza();
                    return;
                }
                zzay().zzj().zza("Next upload time is 0");
                zzm().zzc();
                zzkf zzkfVar4 = this.zzg;
                zzal(zzkfVar4);
                zzkfVar4.zza();
                return;
            }
            j = 0;
            if (j == 0) {
            }
        } else {
            zzay().zzj().zza("Nothing to upload or uploading impossible");
            zzm().zzc();
            zzkf zzkfVar5 = this.zzg;
            zzal(zzkfVar5);
            zzkfVar5.zza();
        }
    }

    /* JADX WARN: Code restructure failed: missing block: B:373:0x0b7e, code lost:
        if (r10 > (com.google.android.gms.measurement.internal.zzag.zzA() + r8)) goto L404;
     */
    /* JADX WARN: Removed duplicated region for block: B:111:0x03a7 A[Catch: all -> 0x0d17, TryCatch #3 {all -> 0x0d17, blocks: (B:3:0x000e, B:5:0x0026, B:8:0x002e, B:9:0x0040, B:12:0x0054, B:15:0x007b, B:17:0x00b1, B:20:0x00c3, B:22:0x00cd, B:173:0x0538, B:24:0x00f3, B:26:0x0101, B:29:0x0125, B:31:0x012b, B:33:0x013d, B:35:0x014b, B:37:0x015b, B:38:0x0168, B:39:0x016d, B:42:0x0186, B:111:0x03a7, B:112:0x03b3, B:115:0x03bd, B:121:0x03e0, B:118:0x03cf, B:143:0x045f, B:145:0x046b, B:148:0x047e, B:150:0x048f, B:152:0x049b, B:172:0x0524, B:157:0x04c5, B:159:0x04d5, B:162:0x04ea, B:164:0x04fb, B:166:0x0507, B:125:0x03e8, B:127:0x03f4, B:129:0x0400, B:141:0x0445, B:133:0x041d, B:136:0x042f, B:138:0x0435, B:140:0x043f, B:68:0x01e4, B:71:0x01ee, B:73:0x01fc, B:77:0x0243, B:74:0x0219, B:76:0x022a, B:81:0x0252, B:83:0x027e, B:84:0x02a8, B:86:0x02de, B:88:0x02e4, B:91:0x02f0, B:93:0x0326, B:94:0x0341, B:96:0x0347, B:98:0x0355, B:102:0x0368, B:99:0x035d, B:105:0x036f, B:108:0x0376, B:109:0x038e, B:176:0x054d, B:178:0x055b, B:180:0x0566, B:191:0x0598, B:181:0x056e, B:183:0x0579, B:185:0x057f, B:188:0x058b, B:190:0x0593, B:192:0x059b, B:193:0x05a7, B:196:0x05af, B:198:0x05c1, B:199:0x05cd, B:201:0x05d5, B:205:0x05fa, B:207:0x061f, B:209:0x0630, B:211:0x0636, B:213:0x0642, B:214:0x0673, B:216:0x0679, B:218:0x0687, B:219:0x068b, B:220:0x068e, B:221:0x0691, B:222:0x069f, B:224:0x06a5, B:226:0x06b5, B:227:0x06bc, B:229:0x06c8, B:230:0x06cf, B:231:0x06d2, B:233:0x0712, B:234:0x0725, B:236:0x072b, B:239:0x0745, B:241:0x0760, B:243:0x0779, B:245:0x077e, B:247:0x0782, B:249:0x0786, B:251:0x0790, B:252:0x079a, B:254:0x079e, B:256:0x07a4, B:257:0x07b2, B:258:0x07bb, B:326:0x0a0b, B:260:0x07c8, B:262:0x07df, B:268:0x07fb, B:270:0x081f, B:271:0x0827, B:273:0x082d, B:275:0x083f, B:282:0x0868, B:283:0x088b, B:285:0x0897, B:287:0x08ac, B:289:0x08ed, B:293:0x0905, B:295:0x090c, B:297:0x091b, B:299:0x091f, B:301:0x0923, B:303:0x0927, B:304:0x0933, B:305:0x0938, B:307:0x093e, B:309:0x095a, B:310:0x095f, B:325:0x0a08, B:311:0x097a, B:313:0x0982, B:317:0x09a9, B:319:0x09d5, B:320:0x09dc, B:321:0x09ee, B:323:0x09f8, B:314:0x098f, B:280:0x0853, B:266:0x07e6, B:327:0x0a17, B:329:0x0a25, B:330:0x0a2b, B:331:0x0a33, B:333:0x0a39, B:336:0x0a53, B:338:0x0a64, B:358:0x0ad8, B:360:0x0ade, B:362:0x0af6, B:365:0x0afd, B:370:0x0b2c, B:372:0x0b6e, B:375:0x0ba3, B:376:0x0ba7, B:377:0x0bb2, B:379:0x0bf5, B:380:0x0c02, B:382:0x0c11, B:386:0x0c2b, B:388:0x0c44, B:374:0x0b80, B:366:0x0b05, B:368:0x0b11, B:369:0x0b15, B:389:0x0c5c, B:390:0x0c74, B:393:0x0c7c, B:394:0x0c81, B:395:0x0c91, B:397:0x0cab, B:398:0x0cc6, B:400:0x0cd0, B:405:0x0cf3, B:404:0x0ce0, B:339:0x0a7c, B:341:0x0a82, B:343:0x0a8c, B:345:0x0a93, B:351:0x0aa3, B:353:0x0aaa, B:355:0x0ac9, B:357:0x0ad0, B:356:0x0acd, B:352:0x0aa7, B:344:0x0a90, B:202:0x05da, B:204:0x05e0, B:408:0x0d05), top: B:420:0x000e, inners: #0, #1, #2, #4 }] */
    /* JADX WARN: Removed duplicated region for block: B:145:0x046b A[Catch: all -> 0x0d17, TryCatch #3 {all -> 0x0d17, blocks: (B:3:0x000e, B:5:0x0026, B:8:0x002e, B:9:0x0040, B:12:0x0054, B:15:0x007b, B:17:0x00b1, B:20:0x00c3, B:22:0x00cd, B:173:0x0538, B:24:0x00f3, B:26:0x0101, B:29:0x0125, B:31:0x012b, B:33:0x013d, B:35:0x014b, B:37:0x015b, B:38:0x0168, B:39:0x016d, B:42:0x0186, B:111:0x03a7, B:112:0x03b3, B:115:0x03bd, B:121:0x03e0, B:118:0x03cf, B:143:0x045f, B:145:0x046b, B:148:0x047e, B:150:0x048f, B:152:0x049b, B:172:0x0524, B:157:0x04c5, B:159:0x04d5, B:162:0x04ea, B:164:0x04fb, B:166:0x0507, B:125:0x03e8, B:127:0x03f4, B:129:0x0400, B:141:0x0445, B:133:0x041d, B:136:0x042f, B:138:0x0435, B:140:0x043f, B:68:0x01e4, B:71:0x01ee, B:73:0x01fc, B:77:0x0243, B:74:0x0219, B:76:0x022a, B:81:0x0252, B:83:0x027e, B:84:0x02a8, B:86:0x02de, B:88:0x02e4, B:91:0x02f0, B:93:0x0326, B:94:0x0341, B:96:0x0347, B:98:0x0355, B:102:0x0368, B:99:0x035d, B:105:0x036f, B:108:0x0376, B:109:0x038e, B:176:0x054d, B:178:0x055b, B:180:0x0566, B:191:0x0598, B:181:0x056e, B:183:0x0579, B:185:0x057f, B:188:0x058b, B:190:0x0593, B:192:0x059b, B:193:0x05a7, B:196:0x05af, B:198:0x05c1, B:199:0x05cd, B:201:0x05d5, B:205:0x05fa, B:207:0x061f, B:209:0x0630, B:211:0x0636, B:213:0x0642, B:214:0x0673, B:216:0x0679, B:218:0x0687, B:219:0x068b, B:220:0x068e, B:221:0x0691, B:222:0x069f, B:224:0x06a5, B:226:0x06b5, B:227:0x06bc, B:229:0x06c8, B:230:0x06cf, B:231:0x06d2, B:233:0x0712, B:234:0x0725, B:236:0x072b, B:239:0x0745, B:241:0x0760, B:243:0x0779, B:245:0x077e, B:247:0x0782, B:249:0x0786, B:251:0x0790, B:252:0x079a, B:254:0x079e, B:256:0x07a4, B:257:0x07b2, B:258:0x07bb, B:326:0x0a0b, B:260:0x07c8, B:262:0x07df, B:268:0x07fb, B:270:0x081f, B:271:0x0827, B:273:0x082d, B:275:0x083f, B:282:0x0868, B:283:0x088b, B:285:0x0897, B:287:0x08ac, B:289:0x08ed, B:293:0x0905, B:295:0x090c, B:297:0x091b, B:299:0x091f, B:301:0x0923, B:303:0x0927, B:304:0x0933, B:305:0x0938, B:307:0x093e, B:309:0x095a, B:310:0x095f, B:325:0x0a08, B:311:0x097a, B:313:0x0982, B:317:0x09a9, B:319:0x09d5, B:320:0x09dc, B:321:0x09ee, B:323:0x09f8, B:314:0x098f, B:280:0x0853, B:266:0x07e6, B:327:0x0a17, B:329:0x0a25, B:330:0x0a2b, B:331:0x0a33, B:333:0x0a39, B:336:0x0a53, B:338:0x0a64, B:358:0x0ad8, B:360:0x0ade, B:362:0x0af6, B:365:0x0afd, B:370:0x0b2c, B:372:0x0b6e, B:375:0x0ba3, B:376:0x0ba7, B:377:0x0bb2, B:379:0x0bf5, B:380:0x0c02, B:382:0x0c11, B:386:0x0c2b, B:388:0x0c44, B:374:0x0b80, B:366:0x0b05, B:368:0x0b11, B:369:0x0b15, B:389:0x0c5c, B:390:0x0c74, B:393:0x0c7c, B:394:0x0c81, B:395:0x0c91, B:397:0x0cab, B:398:0x0cc6, B:400:0x0cd0, B:405:0x0cf3, B:404:0x0ce0, B:339:0x0a7c, B:341:0x0a82, B:343:0x0a8c, B:345:0x0a93, B:351:0x0aa3, B:353:0x0aaa, B:355:0x0ac9, B:357:0x0ad0, B:356:0x0acd, B:352:0x0aa7, B:344:0x0a90, B:202:0x05da, B:204:0x05e0, B:408:0x0d05), top: B:420:0x000e, inners: #0, #1, #2, #4 }] */
    /* JADX WARN: Removed duplicated region for block: B:157:0x04c5 A[Catch: all -> 0x0d17, TryCatch #3 {all -> 0x0d17, blocks: (B:3:0x000e, B:5:0x0026, B:8:0x002e, B:9:0x0040, B:12:0x0054, B:15:0x007b, B:17:0x00b1, B:20:0x00c3, B:22:0x00cd, B:173:0x0538, B:24:0x00f3, B:26:0x0101, B:29:0x0125, B:31:0x012b, B:33:0x013d, B:35:0x014b, B:37:0x015b, B:38:0x0168, B:39:0x016d, B:42:0x0186, B:111:0x03a7, B:112:0x03b3, B:115:0x03bd, B:121:0x03e0, B:118:0x03cf, B:143:0x045f, B:145:0x046b, B:148:0x047e, B:150:0x048f, B:152:0x049b, B:172:0x0524, B:157:0x04c5, B:159:0x04d5, B:162:0x04ea, B:164:0x04fb, B:166:0x0507, B:125:0x03e8, B:127:0x03f4, B:129:0x0400, B:141:0x0445, B:133:0x041d, B:136:0x042f, B:138:0x0435, B:140:0x043f, B:68:0x01e4, B:71:0x01ee, B:73:0x01fc, B:77:0x0243, B:74:0x0219, B:76:0x022a, B:81:0x0252, B:83:0x027e, B:84:0x02a8, B:86:0x02de, B:88:0x02e4, B:91:0x02f0, B:93:0x0326, B:94:0x0341, B:96:0x0347, B:98:0x0355, B:102:0x0368, B:99:0x035d, B:105:0x036f, B:108:0x0376, B:109:0x038e, B:176:0x054d, B:178:0x055b, B:180:0x0566, B:191:0x0598, B:181:0x056e, B:183:0x0579, B:185:0x057f, B:188:0x058b, B:190:0x0593, B:192:0x059b, B:193:0x05a7, B:196:0x05af, B:198:0x05c1, B:199:0x05cd, B:201:0x05d5, B:205:0x05fa, B:207:0x061f, B:209:0x0630, B:211:0x0636, B:213:0x0642, B:214:0x0673, B:216:0x0679, B:218:0x0687, B:219:0x068b, B:220:0x068e, B:221:0x0691, B:222:0x069f, B:224:0x06a5, B:226:0x06b5, B:227:0x06bc, B:229:0x06c8, B:230:0x06cf, B:231:0x06d2, B:233:0x0712, B:234:0x0725, B:236:0x072b, B:239:0x0745, B:241:0x0760, B:243:0x0779, B:245:0x077e, B:247:0x0782, B:249:0x0786, B:251:0x0790, B:252:0x079a, B:254:0x079e, B:256:0x07a4, B:257:0x07b2, B:258:0x07bb, B:326:0x0a0b, B:260:0x07c8, B:262:0x07df, B:268:0x07fb, B:270:0x081f, B:271:0x0827, B:273:0x082d, B:275:0x083f, B:282:0x0868, B:283:0x088b, B:285:0x0897, B:287:0x08ac, B:289:0x08ed, B:293:0x0905, B:295:0x090c, B:297:0x091b, B:299:0x091f, B:301:0x0923, B:303:0x0927, B:304:0x0933, B:305:0x0938, B:307:0x093e, B:309:0x095a, B:310:0x095f, B:325:0x0a08, B:311:0x097a, B:313:0x0982, B:317:0x09a9, B:319:0x09d5, B:320:0x09dc, B:321:0x09ee, B:323:0x09f8, B:314:0x098f, B:280:0x0853, B:266:0x07e6, B:327:0x0a17, B:329:0x0a25, B:330:0x0a2b, B:331:0x0a33, B:333:0x0a39, B:336:0x0a53, B:338:0x0a64, B:358:0x0ad8, B:360:0x0ade, B:362:0x0af6, B:365:0x0afd, B:370:0x0b2c, B:372:0x0b6e, B:375:0x0ba3, B:376:0x0ba7, B:377:0x0bb2, B:379:0x0bf5, B:380:0x0c02, B:382:0x0c11, B:386:0x0c2b, B:388:0x0c44, B:374:0x0b80, B:366:0x0b05, B:368:0x0b11, B:369:0x0b15, B:389:0x0c5c, B:390:0x0c74, B:393:0x0c7c, B:394:0x0c81, B:395:0x0c91, B:397:0x0cab, B:398:0x0cc6, B:400:0x0cd0, B:405:0x0cf3, B:404:0x0ce0, B:339:0x0a7c, B:341:0x0a82, B:343:0x0a8c, B:345:0x0a93, B:351:0x0aa3, B:353:0x0aaa, B:355:0x0ac9, B:357:0x0ad0, B:356:0x0acd, B:352:0x0aa7, B:344:0x0a90, B:202:0x05da, B:204:0x05e0, B:408:0x0d05), top: B:420:0x000e, inners: #0, #1, #2, #4 }] */
    /* JADX WARN: Removed duplicated region for block: B:270:0x081f A[Catch: all -> 0x0d17, TryCatch #3 {all -> 0x0d17, blocks: (B:3:0x000e, B:5:0x0026, B:8:0x002e, B:9:0x0040, B:12:0x0054, B:15:0x007b, B:17:0x00b1, B:20:0x00c3, B:22:0x00cd, B:173:0x0538, B:24:0x00f3, B:26:0x0101, B:29:0x0125, B:31:0x012b, B:33:0x013d, B:35:0x014b, B:37:0x015b, B:38:0x0168, B:39:0x016d, B:42:0x0186, B:111:0x03a7, B:112:0x03b3, B:115:0x03bd, B:121:0x03e0, B:118:0x03cf, B:143:0x045f, B:145:0x046b, B:148:0x047e, B:150:0x048f, B:152:0x049b, B:172:0x0524, B:157:0x04c5, B:159:0x04d5, B:162:0x04ea, B:164:0x04fb, B:166:0x0507, B:125:0x03e8, B:127:0x03f4, B:129:0x0400, B:141:0x0445, B:133:0x041d, B:136:0x042f, B:138:0x0435, B:140:0x043f, B:68:0x01e4, B:71:0x01ee, B:73:0x01fc, B:77:0x0243, B:74:0x0219, B:76:0x022a, B:81:0x0252, B:83:0x027e, B:84:0x02a8, B:86:0x02de, B:88:0x02e4, B:91:0x02f0, B:93:0x0326, B:94:0x0341, B:96:0x0347, B:98:0x0355, B:102:0x0368, B:99:0x035d, B:105:0x036f, B:108:0x0376, B:109:0x038e, B:176:0x054d, B:178:0x055b, B:180:0x0566, B:191:0x0598, B:181:0x056e, B:183:0x0579, B:185:0x057f, B:188:0x058b, B:190:0x0593, B:192:0x059b, B:193:0x05a7, B:196:0x05af, B:198:0x05c1, B:199:0x05cd, B:201:0x05d5, B:205:0x05fa, B:207:0x061f, B:209:0x0630, B:211:0x0636, B:213:0x0642, B:214:0x0673, B:216:0x0679, B:218:0x0687, B:219:0x068b, B:220:0x068e, B:221:0x0691, B:222:0x069f, B:224:0x06a5, B:226:0x06b5, B:227:0x06bc, B:229:0x06c8, B:230:0x06cf, B:231:0x06d2, B:233:0x0712, B:234:0x0725, B:236:0x072b, B:239:0x0745, B:241:0x0760, B:243:0x0779, B:245:0x077e, B:247:0x0782, B:249:0x0786, B:251:0x0790, B:252:0x079a, B:254:0x079e, B:256:0x07a4, B:257:0x07b2, B:258:0x07bb, B:326:0x0a0b, B:260:0x07c8, B:262:0x07df, B:268:0x07fb, B:270:0x081f, B:271:0x0827, B:273:0x082d, B:275:0x083f, B:282:0x0868, B:283:0x088b, B:285:0x0897, B:287:0x08ac, B:289:0x08ed, B:293:0x0905, B:295:0x090c, B:297:0x091b, B:299:0x091f, B:301:0x0923, B:303:0x0927, B:304:0x0933, B:305:0x0938, B:307:0x093e, B:309:0x095a, B:310:0x095f, B:325:0x0a08, B:311:0x097a, B:313:0x0982, B:317:0x09a9, B:319:0x09d5, B:320:0x09dc, B:321:0x09ee, B:323:0x09f8, B:314:0x098f, B:280:0x0853, B:266:0x07e6, B:327:0x0a17, B:329:0x0a25, B:330:0x0a2b, B:331:0x0a33, B:333:0x0a39, B:336:0x0a53, B:338:0x0a64, B:358:0x0ad8, B:360:0x0ade, B:362:0x0af6, B:365:0x0afd, B:370:0x0b2c, B:372:0x0b6e, B:375:0x0ba3, B:376:0x0ba7, B:377:0x0bb2, B:379:0x0bf5, B:380:0x0c02, B:382:0x0c11, B:386:0x0c2b, B:388:0x0c44, B:374:0x0b80, B:366:0x0b05, B:368:0x0b11, B:369:0x0b15, B:389:0x0c5c, B:390:0x0c74, B:393:0x0c7c, B:394:0x0c81, B:395:0x0c91, B:397:0x0cab, B:398:0x0cc6, B:400:0x0cd0, B:405:0x0cf3, B:404:0x0ce0, B:339:0x0a7c, B:341:0x0a82, B:343:0x0a8c, B:345:0x0a93, B:351:0x0aa3, B:353:0x0aaa, B:355:0x0ac9, B:357:0x0ad0, B:356:0x0acd, B:352:0x0aa7, B:344:0x0a90, B:202:0x05da, B:204:0x05e0, B:408:0x0d05), top: B:420:0x000e, inners: #0, #1, #2, #4 }] */
    /* JADX WARN: Removed duplicated region for block: B:282:0x0868 A[Catch: all -> 0x0d17, TryCatch #3 {all -> 0x0d17, blocks: (B:3:0x000e, B:5:0x0026, B:8:0x002e, B:9:0x0040, B:12:0x0054, B:15:0x007b, B:17:0x00b1, B:20:0x00c3, B:22:0x00cd, B:173:0x0538, B:24:0x00f3, B:26:0x0101, B:29:0x0125, B:31:0x012b, B:33:0x013d, B:35:0x014b, B:37:0x015b, B:38:0x0168, B:39:0x016d, B:42:0x0186, B:111:0x03a7, B:112:0x03b3, B:115:0x03bd, B:121:0x03e0, B:118:0x03cf, B:143:0x045f, B:145:0x046b, B:148:0x047e, B:150:0x048f, B:152:0x049b, B:172:0x0524, B:157:0x04c5, B:159:0x04d5, B:162:0x04ea, B:164:0x04fb, B:166:0x0507, B:125:0x03e8, B:127:0x03f4, B:129:0x0400, B:141:0x0445, B:133:0x041d, B:136:0x042f, B:138:0x0435, B:140:0x043f, B:68:0x01e4, B:71:0x01ee, B:73:0x01fc, B:77:0x0243, B:74:0x0219, B:76:0x022a, B:81:0x0252, B:83:0x027e, B:84:0x02a8, B:86:0x02de, B:88:0x02e4, B:91:0x02f0, B:93:0x0326, B:94:0x0341, B:96:0x0347, B:98:0x0355, B:102:0x0368, B:99:0x035d, B:105:0x036f, B:108:0x0376, B:109:0x038e, B:176:0x054d, B:178:0x055b, B:180:0x0566, B:191:0x0598, B:181:0x056e, B:183:0x0579, B:185:0x057f, B:188:0x058b, B:190:0x0593, B:192:0x059b, B:193:0x05a7, B:196:0x05af, B:198:0x05c1, B:199:0x05cd, B:201:0x05d5, B:205:0x05fa, B:207:0x061f, B:209:0x0630, B:211:0x0636, B:213:0x0642, B:214:0x0673, B:216:0x0679, B:218:0x0687, B:219:0x068b, B:220:0x068e, B:221:0x0691, B:222:0x069f, B:224:0x06a5, B:226:0x06b5, B:227:0x06bc, B:229:0x06c8, B:230:0x06cf, B:231:0x06d2, B:233:0x0712, B:234:0x0725, B:236:0x072b, B:239:0x0745, B:241:0x0760, B:243:0x0779, B:245:0x077e, B:247:0x0782, B:249:0x0786, B:251:0x0790, B:252:0x079a, B:254:0x079e, B:256:0x07a4, B:257:0x07b2, B:258:0x07bb, B:326:0x0a0b, B:260:0x07c8, B:262:0x07df, B:268:0x07fb, B:270:0x081f, B:271:0x0827, B:273:0x082d, B:275:0x083f, B:282:0x0868, B:283:0x088b, B:285:0x0897, B:287:0x08ac, B:289:0x08ed, B:293:0x0905, B:295:0x090c, B:297:0x091b, B:299:0x091f, B:301:0x0923, B:303:0x0927, B:304:0x0933, B:305:0x0938, B:307:0x093e, B:309:0x095a, B:310:0x095f, B:325:0x0a08, B:311:0x097a, B:313:0x0982, B:317:0x09a9, B:319:0x09d5, B:320:0x09dc, B:321:0x09ee, B:323:0x09f8, B:314:0x098f, B:280:0x0853, B:266:0x07e6, B:327:0x0a17, B:329:0x0a25, B:330:0x0a2b, B:331:0x0a33, B:333:0x0a39, B:336:0x0a53, B:338:0x0a64, B:358:0x0ad8, B:360:0x0ade, B:362:0x0af6, B:365:0x0afd, B:370:0x0b2c, B:372:0x0b6e, B:375:0x0ba3, B:376:0x0ba7, B:377:0x0bb2, B:379:0x0bf5, B:380:0x0c02, B:382:0x0c11, B:386:0x0c2b, B:388:0x0c44, B:374:0x0b80, B:366:0x0b05, B:368:0x0b11, B:369:0x0b15, B:389:0x0c5c, B:390:0x0c74, B:393:0x0c7c, B:394:0x0c81, B:395:0x0c91, B:397:0x0cab, B:398:0x0cc6, B:400:0x0cd0, B:405:0x0cf3, B:404:0x0ce0, B:339:0x0a7c, B:341:0x0a82, B:343:0x0a8c, B:345:0x0a93, B:351:0x0aa3, B:353:0x0aaa, B:355:0x0ac9, B:357:0x0ad0, B:356:0x0acd, B:352:0x0aa7, B:344:0x0a90, B:202:0x05da, B:204:0x05e0, B:408:0x0d05), top: B:420:0x000e, inners: #0, #1, #2, #4 }] */
    /* JADX WARN: Removed duplicated region for block: B:283:0x088b A[Catch: all -> 0x0d17, TryCatch #3 {all -> 0x0d17, blocks: (B:3:0x000e, B:5:0x0026, B:8:0x002e, B:9:0x0040, B:12:0x0054, B:15:0x007b, B:17:0x00b1, B:20:0x00c3, B:22:0x00cd, B:173:0x0538, B:24:0x00f3, B:26:0x0101, B:29:0x0125, B:31:0x012b, B:33:0x013d, B:35:0x014b, B:37:0x015b, B:38:0x0168, B:39:0x016d, B:42:0x0186, B:111:0x03a7, B:112:0x03b3, B:115:0x03bd, B:121:0x03e0, B:118:0x03cf, B:143:0x045f, B:145:0x046b, B:148:0x047e, B:150:0x048f, B:152:0x049b, B:172:0x0524, B:157:0x04c5, B:159:0x04d5, B:162:0x04ea, B:164:0x04fb, B:166:0x0507, B:125:0x03e8, B:127:0x03f4, B:129:0x0400, B:141:0x0445, B:133:0x041d, B:136:0x042f, B:138:0x0435, B:140:0x043f, B:68:0x01e4, B:71:0x01ee, B:73:0x01fc, B:77:0x0243, B:74:0x0219, B:76:0x022a, B:81:0x0252, B:83:0x027e, B:84:0x02a8, B:86:0x02de, B:88:0x02e4, B:91:0x02f0, B:93:0x0326, B:94:0x0341, B:96:0x0347, B:98:0x0355, B:102:0x0368, B:99:0x035d, B:105:0x036f, B:108:0x0376, B:109:0x038e, B:176:0x054d, B:178:0x055b, B:180:0x0566, B:191:0x0598, B:181:0x056e, B:183:0x0579, B:185:0x057f, B:188:0x058b, B:190:0x0593, B:192:0x059b, B:193:0x05a7, B:196:0x05af, B:198:0x05c1, B:199:0x05cd, B:201:0x05d5, B:205:0x05fa, B:207:0x061f, B:209:0x0630, B:211:0x0636, B:213:0x0642, B:214:0x0673, B:216:0x0679, B:218:0x0687, B:219:0x068b, B:220:0x068e, B:221:0x0691, B:222:0x069f, B:224:0x06a5, B:226:0x06b5, B:227:0x06bc, B:229:0x06c8, B:230:0x06cf, B:231:0x06d2, B:233:0x0712, B:234:0x0725, B:236:0x072b, B:239:0x0745, B:241:0x0760, B:243:0x0779, B:245:0x077e, B:247:0x0782, B:249:0x0786, B:251:0x0790, B:252:0x079a, B:254:0x079e, B:256:0x07a4, B:257:0x07b2, B:258:0x07bb, B:326:0x0a0b, B:260:0x07c8, B:262:0x07df, B:268:0x07fb, B:270:0x081f, B:271:0x0827, B:273:0x082d, B:275:0x083f, B:282:0x0868, B:283:0x088b, B:285:0x0897, B:287:0x08ac, B:289:0x08ed, B:293:0x0905, B:295:0x090c, B:297:0x091b, B:299:0x091f, B:301:0x0923, B:303:0x0927, B:304:0x0933, B:305:0x0938, B:307:0x093e, B:309:0x095a, B:310:0x095f, B:325:0x0a08, B:311:0x097a, B:313:0x0982, B:317:0x09a9, B:319:0x09d5, B:320:0x09dc, B:321:0x09ee, B:323:0x09f8, B:314:0x098f, B:280:0x0853, B:266:0x07e6, B:327:0x0a17, B:329:0x0a25, B:330:0x0a2b, B:331:0x0a33, B:333:0x0a39, B:336:0x0a53, B:338:0x0a64, B:358:0x0ad8, B:360:0x0ade, B:362:0x0af6, B:365:0x0afd, B:370:0x0b2c, B:372:0x0b6e, B:375:0x0ba3, B:376:0x0ba7, B:377:0x0bb2, B:379:0x0bf5, B:380:0x0c02, B:382:0x0c11, B:386:0x0c2b, B:388:0x0c44, B:374:0x0b80, B:366:0x0b05, B:368:0x0b11, B:369:0x0b15, B:389:0x0c5c, B:390:0x0c74, B:393:0x0c7c, B:394:0x0c81, B:395:0x0c91, B:397:0x0cab, B:398:0x0cc6, B:400:0x0cd0, B:405:0x0cf3, B:404:0x0ce0, B:339:0x0a7c, B:341:0x0a82, B:343:0x0a8c, B:345:0x0a93, B:351:0x0aa3, B:353:0x0aaa, B:355:0x0ac9, B:357:0x0ad0, B:356:0x0acd, B:352:0x0aa7, B:344:0x0a90, B:202:0x05da, B:204:0x05e0, B:408:0x0d05), top: B:420:0x000e, inners: #0, #1, #2, #4 }] */
    /* JADX WARN: Removed duplicated region for block: B:291:0x0902  */
    /* JADX WARN: Removed duplicated region for block: B:292:0x0904  */
    /* JADX WARN: Removed duplicated region for block: B:295:0x090c A[Catch: all -> 0x0d17, TryCatch #3 {all -> 0x0d17, blocks: (B:3:0x000e, B:5:0x0026, B:8:0x002e, B:9:0x0040, B:12:0x0054, B:15:0x007b, B:17:0x00b1, B:20:0x00c3, B:22:0x00cd, B:173:0x0538, B:24:0x00f3, B:26:0x0101, B:29:0x0125, B:31:0x012b, B:33:0x013d, B:35:0x014b, B:37:0x015b, B:38:0x0168, B:39:0x016d, B:42:0x0186, B:111:0x03a7, B:112:0x03b3, B:115:0x03bd, B:121:0x03e0, B:118:0x03cf, B:143:0x045f, B:145:0x046b, B:148:0x047e, B:150:0x048f, B:152:0x049b, B:172:0x0524, B:157:0x04c5, B:159:0x04d5, B:162:0x04ea, B:164:0x04fb, B:166:0x0507, B:125:0x03e8, B:127:0x03f4, B:129:0x0400, B:141:0x0445, B:133:0x041d, B:136:0x042f, B:138:0x0435, B:140:0x043f, B:68:0x01e4, B:71:0x01ee, B:73:0x01fc, B:77:0x0243, B:74:0x0219, B:76:0x022a, B:81:0x0252, B:83:0x027e, B:84:0x02a8, B:86:0x02de, B:88:0x02e4, B:91:0x02f0, B:93:0x0326, B:94:0x0341, B:96:0x0347, B:98:0x0355, B:102:0x0368, B:99:0x035d, B:105:0x036f, B:108:0x0376, B:109:0x038e, B:176:0x054d, B:178:0x055b, B:180:0x0566, B:191:0x0598, B:181:0x056e, B:183:0x0579, B:185:0x057f, B:188:0x058b, B:190:0x0593, B:192:0x059b, B:193:0x05a7, B:196:0x05af, B:198:0x05c1, B:199:0x05cd, B:201:0x05d5, B:205:0x05fa, B:207:0x061f, B:209:0x0630, B:211:0x0636, B:213:0x0642, B:214:0x0673, B:216:0x0679, B:218:0x0687, B:219:0x068b, B:220:0x068e, B:221:0x0691, B:222:0x069f, B:224:0x06a5, B:226:0x06b5, B:227:0x06bc, B:229:0x06c8, B:230:0x06cf, B:231:0x06d2, B:233:0x0712, B:234:0x0725, B:236:0x072b, B:239:0x0745, B:241:0x0760, B:243:0x0779, B:245:0x077e, B:247:0x0782, B:249:0x0786, B:251:0x0790, B:252:0x079a, B:254:0x079e, B:256:0x07a4, B:257:0x07b2, B:258:0x07bb, B:326:0x0a0b, B:260:0x07c8, B:262:0x07df, B:268:0x07fb, B:270:0x081f, B:271:0x0827, B:273:0x082d, B:275:0x083f, B:282:0x0868, B:283:0x088b, B:285:0x0897, B:287:0x08ac, B:289:0x08ed, B:293:0x0905, B:295:0x090c, B:297:0x091b, B:299:0x091f, B:301:0x0923, B:303:0x0927, B:304:0x0933, B:305:0x0938, B:307:0x093e, B:309:0x095a, B:310:0x095f, B:325:0x0a08, B:311:0x097a, B:313:0x0982, B:317:0x09a9, B:319:0x09d5, B:320:0x09dc, B:321:0x09ee, B:323:0x09f8, B:314:0x098f, B:280:0x0853, B:266:0x07e6, B:327:0x0a17, B:329:0x0a25, B:330:0x0a2b, B:331:0x0a33, B:333:0x0a39, B:336:0x0a53, B:338:0x0a64, B:358:0x0ad8, B:360:0x0ade, B:362:0x0af6, B:365:0x0afd, B:370:0x0b2c, B:372:0x0b6e, B:375:0x0ba3, B:376:0x0ba7, B:377:0x0bb2, B:379:0x0bf5, B:380:0x0c02, B:382:0x0c11, B:386:0x0c2b, B:388:0x0c44, B:374:0x0b80, B:366:0x0b05, B:368:0x0b11, B:369:0x0b15, B:389:0x0c5c, B:390:0x0c74, B:393:0x0c7c, B:394:0x0c81, B:395:0x0c91, B:397:0x0cab, B:398:0x0cc6, B:400:0x0cd0, B:405:0x0cf3, B:404:0x0ce0, B:339:0x0a7c, B:341:0x0a82, B:343:0x0a8c, B:345:0x0a93, B:351:0x0aa3, B:353:0x0aaa, B:355:0x0ac9, B:357:0x0ad0, B:356:0x0acd, B:352:0x0aa7, B:344:0x0a90, B:202:0x05da, B:204:0x05e0, B:408:0x0d05), top: B:420:0x000e, inners: #0, #1, #2, #4 }] */
    /* JADX WARN: Removed duplicated region for block: B:305:0x0938 A[Catch: all -> 0x0d17, TryCatch #3 {all -> 0x0d17, blocks: (B:3:0x000e, B:5:0x0026, B:8:0x002e, B:9:0x0040, B:12:0x0054, B:15:0x007b, B:17:0x00b1, B:20:0x00c3, B:22:0x00cd, B:173:0x0538, B:24:0x00f3, B:26:0x0101, B:29:0x0125, B:31:0x012b, B:33:0x013d, B:35:0x014b, B:37:0x015b, B:38:0x0168, B:39:0x016d, B:42:0x0186, B:111:0x03a7, B:112:0x03b3, B:115:0x03bd, B:121:0x03e0, B:118:0x03cf, B:143:0x045f, B:145:0x046b, B:148:0x047e, B:150:0x048f, B:152:0x049b, B:172:0x0524, B:157:0x04c5, B:159:0x04d5, B:162:0x04ea, B:164:0x04fb, B:166:0x0507, B:125:0x03e8, B:127:0x03f4, B:129:0x0400, B:141:0x0445, B:133:0x041d, B:136:0x042f, B:138:0x0435, B:140:0x043f, B:68:0x01e4, B:71:0x01ee, B:73:0x01fc, B:77:0x0243, B:74:0x0219, B:76:0x022a, B:81:0x0252, B:83:0x027e, B:84:0x02a8, B:86:0x02de, B:88:0x02e4, B:91:0x02f0, B:93:0x0326, B:94:0x0341, B:96:0x0347, B:98:0x0355, B:102:0x0368, B:99:0x035d, B:105:0x036f, B:108:0x0376, B:109:0x038e, B:176:0x054d, B:178:0x055b, B:180:0x0566, B:191:0x0598, B:181:0x056e, B:183:0x0579, B:185:0x057f, B:188:0x058b, B:190:0x0593, B:192:0x059b, B:193:0x05a7, B:196:0x05af, B:198:0x05c1, B:199:0x05cd, B:201:0x05d5, B:205:0x05fa, B:207:0x061f, B:209:0x0630, B:211:0x0636, B:213:0x0642, B:214:0x0673, B:216:0x0679, B:218:0x0687, B:219:0x068b, B:220:0x068e, B:221:0x0691, B:222:0x069f, B:224:0x06a5, B:226:0x06b5, B:227:0x06bc, B:229:0x06c8, B:230:0x06cf, B:231:0x06d2, B:233:0x0712, B:234:0x0725, B:236:0x072b, B:239:0x0745, B:241:0x0760, B:243:0x0779, B:245:0x077e, B:247:0x0782, B:249:0x0786, B:251:0x0790, B:252:0x079a, B:254:0x079e, B:256:0x07a4, B:257:0x07b2, B:258:0x07bb, B:326:0x0a0b, B:260:0x07c8, B:262:0x07df, B:268:0x07fb, B:270:0x081f, B:271:0x0827, B:273:0x082d, B:275:0x083f, B:282:0x0868, B:283:0x088b, B:285:0x0897, B:287:0x08ac, B:289:0x08ed, B:293:0x0905, B:295:0x090c, B:297:0x091b, B:299:0x091f, B:301:0x0923, B:303:0x0927, B:304:0x0933, B:305:0x0938, B:307:0x093e, B:309:0x095a, B:310:0x095f, B:325:0x0a08, B:311:0x097a, B:313:0x0982, B:317:0x09a9, B:319:0x09d5, B:320:0x09dc, B:321:0x09ee, B:323:0x09f8, B:314:0x098f, B:280:0x0853, B:266:0x07e6, B:327:0x0a17, B:329:0x0a25, B:330:0x0a2b, B:331:0x0a33, B:333:0x0a39, B:336:0x0a53, B:338:0x0a64, B:358:0x0ad8, B:360:0x0ade, B:362:0x0af6, B:365:0x0afd, B:370:0x0b2c, B:372:0x0b6e, B:375:0x0ba3, B:376:0x0ba7, B:377:0x0bb2, B:379:0x0bf5, B:380:0x0c02, B:382:0x0c11, B:386:0x0c2b, B:388:0x0c44, B:374:0x0b80, B:366:0x0b05, B:368:0x0b11, B:369:0x0b15, B:389:0x0c5c, B:390:0x0c74, B:393:0x0c7c, B:394:0x0c81, B:395:0x0c91, B:397:0x0cab, B:398:0x0cc6, B:400:0x0cd0, B:405:0x0cf3, B:404:0x0ce0, B:339:0x0a7c, B:341:0x0a82, B:343:0x0a8c, B:345:0x0a93, B:351:0x0aa3, B:353:0x0aaa, B:355:0x0ac9, B:357:0x0ad0, B:356:0x0acd, B:352:0x0aa7, B:344:0x0a90, B:202:0x05da, B:204:0x05e0, B:408:0x0d05), top: B:420:0x000e, inners: #0, #1, #2, #4 }] */
    /* JADX WARN: Removed duplicated region for block: B:372:0x0b6e A[Catch: all -> 0x0d17, TryCatch #3 {all -> 0x0d17, blocks: (B:3:0x000e, B:5:0x0026, B:8:0x002e, B:9:0x0040, B:12:0x0054, B:15:0x007b, B:17:0x00b1, B:20:0x00c3, B:22:0x00cd, B:173:0x0538, B:24:0x00f3, B:26:0x0101, B:29:0x0125, B:31:0x012b, B:33:0x013d, B:35:0x014b, B:37:0x015b, B:38:0x0168, B:39:0x016d, B:42:0x0186, B:111:0x03a7, B:112:0x03b3, B:115:0x03bd, B:121:0x03e0, B:118:0x03cf, B:143:0x045f, B:145:0x046b, B:148:0x047e, B:150:0x048f, B:152:0x049b, B:172:0x0524, B:157:0x04c5, B:159:0x04d5, B:162:0x04ea, B:164:0x04fb, B:166:0x0507, B:125:0x03e8, B:127:0x03f4, B:129:0x0400, B:141:0x0445, B:133:0x041d, B:136:0x042f, B:138:0x0435, B:140:0x043f, B:68:0x01e4, B:71:0x01ee, B:73:0x01fc, B:77:0x0243, B:74:0x0219, B:76:0x022a, B:81:0x0252, B:83:0x027e, B:84:0x02a8, B:86:0x02de, B:88:0x02e4, B:91:0x02f0, B:93:0x0326, B:94:0x0341, B:96:0x0347, B:98:0x0355, B:102:0x0368, B:99:0x035d, B:105:0x036f, B:108:0x0376, B:109:0x038e, B:176:0x054d, B:178:0x055b, B:180:0x0566, B:191:0x0598, B:181:0x056e, B:183:0x0579, B:185:0x057f, B:188:0x058b, B:190:0x0593, B:192:0x059b, B:193:0x05a7, B:196:0x05af, B:198:0x05c1, B:199:0x05cd, B:201:0x05d5, B:205:0x05fa, B:207:0x061f, B:209:0x0630, B:211:0x0636, B:213:0x0642, B:214:0x0673, B:216:0x0679, B:218:0x0687, B:219:0x068b, B:220:0x068e, B:221:0x0691, B:222:0x069f, B:224:0x06a5, B:226:0x06b5, B:227:0x06bc, B:229:0x06c8, B:230:0x06cf, B:231:0x06d2, B:233:0x0712, B:234:0x0725, B:236:0x072b, B:239:0x0745, B:241:0x0760, B:243:0x0779, B:245:0x077e, B:247:0x0782, B:249:0x0786, B:251:0x0790, B:252:0x079a, B:254:0x079e, B:256:0x07a4, B:257:0x07b2, B:258:0x07bb, B:326:0x0a0b, B:260:0x07c8, B:262:0x07df, B:268:0x07fb, B:270:0x081f, B:271:0x0827, B:273:0x082d, B:275:0x083f, B:282:0x0868, B:283:0x088b, B:285:0x0897, B:287:0x08ac, B:289:0x08ed, B:293:0x0905, B:295:0x090c, B:297:0x091b, B:299:0x091f, B:301:0x0923, B:303:0x0927, B:304:0x0933, B:305:0x0938, B:307:0x093e, B:309:0x095a, B:310:0x095f, B:325:0x0a08, B:311:0x097a, B:313:0x0982, B:317:0x09a9, B:319:0x09d5, B:320:0x09dc, B:321:0x09ee, B:323:0x09f8, B:314:0x098f, B:280:0x0853, B:266:0x07e6, B:327:0x0a17, B:329:0x0a25, B:330:0x0a2b, B:331:0x0a33, B:333:0x0a39, B:336:0x0a53, B:338:0x0a64, B:358:0x0ad8, B:360:0x0ade, B:362:0x0af6, B:365:0x0afd, B:370:0x0b2c, B:372:0x0b6e, B:375:0x0ba3, B:376:0x0ba7, B:377:0x0bb2, B:379:0x0bf5, B:380:0x0c02, B:382:0x0c11, B:386:0x0c2b, B:388:0x0c44, B:374:0x0b80, B:366:0x0b05, B:368:0x0b11, B:369:0x0b15, B:389:0x0c5c, B:390:0x0c74, B:393:0x0c7c, B:394:0x0c81, B:395:0x0c91, B:397:0x0cab, B:398:0x0cc6, B:400:0x0cd0, B:405:0x0cf3, B:404:0x0ce0, B:339:0x0a7c, B:341:0x0a82, B:343:0x0a8c, B:345:0x0a93, B:351:0x0aa3, B:353:0x0aaa, B:355:0x0ac9, B:357:0x0ad0, B:356:0x0acd, B:352:0x0aa7, B:344:0x0a90, B:202:0x05da, B:204:0x05e0, B:408:0x0d05), top: B:420:0x000e, inners: #0, #1, #2, #4 }] */
    /* JADX WARN: Removed duplicated region for block: B:379:0x0bf5 A[Catch: all -> 0x0d17, TRY_LEAVE, TryCatch #3 {all -> 0x0d17, blocks: (B:3:0x000e, B:5:0x0026, B:8:0x002e, B:9:0x0040, B:12:0x0054, B:15:0x007b, B:17:0x00b1, B:20:0x00c3, B:22:0x00cd, B:173:0x0538, B:24:0x00f3, B:26:0x0101, B:29:0x0125, B:31:0x012b, B:33:0x013d, B:35:0x014b, B:37:0x015b, B:38:0x0168, B:39:0x016d, B:42:0x0186, B:111:0x03a7, B:112:0x03b3, B:115:0x03bd, B:121:0x03e0, B:118:0x03cf, B:143:0x045f, B:145:0x046b, B:148:0x047e, B:150:0x048f, B:152:0x049b, B:172:0x0524, B:157:0x04c5, B:159:0x04d5, B:162:0x04ea, B:164:0x04fb, B:166:0x0507, B:125:0x03e8, B:127:0x03f4, B:129:0x0400, B:141:0x0445, B:133:0x041d, B:136:0x042f, B:138:0x0435, B:140:0x043f, B:68:0x01e4, B:71:0x01ee, B:73:0x01fc, B:77:0x0243, B:74:0x0219, B:76:0x022a, B:81:0x0252, B:83:0x027e, B:84:0x02a8, B:86:0x02de, B:88:0x02e4, B:91:0x02f0, B:93:0x0326, B:94:0x0341, B:96:0x0347, B:98:0x0355, B:102:0x0368, B:99:0x035d, B:105:0x036f, B:108:0x0376, B:109:0x038e, B:176:0x054d, B:178:0x055b, B:180:0x0566, B:191:0x0598, B:181:0x056e, B:183:0x0579, B:185:0x057f, B:188:0x058b, B:190:0x0593, B:192:0x059b, B:193:0x05a7, B:196:0x05af, B:198:0x05c1, B:199:0x05cd, B:201:0x05d5, B:205:0x05fa, B:207:0x061f, B:209:0x0630, B:211:0x0636, B:213:0x0642, B:214:0x0673, B:216:0x0679, B:218:0x0687, B:219:0x068b, B:220:0x068e, B:221:0x0691, B:222:0x069f, B:224:0x06a5, B:226:0x06b5, B:227:0x06bc, B:229:0x06c8, B:230:0x06cf, B:231:0x06d2, B:233:0x0712, B:234:0x0725, B:236:0x072b, B:239:0x0745, B:241:0x0760, B:243:0x0779, B:245:0x077e, B:247:0x0782, B:249:0x0786, B:251:0x0790, B:252:0x079a, B:254:0x079e, B:256:0x07a4, B:257:0x07b2, B:258:0x07bb, B:326:0x0a0b, B:260:0x07c8, B:262:0x07df, B:268:0x07fb, B:270:0x081f, B:271:0x0827, B:273:0x082d, B:275:0x083f, B:282:0x0868, B:283:0x088b, B:285:0x0897, B:287:0x08ac, B:289:0x08ed, B:293:0x0905, B:295:0x090c, B:297:0x091b, B:299:0x091f, B:301:0x0923, B:303:0x0927, B:304:0x0933, B:305:0x0938, B:307:0x093e, B:309:0x095a, B:310:0x095f, B:325:0x0a08, B:311:0x097a, B:313:0x0982, B:317:0x09a9, B:319:0x09d5, B:320:0x09dc, B:321:0x09ee, B:323:0x09f8, B:314:0x098f, B:280:0x0853, B:266:0x07e6, B:327:0x0a17, B:329:0x0a25, B:330:0x0a2b, B:331:0x0a33, B:333:0x0a39, B:336:0x0a53, B:338:0x0a64, B:358:0x0ad8, B:360:0x0ade, B:362:0x0af6, B:365:0x0afd, B:370:0x0b2c, B:372:0x0b6e, B:375:0x0ba3, B:376:0x0ba7, B:377:0x0bb2, B:379:0x0bf5, B:380:0x0c02, B:382:0x0c11, B:386:0x0c2b, B:388:0x0c44, B:374:0x0b80, B:366:0x0b05, B:368:0x0b11, B:369:0x0b15, B:389:0x0c5c, B:390:0x0c74, B:393:0x0c7c, B:394:0x0c81, B:395:0x0c91, B:397:0x0cab, B:398:0x0cc6, B:400:0x0cd0, B:405:0x0cf3, B:404:0x0ce0, B:339:0x0a7c, B:341:0x0a82, B:343:0x0a8c, B:345:0x0a93, B:351:0x0aa3, B:353:0x0aaa, B:355:0x0ac9, B:357:0x0ad0, B:356:0x0acd, B:352:0x0aa7, B:344:0x0a90, B:202:0x05da, B:204:0x05e0, B:408:0x0d05), top: B:420:0x000e, inners: #0, #1, #2, #4 }] */
    /* JADX WARN: Removed duplicated region for block: B:382:0x0c11 A[Catch: SQLiteException -> 0x0c29, all -> 0x0d17, TRY_LEAVE, TryCatch #0 {SQLiteException -> 0x0c29, blocks: (B:380:0x0c02, B:382:0x0c11), top: B:414:0x0c02, outer: #3 }] */
    /* JADX WARN: Removed duplicated region for block: B:61:0x01cb  */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    private final boolean zzah(String str, long j) {
        int i;
        String str2;
        com.google.android.gms.internal.measurement.zzgc zzgcVar;
        zzkq zzkqVar;
        zzam zzamVar;
        com.google.android.gms.internal.measurement.zzgd zzgdVar;
        long currentTimeMillis;
        long zzk;
        ContentValues contentValues;
        long parseLong;
        int zzc;
        long j2;
        SecureRandom secureRandom;
        com.google.android.gms.internal.measurement.zzgc zzgcVar2;
        Long l;
        zzkq zzkqVar2;
        HashMap hashMap;
        long zzr;
        HashMap hashMap2;
        com.google.android.gms.internal.measurement.zzgc zzgcVar3;
        com.google.android.gms.internal.measurement.zzfs zzfsVar;
        int i2;
        String str3;
        com.google.android.gms.internal.measurement.zzgc zzgcVar4;
        com.google.android.gms.internal.measurement.zzfs zzfsVar2;
        int i3;
        com.google.android.gms.internal.measurement.zzfs zzfsVar3;
        int i4;
        int i5;
        com.google.android.gms.internal.measurement.zzgc zzgcVar5;
        int i6;
        com.google.android.gms.internal.measurement.zzfs zzfsVar4;
        int i7;
        int i8;
        com.google.android.gms.internal.measurement.zzfs zzfsVar5;
        char c2;
        String str4 = "_npa";
        String str5 = "_ai";
        zzam zzamVar2 = this.zze;
        zzal(zzamVar2);
        zzamVar2.zzw();
        try {
            zzkq zzkqVar3 = new zzkq(this, null);
            zzam zzamVar3 = this.zze;
            zzal(zzamVar3);
            zzamVar3.zzU(null, j, this.zzA, zzkqVar3);
            List list = zzkqVar3.zzc;
            if (list != null && !list.isEmpty()) {
                com.google.android.gms.internal.measurement.zzgc zzgcVar6 = (com.google.android.gms.internal.measurement.zzgc) zzkqVar3.zza.zzby();
                zzgcVar6.zzr();
                com.google.android.gms.internal.measurement.zzfs zzfsVar6 = null;
                com.google.android.gms.internal.measurement.zzfs zzfsVar7 = null;
                int i9 = 0;
                int i10 = 0;
                int i11 = -1;
                int i12 = -1;
                int i13 = 0;
                while (true) {
                    i = i13;
                    str2 = str4;
                    String str6 = str5;
                    if (i9 >= zzkqVar3.zzc.size()) {
                        break;
                    }
                    com.google.android.gms.internal.measurement.zzfs zzfsVar8 = (com.google.android.gms.internal.measurement.zzfs) ((com.google.android.gms.internal.measurement.zzft) zzkqVar3.zzc.get(i9)).zzby();
                    zzfi zzfiVar = this.zzc;
                    zzal(zzfiVar);
                    int i14 = i10;
                    if (zzfiVar.zzr(zzkqVar3.zza.zzx(), zzfsVar8.zzo())) {
                        zzay().zzk().zzc("Dropping blocked raw event. appId", zzeh.zzn(zzkqVar3.zza.zzx()), this.zzn.zzj().zzd(zzfsVar8.zzo()));
                        zzfi zzfiVar2 = this.zzc;
                        zzal(zzfiVar2);
                        if (!zzfiVar2.zzp(zzkqVar3.zza.zzx())) {
                            zzfi zzfiVar3 = this.zzc;
                            zzal(zzfiVar3);
                            if (!zzfiVar3.zzs(zzkqVar3.zza.zzx()) && !"_err".equals(zzfsVar8.zzo())) {
                                zzv().zzN(this.zzF, zzkqVar3.zza.zzx(), 11, "_ev", zzfsVar8.zzo(), 0);
                            }
                        }
                        i7 = i9;
                        zzfsVar = zzfsVar6;
                        i13 = i;
                        i10 = i14;
                        zzgcVar5 = zzgcVar6;
                    } else {
                        if (zzfsVar8.zzo().equals(zzgo.zza(str6))) {
                            zzfsVar8.zzi(str6);
                            str6 = str6;
                            zzay().zzj().zza("Renaming ad_impression to _ai");
                            if (Log.isLoggable(zzay().zzq(), 5)) {
                                int i15 = 0;
                                while (i15 < zzfsVar8.zza()) {
                                    int i16 = i9;
                                    if (FirebaseAnalytics.Param.AD_PLATFORM.equals(zzfsVar8.zzn(i15).zzg()) && !zzfsVar8.zzn(i15).zzh().isEmpty() && "admob".equalsIgnoreCase(zzfsVar8.zzn(i15).zzh())) {
                                        zzay().zzl().zza("AdMob ad impression logged from app. Potentially duplicative.");
                                    }
                                    i15++;
                                    i9 = i16;
                                }
                            }
                        }
                        int i17 = i9;
                        zzfi zzfiVar4 = this.zzc;
                        zzal(zzfiVar4);
                        boolean zzq = zzfiVar4.zzq(zzkqVar3.zza.zzx(), zzfsVar8.zzo());
                        if (zzq) {
                            zzfsVar = zzfsVar6;
                            i2 = i11;
                        } else {
                            zzal(this.zzi);
                            String zzo = zzfsVar8.zzo();
                            Preconditions.checkNotEmpty(zzo);
                            i2 = i11;
                            int hashCode = zzo.hashCode();
                            zzfsVar = zzfsVar6;
                            if (hashCode == 94660) {
                                if (zzo.equals("_in")) {
                                    c2 = 0;
                                    if (c2 != 0) {
                                    }
                                }
                                c2 = 65535;
                                if (c2 != 0) {
                                }
                            } else if (hashCode != 95025) {
                                if (hashCode == 95027 && zzo.equals("_ui")) {
                                    c2 = 1;
                                    if (c2 != 0 && c2 != 1 && c2 != 2) {
                                        zzgcVar4 = zzgcVar6;
                                        str3 = "_et";
                                        zzfsVar2 = zzfsVar7;
                                        i3 = i12;
                                        zzq = false;
                                        if (zzq) {
                                            ArrayList arrayList = new ArrayList(zzfsVar8.zzp());
                                            int i18 = -1;
                                            int i19 = -1;
                                            for (int i20 = 0; i20 < arrayList.size(); i20++) {
                                                if ("value".equals(((com.google.android.gms.internal.measurement.zzfx) arrayList.get(i20)).zzg())) {
                                                    i18 = i20;
                                                } else if (FirebaseAnalytics.Param.CURRENCY.equals(((com.google.android.gms.internal.measurement.zzfx) arrayList.get(i20)).zzg())) {
                                                    i19 = i20;
                                                }
                                            }
                                            if (i18 != -1) {
                                                if (!((com.google.android.gms.internal.measurement.zzfx) arrayList.get(i18)).zzw() && !((com.google.android.gms.internal.measurement.zzfx) arrayList.get(i18)).zzu()) {
                                                    zzay().zzl().zza("Value must be specified with a numeric type.");
                                                    zzfsVar8.zzh(i18);
                                                    zzab(zzfsVar8, "_c");
                                                    zzaa(zzfsVar8, 18, "value");
                                                } else {
                                                    if (i19 != -1) {
                                                        String zzh = ((com.google.android.gms.internal.measurement.zzfx) arrayList.get(i19)).zzh();
                                                        if (zzh.length() == 3) {
                                                            int i21 = 0;
                                                            while (i21 < zzh.length()) {
                                                                int codePointAt = zzh.codePointAt(i21);
                                                                if (Character.isLetter(codePointAt)) {
                                                                    i21 += Character.charCount(codePointAt);
                                                                }
                                                            }
                                                        }
                                                    }
                                                    zzay().zzl().zza("Value parameter discarded. You must also supply a 3-letter ISO_4217 currency code in the currency parameter.");
                                                    zzfsVar8.zzh(i18);
                                                    zzab(zzfsVar8, "_c");
                                                    zzaa(zzfsVar8, 19, FirebaseAnalytics.Param.CURRENCY);
                                                    break;
                                                }
                                            }
                                            if (!"_e".equals(zzfsVar8.zzo())) {
                                                zzal(this.zzi);
                                                if (zzkv.zzB((com.google.android.gms.internal.measurement.zzft) zzfsVar8.zzaC(), "_fr") != null) {
                                                    i5 = i3;
                                                    zzgcVar5 = zzgcVar4;
                                                    i6 = i2;
                                                    i12 = i5;
                                                    zzfsVar7 = zzfsVar2;
                                                    i11 = i6;
                                                } else if (zzfsVar2 == null || Math.abs(zzfsVar2.zzc() - zzfsVar8.zzc()) > 1000) {
                                                    zzgcVar5 = zzgcVar4;
                                                    zzfsVar = zzfsVar8;
                                                    i12 = i3;
                                                    zzfsVar7 = zzfsVar2;
                                                    i11 = i14;
                                                } else {
                                                    com.google.android.gms.internal.measurement.zzfs zzfsVar9 = (com.google.android.gms.internal.measurement.zzfs) zzfsVar2.zzau();
                                                    if (zzaj(zzfsVar8, zzfsVar9)) {
                                                        i8 = i3;
                                                        zzgcVar5 = zzgcVar4;
                                                        zzgcVar5.zzS(i8, zzfsVar9);
                                                        i11 = i2;
                                                        zzfsVar5 = null;
                                                        zzfsVar7 = null;
                                                    } else {
                                                        i8 = i3;
                                                        zzgcVar5 = zzgcVar4;
                                                        zzfsVar5 = zzfsVar8;
                                                        zzfsVar7 = zzfsVar2;
                                                        i11 = i14;
                                                    }
                                                    zzfsVar = zzfsVar5;
                                                    i12 = i8;
                                                }
                                            } else {
                                                i5 = i3;
                                                zzgcVar5 = zzgcVar4;
                                                if ("_vs".equals(zzfsVar8.zzo())) {
                                                    zzal(this.zzi);
                                                    if (zzkv.zzB((com.google.android.gms.internal.measurement.zzft) zzfsVar8.zzaC(), str3) == null) {
                                                        if (zzfsVar == null || Math.abs(zzfsVar.zzc() - zzfsVar8.zzc()) > 1000) {
                                                            zzfsVar7 = zzfsVar8;
                                                            i11 = i2;
                                                            i12 = i14;
                                                        } else {
                                                            com.google.android.gms.internal.measurement.zzfs zzfsVar10 = (com.google.android.gms.internal.measurement.zzfs) zzfsVar.zzau();
                                                            if (zzaj(zzfsVar10, zzfsVar8)) {
                                                                i6 = i2;
                                                                zzgcVar5.zzS(i6, zzfsVar10);
                                                                i12 = i5;
                                                                zzfsVar4 = null;
                                                                zzfsVar = null;
                                                            } else {
                                                                i6 = i2;
                                                                zzfsVar4 = zzfsVar8;
                                                                i12 = i14;
                                                            }
                                                            zzfsVar7 = zzfsVar4;
                                                            i11 = i6;
                                                        }
                                                    }
                                                }
                                                i6 = i2;
                                                i12 = i5;
                                                zzfsVar7 = zzfsVar2;
                                                i11 = i6;
                                            }
                                            i7 = i17;
                                            zzkqVar3.zzc.set(i7, (com.google.android.gms.internal.measurement.zzft) zzfsVar8.zzaC());
                                            i10 = i14 + 1;
                                            zzgcVar5.zzk(zzfsVar8);
                                            i13 = i;
                                        }
                                        if (!"_e".equals(zzfsVar8.zzo())) {
                                        }
                                        i7 = i17;
                                        zzkqVar3.zzc.set(i7, (com.google.android.gms.internal.measurement.zzft) zzfsVar8.zzaC());
                                        i10 = i14 + 1;
                                        zzgcVar5.zzk(zzfsVar8);
                                        i13 = i;
                                    }
                                }
                                c2 = 65535;
                                if (c2 != 0) {
                                    zzgcVar4 = zzgcVar6;
                                    str3 = "_et";
                                    zzfsVar2 = zzfsVar7;
                                    i3 = i12;
                                    zzq = false;
                                    if (zzq) {
                                    }
                                    if (!"_e".equals(zzfsVar8.zzo())) {
                                    }
                                    i7 = i17;
                                    zzkqVar3.zzc.set(i7, (com.google.android.gms.internal.measurement.zzft) zzfsVar8.zzaC());
                                    i10 = i14 + 1;
                                    zzgcVar5.zzk(zzfsVar8);
                                    i13 = i;
                                }
                            } else {
                                if (zzo.equals("_ug")) {
                                    c2 = 2;
                                    if (c2 != 0) {
                                    }
                                }
                                c2 = 65535;
                                if (c2 != 0) {
                                }
                            }
                        }
                        str3 = "_et";
                        int i22 = 0;
                        boolean z = false;
                        boolean z2 = false;
                        while (true) {
                            zzgcVar4 = zzgcVar6;
                            if (i22 >= zzfsVar8.zza()) {
                                break;
                            }
                            if ("_c".equals(zzfsVar8.zzn(i22).zzg())) {
                                com.google.android.gms.internal.measurement.zzfw zzfwVar = (com.google.android.gms.internal.measurement.zzfw) zzfsVar8.zzn(i22).zzby();
                                zzfsVar3 = zzfsVar7;
                                i4 = i12;
                                zzfwVar.zzi(1L);
                                zzfsVar8.zzk(i22, (com.google.android.gms.internal.measurement.zzfx) zzfwVar.zzaC());
                                z = true;
                            } else {
                                zzfsVar3 = zzfsVar7;
                                i4 = i12;
                                if ("_r".equals(zzfsVar8.zzn(i22).zzg())) {
                                    com.google.android.gms.internal.measurement.zzfw zzfwVar2 = (com.google.android.gms.internal.measurement.zzfw) zzfsVar8.zzn(i22).zzby();
                                    zzfwVar2.zzi(1L);
                                    zzfsVar8.zzk(i22, (com.google.android.gms.internal.measurement.zzfx) zzfwVar2.zzaC());
                                    z2 = true;
                                }
                            }
                            i22++;
                            zzfsVar7 = zzfsVar3;
                            i12 = i4;
                            zzgcVar6 = zzgcVar4;
                        }
                        zzfsVar2 = zzfsVar7;
                        i3 = i12;
                        if (!z && zzq) {
                            zzay().zzj().zzb("Marking event as conversion", this.zzn.zzj().zzd(zzfsVar8.zzo()));
                            com.google.android.gms.internal.measurement.zzfw zze = com.google.android.gms.internal.measurement.zzfx.zze();
                            zze.zzj("_c");
                            zze.zzi(1L);
                            zzfsVar8.zze(zze);
                        }
                        if (!z2) {
                            zzay().zzj().zzb("Marking event as real-time", this.zzn.zzj().zzd(zzfsVar8.zzo()));
                            com.google.android.gms.internal.measurement.zzfw zze2 = com.google.android.gms.internal.measurement.zzfx.zze();
                            zze2.zzj("_r");
                            zze2.zzi(1L);
                            zzfsVar8.zze(zze2);
                        }
                        zzam zzamVar4 = this.zze;
                        zzal(zzamVar4);
                        if (zzamVar4.zzl(zza(), zzkqVar3.zza.zzx(), false, false, false, false, true).zze > zzg().zze(zzkqVar3.zza.zzx(), zzdu.zzn)) {
                            zzab(zzfsVar8, "_r");
                        } else {
                            i = 1;
                        }
                        if (zzlb.zzai(zzfsVar8.zzo()) && zzq) {
                            zzam zzamVar5 = this.zze;
                            zzal(zzamVar5);
                            if (zzamVar5.zzl(zza(), zzkqVar3.zza.zzx(), false, false, true, false, false).zzc > zzg().zze(zzkqVar3.zza.zzx(), zzdu.zzm)) {
                                zzay().zzk().zzb("Too many conversions. Not logging as conversion. appId", zzeh.zzn(zzkqVar3.zza.zzx()));
                                com.google.android.gms.internal.measurement.zzfw zzfwVar3 = null;
                                boolean z3 = false;
                                int i23 = -1;
                                for (int i24 = 0; i24 < zzfsVar8.zza(); i24++) {
                                    com.google.android.gms.internal.measurement.zzfx zzn = zzfsVar8.zzn(i24);
                                    if ("_c".equals(zzn.zzg())) {
                                        zzfwVar3 = (com.google.android.gms.internal.measurement.zzfw) zzn.zzby();
                                        i23 = i24;
                                    } else if ("_err".equals(zzn.zzg())) {
                                        z3 = true;
                                    }
                                }
                                if (z3) {
                                    if (zzfwVar3 != null) {
                                        zzfsVar8.zzh(i23);
                                    } else {
                                        zzfwVar3 = null;
                                    }
                                }
                                if (zzfwVar3 != null) {
                                    com.google.android.gms.internal.measurement.zzfw zzfwVar4 = (com.google.android.gms.internal.measurement.zzfw) zzfwVar3.zzau();
                                    zzfwVar4.zzj("_err");
                                    zzfwVar4.zzi(10L);
                                    zzfsVar8.zzk(i23, (com.google.android.gms.internal.measurement.zzfx) zzfwVar4.zzaC());
                                } else {
                                    zzay().zzd().zzb("Did not find conversion parameter. appId", zzeh.zzn(zzkqVar3.zza.zzx()));
                                }
                            }
                        }
                        if (zzq) {
                        }
                        if (!"_e".equals(zzfsVar8.zzo())) {
                        }
                        i7 = i17;
                        zzkqVar3.zzc.set(i7, (com.google.android.gms.internal.measurement.zzft) zzfsVar8.zzaC());
                        i10 = i14 + 1;
                        zzgcVar5.zzk(zzfsVar8);
                        i13 = i;
                    }
                    i9 = i7 + 1;
                    zzgcVar6 = zzgcVar5;
                    str4 = str2;
                    str5 = str6;
                    zzfsVar6 = zzfsVar;
                }
                com.google.android.gms.internal.measurement.zzgc zzgcVar7 = zzgcVar6;
                long j3 = 0;
                int i25 = 0;
                while (i25 < i10) {
                    com.google.android.gms.internal.measurement.zzft zze3 = zzgcVar7.zze(i25);
                    if ("_e".equals(zze3.zzh())) {
                        zzal(this.zzi);
                        if (zzkv.zzB(zze3, "_fr") != null) {
                            zzgcVar7.zzA(i25);
                            i10--;
                            i25--;
                            i25++;
                        }
                    }
                    zzal(this.zzi);
                    com.google.android.gms.internal.measurement.zzfx zzB = zzkv.zzB(zze3, "_et");
                    if (zzB != null) {
                        Long valueOf = zzB.zzw() ? Long.valueOf(zzB.zzd()) : null;
                        if (valueOf != null && valueOf.longValue() > 0) {
                            j3 += valueOf.longValue();
                        }
                    }
                    i25++;
                }
                zzaf(zzgcVar7, j3, false);
                Iterator it = zzgcVar7.zzas().iterator();
                while (true) {
                    if (it.hasNext()) {
                        if ("_s".equals(((com.google.android.gms.internal.measurement.zzft) it.next()).zzh())) {
                            zzam zzamVar6 = this.zze;
                            zzal(zzamVar6);
                            zzamVar6.zzA(zzgcVar7.zzap(), "_se");
                            break;
                        }
                    } else {
                        break;
                    }
                }
                if (zzkv.zza(zzgcVar7, "_sid") >= 0) {
                    zzaf(zzgcVar7, j3, true);
                } else {
                    int zza = zzkv.zza(zzgcVar7, "_se");
                    if (zza >= 0) {
                        zzgcVar7.zzB(zza);
                        zzay().zzd().zzb("Session engagement user property is in the bundle without session ID. appId", zzeh.zzn(zzkqVar3.zza.zzx()));
                    }
                }
                zzkv zzkvVar = this.zzi;
                zzal(zzkvVar);
                zzkvVar.zzt.zzay().zzj().zza("Checking account type status for ad personalization signals");
                zzfi zzfiVar5 = zzkvVar.zzf.zzc;
                zzal(zzfiVar5);
                if (zzfiVar5.zzn(zzgcVar7.zzap())) {
                    zzam zzamVar7 = zzkvVar.zzf.zze;
                    zzal(zzamVar7);
                    zzh zzj = zzamVar7.zzj(zzgcVar7.zzap());
                    if (zzj != null && zzj.zzah() && zzkvVar.zzt.zzg().zze()) {
                        zzkvVar.zzt.zzay().zzc().zza("Turning off ad personalization due to account type");
                        com.google.android.gms.internal.measurement.zzgl zzd = com.google.android.gms.internal.measurement.zzgm.zzd();
                        zzd.zzf(str2);
                        zzd.zzg(zzkvVar.zzt.zzg().zza());
                        zzd.zze(1L);
                        com.google.android.gms.internal.measurement.zzgm zzgmVar = (com.google.android.gms.internal.measurement.zzgm) zzd.zzaC();
                        int i26 = 0;
                        while (true) {
                            if (i26 < zzgcVar7.zzb()) {
                                if (str2.equals(zzgcVar7.zzao(i26).zzf())) {
                                    zzgcVar7.zzam(i26, zzgmVar);
                                    break;
                                }
                                i26++;
                            } else {
                                zzgcVar7.zzm(zzgmVar);
                                break;
                            }
                        }
                    }
                }
                zzgcVar7.zzai(RecyclerView.FOREVER_NS);
                zzgcVar7.zzQ(Long.MIN_VALUE);
                for (int i27 = 0; i27 < zzgcVar7.zza(); i27++) {
                    com.google.android.gms.internal.measurement.zzft zze4 = zzgcVar7.zze(i27);
                    if (zze4.zzd() < zzgcVar7.zzd()) {
                        zzgcVar7.zzai(zze4.zzd());
                    }
                    if (zze4.zzd() > zzgcVar7.zzc()) {
                        zzgcVar7.zzQ(zze4.zzd());
                    }
                }
                zzgcVar7.zzz();
                zzgcVar7.zzo();
                zzaa zzaaVar = this.zzh;
                zzal(zzaaVar);
                zzgcVar7.zzf(zzaaVar.zza(zzgcVar7.zzap(), zzgcVar7.zzas(), zzgcVar7.zzat(), Long.valueOf(zzgcVar7.zzd()), Long.valueOf(zzgcVar7.zzc())));
                if (zzg().zzw(zzkqVar3.zza.zzx())) {
                    HashMap hashMap3 = new HashMap();
                    ArrayList arrayList2 = new ArrayList();
                    SecureRandom zzG = zzv().zzG();
                    int i28 = 0;
                    while (i28 < zzgcVar7.zza()) {
                        com.google.android.gms.internal.measurement.zzfs zzfsVar11 = (com.google.android.gms.internal.measurement.zzfs) zzgcVar7.zze(i28).zzby();
                        if (zzfsVar11.zzo().equals("_ep")) {
                            zzal(this.zzi);
                            String str7 = (String) zzkv.zzC((com.google.android.gms.internal.measurement.zzft) zzfsVar11.zzaC(), "_en");
                            zzas zzasVar = (zzas) hashMap3.get(str7);
                            if (zzasVar == null) {
                                zzam zzamVar8 = this.zze;
                                zzal(zzamVar8);
                                zzasVar = zzamVar8.zzn(zzkqVar3.zza.zzx(), (String) Preconditions.checkNotNull(str7));
                                if (zzasVar != null) {
                                    hashMap3.put(str7, zzasVar);
                                }
                            }
                            if (zzasVar != null && zzasVar.zzi == null) {
                                Long l2 = zzasVar.zzj;
                                if (l2 != null && l2.longValue() > 1) {
                                    zzal(this.zzi);
                                    zzkv.zzz(zzfsVar11, "_sr", zzasVar.zzj);
                                }
                                Boolean bool = zzasVar.zzk;
                                if (bool != null && bool.booleanValue()) {
                                    zzal(this.zzi);
                                    zzkv.zzz(zzfsVar11, "_efs", 1L);
                                }
                                arrayList2.add((com.google.android.gms.internal.measurement.zzft) zzfsVar11.zzaC());
                            }
                            zzgcVar7.zzS(i28, zzfsVar11);
                        } else {
                            zzfi zzfiVar6 = this.zzc;
                            zzal(zzfiVar6);
                            String zzx = zzkqVar3.zza.zzx();
                            String zza2 = zzfiVar6.zza(zzx, "measurement.account.time_zone_offset_minutes");
                            if (!TextUtils.isEmpty(zza2)) {
                                try {
                                    parseLong = Long.parseLong(zza2);
                                } catch (NumberFormatException e2) {
                                    zzfiVar6.zzt.zzay().zzk().zzc("Unable to parse timezone offset. appId", zzeh.zzn(zzx), e2);
                                }
                                long zzr22 = zzv().zzr(zzfsVar11.zzc(), parseLong);
                                com.google.android.gms.internal.measurement.zzft zzftVar2 = (com.google.android.gms.internal.measurement.zzft) zzfsVar11.zzaC();
                                Long l32 = 1L;
                                long j42 = parseLong;
                                if (!TextUtils.isEmpty("_dbg")) {
                                    Iterator it2 = zzftVar2.zzi().iterator();
                                    while (true) {
                                        if (!it2.hasNext()) {
                                            break;
                                        }
                                        com.google.android.gms.internal.measurement.zzfx zzfxVar = (com.google.android.gms.internal.measurement.zzfx) it2.next();
                                        Iterator it3 = it2;
                                        if (!"_dbg".equals(zzfxVar.zzg())) {
                                            it2 = it3;
                                        } else if (l32.equals(Long.valueOf(zzfxVar.zzd()))) {
                                            zzc = 1;
                                        }
                                    }
                                }
                                zzfi zzfiVar72 = this.zzc;
                                zzal(zzfiVar72);
                                zzc = zzfiVar72.zzc(zzkqVar3.zza.zzx(), zzfsVar11.zzo());
                                if (zzc > 0) {
                                    zzay().zzk().zzc("Sample rate must be positive. event, rate", zzfsVar11.zzo(), Integer.valueOf(zzc));
                                    arrayList2.add((com.google.android.gms.internal.measurement.zzft) zzfsVar11.zzaC());
                                    zzgcVar7.zzS(i28, zzfsVar11);
                                } else {
                                    zzas zzasVar2 = (zzas) hashMap3.get(zzfsVar11.zzo());
                                    if (zzasVar2 == null) {
                                        zzam zzamVar9 = this.zze;
                                        zzal(zzamVar9);
                                        zzasVar2 = zzamVar9.zzn(zzkqVar3.zza.zzx(), zzfsVar11.zzo());
                                        if (zzasVar2 == null) {
                                            j2 = zzr22;
                                            zzay().zzk().zzc("Event being bundled has no eventAggregate. appId, eventName", zzkqVar3.zza.zzx(), zzfsVar11.zzo());
                                            zzasVar2 = new zzas(zzkqVar3.zza.zzx(), zzfsVar11.zzo(), 1L, 1L, 1L, zzfsVar11.zzc(), 0L, null, null, null, null);
                                            zzal(this.zzi);
                                            Long l42 = (Long) zzkv.zzC((com.google.android.gms.internal.measurement.zzft) zzfsVar11.zzaC(), "_eid");
                                            Boolean valueOf22 = Boolean.valueOf(l42 == null);
                                            if (zzc != 1) {
                                                arrayList2.add((com.google.android.gms.internal.measurement.zzft) zzfsVar11.zzaC());
                                                if (valueOf22.booleanValue() && (zzasVar2.zzi != null || zzasVar2.zzj != null || zzasVar2.zzk != null)) {
                                                    hashMap3.put(zzfsVar11.zzo(), zzasVar2.zza(null, null, null));
                                                }
                                                zzgcVar7.zzS(i28, zzfsVar11);
                                            } else {
                                                if (zzG.nextInt(zzc) == 0) {
                                                    zzal(this.zzi);
                                                    Long valueOf3 = Long.valueOf(zzc);
                                                    zzkv.zzz(zzfsVar11, "_sr", valueOf3);
                                                    arrayList2.add((com.google.android.gms.internal.measurement.zzft) zzfsVar11.zzaC());
                                                    if (valueOf22.booleanValue()) {
                                                        zzasVar2 = zzasVar2.zza(null, valueOf3, null);
                                                    }
                                                    hashMap3.put(zzfsVar11.zzo(), zzasVar2.zzb(zzfsVar11.zzc(), j2));
                                                    zzkqVar2 = zzkqVar3;
                                                    secureRandom = zzG;
                                                    zzgcVar3 = zzgcVar7;
                                                    hashMap2 = hashMap3;
                                                } else {
                                                    long j5 = j2;
                                                    secureRandom = zzG;
                                                    Long l5 = zzasVar2.zzh;
                                                    if (l5 != null) {
                                                        zzr = l5.longValue();
                                                        zzkqVar2 = zzkqVar3;
                                                        hashMap = hashMap3;
                                                        zzgcVar2 = zzgcVar7;
                                                        l = l42;
                                                    } else {
                                                        zzgcVar2 = zzgcVar7;
                                                        l = l42;
                                                        zzkqVar2 = zzkqVar3;
                                                        hashMap = hashMap3;
                                                        zzr = zzv().zzr(zzfsVar11.zzb(), j42);
                                                    }
                                                    if (zzr != j5) {
                                                        zzal(this.zzi);
                                                        zzkv.zzz(zzfsVar11, "_efs", 1L);
                                                        zzal(this.zzi);
                                                        Long valueOf4 = Long.valueOf(zzc);
                                                        zzkv.zzz(zzfsVar11, "_sr", valueOf4);
                                                        arrayList2.add((com.google.android.gms.internal.measurement.zzft) zzfsVar11.zzaC());
                                                        if (valueOf22.booleanValue()) {
                                                            zzasVar2 = zzasVar2.zza(null, valueOf4, Boolean.TRUE);
                                                        }
                                                        hashMap2 = hashMap;
                                                        hashMap2.put(zzfsVar11.zzo(), zzasVar2.zzb(zzfsVar11.zzc(), j5));
                                                    } else {
                                                        hashMap2 = hashMap;
                                                        if (valueOf22.booleanValue()) {
                                                            hashMap2.put(zzfsVar11.zzo(), zzasVar2.zza(l, null, null));
                                                        }
                                                    }
                                                    zzgcVar3 = zzgcVar2;
                                                }
                                                zzgcVar3.zzS(i28, zzfsVar11);
                                                i28++;
                                                zzgcVar7 = zzgcVar3;
                                                hashMap3 = hashMap2;
                                                zzG = secureRandom;
                                                zzkqVar3 = zzkqVar2;
                                            }
                                        }
                                    }
                                    j2 = zzr22;
                                    zzal(this.zzi);
                                    Long l422 = (Long) zzkv.zzC((com.google.android.gms.internal.measurement.zzft) zzfsVar11.zzaC(), "_eid");
                                    Boolean valueOf222 = Boolean.valueOf(l422 == null);
                                    if (zzc != 1) {
                                    }
                                }
                            }
                            parseLong = 0;
                            long zzr222 = zzv().zzr(zzfsVar11.zzc(), parseLong);
                            com.google.android.gms.internal.measurement.zzft zzftVar22 = (com.google.android.gms.internal.measurement.zzft) zzfsVar11.zzaC();
                            Long l322 = 1L;
                            long j422 = parseLong;
                            if (!TextUtils.isEmpty("_dbg")) {
                            }
                            zzfi zzfiVar722 = this.zzc;
                            zzal(zzfiVar722);
                            zzc = zzfiVar722.zzc(zzkqVar3.zza.zzx(), zzfsVar11.zzo());
                            if (zzc > 0) {
                            }
                        }
                        zzkqVar2 = zzkqVar3;
                        secureRandom = zzG;
                        zzgcVar3 = zzgcVar7;
                        hashMap2 = hashMap3;
                        i28++;
                        zzgcVar7 = zzgcVar3;
                        hashMap3 = hashMap2;
                        zzG = secureRandom;
                        zzkqVar3 = zzkqVar2;
                    }
                    zzkq zzkqVar4 = zzkqVar3;
                    HashMap hashMap4 = hashMap3;
                    zzgcVar = zzgcVar7;
                    if (arrayList2.size() < zzgcVar.zza()) {
                        zzgcVar.zzr();
                        zzgcVar.zzg(arrayList2);
                    }
                    for (Map.Entry entry : hashMap4.entrySet()) {
                        zzam zzamVar10 = this.zze;
                        zzal(zzamVar10);
                        zzamVar10.zzE((zzas) entry.getValue());
                    }
                    zzkqVar = zzkqVar4;
                } else {
                    zzgcVar = zzgcVar7;
                    zzkqVar = zzkqVar3;
                }
                String zzx2 = zzkqVar.zza.zzx();
                zzam zzamVar11 = this.zze;
                zzal(zzamVar11);
                zzh zzj2 = zzamVar11.zzj(zzx2);
                if (zzj2 == null) {
                    zzay().zzd().zzb("Bundling raw events w/o app info. appId", zzeh.zzn(zzkqVar.zza.zzx()));
                } else if (zzgcVar.zza() > 0) {
                    long zzn2 = zzj2.zzn();
                    if (zzn2 != 0) {
                        zzgcVar.zzab(zzn2);
                    } else {
                        zzgcVar.zzv();
                    }
                    long zzp = zzj2.zzp();
                    if (zzp != 0) {
                        zzn2 = zzp;
                    }
                    if (zzn2 != 0) {
                        zzgcVar.zzac(zzn2);
                    } else {
                        zzgcVar.zzw();
                    }
                    zzj2.zzE();
                    zzgcVar.zzI((int) zzj2.zzo());
                    zzj2.zzab(zzgcVar.zzd());
                    zzj2.zzZ(zzgcVar.zzc());
                    String zzs = zzj2.zzs();
                    if (zzs != null) {
                        zzgcVar.zzW(zzs);
                    } else {
                        zzgcVar.zzs();
                    }
                    zzam zzamVar12 = this.zze;
                    zzal(zzamVar12);
                    zzamVar12.zzD(zzj2);
                }
                if (zzgcVar.zza() > 0) {
                    this.zzn.zzaw();
                    zzfi zzfiVar8 = this.zzc;
                    zzal(zzfiVar8);
                    com.google.android.gms.internal.measurement.zzff zze5 = zzfiVar8.zze(zzkqVar.zza.zzx());
                    try {
                        try {
                            if (zze5 != null && zze5.zzs()) {
                                zzgcVar.zzK(zze5.zzc());
                                zzamVar = this.zze;
                                zzal(zzamVar);
                                zzgdVar = (com.google.android.gms.internal.measurement.zzgd) zzgcVar.zzaC();
                                zzamVar.zzg();
                                zzamVar.zzW();
                                Preconditions.checkNotNull(zzgdVar);
                                Preconditions.checkNotEmpty(zzgdVar.zzx());
                                Preconditions.checkState(zzgdVar.zzbe());
                                zzamVar.zzz();
                                currentTimeMillis = zzamVar.zzt.zzav().currentTimeMillis();
                                zzk = zzgdVar.zzk();
                                zzamVar.zzt.zzf();
                                if (zzk >= currentTimeMillis - zzag.zzA()) {
                                    long zzk2 = zzgdVar.zzk();
                                    zzamVar.zzt.zzf();
                                }
                                zzamVar.zzt.zzay().zzk().zzd("Storing bundle outside of the max uploading time span. appId, now, timestamp", zzeh.zzn(zzgdVar.zzx()), Long.valueOf(currentTimeMillis), Long.valueOf(zzgdVar.zzk()));
                                byte[] zzbu2 = zzgdVar.zzbu();
                                zzkv zzkvVar22 = zzamVar.zzf.zzi;
                                zzal(zzkvVar22);
                                byte[] zzy2 = zzkvVar22.zzy(zzbu2);
                                zzamVar.zzt.zzay().zzj().zzb("Saving bundle, size", Integer.valueOf(zzy2.length));
                                contentValues = new ContentValues();
                                contentValues.put("app_id", zzgdVar.zzx());
                                contentValues.put("bundle_end_timestamp", Long.valueOf(zzgdVar.zzk()));
                                contentValues.put("data", zzy2);
                                contentValues.put("has_realtime", Integer.valueOf(i));
                                if (zzgdVar.zzbk()) {
                                    contentValues.put("retry_count", Integer.valueOf(zzgdVar.zze()));
                                }
                                if (zzamVar.zzh().insert("queue", null, contentValues) == -1) {
                                    zzamVar.zzt.zzay().zzd().zzb("Failed to insert bundle (got -1). appId", zzeh.zzn(zzgdVar.zzx()));
                                }
                            }
                            if (zzamVar.zzh().insert("queue", null, contentValues) == -1) {
                            }
                        } catch (SQLiteException e3) {
                            zzamVar.zzt.zzay().zzd().zzc("Error storing bundle. appId", zzeh.zzn(zzgdVar.zzx()), e3);
                        }
                        zzkv zzkvVar222 = zzamVar.zzf.zzi;
                        zzal(zzkvVar222);
                        byte[] zzy22 = zzkvVar222.zzy(zzbu2);
                        zzamVar.zzt.zzay().zzj().zzb("Saving bundle, size", Integer.valueOf(zzy22.length));
                        contentValues = new ContentValues();
                        contentValues.put("app_id", zzgdVar.zzx());
                        contentValues.put("bundle_end_timestamp", Long.valueOf(zzgdVar.zzk()));
                        contentValues.put("data", zzy22);
                        contentValues.put("has_realtime", Integer.valueOf(i));
                        if (zzgdVar.zzbk()) {
                        }
                    } catch (IOException e4) {
                        zzamVar.zzt.zzay().zzd().zzc("Data loss. Failed to serialize bundle. appId", zzeh.zzn(zzgdVar.zzx()), e4);
                    }
                    if (zzkqVar.zza.zzF().isEmpty()) {
                        zzgcVar.zzK(-1L);
                    } else {
                        zzay().zzk().zzb("Did not find measurement config or missing version info. appId", zzeh.zzn(zzkqVar.zza.zzx()));
                    }
                    zzamVar = this.zze;
                    zzal(zzamVar);
                    zzgdVar = (com.google.android.gms.internal.measurement.zzgd) zzgcVar.zzaC();
                    zzamVar.zzg();
                    zzamVar.zzW();
                    Preconditions.checkNotNull(zzgdVar);
                    Preconditions.checkNotEmpty(zzgdVar.zzx());
                    Preconditions.checkState(zzgdVar.zzbe());
                    zzamVar.zzz();
                    currentTimeMillis = zzamVar.zzt.zzav().currentTimeMillis();
                    zzk = zzgdVar.zzk();
                    zzamVar.zzt.zzf();
                    if (zzk >= currentTimeMillis - zzag.zzA()) {
                    }
                    zzamVar.zzt.zzay().zzk().zzd("Storing bundle outside of the max uploading time span. appId, now, timestamp", zzeh.zzn(zzgdVar.zzx()), Long.valueOf(currentTimeMillis), Long.valueOf(zzgdVar.zzk()));
                    byte[] zzbu22 = zzgdVar.zzbu();
                }
                zzam zzamVar13 = this.zze;
                zzal(zzamVar13);
                List list2 = zzkqVar.zzb;
                Preconditions.checkNotNull(list2);
                zzamVar13.zzg();
                zzamVar13.zzW();
                StringBuilder sb = new StringBuilder("rowid in (");
                for (int i29 = 0; i29 < list2.size(); i29++) {
                    if (i29 != 0) {
                        sb.append(",");
                    }
                    sb.append(((Long) list2.get(i29)).longValue());
                }
                sb.append(")");
                int delete = zzamVar13.zzh().delete("raw_events", sb.toString(), null);
                if (delete != list2.size()) {
                    zzamVar13.zzt.zzay().zzd().zzc("Deleted fewer rows from raw events table than expected", Integer.valueOf(delete), Integer.valueOf(list2.size()));
                }
                zzam zzamVar14 = this.zze;
                zzal(zzamVar14);
                try {
                    zzamVar14.zzh().execSQL("delete from raw_events_metadata where app_id=? and metadata_fingerprint not in (select distinct metadata_fingerprint from raw_events where app_id=?)", new String[]{zzx2, zzx2});
                } catch (SQLiteException e5) {
                    zzamVar14.zzt.zzay().zzd().zzc("Failed to remove unused event metadata. appId", zzeh.zzn(zzx2), e5);
                }
                zzam zzamVar15 = this.zze;
                zzal(zzamVar15);
                zzamVar15.zzC();
                zzam zzamVar16 = this.zze;
                zzal(zzamVar16);
                zzamVar16.zzx();
                return true;
            }
            zzam zzamVar17 = this.zze;
            zzal(zzamVar17);
            zzamVar17.zzC();
            zzam zzamVar18 = this.zze;
            zzal(zzamVar18);
            zzamVar18.zzx();
            return false;
        } catch (Throwable th) {
            zzam zzamVar19 = this.zze;
            zzal(zzamVar19);
            zzamVar19.zzx();
            throw th;
        }
    }

    private final boolean zzai() {
        zzaz().zzg();
        zzB();
        zzam zzamVar = this.zze;
        zzal(zzamVar);
        if (zzamVar.zzF()) {
            return true;
        }
        zzam zzamVar2 = this.zze;
        zzal(zzamVar2);
        return !TextUtils.isEmpty(zzamVar2.zzr());
    }

    private final boolean zzaj(com.google.android.gms.internal.measurement.zzfs zzfsVar, com.google.android.gms.internal.measurement.zzfs zzfsVar2) {
        Preconditions.checkArgument("_e".equals(zzfsVar.zzo()));
        zzal(this.zzi);
        com.google.android.gms.internal.measurement.zzfx zzB = zzkv.zzB((com.google.android.gms.internal.measurement.zzft) zzfsVar.zzaC(), "_sc");
        String zzh = zzB == null ? null : zzB.zzh();
        zzal(this.zzi);
        com.google.android.gms.internal.measurement.zzfx zzB2 = zzkv.zzB((com.google.android.gms.internal.measurement.zzft) zzfsVar2.zzaC(), "_pc");
        String zzh2 = zzB2 != null ? zzB2.zzh() : null;
        if (zzh2 == null || !zzh2.equals(zzh)) {
            return false;
        }
        Preconditions.checkArgument("_e".equals(zzfsVar.zzo()));
        zzal(this.zzi);
        com.google.android.gms.internal.measurement.zzfx zzB3 = zzkv.zzB((com.google.android.gms.internal.measurement.zzft) zzfsVar.zzaC(), "_et");
        if (zzB3 == null || !zzB3.zzw() || zzB3.zzd() <= 0) {
            return true;
        }
        long zzd = zzB3.zzd();
        zzal(this.zzi);
        com.google.android.gms.internal.measurement.zzfx zzB4 = zzkv.zzB((com.google.android.gms.internal.measurement.zzft) zzfsVar2.zzaC(), "_et");
        if (zzB4 != null && zzB4.zzd() > 0) {
            zzd += zzB4.zzd();
        }
        zzal(this.zzi);
        zzkv.zzz(zzfsVar2, "_et", Long.valueOf(zzd));
        zzal(this.zzi);
        zzkv.zzz(zzfsVar, "_fr", 1L);
        return true;
    }

    private static final boolean zzak(zzq zzqVar) {
        return (TextUtils.isEmpty(zzqVar.zzb) && TextUtils.isEmpty(zzqVar.zzq)) ? false : true;
    }

    private static final zzkh zzal(zzkh zzkhVar) {
        if (zzkhVar != null) {
            if (zzkhVar.zzY()) {
                return zzkhVar;
            }
            throw new IllegalStateException("Component not initialized: ".concat(String.valueOf(zzkhVar.getClass())));
        }
        throw new IllegalStateException("Upload Component not created");
    }

    public static zzkt zzt(Context context) {
        Preconditions.checkNotNull(context);
        Preconditions.checkNotNull(context.getApplicationContext());
        if (zzb == null) {
            synchronized (zzkt.class) {
                if (zzb == null) {
                    zzb = new zzkt((zzku) Preconditions.checkNotNull(new zzku(context)), null);
                }
            }
        }
        return zzb;
    }

    public static /* bridge */ /* synthetic */ void zzy(zzkt zzktVar, zzku zzkuVar) {
        zzktVar.zzaz().zzg();
        zzktVar.zzm = new zzez(zzktVar);
        zzam zzamVar = new zzam(zzktVar);
        zzamVar.zzX();
        zzktVar.zze = zzamVar;
        zzktVar.zzg().zzq((zzaf) Preconditions.checkNotNull(zzktVar.zzc));
        zzjo zzjoVar = new zzjo(zzktVar);
        zzjoVar.zzX();
        zzktVar.zzk = zzjoVar;
        zzaa zzaaVar = new zzaa(zzktVar);
        zzaaVar.zzX();
        zzktVar.zzh = zzaaVar;
        zzic zzicVar = new zzic(zzktVar);
        zzicVar.zzX();
        zzktVar.zzj = zzicVar;
        zzkf zzkfVar = new zzkf(zzktVar);
        zzkfVar.zzX();
        zzktVar.zzg = zzkfVar;
        zzktVar.zzf = new zzep(zzktVar);
        if (zzktVar.zzr != zzktVar.zzs) {
            zzktVar.zzay().zzd().zzc("Not all upload components initialized", Integer.valueOf(zzktVar.zzr), Integer.valueOf(zzktVar.zzs));
        }
        zzktVar.zzo = true;
    }

    @VisibleForTesting
    public final void zzA() {
        zzaz().zzg();
        zzB();
        if (this.zzp) {
            return;
        }
        this.zzp = true;
        if (zzZ()) {
            FileChannel fileChannel = this.zzx;
            zzaz().zzg();
            int i = 0;
            if (fileChannel != null && fileChannel.isOpen()) {
                ByteBuffer allocate = ByteBuffer.allocate(4);
                try {
                    fileChannel.position(0L);
                    int read = fileChannel.read(allocate);
                    if (read == 4) {
                        allocate.flip();
                        i = allocate.getInt();
                    } else if (read != -1) {
                        zzay().zzk().zzb("Unexpected data length. Bytes read", Integer.valueOf(read));
                    }
                } catch (IOException e2) {
                    zzay().zzd().zzb("Failed to read from channel", e2);
                }
            } else {
                zzay().zzd().zza("Bad channel to read from");
            }
            int zzi = this.zzn.zzh().zzi();
            zzaz().zzg();
            if (i > zzi) {
                zzay().zzd().zzc("Panic: can't downgrade version. Previous, current version", Integer.valueOf(i), Integer.valueOf(zzi));
            } else if (i < zzi) {
                FileChannel fileChannel2 = this.zzx;
                zzaz().zzg();
                if (fileChannel2 != null && fileChannel2.isOpen()) {
                    ByteBuffer allocate2 = ByteBuffer.allocate(4);
                    allocate2.putInt(zzi);
                    allocate2.flip();
                    try {
                        fileChannel2.truncate(0L);
                        fileChannel2.write(allocate2);
                        fileChannel2.force(true);
                        if (fileChannel2.size() != 4) {
                            zzay().zzd().zzb("Error writing to channel. Bytes written", Long.valueOf(fileChannel2.size()));
                        }
                        zzay().zzj().zzc("Storage version upgraded. Previous, current version", Integer.valueOf(i), Integer.valueOf(zzi));
                        return;
                    } catch (IOException e3) {
                        zzay().zzd().zzb("Failed to write to channel", e3);
                    }
                } else {
                    zzay().zzd().zza("Bad channel to read from");
                }
                zzay().zzd().zzc("Storage version upgrade failed. Previous, current version", Integer.valueOf(i), Integer.valueOf(zzi));
            }
        }
    }

    public final void zzB() {
        if (!this.zzo) {
            throw new IllegalStateException("UploadController is not initialized");
        }
    }

    /* JADX WARN: Code restructure failed: missing block: B:31:0x00a5, code lost:
        if ((zzg().zzi(r6, com.google.android.gms.measurement.internal.zzdu.zzR) + r0.zzb) < zzav().elapsedRealtime()) goto L32;
     */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public final void zzC(String str, com.google.android.gms.internal.measurement.zzgc zzgcVar) {
        int zza;
        int indexOf;
        zzfi zzfiVar = this.zzc;
        zzal(zzfiVar);
        Set zzk = zzfiVar.zzk(str);
        if (zzk != null) {
            zzgcVar.zzi(zzk);
        }
        zzfi zzfiVar2 = this.zzc;
        zzal(zzfiVar2);
        if (zzfiVar2.zzv(str)) {
            zzgcVar.zzp();
        }
        zzfi zzfiVar3 = this.zzc;
        zzal(zzfiVar3);
        if (zzfiVar3.zzy(str)) {
            if (zzg().zzs(str, zzdu.zzaq)) {
                String zzar = zzgcVar.zzar();
                if (!TextUtils.isEmpty(zzar) && (indexOf = zzar.indexOf(".")) != -1) {
                    zzgcVar.zzY(zzar.substring(0, indexOf));
                }
            } else {
                zzgcVar.zzu();
            }
        }
        zzfi zzfiVar4 = this.zzc;
        zzal(zzfiVar4);
        if (zzfiVar4.zzz(str) && (zza = zzkv.zza(zzgcVar, "_id")) != -1) {
            zzgcVar.zzB(zza);
        }
        zzfi zzfiVar5 = this.zzc;
        zzal(zzfiVar5);
        if (zzfiVar5.zzx(str)) {
            zzgcVar.zzq();
        }
        zzfi zzfiVar6 = this.zzc;
        zzal(zzfiVar6);
        if (zzfiVar6.zzu(str)) {
            zzgcVar.zzn();
            zzks zzksVar = (zzks) this.zzC.get(str);
            if (zzksVar != null) {
            }
            zzksVar = new zzks(this);
            this.zzC.put(str, zzksVar);
            zzgcVar.zzR(zzksVar.zza);
        }
        zzfi zzfiVar7 = this.zzc;
        zzal(zzfiVar7);
        if (zzfiVar7.zzw(str)) {
            zzgcVar.zzy();
        }
    }

    public final void zzD(zzh zzhVar) {
        a aVar;
        a aVar2;
        zzaz().zzg();
        if (TextUtils.isEmpty(zzhVar.zzy()) && TextUtils.isEmpty(zzhVar.zzr())) {
            zzI((String) Preconditions.checkNotNull(zzhVar.zzt()), 204, null, null, null);
            return;
        }
        zzki zzkiVar = this.zzl;
        Uri.Builder builder = new Uri.Builder();
        String zzy = zzhVar.zzy();
        if (TextUtils.isEmpty(zzy)) {
            zzy = zzhVar.zzr();
        }
        a aVar3 = null;
        Uri.Builder appendQueryParameter = builder.scheme((String) zzdu.zzd.zza(null)).encodedAuthority((String) zzdu.zze.zza(null)).path("config/app/".concat(String.valueOf(zzy))).appendQueryParameter("platform", DefaultSettingsSpiCall.ANDROID_CLIENT_TYPE);
        zzkiVar.zzt.zzf().zzh();
        appendQueryParameter.appendQueryParameter("gmp_version", String.valueOf(74029L)).appendQueryParameter("runtime_version", CrashlyticsReportDataCapture.SIGNAL_DEFAULT);
        String uri = builder.build().toString();
        try {
            String str = (String) Preconditions.checkNotNull(zzhVar.zzt());
            URL url = new URL(uri);
            zzay().zzj().zzb("Fetching remote configuration", str);
            zzfi zzfiVar = this.zzc;
            zzal(zzfiVar);
            com.google.android.gms.internal.measurement.zzff zze = zzfiVar.zze(str);
            zzfi zzfiVar2 = this.zzc;
            zzal(zzfiVar2);
            String zzh = zzfiVar2.zzh(str);
            if (zze != null) {
                if (TextUtils.isEmpty(zzh)) {
                    aVar2 = null;
                } else {
                    aVar2 = new a();
                    aVar2.put(HttpHeaders.IF_MODIFIED_SINCE, zzh);
                }
                zzox.zzc();
                if (zzg().zzs(null, zzdu.zzao)) {
                    zzfi zzfiVar3 = this.zzc;
                    zzal(zzfiVar3);
                    String zzf = zzfiVar3.zzf(str);
                    if (!TextUtils.isEmpty(zzf)) {
                        if (aVar2 == null) {
                            aVar2 = new a();
                        }
                        aVar3 = aVar2;
                        aVar3.put(HttpHeaders.IF_NONE_MATCH, zzf);
                    }
                }
                aVar = aVar2;
                this.zzt = true;
                zzen zzenVar2 = this.zzd;
                zzal(zzenVar2);
                zzkl zzklVar2 = new zzkl(this);
                zzenVar2.zzg();
                zzenVar2.zzW();
                Preconditions.checkNotNull(url);
                Preconditions.checkNotNull(zzklVar2);
                zzenVar2.zzt.zzaz().zzo(new zzem(zzenVar2, str, url, null, aVar, zzklVar2));
            }
            aVar = aVar3;
            this.zzt = true;
            zzen zzenVar22 = this.zzd;
            zzal(zzenVar22);
            zzkl zzklVar22 = new zzkl(this);
            zzenVar22.zzg();
            zzenVar22.zzW();
            Preconditions.checkNotNull(url);
            Preconditions.checkNotNull(zzklVar22);
            zzenVar22.zzt.zzaz().zzo(new zzem(zzenVar22, str, url, null, aVar, zzklVar22));
        } catch (MalformedURLException unused) {
            zzay().zzd().zzc("Failed to parse config URL. Not fetching. appId", zzeh.zzn(zzhVar.zzt()), uri);
        }
    }

    public final void zzE(zzaw zzawVar, zzq zzqVar) {
        zzaw zzawVar2;
        List<zzac> zzt;
        List<zzac> zzt2;
        List<zzac> zzt3;
        String str;
        Preconditions.checkNotNull(zzqVar);
        Preconditions.checkNotEmpty(zzqVar.zza);
        zzaz().zzg();
        zzB();
        String str2 = zzqVar.zza;
        long j = zzawVar.zzd;
        zzei zzb2 = zzei.zzb(zzawVar);
        zzaz().zzg();
        zzie zzieVar = null;
        if (this.zzD != null && (str = this.zzE) != null && str.equals(str2)) {
            zzieVar = this.zzD;
        }
        zzlb.zzK(zzieVar, zzb2.zzd, false);
        zzaw zza = zzb2.zza();
        zzal(this.zzi);
        if (zzkv.zzA(zza, zzqVar)) {
            if (!zzqVar.zzh) {
                zzd(zzqVar);
                return;
            }
            List list = zzqVar.zzt;
            if (list == null) {
                zzawVar2 = zza;
            } else if (list.contains(zza.zza)) {
                Bundle zzc = zza.zzb.zzc();
                zzc.putLong("ga_safelisted", 1L);
                zzawVar2 = new zzaw(zza.zza, new zzau(zzc), zza.zzc, zza.zzd);
            } else {
                zzay().zzc().zzd("Dropping non-safelisted event. appId, event name, origin", str2, zza.zza, zza.zzc);
                return;
            }
            zzam zzamVar = this.zze;
            zzal(zzamVar);
            zzamVar.zzw();
            try {
                zzam zzamVar2 = this.zze;
                zzal(zzamVar2);
                Preconditions.checkNotEmpty(str2);
                zzamVar2.zzg();
                zzamVar2.zzW();
                int i = (j > 0L ? 1 : (j == 0L ? 0 : -1));
                if (i < 0) {
                    zzamVar2.zzt.zzay().zzk().zzc("Invalid time querying timed out conditional properties", zzeh.zzn(str2), Long.valueOf(j));
                    zzt = Collections.emptyList();
                } else {
                    zzt = zzamVar2.zzt("active=0 and app_id=? and abs(? - creation_timestamp) > trigger_timeout", new String[]{str2, String.valueOf(j)});
                }
                for (zzac zzacVar : zzt) {
                    if (zzacVar != null) {
                        zzay().zzj().zzd("User property timed out", zzacVar.zza, this.zzn.zzj().zzf(zzacVar.zzc.zzb), zzacVar.zzc.zza());
                        zzaw zzawVar3 = zzacVar.zzg;
                        if (zzawVar3 != null) {
                            zzY(new zzaw(zzawVar3, j), zzqVar);
                        }
                        zzam zzamVar3 = this.zze;
                        zzal(zzamVar3);
                        zzamVar3.zza(str2, zzacVar.zzc.zzb);
                    }
                }
                zzam zzamVar4 = this.zze;
                zzal(zzamVar4);
                Preconditions.checkNotEmpty(str2);
                zzamVar4.zzg();
                zzamVar4.zzW();
                if (i < 0) {
                    zzamVar4.zzt.zzay().zzk().zzc("Invalid time querying expired conditional properties", zzeh.zzn(str2), Long.valueOf(j));
                    zzt2 = Collections.emptyList();
                } else {
                    zzt2 = zzamVar4.zzt("active<>0 and app_id=? and abs(? - triggered_timestamp) > time_to_live", new String[]{str2, String.valueOf(j)});
                }
                ArrayList arrayList = new ArrayList(zzt2.size());
                for (zzac zzacVar2 : zzt2) {
                    if (zzacVar2 != null) {
                        zzay().zzj().zzd("User property expired", zzacVar2.zza, this.zzn.zzj().zzf(zzacVar2.zzc.zzb), zzacVar2.zzc.zza());
                        zzam zzamVar5 = this.zze;
                        zzal(zzamVar5);
                        zzamVar5.zzA(str2, zzacVar2.zzc.zzb);
                        zzaw zzawVar4 = zzacVar2.zzk;
                        if (zzawVar4 != null) {
                            arrayList.add(zzawVar4);
                        }
                        zzam zzamVar6 = this.zze;
                        zzal(zzamVar6);
                        zzamVar6.zza(str2, zzacVar2.zzc.zzb);
                    }
                }
                Iterator it = arrayList.iterator();
                while (it.hasNext()) {
                    zzY(new zzaw((zzaw) it.next(), j), zzqVar);
                }
                zzam zzamVar7 = this.zze;
                zzal(zzamVar7);
                String str3 = zzawVar2.zza;
                Preconditions.checkNotEmpty(str2);
                Preconditions.checkNotEmpty(str3);
                zzamVar7.zzg();
                zzamVar7.zzW();
                if (i < 0) {
                    zzamVar7.zzt.zzay().zzk().zzd("Invalid time querying triggered conditional properties", zzeh.zzn(str2), zzamVar7.zzt.zzj().zzd(str3), Long.valueOf(j));
                    zzt3 = Collections.emptyList();
                } else {
                    zzt3 = zzamVar7.zzt("active=0 and app_id=? and trigger_event_name=? and abs(? - creation_timestamp) <= trigger_timeout", new String[]{str2, str3, String.valueOf(j)});
                }
                ArrayList arrayList2 = new ArrayList(zzt3.size());
                for (zzac zzacVar3 : zzt3) {
                    if (zzacVar3 != null) {
                        zzkw zzkwVar = zzacVar3.zzc;
                        zzky zzkyVar = new zzky((String) Preconditions.checkNotNull(zzacVar3.zza), zzacVar3.zzb, zzkwVar.zzb, j, Preconditions.checkNotNull(zzkwVar.zza()));
                        zzam zzamVar8 = this.zze;
                        zzal(zzamVar8);
                        if (zzamVar8.zzL(zzkyVar)) {
                            zzay().zzj().zzd("User property triggered", zzacVar3.zza, this.zzn.zzj().zzf(zzkyVar.zzc), zzkyVar.zze);
                        } else {
                            zzay().zzd().zzd("Too many active user properties, ignoring", zzeh.zzn(zzacVar3.zza), this.zzn.zzj().zzf(zzkyVar.zzc), zzkyVar.zze);
                        }
                        zzaw zzawVar5 = zzacVar3.zzi;
                        if (zzawVar5 != null) {
                            arrayList2.add(zzawVar5);
                        }
                        zzacVar3.zzc = new zzkw(zzkyVar);
                        zzacVar3.zze = true;
                        zzam zzamVar9 = this.zze;
                        zzal(zzamVar9);
                        zzamVar9.zzK(zzacVar3);
                    }
                }
                zzY(zzawVar2, zzqVar);
                Iterator it2 = arrayList2.iterator();
                while (it2.hasNext()) {
                    zzY(new zzaw((zzaw) it2.next(), j), zzqVar);
                }
                zzam zzamVar10 = this.zze;
                zzal(zzamVar10);
                zzamVar10.zzC();
            } finally {
                zzam zzamVar11 = this.zze;
                zzal(zzamVar11);
                zzamVar11.zzx();
            }
        }
    }

    public final void zzF(zzaw zzawVar, String str) {
        zzam zzamVar = this.zze;
        zzal(zzamVar);
        zzh zzj = zzamVar.zzj(str);
        if (zzj != null && !TextUtils.isEmpty(zzj.zzw())) {
            Boolean zzad = zzad(zzj);
            if (zzad == null) {
                if (!"_ui".equals(zzawVar.zza)) {
                    zzay().zzk().zzb("Could not find package. appId", zzeh.zzn(str));
                }
            } else if (!zzad.booleanValue()) {
                zzay().zzd().zzb("App version does not match; dropping event. appId", zzeh.zzn(str));
                return;
            }
            String zzy = zzj.zzy();
            String zzw = zzj.zzw();
            long zzb2 = zzj.zzb();
            String zzv = zzj.zzv();
            long zzm = zzj.zzm();
            long zzj2 = zzj.zzj();
            boolean zzai = zzj.zzai();
            String zzx = zzj.zzx();
            zzj.zza();
            zzG(zzawVar, new zzq(str, zzy, zzw, zzb2, zzv, zzm, zzj2, (String) null, zzai, false, zzx, 0L, 0L, 0, zzj.zzah(), false, zzj.zzr(), zzj.zzq(), zzj.zzk(), zzj.zzC(), (String) null, zzh(str).zzh(), "", (String) null));
            return;
        }
        zzay().zzc().zzb("No app data available; dropping event", str);
    }

    public final void zzG(zzaw zzawVar, zzq zzqVar) {
        Preconditions.checkNotEmpty(zzqVar.zza);
        zzei zzb2 = zzei.zzb(zzawVar);
        zzlb zzv = zzv();
        Bundle bundle = zzb2.zzd;
        zzam zzamVar = this.zze;
        zzal(zzamVar);
        zzv.zzL(bundle, zzamVar.zzi(zzqVar.zza));
        zzv().zzM(zzb2, zzg().zzd(zzqVar.zza));
        zzaw zza = zzb2.zza();
        if ("_cmp".equals(zza.zza) && "referrer API v2".equals(zza.zzb.zzg("_cis"))) {
            String zzg = zza.zzb.zzg("gclid");
            if (!TextUtils.isEmpty(zzg)) {
                zzW(new zzkw("_lgclid", zza.zzd, zzg, "auto"), zzqVar);
            }
        }
        zzE(zza, zzqVar);
    }

    public final void zzH() {
        this.zzs++;
    }

    /* JADX WARN: Removed duplicated region for block: B:16:0x0049 A[Catch: all -> 0x0185, TryCatch #1 {all -> 0x018f, blocks: (B:4:0x0010, B:5:0x0012, B:64:0x0177, B:6:0x002c, B:16:0x0049, B:63:0x016f, B:21:0x0063, B:26:0x00b5, B:25:0x00a6, B:29:0x00bd, B:32:0x00c9, B:34:0x00cf, B:36:0x00d7, B:39:0x00e8, B:42:0x00f4, B:44:0x00fa, B:49:0x0107, B:53:0x0123, B:55:0x0138, B:57:0x0157, B:59:0x0162, B:61:0x0168, B:62:0x016c, B:56:0x0146, B:50:0x0110, B:52:0x011b), top: B:73:0x0010 }] */
    /* JADX WARN: Removed duplicated region for block: B:17:0x005c  */
    /* JADX WARN: Removed duplicated region for block: B:52:0x011b A[Catch: all -> 0x0185, TryCatch #1 {all -> 0x018f, blocks: (B:4:0x0010, B:5:0x0012, B:64:0x0177, B:6:0x002c, B:16:0x0049, B:63:0x016f, B:21:0x0063, B:26:0x00b5, B:25:0x00a6, B:29:0x00bd, B:32:0x00c9, B:34:0x00cf, B:36:0x00d7, B:39:0x00e8, B:42:0x00f4, B:44:0x00fa, B:49:0x0107, B:53:0x0123, B:55:0x0138, B:57:0x0157, B:59:0x0162, B:61:0x0168, B:62:0x016c, B:56:0x0146, B:50:0x0110, B:52:0x011b), top: B:73:0x0010 }] */
    /* JADX WARN: Removed duplicated region for block: B:55:0x0138 A[Catch: all -> 0x0185, TryCatch #1 {all -> 0x018f, blocks: (B:4:0x0010, B:5:0x0012, B:64:0x0177, B:6:0x002c, B:16:0x0049, B:63:0x016f, B:21:0x0063, B:26:0x00b5, B:25:0x00a6, B:29:0x00bd, B:32:0x00c9, B:34:0x00cf, B:36:0x00d7, B:39:0x00e8, B:42:0x00f4, B:44:0x00fa, B:49:0x0107, B:53:0x0123, B:55:0x0138, B:57:0x0157, B:59:0x0162, B:61:0x0168, B:62:0x016c, B:56:0x0146, B:50:0x0110, B:52:0x011b), top: B:73:0x0010 }] */
    /* JADX WARN: Removed duplicated region for block: B:56:0x0146 A[Catch: all -> 0x0185, TryCatch #1 {all -> 0x018f, blocks: (B:4:0x0010, B:5:0x0012, B:64:0x0177, B:6:0x002c, B:16:0x0049, B:63:0x016f, B:21:0x0063, B:26:0x00b5, B:25:0x00a6, B:29:0x00bd, B:32:0x00c9, B:34:0x00cf, B:36:0x00d7, B:39:0x00e8, B:42:0x00f4, B:44:0x00fa, B:49:0x0107, B:53:0x0123, B:55:0x0138, B:57:0x0157, B:59:0x0162, B:61:0x0168, B:62:0x016c, B:56:0x0146, B:50:0x0110, B:52:0x011b), top: B:73:0x0010 }] */
    /* JADX WARN: Removed duplicated region for block: B:59:0x0162 A[Catch: all -> 0x0185, TryCatch #1 {all -> 0x018f, blocks: (B:4:0x0010, B:5:0x0012, B:64:0x0177, B:6:0x002c, B:16:0x0049, B:63:0x016f, B:21:0x0063, B:26:0x00b5, B:25:0x00a6, B:29:0x00bd, B:32:0x00c9, B:34:0x00cf, B:36:0x00d7, B:39:0x00e8, B:42:0x00f4, B:44:0x00fa, B:49:0x0107, B:53:0x0123, B:55:0x0138, B:57:0x0157, B:59:0x0162, B:61:0x0168, B:62:0x016c, B:56:0x0146, B:50:0x0110, B:52:0x011b), top: B:73:0x0010 }] */
    @VisibleForTesting
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public final void zzI(String str, int i, Throwable th, byte[] bArr, Map map) {
        boolean z;
        String str2;
        zzfi zzfiVar;
        zzen zzenVar;
        zzaz().zzg();
        zzB();
        Preconditions.checkNotEmpty(str);
        if (bArr == null) {
            try {
                bArr = new byte[0];
            } finally {
                this.zzt = false;
                zzae();
            }
        }
        zzef zzj = zzay().zzj();
        Integer valueOf = Integer.valueOf(bArr.length);
        zzj.zzb("onConfigFetched. Response size", valueOf);
        zzam zzamVar = this.zze;
        zzal(zzamVar);
        zzamVar.zzw();
        zzam zzamVar2 = this.zze;
        zzal(zzamVar2);
        zzh zzj2 = zzamVar2.zzj(str);
        if (i != 200 && i != 204) {
            if (i == 304) {
                i = 304;
            }
            z = false;
            if (zzj2 == null) {
                zzay().zzk().zzb("App does not exist in onConfigFetched. appId", zzeh.zzn(str));
            } else {
                if (!z && i != 404) {
                    zzj2.zzU(zzav().currentTimeMillis());
                    zzam zzamVar3 = this.zze;
                    zzal(zzamVar3);
                    zzamVar3.zzD(zzj2);
                    zzay().zzj().zzc("Fetching config failed. code, error", Integer.valueOf(i), th);
                    zzfi zzfiVar2 = this.zzc;
                    zzal(zzfiVar2);
                    zzfiVar2.zzl(str);
                    this.zzk.zzd.zzb(zzav().currentTimeMillis());
                    if (i == 503 || i == 429) {
                        this.zzk.zzb.zzb(zzav().currentTimeMillis());
                    }
                    zzag();
                }
                List list = map != null ? (List) map.get(HttpHeaders.LAST_MODIFIED) : null;
                String str3 = (list == null || list.isEmpty()) ? null : (String) list.get(0);
                zzox.zzc();
                if (zzg().zzs(null, zzdu.zzao)) {
                    List list2 = map != null ? (List) map.get(HttpHeaders.ETAG) : null;
                    if (list2 != null && !list2.isEmpty()) {
                        str2 = (String) list2.get(0);
                        if (i != 404 && i != 304) {
                            zzfi zzfiVar32 = this.zzc;
                            zzal(zzfiVar32);
                            zzfiVar32.zzt(str, bArr, str3, str2);
                            zzj2.zzL(zzav().currentTimeMillis());
                            zzam zzamVar4222 = this.zze;
                            zzal(zzamVar4222);
                            zzamVar4222.zzD(zzj2);
                            if (i != 404) {
                                zzay().zzl().zzb("Config not found. Using empty config. appId", str);
                            } else {
                                zzay().zzj().zzc("Successfully fetched config. Got network response. code, size", Integer.valueOf(i), valueOf);
                            }
                            zzenVar = this.zzd;
                            zzal(zzenVar);
                            if (!zzenVar.zza() && zzai()) {
                                zzX();
                            } else {
                                zzag();
                            }
                        }
                        zzfiVar = this.zzc;
                        zzal(zzfiVar);
                        if (zzfiVar.zze(str) == null) {
                            zzfi zzfiVar4 = this.zzc;
                            zzal(zzfiVar4);
                            zzfiVar4.zzt(str, null, null, null);
                        }
                        zzj2.zzL(zzav().currentTimeMillis());
                        zzam zzamVar42222 = this.zze;
                        zzal(zzamVar42222);
                        zzamVar42222.zzD(zzj2);
                        if (i != 404) {
                        }
                        zzenVar = this.zzd;
                        zzal(zzenVar);
                        if (!zzenVar.zza()) {
                        }
                        zzag();
                    }
                }
                str2 = null;
                if (i != 404) {
                    zzfi zzfiVar322 = this.zzc;
                    zzal(zzfiVar322);
                    zzfiVar322.zzt(str, bArr, str3, str2);
                    zzj2.zzL(zzav().currentTimeMillis());
                    zzam zzamVar422222 = this.zze;
                    zzal(zzamVar422222);
                    zzamVar422222.zzD(zzj2);
                    if (i != 404) {
                    }
                    zzenVar = this.zzd;
                    zzal(zzenVar);
                    if (!zzenVar.zza()) {
                    }
                    zzag();
                }
                zzfiVar = this.zzc;
                zzal(zzfiVar);
                if (zzfiVar.zze(str) == null) {
                }
                zzj2.zzL(zzav().currentTimeMillis());
                zzam zzamVar4222222 = this.zze;
                zzal(zzamVar4222222);
                zzamVar4222222.zzD(zzj2);
                if (i != 404) {
                }
                zzenVar = this.zzd;
                zzal(zzenVar);
                if (!zzenVar.zza()) {
                }
                zzag();
            }
            zzam zzamVar522 = this.zze;
            zzal(zzamVar522);
            zzamVar522.zzC();
            zzam zzamVar622 = this.zze;
            zzal(zzamVar622);
            zzamVar622.zzx();
        }
        if (th == null) {
            z = true;
            if (zzj2 == null) {
            }
            zzam zzamVar5222 = this.zze;
            zzal(zzamVar5222);
            zzamVar5222.zzC();
            zzam zzamVar6222 = this.zze;
            zzal(zzamVar6222);
            zzamVar6222.zzx();
        }
        z = false;
        if (zzj2 == null) {
        }
        zzam zzamVar52222 = this.zze;
        zzal(zzamVar52222);
        zzamVar52222.zzC();
        zzam zzamVar62222 = this.zze;
        zzal(zzamVar62222);
        zzamVar62222.zzx();
    }

    public final void zzJ(boolean z) {
        zzag();
    }

    @VisibleForTesting
    public final void zzK(int i, Throwable th, byte[] bArr, String str) {
        zzam zzamVar;
        long longValue;
        zzaz().zzg();
        zzB();
        if (bArr == null) {
            try {
                bArr = new byte[0];
            } finally {
                this.zzu = false;
                zzae();
            }
        }
        List<Long> list = (List) Preconditions.checkNotNull(this.zzy);
        this.zzy = null;
        if (i != 200) {
            if (i == 204) {
                i = 204;
            }
            zzay().zzj().zzc("Network upload failed. Will retry later. code, error", Integer.valueOf(i), th);
            this.zzk.zzd.zzb(zzav().currentTimeMillis());
            if (i != 503 || i == 429) {
                this.zzk.zzb.zzb(zzav().currentTimeMillis());
            }
            zzam zzamVar22 = this.zze;
            zzal(zzamVar22);
            zzamVar22.zzy(list);
            zzag();
        }
        if (th == null) {
            try {
                this.zzk.zzc.zzb(zzav().currentTimeMillis());
                this.zzk.zzd.zzb(0L);
                zzag();
                zzay().zzj().zzc("Successful upload. Got network response. code, size", Integer.valueOf(i), Integer.valueOf(bArr.length));
                zzam zzamVar3 = this.zze;
                zzal(zzamVar3);
                zzamVar3.zzw();
                try {
                    for (Long l : list) {
                        try {
                            zzamVar = this.zze;
                            zzal(zzamVar);
                            longValue = l.longValue();
                            zzamVar.zzg();
                            zzamVar.zzW();
                            try {
                            } catch (SQLiteException e2) {
                                zzamVar.zzt.zzay().zzd().zzb("Failed to delete a bundle in a queue table", e2);
                                throw e2;
                                break;
                            }
                        } catch (SQLiteException e3) {
                            List list2 = this.zzz;
                            if (list2 == null || !list2.contains(l)) {
                                throw e3;
                            }
                        }
                        if (zzamVar.zzh().delete("queue", "rowid=?", new String[]{String.valueOf(longValue)}) != 1) {
                            throw new SQLiteException("Deleted fewer rows from queue than expected");
                            break;
                        }
                    }
                    zzam zzamVar4 = this.zze;
                    zzal(zzamVar4);
                    zzamVar4.zzC();
                    zzam zzamVar5 = this.zze;
                    zzal(zzamVar5);
                    zzamVar5.zzx();
                    this.zzz = null;
                    zzen zzenVar = this.zzd;
                    zzal(zzenVar);
                    if (zzenVar.zza() && zzai()) {
                        zzX();
                    } else {
                        this.zzA = -1L;
                        zzag();
                    }
                    this.zza = 0L;
                } catch (Throwable th2) {
                    zzam zzamVar6 = this.zze;
                    zzal(zzamVar6);
                    zzamVar6.zzx();
                    throw th2;
                }
            } catch (SQLiteException e4) {
                zzay().zzd().zzb("Database error while trying to delete uploaded bundles", e4);
                this.zza = zzav().elapsedRealtime();
                zzay().zzj().zzb("Disable upload, time", Long.valueOf(this.zza));
            }
        }
        zzay().zzj().zzc("Network upload failed. Will retry later. code, error", Integer.valueOf(i), th);
        this.zzk.zzd.zzb(zzav().currentTimeMillis());
        if (i != 503) {
        }
        this.zzk.zzb.zzb(zzav().currentTimeMillis());
        zzam zzamVar222 = this.zze;
        zzal(zzamVar222);
        zzamVar222.zzy(list);
        zzag();
    }

    /* JADX WARN: Can't wrap try/catch for region: R(9:(2:93|94)|(2:96|(11:98|(3:100|(2:102|(1:104))(1:129)|128)(1:130)|105|(1:107)(1:127)|108|109|110|111|112|113|(4:115|(1:117)|118|(1:120))))|131|109|110|111|112|113|(0)) */
    /* JADX WARN: Code restructure failed: missing block: B:159:0x04b7, code lost:
        r0 = e;
     */
    /* JADX WARN: Code restructure failed: missing block: B:161:0x04b9, code lost:
        r0 = e;
     */
    /* JADX WARN: Code restructure failed: missing block: B:162:0x04ba, code lost:
        r3 = r21;
     */
    /* JADX WARN: Code restructure failed: missing block: B:163:0x04bc, code lost:
        zzay().zzd().zzc("Application info is null, first open report might be inaccurate. appId", com.google.android.gms.measurement.internal.zzeh.zzn(r3), r0);
        r0 = r4;
     */
    /* JADX WARN: Removed duplicated region for block: B:126:0x03e9 A[Catch: all -> 0x057a, TryCatch #6 {all -> 0x057a, blocks: (B:23:0x00a4, B:25:0x00b3, B:43:0x0119, B:45:0x012c, B:47:0x0142, B:48:0x0169, B:50:0x01c5, B:52:0x01cb, B:54:0x01d4, B:64:0x0206, B:66:0x0211, B:70:0x021e, B:73:0x022f, B:77:0x023a, B:79:0x023d, B:80:0x025b, B:82:0x0260, B:85:0x027f, B:88:0x0292, B:90:0x02b8, B:93:0x02c0, B:95:0x02cf, B:124:0x03b5, B:126:0x03e9, B:127:0x03ec, B:129:0x0415, B:173:0x04ec, B:174:0x04ef, B:182:0x0569, B:131:0x042a, B:136:0x044f, B:138:0x0457, B:140:0x045f, B:144:0x0472, B:148:0x0485, B:152:0x0491, B:155:0x04a5, B:157:0x04b2, B:165:0x04d0, B:167:0x04d6, B:168:0x04db, B:170:0x04e1, B:163:0x04bc, B:146:0x047d, B:134:0x043b, B:96:0x02e0, B:98:0x030b, B:99:0x031c, B:101:0x0323, B:103:0x0329, B:105:0x0333, B:107:0x0339, B:109:0x033f, B:111:0x0345, B:112:0x034a, B:117:0x036d, B:120:0x0372, B:121:0x0386, B:122:0x0396, B:123:0x03a6, B:175:0x0504, B:177:0x0534, B:178:0x0537, B:179:0x054c, B:181:0x0550, B:83:0x026f, B:60:0x01ed, B:29:0x00c5, B:31:0x00c9, B:35:0x00da, B:37:0x00f3, B:39:0x00fd, B:42:0x0109), top: B:201:0x00a4, inners: #0, #5 }] */
    /* JADX WARN: Removed duplicated region for block: B:129:0x0415 A[Catch: all -> 0x057a, TRY_LEAVE, TryCatch #6 {all -> 0x057a, blocks: (B:23:0x00a4, B:25:0x00b3, B:43:0x0119, B:45:0x012c, B:47:0x0142, B:48:0x0169, B:50:0x01c5, B:52:0x01cb, B:54:0x01d4, B:64:0x0206, B:66:0x0211, B:70:0x021e, B:73:0x022f, B:77:0x023a, B:79:0x023d, B:80:0x025b, B:82:0x0260, B:85:0x027f, B:88:0x0292, B:90:0x02b8, B:93:0x02c0, B:95:0x02cf, B:124:0x03b5, B:126:0x03e9, B:127:0x03ec, B:129:0x0415, B:173:0x04ec, B:174:0x04ef, B:182:0x0569, B:131:0x042a, B:136:0x044f, B:138:0x0457, B:140:0x045f, B:144:0x0472, B:148:0x0485, B:152:0x0491, B:155:0x04a5, B:157:0x04b2, B:165:0x04d0, B:167:0x04d6, B:168:0x04db, B:170:0x04e1, B:163:0x04bc, B:146:0x047d, B:134:0x043b, B:96:0x02e0, B:98:0x030b, B:99:0x031c, B:101:0x0323, B:103:0x0329, B:105:0x0333, B:107:0x0339, B:109:0x033f, B:111:0x0345, B:112:0x034a, B:117:0x036d, B:120:0x0372, B:121:0x0386, B:122:0x0396, B:123:0x03a6, B:175:0x0504, B:177:0x0534, B:178:0x0537, B:179:0x054c, B:181:0x0550, B:83:0x026f, B:60:0x01ed, B:29:0x00c5, B:31:0x00c9, B:35:0x00da, B:37:0x00f3, B:39:0x00fd, B:42:0x0109), top: B:201:0x00a4, inners: #0, #5 }] */
    /* JADX WARN: Removed duplicated region for block: B:165:0x04d0 A[Catch: all -> 0x057a, TryCatch #6 {all -> 0x057a, blocks: (B:23:0x00a4, B:25:0x00b3, B:43:0x0119, B:45:0x012c, B:47:0x0142, B:48:0x0169, B:50:0x01c5, B:52:0x01cb, B:54:0x01d4, B:64:0x0206, B:66:0x0211, B:70:0x021e, B:73:0x022f, B:77:0x023a, B:79:0x023d, B:80:0x025b, B:82:0x0260, B:85:0x027f, B:88:0x0292, B:90:0x02b8, B:93:0x02c0, B:95:0x02cf, B:124:0x03b5, B:126:0x03e9, B:127:0x03ec, B:129:0x0415, B:173:0x04ec, B:174:0x04ef, B:182:0x0569, B:131:0x042a, B:136:0x044f, B:138:0x0457, B:140:0x045f, B:144:0x0472, B:148:0x0485, B:152:0x0491, B:155:0x04a5, B:157:0x04b2, B:165:0x04d0, B:167:0x04d6, B:168:0x04db, B:170:0x04e1, B:163:0x04bc, B:146:0x047d, B:134:0x043b, B:96:0x02e0, B:98:0x030b, B:99:0x031c, B:101:0x0323, B:103:0x0329, B:105:0x0333, B:107:0x0339, B:109:0x033f, B:111:0x0345, B:112:0x034a, B:117:0x036d, B:120:0x0372, B:121:0x0386, B:122:0x0396, B:123:0x03a6, B:175:0x0504, B:177:0x0534, B:178:0x0537, B:179:0x054c, B:181:0x0550, B:83:0x026f, B:60:0x01ed, B:29:0x00c5, B:31:0x00c9, B:35:0x00da, B:37:0x00f3, B:39:0x00fd, B:42:0x0109), top: B:201:0x00a4, inners: #0, #5 }] */
    /* JADX WARN: Removed duplicated region for block: B:173:0x04ec A[Catch: all -> 0x057a, TryCatch #6 {all -> 0x057a, blocks: (B:23:0x00a4, B:25:0x00b3, B:43:0x0119, B:45:0x012c, B:47:0x0142, B:48:0x0169, B:50:0x01c5, B:52:0x01cb, B:54:0x01d4, B:64:0x0206, B:66:0x0211, B:70:0x021e, B:73:0x022f, B:77:0x023a, B:79:0x023d, B:80:0x025b, B:82:0x0260, B:85:0x027f, B:88:0x0292, B:90:0x02b8, B:93:0x02c0, B:95:0x02cf, B:124:0x03b5, B:126:0x03e9, B:127:0x03ec, B:129:0x0415, B:173:0x04ec, B:174:0x04ef, B:182:0x0569, B:131:0x042a, B:136:0x044f, B:138:0x0457, B:140:0x045f, B:144:0x0472, B:148:0x0485, B:152:0x0491, B:155:0x04a5, B:157:0x04b2, B:165:0x04d0, B:167:0x04d6, B:168:0x04db, B:170:0x04e1, B:163:0x04bc, B:146:0x047d, B:134:0x043b, B:96:0x02e0, B:98:0x030b, B:99:0x031c, B:101:0x0323, B:103:0x0329, B:105:0x0333, B:107:0x0339, B:109:0x033f, B:111:0x0345, B:112:0x034a, B:117:0x036d, B:120:0x0372, B:121:0x0386, B:122:0x0396, B:123:0x03a6, B:175:0x0504, B:177:0x0534, B:178:0x0537, B:179:0x054c, B:181:0x0550, B:83:0x026f, B:60:0x01ed, B:29:0x00c5, B:31:0x00c9, B:35:0x00da, B:37:0x00f3, B:39:0x00fd, B:42:0x0109), top: B:201:0x00a4, inners: #0, #5 }] */
    /* JADX WARN: Removed duplicated region for block: B:179:0x054c A[Catch: all -> 0x057a, TryCatch #6 {all -> 0x057a, blocks: (B:23:0x00a4, B:25:0x00b3, B:43:0x0119, B:45:0x012c, B:47:0x0142, B:48:0x0169, B:50:0x01c5, B:52:0x01cb, B:54:0x01d4, B:64:0x0206, B:66:0x0211, B:70:0x021e, B:73:0x022f, B:77:0x023a, B:79:0x023d, B:80:0x025b, B:82:0x0260, B:85:0x027f, B:88:0x0292, B:90:0x02b8, B:93:0x02c0, B:95:0x02cf, B:124:0x03b5, B:126:0x03e9, B:127:0x03ec, B:129:0x0415, B:173:0x04ec, B:174:0x04ef, B:182:0x0569, B:131:0x042a, B:136:0x044f, B:138:0x0457, B:140:0x045f, B:144:0x0472, B:148:0x0485, B:152:0x0491, B:155:0x04a5, B:157:0x04b2, B:165:0x04d0, B:167:0x04d6, B:168:0x04db, B:170:0x04e1, B:163:0x04bc, B:146:0x047d, B:134:0x043b, B:96:0x02e0, B:98:0x030b, B:99:0x031c, B:101:0x0323, B:103:0x0329, B:105:0x0333, B:107:0x0339, B:109:0x033f, B:111:0x0345, B:112:0x034a, B:117:0x036d, B:120:0x0372, B:121:0x0386, B:122:0x0396, B:123:0x03a6, B:175:0x0504, B:177:0x0534, B:178:0x0537, B:179:0x054c, B:181:0x0550, B:83:0x026f, B:60:0x01ed, B:29:0x00c5, B:31:0x00c9, B:35:0x00da, B:37:0x00f3, B:39:0x00fd, B:42:0x0109), top: B:201:0x00a4, inners: #0, #5 }] */
    /* JADX WARN: Removed duplicated region for block: B:189:0x042a A[EXC_TOP_SPLITTER, SYNTHETIC] */
    /* JADX WARN: Removed duplicated region for block: B:64:0x0206 A[Catch: all -> 0x057a, TryCatch #6 {all -> 0x057a, blocks: (B:23:0x00a4, B:25:0x00b3, B:43:0x0119, B:45:0x012c, B:47:0x0142, B:48:0x0169, B:50:0x01c5, B:52:0x01cb, B:54:0x01d4, B:64:0x0206, B:66:0x0211, B:70:0x021e, B:73:0x022f, B:77:0x023a, B:79:0x023d, B:80:0x025b, B:82:0x0260, B:85:0x027f, B:88:0x0292, B:90:0x02b8, B:93:0x02c0, B:95:0x02cf, B:124:0x03b5, B:126:0x03e9, B:127:0x03ec, B:129:0x0415, B:173:0x04ec, B:174:0x04ef, B:182:0x0569, B:131:0x042a, B:136:0x044f, B:138:0x0457, B:140:0x045f, B:144:0x0472, B:148:0x0485, B:152:0x0491, B:155:0x04a5, B:157:0x04b2, B:165:0x04d0, B:167:0x04d6, B:168:0x04db, B:170:0x04e1, B:163:0x04bc, B:146:0x047d, B:134:0x043b, B:96:0x02e0, B:98:0x030b, B:99:0x031c, B:101:0x0323, B:103:0x0329, B:105:0x0333, B:107:0x0339, B:109:0x033f, B:111:0x0345, B:112:0x034a, B:117:0x036d, B:120:0x0372, B:121:0x0386, B:122:0x0396, B:123:0x03a6, B:175:0x0504, B:177:0x0534, B:178:0x0537, B:179:0x054c, B:181:0x0550, B:83:0x026f, B:60:0x01ed, B:29:0x00c5, B:31:0x00c9, B:35:0x00da, B:37:0x00f3, B:39:0x00fd, B:42:0x0109), top: B:201:0x00a4, inners: #0, #5 }] */
    /* JADX WARN: Removed duplicated region for block: B:82:0x0260 A[Catch: all -> 0x057a, TryCatch #6 {all -> 0x057a, blocks: (B:23:0x00a4, B:25:0x00b3, B:43:0x0119, B:45:0x012c, B:47:0x0142, B:48:0x0169, B:50:0x01c5, B:52:0x01cb, B:54:0x01d4, B:64:0x0206, B:66:0x0211, B:70:0x021e, B:73:0x022f, B:77:0x023a, B:79:0x023d, B:80:0x025b, B:82:0x0260, B:85:0x027f, B:88:0x0292, B:90:0x02b8, B:93:0x02c0, B:95:0x02cf, B:124:0x03b5, B:126:0x03e9, B:127:0x03ec, B:129:0x0415, B:173:0x04ec, B:174:0x04ef, B:182:0x0569, B:131:0x042a, B:136:0x044f, B:138:0x0457, B:140:0x045f, B:144:0x0472, B:148:0x0485, B:152:0x0491, B:155:0x04a5, B:157:0x04b2, B:165:0x04d0, B:167:0x04d6, B:168:0x04db, B:170:0x04e1, B:163:0x04bc, B:146:0x047d, B:134:0x043b, B:96:0x02e0, B:98:0x030b, B:99:0x031c, B:101:0x0323, B:103:0x0329, B:105:0x0333, B:107:0x0339, B:109:0x033f, B:111:0x0345, B:112:0x034a, B:117:0x036d, B:120:0x0372, B:121:0x0386, B:122:0x0396, B:123:0x03a6, B:175:0x0504, B:177:0x0534, B:178:0x0537, B:179:0x054c, B:181:0x0550, B:83:0x026f, B:60:0x01ed, B:29:0x00c5, B:31:0x00c9, B:35:0x00da, B:37:0x00f3, B:39:0x00fd, B:42:0x0109), top: B:201:0x00a4, inners: #0, #5 }] */
    /* JADX WARN: Removed duplicated region for block: B:83:0x026f A[Catch: all -> 0x057a, TryCatch #6 {all -> 0x057a, blocks: (B:23:0x00a4, B:25:0x00b3, B:43:0x0119, B:45:0x012c, B:47:0x0142, B:48:0x0169, B:50:0x01c5, B:52:0x01cb, B:54:0x01d4, B:64:0x0206, B:66:0x0211, B:70:0x021e, B:73:0x022f, B:77:0x023a, B:79:0x023d, B:80:0x025b, B:82:0x0260, B:85:0x027f, B:88:0x0292, B:90:0x02b8, B:93:0x02c0, B:95:0x02cf, B:124:0x03b5, B:126:0x03e9, B:127:0x03ec, B:129:0x0415, B:173:0x04ec, B:174:0x04ef, B:182:0x0569, B:131:0x042a, B:136:0x044f, B:138:0x0457, B:140:0x045f, B:144:0x0472, B:148:0x0485, B:152:0x0491, B:155:0x04a5, B:157:0x04b2, B:165:0x04d0, B:167:0x04d6, B:168:0x04db, B:170:0x04e1, B:163:0x04bc, B:146:0x047d, B:134:0x043b, B:96:0x02e0, B:98:0x030b, B:99:0x031c, B:101:0x0323, B:103:0x0329, B:105:0x0333, B:107:0x0339, B:109:0x033f, B:111:0x0345, B:112:0x034a, B:117:0x036d, B:120:0x0372, B:121:0x0386, B:122:0x0396, B:123:0x03a6, B:175:0x0504, B:177:0x0534, B:178:0x0537, B:179:0x054c, B:181:0x0550, B:83:0x026f, B:60:0x01ed, B:29:0x00c5, B:31:0x00c9, B:35:0x00da, B:37:0x00f3, B:39:0x00fd, B:42:0x0109), top: B:201:0x00a4, inners: #0, #5 }] */
    /* JADX WARN: Removed duplicated region for block: B:85:0x027f A[Catch: all -> 0x057a, TRY_LEAVE, TryCatch #6 {all -> 0x057a, blocks: (B:23:0x00a4, B:25:0x00b3, B:43:0x0119, B:45:0x012c, B:47:0x0142, B:48:0x0169, B:50:0x01c5, B:52:0x01cb, B:54:0x01d4, B:64:0x0206, B:66:0x0211, B:70:0x021e, B:73:0x022f, B:77:0x023a, B:79:0x023d, B:80:0x025b, B:82:0x0260, B:85:0x027f, B:88:0x0292, B:90:0x02b8, B:93:0x02c0, B:95:0x02cf, B:124:0x03b5, B:126:0x03e9, B:127:0x03ec, B:129:0x0415, B:173:0x04ec, B:174:0x04ef, B:182:0x0569, B:131:0x042a, B:136:0x044f, B:138:0x0457, B:140:0x045f, B:144:0x0472, B:148:0x0485, B:152:0x0491, B:155:0x04a5, B:157:0x04b2, B:165:0x04d0, B:167:0x04d6, B:168:0x04db, B:170:0x04e1, B:163:0x04bc, B:146:0x047d, B:134:0x043b, B:96:0x02e0, B:98:0x030b, B:99:0x031c, B:101:0x0323, B:103:0x0329, B:105:0x0333, B:107:0x0339, B:109:0x033f, B:111:0x0345, B:112:0x034a, B:117:0x036d, B:120:0x0372, B:121:0x0386, B:122:0x0396, B:123:0x03a6, B:175:0x0504, B:177:0x0534, B:178:0x0537, B:179:0x054c, B:181:0x0550, B:83:0x026f, B:60:0x01ed, B:29:0x00c5, B:31:0x00c9, B:35:0x00da, B:37:0x00f3, B:39:0x00fd, B:42:0x0109), top: B:201:0x00a4, inners: #0, #5 }] */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public final void zzL(zzq zzqVar) {
        String str;
        String str2;
        int i;
        zzh zzj;
        String str3;
        zzas zzn;
        boolean z;
        long zzc;
        PackageInfo packageInfo;
        String str4;
        String str5;
        ApplicationInfo applicationInfo;
        ApplicationInfo applicationInfo2;
        boolean z2;
        SQLiteDatabase zzh;
        String[] strArr;
        int delete;
        zzaz().zzg();
        zzB();
        Preconditions.checkNotNull(zzqVar);
        Preconditions.checkNotEmpty(zzqVar.zza);
        if (zzak(zzqVar)) {
            zzam zzamVar = this.zze;
            zzal(zzamVar);
            zzh zzj2 = zzamVar.zzj(zzqVar.zza);
            if (zzj2 != null && TextUtils.isEmpty(zzj2.zzy()) && !TextUtils.isEmpty(zzqVar.zzb)) {
                zzj2.zzL(0L);
                zzam zzamVar2 = this.zze;
                zzal(zzamVar2);
                zzamVar2.zzD(zzj2);
                zzfi zzfiVar = this.zzc;
                zzal(zzfiVar);
                zzfiVar.zzm(zzqVar.zza);
            }
            if (!zzqVar.zzh) {
                zzd(zzqVar);
                return;
            }
            long j = zzqVar.zzm;
            if (j == 0) {
                j = zzav().currentTimeMillis();
            }
            this.zzn.zzg().zzd();
            int i2 = zzqVar.zzn;
            if (i2 != 0 && i2 != 1) {
                zzay().zzk().zzc("Incorrect app type, assuming installed app. appId, appType", zzeh.zzn(zzqVar.zza), Integer.valueOf(i2));
                i2 = 0;
            }
            zzam zzamVar3 = this.zze;
            zzal(zzamVar3);
            zzamVar3.zzw();
            try {
                zzam zzamVar4 = this.zze;
                zzal(zzamVar4);
                zzky zzp = zzamVar4.zzp(zzqVar.zza, "_npa");
                if (zzp != null && !"auto".equals(zzp.zzb)) {
                    str = "_sysu";
                    str2 = "_sys";
                    i = 1;
                    zzam zzamVar52 = this.zze;
                    zzal(zzamVar52);
                    zzj = zzamVar52.zzj((String) Preconditions.checkNotNull(zzqVar.zza));
                    if (zzj == null && zzv().zzam(zzqVar.zzb, zzj.zzy(), zzqVar.zzq, zzj.zzr())) {
                        zzay().zzk().zzb("New GMP App Id passed in. Removing cached database data. appId", zzeh.zzn(zzj.zzt()));
                        zzam zzamVar6 = this.zze;
                        zzal(zzamVar6);
                        String zzt = zzj.zzt();
                        zzamVar6.zzW();
                        zzamVar6.zzg();
                        Preconditions.checkNotEmpty(zzt);
                        try {
                            zzh = zzamVar6.zzh();
                            strArr = new String[i];
                            strArr[0] = zzt;
                            delete = zzh.delete("events", "app_id=?", strArr) + zzh.delete("user_attributes", "app_id=?", strArr) + zzh.delete("conditional_properties", "app_id=?", strArr) + zzh.delete("apps", "app_id=?", strArr) + zzh.delete("raw_events", "app_id=?", strArr) + zzh.delete("raw_events_metadata", "app_id=?", strArr) + zzh.delete("event_filters", "app_id=?", strArr) + zzh.delete("property_filters", "app_id=?", strArr) + zzh.delete("audience_filter_values", "app_id=?", strArr) + zzh.delete("consent_settings", "app_id=?", strArr);
                            zzoi.zzc();
                            str3 = "_pfo";
                        } catch (SQLiteException e2) {
                            e = e2;
                            str3 = "_pfo";
                        }
                        try {
                            if (zzamVar6.zzt.zzf().zzs(null, zzdu.zzat)) {
                                delete += zzh.delete("default_event_params", "app_id=?", strArr);
                            }
                            if (delete > 0) {
                                zzamVar6.zzt.zzay().zzj().zzc("Deleted application data. app, records", zzt, Integer.valueOf(delete));
                            }
                        } catch (SQLiteException e3) {
                            e = e3;
                            zzamVar6.zzt.zzay().zzd().zzc("Error deleting application data. appId, error", zzeh.zzn(zzt), e);
                            zzj = null;
                            if (zzj != null) {
                            }
                            zzd(zzqVar);
                            if (i2 == 0) {
                            }
                            if (zzn == null) {
                            }
                            zzam zzamVar722 = this.zze;
                            zzal(zzamVar722);
                            zzamVar722.zzC();
                        }
                        zzj = null;
                    } else {
                        str3 = "_pfo";
                    }
                    if (zzj != null) {
                        boolean z3 = (zzj.zzb() == -2147483648L || zzj.zzb() == zzqVar.zzj) ? false : true;
                        String zzw = zzj.zzw();
                        if (z3 | ((zzj.zzb() != -2147483648L || zzw == null || zzw.equals(zzqVar.zzc)) ? false : true)) {
                            Bundle bundle = new Bundle();
                            bundle.putString("_pv", zzw);
                            zzE(new zzaw("_au", new zzau(bundle), "auto", j), zzqVar);
                        }
                    }
                    zzd(zzqVar);
                    if (i2 == 0) {
                        zzam zzamVar8 = this.zze;
                        zzal(zzamVar8);
                        zzn = zzamVar8.zzn(zzqVar.zza, "_f");
                        z = false;
                    } else {
                        zzam zzamVar9 = this.zze;
                        zzal(zzamVar9);
                        zzn = zzamVar9.zzn(zzqVar.zza, "_v");
                        z = true;
                    }
                    if (zzn == null) {
                        long j2 = ((j / 3600000) + 1) * 3600000;
                        if (!z) {
                            zzW(new zzkw("_fot", j, Long.valueOf(j2), "auto"), zzqVar);
                            zzaz().zzg();
                            zzez zzezVar = (zzez) Preconditions.checkNotNull(this.zzm);
                            String str6 = zzqVar.zza;
                            if (str6 != null && !str6.isEmpty()) {
                                zzezVar.zza.zzaz().zzg();
                                if (!zzezVar.zza()) {
                                    zzezVar.zza.zzay().zzi().zza("Install Referrer Reporter is not available");
                                } else {
                                    zzey zzeyVar = new zzey(zzezVar, str6);
                                    zzezVar.zza.zzaz().zzg();
                                    Intent intent = new Intent("com.google.android.finsky.BIND_GET_INSTALL_REFERRER_SERVICE");
                                    intent.setComponent(new ComponentName("com.android.vending", "com.google.android.finsky.externalreferrer.GetInstallReferrerService"));
                                    PackageManager packageManager = zzezVar.zza.zzau().getPackageManager();
                                    if (packageManager == null) {
                                        zzezVar.zza.zzay().zzm().zza("Failed to obtain Package Manager to verify binding conditions for Install Referrer");
                                    } else {
                                        List<ResolveInfo> queryIntentServices = packageManager.queryIntentServices(intent, 0);
                                        if (queryIntentServices != null && !queryIntentServices.isEmpty()) {
                                            ServiceInfo serviceInfo = queryIntentServices.get(0).serviceInfo;
                                            if (serviceInfo != null) {
                                                String str7 = serviceInfo.packageName;
                                                if (serviceInfo.name != null && "com.android.vending".equals(str7) && zzezVar.zza()) {
                                                    try {
                                                        zzezVar.zza.zzay().zzj().zzb("Install Referrer Service is", true != ConnectionTracker.getInstance().bindService(zzezVar.zza.zzau(), new Intent(intent), zzeyVar, 1) ? "not available" : "available");
                                                    } catch (RuntimeException e4) {
                                                        zzezVar.zza.zzay().zzd().zzb("Exception occurred while binding to Install Referrer Service", e4.getMessage());
                                                    }
                                                } else {
                                                    zzezVar.zza.zzay().zzk().zza("Play Store version 8.3.73 or higher required for Install Referrer");
                                                }
                                            }
                                        } else {
                                            zzezVar.zza.zzay().zzi().zza("Play Service for fetching Install Referrer is unavailable on device");
                                        }
                                    }
                                }
                                zzaz().zzg();
                                zzB();
                                Bundle bundle22 = new Bundle();
                                bundle22.putLong("_c", 1L);
                                bundle22.putLong("_r", 1L);
                                bundle22.putLong("_uwa", 0L);
                                String str82 = str3;
                                bundle22.putLong(str82, 0L);
                                String str92 = str2;
                                bundle22.putLong(str92, 0L);
                                String str102 = str;
                                bundle22.putLong(str102, 0L);
                                bundle22.putLong("_et", 1L);
                                if (zzqVar.zzp) {
                                    bundle22.putLong("_dac", 1L);
                                }
                                String str112 = (String) Preconditions.checkNotNull(zzqVar.zza);
                                zzam zzamVar102 = this.zze;
                                zzal(zzamVar102);
                                Preconditions.checkNotEmpty(str112);
                                zzamVar102.zzg();
                                zzamVar102.zzW();
                                zzc = zzamVar102.zzc(str112, "first_open_count");
                                if (this.zzn.zzau().getPackageManager() != null) {
                                    zzay().zzd().zzb("PackageManager is null, first open report might be inaccurate. appId", zzeh.zzn(str112));
                                } else {
                                    try {
                                        packageInfo = Wrappers.packageManager(this.zzn.zzau()).getPackageInfo(str112, 0);
                                    } catch (PackageManager.NameNotFoundException e5) {
                                        zzay().zzd().zzc("Package info is null, first open report might be inaccurate. appId", zzeh.zzn(str112), e5);
                                        packageInfo = null;
                                    }
                                    if (packageInfo != null) {
                                        long j3 = packageInfo.firstInstallTime;
                                        if (j3 != 0) {
                                            str4 = str112;
                                            if (j3 != packageInfo.lastUpdateTime) {
                                                applicationInfo = null;
                                                if (!zzg().zzs(null, zzdu.zzab)) {
                                                    bundle22.putLong("_uwa", 1L);
                                                } else if (zzc == 0) {
                                                    bundle22.putLong("_uwa", 1L);
                                                    z2 = false;
                                                    zzc = 0;
                                                }
                                                z2 = false;
                                            } else {
                                                applicationInfo = null;
                                                z2 = true;
                                            }
                                            str5 = str102;
                                            zzW(new zzkw("_fi", j, Long.valueOf(true != z2 ? 0L : 1L), "auto"), zzqVar);
                                            String str122 = str4;
                                            applicationInfo2 = Wrappers.packageManager(this.zzn.zzau()).getApplicationInfo(str122, 0);
                                            if (applicationInfo2 != null) {
                                                if ((applicationInfo2.flags & 1) != 0) {
                                                    bundle22.putLong(str92, 1L);
                                                }
                                                if ((applicationInfo2.flags & 128) != 0) {
                                                    bundle22.putLong(str5, 1L);
                                                }
                                            }
                                        }
                                    }
                                    str4 = str112;
                                    str5 = str102;
                                    applicationInfo = null;
                                    String str1222 = str4;
                                    applicationInfo2 = Wrappers.packageManager(this.zzn.zzau()).getApplicationInfo(str1222, 0);
                                    if (applicationInfo2 != null) {
                                    }
                                }
                                if (zzc >= 0) {
                                    bundle22.putLong(str82, zzc);
                                }
                                zzG(new zzaw("_f", new zzau(bundle22), "auto", j), zzqVar);
                            }
                            zzezVar.zza.zzay().zzm().zza("Install Referrer Reporter was called with invalid app package name");
                            zzaz().zzg();
                            zzB();
                            Bundle bundle222 = new Bundle();
                            bundle222.putLong("_c", 1L);
                            bundle222.putLong("_r", 1L);
                            bundle222.putLong("_uwa", 0L);
                            String str822 = str3;
                            bundle222.putLong(str822, 0L);
                            String str922 = str2;
                            bundle222.putLong(str922, 0L);
                            String str1022 = str;
                            bundle222.putLong(str1022, 0L);
                            bundle222.putLong("_et", 1L);
                            if (zzqVar.zzp) {
                            }
                            String str1122 = (String) Preconditions.checkNotNull(zzqVar.zza);
                            zzam zzamVar1022 = this.zze;
                            zzal(zzamVar1022);
                            Preconditions.checkNotEmpty(str1122);
                            zzamVar1022.zzg();
                            zzamVar1022.zzW();
                            zzc = zzamVar1022.zzc(str1122, "first_open_count");
                            if (this.zzn.zzau().getPackageManager() != null) {
                            }
                            if (zzc >= 0) {
                            }
                            zzG(new zzaw("_f", new zzau(bundle222), "auto", j), zzqVar);
                        } else {
                            zzW(new zzkw("_fvt", j, Long.valueOf(j2), "auto"), zzqVar);
                            zzaz().zzg();
                            zzB();
                            Bundle bundle3 = new Bundle();
                            bundle3.putLong("_c", 1L);
                            bundle3.putLong("_r", 1L);
                            bundle3.putLong("_et", 1L);
                            if (zzqVar.zzp) {
                                bundle3.putLong("_dac", 1L);
                            }
                            zzG(new zzaw("_v", new zzau(bundle3), "auto", j), zzqVar);
                        }
                    } else if (zzqVar.zzi) {
                        zzG(new zzaw("_cd", new zzau(new Bundle()), "auto", j), zzqVar);
                    }
                    zzam zzamVar7222 = this.zze;
                    zzal(zzamVar7222);
                    zzamVar7222.zzC();
                }
                if (zzqVar.zzr != null) {
                    str = "_sysu";
                    str2 = "_sys";
                    i = 1;
                    zzkw zzkwVar = new zzkw("_npa", j, Long.valueOf(true != zzqVar.zzr.booleanValue() ? 0L : 1L), "auto");
                    if (zzp == null || !zzp.zze.equals(zzkwVar.zzd)) {
                        zzW(zzkwVar, zzqVar);
                    }
                } else {
                    str = "_sysu";
                    str2 = "_sys";
                    i = 1;
                    if (zzp != null) {
                        zzP(new zzkw("_npa", j, null, "auto"), zzqVar);
                    }
                }
                zzam zzamVar522 = this.zze;
                zzal(zzamVar522);
                zzj = zzamVar522.zzj((String) Preconditions.checkNotNull(zzqVar.zza));
                if (zzj == null) {
                }
                str3 = "_pfo";
                if (zzj != null) {
                }
                zzd(zzqVar);
                if (i2 == 0) {
                }
                if (zzn == null) {
                }
                zzam zzamVar72222 = this.zze;
                zzal(zzamVar72222);
                zzamVar72222.zzC();
            } finally {
                zzam zzamVar11 = this.zze;
                zzal(zzamVar11);
                zzamVar11.zzx();
            }
        }
    }

    public final void zzM() {
        this.zzr++;
    }

    public final void zzN(zzac zzacVar) {
        zzq zzac = zzac((String) Preconditions.checkNotNull(zzacVar.zza));
        if (zzac != null) {
            zzO(zzacVar, zzac);
        }
    }

    public final void zzO(zzac zzacVar, zzq zzqVar) {
        Preconditions.checkNotNull(zzacVar);
        Preconditions.checkNotEmpty(zzacVar.zza);
        Preconditions.checkNotNull(zzacVar.zzc);
        Preconditions.checkNotEmpty(zzacVar.zzc.zzb);
        zzaz().zzg();
        zzB();
        if (zzak(zzqVar)) {
            if (zzqVar.zzh) {
                zzam zzamVar = this.zze;
                zzal(zzamVar);
                zzamVar.zzw();
                try {
                    zzd(zzqVar);
                    String str = (String) Preconditions.checkNotNull(zzacVar.zza);
                    zzam zzamVar2 = this.zze;
                    zzal(zzamVar2);
                    zzac zzk = zzamVar2.zzk(str, zzacVar.zzc.zzb);
                    if (zzk != null) {
                        zzay().zzc().zzc("Removing conditional user property", zzacVar.zza, this.zzn.zzj().zzf(zzacVar.zzc.zzb));
                        zzam zzamVar3 = this.zze;
                        zzal(zzamVar3);
                        zzamVar3.zza(str, zzacVar.zzc.zzb);
                        if (zzk.zze) {
                            zzam zzamVar4 = this.zze;
                            zzal(zzamVar4);
                            zzamVar4.zzA(str, zzacVar.zzc.zzb);
                        }
                        zzaw zzawVar = zzacVar.zzk;
                        if (zzawVar != null) {
                            zzau zzauVar = zzawVar.zzb;
                            zzY((zzaw) Preconditions.checkNotNull(zzv().zzz(str, ((zzaw) Preconditions.checkNotNull(zzacVar.zzk)).zza, zzauVar != null ? zzauVar.zzc() : null, zzk.zzb, zzacVar.zzk.zzd, true, true)), zzqVar);
                        }
                    } else {
                        zzay().zzk().zzc("Conditional user property doesn't exist", zzeh.zzn(zzacVar.zza), this.zzn.zzj().zzf(zzacVar.zzc.zzb));
                    }
                    zzam zzamVar5 = this.zze;
                    zzal(zzamVar5);
                    zzamVar5.zzC();
                    return;
                } finally {
                    zzam zzamVar6 = this.zze;
                    zzal(zzamVar6);
                    zzamVar6.zzx();
                }
            }
            zzd(zzqVar);
        }
    }

    public final void zzP(zzkw zzkwVar, zzq zzqVar) {
        zzaz().zzg();
        zzB();
        if (zzak(zzqVar)) {
            if (!zzqVar.zzh) {
                zzd(zzqVar);
            } else if ("_npa".equals(zzkwVar.zzb) && zzqVar.zzr != null) {
                zzay().zzc().zza("Falling back to manifest metadata value for ad personalization");
                zzW(new zzkw("_npa", zzav().currentTimeMillis(), Long.valueOf(true != zzqVar.zzr.booleanValue() ? 0L : 1L), "auto"), zzqVar);
            } else {
                zzay().zzc().zzb("Removing user property", this.zzn.zzj().zzf(zzkwVar.zzb));
                zzam zzamVar = this.zze;
                zzal(zzamVar);
                zzamVar.zzw();
                try {
                    zzd(zzqVar);
                    if ("_id".equals(zzkwVar.zzb)) {
                        zzam zzamVar2 = this.zze;
                        zzal(zzamVar2);
                        zzamVar2.zzA((String) Preconditions.checkNotNull(zzqVar.zza), "_lair");
                    }
                    zzam zzamVar3 = this.zze;
                    zzal(zzamVar3);
                    zzamVar3.zzA((String) Preconditions.checkNotNull(zzqVar.zza), zzkwVar.zzb);
                    zzam zzamVar4 = this.zze;
                    zzal(zzamVar4);
                    zzamVar4.zzC();
                    zzay().zzc().zzb("User property removed", this.zzn.zzj().zzf(zzkwVar.zzb));
                } finally {
                    zzam zzamVar5 = this.zze;
                    zzal(zzamVar5);
                    zzamVar5.zzx();
                }
            }
        }
    }

    @VisibleForTesting
    public final void zzQ(zzq zzqVar) {
        if (this.zzy != null) {
            ArrayList arrayList = new ArrayList();
            this.zzz = arrayList;
            arrayList.addAll(this.zzy);
        }
        zzam zzamVar = this.zze;
        zzal(zzamVar);
        String str = (String) Preconditions.checkNotNull(zzqVar.zza);
        Preconditions.checkNotEmpty(str);
        zzamVar.zzg();
        zzamVar.zzW();
        try {
            SQLiteDatabase zzh = zzamVar.zzh();
            String[] strArr = {str};
            int delete = zzh.delete("apps", "app_id=?", strArr) + zzh.delete("events", "app_id=?", strArr) + zzh.delete("user_attributes", "app_id=?", strArr) + zzh.delete("conditional_properties", "app_id=?", strArr) + zzh.delete("raw_events", "app_id=?", strArr) + zzh.delete("raw_events_metadata", "app_id=?", strArr) + zzh.delete("queue", "app_id=?", strArr) + zzh.delete("audience_filter_values", "app_id=?", strArr) + zzh.delete("main_event_params", "app_id=?", strArr) + zzh.delete("default_event_params", "app_id=?", strArr);
            if (delete > 0) {
                zzamVar.zzt.zzay().zzj().zzc("Reset analytics data. app, records", str, Integer.valueOf(delete));
            }
        } catch (SQLiteException e2) {
            zzamVar.zzt.zzay().zzd().zzc("Error resetting analytics data. appId, error", zzeh.zzn(str), e2);
        }
        if (zzqVar.zzh) {
            zzL(zzqVar);
        }
    }

    public final void zzR(String str, zzie zzieVar) {
        zzaz().zzg();
        String str2 = this.zzE;
        if (str2 == null || str2.equals(str) || zzieVar != null) {
            this.zzE = str;
            this.zzD = zzieVar;
        }
    }

    public final void zzS() {
        zzaz().zzg();
        zzam zzamVar = this.zze;
        zzal(zzamVar);
        zzamVar.zzz();
        if (this.zzk.zzc.zza() == 0) {
            this.zzk.zzc.zzb(zzav().currentTimeMillis());
        }
        zzag();
    }

    public final void zzT(zzac zzacVar) {
        zzq zzac = zzac((String) Preconditions.checkNotNull(zzacVar.zza));
        if (zzac != null) {
            zzU(zzacVar, zzac);
        }
    }

    public final void zzU(zzac zzacVar, zzq zzqVar) {
        Preconditions.checkNotNull(zzacVar);
        Preconditions.checkNotEmpty(zzacVar.zza);
        Preconditions.checkNotNull(zzacVar.zzb);
        Preconditions.checkNotNull(zzacVar.zzc);
        Preconditions.checkNotEmpty(zzacVar.zzc.zzb);
        zzaz().zzg();
        zzB();
        if (zzak(zzqVar)) {
            if (!zzqVar.zzh) {
                zzd(zzqVar);
                return;
            }
            zzac zzacVar2 = new zzac(zzacVar);
            boolean z = false;
            zzacVar2.zze = false;
            zzam zzamVar = this.zze;
            zzal(zzamVar);
            zzamVar.zzw();
            try {
                zzam zzamVar2 = this.zze;
                zzal(zzamVar2);
                zzac zzk = zzamVar2.zzk((String) Preconditions.checkNotNull(zzacVar2.zza), zzacVar2.zzc.zzb);
                if (zzk != null && !zzk.zzb.equals(zzacVar2.zzb)) {
                    zzay().zzk().zzd("Updating a conditional user property with different origin. name, origin, origin (from DB)", this.zzn.zzj().zzf(zzacVar2.zzc.zzb), zzacVar2.zzb, zzk.zzb);
                }
                if (zzk != null && zzk.zze) {
                    zzacVar2.zzb = zzk.zzb;
                    zzacVar2.zzd = zzk.zzd;
                    zzacVar2.zzh = zzk.zzh;
                    zzacVar2.zzf = zzk.zzf;
                    zzacVar2.zzi = zzk.zzi;
                    zzacVar2.zze = true;
                    zzkw zzkwVar = zzacVar2.zzc;
                    zzacVar2.zzc = new zzkw(zzkwVar.zzb, zzk.zzc.zzc, zzkwVar.zza(), zzk.zzc.zzf);
                } else if (TextUtils.isEmpty(zzacVar2.zzf)) {
                    zzkw zzkwVar2 = zzacVar2.zzc;
                    zzacVar2.zzc = new zzkw(zzkwVar2.zzb, zzacVar2.zzd, zzkwVar2.zza(), zzacVar2.zzc.zzf);
                    zzacVar2.zze = true;
                    z = true;
                }
                if (zzacVar2.zze) {
                    zzkw zzkwVar3 = zzacVar2.zzc;
                    zzky zzkyVar = new zzky((String) Preconditions.checkNotNull(zzacVar2.zza), zzacVar2.zzb, zzkwVar3.zzb, zzkwVar3.zzc, Preconditions.checkNotNull(zzkwVar3.zza()));
                    zzam zzamVar3 = this.zze;
                    zzal(zzamVar3);
                    if (zzamVar3.zzL(zzkyVar)) {
                        zzay().zzc().zzd("User property updated immediately", zzacVar2.zza, this.zzn.zzj().zzf(zzkyVar.zzc), zzkyVar.zze);
                    } else {
                        zzay().zzd().zzd("(2)Too many active user properties, ignoring", zzeh.zzn(zzacVar2.zza), this.zzn.zzj().zzf(zzkyVar.zzc), zzkyVar.zze);
                    }
                    if (z && zzacVar2.zzi != null) {
                        zzY(new zzaw(zzacVar2.zzi, zzacVar2.zzd), zzqVar);
                    }
                }
                zzam zzamVar4 = this.zze;
                zzal(zzamVar4);
                if (zzamVar4.zzK(zzacVar2)) {
                    zzay().zzc().zzd("Conditional property added", zzacVar2.zza, this.zzn.zzj().zzf(zzacVar2.zzc.zzb), zzacVar2.zzc.zza());
                } else {
                    zzay().zzd().zzd("Too many conditional properties, ignoring", zzeh.zzn(zzacVar2.zza), this.zzn.zzj().zzf(zzacVar2.zzc.zzb), zzacVar2.zzc.zza());
                }
                zzam zzamVar5 = this.zze;
                zzal(zzamVar5);
                zzamVar5.zzC();
            } finally {
                zzam zzamVar6 = this.zze;
                zzal(zzamVar6);
                zzamVar6.zzx();
            }
        }
    }

    public final void zzV(String str, zzai zzaiVar) {
        zzaz().zzg();
        zzB();
        this.zzB.put(str, zzaiVar);
        zzam zzamVar = this.zze;
        zzal(zzamVar);
        Preconditions.checkNotNull(str);
        Preconditions.checkNotNull(zzaiVar);
        zzamVar.zzg();
        zzamVar.zzW();
        ContentValues contentValues = new ContentValues();
        contentValues.put("app_id", str);
        contentValues.put("consent_state", zzaiVar.zzh());
        try {
            if (zzamVar.zzh().insertWithOnConflict("consent_settings", null, contentValues, 5) == -1) {
                zzamVar.zzt.zzay().zzd().zzb("Failed to insert/update consent setting (got -1). appId", zzeh.zzn(str));
            }
        } catch (SQLiteException e2) {
            zzamVar.zzt.zzay().zzd().zzc("Error storing consent setting. appId, error", zzeh.zzn(str), e2);
        }
    }

    public final void zzW(zzkw zzkwVar, zzq zzqVar) {
        long j;
        zzaz().zzg();
        zzB();
        if (zzak(zzqVar)) {
            if (!zzqVar.zzh) {
                zzd(zzqVar);
                return;
            }
            int zzl = zzv().zzl(zzkwVar.zzb);
            int i = 0;
            if (zzl != 0) {
                zzlb zzv = zzv();
                String str = zzkwVar.zzb;
                zzg();
                String zzD = zzv.zzD(str, 24, true);
                String str2 = zzkwVar.zzb;
                zzv().zzN(this.zzF, zzqVar.zza, zzl, "_ev", zzD, str2 != null ? str2.length() : 0);
                return;
            }
            int zzd = zzv().zzd(zzkwVar.zzb, zzkwVar.zza());
            if (zzd != 0) {
                zzlb zzv2 = zzv();
                String str3 = zzkwVar.zzb;
                zzg();
                String zzD2 = zzv2.zzD(str3, 24, true);
                Object zza = zzkwVar.zza();
                if (zza != null && ((zza instanceof String) || (zza instanceof CharSequence))) {
                    i = zza.toString().length();
                }
                zzv().zzN(this.zzF, zzqVar.zza, zzd, "_ev", zzD2, i);
                return;
            }
            Object zzB = zzv().zzB(zzkwVar.zzb, zzkwVar.zza());
            if (zzB == null) {
                return;
            }
            if ("_sid".equals(zzkwVar.zzb)) {
                long j2 = zzkwVar.zzc;
                String str4 = zzkwVar.zzf;
                String str5 = (String) Preconditions.checkNotNull(zzqVar.zza);
                zzam zzamVar = this.zze;
                zzal(zzamVar);
                zzky zzp = zzamVar.zzp(str5, "_sno");
                if (zzp != null) {
                    Object obj = zzp.zze;
                    if (obj instanceof Long) {
                        j = ((Long) obj).longValue();
                        zzW(new zzkw("_sno", j2, Long.valueOf(j + 1), str4), zzqVar);
                    }
                }
                if (zzp != null) {
                    zzay().zzk().zzb("Retrieved last session number from database does not contain a valid (long) value", zzp.zze);
                }
                zzam zzamVar2 = this.zze;
                zzal(zzamVar2);
                zzas zzn = zzamVar2.zzn(str5, "_s");
                if (zzn != null) {
                    j = zzn.zzc;
                    zzay().zzj().zzb("Backfill the session number. Last used session number", Long.valueOf(j));
                } else {
                    j = 0;
                }
                zzW(new zzkw("_sno", j2, Long.valueOf(j + 1), str4), zzqVar);
            }
            zzky zzkyVar = new zzky((String) Preconditions.checkNotNull(zzqVar.zza), (String) Preconditions.checkNotNull(zzkwVar.zzf), zzkwVar.zzb, zzkwVar.zzc, zzB);
            zzay().zzj().zzc("Setting user property", this.zzn.zzj().zzf(zzkyVar.zzc), zzB);
            zzam zzamVar3 = this.zze;
            zzal(zzamVar3);
            zzamVar3.zzw();
            try {
                if ("_id".equals(zzkyVar.zzc)) {
                    zzam zzamVar4 = this.zze;
                    zzal(zzamVar4);
                    zzky zzp2 = zzamVar4.zzp(zzqVar.zza, "_id");
                    if (zzp2 != null && !zzkyVar.zze.equals(zzp2.zze)) {
                        zzam zzamVar5 = this.zze;
                        zzal(zzamVar5);
                        zzamVar5.zzA(zzqVar.zza, "_lair");
                    }
                }
                zzd(zzqVar);
                zzam zzamVar6 = this.zze;
                zzal(zzamVar6);
                boolean zzL = zzamVar6.zzL(zzkyVar);
                zzam zzamVar7 = this.zze;
                zzal(zzamVar7);
                zzamVar7.zzC();
                if (!zzL) {
                    zzay().zzd().zzc("Too many unique user properties are set. Ignoring user property", this.zzn.zzj().zzf(zzkyVar.zzc), zzkyVar.zze);
                    zzv().zzN(this.zzF, zzqVar.zza, 9, null, null, 0);
                }
            } finally {
                zzam zzamVar8 = this.zze;
                zzal(zzamVar8);
                zzamVar8.zzx();
            }
        }
    }

    /* JADX WARN: Code restructure failed: missing block: B:151:0x02f6, code lost:
        r0 = r0.subList(0, r3);
     */
    /* JADX WARN: Code restructure failed: missing block: B:153:0x02fb, code lost:
        r0 = th;
     */
    /* JADX WARN: Code restructure failed: missing block: B:154:0x02fc, code lost:
        r2 = false;
     */
    /* JADX WARN: Code restructure failed: missing block: B:231:0x0563, code lost:
        if (r11 == null) goto L254;
     */
    /* JADX WARN: Code restructure failed: missing block: B:52:0x0126, code lost:
        if (r11 == null) goto L229;
     */
    /* JADX WARN: Not initialized variable reg: 11, insn: 0x0581: MOVE  (r9 I:??[OBJECT, ARRAY]) = (r11 I:??[OBJECT, ARRAY]), block:B:241:0x0581 */
    /* JADX WARN: Removed duplicated region for block: B:132:0x029d A[Catch: all -> 0x0588, TryCatch #5 {all -> 0x0588, blocks: (B:130:0x0297, B:132:0x029d, B:134:0x02a9, B:135:0x02ad, B:137:0x02b3, B:139:0x02c7, B:143:0x02d0, B:145:0x02d6, B:148:0x02eb, B:156:0x0302, B:158:0x031d, B:162:0x032c, B:164:0x0350, B:170:0x0362, B:174:0x039c, B:176:0x03a1, B:178:0x03a9, B:179:0x03ac, B:181:0x03b1, B:182:0x03b4, B:184:0x03c0, B:185:0x03d6, B:188:0x03e2, B:190:0x03f3, B:192:0x0405, B:194:0x0427, B:196:0x0465, B:198:0x0477, B:200:0x048c, B:204:0x049c, B:205:0x04a0, B:199:0x0485, B:207:0x04e4, B:195:0x045c, B:117:0x0268, B:129:0x0294, B:211:0x04fb, B:212:0x04fe, B:213:0x04ff, B:218:0x0540, B:234:0x0567, B:236:0x056d, B:238:0x0578, B:222:0x0549, B:243:0x0584, B:244:0x0587, B:203:0x0498), top: B:256:0x00eb, inners: #2 }] */
    /* JADX WARN: Removed duplicated region for block: B:236:0x056d A[Catch: all -> 0x0588, TryCatch #5 {all -> 0x0588, blocks: (B:130:0x0297, B:132:0x029d, B:134:0x02a9, B:135:0x02ad, B:137:0x02b3, B:139:0x02c7, B:143:0x02d0, B:145:0x02d6, B:148:0x02eb, B:156:0x0302, B:158:0x031d, B:162:0x032c, B:164:0x0350, B:170:0x0362, B:174:0x039c, B:176:0x03a1, B:178:0x03a9, B:179:0x03ac, B:181:0x03b1, B:182:0x03b4, B:184:0x03c0, B:185:0x03d6, B:188:0x03e2, B:190:0x03f3, B:192:0x0405, B:194:0x0427, B:196:0x0465, B:198:0x0477, B:200:0x048c, B:204:0x049c, B:205:0x04a0, B:199:0x0485, B:207:0x04e4, B:195:0x045c, B:117:0x0268, B:129:0x0294, B:211:0x04fb, B:212:0x04fe, B:213:0x04ff, B:218:0x0540, B:234:0x0567, B:236:0x056d, B:238:0x0578, B:222:0x0549, B:243:0x0584, B:244:0x0587, B:203:0x0498), top: B:256:0x00eb, inners: #2 }] */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public final void zzX() {
        boolean z;
        Cursor cursor;
        Cursor cursor2;
        Cursor cursor3;
        String str;
        Cursor cursor4;
        Cursor cursor5;
        long j;
        Cursor cursor6;
        List emptyList;
        String str2;
        String str3;
        String str4;
        byte[] blob;
        zzkv zzkvVar;
        zzaz().zzg();
        zzB();
        int i = 1;
        this.zzv = true;
        int i2 = 0;
        try {
            this.zzn.zzaw();
            Boolean zzj = this.zzn.zzt().zzj();
            if (zzj == null) {
                zzay().zzk().zza("Upload data called on the client side before use of service was decided");
                this.zzv = false;
            } else if (zzj.booleanValue()) {
                zzay().zzd().zza("Upload called in the client side when service should be used");
                this.zzv = false;
            } else if (this.zza > 0) {
                zzag();
                this.zzv = false;
            } else {
                zzaz().zzg();
                if (this.zzy != null) {
                    zzay().zzj().zza("Uploading requested multiple times");
                    this.zzv = false;
                } else {
                    zzen zzenVar = this.zzd;
                    zzal(zzenVar);
                    if (!zzenVar.zza()) {
                        zzay().zzj().zza("Network not connected, ignoring upload request");
                        zzag();
                        this.zzv = false;
                    } else {
                        long currentTimeMillis = zzav().currentTimeMillis();
                        Cursor cursor7 = null;
                        int zze = zzg().zze(null, zzdu.zzP);
                        zzg();
                        long zzz = currentTimeMillis - zzag.zzz();
                        for (int i3 = 0; i3 < zze && zzah(null, zzz); i3++) {
                        }
                        long zza = this.zzk.zzc.zza();
                        if (zza != 0) {
                            zzay().zzc().zzb("Uploading events. Elapsed time since last upload attempt (ms)", Long.valueOf(Math.abs(currentTimeMillis - zza)));
                        }
                        zzam zzamVar = this.zze;
                        zzal(zzamVar);
                        String zzr = zzamVar.zzr();
                        long j2 = -1;
                        try {
                            if (!TextUtils.isEmpty(zzr)) {
                                if (this.zzA == -1) {
                                    try {
                                        zzam zzamVar2 = this.zze;
                                        zzal(zzamVar2);
                                        try {
                                            cursor4 = zzamVar2.zzh().rawQuery("select rowid from raw_events order by rowid desc limit 1;", null);
                                        } catch (SQLiteException e2) {
                                            e = e2;
                                            cursor4 = null;
                                        } catch (Throwable th) {
                                            th = th;
                                            if (cursor7 != null) {
                                                cursor7.close();
                                            }
                                            throw th;
                                        }
                                        try {
                                            if (cursor4.moveToFirst()) {
                                                j2 = cursor4.getLong(0);
                                            }
                                        } catch (SQLiteException e3) {
                                            e = e3;
                                            zzamVar2.zzt.zzay().zzd().zzb("Error querying raw events", e);
                                        }
                                        cursor4.close();
                                        this.zzA = j2;
                                    } catch (Throwable th2) {
                                        th = th2;
                                        cursor7 = cursor4;
                                    }
                                }
                                int zze2 = zzg().zze(zzr, zzdu.zzf);
                                int max = Math.max(0, zzg().zze(zzr, zzdu.zzg));
                                zzam zzamVar3 = this.zze;
                                zzal(zzamVar3);
                                zzamVar3.zzg();
                                zzamVar3.zzW();
                                Preconditions.checkArgument(zze2 > 0);
                                try {
                                    Preconditions.checkArgument(max > 0);
                                    Preconditions.checkNotEmpty(zzr);
                                    try {
                                        cursor6 = zzamVar3.zzh().query("queue", new String[]{"rowid", "data", "retry_count"}, "app_id=?", new String[]{zzr}, null, null, "rowid", String.valueOf(zze2));
                                        try {
                                            if (!cursor6.moveToFirst()) {
                                                emptyList = Collections.emptyList();
                                                cursor6.close();
                                                j = currentTimeMillis;
                                            } else {
                                                ArrayList arrayList = new ArrayList();
                                                int i4 = 0;
                                                while (true) {
                                                    long j3 = cursor6.getLong(i2);
                                                    try {
                                                        blob = cursor6.getBlob(i);
                                                        zzkvVar = zzamVar3.zzf.zzi;
                                                        zzal(zzkvVar);
                                                    } catch (IOException e4) {
                                                        e = e4;
                                                        j = currentTimeMillis;
                                                    }
                                                    try {
                                                        ByteArrayInputStream byteArrayInputStream = new ByteArrayInputStream(blob);
                                                        GZIPInputStream gZIPInputStream = new GZIPInputStream(byteArrayInputStream);
                                                        ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
                                                        byte[] bArr = new byte[1024];
                                                        j = currentTimeMillis;
                                                        while (true) {
                                                            try {
                                                                try {
                                                                    int read = gZIPInputStream.read(bArr);
                                                                    if (read <= 0) {
                                                                        break;
                                                                    }
                                                                    byteArrayOutputStream.write(bArr, 0, read);
                                                                } catch (SQLiteException e5) {
                                                                    e = e5;
                                                                    zzamVar3.zzt.zzay().zzd().zzc("Error querying bundles. appId", zzeh.zzn(zzr), e);
                                                                    emptyList = Collections.emptyList();
                                                                    if (cursor6 != null) {
                                                                        cursor6.close();
                                                                    }
                                                                    if (!emptyList.isEmpty()) {
                                                                    }
                                                                    this.zzv = false;
                                                                    zzae();
                                                                }
                                                            } catch (IOException e6) {
                                                                e = e6;
                                                                try {
                                                                    zzkvVar.zzt.zzay().zzd().zzb("Failed to ungzip content", e);
                                                                    throw e;
                                                                    break;
                                                                } catch (IOException e7) {
                                                                    e = e7;
                                                                    zzamVar3.zzt.zzay().zzd().zzc("Failed to unzip queued bundle. appId", zzeh.zzn(zzr), e);
                                                                    if (cursor6.moveToNext()) {
                                                                        break;
                                                                    }
                                                                    currentTimeMillis = j;
                                                                    i = 1;
                                                                    i2 = 0;
                                                                    cursor6.close();
                                                                    emptyList = arrayList;
                                                                    if (!emptyList.isEmpty()) {
                                                                    }
                                                                    this.zzv = false;
                                                                    zzae();
                                                                }
                                                            }
                                                        }
                                                        gZIPInputStream.close();
                                                        byteArrayInputStream.close();
                                                        byte[] byteArray = byteArrayOutputStream.toByteArray();
                                                        if (!arrayList.isEmpty() && byteArray.length + i4 > max) {
                                                            break;
                                                        }
                                                        try {
                                                            com.google.android.gms.internal.measurement.zzgc zzgcVar = (com.google.android.gms.internal.measurement.zzgc) zzkv.zzl(com.google.android.gms.internal.measurement.zzgd.zzt(), byteArray);
                                                            if (!cursor6.isNull(2)) {
                                                                zzgcVar.zzaf(cursor6.getInt(2));
                                                            }
                                                            i4 += byteArray.length;
                                                            arrayList.add(Pair.create((com.google.android.gms.internal.measurement.zzgd) zzgcVar.zzaC(), Long.valueOf(j3)));
                                                        } catch (IOException e8) {
                                                            zzamVar3.zzt.zzay().zzd().zzc("Failed to merge queued bundle. appId", zzeh.zzn(zzr), e8);
                                                        }
                                                        if (cursor6.moveToNext() || i4 > max) {
                                                            break;
                                                            break;
                                                        }
                                                        currentTimeMillis = j;
                                                        i = 1;
                                                        i2 = 0;
                                                    } catch (IOException e9) {
                                                        e = e9;
                                                        j = currentTimeMillis;
                                                    }
                                                }
                                            }
                                        } catch (SQLiteException e10) {
                                            e = e10;
                                            j = currentTimeMillis;
                                        }
                                    } catch (SQLiteException e11) {
                                        e = e11;
                                        j = currentTimeMillis;
                                        cursor6 = null;
                                    } catch (Throwable th3) {
                                        th = th3;
                                        cursor5 = null;
                                        if (cursor5 != null) {
                                            cursor5.close();
                                        }
                                        throw th;
                                    }
                                    if (!emptyList.isEmpty()) {
                                        if (zzh(zzr).zzi(zzah.AD_STORAGE)) {
                                            Iterator it = emptyList.iterator();
                                            while (true) {
                                                if (!it.hasNext()) {
                                                    str4 = null;
                                                    break;
                                                }
                                                com.google.android.gms.internal.measurement.zzgd zzgdVar = (com.google.android.gms.internal.measurement.zzgd) ((Pair) it.next()).first;
                                                if (!zzgdVar.zzJ().isEmpty()) {
                                                    str4 = zzgdVar.zzJ();
                                                    break;
                                                }
                                            }
                                            if (str4 != null) {
                                                int i5 = 0;
                                                while (true) {
                                                    if (i5 >= emptyList.size()) {
                                                        break;
                                                    }
                                                    com.google.android.gms.internal.measurement.zzgd zzgdVar2 = (com.google.android.gms.internal.measurement.zzgd) ((Pair) emptyList.get(i5)).first;
                                                    if (!zzgdVar2.zzJ().isEmpty() && !zzgdVar2.zzJ().equals(str4)) {
                                                        break;
                                                    }
                                                    i5++;
                                                }
                                            }
                                        }
                                        com.google.android.gms.internal.measurement.zzga zza2 = com.google.android.gms.internal.measurement.zzgb.zza();
                                        int size = emptyList.size();
                                        ArrayList arrayList2 = new ArrayList(emptyList.size());
                                        boolean z2 = zzg().zzt(zzr) && zzh(zzr).zzi(zzah.AD_STORAGE);
                                        boolean zzi = zzh(zzr).zzi(zzah.AD_STORAGE);
                                        boolean zzi2 = zzh(zzr).zzi(zzah.ANALYTICS_STORAGE);
                                        zzpd.zzc();
                                        boolean z3 = zzg().zzs(null, zzdu.zzal) && zzg().zzs(zzr, zzdu.zzan);
                                        int i6 = 0;
                                        while (i6 < size) {
                                            com.google.android.gms.internal.measurement.zzgc zzgcVar2 = (com.google.android.gms.internal.measurement.zzgc) ((com.google.android.gms.internal.measurement.zzgd) ((Pair) emptyList.get(i6)).first).zzby();
                                            arrayList2.add((Long) ((Pair) emptyList.get(i6)).second);
                                            zzg().zzh();
                                            zzgcVar2.zzal(74029L);
                                            long j4 = j;
                                            zzgcVar2.zzak(j4);
                                            this.zzn.zzaw();
                                            try {
                                                zzgcVar2.zzag(false);
                                                if (!z2) {
                                                    zzgcVar2.zzq();
                                                }
                                                if (!zzi) {
                                                    zzgcVar2.zzx();
                                                    zzgcVar2.zzt();
                                                }
                                                if (!zzi2) {
                                                    zzgcVar2.zzn();
                                                }
                                                zzC(zzr, zzgcVar2);
                                                if (!z3) {
                                                    zzgcVar2.zzy();
                                                }
                                                if (zzg().zzs(zzr, zzdu.zzT)) {
                                                    byte[] zzbu = ((com.google.android.gms.internal.measurement.zzgd) zzgcVar2.zzaC()).zzbu();
                                                    zzkv zzkvVar2 = this.zzi;
                                                    zzal(zzkvVar2);
                                                    zzgcVar2.zzJ(zzkvVar2.zzd(zzbu));
                                                }
                                                zza2.zza(zzgcVar2);
                                                i6++;
                                                j = j4;
                                            } catch (Throwable th4) {
                                                th = th4;
                                                z = false;
                                                this.zzv = z;
                                                zzae();
                                                throw th;
                                            }
                                        }
                                        long j5 = j;
                                        if (Log.isLoggable(zzay().zzq(), 2)) {
                                            zzkv zzkvVar3 = this.zzi;
                                            zzal(zzkvVar3);
                                            str2 = zzkvVar3.zzm((com.google.android.gms.internal.measurement.zzgb) zza2.zzaC());
                                        } else {
                                            str2 = null;
                                        }
                                        zzal(this.zzi);
                                        byte[] zzbu2 = ((com.google.android.gms.internal.measurement.zzgb) zza2.zzaC()).zzbu();
                                        zzfi zzfiVar = this.zzl.zzf.zzc;
                                        zzal(zzfiVar);
                                        String zzi3 = zzfiVar.zzi(zzr);
                                        if (!TextUtils.isEmpty(zzi3)) {
                                            Uri parse = Uri.parse((String) zzdu.zzp.zza(null));
                                            Uri.Builder buildUpon = parse.buildUpon();
                                            buildUpon.authority(zzi3 + "." + parse.getAuthority());
                                            str3 = buildUpon.build().toString();
                                        } else {
                                            str3 = (String) zzdu.zzp.zza(null);
                                        }
                                        try {
                                            URL url = new URL(str3);
                                            Preconditions.checkArgument(!arrayList2.isEmpty());
                                            if (this.zzy != null) {
                                                zzay().zzd().zza("Set uploading progress before finishing the previous upload");
                                            } else {
                                                this.zzy = new ArrayList(arrayList2);
                                            }
                                            this.zzk.zzd.zzb(j5);
                                            zzay().zzj().zzd("Uploading data. app, uncompressed size, data", size > 0 ? zza2.zzb(0).zzx() : "?", Integer.valueOf(zzbu2.length), str2);
                                            this.zzu = true;
                                            zzen zzenVar2 = this.zzd;
                                            zzal(zzenVar2);
                                            zzkk zzkkVar = new zzkk(this, zzr);
                                            zzenVar2.zzg();
                                            zzenVar2.zzW();
                                            Preconditions.checkNotNull(url);
                                            Preconditions.checkNotNull(zzbu2);
                                            Preconditions.checkNotNull(zzkkVar);
                                            zzenVar2.zzt.zzaz().zzo(new zzem(zzenVar2, zzr, url, zzbu2, null, zzkkVar));
                                        } catch (MalformedURLException unused) {
                                            zzay().zzd().zzc("Failed to parse upload URL. Not uploading. appId", zzeh.zzn(zzr), str3);
                                        }
                                    }
                                } catch (Throwable th5) {
                                    th = th5;
                                    cursor5 = cursor4;
                                }
                            } else {
                                try {
                                    this.zzA = -1L;
                                    zzam zzamVar4 = this.zze;
                                    zzal(zzamVar4);
                                    zzg();
                                    long zzz2 = currentTimeMillis - zzag.zzz();
                                    zzamVar4.zzg();
                                    zzamVar4.zzW();
                                    try {
                                        cursor3 = zzamVar4.zzh().rawQuery("select app_id from apps where app_id in (select distinct app_id from raw_events) and config_fetched_time < ? order by failed_config_fetch_time limit 1;", new String[]{String.valueOf(zzz2)});
                                    } catch (SQLiteException e12) {
                                        e = e12;
                                        cursor3 = null;
                                    } catch (Throwable th6) {
                                        th = th6;
                                        cursor2 = null;
                                        if (cursor2 != null) {
                                            cursor2.close();
                                        }
                                        throw th;
                                    }
                                    try {
                                    } catch (SQLiteException e13) {
                                        e = e13;
                                        zzamVar4.zzt.zzay().zzd().zzb("Error selecting expired configs", e);
                                    }
                                    if (!cursor3.moveToFirst()) {
                                        zzamVar4.zzt.zzay().zzj().zza("No expired configs for apps with pending events");
                                        cursor3.close();
                                        str = null;
                                        if (!TextUtils.isEmpty(str)) {
                                            zzam zzamVar5 = this.zze;
                                            zzal(zzamVar5);
                                            zzh zzj2 = zzamVar5.zzj(str);
                                            if (zzj2 != null) {
                                                zzD(zzj2);
                                            }
                                        }
                                    } else {
                                        str = cursor3.getString(0);
                                        cursor3.close();
                                        if (!TextUtils.isEmpty(str)) {
                                        }
                                    }
                                } catch (Throwable th7) {
                                    th = th7;
                                    cursor2 = cursor;
                                }
                            }
                            this.zzv = false;
                        } catch (Throwable th8) {
                            th = th8;
                            z = false;
                            this.zzv = z;
                            zzae();
                            throw th;
                        }
                    }
                }
            }
            zzae();
        } catch (Throwable th9) {
            th = th9;
            z = false;
        }
    }

    /*  JADX ERROR: IF instruction can be used only in fallback mode
        jadx.core.utils.exceptions.CodegenException: IF instruction can be used only in fallback mode
        	at jadx.core.codegen.InsnGen.fallbackOnlyInsn(InsnGen.java:666)
        	at jadx.core.codegen.InsnGen.makeInsnBody(InsnGen.java:524)
        	at jadx.core.codegen.InsnGen.makeInsn(InsnGen.java:282)
        	at jadx.core.codegen.InsnGen.makeInsn(InsnGen.java:252)
        	at jadx.core.codegen.RegionGen.makeSimpleBlock(RegionGen.java:91)
        	at jadx.core.dex.nodes.IBlock.generate(IBlock.java:15)
        	at jadx.core.codegen.RegionGen.makeRegion(RegionGen.java:63)
        	at jadx.core.dex.regions.Region.generate(Region.java:35)
        	at jadx.core.codegen.RegionGen.makeRegion(RegionGen.java:63)
        	at jadx.core.codegen.RegionGen.makeRegionIndent(RegionGen.java:80)
        	at jadx.core.codegen.RegionGen.makeLoop(RegionGen.java:175)
        	at jadx.core.dex.regions.loops.LoopRegion.generate(LoopRegion.java:171)
        	at jadx.core.codegen.RegionGen.makeRegion(RegionGen.java:63)
        	at jadx.core.dex.regions.Region.generate(Region.java:35)
        	at jadx.core.codegen.RegionGen.makeRegion(RegionGen.java:63)
        	at jadx.core.codegen.RegionGen.makeRegionIndent(RegionGen.java:80)
        	at jadx.core.codegen.RegionGen.makeTryCatch(RegionGen.java:302)
        	at jadx.core.dex.regions.TryCatchRegion.generate(TryCatchRegion.java:85)
        	at jadx.core.codegen.RegionGen.makeRegion(RegionGen.java:63)
        	at jadx.core.dex.regions.Region.generate(Region.java:35)
        	at jadx.core.codegen.RegionGen.makeRegion(RegionGen.java:63)
        	at jadx.core.codegen.RegionGen.makeRegionIndent(RegionGen.java:80)
        	at jadx.core.codegen.RegionGen.makeCatchBlock(RegionGen.java:365)
        	at jadx.core.codegen.RegionGen.makeTryCatch(RegionGen.java:313)
        	at jadx.core.dex.regions.TryCatchRegion.generate(TryCatchRegion.java:85)
        	at jadx.core.codegen.RegionGen.makeRegion(RegionGen.java:63)
        	at jadx.core.dex.regions.Region.generate(Region.java:35)
        	at jadx.core.codegen.RegionGen.makeRegion(RegionGen.java:63)
        	at jadx.core.dex.regions.Region.generate(Region.java:35)
        	at jadx.core.codegen.RegionGen.makeRegion(RegionGen.java:63)
        	at jadx.core.dex.regions.Region.generate(Region.java:35)
        	at jadx.core.codegen.RegionGen.makeRegion(RegionGen.java:63)
        	at jadx.core.codegen.RegionGen.makeRegionIndent(RegionGen.java:80)
        	at jadx.core.codegen.RegionGen.makeIf(RegionGen.java:123)
        	at jadx.core.dex.regions.conditions.IfRegion.generate(IfRegion.java:90)
        	at jadx.core.codegen.RegionGen.makeRegion(RegionGen.java:63)
        	at jadx.core.dex.regions.Region.generate(Region.java:35)
        	at jadx.core.codegen.RegionGen.makeRegion(RegionGen.java:63)
        	at jadx.core.dex.regions.Region.generate(Region.java:35)
        	at jadx.core.codegen.RegionGen.makeRegion(RegionGen.java:63)
        	at jadx.core.codegen.RegionGen.makeRegionIndent(RegionGen.java:80)
        	at jadx.core.codegen.RegionGen.makeIf(RegionGen.java:123)
        	at jadx.core.dex.regions.conditions.IfRegion.generate(IfRegion.java:90)
        	at jadx.core.codegen.RegionGen.makeRegion(RegionGen.java:63)
        	at jadx.core.dex.regions.Region.generate(Region.java:35)
        	at jadx.core.codegen.RegionGen.makeRegion(RegionGen.java:63)
        	at jadx.core.codegen.RegionGen.makeRegionIndent(RegionGen.java:80)
        	at jadx.core.codegen.RegionGen.makeIf(RegionGen.java:123)
        	at jadx.core.dex.regions.conditions.IfRegion.generate(IfRegion.java:90)
        	at jadx.core.codegen.RegionGen.makeRegion(RegionGen.java:63)
        	at jadx.core.dex.regions.Region.generate(Region.java:35)
        	at jadx.core.codegen.RegionGen.makeRegion(RegionGen.java:63)
        	at jadx.core.codegen.RegionGen.makeRegionIndent(RegionGen.java:80)
        	at jadx.core.codegen.RegionGen.makeTryCatch(RegionGen.java:302)
        	at jadx.core.dex.regions.TryCatchRegion.generate(TryCatchRegion.java:85)
        	at jadx.core.codegen.RegionGen.makeRegion(RegionGen.java:63)
        	at jadx.core.dex.regions.Region.generate(Region.java:35)
        	at jadx.core.codegen.RegionGen.makeRegion(RegionGen.java:63)
        	at jadx.core.dex.regions.Region.generate(Region.java:35)
        	at jadx.core.codegen.RegionGen.makeRegion(RegionGen.java:63)
        	at jadx.core.dex.regions.Region.generate(Region.java:35)
        	at jadx.core.codegen.RegionGen.makeRegion(RegionGen.java:63)
        	at jadx.core.codegen.RegionGen.makeRegionIndent(RegionGen.java:80)
        	at jadx.core.codegen.RegionGen.makeIf(RegionGen.java:123)
        	at jadx.core.dex.regions.conditions.IfRegion.generate(IfRegion.java:90)
        	at jadx.core.codegen.RegionGen.makeRegion(RegionGen.java:63)
        	at jadx.core.dex.regions.Region.generate(Region.java:35)
        	at jadx.core.codegen.RegionGen.makeRegion(RegionGen.java:63)
        	at jadx.core.dex.regions.Region.generate(Region.java:35)
        	at jadx.core.codegen.RegionGen.makeRegion(RegionGen.java:63)
        	at jadx.core.dex.regions.Region.generate(Region.java:35)
        	at jadx.core.codegen.RegionGen.makeRegion(RegionGen.java:63)
        	at jadx.core.codegen.MethodGen.addRegionInsns(MethodGen.java:296)
        	at jadx.core.codegen.MethodGen.addInstructions(MethodGen.java:275)
        	at jadx.core.codegen.ClassGen.addMethodCode(ClassGen.java:377)
        	at jadx.core.codegen.ClassGen.addMethod(ClassGen.java:306)
        	at jadx.core.codegen.ClassGen.lambda$addInnerClsAndMethods$2(ClassGen.java:272)
        	at java.util.stream.ForEachOps$ForEachOp$OfRef.accept(ForEachOps.java:184)
        	at java.util.ArrayList.forEach(ArrayList.java:1259)
        	at java.util.stream.SortedOps$RefSortingSink.end(SortedOps.java:390)
        	at java.util.stream.Sink$ChainedReference.end(Sink.java:258)
        */
    /* JADX WARN: Can't wrap try/catch for region: R(18:286|(2:288|(1:290)(7:291|292|(1:294)|46|(0)(0)|49|(0)(0)))|295|296|297|298|299|300|301|302|303|304|292|(0)|46|(0)(0)|49|(0)(0)) */
    /* JADX WARN: Code restructure failed: missing block: B:219:0x0745, code lost:
        if (r14.isEmpty() == false) goto L161;
     */
    /* JADX WARN: Code restructure failed: missing block: B:265:0x0939, code lost:
        r13 = 1;
     */
    /* JADX WARN: Code restructure failed: missing block: B:80:0x0277, code lost:
        r0 = e;
     */
    /* JADX WARN: Code restructure failed: missing block: B:82:0x0279, code lost:
        r0 = e;
     */
    /* JADX WARN: Code restructure failed: missing block: B:83:0x027a, code lost:
        r33 = "metadata_fingerprint";
     */
    /* JADX WARN: Code restructure failed: missing block: B:84:0x027d, code lost:
        r0 = e;
     */
    /* JADX WARN: Code restructure failed: missing block: B:85:0x027e, code lost:
        r33 = "metadata_fingerprint";
        r21 = r15;
     */
    /* JADX WARN: Code restructure failed: missing block: B:88:0x0284, code lost:
        r11.zzt.zzay().zzd().zzc("Error pruning currencies. appId", com.google.android.gms.measurement.internal.zzeh.zzn(r10), r0);
     */
    /* JADX WARN: Removed duplicated region for block: B:105:0x036b A[Catch: all -> 0x0a72, TryCatch #1 {all -> 0x0a72, blocks: (B:28:0x0124, B:31:0x0135, B:33:0x013f, B:38:0x014b, B:94:0x02f5, B:103:0x032b, B:105:0x036b, B:107:0x0371, B:108:0x0388, B:112:0x039b, B:114:0x03b2, B:116:0x03b8, B:117:0x03cf, B:122:0x03f9, B:126:0x041a, B:127:0x0431, B:130:0x0442, B:133:0x045f, B:134:0x0473, B:136:0x047d, B:138:0x048c, B:140:0x0492, B:141:0x049b, B:142:0x04a9, B:144:0x04be, B:146:0x04d3, B:158:0x04ff, B:159:0x0514, B:161:0x053e, B:164:0x0556, B:167:0x0599, B:169:0x05c5, B:171:0x0604, B:172:0x0609, B:174:0x0611, B:175:0x0616, B:177:0x061e, B:178:0x0623, B:180:0x0632, B:182:0x0640, B:184:0x0648, B:185:0x064d, B:187:0x0656, B:188:0x065a, B:190:0x0667, B:191:0x066c, B:193:0x0693, B:195:0x069b, B:196:0x06a0, B:198:0x06a8, B:199:0x06ab, B:201:0x06c3, B:204:0x06cb, B:205:0x06e4, B:207:0x06ea, B:209:0x06fe, B:211:0x070a, B:213:0x0717, B:217:0x0731, B:218:0x0741, B:222:0x074a, B:223:0x074d, B:225:0x076b, B:227:0x076f, B:229:0x0781, B:231:0x0785, B:233:0x0790, B:234:0x0799, B:236:0x07d8, B:238:0x07e1, B:239:0x07e4, B:241:0x07f1, B:243:0x0811, B:244:0x081e, B:245:0x0854, B:247:0x085c, B:249:0x0866, B:250:0x0873, B:252:0x087d, B:253:0x088a, B:254:0x0897, B:256:0x089d, B:258:0x08cd, B:259:0x0913, B:260:0x091d, B:261:0x0929, B:263:0x092f, B:272:0x097d, B:273:0x09cb, B:275:0x09db, B:289:0x0a3f, B:278:0x09f3, B:280:0x09f7, B:266:0x093b, B:268:0x0967, B:284:0x0a10, B:285:0x0a27, B:288:0x0a2a, B:168:0x05b7, B:155:0x04e4, B:97:0x030b, B:98:0x0312, B:100:0x0318, B:102:0x0324, B:44:0x015f, B:47:0x016b, B:49:0x0182, B:55:0x01a0, B:63:0x01e0, B:65:0x01e6, B:67:0x01f4, B:69:0x0205, B:72:0x020c, B:90:0x02b3, B:92:0x02be, B:73:0x023a, B:74:0x0257, B:76:0x025e, B:78:0x026f, B:89:0x0297, B:88:0x0284, B:58:0x01ae, B:62:0x01d6), top: B:299:0x0124, inners: #3, #5, #7 }] */
    /* JADX WARN: Removed duplicated region for block: B:111:0x0399  */
    /* JADX WARN: Removed duplicated region for block: B:158:0x04ff A[Catch: all -> 0x0a72, TryCatch #1 {all -> 0x0a72, blocks: (B:28:0x0124, B:31:0x0135, B:33:0x013f, B:38:0x014b, B:94:0x02f5, B:103:0x032b, B:105:0x036b, B:107:0x0371, B:108:0x0388, B:112:0x039b, B:114:0x03b2, B:116:0x03b8, B:117:0x03cf, B:122:0x03f9, B:126:0x041a, B:127:0x0431, B:130:0x0442, B:133:0x045f, B:134:0x0473, B:136:0x047d, B:138:0x048c, B:140:0x0492, B:141:0x049b, B:142:0x04a9, B:144:0x04be, B:146:0x04d3, B:158:0x04ff, B:159:0x0514, B:161:0x053e, B:164:0x0556, B:167:0x0599, B:169:0x05c5, B:171:0x0604, B:172:0x0609, B:174:0x0611, B:175:0x0616, B:177:0x061e, B:178:0x0623, B:180:0x0632, B:182:0x0640, B:184:0x0648, B:185:0x064d, B:187:0x0656, B:188:0x065a, B:190:0x0667, B:191:0x066c, B:193:0x0693, B:195:0x069b, B:196:0x06a0, B:198:0x06a8, B:199:0x06ab, B:201:0x06c3, B:204:0x06cb, B:205:0x06e4, B:207:0x06ea, B:209:0x06fe, B:211:0x070a, B:213:0x0717, B:217:0x0731, B:218:0x0741, B:222:0x074a, B:223:0x074d, B:225:0x076b, B:227:0x076f, B:229:0x0781, B:231:0x0785, B:233:0x0790, B:234:0x0799, B:236:0x07d8, B:238:0x07e1, B:239:0x07e4, B:241:0x07f1, B:243:0x0811, B:244:0x081e, B:245:0x0854, B:247:0x085c, B:249:0x0866, B:250:0x0873, B:252:0x087d, B:253:0x088a, B:254:0x0897, B:256:0x089d, B:258:0x08cd, B:259:0x0913, B:260:0x091d, B:261:0x0929, B:263:0x092f, B:272:0x097d, B:273:0x09cb, B:275:0x09db, B:289:0x0a3f, B:278:0x09f3, B:280:0x09f7, B:266:0x093b, B:268:0x0967, B:284:0x0a10, B:285:0x0a27, B:288:0x0a2a, B:168:0x05b7, B:155:0x04e4, B:97:0x030b, B:98:0x0312, B:100:0x0318, B:102:0x0324, B:44:0x015f, B:47:0x016b, B:49:0x0182, B:55:0x01a0, B:63:0x01e0, B:65:0x01e6, B:67:0x01f4, B:69:0x0205, B:72:0x020c, B:90:0x02b3, B:92:0x02be, B:73:0x023a, B:74:0x0257, B:76:0x025e, B:78:0x026f, B:89:0x0297, B:88:0x0284, B:58:0x01ae, B:62:0x01d6), top: B:299:0x0124, inners: #3, #5, #7 }] */
    /* JADX WARN: Removed duplicated region for block: B:161:0x053e A[Catch: all -> 0x0a72, TryCatch #1 {all -> 0x0a72, blocks: (B:28:0x0124, B:31:0x0135, B:33:0x013f, B:38:0x014b, B:94:0x02f5, B:103:0x032b, B:105:0x036b, B:107:0x0371, B:108:0x0388, B:112:0x039b, B:114:0x03b2, B:116:0x03b8, B:117:0x03cf, B:122:0x03f9, B:126:0x041a, B:127:0x0431, B:130:0x0442, B:133:0x045f, B:134:0x0473, B:136:0x047d, B:138:0x048c, B:140:0x0492, B:141:0x049b, B:142:0x04a9, B:144:0x04be, B:146:0x04d3, B:158:0x04ff, B:159:0x0514, B:161:0x053e, B:164:0x0556, B:167:0x0599, B:169:0x05c5, B:171:0x0604, B:172:0x0609, B:174:0x0611, B:175:0x0616, B:177:0x061e, B:178:0x0623, B:180:0x0632, B:182:0x0640, B:184:0x0648, B:185:0x064d, B:187:0x0656, B:188:0x065a, B:190:0x0667, B:191:0x066c, B:193:0x0693, B:195:0x069b, B:196:0x06a0, B:198:0x06a8, B:199:0x06ab, B:201:0x06c3, B:204:0x06cb, B:205:0x06e4, B:207:0x06ea, B:209:0x06fe, B:211:0x070a, B:213:0x0717, B:217:0x0731, B:218:0x0741, B:222:0x074a, B:223:0x074d, B:225:0x076b, B:227:0x076f, B:229:0x0781, B:231:0x0785, B:233:0x0790, B:234:0x0799, B:236:0x07d8, B:238:0x07e1, B:239:0x07e4, B:241:0x07f1, B:243:0x0811, B:244:0x081e, B:245:0x0854, B:247:0x085c, B:249:0x0866, B:250:0x0873, B:252:0x087d, B:253:0x088a, B:254:0x0897, B:256:0x089d, B:258:0x08cd, B:259:0x0913, B:260:0x091d, B:261:0x0929, B:263:0x092f, B:272:0x097d, B:273:0x09cb, B:275:0x09db, B:289:0x0a3f, B:278:0x09f3, B:280:0x09f7, B:266:0x093b, B:268:0x0967, B:284:0x0a10, B:285:0x0a27, B:288:0x0a2a, B:168:0x05b7, B:155:0x04e4, B:97:0x030b, B:98:0x0312, B:100:0x0318, B:102:0x0324, B:44:0x015f, B:47:0x016b, B:49:0x0182, B:55:0x01a0, B:63:0x01e0, B:65:0x01e6, B:67:0x01f4, B:69:0x0205, B:72:0x020c, B:90:0x02b3, B:92:0x02be, B:73:0x023a, B:74:0x0257, B:76:0x025e, B:78:0x026f, B:89:0x0297, B:88:0x0284, B:58:0x01ae, B:62:0x01d6), top: B:299:0x0124, inners: #3, #5, #7 }] */
    /* JADX WARN: Removed duplicated region for block: B:168:0x05b7 A[Catch: all -> 0x0a72, TryCatch #1 {all -> 0x0a72, blocks: (B:28:0x0124, B:31:0x0135, B:33:0x013f, B:38:0x014b, B:94:0x02f5, B:103:0x032b, B:105:0x036b, B:107:0x0371, B:108:0x0388, B:112:0x039b, B:114:0x03b2, B:116:0x03b8, B:117:0x03cf, B:122:0x03f9, B:126:0x041a, B:127:0x0431, B:130:0x0442, B:133:0x045f, B:134:0x0473, B:136:0x047d, B:138:0x048c, B:140:0x0492, B:141:0x049b, B:142:0x04a9, B:144:0x04be, B:146:0x04d3, B:158:0x04ff, B:159:0x0514, B:161:0x053e, B:164:0x0556, B:167:0x0599, B:169:0x05c5, B:171:0x0604, B:172:0x0609, B:174:0x0611, B:175:0x0616, B:177:0x061e, B:178:0x0623, B:180:0x0632, B:182:0x0640, B:184:0x0648, B:185:0x064d, B:187:0x0656, B:188:0x065a, B:190:0x0667, B:191:0x066c, B:193:0x0693, B:195:0x069b, B:196:0x06a0, B:198:0x06a8, B:199:0x06ab, B:201:0x06c3, B:204:0x06cb, B:205:0x06e4, B:207:0x06ea, B:209:0x06fe, B:211:0x070a, B:213:0x0717, B:217:0x0731, B:218:0x0741, B:222:0x074a, B:223:0x074d, B:225:0x076b, B:227:0x076f, B:229:0x0781, B:231:0x0785, B:233:0x0790, B:234:0x0799, B:236:0x07d8, B:238:0x07e1, B:239:0x07e4, B:241:0x07f1, B:243:0x0811, B:244:0x081e, B:245:0x0854, B:247:0x085c, B:249:0x0866, B:250:0x0873, B:252:0x087d, B:253:0x088a, B:254:0x0897, B:256:0x089d, B:258:0x08cd, B:259:0x0913, B:260:0x091d, B:261:0x0929, B:263:0x092f, B:272:0x097d, B:273:0x09cb, B:275:0x09db, B:289:0x0a3f, B:278:0x09f3, B:280:0x09f7, B:266:0x093b, B:268:0x0967, B:284:0x0a10, B:285:0x0a27, B:288:0x0a2a, B:168:0x05b7, B:155:0x04e4, B:97:0x030b, B:98:0x0312, B:100:0x0318, B:102:0x0324, B:44:0x015f, B:47:0x016b, B:49:0x0182, B:55:0x01a0, B:63:0x01e0, B:65:0x01e6, B:67:0x01f4, B:69:0x0205, B:72:0x020c, B:90:0x02b3, B:92:0x02be, B:73:0x023a, B:74:0x0257, B:76:0x025e, B:78:0x026f, B:89:0x0297, B:88:0x0284, B:58:0x01ae, B:62:0x01d6), top: B:299:0x0124, inners: #3, #5, #7 }] */
    /* JADX WARN: Removed duplicated region for block: B:171:0x0604 A[Catch: all -> 0x0a72, TryCatch #1 {all -> 0x0a72, blocks: (B:28:0x0124, B:31:0x0135, B:33:0x013f, B:38:0x014b, B:94:0x02f5, B:103:0x032b, B:105:0x036b, B:107:0x0371, B:108:0x0388, B:112:0x039b, B:114:0x03b2, B:116:0x03b8, B:117:0x03cf, B:122:0x03f9, B:126:0x041a, B:127:0x0431, B:130:0x0442, B:133:0x045f, B:134:0x0473, B:136:0x047d, B:138:0x048c, B:140:0x0492, B:141:0x049b, B:142:0x04a9, B:144:0x04be, B:146:0x04d3, B:158:0x04ff, B:159:0x0514, B:161:0x053e, B:164:0x0556, B:167:0x0599, B:169:0x05c5, B:171:0x0604, B:172:0x0609, B:174:0x0611, B:175:0x0616, B:177:0x061e, B:178:0x0623, B:180:0x0632, B:182:0x0640, B:184:0x0648, B:185:0x064d, B:187:0x0656, B:188:0x065a, B:190:0x0667, B:191:0x066c, B:193:0x0693, B:195:0x069b, B:196:0x06a0, B:198:0x06a8, B:199:0x06ab, B:201:0x06c3, B:204:0x06cb, B:205:0x06e4, B:207:0x06ea, B:209:0x06fe, B:211:0x070a, B:213:0x0717, B:217:0x0731, B:218:0x0741, B:222:0x074a, B:223:0x074d, B:225:0x076b, B:227:0x076f, B:229:0x0781, B:231:0x0785, B:233:0x0790, B:234:0x0799, B:236:0x07d8, B:238:0x07e1, B:239:0x07e4, B:241:0x07f1, B:243:0x0811, B:244:0x081e, B:245:0x0854, B:247:0x085c, B:249:0x0866, B:250:0x0873, B:252:0x087d, B:253:0x088a, B:254:0x0897, B:256:0x089d, B:258:0x08cd, B:259:0x0913, B:260:0x091d, B:261:0x0929, B:263:0x092f, B:272:0x097d, B:273:0x09cb, B:275:0x09db, B:289:0x0a3f, B:278:0x09f3, B:280:0x09f7, B:266:0x093b, B:268:0x0967, B:284:0x0a10, B:285:0x0a27, B:288:0x0a2a, B:168:0x05b7, B:155:0x04e4, B:97:0x030b, B:98:0x0312, B:100:0x0318, B:102:0x0324, B:44:0x015f, B:47:0x016b, B:49:0x0182, B:55:0x01a0, B:63:0x01e0, B:65:0x01e6, B:67:0x01f4, B:69:0x0205, B:72:0x020c, B:90:0x02b3, B:92:0x02be, B:73:0x023a, B:74:0x0257, B:76:0x025e, B:78:0x026f, B:89:0x0297, B:88:0x0284, B:58:0x01ae, B:62:0x01d6), top: B:299:0x0124, inners: #3, #5, #7 }] */
    /* JADX WARN: Removed duplicated region for block: B:174:0x0611 A[Catch: all -> 0x0a72, TryCatch #1 {all -> 0x0a72, blocks: (B:28:0x0124, B:31:0x0135, B:33:0x013f, B:38:0x014b, B:94:0x02f5, B:103:0x032b, B:105:0x036b, B:107:0x0371, B:108:0x0388, B:112:0x039b, B:114:0x03b2, B:116:0x03b8, B:117:0x03cf, B:122:0x03f9, B:126:0x041a, B:127:0x0431, B:130:0x0442, B:133:0x045f, B:134:0x0473, B:136:0x047d, B:138:0x048c, B:140:0x0492, B:141:0x049b, B:142:0x04a9, B:144:0x04be, B:146:0x04d3, B:158:0x04ff, B:159:0x0514, B:161:0x053e, B:164:0x0556, B:167:0x0599, B:169:0x05c5, B:171:0x0604, B:172:0x0609, B:174:0x0611, B:175:0x0616, B:177:0x061e, B:178:0x0623, B:180:0x0632, B:182:0x0640, B:184:0x0648, B:185:0x064d, B:187:0x0656, B:188:0x065a, B:190:0x0667, B:191:0x066c, B:193:0x0693, B:195:0x069b, B:196:0x06a0, B:198:0x06a8, B:199:0x06ab, B:201:0x06c3, B:204:0x06cb, B:205:0x06e4, B:207:0x06ea, B:209:0x06fe, B:211:0x070a, B:213:0x0717, B:217:0x0731, B:218:0x0741, B:222:0x074a, B:223:0x074d, B:225:0x076b, B:227:0x076f, B:229:0x0781, B:231:0x0785, B:233:0x0790, B:234:0x0799, B:236:0x07d8, B:238:0x07e1, B:239:0x07e4, B:241:0x07f1, B:243:0x0811, B:244:0x081e, B:245:0x0854, B:247:0x085c, B:249:0x0866, B:250:0x0873, B:252:0x087d, B:253:0x088a, B:254:0x0897, B:256:0x089d, B:258:0x08cd, B:259:0x0913, B:260:0x091d, B:261:0x0929, B:263:0x092f, B:272:0x097d, B:273:0x09cb, B:275:0x09db, B:289:0x0a3f, B:278:0x09f3, B:280:0x09f7, B:266:0x093b, B:268:0x0967, B:284:0x0a10, B:285:0x0a27, B:288:0x0a2a, B:168:0x05b7, B:155:0x04e4, B:97:0x030b, B:98:0x0312, B:100:0x0318, B:102:0x0324, B:44:0x015f, B:47:0x016b, B:49:0x0182, B:55:0x01a0, B:63:0x01e0, B:65:0x01e6, B:67:0x01f4, B:69:0x0205, B:72:0x020c, B:90:0x02b3, B:92:0x02be, B:73:0x023a, B:74:0x0257, B:76:0x025e, B:78:0x026f, B:89:0x0297, B:88:0x0284, B:58:0x01ae, B:62:0x01d6), top: B:299:0x0124, inners: #3, #5, #7 }] */
    /* JADX WARN: Removed duplicated region for block: B:177:0x061e A[Catch: all -> 0x0a72, TryCatch #1 {all -> 0x0a72, blocks: (B:28:0x0124, B:31:0x0135, B:33:0x013f, B:38:0x014b, B:94:0x02f5, B:103:0x032b, B:105:0x036b, B:107:0x0371, B:108:0x0388, B:112:0x039b, B:114:0x03b2, B:116:0x03b8, B:117:0x03cf, B:122:0x03f9, B:126:0x041a, B:127:0x0431, B:130:0x0442, B:133:0x045f, B:134:0x0473, B:136:0x047d, B:138:0x048c, B:140:0x0492, B:141:0x049b, B:142:0x04a9, B:144:0x04be, B:146:0x04d3, B:158:0x04ff, B:159:0x0514, B:161:0x053e, B:164:0x0556, B:167:0x0599, B:169:0x05c5, B:171:0x0604, B:172:0x0609, B:174:0x0611, B:175:0x0616, B:177:0x061e, B:178:0x0623, B:180:0x0632, B:182:0x0640, B:184:0x0648, B:185:0x064d, B:187:0x0656, B:188:0x065a, B:190:0x0667, B:191:0x066c, B:193:0x0693, B:195:0x069b, B:196:0x06a0, B:198:0x06a8, B:199:0x06ab, B:201:0x06c3, B:204:0x06cb, B:205:0x06e4, B:207:0x06ea, B:209:0x06fe, B:211:0x070a, B:213:0x0717, B:217:0x0731, B:218:0x0741, B:222:0x074a, B:223:0x074d, B:225:0x076b, B:227:0x076f, B:229:0x0781, B:231:0x0785, B:233:0x0790, B:234:0x0799, B:236:0x07d8, B:238:0x07e1, B:239:0x07e4, B:241:0x07f1, B:243:0x0811, B:244:0x081e, B:245:0x0854, B:247:0x085c, B:249:0x0866, B:250:0x0873, B:252:0x087d, B:253:0x088a, B:254:0x0897, B:256:0x089d, B:258:0x08cd, B:259:0x0913, B:260:0x091d, B:261:0x0929, B:263:0x092f, B:272:0x097d, B:273:0x09cb, B:275:0x09db, B:289:0x0a3f, B:278:0x09f3, B:280:0x09f7, B:266:0x093b, B:268:0x0967, B:284:0x0a10, B:285:0x0a27, B:288:0x0a2a, B:168:0x05b7, B:155:0x04e4, B:97:0x030b, B:98:0x0312, B:100:0x0318, B:102:0x0324, B:44:0x015f, B:47:0x016b, B:49:0x0182, B:55:0x01a0, B:63:0x01e0, B:65:0x01e6, B:67:0x01f4, B:69:0x0205, B:72:0x020c, B:90:0x02b3, B:92:0x02be, B:73:0x023a, B:74:0x0257, B:76:0x025e, B:78:0x026f, B:89:0x0297, B:88:0x0284, B:58:0x01ae, B:62:0x01d6), top: B:299:0x0124, inners: #3, #5, #7 }] */
    /* JADX WARN: Removed duplicated region for block: B:187:0x0656 A[Catch: all -> 0x0a72, TryCatch #1 {all -> 0x0a72, blocks: (B:28:0x0124, B:31:0x0135, B:33:0x013f, B:38:0x014b, B:94:0x02f5, B:103:0x032b, B:105:0x036b, B:107:0x0371, B:108:0x0388, B:112:0x039b, B:114:0x03b2, B:116:0x03b8, B:117:0x03cf, B:122:0x03f9, B:126:0x041a, B:127:0x0431, B:130:0x0442, B:133:0x045f, B:134:0x0473, B:136:0x047d, B:138:0x048c, B:140:0x0492, B:141:0x049b, B:142:0x04a9, B:144:0x04be, B:146:0x04d3, B:158:0x04ff, B:159:0x0514, B:161:0x053e, B:164:0x0556, B:167:0x0599, B:169:0x05c5, B:171:0x0604, B:172:0x0609, B:174:0x0611, B:175:0x0616, B:177:0x061e, B:178:0x0623, B:180:0x0632, B:182:0x0640, B:184:0x0648, B:185:0x064d, B:187:0x0656, B:188:0x065a, B:190:0x0667, B:191:0x066c, B:193:0x0693, B:195:0x069b, B:196:0x06a0, B:198:0x06a8, B:199:0x06ab, B:201:0x06c3, B:204:0x06cb, B:205:0x06e4, B:207:0x06ea, B:209:0x06fe, B:211:0x070a, B:213:0x0717, B:217:0x0731, B:218:0x0741, B:222:0x074a, B:223:0x074d, B:225:0x076b, B:227:0x076f, B:229:0x0781, B:231:0x0785, B:233:0x0790, B:234:0x0799, B:236:0x07d8, B:238:0x07e1, B:239:0x07e4, B:241:0x07f1, B:243:0x0811, B:244:0x081e, B:245:0x0854, B:247:0x085c, B:249:0x0866, B:250:0x0873, B:252:0x087d, B:253:0x088a, B:254:0x0897, B:256:0x089d, B:258:0x08cd, B:259:0x0913, B:260:0x091d, B:261:0x0929, B:263:0x092f, B:272:0x097d, B:273:0x09cb, B:275:0x09db, B:289:0x0a3f, B:278:0x09f3, B:280:0x09f7, B:266:0x093b, B:268:0x0967, B:284:0x0a10, B:285:0x0a27, B:288:0x0a2a, B:168:0x05b7, B:155:0x04e4, B:97:0x030b, B:98:0x0312, B:100:0x0318, B:102:0x0324, B:44:0x015f, B:47:0x016b, B:49:0x0182, B:55:0x01a0, B:63:0x01e0, B:65:0x01e6, B:67:0x01f4, B:69:0x0205, B:72:0x020c, B:90:0x02b3, B:92:0x02be, B:73:0x023a, B:74:0x0257, B:76:0x025e, B:78:0x026f, B:89:0x0297, B:88:0x0284, B:58:0x01ae, B:62:0x01d6), top: B:299:0x0124, inners: #3, #5, #7 }] */
    /* JADX WARN: Removed duplicated region for block: B:190:0x0667 A[Catch: all -> 0x0a72, TryCatch #1 {all -> 0x0a72, blocks: (B:28:0x0124, B:31:0x0135, B:33:0x013f, B:38:0x014b, B:94:0x02f5, B:103:0x032b, B:105:0x036b, B:107:0x0371, B:108:0x0388, B:112:0x039b, B:114:0x03b2, B:116:0x03b8, B:117:0x03cf, B:122:0x03f9, B:126:0x041a, B:127:0x0431, B:130:0x0442, B:133:0x045f, B:134:0x0473, B:136:0x047d, B:138:0x048c, B:140:0x0492, B:141:0x049b, B:142:0x04a9, B:144:0x04be, B:146:0x04d3, B:158:0x04ff, B:159:0x0514, B:161:0x053e, B:164:0x0556, B:167:0x0599, B:169:0x05c5, B:171:0x0604, B:172:0x0609, B:174:0x0611, B:175:0x0616, B:177:0x061e, B:178:0x0623, B:180:0x0632, B:182:0x0640, B:184:0x0648, B:185:0x064d, B:187:0x0656, B:188:0x065a, B:190:0x0667, B:191:0x066c, B:193:0x0693, B:195:0x069b, B:196:0x06a0, B:198:0x06a8, B:199:0x06ab, B:201:0x06c3, B:204:0x06cb, B:205:0x06e4, B:207:0x06ea, B:209:0x06fe, B:211:0x070a, B:213:0x0717, B:217:0x0731, B:218:0x0741, B:222:0x074a, B:223:0x074d, B:225:0x076b, B:227:0x076f, B:229:0x0781, B:231:0x0785, B:233:0x0790, B:234:0x0799, B:236:0x07d8, B:238:0x07e1, B:239:0x07e4, B:241:0x07f1, B:243:0x0811, B:244:0x081e, B:245:0x0854, B:247:0x085c, B:249:0x0866, B:250:0x0873, B:252:0x087d, B:253:0x088a, B:254:0x0897, B:256:0x089d, B:258:0x08cd, B:259:0x0913, B:260:0x091d, B:261:0x0929, B:263:0x092f, B:272:0x097d, B:273:0x09cb, B:275:0x09db, B:289:0x0a3f, B:278:0x09f3, B:280:0x09f7, B:266:0x093b, B:268:0x0967, B:284:0x0a10, B:285:0x0a27, B:288:0x0a2a, B:168:0x05b7, B:155:0x04e4, B:97:0x030b, B:98:0x0312, B:100:0x0318, B:102:0x0324, B:44:0x015f, B:47:0x016b, B:49:0x0182, B:55:0x01a0, B:63:0x01e0, B:65:0x01e6, B:67:0x01f4, B:69:0x0205, B:72:0x020c, B:90:0x02b3, B:92:0x02be, B:73:0x023a, B:74:0x0257, B:76:0x025e, B:78:0x026f, B:89:0x0297, B:88:0x0284, B:58:0x01ae, B:62:0x01d6), top: B:299:0x0124, inners: #3, #5, #7 }] */
    /* JADX WARN: Removed duplicated region for block: B:198:0x06a8 A[Catch: all -> 0x0a72, TryCatch #1 {all -> 0x0a72, blocks: (B:28:0x0124, B:31:0x0135, B:33:0x013f, B:38:0x014b, B:94:0x02f5, B:103:0x032b, B:105:0x036b, B:107:0x0371, B:108:0x0388, B:112:0x039b, B:114:0x03b2, B:116:0x03b8, B:117:0x03cf, B:122:0x03f9, B:126:0x041a, B:127:0x0431, B:130:0x0442, B:133:0x045f, B:134:0x0473, B:136:0x047d, B:138:0x048c, B:140:0x0492, B:141:0x049b, B:142:0x04a9, B:144:0x04be, B:146:0x04d3, B:158:0x04ff, B:159:0x0514, B:161:0x053e, B:164:0x0556, B:167:0x0599, B:169:0x05c5, B:171:0x0604, B:172:0x0609, B:174:0x0611, B:175:0x0616, B:177:0x061e, B:178:0x0623, B:180:0x0632, B:182:0x0640, B:184:0x0648, B:185:0x064d, B:187:0x0656, B:188:0x065a, B:190:0x0667, B:191:0x066c, B:193:0x0693, B:195:0x069b, B:196:0x06a0, B:198:0x06a8, B:199:0x06ab, B:201:0x06c3, B:204:0x06cb, B:205:0x06e4, B:207:0x06ea, B:209:0x06fe, B:211:0x070a, B:213:0x0717, B:217:0x0731, B:218:0x0741, B:222:0x074a, B:223:0x074d, B:225:0x076b, B:227:0x076f, B:229:0x0781, B:231:0x0785, B:233:0x0790, B:234:0x0799, B:236:0x07d8, B:238:0x07e1, B:239:0x07e4, B:241:0x07f1, B:243:0x0811, B:244:0x081e, B:245:0x0854, B:247:0x085c, B:249:0x0866, B:250:0x0873, B:252:0x087d, B:253:0x088a, B:254:0x0897, B:256:0x089d, B:258:0x08cd, B:259:0x0913, B:260:0x091d, B:261:0x0929, B:263:0x092f, B:272:0x097d, B:273:0x09cb, B:275:0x09db, B:289:0x0a3f, B:278:0x09f3, B:280:0x09f7, B:266:0x093b, B:268:0x0967, B:284:0x0a10, B:285:0x0a27, B:288:0x0a2a, B:168:0x05b7, B:155:0x04e4, B:97:0x030b, B:98:0x0312, B:100:0x0318, B:102:0x0324, B:44:0x015f, B:47:0x016b, B:49:0x0182, B:55:0x01a0, B:63:0x01e0, B:65:0x01e6, B:67:0x01f4, B:69:0x0205, B:72:0x020c, B:90:0x02b3, B:92:0x02be, B:73:0x023a, B:74:0x0257, B:76:0x025e, B:78:0x026f, B:89:0x0297, B:88:0x0284, B:58:0x01ae, B:62:0x01d6), top: B:299:0x0124, inners: #3, #5, #7 }] */
    /* JADX WARN: Removed duplicated region for block: B:207:0x06ea A[Catch: all -> 0x0a72, TRY_LEAVE, TryCatch #1 {all -> 0x0a72, blocks: (B:28:0x0124, B:31:0x0135, B:33:0x013f, B:38:0x014b, B:94:0x02f5, B:103:0x032b, B:105:0x036b, B:107:0x0371, B:108:0x0388, B:112:0x039b, B:114:0x03b2, B:116:0x03b8, B:117:0x03cf, B:122:0x03f9, B:126:0x041a, B:127:0x0431, B:130:0x0442, B:133:0x045f, B:134:0x0473, B:136:0x047d, B:138:0x048c, B:140:0x0492, B:141:0x049b, B:142:0x04a9, B:144:0x04be, B:146:0x04d3, B:158:0x04ff, B:159:0x0514, B:161:0x053e, B:164:0x0556, B:167:0x0599, B:169:0x05c5, B:171:0x0604, B:172:0x0609, B:174:0x0611, B:175:0x0616, B:177:0x061e, B:178:0x0623, B:180:0x0632, B:182:0x0640, B:184:0x0648, B:185:0x064d, B:187:0x0656, B:188:0x065a, B:190:0x0667, B:191:0x066c, B:193:0x0693, B:195:0x069b, B:196:0x06a0, B:198:0x06a8, B:199:0x06ab, B:201:0x06c3, B:204:0x06cb, B:205:0x06e4, B:207:0x06ea, B:209:0x06fe, B:211:0x070a, B:213:0x0717, B:217:0x0731, B:218:0x0741, B:222:0x074a, B:223:0x074d, B:225:0x076b, B:227:0x076f, B:229:0x0781, B:231:0x0785, B:233:0x0790, B:234:0x0799, B:236:0x07d8, B:238:0x07e1, B:239:0x07e4, B:241:0x07f1, B:243:0x0811, B:244:0x081e, B:245:0x0854, B:247:0x085c, B:249:0x0866, B:250:0x0873, B:252:0x087d, B:253:0x088a, B:254:0x0897, B:256:0x089d, B:258:0x08cd, B:259:0x0913, B:260:0x091d, B:261:0x0929, B:263:0x092f, B:272:0x097d, B:273:0x09cb, B:275:0x09db, B:289:0x0a3f, B:278:0x09f3, B:280:0x09f7, B:266:0x093b, B:268:0x0967, B:284:0x0a10, B:285:0x0a27, B:288:0x0a2a, B:168:0x05b7, B:155:0x04e4, B:97:0x030b, B:98:0x0312, B:100:0x0318, B:102:0x0324, B:44:0x015f, B:47:0x016b, B:49:0x0182, B:55:0x01a0, B:63:0x01e0, B:65:0x01e6, B:67:0x01f4, B:69:0x0205, B:72:0x020c, B:90:0x02b3, B:92:0x02be, B:73:0x023a, B:74:0x0257, B:76:0x025e, B:78:0x026f, B:89:0x0297, B:88:0x0284, B:58:0x01ae, B:62:0x01d6), top: B:299:0x0124, inners: #3, #5, #7 }] */
    /* JADX WARN: Removed duplicated region for block: B:222:0x074a A[Catch: all -> 0x0a72, TryCatch #1 {all -> 0x0a72, blocks: (B:28:0x0124, B:31:0x0135, B:33:0x013f, B:38:0x014b, B:94:0x02f5, B:103:0x032b, B:105:0x036b, B:107:0x0371, B:108:0x0388, B:112:0x039b, B:114:0x03b2, B:116:0x03b8, B:117:0x03cf, B:122:0x03f9, B:126:0x041a, B:127:0x0431, B:130:0x0442, B:133:0x045f, B:134:0x0473, B:136:0x047d, B:138:0x048c, B:140:0x0492, B:141:0x049b, B:142:0x04a9, B:144:0x04be, B:146:0x04d3, B:158:0x04ff, B:159:0x0514, B:161:0x053e, B:164:0x0556, B:167:0x0599, B:169:0x05c5, B:171:0x0604, B:172:0x0609, B:174:0x0611, B:175:0x0616, B:177:0x061e, B:178:0x0623, B:180:0x0632, B:182:0x0640, B:184:0x0648, B:185:0x064d, B:187:0x0656, B:188:0x065a, B:190:0x0667, B:191:0x066c, B:193:0x0693, B:195:0x069b, B:196:0x06a0, B:198:0x06a8, B:199:0x06ab, B:201:0x06c3, B:204:0x06cb, B:205:0x06e4, B:207:0x06ea, B:209:0x06fe, B:211:0x070a, B:213:0x0717, B:217:0x0731, B:218:0x0741, B:222:0x074a, B:223:0x074d, B:225:0x076b, B:227:0x076f, B:229:0x0781, B:231:0x0785, B:233:0x0790, B:234:0x0799, B:236:0x07d8, B:238:0x07e1, B:239:0x07e4, B:241:0x07f1, B:243:0x0811, B:244:0x081e, B:245:0x0854, B:247:0x085c, B:249:0x0866, B:250:0x0873, B:252:0x087d, B:253:0x088a, B:254:0x0897, B:256:0x089d, B:258:0x08cd, B:259:0x0913, B:260:0x091d, B:261:0x0929, B:263:0x092f, B:272:0x097d, B:273:0x09cb, B:275:0x09db, B:289:0x0a3f, B:278:0x09f3, B:280:0x09f7, B:266:0x093b, B:268:0x0967, B:284:0x0a10, B:285:0x0a27, B:288:0x0a2a, B:168:0x05b7, B:155:0x04e4, B:97:0x030b, B:98:0x0312, B:100:0x0318, B:102:0x0324, B:44:0x015f, B:47:0x016b, B:49:0x0182, B:55:0x01a0, B:63:0x01e0, B:65:0x01e6, B:67:0x01f4, B:69:0x0205, B:72:0x020c, B:90:0x02b3, B:92:0x02be, B:73:0x023a, B:74:0x0257, B:76:0x025e, B:78:0x026f, B:89:0x0297, B:88:0x0284, B:58:0x01ae, B:62:0x01d6), top: B:299:0x0124, inners: #3, #5, #7 }] */
    /* JADX WARN: Removed duplicated region for block: B:233:0x0790 A[Catch: all -> 0x0a72, TryCatch #1 {all -> 0x0a72, blocks: (B:28:0x0124, B:31:0x0135, B:33:0x013f, B:38:0x014b, B:94:0x02f5, B:103:0x032b, B:105:0x036b, B:107:0x0371, B:108:0x0388, B:112:0x039b, B:114:0x03b2, B:116:0x03b8, B:117:0x03cf, B:122:0x03f9, B:126:0x041a, B:127:0x0431, B:130:0x0442, B:133:0x045f, B:134:0x0473, B:136:0x047d, B:138:0x048c, B:140:0x0492, B:141:0x049b, B:142:0x04a9, B:144:0x04be, B:146:0x04d3, B:158:0x04ff, B:159:0x0514, B:161:0x053e, B:164:0x0556, B:167:0x0599, B:169:0x05c5, B:171:0x0604, B:172:0x0609, B:174:0x0611, B:175:0x0616, B:177:0x061e, B:178:0x0623, B:180:0x0632, B:182:0x0640, B:184:0x0648, B:185:0x064d, B:187:0x0656, B:188:0x065a, B:190:0x0667, B:191:0x066c, B:193:0x0693, B:195:0x069b, B:196:0x06a0, B:198:0x06a8, B:199:0x06ab, B:201:0x06c3, B:204:0x06cb, B:205:0x06e4, B:207:0x06ea, B:209:0x06fe, B:211:0x070a, B:213:0x0717, B:217:0x0731, B:218:0x0741, B:222:0x074a, B:223:0x074d, B:225:0x076b, B:227:0x076f, B:229:0x0781, B:231:0x0785, B:233:0x0790, B:234:0x0799, B:236:0x07d8, B:238:0x07e1, B:239:0x07e4, B:241:0x07f1, B:243:0x0811, B:244:0x081e, B:245:0x0854, B:247:0x085c, B:249:0x0866, B:250:0x0873, B:252:0x087d, B:253:0x088a, B:254:0x0897, B:256:0x089d, B:258:0x08cd, B:259:0x0913, B:260:0x091d, B:261:0x0929, B:263:0x092f, B:272:0x097d, B:273:0x09cb, B:275:0x09db, B:289:0x0a3f, B:278:0x09f3, B:280:0x09f7, B:266:0x093b, B:268:0x0967, B:284:0x0a10, B:285:0x0a27, B:288:0x0a2a, B:168:0x05b7, B:155:0x04e4, B:97:0x030b, B:98:0x0312, B:100:0x0318, B:102:0x0324, B:44:0x015f, B:47:0x016b, B:49:0x0182, B:55:0x01a0, B:63:0x01e0, B:65:0x01e6, B:67:0x01f4, B:69:0x0205, B:72:0x020c, B:90:0x02b3, B:92:0x02be, B:73:0x023a, B:74:0x0257, B:76:0x025e, B:78:0x026f, B:89:0x0297, B:88:0x0284, B:58:0x01ae, B:62:0x01d6), top: B:299:0x0124, inners: #3, #5, #7 }] */
    /* JADX WARN: Removed duplicated region for block: B:236:0x07d8 A[Catch: all -> 0x0a72, TryCatch #1 {all -> 0x0a72, blocks: (B:28:0x0124, B:31:0x0135, B:33:0x013f, B:38:0x014b, B:94:0x02f5, B:103:0x032b, B:105:0x036b, B:107:0x0371, B:108:0x0388, B:112:0x039b, B:114:0x03b2, B:116:0x03b8, B:117:0x03cf, B:122:0x03f9, B:126:0x041a, B:127:0x0431, B:130:0x0442, B:133:0x045f, B:134:0x0473, B:136:0x047d, B:138:0x048c, B:140:0x0492, B:141:0x049b, B:142:0x04a9, B:144:0x04be, B:146:0x04d3, B:158:0x04ff, B:159:0x0514, B:161:0x053e, B:164:0x0556, B:167:0x0599, B:169:0x05c5, B:171:0x0604, B:172:0x0609, B:174:0x0611, B:175:0x0616, B:177:0x061e, B:178:0x0623, B:180:0x0632, B:182:0x0640, B:184:0x0648, B:185:0x064d, B:187:0x0656, B:188:0x065a, B:190:0x0667, B:191:0x066c, B:193:0x0693, B:195:0x069b, B:196:0x06a0, B:198:0x06a8, B:199:0x06ab, B:201:0x06c3, B:204:0x06cb, B:205:0x06e4, B:207:0x06ea, B:209:0x06fe, B:211:0x070a, B:213:0x0717, B:217:0x0731, B:218:0x0741, B:222:0x074a, B:223:0x074d, B:225:0x076b, B:227:0x076f, B:229:0x0781, B:231:0x0785, B:233:0x0790, B:234:0x0799, B:236:0x07d8, B:238:0x07e1, B:239:0x07e4, B:241:0x07f1, B:243:0x0811, B:244:0x081e, B:245:0x0854, B:247:0x085c, B:249:0x0866, B:250:0x0873, B:252:0x087d, B:253:0x088a, B:254:0x0897, B:256:0x089d, B:258:0x08cd, B:259:0x0913, B:260:0x091d, B:261:0x0929, B:263:0x092f, B:272:0x097d, B:273:0x09cb, B:275:0x09db, B:289:0x0a3f, B:278:0x09f3, B:280:0x09f7, B:266:0x093b, B:268:0x0967, B:284:0x0a10, B:285:0x0a27, B:288:0x0a2a, B:168:0x05b7, B:155:0x04e4, B:97:0x030b, B:98:0x0312, B:100:0x0318, B:102:0x0324, B:44:0x015f, B:47:0x016b, B:49:0x0182, B:55:0x01a0, B:63:0x01e0, B:65:0x01e6, B:67:0x01f4, B:69:0x0205, B:72:0x020c, B:90:0x02b3, B:92:0x02be, B:73:0x023a, B:74:0x0257, B:76:0x025e, B:78:0x026f, B:89:0x0297, B:88:0x0284, B:58:0x01ae, B:62:0x01d6), top: B:299:0x0124, inners: #3, #5, #7 }] */
    /* JADX WARN: Removed duplicated region for block: B:241:0x07f1 A[Catch: all -> 0x0a72, TryCatch #1 {all -> 0x0a72, blocks: (B:28:0x0124, B:31:0x0135, B:33:0x013f, B:38:0x014b, B:94:0x02f5, B:103:0x032b, B:105:0x036b, B:107:0x0371, B:108:0x0388, B:112:0x039b, B:114:0x03b2, B:116:0x03b8, B:117:0x03cf, B:122:0x03f9, B:126:0x041a, B:127:0x0431, B:130:0x0442, B:133:0x045f, B:134:0x0473, B:136:0x047d, B:138:0x048c, B:140:0x0492, B:141:0x049b, B:142:0x04a9, B:144:0x04be, B:146:0x04d3, B:158:0x04ff, B:159:0x0514, B:161:0x053e, B:164:0x0556, B:167:0x0599, B:169:0x05c5, B:171:0x0604, B:172:0x0609, B:174:0x0611, B:175:0x0616, B:177:0x061e, B:178:0x0623, B:180:0x0632, B:182:0x0640, B:184:0x0648, B:185:0x064d, B:187:0x0656, B:188:0x065a, B:190:0x0667, B:191:0x066c, B:193:0x0693, B:195:0x069b, B:196:0x06a0, B:198:0x06a8, B:199:0x06ab, B:201:0x06c3, B:204:0x06cb, B:205:0x06e4, B:207:0x06ea, B:209:0x06fe, B:211:0x070a, B:213:0x0717, B:217:0x0731, B:218:0x0741, B:222:0x074a, B:223:0x074d, B:225:0x076b, B:227:0x076f, B:229:0x0781, B:231:0x0785, B:233:0x0790, B:234:0x0799, B:236:0x07d8, B:238:0x07e1, B:239:0x07e4, B:241:0x07f1, B:243:0x0811, B:244:0x081e, B:245:0x0854, B:247:0x085c, B:249:0x0866, B:250:0x0873, B:252:0x087d, B:253:0x088a, B:254:0x0897, B:256:0x089d, B:258:0x08cd, B:259:0x0913, B:260:0x091d, B:261:0x0929, B:263:0x092f, B:272:0x097d, B:273:0x09cb, B:275:0x09db, B:289:0x0a3f, B:278:0x09f3, B:280:0x09f7, B:266:0x093b, B:268:0x0967, B:284:0x0a10, B:285:0x0a27, B:288:0x0a2a, B:168:0x05b7, B:155:0x04e4, B:97:0x030b, B:98:0x0312, B:100:0x0318, B:102:0x0324, B:44:0x015f, B:47:0x016b, B:49:0x0182, B:55:0x01a0, B:63:0x01e0, B:65:0x01e6, B:67:0x01f4, B:69:0x0205, B:72:0x020c, B:90:0x02b3, B:92:0x02be, B:73:0x023a, B:74:0x0257, B:76:0x025e, B:78:0x026f, B:89:0x0297, B:88:0x0284, B:58:0x01ae, B:62:0x01d6), top: B:299:0x0124, inners: #3, #5, #7 }] */
    /* JADX WARN: Removed duplicated region for block: B:252:0x087d A[Catch: all -> 0x0a72, TryCatch #1 {all -> 0x0a72, blocks: (B:28:0x0124, B:31:0x0135, B:33:0x013f, B:38:0x014b, B:94:0x02f5, B:103:0x032b, B:105:0x036b, B:107:0x0371, B:108:0x0388, B:112:0x039b, B:114:0x03b2, B:116:0x03b8, B:117:0x03cf, B:122:0x03f9, B:126:0x041a, B:127:0x0431, B:130:0x0442, B:133:0x045f, B:134:0x0473, B:136:0x047d, B:138:0x048c, B:140:0x0492, B:141:0x049b, B:142:0x04a9, B:144:0x04be, B:146:0x04d3, B:158:0x04ff, B:159:0x0514, B:161:0x053e, B:164:0x0556, B:167:0x0599, B:169:0x05c5, B:171:0x0604, B:172:0x0609, B:174:0x0611, B:175:0x0616, B:177:0x061e, B:178:0x0623, B:180:0x0632, B:182:0x0640, B:184:0x0648, B:185:0x064d, B:187:0x0656, B:188:0x065a, B:190:0x0667, B:191:0x066c, B:193:0x0693, B:195:0x069b, B:196:0x06a0, B:198:0x06a8, B:199:0x06ab, B:201:0x06c3, B:204:0x06cb, B:205:0x06e4, B:207:0x06ea, B:209:0x06fe, B:211:0x070a, B:213:0x0717, B:217:0x0731, B:218:0x0741, B:222:0x074a, B:223:0x074d, B:225:0x076b, B:227:0x076f, B:229:0x0781, B:231:0x0785, B:233:0x0790, B:234:0x0799, B:236:0x07d8, B:238:0x07e1, B:239:0x07e4, B:241:0x07f1, B:243:0x0811, B:244:0x081e, B:245:0x0854, B:247:0x085c, B:249:0x0866, B:250:0x0873, B:252:0x087d, B:253:0x088a, B:254:0x0897, B:256:0x089d, B:258:0x08cd, B:259:0x0913, B:260:0x091d, B:261:0x0929, B:263:0x092f, B:272:0x097d, B:273:0x09cb, B:275:0x09db, B:289:0x0a3f, B:278:0x09f3, B:280:0x09f7, B:266:0x093b, B:268:0x0967, B:284:0x0a10, B:285:0x0a27, B:288:0x0a2a, B:168:0x05b7, B:155:0x04e4, B:97:0x030b, B:98:0x0312, B:100:0x0318, B:102:0x0324, B:44:0x015f, B:47:0x016b, B:49:0x0182, B:55:0x01a0, B:63:0x01e0, B:65:0x01e6, B:67:0x01f4, B:69:0x0205, B:72:0x020c, B:90:0x02b3, B:92:0x02be, B:73:0x023a, B:74:0x0257, B:76:0x025e, B:78:0x026f, B:89:0x0297, B:88:0x0284, B:58:0x01ae, B:62:0x01d6), top: B:299:0x0124, inners: #3, #5, #7 }] */
    /* JADX WARN: Removed duplicated region for block: B:256:0x089d A[Catch: all -> 0x0a72, TRY_LEAVE, TryCatch #1 {all -> 0x0a72, blocks: (B:28:0x0124, B:31:0x0135, B:33:0x013f, B:38:0x014b, B:94:0x02f5, B:103:0x032b, B:105:0x036b, B:107:0x0371, B:108:0x0388, B:112:0x039b, B:114:0x03b2, B:116:0x03b8, B:117:0x03cf, B:122:0x03f9, B:126:0x041a, B:127:0x0431, B:130:0x0442, B:133:0x045f, B:134:0x0473, B:136:0x047d, B:138:0x048c, B:140:0x0492, B:141:0x049b, B:142:0x04a9, B:144:0x04be, B:146:0x04d3, B:158:0x04ff, B:159:0x0514, B:161:0x053e, B:164:0x0556, B:167:0x0599, B:169:0x05c5, B:171:0x0604, B:172:0x0609, B:174:0x0611, B:175:0x0616, B:177:0x061e, B:178:0x0623, B:180:0x0632, B:182:0x0640, B:184:0x0648, B:185:0x064d, B:187:0x0656, B:188:0x065a, B:190:0x0667, B:191:0x066c, B:193:0x0693, B:195:0x069b, B:196:0x06a0, B:198:0x06a8, B:199:0x06ab, B:201:0x06c3, B:204:0x06cb, B:205:0x06e4, B:207:0x06ea, B:209:0x06fe, B:211:0x070a, B:213:0x0717, B:217:0x0731, B:218:0x0741, B:222:0x074a, B:223:0x074d, B:225:0x076b, B:227:0x076f, B:229:0x0781, B:231:0x0785, B:233:0x0790, B:234:0x0799, B:236:0x07d8, B:238:0x07e1, B:239:0x07e4, B:241:0x07f1, B:243:0x0811, B:244:0x081e, B:245:0x0854, B:247:0x085c, B:249:0x0866, B:250:0x0873, B:252:0x087d, B:253:0x088a, B:254:0x0897, B:256:0x089d, B:258:0x08cd, B:259:0x0913, B:260:0x091d, B:261:0x0929, B:263:0x092f, B:272:0x097d, B:273:0x09cb, B:275:0x09db, B:289:0x0a3f, B:278:0x09f3, B:280:0x09f7, B:266:0x093b, B:268:0x0967, B:284:0x0a10, B:285:0x0a27, B:288:0x0a2a, B:168:0x05b7, B:155:0x04e4, B:97:0x030b, B:98:0x0312, B:100:0x0318, B:102:0x0324, B:44:0x015f, B:47:0x016b, B:49:0x0182, B:55:0x01a0, B:63:0x01e0, B:65:0x01e6, B:67:0x01f4, B:69:0x0205, B:72:0x020c, B:90:0x02b3, B:92:0x02be, B:73:0x023a, B:74:0x0257, B:76:0x025e, B:78:0x026f, B:89:0x0297, B:88:0x0284, B:58:0x01ae, B:62:0x01d6), top: B:299:0x0124, inners: #3, #5, #7 }] */
    /* JADX WARN: Removed duplicated region for block: B:263:0x092f A[Catch: all -> 0x0a72, TryCatch #1 {all -> 0x0a72, blocks: (B:28:0x0124, B:31:0x0135, B:33:0x013f, B:38:0x014b, B:94:0x02f5, B:103:0x032b, B:105:0x036b, B:107:0x0371, B:108:0x0388, B:112:0x039b, B:114:0x03b2, B:116:0x03b8, B:117:0x03cf, B:122:0x03f9, B:126:0x041a, B:127:0x0431, B:130:0x0442, B:133:0x045f, B:134:0x0473, B:136:0x047d, B:138:0x048c, B:140:0x0492, B:141:0x049b, B:142:0x04a9, B:144:0x04be, B:146:0x04d3, B:158:0x04ff, B:159:0x0514, B:161:0x053e, B:164:0x0556, B:167:0x0599, B:169:0x05c5, B:171:0x0604, B:172:0x0609, B:174:0x0611, B:175:0x0616, B:177:0x061e, B:178:0x0623, B:180:0x0632, B:182:0x0640, B:184:0x0648, B:185:0x064d, B:187:0x0656, B:188:0x065a, B:190:0x0667, B:191:0x066c, B:193:0x0693, B:195:0x069b, B:196:0x06a0, B:198:0x06a8, B:199:0x06ab, B:201:0x06c3, B:204:0x06cb, B:205:0x06e4, B:207:0x06ea, B:209:0x06fe, B:211:0x070a, B:213:0x0717, B:217:0x0731, B:218:0x0741, B:222:0x074a, B:223:0x074d, B:225:0x076b, B:227:0x076f, B:229:0x0781, B:231:0x0785, B:233:0x0790, B:234:0x0799, B:236:0x07d8, B:238:0x07e1, B:239:0x07e4, B:241:0x07f1, B:243:0x0811, B:244:0x081e, B:245:0x0854, B:247:0x085c, B:249:0x0866, B:250:0x0873, B:252:0x087d, B:253:0x088a, B:254:0x0897, B:256:0x089d, B:258:0x08cd, B:259:0x0913, B:260:0x091d, B:261:0x0929, B:263:0x092f, B:272:0x097d, B:273:0x09cb, B:275:0x09db, B:289:0x0a3f, B:278:0x09f3, B:280:0x09f7, B:266:0x093b, B:268:0x0967, B:284:0x0a10, B:285:0x0a27, B:288:0x0a2a, B:168:0x05b7, B:155:0x04e4, B:97:0x030b, B:98:0x0312, B:100:0x0318, B:102:0x0324, B:44:0x015f, B:47:0x016b, B:49:0x0182, B:55:0x01a0, B:63:0x01e0, B:65:0x01e6, B:67:0x01f4, B:69:0x0205, B:72:0x020c, B:90:0x02b3, B:92:0x02be, B:73:0x023a, B:74:0x0257, B:76:0x025e, B:78:0x026f, B:89:0x0297, B:88:0x0284, B:58:0x01ae, B:62:0x01d6), top: B:299:0x0124, inners: #3, #5, #7 }] */
    /* JADX WARN: Removed duplicated region for block: B:275:0x09db A[Catch: SQLiteException -> 0x09f6, all -> 0x0a72, TRY_LEAVE, TryCatch #5 {SQLiteException -> 0x09f6, blocks: (B:273:0x09cb, B:275:0x09db), top: B:305:0x09cb, outer: #1 }] */
    /* JADX WARN: Removed duplicated region for block: B:277:0x09f1  */
    /* JADX WARN: Removed duplicated region for block: B:327:0x093b A[SYNTHETIC] */
    /* JADX WARN: Removed duplicated region for block: B:40:0x0155  */
    /* JADX WARN: Removed duplicated region for block: B:47:0x016b A[Catch: all -> 0x0a72, TRY_ENTER, TryCatch #1 {all -> 0x0a72, blocks: (B:28:0x0124, B:31:0x0135, B:33:0x013f, B:38:0x014b, B:94:0x02f5, B:103:0x032b, B:105:0x036b, B:107:0x0371, B:108:0x0388, B:112:0x039b, B:114:0x03b2, B:116:0x03b8, B:117:0x03cf, B:122:0x03f9, B:126:0x041a, B:127:0x0431, B:130:0x0442, B:133:0x045f, B:134:0x0473, B:136:0x047d, B:138:0x048c, B:140:0x0492, B:141:0x049b, B:142:0x04a9, B:144:0x04be, B:146:0x04d3, B:158:0x04ff, B:159:0x0514, B:161:0x053e, B:164:0x0556, B:167:0x0599, B:169:0x05c5, B:171:0x0604, B:172:0x0609, B:174:0x0611, B:175:0x0616, B:177:0x061e, B:178:0x0623, B:180:0x0632, B:182:0x0640, B:184:0x0648, B:185:0x064d, B:187:0x0656, B:188:0x065a, B:190:0x0667, B:191:0x066c, B:193:0x0693, B:195:0x069b, B:196:0x06a0, B:198:0x06a8, B:199:0x06ab, B:201:0x06c3, B:204:0x06cb, B:205:0x06e4, B:207:0x06ea, B:209:0x06fe, B:211:0x070a, B:213:0x0717, B:217:0x0731, B:218:0x0741, B:222:0x074a, B:223:0x074d, B:225:0x076b, B:227:0x076f, B:229:0x0781, B:231:0x0785, B:233:0x0790, B:234:0x0799, B:236:0x07d8, B:238:0x07e1, B:239:0x07e4, B:241:0x07f1, B:243:0x0811, B:244:0x081e, B:245:0x0854, B:247:0x085c, B:249:0x0866, B:250:0x0873, B:252:0x087d, B:253:0x088a, B:254:0x0897, B:256:0x089d, B:258:0x08cd, B:259:0x0913, B:260:0x091d, B:261:0x0929, B:263:0x092f, B:272:0x097d, B:273:0x09cb, B:275:0x09db, B:289:0x0a3f, B:278:0x09f3, B:280:0x09f7, B:266:0x093b, B:268:0x0967, B:284:0x0a10, B:285:0x0a27, B:288:0x0a2a, B:168:0x05b7, B:155:0x04e4, B:97:0x030b, B:98:0x0312, B:100:0x0318, B:102:0x0324, B:44:0x015f, B:47:0x016b, B:49:0x0182, B:55:0x01a0, B:63:0x01e0, B:65:0x01e6, B:67:0x01f4, B:69:0x0205, B:72:0x020c, B:90:0x02b3, B:92:0x02be, B:73:0x023a, B:74:0x0257, B:76:0x025e, B:78:0x026f, B:89:0x0297, B:88:0x0284, B:58:0x01ae, B:62:0x01d6), top: B:299:0x0124, inners: #3, #5, #7 }] */
    /* JADX WARN: Removed duplicated region for block: B:61:0x01d4  */
    /* JADX WARN: Removed duplicated region for block: B:65:0x01e6 A[Catch: all -> 0x0a72, TryCatch #1 {all -> 0x0a72, blocks: (B:28:0x0124, B:31:0x0135, B:33:0x013f, B:38:0x014b, B:94:0x02f5, B:103:0x032b, B:105:0x036b, B:107:0x0371, B:108:0x0388, B:112:0x039b, B:114:0x03b2, B:116:0x03b8, B:117:0x03cf, B:122:0x03f9, B:126:0x041a, B:127:0x0431, B:130:0x0442, B:133:0x045f, B:134:0x0473, B:136:0x047d, B:138:0x048c, B:140:0x0492, B:141:0x049b, B:142:0x04a9, B:144:0x04be, B:146:0x04d3, B:158:0x04ff, B:159:0x0514, B:161:0x053e, B:164:0x0556, B:167:0x0599, B:169:0x05c5, B:171:0x0604, B:172:0x0609, B:174:0x0611, B:175:0x0616, B:177:0x061e, B:178:0x0623, B:180:0x0632, B:182:0x0640, B:184:0x0648, B:185:0x064d, B:187:0x0656, B:188:0x065a, B:190:0x0667, B:191:0x066c, B:193:0x0693, B:195:0x069b, B:196:0x06a0, B:198:0x06a8, B:199:0x06ab, B:201:0x06c3, B:204:0x06cb, B:205:0x06e4, B:207:0x06ea, B:209:0x06fe, B:211:0x070a, B:213:0x0717, B:217:0x0731, B:218:0x0741, B:222:0x074a, B:223:0x074d, B:225:0x076b, B:227:0x076f, B:229:0x0781, B:231:0x0785, B:233:0x0790, B:234:0x0799, B:236:0x07d8, B:238:0x07e1, B:239:0x07e4, B:241:0x07f1, B:243:0x0811, B:244:0x081e, B:245:0x0854, B:247:0x085c, B:249:0x0866, B:250:0x0873, B:252:0x087d, B:253:0x088a, B:254:0x0897, B:256:0x089d, B:258:0x08cd, B:259:0x0913, B:260:0x091d, B:261:0x0929, B:263:0x092f, B:272:0x097d, B:273:0x09cb, B:275:0x09db, B:289:0x0a3f, B:278:0x09f3, B:280:0x09f7, B:266:0x093b, B:268:0x0967, B:284:0x0a10, B:285:0x0a27, B:288:0x0a2a, B:168:0x05b7, B:155:0x04e4, B:97:0x030b, B:98:0x0312, B:100:0x0318, B:102:0x0324, B:44:0x015f, B:47:0x016b, B:49:0x0182, B:55:0x01a0, B:63:0x01e0, B:65:0x01e6, B:67:0x01f4, B:69:0x0205, B:72:0x020c, B:90:0x02b3, B:92:0x02be, B:73:0x023a, B:74:0x0257, B:76:0x025e, B:78:0x026f, B:89:0x0297, B:88:0x0284, B:58:0x01ae, B:62:0x01d6), top: B:299:0x0124, inners: #3, #5, #7 }] */
    /* JADX WARN: Removed duplicated region for block: B:92:0x02be A[Catch: all -> 0x0a72, TryCatch #1 {all -> 0x0a72, blocks: (B:28:0x0124, B:31:0x0135, B:33:0x013f, B:38:0x014b, B:94:0x02f5, B:103:0x032b, B:105:0x036b, B:107:0x0371, B:108:0x0388, B:112:0x039b, B:114:0x03b2, B:116:0x03b8, B:117:0x03cf, B:122:0x03f9, B:126:0x041a, B:127:0x0431, B:130:0x0442, B:133:0x045f, B:134:0x0473, B:136:0x047d, B:138:0x048c, B:140:0x0492, B:141:0x049b, B:142:0x04a9, B:144:0x04be, B:146:0x04d3, B:158:0x04ff, B:159:0x0514, B:161:0x053e, B:164:0x0556, B:167:0x0599, B:169:0x05c5, B:171:0x0604, B:172:0x0609, B:174:0x0611, B:175:0x0616, B:177:0x061e, B:178:0x0623, B:180:0x0632, B:182:0x0640, B:184:0x0648, B:185:0x064d, B:187:0x0656, B:188:0x065a, B:190:0x0667, B:191:0x066c, B:193:0x0693, B:195:0x069b, B:196:0x06a0, B:198:0x06a8, B:199:0x06ab, B:201:0x06c3, B:204:0x06cb, B:205:0x06e4, B:207:0x06ea, B:209:0x06fe, B:211:0x070a, B:213:0x0717, B:217:0x0731, B:218:0x0741, B:222:0x074a, B:223:0x074d, B:225:0x076b, B:227:0x076f, B:229:0x0781, B:231:0x0785, B:233:0x0790, B:234:0x0799, B:236:0x07d8, B:238:0x07e1, B:239:0x07e4, B:241:0x07f1, B:243:0x0811, B:244:0x081e, B:245:0x0854, B:247:0x085c, B:249:0x0866, B:250:0x0873, B:252:0x087d, B:253:0x088a, B:254:0x0897, B:256:0x089d, B:258:0x08cd, B:259:0x0913, B:260:0x091d, B:261:0x0929, B:263:0x092f, B:272:0x097d, B:273:0x09cb, B:275:0x09db, B:289:0x0a3f, B:278:0x09f3, B:280:0x09f7, B:266:0x093b, B:268:0x0967, B:284:0x0a10, B:285:0x0a27, B:288:0x0a2a, B:168:0x05b7, B:155:0x04e4, B:97:0x030b, B:98:0x0312, B:100:0x0318, B:102:0x0324, B:44:0x015f, B:47:0x016b, B:49:0x0182, B:55:0x01a0, B:63:0x01e0, B:65:0x01e6, B:67:0x01f4, B:69:0x0205, B:72:0x020c, B:90:0x02b3, B:92:0x02be, B:73:0x023a, B:74:0x0257, B:76:0x025e, B:78:0x026f, B:89:0x0297, B:88:0x0284, B:58:0x01ae, B:62:0x01d6), top: B:299:0x0124, inners: #3, #5, #7 }] */
    /* JADX WARN: Removed duplicated region for block: B:96:0x0308  */
    /* JADX WARN: Removed duplicated region for block: B:97:0x030b A[Catch: all -> 0x0a72, TryCatch #1 {all -> 0x0a72, blocks: (B:28:0x0124, B:31:0x0135, B:33:0x013f, B:38:0x014b, B:94:0x02f5, B:103:0x032b, B:105:0x036b, B:107:0x0371, B:108:0x0388, B:112:0x039b, B:114:0x03b2, B:116:0x03b8, B:117:0x03cf, B:122:0x03f9, B:126:0x041a, B:127:0x0431, B:130:0x0442, B:133:0x045f, B:134:0x0473, B:136:0x047d, B:138:0x048c, B:140:0x0492, B:141:0x049b, B:142:0x04a9, B:144:0x04be, B:146:0x04d3, B:158:0x04ff, B:159:0x0514, B:161:0x053e, B:164:0x0556, B:167:0x0599, B:169:0x05c5, B:171:0x0604, B:172:0x0609, B:174:0x0611, B:175:0x0616, B:177:0x061e, B:178:0x0623, B:180:0x0632, B:182:0x0640, B:184:0x0648, B:185:0x064d, B:187:0x0656, B:188:0x065a, B:190:0x0667, B:191:0x066c, B:193:0x0693, B:195:0x069b, B:196:0x06a0, B:198:0x06a8, B:199:0x06ab, B:201:0x06c3, B:204:0x06cb, B:205:0x06e4, B:207:0x06ea, B:209:0x06fe, B:211:0x070a, B:213:0x0717, B:217:0x0731, B:218:0x0741, B:222:0x074a, B:223:0x074d, B:225:0x076b, B:227:0x076f, B:229:0x0781, B:231:0x0785, B:233:0x0790, B:234:0x0799, B:236:0x07d8, B:238:0x07e1, B:239:0x07e4, B:241:0x07f1, B:243:0x0811, B:244:0x081e, B:245:0x0854, B:247:0x085c, B:249:0x0866, B:250:0x0873, B:252:0x087d, B:253:0x088a, B:254:0x0897, B:256:0x089d, B:258:0x08cd, B:259:0x0913, B:260:0x091d, B:261:0x0929, B:263:0x092f, B:272:0x097d, B:273:0x09cb, B:275:0x09db, B:289:0x0a3f, B:278:0x09f3, B:280:0x09f7, B:266:0x093b, B:268:0x0967, B:284:0x0a10, B:285:0x0a27, B:288:0x0a2a, B:168:0x05b7, B:155:0x04e4, B:97:0x030b, B:98:0x0312, B:100:0x0318, B:102:0x0324, B:44:0x015f, B:47:0x016b, B:49:0x0182, B:55:0x01a0, B:63:0x01e0, B:65:0x01e6, B:67:0x01f4, B:69:0x0205, B:72:0x020c, B:90:0x02b3, B:92:0x02be, B:73:0x023a, B:74:0x0257, B:76:0x025e, B:78:0x026f, B:89:0x0297, B:88:0x0284, B:58:0x01ae, B:62:0x01d6), top: B:299:0x0124, inners: #3, #5, #7 }] */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public final void zzY(com.google.android.gms.measurement.internal.zzaw r35, com.google.android.gms.measurement.internal.zzq r36) {
        /*
            r34 = this;
            r1 = r34
            r2 = r35
            r3 = r36
            java.lang.String r4 = "metadata_fingerprint"
            java.lang.String r5 = "app_id"
            java.lang.String r6 = "raw_events"
            java.lang.String r7 = "_sno"
            com.google.android.gms.common.internal.Preconditions.checkNotNull(r36)
            java.lang.String r8 = r3.zza
            com.google.android.gms.common.internal.Preconditions.checkNotEmpty(r8)
            long r8 = java.lang.System.nanoTime()
            com.google.android.gms.measurement.internal.zzfo r10 = r34.zzaz()
            r10.zzg()
            r34.zzB()
            java.lang.String r10 = r3.zza
            com.google.android.gms.measurement.internal.zzkv r11 = r1.zzi
            zzal(r11)
            boolean r11 = com.google.android.gms.measurement.internal.zzkv.zzA(r35, r36)
            if (r11 != 0) goto L32
            return
        L32:
            boolean r11 = r3.zzh
            if (r11 == 0) goto La7d
            com.google.android.gms.measurement.internal.zzfi r11 = r1.zzc
            zzal(r11)
            java.lang.String r12 = r2.zza
            boolean r11 = r11.zzr(r10, r12)
            java.lang.String r15 = "_err"
            r14 = 0
            if (r11 == 0) goto Ldf
            com.google.android.gms.measurement.internal.zzeh r3 = r34.zzay()
            com.google.android.gms.measurement.internal.zzef r3 = r3.zzk()
            java.lang.Object r4 = com.google.android.gms.measurement.internal.zzeh.zzn(r10)
            com.google.android.gms.measurement.internal.zzfr r5 = r1.zzn
            com.google.android.gms.measurement.internal.zzec r5 = r5.zzj()
            java.lang.String r6 = r2.zza
            java.lang.String r5 = r5.zzd(r6)
            java.lang.String r6 = "Dropping blocked event. appId"
            r3.zzc(r6, r4, r5)
            com.google.android.gms.measurement.internal.zzfi r3 = r1.zzc
            zzal(r3)
            boolean r3 = r3.zzp(r10)
            if (r3 != 0) goto L97
            com.google.android.gms.measurement.internal.zzfi r3 = r1.zzc
            zzal(r3)
            boolean r3 = r3.zzs(r10)
            if (r3 == 0) goto L7a
            goto L97
        L7a:
            java.lang.String r3 = r2.zza
            boolean r3 = r15.equals(r3)
            if (r3 != 0) goto Lde
            com.google.android.gms.measurement.internal.zzlb r11 = r34.zzv()
            com.google.android.gms.measurement.internal.zzla r12 = r1.zzF
            r14 = 11
            java.lang.String r2 = r2.zza
            r17 = 0
            java.lang.String r15 = "_ev"
            r13 = r10
            r16 = r2
            r11.zzN(r12, r13, r14, r15, r16, r17)
            return
        L97:
            com.google.android.gms.measurement.internal.zzam r2 = r1.zze
            zzal(r2)
            com.google.android.gms.measurement.internal.zzh r2 = r2.zzj(r10)
            if (r2 == 0) goto Lde
            long r3 = r2.zzl()
            long r5 = r2.zzc()
            long r3 = java.lang.Math.max(r3, r5)
            com.google.android.gms.common.util.Clock r5 = r34.zzav()
            long r5 = r5.currentTimeMillis()
            long r5 = r5 - r3
            long r3 = java.lang.Math.abs(r5)
            r34.zzg()
            com.google.android.gms.measurement.internal.zzdt r5 = com.google.android.gms.measurement.internal.zzdu.zzy
            java.lang.Object r5 = r5.zza(r14)
            java.lang.Long r5 = (java.lang.Long) r5
            long r5 = r5.longValue()
            int r3 = (r3 > r5 ? 1 : (r3 == r5 ? 0 : -1))
            if (r3 <= 0) goto Lde
            com.google.android.gms.measurement.internal.zzeh r3 = r34.zzay()
            com.google.android.gms.measurement.internal.zzef r3 = r3.zzc()
            java.lang.String r4 = "Fetching config for blocked app"
            r3.zza(r4)
            r1.zzD(r2)
        Lde:
            return
        Ldf:
            com.google.android.gms.measurement.internal.zzei r2 = com.google.android.gms.measurement.internal.zzei.zzb(r35)
            com.google.android.gms.measurement.internal.zzlb r11 = r34.zzv()
            com.google.android.gms.measurement.internal.zzag r12 = r34.zzg()
            int r12 = r12.zzd(r10)
            r11.zzM(r2, r12)
            com.google.android.gms.measurement.internal.zzaw r2 = r2.zza()
            com.google.android.gms.measurement.internal.zzeh r11 = r34.zzay()
            java.lang.String r11 = r11.zzq()
            r13 = 2
            boolean r11 = android.util.Log.isLoggable(r11, r13)
            if (r11 == 0) goto L11c
            com.google.android.gms.measurement.internal.zzeh r11 = r34.zzay()
            com.google.android.gms.measurement.internal.zzef r11 = r11.zzj()
            com.google.android.gms.measurement.internal.zzfr r12 = r1.zzn
            com.google.android.gms.measurement.internal.zzec r12 = r12.zzj()
            java.lang.String r12 = r12.zzc(r2)
            java.lang.String r13 = "Logging event"
            r11.zzb(r13, r12)
        L11c:
            com.google.android.gms.measurement.internal.zzam r11 = r1.zze
            zzal(r11)
            r11.zzw()
            r1.zzd(r3)     // Catch: java.lang.Throwable -> La72
            java.lang.String r11 = "ecommerce_purchase"
            java.lang.String r12 = r2.zza     // Catch: java.lang.Throwable -> La72
            boolean r11 = r11.equals(r12)     // Catch: java.lang.Throwable -> La72
            java.lang.String r12 = "refund"
            r28 = r8
            if (r11 != 0) goto L14a
            java.lang.String r9 = "purchase"
            java.lang.String r11 = r2.zza     // Catch: java.lang.Throwable -> La72
            boolean r9 = r9.equals(r11)     // Catch: java.lang.Throwable -> La72
            if (r9 != 0) goto L14a
            java.lang.String r9 = r2.zza     // Catch: java.lang.Throwable -> La72
            boolean r9 = r12.equals(r9)     // Catch: java.lang.Throwable -> La72
            if (r9 == 0) goto L148
            goto L14a
        L148:
            r9 = 0
            goto L14b
        L14a:
            r9 = 1
        L14b:
            java.lang.String r11 = "_iap"
            java.lang.String r13 = r2.zza     // Catch: java.lang.Throwable -> La72
            boolean r11 = r11.equals(r13)     // Catch: java.lang.Throwable -> La72
            if (r11 != 0) goto L15f
            if (r9 == 0) goto L159
            r9 = 1
            goto L15f
        L159:
            r33 = r4
            r8 = r15
        L15c:
            r4 = 2
            goto L2f5
        L15f:
            com.google.android.gms.measurement.internal.zzau r11 = r2.zzb     // Catch: java.lang.Throwable -> La72
            java.lang.String r13 = "currency"
            java.lang.String r11 = r11.zzg(r13)     // Catch: java.lang.Throwable -> La72
            java.lang.String r13 = "value"
            if (r9 == 0) goto L1d4
            com.google.android.gms.measurement.internal.zzau r9 = r2.zzb     // Catch: java.lang.Throwable -> La72
            java.lang.Double r9 = r9.zzd(r13)     // Catch: java.lang.Throwable -> La72
            double r17 = r9.doubleValue()     // Catch: java.lang.Throwable -> La72
            r19 = 4696837146684686336(0x412e848000000000, double:1000000.0)
            double r17 = r17 * r19
            r21 = 0
            int r9 = (r17 > r21 ? 1 : (r17 == r21 ? 0 : -1))
            if (r9 != 0) goto L192
            com.google.android.gms.measurement.internal.zzau r9 = r2.zzb     // Catch: java.lang.Throwable -> La72
            java.lang.Long r9 = r9.zze(r13)     // Catch: java.lang.Throwable -> La72
            r21 = r15
            long r14 = r9.longValue()     // Catch: java.lang.Throwable -> La72
            double r13 = (double) r14     // Catch: java.lang.Throwable -> La72
            double r17 = r13 * r19
            goto L194
        L192:
            r21 = r15
        L194:
            r13 = 4890909195324358656(0x43e0000000000000, double:9.223372036854776E18)
            int r9 = (r17 > r13 ? 1 : (r17 == r13 ? 0 : -1))
            if (r9 > 0) goto L1ae
            r13 = -4332462841530417152(0xc3e0000000000000, double:-9.223372036854776E18)
            int r9 = (r17 > r13 ? 1 : (r17 == r13 ? 0 : -1))
            if (r9 < 0) goto L1ae
            long r13 = java.lang.Math.round(r17)     // Catch: java.lang.Throwable -> La72
            java.lang.String r9 = r2.zza     // Catch: java.lang.Throwable -> La72
            boolean r9 = r12.equals(r9)     // Catch: java.lang.Throwable -> La72
            if (r9 == 0) goto L1e0
            long r13 = -r13
            goto L1e0
        L1ae:
            com.google.android.gms.measurement.internal.zzeh r2 = r34.zzay()     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzef r2 = r2.zzk()     // Catch: java.lang.Throwable -> La72
            java.lang.String r3 = "Data lost. Currency value is too big. appId"
            java.lang.Object r4 = com.google.android.gms.measurement.internal.zzeh.zzn(r10)     // Catch: java.lang.Throwable -> La72
            java.lang.Double r5 = java.lang.Double.valueOf(r17)     // Catch: java.lang.Throwable -> La72
            r2.zzc(r3, r4, r5)     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzam r2 = r1.zze     // Catch: java.lang.Throwable -> La72
            zzal(r2)     // Catch: java.lang.Throwable -> La72
            r2.zzC()     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzam r2 = r1.zze
            zzal(r2)
            r2.zzx()
            return
        L1d4:
            r21 = r15
            com.google.android.gms.measurement.internal.zzau r9 = r2.zzb     // Catch: java.lang.Throwable -> La72
            java.lang.Long r9 = r9.zze(r13)     // Catch: java.lang.Throwable -> La72
            long r13 = r9.longValue()     // Catch: java.lang.Throwable -> La72
        L1e0:
            boolean r9 = android.text.TextUtils.isEmpty(r11)     // Catch: java.lang.Throwable -> La72
            if (r9 != 0) goto L2ef
            java.util.Locale r9 = java.util.Locale.US     // Catch: java.lang.Throwable -> La72
            java.lang.String r9 = r11.toUpperCase(r9)     // Catch: java.lang.Throwable -> La72
            java.lang.String r11 = "[A-Z]{3}"
            boolean r11 = r9.matches(r11)     // Catch: java.lang.Throwable -> La72
            if (r11 == 0) goto L2ef
            java.lang.String r11 = "_ltv_"
            java.lang.String r9 = r11.concat(r9)     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzam r11 = r1.zze     // Catch: java.lang.Throwable -> La72
            zzal(r11)     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzky r11 = r11.zzp(r10, r9)     // Catch: java.lang.Throwable -> La72
            if (r11 == 0) goto L23a
            java.lang.Object r11 = r11.zze     // Catch: java.lang.Throwable -> La72
            boolean r12 = r11 instanceof java.lang.Long     // Catch: java.lang.Throwable -> La72
            if (r12 != 0) goto L20c
            goto L23a
        L20c:
            java.lang.Long r11 = (java.lang.Long) r11     // Catch: java.lang.Throwable -> La72
            long r11 = r11.longValue()     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzky r18 = new com.google.android.gms.measurement.internal.zzky     // Catch: java.lang.Throwable -> La72
            java.lang.String r15 = r2.zzc     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.common.util.Clock r17 = r34.zzav()     // Catch: java.lang.Throwable -> La72
            long r19 = r17.currentTimeMillis()     // Catch: java.lang.Throwable -> La72
            long r11 = r11 + r13
            java.lang.Long r17 = java.lang.Long.valueOf(r11)     // Catch: java.lang.Throwable -> La72
            r11 = r18
            r12 = r10
            r14 = 0
            r13 = r15
            r8 = r14
            r15 = 0
            r14 = r9
            r9 = r21
            r15 = r19
            r11.<init>(r12, r13, r14, r15, r17)     // Catch: java.lang.Throwable -> La72
            r33 = r4
            r8 = r9
            r9 = r18
            r4 = 2
            goto L2b3
        L23a:
            r15 = r21
            r8 = 0
            com.google.android.gms.measurement.internal.zzam r11 = r1.zze     // Catch: java.lang.Throwable -> La72
            zzal(r11)     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzag r12 = r34.zzg()     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzdt r8 = com.google.android.gms.measurement.internal.zzdu.zzD     // Catch: java.lang.Throwable -> La72
            int r8 = r12.zze(r10, r8)     // Catch: java.lang.Throwable -> La72
            int r8 = r8 + (-1)
            com.google.android.gms.common.internal.Preconditions.checkNotEmpty(r10)     // Catch: java.lang.Throwable -> La72
            r11.zzg()     // Catch: java.lang.Throwable -> La72
            r11.zzW()     // Catch: java.lang.Throwable -> La72
            android.database.sqlite.SQLiteDatabase r12 = r11.zzh()     // Catch: android.database.sqlite.SQLiteException -> L27d java.lang.Throwable -> La72
            r21 = r15
            r15 = 3
            java.lang.String[] r15 = new java.lang.String[r15]     // Catch: android.database.sqlite.SQLiteException -> L279 java.lang.Throwable -> La72
            r16 = 0
            r15[r16] = r10     // Catch: android.database.sqlite.SQLiteException -> L279 java.lang.Throwable -> La72
            r16 = 1
            r15[r16] = r10     // Catch: android.database.sqlite.SQLiteException -> L279 java.lang.Throwable -> La72
            java.lang.String r8 = java.lang.String.valueOf(r8)     // Catch: android.database.sqlite.SQLiteException -> L279 java.lang.Throwable -> La72
            r33 = r4
            r4 = 2
            r15[r4] = r8     // Catch: android.database.sqlite.SQLiteException -> L277 java.lang.Throwable -> La72
            java.lang.String r8 = "delete from user_attributes where app_id=? and name in (select name from user_attributes where app_id=? and name like '_ltv_%' order by set_timestamp desc limit ?,10);"
            r12.execSQL(r8, r15)     // Catch: android.database.sqlite.SQLiteException -> L277 java.lang.Throwable -> La72
            goto L297
        L277:
            r0 = move-exception
            goto L283
        L279:
            r0 = move-exception
            r33 = r4
            goto L282
        L27d:
            r0 = move-exception
            r33 = r4
            r21 = r15
        L282:
            r4 = 2
        L283:
            r8 = r0
            com.google.android.gms.measurement.internal.zzfr r11 = r11.zzt     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzeh r11 = r11.zzay()     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzef r11 = r11.zzd()     // Catch: java.lang.Throwable -> La72
            java.lang.String r12 = "Error pruning currencies. appId"
            java.lang.Object r15 = com.google.android.gms.measurement.internal.zzeh.zzn(r10)     // Catch: java.lang.Throwable -> La72
            r11.zzc(r12, r15, r8)     // Catch: java.lang.Throwable -> La72
        L297:
            com.google.android.gms.measurement.internal.zzky r18 = new com.google.android.gms.measurement.internal.zzky     // Catch: java.lang.Throwable -> La72
            java.lang.String r8 = r2.zzc     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.common.util.Clock r11 = r34.zzav()     // Catch: java.lang.Throwable -> La72
            long r15 = r11.currentTimeMillis()     // Catch: java.lang.Throwable -> La72
            java.lang.Long r17 = java.lang.Long.valueOf(r13)     // Catch: java.lang.Throwable -> La72
            r11 = r18
            r12 = r10
            r13 = r8
            r14 = r9
            r8 = r21
            r11.<init>(r12, r13, r14, r15, r17)     // Catch: java.lang.Throwable -> La72
            r9 = r18
        L2b3:
            com.google.android.gms.measurement.internal.zzam r11 = r1.zze     // Catch: java.lang.Throwable -> La72
            zzal(r11)     // Catch: java.lang.Throwable -> La72
            boolean r11 = r11.zzL(r9)     // Catch: java.lang.Throwable -> La72
            if (r11 != 0) goto L2f5
            com.google.android.gms.measurement.internal.zzeh r11 = r34.zzay()     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzef r11 = r11.zzd()     // Catch: java.lang.Throwable -> La72
            java.lang.String r12 = "Too many unique user properties are set. Ignoring user property. appId"
            java.lang.Object r13 = com.google.android.gms.measurement.internal.zzeh.zzn(r10)     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzfr r14 = r1.zzn     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzec r14 = r14.zzj()     // Catch: java.lang.Throwable -> La72
            java.lang.String r15 = r9.zzc     // Catch: java.lang.Throwable -> La72
            java.lang.String r14 = r14.zzf(r15)     // Catch: java.lang.Throwable -> La72
            java.lang.Object r9 = r9.zze     // Catch: java.lang.Throwable -> La72
            r11.zzd(r12, r13, r14, r9)     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzlb r11 = r34.zzv()     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzla r12 = r1.zzF     // Catch: java.lang.Throwable -> La72
            r14 = 9
            r15 = 0
            r16 = 0
            r17 = 0
            r13 = r10
            r11.zzN(r12, r13, r14, r15, r16, r17)     // Catch: java.lang.Throwable -> La72
            goto L2f5
        L2ef:
            r33 = r4
            r8 = r21
            goto L15c
        L2f5:
            java.lang.String r9 = r2.zza     // Catch: java.lang.Throwable -> La72
            boolean r9 = com.google.android.gms.measurement.internal.zzlb.zzai(r9)     // Catch: java.lang.Throwable -> La72
            java.lang.String r11 = r2.zza     // Catch: java.lang.Throwable -> La72
            boolean r8 = r8.equals(r11)     // Catch: java.lang.Throwable -> La72
            r34.zzv()     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzau r11 = r2.zzb     // Catch: java.lang.Throwable -> La72
            if (r11 != 0) goto L30b
            r16 = 0
            goto L32b
        L30b:
            com.google.android.gms.measurement.internal.zzat r12 = new com.google.android.gms.measurement.internal.zzat     // Catch: java.lang.Throwable -> La72
            r12.<init>(r11)     // Catch: java.lang.Throwable -> La72
            r16 = 0
        L312:
            boolean r13 = r12.hasNext()     // Catch: java.lang.Throwable -> La72
            if (r13 == 0) goto L32b
            java.lang.String r13 = r12.next()     // Catch: java.lang.Throwable -> La72
            java.lang.Object r13 = r11.zzf(r13)     // Catch: java.lang.Throwable -> La72
            boolean r14 = r13 instanceof android.os.Parcelable[]     // Catch: java.lang.Throwable -> La72
            if (r14 == 0) goto L312
            android.os.Parcelable[] r13 = (android.os.Parcelable[]) r13     // Catch: java.lang.Throwable -> La72
            int r13 = r13.length     // Catch: java.lang.Throwable -> La72
            long r13 = (long) r13     // Catch: java.lang.Throwable -> La72
            long r16 = r16 + r13
            goto L312
        L32b:
            r22 = 1
            long r15 = r16 + r22
            com.google.android.gms.measurement.internal.zzam r11 = r1.zze     // Catch: java.lang.Throwable -> La72
            zzal(r11)     // Catch: java.lang.Throwable -> La72
            long r12 = r34.zza()     // Catch: java.lang.Throwable -> La72
            r17 = 1
            r20 = 0
            r21 = 0
            r30 = r5
            r4 = 0
            r14 = r10
            r18 = r9
            r19 = r20
            r20 = r8
            com.google.android.gms.measurement.internal.zzak r11 = r11.zzm(r12, r14, r15, r17, r18, r19, r20, r21)     // Catch: java.lang.Throwable -> La72
            long r12 = r11.zzb     // Catch: java.lang.Throwable -> La72
            r34.zzg()     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzdt r14 = com.google.android.gms.measurement.internal.zzdu.zzj     // Catch: java.lang.Throwable -> La72
            r15 = 0
            java.lang.Object r14 = r14.zza(r15)     // Catch: java.lang.Throwable -> La72
            java.lang.Integer r14 = (java.lang.Integer) r14     // Catch: java.lang.Throwable -> La72
            int r14 = r14.intValue()     // Catch: java.lang.Throwable -> La72
            r31 = r6
            r16 = r7
            long r6 = (long) r14     // Catch: java.lang.Throwable -> La72
            long r12 = r12 - r6
            int r6 = (r12 > r4 ? 1 : (r12 == r4 ? 0 : -1))
            r17 = 1000(0x3e8, double:4.94E-321)
            if (r6 <= 0) goto L399
            long r12 = r12 % r17
            int r2 = (r12 > r22 ? 1 : (r12 == r22 ? 0 : -1))
            if (r2 != 0) goto L388
            com.google.android.gms.measurement.internal.zzeh r2 = r34.zzay()     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzef r2 = r2.zzd()     // Catch: java.lang.Throwable -> La72
            java.lang.String r3 = "Data loss. Too many events logged. appId, count"
            java.lang.Object r4 = com.google.android.gms.measurement.internal.zzeh.zzn(r10)     // Catch: java.lang.Throwable -> La72
            long r5 = r11.zzb     // Catch: java.lang.Throwable -> La72
            java.lang.Long r5 = java.lang.Long.valueOf(r5)     // Catch: java.lang.Throwable -> La72
            r2.zzc(r3, r4, r5)     // Catch: java.lang.Throwable -> La72
        L388:
            com.google.android.gms.measurement.internal.zzam r2 = r1.zze     // Catch: java.lang.Throwable -> La72
            zzal(r2)     // Catch: java.lang.Throwable -> La72
            r2.zzC()     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzam r2 = r1.zze
            zzal(r2)
            r2.zzx()
            return
        L399:
            if (r9 == 0) goto L3f4
            long r6 = r11.zza     // Catch: java.lang.Throwable -> La72
            r34.zzg()     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzdt r12 = com.google.android.gms.measurement.internal.zzdu.zzl     // Catch: java.lang.Throwable -> La72
            java.lang.Object r12 = r12.zza(r15)     // Catch: java.lang.Throwable -> La72
            java.lang.Integer r12 = (java.lang.Integer) r12     // Catch: java.lang.Throwable -> La72
            int r12 = r12.intValue()     // Catch: java.lang.Throwable -> La72
            long r12 = (long) r12     // Catch: java.lang.Throwable -> La72
            long r6 = r6 - r12
            int r12 = (r6 > r4 ? 1 : (r6 == r4 ? 0 : -1))
            if (r12 <= 0) goto L3f4
            long r6 = r6 % r17
            int r3 = (r6 > r22 ? 1 : (r6 == r22 ? 0 : -1))
            if (r3 != 0) goto L3cf
            com.google.android.gms.measurement.internal.zzeh r3 = r34.zzay()     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzef r3 = r3.zzd()     // Catch: java.lang.Throwable -> La72
            java.lang.String r4 = "Data loss. Too many public events logged. appId, count"
            java.lang.Object r5 = com.google.android.gms.measurement.internal.zzeh.zzn(r10)     // Catch: java.lang.Throwable -> La72
            long r6 = r11.zza     // Catch: java.lang.Throwable -> La72
            java.lang.Long r6 = java.lang.Long.valueOf(r6)     // Catch: java.lang.Throwable -> La72
            r3.zzc(r4, r5, r6)     // Catch: java.lang.Throwable -> La72
        L3cf:
            com.google.android.gms.measurement.internal.zzlb r11 = r34.zzv()     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzla r12 = r1.zzF     // Catch: java.lang.Throwable -> La72
            r14 = 16
            java.lang.String r15 = "_ev"
            java.lang.String r2 = r2.zza     // Catch: java.lang.Throwable -> La72
            r17 = 0
            r13 = r10
            r16 = r2
            r11.zzN(r12, r13, r14, r15, r16, r17)     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzam r2 = r1.zze     // Catch: java.lang.Throwable -> La72
            zzal(r2)     // Catch: java.lang.Throwable -> La72
            r2.zzC()     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzam r2 = r1.zze
            zzal(r2)
            r2.zzx()
            return
        L3f4:
            r6 = 1000000(0xf4240, float:1.401298E-39)
            if (r8 == 0) goto L442
            long r7 = r11.zzd     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzag r12 = r34.zzg()     // Catch: java.lang.Throwable -> La72
            java.lang.String r13 = r3.zza     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzdt r14 = com.google.android.gms.measurement.internal.zzdu.zzk     // Catch: java.lang.Throwable -> La72
            int r12 = r12.zze(r13, r14)     // Catch: java.lang.Throwable -> La72
            int r12 = java.lang.Math.min(r6, r12)     // Catch: java.lang.Throwable -> La72
            r13 = 0
            int r12 = java.lang.Math.max(r13, r12)     // Catch: java.lang.Throwable -> La72
            long r12 = (long) r12     // Catch: java.lang.Throwable -> La72
            long r7 = r7 - r12
            int r12 = (r7 > r4 ? 1 : (r7 == r4 ? 0 : -1))
            if (r12 <= 0) goto L442
            int r2 = (r7 > r22 ? 1 : (r7 == r22 ? 0 : -1))
            if (r2 != 0) goto L431
            com.google.android.gms.measurement.internal.zzeh r2 = r34.zzay()     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzef r2 = r2.zzd()     // Catch: java.lang.Throwable -> La72
            java.lang.String r3 = "Too many error events logged. appId, count"
            java.lang.Object r4 = com.google.android.gms.measurement.internal.zzeh.zzn(r10)     // Catch: java.lang.Throwable -> La72
            long r5 = r11.zzd     // Catch: java.lang.Throwable -> La72
            java.lang.Long r5 = java.lang.Long.valueOf(r5)     // Catch: java.lang.Throwable -> La72
            r2.zzc(r3, r4, r5)     // Catch: java.lang.Throwable -> La72
        L431:
            com.google.android.gms.measurement.internal.zzam r2 = r1.zze     // Catch: java.lang.Throwable -> La72
            zzal(r2)     // Catch: java.lang.Throwable -> La72
            r2.zzC()     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzam r2 = r1.zze
            zzal(r2)
            r2.zzx()
            return
        L442:
            com.google.android.gms.measurement.internal.zzau r7 = r2.zzb     // Catch: java.lang.Throwable -> La72
            android.os.Bundle r7 = r7.zzc()     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzlb r8 = r34.zzv()     // Catch: java.lang.Throwable -> La72
            java.lang.String r11 = "_o"
            java.lang.String r12 = r2.zzc     // Catch: java.lang.Throwable -> La72
            r8.zzO(r7, r11, r12)     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzlb r8 = r34.zzv()     // Catch: java.lang.Throwable -> La72
            boolean r8 = r8.zzae(r10)     // Catch: java.lang.Throwable -> La72
            java.lang.String r14 = "_r"
            if (r8 == 0) goto L473
            com.google.android.gms.measurement.internal.zzlb r8 = r34.zzv()     // Catch: java.lang.Throwable -> La72
            java.lang.Long r11 = java.lang.Long.valueOf(r22)     // Catch: java.lang.Throwable -> La72
            java.lang.String r12 = "_dbg"
            r8.zzO(r7, r12, r11)     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzlb r8 = r34.zzv()     // Catch: java.lang.Throwable -> La72
            r8.zzO(r7, r14, r11)     // Catch: java.lang.Throwable -> La72
        L473:
            java.lang.String r8 = "_s"
            java.lang.String r11 = r2.zza     // Catch: java.lang.Throwable -> La72
            boolean r8 = r8.equals(r11)     // Catch: java.lang.Throwable -> La72
            if (r8 == 0) goto L49b
            com.google.android.gms.measurement.internal.zzam r8 = r1.zze     // Catch: java.lang.Throwable -> La72
            zzal(r8)     // Catch: java.lang.Throwable -> La72
            java.lang.String r11 = r3.zza     // Catch: java.lang.Throwable -> La72
            r12 = r16
            com.google.android.gms.measurement.internal.zzky r8 = r8.zzp(r11, r12)     // Catch: java.lang.Throwable -> La72
            if (r8 == 0) goto L49b
            java.lang.Object r11 = r8.zze     // Catch: java.lang.Throwable -> La72
            boolean r11 = r11 instanceof java.lang.Long     // Catch: java.lang.Throwable -> La72
            if (r11 == 0) goto L49b
            com.google.android.gms.measurement.internal.zzlb r11 = r34.zzv()     // Catch: java.lang.Throwable -> La72
            java.lang.Object r8 = r8.zze     // Catch: java.lang.Throwable -> La72
            r11.zzO(r7, r12, r8)     // Catch: java.lang.Throwable -> La72
        L49b:
            com.google.android.gms.measurement.internal.zzam r8 = r1.zze     // Catch: java.lang.Throwable -> La72
            zzal(r8)     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.common.internal.Preconditions.checkNotEmpty(r10)     // Catch: java.lang.Throwable -> La72
            r8.zzg()     // Catch: java.lang.Throwable -> La72
            r8.zzW()     // Catch: java.lang.Throwable -> La72
            android.database.sqlite.SQLiteDatabase r11 = r8.zzh()     // Catch: android.database.sqlite.SQLiteException -> L4df java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzfr r12 = r8.zzt     // Catch: android.database.sqlite.SQLiteException -> L4df java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzag r12 = r12.zzf()     // Catch: android.database.sqlite.SQLiteException -> L4df java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzdt r13 = com.google.android.gms.measurement.internal.zzdu.zzo     // Catch: android.database.sqlite.SQLiteException -> L4df java.lang.Throwable -> La72
            int r12 = r12.zze(r10, r13)     // Catch: android.database.sqlite.SQLiteException -> L4df java.lang.Throwable -> La72
            int r6 = java.lang.Math.min(r6, r12)     // Catch: android.database.sqlite.SQLiteException -> L4df java.lang.Throwable -> La72
            r13 = 0
            int r6 = java.lang.Math.max(r13, r6)     // Catch: android.database.sqlite.SQLiteException -> L4db java.lang.Throwable -> La72
            java.lang.String r6 = java.lang.String.valueOf(r6)     // Catch: android.database.sqlite.SQLiteException -> L4db java.lang.Throwable -> La72
            r12 = 2
            java.lang.String[] r12 = new java.lang.String[r12]     // Catch: android.database.sqlite.SQLiteException -> L4db java.lang.Throwable -> La72
            r12[r13] = r10     // Catch: android.database.sqlite.SQLiteException -> L4db java.lang.Throwable -> La72
            r16 = 1
            r12[r16] = r6     // Catch: android.database.sqlite.SQLiteException -> L4db java.lang.Throwable -> La72
            java.lang.String r6 = "rowid in (select rowid from raw_events where app_id=? order by rowid desc limit -1 offset ?)"
            r4 = r31
            int r5 = r11.delete(r4, r6, r12)     // Catch: android.database.sqlite.SQLiteException -> L4d9 java.lang.Throwable -> La72
            long r5 = (long) r5
            goto L4f9
        L4d9:
            r0 = move-exception
            goto L4e3
        L4db:
            r0 = move-exception
            r4 = r31
            goto L4e3
        L4df:
            r0 = move-exception
            r4 = r31
            r13 = 0
        L4e3:
            r5 = r0
            com.google.android.gms.measurement.internal.zzfr r6 = r8.zzt     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzeh r6 = r6.zzay()     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzef r6 = r6.zzd()     // Catch: java.lang.Throwable -> La72
            java.lang.String r8 = "Error deleting over the limit events. appId"
            java.lang.Object r11 = com.google.android.gms.measurement.internal.zzeh.zzn(r10)     // Catch: java.lang.Throwable -> La72
            r6.zzc(r8, r11, r5)     // Catch: java.lang.Throwable -> La72
            r5 = 0
        L4f9:
            r11 = 0
            int r8 = (r5 > r11 ? 1 : (r5 == r11 ? 0 : -1))
            if (r8 <= 0) goto L514
            com.google.android.gms.measurement.internal.zzeh r8 = r34.zzay()     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzef r8 = r8.zzk()     // Catch: java.lang.Throwable -> La72
            java.lang.String r11 = "Data lost. Too many events stored on disk, deleted. appId"
            java.lang.Object r12 = com.google.android.gms.measurement.internal.zzeh.zzn(r10)     // Catch: java.lang.Throwable -> La72
            java.lang.Long r5 = java.lang.Long.valueOf(r5)     // Catch: java.lang.Throwable -> La72
            r8.zzc(r11, r12, r5)     // Catch: java.lang.Throwable -> La72
        L514:
            com.google.android.gms.measurement.internal.zzar r5 = new com.google.android.gms.measurement.internal.zzar     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzfr r12 = r1.zzn     // Catch: java.lang.Throwable -> La72
            java.lang.String r6 = r2.zzc     // Catch: java.lang.Throwable -> La72
            java.lang.String r8 = r2.zza     // Catch: java.lang.Throwable -> La72
            long r2 = r2.zzd     // Catch: java.lang.Throwable -> La72
            r18 = 0
            r11 = r5
            r31 = r13
            r13 = r6
            r6 = r14
            r14 = r10
            r32 = r4
            r4 = r15
            r15 = r8
            r16 = r2
            r20 = r7
            r11.<init>(r12, r13, r14, r15, r16, r18, r20)     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzam r2 = r1.zze     // Catch: java.lang.Throwable -> La72
            zzal(r2)     // Catch: java.lang.Throwable -> La72
            java.lang.String r3 = r5.zzb     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzas r2 = r2.zzn(r10, r3)     // Catch: java.lang.Throwable -> La72
            if (r2 != 0) goto L5b7
            com.google.android.gms.measurement.internal.zzam r2 = r1.zze     // Catch: java.lang.Throwable -> La72
            zzal(r2)     // Catch: java.lang.Throwable -> La72
            long r2 = r2.zzf(r10)     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzag r7 = r34.zzg()     // Catch: java.lang.Throwable -> La72
            int r7 = r7.zzb(r10)     // Catch: java.lang.Throwable -> La72
            long r7 = (long) r7     // Catch: java.lang.Throwable -> La72
            int r2 = (r2 > r7 ? 1 : (r2 == r7 ? 0 : -1))
            if (r2 < 0) goto L599
            if (r9 == 0) goto L599
            com.google.android.gms.measurement.internal.zzeh r2 = r34.zzay()     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzef r2 = r2.zzd()     // Catch: java.lang.Throwable -> La72
            java.lang.String r3 = "Too many event names used, ignoring event. appId, name, supported count"
            java.lang.Object r4 = com.google.android.gms.measurement.internal.zzeh.zzn(r10)     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzfr r6 = r1.zzn     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzec r6 = r6.zzj()     // Catch: java.lang.Throwable -> La72
            java.lang.String r5 = r5.zzb     // Catch: java.lang.Throwable -> La72
            java.lang.String r5 = r6.zzd(r5)     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzag r6 = r34.zzg()     // Catch: java.lang.Throwable -> La72
            int r6 = r6.zzb(r10)     // Catch: java.lang.Throwable -> La72
            java.lang.Integer r6 = java.lang.Integer.valueOf(r6)     // Catch: java.lang.Throwable -> La72
            r2.zzd(r3, r4, r5, r6)     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzlb r11 = r34.zzv()     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzla r12 = r1.zzF     // Catch: java.lang.Throwable -> La72
            r14 = 8
            r15 = 0
            r16 = 0
            r17 = 0
            r13 = r10
            r11.zzN(r12, r13, r14, r15, r16, r17)     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzam r2 = r1.zze
            zzal(r2)
            r2.zzx()
            return
        L599:
            com.google.android.gms.measurement.internal.zzas r2 = new com.google.android.gms.measurement.internal.zzas     // Catch: java.lang.Throwable -> La72
            java.lang.String r13 = r5.zzb     // Catch: java.lang.Throwable -> La72
            long r7 = r5.zzd     // Catch: java.lang.Throwable -> La72
            r14 = 0
            r16 = 0
            r18 = 0
            r22 = 0
            r24 = 0
            r25 = 0
            r26 = 0
            r27 = 0
            r11 = r2
            r12 = r10
            r20 = r7
            r11.<init>(r12, r13, r14, r16, r18, r20, r22, r24, r25, r26, r27)     // Catch: java.lang.Throwable -> La72
            goto L5c5
        L5b7:
            com.google.android.gms.measurement.internal.zzfr r3 = r1.zzn     // Catch: java.lang.Throwable -> La72
            long r7 = r2.zzf     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzar r5 = r5.zza(r3, r7)     // Catch: java.lang.Throwable -> La72
            long r7 = r5.zzd     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzas r2 = r2.zzc(r7)     // Catch: java.lang.Throwable -> La72
        L5c5:
            com.google.android.gms.measurement.internal.zzam r3 = r1.zze     // Catch: java.lang.Throwable -> La72
            zzal(r3)     // Catch: java.lang.Throwable -> La72
            r3.zzE(r2)     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzfo r2 = r34.zzaz()     // Catch: java.lang.Throwable -> La72
            r2.zzg()     // Catch: java.lang.Throwable -> La72
            r34.zzB()     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.common.internal.Preconditions.checkNotNull(r5)     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.common.internal.Preconditions.checkNotNull(r36)     // Catch: java.lang.Throwable -> La72
            java.lang.String r2 = r5.zza     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.common.internal.Preconditions.checkNotEmpty(r2)     // Catch: java.lang.Throwable -> La72
            java.lang.String r2 = r5.zza     // Catch: java.lang.Throwable -> La72
            r3 = r36
            java.lang.String r7 = r3.zza     // Catch: java.lang.Throwable -> La72
            boolean r2 = r2.equals(r7)     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.common.internal.Preconditions.checkArgument(r2)     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.internal.measurement.zzgc r2 = com.google.android.gms.internal.measurement.zzgd.zzt()     // Catch: java.lang.Throwable -> La72
            r7 = 1
            r2.zzad(r7)     // Catch: java.lang.Throwable -> La72
            java.lang.String r8 = "android"
            r2.zzZ(r8)     // Catch: java.lang.Throwable -> La72
            java.lang.String r8 = r3.zza     // Catch: java.lang.Throwable -> La72
            boolean r8 = android.text.TextUtils.isEmpty(r8)     // Catch: java.lang.Throwable -> La72
            if (r8 != 0) goto L609
            java.lang.String r8 = r3.zza     // Catch: java.lang.Throwable -> La72
            r2.zzD(r8)     // Catch: java.lang.Throwable -> La72
        L609:
            java.lang.String r8 = r3.zzd     // Catch: java.lang.Throwable -> La72
            boolean r8 = android.text.TextUtils.isEmpty(r8)     // Catch: java.lang.Throwable -> La72
            if (r8 != 0) goto L616
            java.lang.String r8 = r3.zzd     // Catch: java.lang.Throwable -> La72
            r2.zzF(r8)     // Catch: java.lang.Throwable -> La72
        L616:
            java.lang.String r8 = r3.zzc     // Catch: java.lang.Throwable -> La72
            boolean r8 = android.text.TextUtils.isEmpty(r8)     // Catch: java.lang.Throwable -> La72
            if (r8 != 0) goto L623
            java.lang.String r8 = r3.zzc     // Catch: java.lang.Throwable -> La72
            r2.zzG(r8)     // Catch: java.lang.Throwable -> La72
        L623:
            com.google.android.gms.internal.measurement.zzpd.zzc()     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzag r8 = r34.zzg()     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzdt r9 = com.google.android.gms.measurement.internal.zzdu.zzal     // Catch: java.lang.Throwable -> La72
            boolean r8 = r8.zzs(r4, r9)     // Catch: java.lang.Throwable -> La72
            if (r8 == 0) goto L64d
            com.google.android.gms.measurement.internal.zzag r8 = r34.zzg()     // Catch: java.lang.Throwable -> La72
            java.lang.String r9 = r3.zza     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzdt r10 = com.google.android.gms.measurement.internal.zzdu.zzan     // Catch: java.lang.Throwable -> La72
            boolean r8 = r8.zzs(r9, r10)     // Catch: java.lang.Throwable -> La72
            if (r8 == 0) goto L64d
            java.lang.String r8 = r3.zzx     // Catch: java.lang.Throwable -> La72
            boolean r8 = android.text.TextUtils.isEmpty(r8)     // Catch: java.lang.Throwable -> La72
            if (r8 != 0) goto L64d
            java.lang.String r8 = r3.zzx     // Catch: java.lang.Throwable -> La72
            r2.zzah(r8)     // Catch: java.lang.Throwable -> La72
        L64d:
            long r8 = r3.zzj     // Catch: java.lang.Throwable -> La72
            r10 = -2147483648(0xffffffff80000000, double:NaN)
            int r10 = (r8 > r10 ? 1 : (r8 == r10 ? 0 : -1))
            if (r10 == 0) goto L65a
            int r8 = (int) r8     // Catch: java.lang.Throwable -> La72
            r2.zzH(r8)     // Catch: java.lang.Throwable -> La72
        L65a:
            long r8 = r3.zze     // Catch: java.lang.Throwable -> La72
            r2.zzV(r8)     // Catch: java.lang.Throwable -> La72
            java.lang.String r8 = r3.zzb     // Catch: java.lang.Throwable -> La72
            boolean r8 = android.text.TextUtils.isEmpty(r8)     // Catch: java.lang.Throwable -> La72
            if (r8 != 0) goto L66c
            java.lang.String r8 = r3.zzb     // Catch: java.lang.Throwable -> La72
            r2.zzU(r8)     // Catch: java.lang.Throwable -> La72
        L66c:
            java.lang.String r8 = r3.zza     // Catch: java.lang.Throwable -> La72
            java.lang.Object r8 = com.google.android.gms.common.internal.Preconditions.checkNotNull(r8)     // Catch: java.lang.Throwable -> La72
            java.lang.String r8 = (java.lang.String) r8     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzai r8 = r1.zzh(r8)     // Catch: java.lang.Throwable -> La72
            java.lang.String r9 = r3.zzv     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzai r9 = com.google.android.gms.measurement.internal.zzai.zzb(r9)     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzai r8 = r8.zzc(r9)     // Catch: java.lang.Throwable -> La72
            java.lang.String r8 = r8.zzh()     // Catch: java.lang.Throwable -> La72
            r2.zzL(r8)     // Catch: java.lang.Throwable -> La72
            java.lang.String r8 = r2.zzaq()     // Catch: java.lang.Throwable -> La72
            boolean r8 = r8.isEmpty()     // Catch: java.lang.Throwable -> La72
            if (r8 == 0) goto L6a0
            java.lang.String r8 = r3.zzq     // Catch: java.lang.Throwable -> La72
            boolean r8 = android.text.TextUtils.isEmpty(r8)     // Catch: java.lang.Throwable -> La72
            if (r8 != 0) goto L6a0
            java.lang.String r8 = r3.zzq     // Catch: java.lang.Throwable -> La72
            r2.zzC(r8)     // Catch: java.lang.Throwable -> La72
        L6a0:
            long r8 = r3.zzf     // Catch: java.lang.Throwable -> La72
            r10 = 0
            int r12 = (r8 > r10 ? 1 : (r8 == r10 ? 0 : -1))
            if (r12 == 0) goto L6ab
            r2.zzM(r8)     // Catch: java.lang.Throwable -> La72
        L6ab:
            long r8 = r3.zzs     // Catch: java.lang.Throwable -> La72
            r2.zzP(r8)     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzkv r8 = r1.zzi     // Catch: java.lang.Throwable -> La72
            zzal(r8)     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzkt r9 = r8.zzf     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzfr r9 = r9.zzn     // Catch: java.lang.Throwable -> La72
            android.content.Context r9 = r9.zzau()     // Catch: java.lang.Throwable -> La72
            java.util.Map r9 = com.google.android.gms.measurement.internal.zzdu.zzc(r9)     // Catch: java.lang.Throwable -> La72
            if (r9 == 0) goto L747
            boolean r10 = r9.isEmpty()     // Catch: java.lang.Throwable -> La72
            if (r10 == 0) goto L6cb
            goto L747
        L6cb:
            java.util.ArrayList r14 = new java.util.ArrayList     // Catch: java.lang.Throwable -> La72
            r14.<init>()     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzdt r10 = com.google.android.gms.measurement.internal.zzdu.zzO     // Catch: java.lang.Throwable -> La72
            java.lang.Object r10 = r10.zza(r4)     // Catch: java.lang.Throwable -> La72
            java.lang.Integer r10 = (java.lang.Integer) r10     // Catch: java.lang.Throwable -> La72
            int r10 = r10.intValue()     // Catch: java.lang.Throwable -> La72
            java.util.Set r9 = r9.entrySet()     // Catch: java.lang.Throwable -> La72
            java.util.Iterator r9 = r9.iterator()     // Catch: java.lang.Throwable -> La72
        L6e4:
            boolean r11 = r9.hasNext()     // Catch: java.lang.Throwable -> La72
            if (r11 == 0) goto L741
            java.lang.Object r11 = r9.next()     // Catch: java.lang.Throwable -> La72
            java.util.Map$Entry r11 = (java.util.Map.Entry) r11     // Catch: java.lang.Throwable -> La72
            java.lang.Object r12 = r11.getKey()     // Catch: java.lang.Throwable -> La72
            java.lang.String r12 = (java.lang.String) r12     // Catch: java.lang.Throwable -> La72
            java.lang.String r13 = "measurement.id."
            boolean r12 = r12.startsWith(r13)     // Catch: java.lang.Throwable -> La72
            if (r12 == 0) goto L6e4
            java.lang.Object r11 = r11.getValue()     // Catch: java.lang.NumberFormatException -> L72f java.lang.Throwable -> La72
            java.lang.String r11 = (java.lang.String) r11     // Catch: java.lang.NumberFormatException -> L72f java.lang.Throwable -> La72
            int r11 = java.lang.Integer.parseInt(r11)     // Catch: java.lang.NumberFormatException -> L72f java.lang.Throwable -> La72
            if (r11 == 0) goto L6e4
            java.lang.Integer r11 = java.lang.Integer.valueOf(r11)     // Catch: java.lang.NumberFormatException -> L72f java.lang.Throwable -> La72
            r14.add(r11)     // Catch: java.lang.NumberFormatException -> L72f java.lang.Throwable -> La72
            int r11 = r14.size()     // Catch: java.lang.NumberFormatException -> L72f java.lang.Throwable -> La72
            if (r11 < r10) goto L6e4
            com.google.android.gms.measurement.internal.zzfr r11 = r8.zzt     // Catch: java.lang.NumberFormatException -> L72f java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzeh r11 = r11.zzay()     // Catch: java.lang.NumberFormatException -> L72f java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzef r11 = r11.zzk()     // Catch: java.lang.NumberFormatException -> L72f java.lang.Throwable -> La72
            java.lang.String r12 = "Too many experiment IDs. Number of IDs"
            int r13 = r14.size()     // Catch: java.lang.NumberFormatException -> L72f java.lang.Throwable -> La72
            java.lang.Integer r13 = java.lang.Integer.valueOf(r13)     // Catch: java.lang.NumberFormatException -> L72f java.lang.Throwable -> La72
            r11.zzb(r12, r13)     // Catch: java.lang.NumberFormatException -> L72f java.lang.Throwable -> La72
            goto L741
        L72f:
            r0 = move-exception
            r11 = r0
            com.google.android.gms.measurement.internal.zzfr r12 = r8.zzt     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzeh r12 = r12.zzay()     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzef r12 = r12.zzk()     // Catch: java.lang.Throwable -> La72
            java.lang.String r13 = "Experiment ID NumberFormatException"
            r12.zzb(r13, r11)     // Catch: java.lang.Throwable -> La72
            goto L6e4
        L741:
            boolean r8 = r14.isEmpty()     // Catch: java.lang.Throwable -> La72
            if (r8 == 0) goto L748
        L747:
            r14 = r4
        L748:
            if (r14 == 0) goto L74d
            r2.zzh(r14)     // Catch: java.lang.Throwable -> La72
        L74d:
            java.lang.String r8 = r3.zza     // Catch: java.lang.Throwable -> La72
            java.lang.Object r8 = com.google.android.gms.common.internal.Preconditions.checkNotNull(r8)     // Catch: java.lang.Throwable -> La72
            java.lang.String r8 = (java.lang.String) r8     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzai r8 = r1.zzh(r8)     // Catch: java.lang.Throwable -> La72
            java.lang.String r9 = r3.zzv     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzai r9 = com.google.android.gms.measurement.internal.zzai.zzb(r9)     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzai r8 = r8.zzc(r9)     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzah r9 = com.google.android.gms.measurement.internal.zzah.AD_STORAGE     // Catch: java.lang.Throwable -> La72
            boolean r10 = r8.zzi(r9)     // Catch: java.lang.Throwable -> La72
            if (r10 == 0) goto L799
            boolean r10 = r3.zzo     // Catch: java.lang.Throwable -> La72
            if (r10 == 0) goto L799
            com.google.android.gms.measurement.internal.zzjo r10 = r1.zzk     // Catch: java.lang.Throwable -> La72
            java.lang.String r11 = r3.zza     // Catch: java.lang.Throwable -> La72
            android.util.Pair r10 = r10.zzd(r11, r8)     // Catch: java.lang.Throwable -> La72
            java.lang.Object r11 = r10.first     // Catch: java.lang.Throwable -> La72
            java.lang.CharSequence r11 = (java.lang.CharSequence) r11     // Catch: java.lang.Throwable -> La72
            boolean r11 = android.text.TextUtils.isEmpty(r11)     // Catch: java.lang.Throwable -> La72
            if (r11 != 0) goto L799
            boolean r11 = r3.zzo     // Catch: java.lang.Throwable -> La72
            if (r11 == 0) goto L799
            java.lang.Object r11 = r10.first     // Catch: java.lang.Throwable -> La72
            java.lang.String r11 = (java.lang.String) r11     // Catch: java.lang.Throwable -> La72
            r2.zzae(r11)     // Catch: java.lang.Throwable -> La72
            java.lang.Object r10 = r10.second     // Catch: java.lang.Throwable -> La72
            if (r10 == 0) goto L799
            java.lang.Boolean r10 = (java.lang.Boolean) r10     // Catch: java.lang.Throwable -> La72
            boolean r10 = r10.booleanValue()     // Catch: java.lang.Throwable -> La72
            r2.zzX(r10)     // Catch: java.lang.Throwable -> La72
        L799:
            com.google.android.gms.measurement.internal.zzfr r10 = r1.zzn     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzaq r10 = r10.zzg()     // Catch: java.lang.Throwable -> La72
            r10.zzu()     // Catch: java.lang.Throwable -> La72
            java.lang.String r10 = android.os.Build.MODEL     // Catch: java.lang.Throwable -> La72
            r2.zzN(r10)     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzfr r10 = r1.zzn     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzaq r10 = r10.zzg()     // Catch: java.lang.Throwable -> La72
            r10.zzu()     // Catch: java.lang.Throwable -> La72
            java.lang.String r10 = android.os.Build.VERSION.RELEASE     // Catch: java.lang.Throwable -> La72
            r2.zzY(r10)     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzfr r10 = r1.zzn     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzaq r10 = r10.zzg()     // Catch: java.lang.Throwable -> La72
            long r10 = r10.zzb()     // Catch: java.lang.Throwable -> La72
            int r10 = (int) r10     // Catch: java.lang.Throwable -> La72
            r2.zzaj(r10)     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzfr r10 = r1.zzn     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzaq r10 = r10.zzg()     // Catch: java.lang.Throwable -> La72
            java.lang.String r10 = r10.zzc()     // Catch: java.lang.Throwable -> La72
            r2.zzan(r10)     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzfr r10 = r1.zzn     // Catch: java.lang.Throwable -> La72
            boolean r10 = r10.zzJ()     // Catch: java.lang.Throwable -> La72
            if (r10 == 0) goto L7e4
            r2.zzap()     // Catch: java.lang.Throwable -> La72
            boolean r10 = android.text.TextUtils.isEmpty(r4)     // Catch: java.lang.Throwable -> La72
            if (r10 != 0) goto L7e4
            r2.zzO(r4)     // Catch: java.lang.Throwable -> La72
        L7e4:
            com.google.android.gms.measurement.internal.zzam r10 = r1.zze     // Catch: java.lang.Throwable -> La72
            zzal(r10)     // Catch: java.lang.Throwable -> La72
            java.lang.String r11 = r3.zza     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzh r10 = r10.zzj(r11)     // Catch: java.lang.Throwable -> La72
            if (r10 != 0) goto L854
            com.google.android.gms.measurement.internal.zzh r10 = new com.google.android.gms.measurement.internal.zzh     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzfr r11 = r1.zzn     // Catch: java.lang.Throwable -> La72
            java.lang.String r12 = r3.zza     // Catch: java.lang.Throwable -> La72
            r10.<init>(r11, r12)     // Catch: java.lang.Throwable -> La72
            java.lang.String r11 = r1.zzw(r8)     // Catch: java.lang.Throwable -> La72
            r10.zzH(r11)     // Catch: java.lang.Throwable -> La72
            java.lang.String r11 = r3.zzk     // Catch: java.lang.Throwable -> La72
            r10.zzV(r11)     // Catch: java.lang.Throwable -> La72
            java.lang.String r11 = r3.zzb     // Catch: java.lang.Throwable -> La72
            r10.zzW(r11)     // Catch: java.lang.Throwable -> La72
            boolean r9 = r8.zzi(r9)     // Catch: java.lang.Throwable -> La72
            if (r9 == 0) goto L81e
            com.google.android.gms.measurement.internal.zzjo r9 = r1.zzk     // Catch: java.lang.Throwable -> La72
            java.lang.String r11 = r3.zza     // Catch: java.lang.Throwable -> La72
            boolean r12 = r3.zzo     // Catch: java.lang.Throwable -> La72
            java.lang.String r9 = r9.zzf(r11, r12)     // Catch: java.lang.Throwable -> La72
            r10.zzae(r9)     // Catch: java.lang.Throwable -> La72
        L81e:
            r11 = 0
            r10.zzaa(r11)     // Catch: java.lang.Throwable -> La72
            r10.zzab(r11)     // Catch: java.lang.Throwable -> La72
            r10.zzZ(r11)     // Catch: java.lang.Throwable -> La72
            java.lang.String r9 = r3.zzc     // Catch: java.lang.Throwable -> La72
            r10.zzJ(r9)     // Catch: java.lang.Throwable -> La72
            long r11 = r3.zzj     // Catch: java.lang.Throwable -> La72
            r10.zzK(r11)     // Catch: java.lang.Throwable -> La72
            java.lang.String r9 = r3.zzd     // Catch: java.lang.Throwable -> La72
            r10.zzI(r9)     // Catch: java.lang.Throwable -> La72
            long r11 = r3.zze     // Catch: java.lang.Throwable -> La72
            r10.zzX(r11)     // Catch: java.lang.Throwable -> La72
            long r11 = r3.zzf     // Catch: java.lang.Throwable -> La72
            r10.zzS(r11)     // Catch: java.lang.Throwable -> La72
            boolean r9 = r3.zzh     // Catch: java.lang.Throwable -> La72
            r10.zzac(r9)     // Catch: java.lang.Throwable -> La72
            long r11 = r3.zzs     // Catch: java.lang.Throwable -> La72
            r10.zzT(r11)     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzam r9 = r1.zze     // Catch: java.lang.Throwable -> La72
            zzal(r9)     // Catch: java.lang.Throwable -> La72
            r9.zzD(r10)     // Catch: java.lang.Throwable -> La72
        L854:
            com.google.android.gms.measurement.internal.zzah r9 = com.google.android.gms.measurement.internal.zzah.ANALYTICS_STORAGE     // Catch: java.lang.Throwable -> La72
            boolean r8 = r8.zzi(r9)     // Catch: java.lang.Throwable -> La72
            if (r8 == 0) goto L873
            java.lang.String r8 = r10.zzu()     // Catch: java.lang.Throwable -> La72
            boolean r8 = android.text.TextUtils.isEmpty(r8)     // Catch: java.lang.Throwable -> La72
            if (r8 != 0) goto L873
            java.lang.String r8 = r10.zzu()     // Catch: java.lang.Throwable -> La72
            java.lang.Object r8 = com.google.android.gms.common.internal.Preconditions.checkNotNull(r8)     // Catch: java.lang.Throwable -> La72
            java.lang.String r8 = (java.lang.String) r8     // Catch: java.lang.Throwable -> La72
            r2.zzE(r8)     // Catch: java.lang.Throwable -> La72
        L873:
            java.lang.String r8 = r10.zzx()     // Catch: java.lang.Throwable -> La72
            boolean r8 = android.text.TextUtils.isEmpty(r8)     // Catch: java.lang.Throwable -> La72
            if (r8 != 0) goto L88a
            java.lang.String r8 = r10.zzx()     // Catch: java.lang.Throwable -> La72
            java.lang.Object r8 = com.google.android.gms.common.internal.Preconditions.checkNotNull(r8)     // Catch: java.lang.Throwable -> La72
            java.lang.String r8 = (java.lang.String) r8     // Catch: java.lang.Throwable -> La72
            r2.zzT(r8)     // Catch: java.lang.Throwable -> La72
        L88a:
            com.google.android.gms.measurement.internal.zzam r8 = r1.zze     // Catch: java.lang.Throwable -> La72
            zzal(r8)     // Catch: java.lang.Throwable -> La72
            java.lang.String r3 = r3.zza     // Catch: java.lang.Throwable -> La72
            java.util.List r3 = r8.zzu(r3)     // Catch: java.lang.Throwable -> La72
            r13 = r31
        L897:
            int r8 = r3.size()     // Catch: java.lang.Throwable -> La72
            if (r13 >= r8) goto L8cd
            com.google.android.gms.internal.measurement.zzgl r8 = com.google.android.gms.internal.measurement.zzgm.zzd()     // Catch: java.lang.Throwable -> La72
            java.lang.Object r9 = r3.get(r13)     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzky r9 = (com.google.android.gms.measurement.internal.zzky) r9     // Catch: java.lang.Throwable -> La72
            java.lang.String r9 = r9.zzc     // Catch: java.lang.Throwable -> La72
            r8.zzf(r9)     // Catch: java.lang.Throwable -> La72
            java.lang.Object r9 = r3.get(r13)     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzky r9 = (com.google.android.gms.measurement.internal.zzky) r9     // Catch: java.lang.Throwable -> La72
            long r9 = r9.zzd     // Catch: java.lang.Throwable -> La72
            r8.zzg(r9)     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzkv r9 = r1.zzi     // Catch: java.lang.Throwable -> La72
            zzal(r9)     // Catch: java.lang.Throwable -> La72
            java.lang.Object r10 = r3.get(r13)     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzky r10 = (com.google.android.gms.measurement.internal.zzky) r10     // Catch: java.lang.Throwable -> La72
            java.lang.Object r10 = r10.zze     // Catch: java.lang.Throwable -> La72
            r9.zzu(r8, r10)     // Catch: java.lang.Throwable -> La72
            r2.zzl(r8)     // Catch: java.lang.Throwable -> La72
            int r13 = r13 + 1
            goto L897
        L8cd:
            com.google.android.gms.measurement.internal.zzam r3 = r1.zze     // Catch: java.io.IOException -> La28 java.lang.Throwable -> La72
            zzal(r3)     // Catch: java.io.IOException -> La28 java.lang.Throwable -> La72
            com.google.android.gms.internal.measurement.zzkf r8 = r2.zzaC()     // Catch: java.io.IOException -> La28 java.lang.Throwable -> La72
            com.google.android.gms.internal.measurement.zzgd r8 = (com.google.android.gms.internal.measurement.zzgd) r8     // Catch: java.io.IOException -> La28 java.lang.Throwable -> La72
            r3.zzg()     // Catch: java.io.IOException -> La28 java.lang.Throwable -> La72
            r3.zzW()     // Catch: java.io.IOException -> La28 java.lang.Throwable -> La72
            com.google.android.gms.common.internal.Preconditions.checkNotNull(r8)     // Catch: java.io.IOException -> La28 java.lang.Throwable -> La72
            java.lang.String r9 = r8.zzx()     // Catch: java.io.IOException -> La28 java.lang.Throwable -> La72
            com.google.android.gms.common.internal.Preconditions.checkNotEmpty(r9)     // Catch: java.io.IOException -> La28 java.lang.Throwable -> La72
            byte[] r9 = r8.zzbu()     // Catch: java.io.IOException -> La28 java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzkt r10 = r3.zzf     // Catch: java.io.IOException -> La28 java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzkv r10 = r10.zzi     // Catch: java.io.IOException -> La28 java.lang.Throwable -> La72
            zzal(r10)     // Catch: java.io.IOException -> La28 java.lang.Throwable -> La72
            long r10 = r10.zzd(r9)     // Catch: java.io.IOException -> La28 java.lang.Throwable -> La72
            android.content.ContentValues r12 = new android.content.ContentValues     // Catch: java.io.IOException -> La28 java.lang.Throwable -> La72
            r12.<init>()     // Catch: java.io.IOException -> La28 java.lang.Throwable -> La72
            java.lang.String r13 = r8.zzx()     // Catch: java.io.IOException -> La28 java.lang.Throwable -> La72
            r14 = r30
            r12.put(r14, r13)     // Catch: java.io.IOException -> La28 java.lang.Throwable -> La72
            java.lang.Long r13 = java.lang.Long.valueOf(r10)     // Catch: java.io.IOException -> La28 java.lang.Throwable -> La72
            r15 = r33
            r12.put(r15, r13)     // Catch: java.io.IOException -> La28 java.lang.Throwable -> La72
            java.lang.String r13 = "metadata"
            r12.put(r13, r9)     // Catch: java.io.IOException -> La28 java.lang.Throwable -> La72
            android.database.sqlite.SQLiteDatabase r9 = r3.zzh()     // Catch: android.database.sqlite.SQLiteException -> La0e java.io.IOException -> La28 java.lang.Throwable -> La72
            java.lang.String r13 = "raw_events_metadata"
            r7 = 4
            r9.insertWithOnConflict(r13, r4, r12, r7)     // Catch: android.database.sqlite.SQLiteException -> La0e java.io.IOException -> La28 java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzam r2 = r1.zze     // Catch: java.lang.Throwable -> La72
            zzal(r2)     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzau r3 = r5.zzf     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzat r7 = new com.google.android.gms.measurement.internal.zzat     // Catch: java.lang.Throwable -> La72
            r7.<init>(r3)     // Catch: java.lang.Throwable -> La72
        L929:
            boolean r3 = r7.hasNext()     // Catch: java.lang.Throwable -> La72
            if (r3 == 0) goto L93b
            java.lang.String r3 = r7.next()     // Catch: java.lang.Throwable -> La72
            boolean r3 = r6.equals(r3)     // Catch: java.lang.Throwable -> La72
            if (r3 == 0) goto L929
        L939:
            r13 = 1
            goto L97d
        L93b:
            com.google.android.gms.measurement.internal.zzfi r3 = r1.zzc     // Catch: java.lang.Throwable -> La72
            zzal(r3)     // Catch: java.lang.Throwable -> La72
            java.lang.String r6 = r5.zza     // Catch: java.lang.Throwable -> La72
            java.lang.String r7 = r5.zzb     // Catch: java.lang.Throwable -> La72
            boolean r3 = r3.zzq(r6, r7)     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzam r6 = r1.zze     // Catch: java.lang.Throwable -> La72
            zzal(r6)     // Catch: java.lang.Throwable -> La72
            long r17 = r34.zza()     // Catch: java.lang.Throwable -> La72
            java.lang.String r7 = r5.zza     // Catch: java.lang.Throwable -> La72
            r20 = 0
            r21 = 0
            r22 = 0
            r23 = 0
            r24 = 0
            r16 = r6
            r19 = r7
            com.google.android.gms.measurement.internal.zzak r6 = r16.zzl(r17, r19, r20, r21, r22, r23, r24)     // Catch: java.lang.Throwable -> La72
            if (r3 == 0) goto L97b
            long r6 = r6.zze     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzag r3 = r34.zzg()     // Catch: java.lang.Throwable -> La72
            java.lang.String r8 = r5.zza     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzdt r9 = com.google.android.gms.measurement.internal.zzdu.zzn     // Catch: java.lang.Throwable -> La72
            int r3 = r3.zze(r8, r9)     // Catch: java.lang.Throwable -> La72
            long r8 = (long) r3     // Catch: java.lang.Throwable -> La72
            int r3 = (r6 > r8 ? 1 : (r6 == r8 ? 0 : -1))
            if (r3 >= 0) goto L97b
            goto L939
        L97b:
            r13 = r31
        L97d:
            r2.zzg()     // Catch: java.lang.Throwable -> La72
            r2.zzW()     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.common.internal.Preconditions.checkNotNull(r5)     // Catch: java.lang.Throwable -> La72
            java.lang.String r3 = r5.zza     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.common.internal.Preconditions.checkNotEmpty(r3)     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzkt r3 = r2.zzf     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzkv r3 = r3.zzi     // Catch: java.lang.Throwable -> La72
            zzal(r3)     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.internal.measurement.zzft r3 = r3.zzj(r5)     // Catch: java.lang.Throwable -> La72
            byte[] r3 = r3.zzbu()     // Catch: java.lang.Throwable -> La72
            android.content.ContentValues r6 = new android.content.ContentValues     // Catch: java.lang.Throwable -> La72
            r6.<init>()     // Catch: java.lang.Throwable -> La72
            java.lang.String r7 = r5.zza     // Catch: java.lang.Throwable -> La72
            r6.put(r14, r7)     // Catch: java.lang.Throwable -> La72
            java.lang.String r7 = "name"
            java.lang.String r8 = r5.zzb     // Catch: java.lang.Throwable -> La72
            r6.put(r7, r8)     // Catch: java.lang.Throwable -> La72
            java.lang.String r7 = "timestamp"
            long r8 = r5.zzd     // Catch: java.lang.Throwable -> La72
            java.lang.Long r8 = java.lang.Long.valueOf(r8)     // Catch: java.lang.Throwable -> La72
            r6.put(r7, r8)     // Catch: java.lang.Throwable -> La72
            java.lang.Long r7 = java.lang.Long.valueOf(r10)     // Catch: java.lang.Throwable -> La72
            r6.put(r15, r7)     // Catch: java.lang.Throwable -> La72
            java.lang.String r7 = "data"
            r6.put(r7, r3)     // Catch: java.lang.Throwable -> La72
            java.lang.String r3 = "realtime"
            java.lang.Integer r7 = java.lang.Integer.valueOf(r13)     // Catch: java.lang.Throwable -> La72
            r6.put(r3, r7)     // Catch: java.lang.Throwable -> La72
            android.database.sqlite.SQLiteDatabase r3 = r2.zzh()     // Catch: android.database.sqlite.SQLiteException -> L9f6 java.lang.Throwable -> La72
            r7 = r32
            long r3 = r3.insert(r7, r4, r6)     // Catch: android.database.sqlite.SQLiteException -> L9f6 java.lang.Throwable -> La72
            r6 = -1
            int r3 = (r3 > r6 ? 1 : (r3 == r6 ? 0 : -1))
            if (r3 != 0) goto L9f1
            com.google.android.gms.measurement.internal.zzfr r3 = r2.zzt     // Catch: android.database.sqlite.SQLiteException -> L9f6 java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzeh r3 = r3.zzay()     // Catch: android.database.sqlite.SQLiteException -> L9f6 java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzef r3 = r3.zzd()     // Catch: android.database.sqlite.SQLiteException -> L9f6 java.lang.Throwable -> La72
            java.lang.String r4 = "Failed to insert raw event (got -1). appId"
            java.lang.String r6 = r5.zza     // Catch: android.database.sqlite.SQLiteException -> L9f6 java.lang.Throwable -> La72
            java.lang.Object r6 = com.google.android.gms.measurement.internal.zzeh.zzn(r6)     // Catch: android.database.sqlite.SQLiteException -> L9f6 java.lang.Throwable -> La72
            r3.zzb(r4, r6)     // Catch: android.database.sqlite.SQLiteException -> L9f6 java.lang.Throwable -> La72
            goto La3f
        L9f1:
            r3 = 0
            r1.zza = r3     // Catch: java.lang.Throwable -> La72
            goto La3f
        L9f6:
            r0 = move-exception
            r3 = r0
            com.google.android.gms.measurement.internal.zzfr r2 = r2.zzt     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzeh r2 = r2.zzay()     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzef r2 = r2.zzd()     // Catch: java.lang.Throwable -> La72
            java.lang.String r4 = "Error storing raw event. appId"
            java.lang.String r5 = r5.zza     // Catch: java.lang.Throwable -> La72
            java.lang.Object r5 = com.google.android.gms.measurement.internal.zzeh.zzn(r5)     // Catch: java.lang.Throwable -> La72
            r2.zzc(r4, r5, r3)     // Catch: java.lang.Throwable -> La72
            goto La3f
        La0e:
            r0 = move-exception
            r4 = r0
            com.google.android.gms.measurement.internal.zzfr r3 = r3.zzt     // Catch: java.io.IOException -> La28 java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzeh r3 = r3.zzay()     // Catch: java.io.IOException -> La28 java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzef r3 = r3.zzd()     // Catch: java.io.IOException -> La28 java.lang.Throwable -> La72
            java.lang.String r5 = "Error storing raw event metadata. appId"
            java.lang.String r6 = r8.zzx()     // Catch: java.io.IOException -> La28 java.lang.Throwable -> La72
            java.lang.Object r6 = com.google.android.gms.measurement.internal.zzeh.zzn(r6)     // Catch: java.io.IOException -> La28 java.lang.Throwable -> La72
            r3.zzc(r5, r6, r4)     // Catch: java.io.IOException -> La28 java.lang.Throwable -> La72
            throw r4     // Catch: java.io.IOException -> La28 java.lang.Throwable -> La72
        La28:
            r0 = move-exception
            r3 = r0
            com.google.android.gms.measurement.internal.zzeh r4 = r34.zzay()     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzef r4 = r4.zzd()     // Catch: java.lang.Throwable -> La72
            java.lang.String r5 = "Data loss. Failed to insert raw event metadata. appId"
            java.lang.String r2 = r2.zzap()     // Catch: java.lang.Throwable -> La72
            java.lang.Object r2 = com.google.android.gms.measurement.internal.zzeh.zzn(r2)     // Catch: java.lang.Throwable -> La72
            r4.zzc(r5, r2, r3)     // Catch: java.lang.Throwable -> La72
        La3f:
            com.google.android.gms.measurement.internal.zzam r2 = r1.zze     // Catch: java.lang.Throwable -> La72
            zzal(r2)     // Catch: java.lang.Throwable -> La72
            r2.zzC()     // Catch: java.lang.Throwable -> La72
            com.google.android.gms.measurement.internal.zzam r2 = r1.zze
            zzal(r2)
            r2.zzx()
            r34.zzag()
            com.google.android.gms.measurement.internal.zzeh r2 = r34.zzay()
            com.google.android.gms.measurement.internal.zzef r2 = r2.zzj()
            long r3 = java.lang.System.nanoTime()
            long r3 = r3 - r28
            r5 = 500000(0x7a120, double:2.47033E-318)
            long r3 = r3 + r5
            r5 = 1000000(0xf4240, double:4.940656E-318)
            long r3 = r3 / r5
            java.lang.Long r3 = java.lang.Long.valueOf(r3)
            java.lang.String r4 = "Background event processing time, ms"
            r2.zzb(r4, r3)
            return
        La72:
            r0 = move-exception
            r2 = r0
            com.google.android.gms.measurement.internal.zzam r3 = r1.zze
            zzal(r3)
            r3.zzx()
            throw r2
        La7d:
            r1.zzd(r3)
            return
        */
        throw new UnsupportedOperationException("Method not decompiled: com.google.android.gms.measurement.internal.zzkt.zzY(com.google.android.gms.measurement.internal.zzaw, com.google.android.gms.measurement.internal.zzq):void");
    }

    @VisibleForTesting
    public final boolean zzZ() {
        zzaz().zzg();
        FileLock fileLock = this.zzw;
        if (fileLock != null && fileLock.isValid()) {
            zzay().zzj().zza("Storage concurrent access okay");
            return true;
        }
        this.zze.zzt.zzf();
        try {
            FileChannel channel = new RandomAccessFile(new File(this.zzn.zzau().getFilesDir(), "google_app_measurement.db"), "rw").getChannel();
            this.zzx = channel;
            FileLock tryLock = channel.tryLock();
            this.zzw = tryLock;
            if (tryLock != null) {
                zzay().zzj().zza("Storage concurrent access okay");
                return true;
            }
            zzay().zzd().zza("Storage concurrent data access panic");
            return false;
        } catch (FileNotFoundException e2) {
            zzay().zzd().zzb("Failed to acquire storage lock", e2);
            return false;
        } catch (IOException e3) {
            zzay().zzd().zzb("Failed to access storage lock file", e3);
            return false;
        } catch (OverlappingFileLockException e4) {
            zzay().zzk().zzb("Storage lock already acquired", e4);
            return false;
        }
    }

    public final long zza() {
        long currentTimeMillis = zzav().currentTimeMillis();
        zzjo zzjoVar = this.zzk;
        zzjoVar.zzW();
        zzjoVar.zzg();
        long zza = zzjoVar.zze.zza();
        if (zza == 0) {
            zza = zzjoVar.zzt.zzv().zzG().nextInt(86400000) + 1;
            zzjoVar.zze.zzb(zza);
        }
        return ((((currentTimeMillis + zza) / 1000) / 60) / 60) / 24;
    }

    @Override // com.google.android.gms.measurement.internal.zzgm
    public final Context zzau() {
        return this.zzn.zzau();
    }

    @Override // com.google.android.gms.measurement.internal.zzgm
    public final Clock zzav() {
        return ((zzfr) Preconditions.checkNotNull(this.zzn)).zzav();
    }

    @Override // com.google.android.gms.measurement.internal.zzgm
    public final zzab zzaw() {
        throw null;
    }

    @Override // com.google.android.gms.measurement.internal.zzgm
    public final zzeh zzay() {
        return ((zzfr) Preconditions.checkNotNull(this.zzn)).zzay();
    }

    @Override // com.google.android.gms.measurement.internal.zzgm
    public final zzfo zzaz() {
        return ((zzfr) Preconditions.checkNotNull(this.zzn)).zzaz();
    }

    public final zzh zzd(zzq zzqVar) {
        zzaz().zzg();
        zzB();
        Preconditions.checkNotNull(zzqVar);
        Preconditions.checkNotEmpty(zzqVar.zza);
        if (!zzqVar.zzw.isEmpty()) {
            this.zzC.put(zzqVar.zza, new zzks(this, zzqVar.zzw));
        }
        zzam zzamVar = this.zze;
        zzal(zzamVar);
        zzh zzj = zzamVar.zzj(zzqVar.zza);
        zzai zzc = zzh(zzqVar.zza).zzc(zzai.zzb(zzqVar.zzv));
        zzah zzahVar = zzah.AD_STORAGE;
        String zzf = zzc.zzi(zzahVar) ? this.zzk.zzf(zzqVar.zza, zzqVar.zzo) : "";
        if (zzj == null) {
            zzj = new zzh(this.zzn, zzqVar.zza);
            if (zzc.zzi(zzah.ANALYTICS_STORAGE)) {
                zzj.zzH(zzw(zzc));
            }
            if (zzc.zzi(zzahVar)) {
                zzj.zzae(zzf);
            }
        } else if (zzc.zzi(zzahVar) && zzf != null && !zzf.equals(zzj.zzA())) {
            zzj.zzae(zzf);
            if (zzqVar.zzo && !"00000000-0000-0000-0000-000000000000".equals(this.zzk.zzd(zzqVar.zza, zzc).first)) {
                zzj.zzH(zzw(zzc));
                zzam zzamVar2 = this.zze;
                zzal(zzamVar2);
                if (zzamVar2.zzp(zzqVar.zza, "_id") != null) {
                    zzam zzamVar3 = this.zze;
                    zzal(zzamVar3);
                    if (zzamVar3.zzp(zzqVar.zza, "_lair") == null) {
                        zzky zzkyVar = new zzky(zzqVar.zza, "auto", "_lair", zzav().currentTimeMillis(), 1L);
                        zzam zzamVar4 = this.zze;
                        zzal(zzamVar4);
                        zzamVar4.zzL(zzkyVar);
                    }
                }
            }
        } else if (TextUtils.isEmpty(zzj.zzu()) && zzc.zzi(zzah.ANALYTICS_STORAGE)) {
            zzj.zzH(zzw(zzc));
        }
        zzj.zzW(zzqVar.zzb);
        zzj.zzF(zzqVar.zzq);
        if (!TextUtils.isEmpty(zzqVar.zzk)) {
            zzj.zzV(zzqVar.zzk);
        }
        long j = zzqVar.zze;
        if (j != 0) {
            zzj.zzX(j);
        }
        if (!TextUtils.isEmpty(zzqVar.zzc)) {
            zzj.zzJ(zzqVar.zzc);
        }
        zzj.zzK(zzqVar.zzj);
        String str = zzqVar.zzd;
        if (str != null) {
            zzj.zzI(str);
        }
        zzj.zzS(zzqVar.zzf);
        zzj.zzac(zzqVar.zzh);
        if (!TextUtils.isEmpty(zzqVar.zzg)) {
            zzj.zzY(zzqVar.zzg);
        }
        zzj.zzG(zzqVar.zzo);
        zzj.zzad(zzqVar.zzr);
        zzj.zzT(zzqVar.zzs);
        zzpd.zzc();
        if (zzg().zzs(null, zzdu.zzal) && zzg().zzs(zzqVar.zza, zzdu.zzan)) {
            zzj.zzag(zzqVar.zzx);
        }
        zznt.zzc();
        if (zzg().zzs(null, zzdu.zzaj)) {
            zzj.zzaf(zzqVar.zzt);
        } else {
            zznt.zzc();
            if (zzg().zzs(null, zzdu.zzai)) {
                zzj.zzaf(null);
            }
        }
        if (zzj.zzaj()) {
            zzam zzamVar5 = this.zze;
            zzal(zzamVar5);
            zzamVar5.zzD(zzj);
        }
        return zzj;
    }

    public final zzaa zzf() {
        zzaa zzaaVar = this.zzh;
        zzal(zzaaVar);
        return zzaaVar;
    }

    public final zzag zzg() {
        return ((zzfr) Preconditions.checkNotNull(this.zzn)).zzf();
    }

    public final zzai zzh(String str) {
        String str2;
        zzai zzaiVar = zzai.zza;
        zzaz().zzg();
        zzB();
        zzai zzaiVar2 = (zzai) this.zzB.get(str);
        if (zzaiVar2 == null) {
            zzam zzamVar = this.zze;
            zzal(zzamVar);
            Preconditions.checkNotNull(str);
            zzamVar.zzg();
            zzamVar.zzW();
            Cursor cursor = null;
            try {
                try {
                    cursor = zzamVar.zzh().rawQuery("select consent_state from consent_settings where app_id=? limit 1;", new String[]{str});
                    if (cursor.moveToFirst()) {
                        str2 = cursor.getString(0);
                        cursor.close();
                    } else {
                        cursor.close();
                        str2 = "G1";
                    }
                    zzai zzb2 = zzai.zzb(str2);
                    zzV(str, zzb2);
                    return zzb2;
                } catch (SQLiteException e2) {
                    zzamVar.zzt.zzay().zzd().zzc("Database error", "select consent_state from consent_settings where app_id=? limit 1;", e2);
                    throw e2;
                }
            } catch (Throwable th) {
                if (cursor != null) {
                    cursor.close();
                }
                throw th;
            }
        }
        return zzaiVar2;
    }

    public final zzam zzi() {
        zzam zzamVar = this.zze;
        zzal(zzamVar);
        return zzamVar;
    }

    public final zzec zzj() {
        return this.zzn.zzj();
    }

    public final zzen zzl() {
        zzen zzenVar = this.zzd;
        zzal(zzenVar);
        return zzenVar;
    }

    public final zzep zzm() {
        zzep zzepVar = this.zzf;
        if (zzepVar != null) {
            return zzepVar;
        }
        throw new IllegalStateException("Network broadcast receiver not created");
    }

    public final zzfi zzo() {
        zzfi zzfiVar = this.zzc;
        zzal(zzfiVar);
        return zzfiVar;
    }

    public final zzfr zzq() {
        return this.zzn;
    }

    public final zzic zzr() {
        zzic zzicVar = this.zzj;
        zzal(zzicVar);
        return zzicVar;
    }

    public final zzjo zzs() {
        return this.zzk;
    }

    public final zzkv zzu() {
        zzkv zzkvVar = this.zzi;
        zzal(zzkvVar);
        return zzkvVar;
    }

    public final zzlb zzv() {
        return ((zzfr) Preconditions.checkNotNull(this.zzn)).zzv();
    }

    public final String zzw(zzai zzaiVar) {
        if (zzaiVar.zzi(zzah.ANALYTICS_STORAGE)) {
            byte[] bArr = new byte[16];
            zzv().zzG().nextBytes(bArr);
            return String.format(Locale.US, "%032x", new BigInteger(1, bArr));
        }
        return null;
    }

    public final String zzx(zzq zzqVar) {
        try {
            return (String) zzaz().zzh(new zzkm(this, zzqVar)).get(30000L, TimeUnit.MILLISECONDS);
        } catch (InterruptedException | ExecutionException | TimeoutException e2) {
            zzay().zzd().zzc("Failed to get app instance id. appId", zzeh.zzn(zzqVar.zza), e2);
            return null;
        }
    }

    public final void zzz(Runnable runnable) {
        zzaz().zzg();
        if (this.zzq == null) {
            this.zzq = new ArrayList();
        }
        this.zzq.add(runnable);
    }
}