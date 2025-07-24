package com.google.android.gms.measurement.internal;

import android.content.Context;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager;
import android.content.pm.Signature;
import android.content.res.Resources;
import android.text.TextUtils;
import c.b.a.a.a;
import com.google.android.gms.common.internal.Preconditions;
import com.google.android.gms.common.wrappers.InstantApps;
import com.google.android.gms.common.wrappers.Wrappers;
import com.google.android.gms.internal.measurement.zzpd;
import com.google.android.gms.internal.measurement.zzpj;
import java.math.BigInteger;
import java.security.MessageDigest;
import java.util.List;
import java.util.Locale;
import org.checkerframework.checker.nullness.qual.EnsuresNonNull;

/* compiled from: com.google.android.gms:play-services-measurement-impl@@21.2.0 */
/* loaded from: classes.dex */
public final class zzdy extends zzf {
    private String zza;
    private String zzb;
    private int zzc;
    private String zzd;
    private String zze;
    private long zzf;
    private final long zzg;
    private List zzh;
    private String zzi;
    private int zzj;
    private String zzk;
    private String zzl;
    private String zzm;
    private long zzn;
    private String zzo;

    public zzdy(zzfr zzfrVar, long j) {
        super(zzfrVar);
        this.zzn = 0L;
        this.zzo = null;
        this.zzg = j;
    }

    /* JADX WARN: Can't wrap try/catch for region: R(20:1|(1:3)(6:64|65|(1:67)(2:82|(1:84))|68|69|(20:71|(1:73)(1:80)|75|76|5|(1:63)(1:9)|10|11|13|(1:15)|16|17|(1:19)|20|(3:22|(1:24)(1:26)|25)|(3:28|(1:30)(1:33)|31)|34|(3:36|(1:38)(3:45|(3:48|(1:50)|46)|51)|(2:40|41)(2:43|44))|52|(0)(0)))|4|5|(1:7)|63|10|11|13|(0)|16|17|(0)|20|(0)|(0)|34|(0)|52|(0)(0)) */
    /* JADX WARN: Code restructure failed: missing block: B:63:0x01c2, code lost:
        r2 = move-exception;
     */
    /* JADX WARN: Code restructure failed: missing block: B:64:0x01c3, code lost:
        r11.zzt.zzay().zzd().zzc("Fetching Google App Id failed with exception. appId", com.google.android.gms.measurement.internal.zzeh.zzn(r0), r2);
     */
    /* JADX WARN: Removed duplicated region for block: B:28:0x00b6  */
    /* JADX WARN: Removed duplicated region for block: B:34:0x00d0  */
    /* JADX WARN: Removed duplicated region for block: B:35:0x00e0  */
    /* JADX WARN: Removed duplicated region for block: B:36:0x00f0  */
    /* JADX WARN: Removed duplicated region for block: B:37:0x0100  */
    /* JADX WARN: Removed duplicated region for block: B:38:0x0108  */
    /* JADX WARN: Removed duplicated region for block: B:39:0x0118  */
    /* JADX WARN: Removed duplicated region for block: B:40:0x0128  */
    /* JADX WARN: Removed duplicated region for block: B:41:0x0130  */
    /* JADX WARN: Removed duplicated region for block: B:42:0x0140  */
    /* JADX WARN: Removed duplicated region for block: B:45:0x0152  */
    /* JADX WARN: Removed duplicated region for block: B:48:0x0172  */
    /* JADX WARN: Removed duplicated region for block: B:51:0x017b A[Catch: IllegalStateException -> 0x01c2, TryCatch #0 {IllegalStateException -> 0x01c2, blocks: (B:46:0x015a, B:49:0x0173, B:51:0x017b, B:55:0x0199, B:54:0x0195, B:57:0x01a3, B:59:0x01b9, B:61:0x01be, B:60:0x01bc), top: B:83:0x015a }] */
    /* JADX WARN: Removed duplicated region for block: B:57:0x01a3 A[Catch: IllegalStateException -> 0x01c2, TryCatch #0 {IllegalStateException -> 0x01c2, blocks: (B:46:0x015a, B:49:0x0173, B:51:0x017b, B:55:0x0199, B:54:0x0195, B:57:0x01a3, B:59:0x01b9, B:61:0x01be, B:60:0x01bc), top: B:83:0x015a }] */
    /* JADX WARN: Removed duplicated region for block: B:68:0x01ed  */
    /* JADX WARN: Removed duplicated region for block: B:79:0x0226  */
    /* JADX WARN: Removed duplicated region for block: B:81:0x0233  */
    @Override // com.google.android.gms.measurement.internal.zzf
    @EnsuresNonNull({"appId", "appStore", "appName", "gmpAppId", "gaAppId"})
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public final void zzd() {
        String str;
        String str2;
        PackageInfo packageInfo;
        byte b2;
        int zza;
        List<String> zzp;
        String zzc;
        String packageName = this.zzt.zzau().getPackageName();
        PackageManager packageManager = this.zzt.zzau().getPackageManager();
        String str3 = "Unknown";
        int i = Integer.MIN_VALUE;
        String str4 = "unknown";
        if (packageManager == null) {
            this.zzt.zzay().zzd().zzb("PackageManager is null, app identity information might be inaccurate. appId", zzeh.zzn(packageName));
        } else {
            try {
                str4 = packageManager.getInstallerPackageName(packageName);
            } catch (IllegalArgumentException unused) {
                this.zzt.zzay().zzd().zzb("Error retrieving app installer package name. appId", zzeh.zzn(packageName));
            }
            if (str4 == null) {
                str4 = "manual_install";
            } else if ("com.android.vending".equals(str4)) {
                str4 = "";
            }
            try {
                packageInfo = packageManager.getPackageInfo(this.zzt.zzau().getPackageName(), 0);
            } catch (PackageManager.NameNotFoundException unused2) {
                str = "Unknown";
            }
            if (packageInfo != null) {
                CharSequence applicationLabel = packageManager.getApplicationLabel(packageInfo.applicationInfo);
                str2 = !TextUtils.isEmpty(applicationLabel) ? applicationLabel.toString() : "Unknown";
                try {
                    str3 = packageInfo.versionName;
                    i = packageInfo.versionCode;
                } catch (PackageManager.NameNotFoundException unused3) {
                    str = str3;
                    str3 = str2;
                    this.zzt.zzay().zzd().zzc("Error retrieving package info. appId, appName", zzeh.zzn(packageName), str3);
                    str2 = str3;
                    str3 = str;
                    this.zza = packageName;
                    this.zzd = str4;
                    this.zzb = str3;
                    this.zzc = i;
                    this.zze = str2;
                    this.zzf = 0L;
                    if (TextUtils.isEmpty(this.zzt.zzw())) {
                    }
                    zza = this.zzt.zza();
                    switch (zza) {
                    }
                    this.zzk = "";
                    this.zzl = "";
                    this.zzt.zzaw();
                    if (b2 != 0) {
                    }
                    zzc = zzid.zzc(this.zzt.zzau(), "google_app_id", this.zzt.zzz());
                    this.zzk = true != TextUtils.isEmpty(zzc) ? zzc : "";
                    if (!TextUtils.isEmpty(zzc)) {
                    }
                    if (zza == 0) {
                    }
                    this.zzh = null;
                    this.zzt.zzaw();
                    zzp = this.zzt.zzf().zzp("analytics.safelisted_events");
                    if (zzp != null) {
                    }
                    this.zzh = zzp;
                    if (packageManager == null) {
                    }
                }
                this.zza = packageName;
                this.zzd = str4;
                this.zzb = str3;
                this.zzc = i;
                this.zze = str2;
                this.zzf = 0L;
                b2 = (TextUtils.isEmpty(this.zzt.zzw()) && "am".equals(this.zzt.zzx())) ? (byte) 1 : (byte) 0;
                zza = this.zzt.zza();
                switch (zza) {
                    case 0:
                        a.F(this.zzt, "App measurement collection enabled");
                        break;
                    case 1:
                        this.zzt.zzay().zzi().zza("App measurement deactivated via the manifest");
                        break;
                    case 2:
                        a.F(this.zzt, "App measurement deactivated via the init parameters");
                        break;
                    case 3:
                        this.zzt.zzay().zzi().zza("App measurement disabled by setAnalyticsCollectionEnabled(false)");
                        break;
                    case 4:
                        this.zzt.zzay().zzi().zza("App measurement disabled via the manifest");
                        break;
                    case 5:
                        a.F(this.zzt, "App measurement disabled via the init parameters");
                        break;
                    case 6:
                        this.zzt.zzay().zzl().zza("App measurement deactivated via resources. This method is being deprecated. Please refer to https://firebase.google.com/support/guides/disable-analytics");
                        break;
                    case 7:
                        this.zzt.zzay().zzi().zza("App measurement disabled via the global data collection setting");
                        break;
                    default:
                        this.zzt.zzay().zzi().zza("App measurement disabled due to denied storage consent");
                        break;
                }
                this.zzk = "";
                this.zzl = "";
                this.zzt.zzaw();
                if (b2 != 0) {
                    this.zzl = this.zzt.zzw();
                }
                zzc = zzid.zzc(this.zzt.zzau(), "google_app_id", this.zzt.zzz());
                this.zzk = true != TextUtils.isEmpty(zzc) ? zzc : "";
                if (!TextUtils.isEmpty(zzc)) {
                    Context zzau = this.zzt.zzau();
                    String zzz = this.zzt.zzz();
                    Preconditions.checkNotNull(zzau);
                    Resources resources = zzau.getResources();
                    if (TextUtils.isEmpty(zzz)) {
                        zzz = zzfj.zza(zzau);
                    }
                    this.zzl = zzfj.zzb("admob_app_id", resources, zzz);
                }
                if (zza == 0) {
                    this.zzt.zzay().zzj().zzc("App measurement enabled for app package, google app id", this.zza, TextUtils.isEmpty(this.zzk) ? this.zzl : this.zzk);
                }
                this.zzh = null;
                this.zzt.zzaw();
                zzp = this.zzt.zzf().zzp("analytics.safelisted_events");
                if (zzp != null) {
                    if (zzp.isEmpty()) {
                        this.zzt.zzay().zzl().zza("Safelisted event list is empty. Ignoring");
                    } else {
                        for (String str5 : zzp) {
                            if (!this.zzt.zzv().zzab("safelisted event", str5)) {
                            }
                        }
                    }
                    if (packageManager == null) {
                        this.zzj = InstantApps.isInstantApp(this.zzt.zzau()) ? 1 : 0;
                        return;
                    } else {
                        this.zzj = 0;
                        return;
                    }
                }
                this.zzh = zzp;
                if (packageManager == null) {
                }
            }
        }
        str2 = "Unknown";
        this.zza = packageName;
        this.zzd = str4;
        this.zzb = str3;
        this.zzc = i;
        this.zze = str2;
        this.zzf = 0L;
        if (TextUtils.isEmpty(this.zzt.zzw())) {
        }
        zza = this.zzt.zza();
        switch (zza) {
        }
        this.zzk = "";
        this.zzl = "";
        this.zzt.zzaw();
        if (b2 != 0) {
        }
        zzc = zzid.zzc(this.zzt.zzau(), "google_app_id", this.zzt.zzz());
        this.zzk = true != TextUtils.isEmpty(zzc) ? zzc : "";
        if (!TextUtils.isEmpty(zzc)) {
        }
        if (zza == 0) {
        }
        this.zzh = null;
        this.zzt.zzaw();
        zzp = this.zzt.zzf().zzp("analytics.safelisted_events");
        if (zzp != null) {
        }
        this.zzh = zzp;
        if (packageManager == null) {
        }
    }

    @Override // com.google.android.gms.measurement.internal.zzf
    public final boolean zzf() {
        return true;
    }

    public final int zzh() {
        zza();
        return this.zzj;
    }

    public final int zzi() {
        zza();
        return this.zzc;
    }

    /* JADX WARN: Removed duplicated region for block: B:41:0x0176  */
    /* JADX WARN: Removed duplicated region for block: B:42:0x017d  */
    /* JADX WARN: Removed duplicated region for block: B:45:0x01bd  */
    /* JADX WARN: Removed duplicated region for block: B:46:0x01bf  */
    /* JADX WARN: Removed duplicated region for block: B:49:0x01e1  */
    /* JADX WARN: Removed duplicated region for block: B:53:0x0204  */
    /* JADX WARN: Removed duplicated region for block: B:56:0x021c  */
    /* JADX WARN: Removed duplicated region for block: B:70:0x0259  */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public final zzq zzj(String str) {
        String str2;
        long zza;
        boolean z;
        long min;
        Boolean zzk;
        long j;
        List list;
        String str3;
        String str4;
        Class<?> loadClass;
        Object invoke;
        zzg();
        String zzl = zzl();
        String zzm = zzm();
        zza();
        String str5 = this.zzb;
        zza();
        long j2 = this.zzc;
        zza();
        Preconditions.checkNotNull(this.zzd);
        String str6 = this.zzd;
        this.zzt.zzf().zzh();
        zza();
        zzg();
        long j3 = this.zzf;
        long j4 = 0;
        if (j3 == 0) {
            zzlb zzv = this.zzt.zzv();
            Context zzau = this.zzt.zzau();
            String packageName = this.zzt.zzau().getPackageName();
            zzv.zzg();
            Preconditions.checkNotNull(zzau);
            Preconditions.checkNotEmpty(packageName);
            PackageManager packageManager = zzau.getPackageManager();
            MessageDigest zzF = zzlb.zzF();
            if (zzF == null) {
                a.E(zzv.zzt, "Could not get MD5 instance");
            } else {
                if (packageManager != null) {
                    try {
                        if (!zzv.zzag(zzau, packageName)) {
                            Signature[] signatureArr = Wrappers.packageManager(zzau).getPackageInfo(zzv.zzt.zzau().getPackageName(), 64).signatures;
                            if (signatureArr != null && signatureArr.length > 0) {
                                j4 = zzlb.zzp(zzF.digest(signatureArr[0].toByteArray()));
                            } else {
                                zzv.zzt.zzay().zzk().zza("Could not get signatures");
                            }
                        }
                    } catch (PackageManager.NameNotFoundException e2) {
                        zzv.zzt.zzay().zzd().zzb("Package name not found", e2);
                    }
                }
                this.zzf = j4;
            }
            j4 = -1;
            this.zzf = j4;
        } else {
            j4 = j3;
        }
        boolean zzJ = this.zzt.zzJ();
        boolean z2 = !this.zzt.zzm().zzl;
        zzg();
        if (this.zzt.zzJ()) {
            zzpj.zzc();
            if (this.zzt.zzf().zzs(null, zzdu.zzaa)) {
                a.F(this.zzt, "Disabled IID for tests.");
            } else {
                try {
                    loadClass = this.zzt.zzau().getClassLoader().loadClass("com.google.firebase.analytics.FirebaseAnalytics");
                } catch (ClassNotFoundException unused) {
                }
                if (loadClass != null) {
                    try {
                        invoke = loadClass.getDeclaredMethod("getInstance", Context.class).invoke(null, this.zzt.zzau());
                    } catch (Exception unused2) {
                        this.zzt.zzay().zzm().zza("Failed to obtain Firebase Analytics instance");
                    }
                    if (invoke != null) {
                        try {
                            str4 = (String) loadClass.getDeclaredMethod("getFirebaseInstanceId", new Class[0]).invoke(invoke, new Object[0]);
                        } catch (Exception unused3) {
                            this.zzt.zzay().zzl().zza("Failed to retrieve Firebase Instance Id");
                        }
                        str2 = str4;
                        zzfr zzfrVar = this.zzt;
                        zza = zzfrVar.zzm().zzc.zza();
                        if (zza == 0) {
                            min = zzfrVar.zzc;
                            z = zzJ;
                        } else {
                            z = zzJ;
                            min = Math.min(zzfrVar.zzc, zza);
                        }
                        zza();
                        int i = this.zzj;
                        boolean zzr = this.zzt.zzf().zzr();
                        zzew zzm2 = this.zzt.zzm();
                        zzm2.zzg();
                        boolean z3 = zzm2.zza().getBoolean("deferred_analytics_collection", false);
                        zza();
                        String str7 = this.zzl;
                        Boolean valueOf = this.zzt.zzf().zzk("google_analytics_default_allow_ad_personalization_signals") == null ? null : Boolean.valueOf(!zzk.booleanValue());
                        long j5 = this.zzg;
                        List list2 = this.zzh;
                        String zzh = this.zzt.zzm().zzc().zzh();
                        if (this.zzi == null) {
                            j = j5;
                            if (this.zzt.zzf().zzs(null, zzdu.zzap)) {
                                this.zzi = this.zzt.zzv().zzC();
                            } else {
                                this.zzi = "";
                            }
                        } else {
                            j = j5;
                        }
                        String str8 = this.zzi;
                        zzpd.zzc();
                        String str9 = null;
                        if (this.zzt.zzf().zzs(null, zzdu.zzam)) {
                            zzg();
                            if (this.zzn == 0) {
                                list = list2;
                                str3 = str7;
                            } else {
                                list = list2;
                                str3 = str7;
                                long currentTimeMillis = this.zzt.zzav().currentTimeMillis() - this.zzn;
                                if (this.zzm != null && currentTimeMillis > 86400000 && this.zzo == null) {
                                    zzo();
                                }
                            }
                            if (this.zzm == null) {
                                zzo();
                            }
                            str9 = this.zzm;
                        } else {
                            list = list2;
                            str3 = str7;
                        }
                        return new zzq(zzl, zzm, str5, j2, str6, 74029L, j4, str, z, z2, str2, 0L, min, i, zzr, z3, str3, valueOf, j, list, (String) null, zzh, str8, str9);
                    }
                    str4 = null;
                    str2 = str4;
                    zzfr zzfrVar2 = this.zzt;
                    zza = zzfrVar2.zzm().zzc.zza();
                    if (zza == 0) {
                    }
                    zza();
                    int i2 = this.zzj;
                    boolean zzr2 = this.zzt.zzf().zzr();
                    zzew zzm22 = this.zzt.zzm();
                    zzm22.zzg();
                    boolean z32 = zzm22.zza().getBoolean("deferred_analytics_collection", false);
                    zza();
                    String str72 = this.zzl;
                    Boolean valueOf2 = this.zzt.zzf().zzk("google_analytics_default_allow_ad_personalization_signals") == null ? null : Boolean.valueOf(!zzk.booleanValue());
                    long j52 = this.zzg;
                    List list22 = this.zzh;
                    String zzh2 = this.zzt.zzm().zzc().zzh();
                    if (this.zzi == null) {
                    }
                    String str82 = this.zzi;
                    zzpd.zzc();
                    String str92 = null;
                    if (this.zzt.zzf().zzs(null, zzdu.zzam)) {
                    }
                    return new zzq(zzl, zzm, str5, j2, str6, 74029L, j4, str, z, z2, str2, 0L, min, i2, zzr2, z32, str3, valueOf2, j, list, (String) null, zzh2, str82, str92);
                }
            }
        }
        str2 = null;
        zzfr zzfrVar22 = this.zzt;
        zza = zzfrVar22.zzm().zzc.zza();
        if (zza == 0) {
        }
        zza();
        int i22 = this.zzj;
        boolean zzr22 = this.zzt.zzf().zzr();
        zzew zzm222 = this.zzt.zzm();
        zzm222.zzg();
        boolean z322 = zzm222.zza().getBoolean("deferred_analytics_collection", false);
        zza();
        String str722 = this.zzl;
        Boolean valueOf22 = this.zzt.zzf().zzk("google_analytics_default_allow_ad_personalization_signals") == null ? null : Boolean.valueOf(!zzk.booleanValue());
        long j522 = this.zzg;
        List list222 = this.zzh;
        String zzh22 = this.zzt.zzm().zzc().zzh();
        if (this.zzi == null) {
        }
        String str822 = this.zzi;
        zzpd.zzc();
        String str922 = null;
        if (this.zzt.zzf().zzs(null, zzdu.zzam)) {
        }
        return new zzq(zzl, zzm, str5, j2, str6, 74029L, j4, str, z, z2, str2, 0L, min, i22, zzr22, z322, str3, valueOf22, j, list, (String) null, zzh22, str822, str922);
    }

    public final String zzk() {
        zza();
        return this.zzl;
    }

    public final String zzl() {
        zza();
        Preconditions.checkNotNull(this.zza);
        return this.zza;
    }

    public final String zzm() {
        zzg();
        zza();
        Preconditions.checkNotNull(this.zzk);
        return this.zzk;
    }

    public final List zzn() {
        return this.zzh;
    }

    public final void zzo() {
        String format;
        zzg();
        if (!this.zzt.zzm().zzc().zzi(zzah.ANALYTICS_STORAGE)) {
            this.zzt.zzay().zzc().zza("Analytics Storage consent is not granted");
            format = null;
        } else {
            byte[] bArr = new byte[16];
            this.zzt.zzv().zzG().nextBytes(bArr);
            format = String.format(Locale.US, "%032x", new BigInteger(1, bArr));
        }
        zzef zzc = this.zzt.zzay().zzc();
        Object[] objArr = new Object[1];
        objArr[0] = format == null ? "null" : "not null";
        zzc.zza(String.format("Resetting session stitching token to %s", objArr));
        this.zzm = format;
        this.zzn = this.zzt.zzav().currentTimeMillis();
    }

    public final boolean zzp(String str) {
        String str2 = this.zzo;
        boolean z = false;
        if (str2 != null && !str2.equals(str)) {
            z = true;
        }
        this.zzo = str;
        return z;
    }
}