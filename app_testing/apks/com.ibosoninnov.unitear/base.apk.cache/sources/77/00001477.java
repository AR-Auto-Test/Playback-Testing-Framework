package com.google.android.gms.measurement.internal;

import android.content.pm.ApplicationInfo;
import android.content.pm.PackageManager;
import android.content.res.Resources;
import android.os.Bundle;
import android.text.TextUtils;
import c.b.a.a.a;
import com.google.android.gms.common.internal.Preconditions;
import com.google.android.gms.common.util.ProcessUtils;
import com.google.android.gms.common.util.VisibleForTesting;
import com.google.android.gms.common.wrappers.Wrappers;
import java.lang.reflect.InvocationTargetException;
import java.util.Arrays;
import java.util.List;
import org.checkerframework.checker.nullness.qual.EnsuresNonNull;

/* compiled from: com.google.android.gms:play-services-measurement-impl@@21.2.0 */
/* loaded from: classes.dex */
public final class zzag extends zzgk {
    private Boolean zza;
    private zzaf zzb;
    private Boolean zzc;

    public zzag(zzfr zzfrVar) {
        super(zzfrVar);
        this.zzb = zzae.zza;
    }

    public static final long zzA() {
        return ((Long) zzdu.zzC.zza(null)).longValue();
    }

    private final String zzB(String str, String str2) {
        try {
            String str3 = (String) Class.forName("android.os.SystemProperties").getMethod("get", String.class, String.class).invoke(null, str, "");
            Preconditions.checkNotNull(str3);
            return str3;
        } catch (ClassNotFoundException e2) {
            this.zzt.zzay().zzd().zzb("Could not find SystemProperties class", e2);
            return "";
        } catch (IllegalAccessException e3) {
            this.zzt.zzay().zzd().zzb("Could not access SystemProperties.get()", e3);
            return "";
        } catch (NoSuchMethodException e4) {
            this.zzt.zzay().zzd().zzb("Could not find SystemProperties.get() method", e4);
            return "";
        } catch (InvocationTargetException e5) {
            this.zzt.zzay().zzd().zzb("SystemProperties.get() threw an exception", e5);
            return "";
        }
    }

    public static final long zzz() {
        return ((Long) zzdu.zzc.zza(null)).longValue();
    }

    public final double zza(String str, zzdt zzdtVar) {
        if (str == null) {
            return ((Double) zzdtVar.zza(null)).doubleValue();
        }
        String zza = this.zzb.zza(str, zzdtVar.zzb());
        if (TextUtils.isEmpty(zza)) {
            return ((Double) zzdtVar.zza(null)).doubleValue();
        }
        try {
            return ((Double) zzdtVar.zza(Double.valueOf(Double.parseDouble(zza)))).doubleValue();
        } catch (NumberFormatException unused) {
            return ((Double) zzdtVar.zza(null)).doubleValue();
        }
    }

    public final int zzb(String str) {
        return zzf(str, zzdu.zzG, 500, 2000);
    }

    public final int zzc() {
        zzlb zzv = this.zzt.zzv();
        Boolean zzj = zzv.zzt.zzt().zzj();
        if (zzv.zzm() < 201500) {
            return (zzj == null || zzj.booleanValue()) ? 25 : 100;
        }
        return 100;
    }

    public final int zzd(String str) {
        return zzf(str, zzdu.zzH, 25, 100);
    }

    public final int zze(String str, zzdt zzdtVar) {
        if (str == null) {
            return ((Integer) zzdtVar.zza(null)).intValue();
        }
        String zza = this.zzb.zza(str, zzdtVar.zzb());
        if (TextUtils.isEmpty(zza)) {
            return ((Integer) zzdtVar.zza(null)).intValue();
        }
        try {
            return ((Integer) zzdtVar.zza(Integer.valueOf(Integer.parseInt(zza)))).intValue();
        } catch (NumberFormatException unused) {
            return ((Integer) zzdtVar.zza(null)).intValue();
        }
    }

    public final int zzf(String str, zzdt zzdtVar, int i, int i2) {
        return Math.max(Math.min(zze(str, zzdtVar), i2), i);
    }

    public final long zzh() {
        this.zzt.zzaw();
        return 74029L;
    }

    public final long zzi(String str, zzdt zzdtVar) {
        if (str == null) {
            return ((Long) zzdtVar.zza(null)).longValue();
        }
        String zza = this.zzb.zza(str, zzdtVar.zzb());
        if (TextUtils.isEmpty(zza)) {
            return ((Long) zzdtVar.zza(null)).longValue();
        }
        try {
            return ((Long) zzdtVar.zza(Long.valueOf(Long.parseLong(zza)))).longValue();
        } catch (NumberFormatException unused) {
            return ((Long) zzdtVar.zza(null)).longValue();
        }
    }

    @VisibleForTesting
    public final Bundle zzj() {
        try {
            if (this.zzt.zzau().getPackageManager() == null) {
                this.zzt.zzay().zzd().zza("Failed to load metadata: PackageManager is null");
                return null;
            }
            ApplicationInfo applicationInfo = Wrappers.packageManager(this.zzt.zzau()).getApplicationInfo(this.zzt.zzau().getPackageName(), 128);
            if (applicationInfo == null) {
                this.zzt.zzay().zzd().zza("Failed to load metadata: ApplicationInfo is null");
                return null;
            }
            return applicationInfo.metaData;
        } catch (PackageManager.NameNotFoundException e2) {
            this.zzt.zzay().zzd().zzb("Failed to load metadata: Package name not found", e2);
            return null;
        }
    }

    @VisibleForTesting
    public final Boolean zzk(String str) {
        Preconditions.checkNotEmpty(str);
        Bundle zzj = zzj();
        if (zzj == null) {
            a.E(this.zzt, "Failed to load metadata: Metadata bundle is null");
            return null;
        } else if (zzj.containsKey(str)) {
            return Boolean.valueOf(zzj.getBoolean(str));
        } else {
            return null;
        }
    }

    public final String zzl() {
        return zzB("debug.firebase.analytics.app", "");
    }

    public final String zzm() {
        return zzB("debug.deferred.deeplink", "");
    }

    public final String zzn() {
        this.zzt.zzaw();
        return "FA";
    }

    public final String zzo(String str, zzdt zzdtVar) {
        if (str == null) {
            return (String) zzdtVar.zza(null);
        }
        return (String) zzdtVar.zza(this.zzb.zza(str, zzdtVar.zzb()));
    }

    /* JADX WARN: Removed duplicated region for block: B:18:0x0026 A[EXC_TOP_SPLITTER, SYNTHETIC] */
    @VisibleForTesting
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public final List zzp(String str) {
        Integer valueOf;
        Preconditions.checkNotEmpty("analytics.safelisted_events");
        Bundle zzj = zzj();
        if (zzj == null) {
            a.E(this.zzt, "Failed to load metadata: Metadata bundle is null");
        } else if (zzj.containsKey("analytics.safelisted_events")) {
            valueOf = Integer.valueOf(zzj.getInt("analytics.safelisted_events"));
            if (valueOf != null) {
                try {
                    String[] stringArray = this.zzt.zzau().getResources().getStringArray(valueOf.intValue());
                    if (stringArray == null) {
                        return null;
                    }
                    return Arrays.asList(stringArray);
                } catch (Resources.NotFoundException e2) {
                    this.zzt.zzay().zzd().zzb("Failed to load string array from metadata: resource not found", e2);
                }
            }
            return null;
        }
        valueOf = null;
        if (valueOf != null) {
        }
        return null;
    }

    public final void zzq(zzaf zzafVar) {
        this.zzb = zzafVar;
    }

    public final boolean zzr() {
        Boolean zzk = zzk("google_analytics_adid_collection_enabled");
        return zzk == null || zzk.booleanValue();
    }

    public final boolean zzs(String str, zzdt zzdtVar) {
        if (str == null) {
            return ((Boolean) zzdtVar.zza(null)).booleanValue();
        }
        String zza = this.zzb.zza(str, zzdtVar.zzb());
        if (TextUtils.isEmpty(zza)) {
            return ((Boolean) zzdtVar.zza(null)).booleanValue();
        }
        return ((Boolean) zzdtVar.zza(Boolean.valueOf("1".equals(zza)))).booleanValue();
    }

    public final boolean zzt(String str) {
        return "1".equals(this.zzb.zza(str, "gaia_collection_enabled"));
    }

    public final boolean zzu() {
        Boolean zzk = zzk("google_analytics_automatic_screen_reporting_enabled");
        return zzk == null || zzk.booleanValue();
    }

    public final boolean zzv() {
        this.zzt.zzaw();
        Boolean zzk = zzk("firebase_analytics_collection_deactivated");
        return zzk != null && zzk.booleanValue();
    }

    public final boolean zzw(String str) {
        return "1".equals(this.zzb.zza(str, "measurement.event_sampling_enabled"));
    }

    public final boolean zzx() {
        if (this.zza == null) {
            Boolean zzk = zzk("app_measurement_lite");
            this.zza = zzk;
            if (zzk == null) {
                this.zza = Boolean.FALSE;
            }
        }
        return this.zza.booleanValue() || !this.zzt.zzN();
    }

    @EnsuresNonNull({"this.isMainProcess"})
    public final boolean zzy() {
        if (this.zzc == null) {
            synchronized (this) {
                if (this.zzc == null) {
                    ApplicationInfo applicationInfo = this.zzt.zzau().getApplicationInfo();
                    String myProcessName = ProcessUtils.getMyProcessName();
                    if (applicationInfo != null) {
                        String str = applicationInfo.processName;
                        boolean z = false;
                        if (str != null && str.equals(myProcessName)) {
                            z = true;
                        }
                        this.zzc = Boolean.valueOf(z);
                    }
                    if (this.zzc == null) {
                        this.zzc = Boolean.TRUE;
                        this.zzt.zzay().zzd().zza("My process not in the list of running processes");
                    }
                }
            }
        }
        return this.zzc.booleanValue();
    }
}