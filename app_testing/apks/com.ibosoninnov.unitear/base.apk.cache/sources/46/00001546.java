package com.google.android.gms.measurement.internal;

import android.app.Application;
import android.content.Context;
import android.content.SharedPreferences;
import android.os.Build;
import android.os.Bundle;
import android.os.Parcelable;
import android.text.TextUtils;
import c.b.a.a.a;
import com.google.android.gms.common.internal.Preconditions;
import com.google.android.gms.common.util.CollectionUtils;
import com.google.android.gms.common.util.Strings;
import com.google.android.gms.common.util.VisibleForTesting;
import com.google.android.gms.internal.measurement.zznw;
import com.google.android.gms.internal.measurement.zzof;
import com.google.android.gms.internal.measurement.zzpd;
import com.google.android.gms.measurement.api.AppMeasurementSdk;
import com.google.firebase.analytics.FirebaseAnalytics;
import com.google.firebase.crashlytics.CrashlyticsAnalyticsListener;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;
import java.util.concurrent.CopyOnWriteArraySet;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;

/* compiled from: com.google.android.gms:play-services-measurement-impl@@21.2.0 */
/* loaded from: classes.dex */
public final class zzhx extends zzf {
    @VisibleForTesting
    public zzhw zza;
    public final zzs zzb;
    @VisibleForTesting
    public boolean zzc;
    private zzgr zzd;
    private final Set zze;
    private boolean zzf;
    private final AtomicReference zzg;
    private final Object zzh;
    private zzai zzi;
    private int zzj;
    private final AtomicLong zzk;
    private long zzl;
    private int zzm;
    private final zzla zzn;

    public zzhx(zzfr zzfrVar) {
        super(zzfrVar);
        this.zze = new CopyOnWriteArraySet();
        this.zzh = new Object();
        this.zzc = true;
        this.zzn = new zzhl(this);
        this.zzg = new AtomicReference();
        this.zzi = new zzai(null, null);
        this.zzj = 100;
        this.zzl = -1L;
        this.zzm = 100;
        this.zzk = new AtomicLong(0L);
        this.zzb = new zzs(zzfrVar);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public final void zzaa(Boolean bool, boolean z) {
        zzg();
        zza();
        this.zzt.zzay().zzc().zzb("Setting app measurement enabled (FE)", bool);
        this.zzt.zzm().zzh(bool);
        if (z) {
            zzew zzm = this.zzt.zzm();
            zzfr zzfrVar = zzm.zzt;
            zzm.zzg();
            SharedPreferences.Editor edit = zzm.zza().edit();
            if (bool != null) {
                edit.putBoolean("measurement_enabled_from_api", bool.booleanValue());
            } else {
                edit.remove("measurement_enabled_from_api");
            }
            edit.apply();
        }
        if (this.zzt.zzK() || !(bool == null || bool.booleanValue())) {
            zzab();
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public final void zzab() {
        zzg();
        String zza = this.zzt.zzm().zzh.zza();
        if (zza != null) {
            if ("unset".equals(zza)) {
                zzY("app", "_npa", null, this.zzt.zzav().currentTimeMillis());
            } else {
                zzY("app", "_npa", Long.valueOf(true != "true".equals(zza) ? 0L : 1L), this.zzt.zzav().currentTimeMillis());
            }
        }
        if (this.zzt.zzJ() && this.zzc) {
            this.zzt.zzay().zzc().zza("Recording app launch after enabling measurement for the first time (FE)");
            zzz();
            zzof.zzc();
            if (this.zzt.zzf().zzs(null, zzdu.zzad)) {
                this.zzt.zzu().zza.zza();
            }
            this.zzt.zzaz().zzp(new zzgz(this));
            return;
        }
        this.zzt.zzay().zzc().zza("Updating Scion state (FE)");
        this.zzt.zzt().zzI();
    }

    public static /* bridge */ /* synthetic */ void zzv(zzhx zzhxVar, zzai zzaiVar, zzai zzaiVar2) {
        boolean z;
        zzah[] zzahVarArr = {zzah.ANALYTICS_STORAGE, zzah.AD_STORAGE};
        int i = 0;
        while (true) {
            if (i >= 2) {
                z = false;
                break;
            }
            zzah zzahVar = zzahVarArr[i];
            if (!zzaiVar2.zzi(zzahVar) && zzaiVar.zzi(zzahVar)) {
                z = true;
                break;
            }
            i++;
        }
        boolean zzl = zzaiVar.zzl(zzaiVar2, zzah.ANALYTICS_STORAGE, zzah.AD_STORAGE);
        if (z || zzl) {
            zzhxVar.zzt.zzh().zzo();
        }
    }

    public static /* synthetic */ void zzw(zzhx zzhxVar, zzai zzaiVar, int i, long j, boolean z, boolean z2) {
        zzhxVar.zzg();
        zzhxVar.zza();
        if (j <= zzhxVar.zzl && zzai.zzj(zzhxVar.zzm, i)) {
            zzhxVar.zzt.zzay().zzi().zzb("Dropped out-of-date consent setting, proposed settings", zzaiVar);
            return;
        }
        zzew zzm = zzhxVar.zzt.zzm();
        zzfr zzfrVar = zzm.zzt;
        zzm.zzg();
        if (zzm.zzl(i)) {
            SharedPreferences.Editor edit = zzm.zza().edit();
            edit.putString("consent_settings", zzaiVar.zzh());
            edit.putInt("consent_source", i);
            edit.apply();
            zzhxVar.zzl = j;
            zzhxVar.zzm = i;
            zzhxVar.zzt.zzt().zzF(z);
            if (z2) {
                zzhxVar.zzt.zzt().zzu(new AtomicReference());
                return;
            }
            return;
        }
        zzhxVar.zzt.zzay().zzi().zzb("Lower precedence consent source ignored, proposed source", Integer.valueOf(i));
    }

    public final void zzA(String str, String str2, Bundle bundle) {
        long currentTimeMillis = this.zzt.zzav().currentTimeMillis();
        Preconditions.checkNotEmpty(str);
        Bundle bundle2 = new Bundle();
        bundle2.putString("name", str);
        bundle2.putLong(AppMeasurementSdk.ConditionalUserProperty.CREATION_TIMESTAMP, currentTimeMillis);
        if (str2 != null) {
            bundle2.putString(AppMeasurementSdk.ConditionalUserProperty.EXPIRED_EVENT_NAME, str2);
            bundle2.putBundle(AppMeasurementSdk.ConditionalUserProperty.EXPIRED_EVENT_PARAMS, bundle);
        }
        this.zzt.zzaz().zzp(new zzhg(this, bundle2));
    }

    public final void zzB() {
        if (!(this.zzt.zzau().getApplicationContext() instanceof Application) || this.zza == null) {
            return;
        }
        ((Application) this.zzt.zzau().getApplicationContext()).unregisterActivityLifecycleCallbacks(this.zza);
    }

    public final /* synthetic */ void zzC(Bundle bundle) {
        if (bundle == null) {
            this.zzt.zzm().zzs.zzb(new Bundle());
            return;
        }
        Bundle zza = this.zzt.zzm().zzs.zza();
        for (String str : bundle.keySet()) {
            Object obj = bundle.get(str);
            if (obj != null && !(obj instanceof String) && !(obj instanceof Long) && !(obj instanceof Double)) {
                if (this.zzt.zzv().zzaf(obj)) {
                    this.zzt.zzv().zzN(this.zzn, null, 27, null, null, 0);
                }
                this.zzt.zzay().zzl().zzc("Invalid default event parameter type. Name, value", str, obj);
            } else if (zzlb.zzah(str)) {
                this.zzt.zzay().zzl().zzb("Invalid default event parameter name. Name", str);
            } else if (obj == null) {
                zza.remove(str);
            } else {
                zzlb zzv = this.zzt.zzv();
                this.zzt.zzf();
                if (zzv.zzaa("param", str, 100, obj)) {
                    this.zzt.zzv().zzO(zza, str, obj);
                }
            }
        }
        this.zzt.zzv();
        int zzc = this.zzt.zzf().zzc();
        if (zza.size() > zzc) {
            Iterator it = new TreeSet(zza.keySet()).iterator();
            int i = 0;
            while (it.hasNext()) {
                String str2 = (String) it.next();
                i++;
                if (i > zzc) {
                    zza.remove(str2);
                }
            }
            this.zzt.zzv().zzN(this.zzn, null, 26, null, null, 0);
            this.zzt.zzay().zzl().zza("Too many default event parameters set. Discarding beyond event parameter limit");
        }
        this.zzt.zzm().zzs.zzb(zza);
        this.zzt.zzt().zzH(zza);
    }

    public final void zzD(String str, String str2, Bundle bundle) {
        zzE(str, str2, bundle, true, true, this.zzt.zzav().currentTimeMillis());
    }

    public final void zzE(String str, String str2, Bundle bundle, boolean z, boolean z2, long j) {
        String str3 = str == null ? "app" : str;
        Bundle bundle2 = bundle == null ? new Bundle() : bundle;
        if (str2 != FirebaseAnalytics.Event.SCREEN_VIEW && (str2 == null || !str2.equals(FirebaseAnalytics.Event.SCREEN_VIEW))) {
            boolean z3 = true;
            if (z2 && this.zzd != null && !zzlb.zzah(str2)) {
                z3 = false;
            }
            zzM(str3, str2, j, bundle2, z2, z3, z, null);
            return;
        }
        this.zzt.zzs().zzx(bundle2, j);
    }

    public final void zzF(String str, String str2, Bundle bundle, String str3) {
        zzfr.zzO();
        zzM("auto", str2, this.zzt.zzav().currentTimeMillis(), bundle, false, true, true, str3);
    }

    public final void zzG(String str, String str2, Bundle bundle) {
        zzg();
        zzH(str, str2, this.zzt.zzav().currentTimeMillis(), bundle);
    }

    public final void zzH(String str, String str2, long j, Bundle bundle) {
        zzg();
        zzI(str, str2, j, bundle, true, this.zzd == null || zzlb.zzah(str2), true, null);
    }

    public final void zzI(String str, String str2, long j, Bundle bundle, boolean z, boolean z2, boolean z3, String str3) {
        boolean z4;
        String str4;
        long j2;
        Bundle[] bundleArr;
        Class<?> cls;
        Preconditions.checkNotEmpty(str);
        Preconditions.checkNotNull(bundle);
        zzg();
        zza();
        if (this.zzt.zzJ()) {
            List zzn = this.zzt.zzh().zzn();
            if (zzn != null && !zzn.contains(str2)) {
                this.zzt.zzay().zzc().zzc("Dropping non-safelisted event. event name, origin", str2, str);
                return;
            }
            String str5 = null;
            if (!this.zzf) {
                this.zzf = true;
                try {
                    if (!this.zzt.zzN()) {
                        cls = Class.forName("com.google.android.gms.tagmanager.TagManagerService", true, this.zzt.zzau().getClassLoader());
                    } else {
                        cls = Class.forName("com.google.android.gms.tagmanager.TagManagerService");
                    }
                    try {
                        cls.getDeclaredMethod("initialize", Context.class).invoke(null, this.zzt.zzau());
                    } catch (Exception e2) {
                        this.zzt.zzay().zzk().zzb("Failed to invoke Tag Manager's initialize() method", e2);
                    }
                } catch (ClassNotFoundException unused) {
                    this.zzt.zzay().zzi().zza("Tag Manager is not found and thus will not be used");
                }
            }
            if ("_cmp".equals(str2) && bundle.containsKey("gclid")) {
                this.zzt.zzaw();
                zzY("auto", "_lgclid", bundle.getString("gclid"), this.zzt.zzav().currentTimeMillis());
            }
            this.zzt.zzaw();
            if (z && zzlb.zzal(str2)) {
                this.zzt.zzv().zzL(bundle, this.zzt.zzm().zzs.zza());
            }
            if (!z3) {
                this.zzt.zzaw();
                if (!"_iap".equals(str2)) {
                    zzlb zzv = this.zzt.zzv();
                    int i = 2;
                    if (zzv.zzac("event", str2)) {
                        if (zzv.zzZ("event", zzgo.zza, zzgo.zzb, str2)) {
                            zzv.zzt.zzf();
                            if (zzv.zzY("event", 40, str2)) {
                                i = 0;
                            }
                        } else {
                            i = 13;
                        }
                    }
                    if (i != 0) {
                        this.zzt.zzay().zze().zzb("Invalid public event name. Event will not be logged (FE)", this.zzt.zzj().zzd(str2));
                        zzlb zzv2 = this.zzt.zzv();
                        this.zzt.zzf();
                        this.zzt.zzv().zzN(this.zzn, null, i, "_ev", zzv2.zzD(str2, 40, true), str2 != null ? str2.length() : 0);
                        return;
                    }
                }
            }
            this.zzt.zzaw();
            zzie zzj = this.zzt.zzs().zzj(false);
            if (zzj != null && !bundle.containsKey("_sc")) {
                zzj.zzd = true;
            }
            zzlb.zzK(zzj, bundle, z && !z3);
            boolean equals = "am".equals(str);
            boolean zzah = zzlb.zzah(str2);
            if (!z || this.zzd == null || zzah) {
                z4 = equals;
            } else if (!equals) {
                this.zzt.zzay().zzc().zzc("Passing event to registered event handler (FE)", this.zzt.zzj().zzd(str2), this.zzt.zzj().zzb(bundle));
                Preconditions.checkNotNull(this.zzd);
                this.zzd.interceptEvent(str, str2, bundle, j);
                return;
            } else {
                z4 = true;
            }
            if (this.zzt.zzM()) {
                int zzh = this.zzt.zzv().zzh(str2);
                if (zzh != 0) {
                    this.zzt.zzay().zze().zzb("Invalid event name. Event will not be logged (FE)", this.zzt.zzj().zzd(str2));
                    zzlb zzv3 = this.zzt.zzv();
                    this.zzt.zzf();
                    this.zzt.zzv().zzN(this.zzn, str3, zzh, "_ev", zzv3.zzD(str2, 40, true), str2 != null ? str2.length() : 0);
                    return;
                }
                Bundle zzy = this.zzt.zzv().zzy(str3, str2, bundle, CollectionUtils.listOf((Object[]) new String[]{CrashlyticsAnalyticsListener.EVENT_ORIGIN_KEY, "_sn", "_sc", "_si"}), z3);
                Preconditions.checkNotNull(zzy);
                this.zzt.zzaw();
                if (this.zzt.zzs().zzj(false) != null && "_ae".equals(str2)) {
                    zzka zzkaVar = this.zzt.zzu().zzb;
                    long elapsedRealtime = zzkaVar.zzc.zzt.zzav().elapsedRealtime();
                    long j3 = elapsedRealtime - zzkaVar.zzb;
                    zzkaVar.zzb = elapsedRealtime;
                    if (j3 > 0) {
                        this.zzt.zzv().zzI(zzy, j3);
                    }
                }
                zznw.zzc();
                if (this.zzt.zzf().zzs(null, zzdu.zzac)) {
                    if (!"auto".equals(str) && "_ssr".equals(str2)) {
                        zzlb zzv4 = this.zzt.zzv();
                        String string = zzy.getString("_ffr");
                        if (!Strings.isEmptyOrWhitespace(string)) {
                            str5 = string != null ? string.trim() : string;
                        }
                        if (!zzkz.zza(str5, zzv4.zzt.zzm().zzp.zza())) {
                            zzv4.zzt.zzm().zzp.zzb(str5);
                        } else {
                            zzv4.zzt.zzay().zzc().zza("Not logging duplicate session_start_with_rollout event");
                            return;
                        }
                    } else if ("_ae".equals(str2)) {
                        String zza = this.zzt.zzv().zzt.zzm().zzp.zza();
                        if (!TextUtils.isEmpty(zza)) {
                            zzy.putString("_ffr", zza);
                        }
                    }
                }
                ArrayList arrayList = new ArrayList();
                arrayList.add(zzy);
                if (this.zzt.zzm().zzj.zza() > 0 && this.zzt.zzm().zzk(j) && this.zzt.zzm().zzm.zzb()) {
                    a.F(this.zzt, "Current session is expired, remove the session number, ID, and engagement time");
                    str4 = "_ae";
                    j2 = 0;
                    zzY("auto", "_sid", null, this.zzt.zzav().currentTimeMillis());
                    zzY("auto", "_sno", null, this.zzt.zzav().currentTimeMillis());
                    zzY("auto", "_se", null, this.zzt.zzav().currentTimeMillis());
                    this.zzt.zzm().zzk.zzb(0L);
                } else {
                    str4 = "_ae";
                    j2 = 0;
                }
                if (zzy.getLong(FirebaseAnalytics.Param.EXTEND_SESSION, j2) == 1) {
                    a.F(this.zzt, "EXTEND_SESSION param attached: initiate a new session or extend the current active session");
                    this.zzt.zzu().zza.zzb(j, true);
                }
                ArrayList arrayList2 = new ArrayList(zzy.keySet());
                Collections.sort(arrayList2);
                int size = arrayList2.size();
                for (int i2 = 0; i2 < size; i2++) {
                    String str6 = (String) arrayList2.get(i2);
                    if (str6 != null) {
                        this.zzt.zzv();
                        Object obj = zzy.get(str6);
                        if (obj instanceof Bundle) {
                            bundleArr = new Bundle[]{(Bundle) obj};
                        } else if (obj instanceof Parcelable[]) {
                            Parcelable[] parcelableArr = (Parcelable[]) obj;
                            bundleArr = (Bundle[]) Arrays.copyOf(parcelableArr, parcelableArr.length, Bundle[].class);
                        } else if (obj instanceof ArrayList) {
                            ArrayList arrayList3 = (ArrayList) obj;
                            bundleArr = (Bundle[]) arrayList3.toArray(new Bundle[arrayList3.size()]);
                        } else {
                            bundleArr = null;
                        }
                        if (bundleArr != null) {
                            zzy.putParcelableArray(str6, bundleArr);
                        }
                    }
                }
                int i3 = 0;
                while (i3 < arrayList.size()) {
                    Bundle bundle2 = (Bundle) arrayList.get(i3);
                    String str7 = i3 != 0 ? "_ep" : str2;
                    bundle2.putString(CrashlyticsAnalyticsListener.EVENT_ORIGIN_KEY, str);
                    if (z2) {
                        bundle2 = this.zzt.zzv().zzt(bundle2);
                    }
                    Bundle bundle3 = bundle2;
                    this.zzt.zzt().zzA(new zzaw(str7, new zzau(bundle3), str, j), str3);
                    if (!z4) {
                        for (zzgs zzgsVar : this.zze) {
                            zzgsVar.onEvent(str, str2, new Bundle(bundle3), j);
                        }
                    }
                    i3++;
                }
                this.zzt.zzaw();
                if (this.zzt.zzs().zzj(false) == null || !str4.equals(str2)) {
                    return;
                }
                this.zzt.zzu().zzb.zzd(true, true, this.zzt.zzav().elapsedRealtime());
                return;
            }
            return;
        }
        this.zzt.zzay().zzc().zza("Event not sent since app measurement is disabled");
    }

    public final void zzJ(zzgs zzgsVar) {
        zza();
        Preconditions.checkNotNull(zzgsVar);
        if (this.zze.add(zzgsVar)) {
            return;
        }
        a.G(this.zzt, "OnEventListener already registered");
    }

    public final void zzK(long j) {
        this.zzg.set(null);
        this.zzt.zzaz().zzp(new zzhe(this, j));
    }

    public final void zzL(long j, boolean z) {
        zzg();
        zza();
        this.zzt.zzay().zzc().zza("Resetting analytics data (FE)");
        zzkc zzu = this.zzt.zzu();
        zzu.zzg();
        zzu.zzb.zza();
        zzpd.zzc();
        if (this.zzt.zzf().zzs(null, zzdu.zzam)) {
            this.zzt.zzh().zzo();
        }
        boolean zzJ = this.zzt.zzJ();
        zzew zzm = this.zzt.zzm();
        zzm.zzc.zzb(j);
        if (!TextUtils.isEmpty(zzm.zzt.zzm().zzp.zza())) {
            zzm.zzp.zzb(null);
        }
        zzof.zzc();
        zzag zzf = zzm.zzt.zzf();
        zzdt zzdtVar = zzdu.zzad;
        if (zzf.zzs(null, zzdtVar)) {
            zzm.zzj.zzb(0L);
        }
        zzm.zzk.zzb(0L);
        if (!zzm.zzt.zzf().zzv()) {
            zzm.zzi(!zzJ);
        }
        zzm.zzq.zzb(null);
        zzm.zzr.zzb(0L);
        zzm.zzs.zzb(null);
        if (z) {
            this.zzt.zzt().zzC();
        }
        zzof.zzc();
        if (this.zzt.zzf().zzs(null, zzdtVar)) {
            this.zzt.zzu().zza.zza();
        }
        this.zzc = !zzJ;
    }

    public final void zzM(String str, String str2, long j, Bundle bundle, boolean z, boolean z2, boolean z3, String str3) {
        Bundle bundle2 = new Bundle(bundle);
        for (String str4 : bundle2.keySet()) {
            Object obj = bundle2.get(str4);
            if (obj instanceof Bundle) {
                bundle2.putBundle(str4, new Bundle((Bundle) obj));
            } else {
                int i = 0;
                if (obj instanceof Parcelable[]) {
                    Parcelable[] parcelableArr = (Parcelable[]) obj;
                    while (i < parcelableArr.length) {
                        Parcelable parcelable = parcelableArr[i];
                        if (parcelable instanceof Bundle) {
                            parcelableArr[i] = new Bundle((Bundle) parcelable);
                        }
                        i++;
                    }
                } else if (obj instanceof List) {
                    List list = (List) obj;
                    while (i < list.size()) {
                        Object obj2 = list.get(i);
                        if (obj2 instanceof Bundle) {
                            list.set(i, new Bundle((Bundle) obj2));
                        }
                        i++;
                    }
                }
            }
        }
        this.zzt.zzaz().zzp(new zzhb(this, str, str2, j, bundle2, z, z2, z3, str3));
    }

    public final void zzN(String str, String str2, long j, Object obj) {
        this.zzt.zzaz().zzp(new zzhc(this, str, str2, obj, j));
    }

    public final void zzO(String str) {
        this.zzg.set(str);
    }

    public final void zzP(Bundle bundle) {
        zzQ(bundle, this.zzt.zzav().currentTimeMillis());
    }

    public final void zzQ(Bundle bundle, long j) {
        Preconditions.checkNotNull(bundle);
        Bundle bundle2 = new Bundle(bundle);
        if (!TextUtils.isEmpty(bundle2.getString("app_id"))) {
            a.G(this.zzt, "Package name should be null when calling setConditionalUserProperty");
        }
        bundle2.remove("app_id");
        Preconditions.checkNotNull(bundle2);
        zzgn.zza(bundle2, "app_id", String.class, null);
        zzgn.zza(bundle2, "origin", String.class, null);
        zzgn.zza(bundle2, "name", String.class, null);
        zzgn.zza(bundle2, "value", Object.class, null);
        zzgn.zza(bundle2, AppMeasurementSdk.ConditionalUserProperty.TRIGGER_EVENT_NAME, String.class, null);
        zzgn.zza(bundle2, AppMeasurementSdk.ConditionalUserProperty.TRIGGER_TIMEOUT, Long.class, 0L);
        zzgn.zza(bundle2, AppMeasurementSdk.ConditionalUserProperty.TIMED_OUT_EVENT_NAME, String.class, null);
        zzgn.zza(bundle2, AppMeasurementSdk.ConditionalUserProperty.TIMED_OUT_EVENT_PARAMS, Bundle.class, null);
        zzgn.zza(bundle2, AppMeasurementSdk.ConditionalUserProperty.TRIGGERED_EVENT_NAME, String.class, null);
        zzgn.zza(bundle2, AppMeasurementSdk.ConditionalUserProperty.TRIGGERED_EVENT_PARAMS, Bundle.class, null);
        zzgn.zza(bundle2, AppMeasurementSdk.ConditionalUserProperty.TIME_TO_LIVE, Long.class, 0L);
        zzgn.zza(bundle2, AppMeasurementSdk.ConditionalUserProperty.EXPIRED_EVENT_NAME, String.class, null);
        zzgn.zza(bundle2, AppMeasurementSdk.ConditionalUserProperty.EXPIRED_EVENT_PARAMS, Bundle.class, null);
        Preconditions.checkNotEmpty(bundle2.getString("name"));
        Preconditions.checkNotEmpty(bundle2.getString("origin"));
        Preconditions.checkNotNull(bundle2.get("value"));
        bundle2.putLong(AppMeasurementSdk.ConditionalUserProperty.CREATION_TIMESTAMP, j);
        String string = bundle2.getString("name");
        Object obj = bundle2.get("value");
        if (this.zzt.zzv().zzl(string) == 0) {
            if (this.zzt.zzv().zzd(string, obj) == 0) {
                Object zzB = this.zzt.zzv().zzB(string, obj);
                if (zzB == null) {
                    this.zzt.zzay().zzd().zzc("Unable to normalize conditional user property value", this.zzt.zzj().zzf(string), obj);
                    return;
                }
                zzgn.zzb(bundle2, zzB);
                long j2 = bundle2.getLong(AppMeasurementSdk.ConditionalUserProperty.TRIGGER_TIMEOUT);
                if (!TextUtils.isEmpty(bundle2.getString(AppMeasurementSdk.ConditionalUserProperty.TRIGGER_EVENT_NAME))) {
                    this.zzt.zzf();
                    if (j2 > 15552000000L || j2 < 1) {
                        this.zzt.zzay().zzd().zzc("Invalid conditional user property timeout", this.zzt.zzj().zzf(string), Long.valueOf(j2));
                        return;
                    }
                }
                long j3 = bundle2.getLong(AppMeasurementSdk.ConditionalUserProperty.TIME_TO_LIVE);
                this.zzt.zzf();
                if (j3 <= 15552000000L && j3 >= 1) {
                    this.zzt.zzaz().zzp(new zzhf(this, bundle2));
                    return;
                } else {
                    this.zzt.zzay().zzd().zzc("Invalid conditional user property time to live", this.zzt.zzj().zzf(string), Long.valueOf(j3));
                    return;
                }
            }
            this.zzt.zzay().zzd().zzc("Invalid conditional user property value", this.zzt.zzj().zzf(string), obj);
            return;
        }
        this.zzt.zzay().zzd().zzb("Invalid conditional user property name", this.zzt.zzj().zzf(string));
    }

    public final void zzR(Bundle bundle, int i, long j) {
        zza();
        String zzg = zzai.zzg(bundle);
        if (zzg != null) {
            this.zzt.zzay().zzl().zzb("Ignoring invalid consent setting", zzg);
            this.zzt.zzay().zzl().zza("Valid consent values are 'granted', 'denied'");
        }
        zzS(zzai.zza(bundle), i, j);
    }

    public final void zzS(zzai zzaiVar, int i, long j) {
        zzai zzaiVar2;
        boolean z;
        boolean z2;
        boolean z3;
        zzai zzaiVar3 = zzaiVar;
        zza();
        if (i != -10 && zzaiVar.zze() == null && zzaiVar.zzf() == null) {
            this.zzt.zzay().zzl().zza("Discarding empty consent settings");
            return;
        }
        synchronized (this.zzh) {
            zzaiVar2 = this.zzi;
            z = true;
            z2 = false;
            if (zzai.zzj(i, this.zzj)) {
                boolean zzk = zzaiVar3.zzk(this.zzi);
                zzah zzahVar = zzah.ANALYTICS_STORAGE;
                if (zzaiVar3.zzi(zzahVar) && !this.zzi.zzi(zzahVar)) {
                    z2 = true;
                }
                zzaiVar3 = zzaiVar3.zzd(this.zzi);
                this.zzi = zzaiVar3;
                this.zzj = i;
                z3 = z2;
                z2 = zzk;
            } else {
                z = false;
                z3 = false;
            }
        }
        if (!z) {
            this.zzt.zzay().zzi().zzb("Ignoring lower-priority consent settings, proposed settings", zzaiVar3);
            return;
        }
        long andIncrement = this.zzk.getAndIncrement();
        if (z2) {
            this.zzg.set(null);
            this.zzt.zzaz().zzq(new zzhr(this, zzaiVar3, j, i, andIncrement, z3, zzaiVar2));
            return;
        }
        zzhs zzhsVar = new zzhs(this, zzaiVar3, i, andIncrement, z3, zzaiVar2);
        if (i != 30 && i != -10) {
            this.zzt.zzaz().zzp(zzhsVar);
        } else {
            this.zzt.zzaz().zzq(zzhsVar);
        }
    }

    public final void zzT(zzgr zzgrVar) {
        zzgr zzgrVar2;
        zzg();
        zza();
        if (zzgrVar != null && zzgrVar != (zzgrVar2 = this.zzd)) {
            Preconditions.checkState(zzgrVar2 == null, "EventInterceptor already set.");
        }
        this.zzd = zzgrVar;
    }

    public final void zzU(Boolean bool) {
        zza();
        this.zzt.zzaz().zzp(new zzhq(this, bool));
    }

    public final void zzV(zzai zzaiVar) {
        zzg();
        boolean z = (zzaiVar.zzi(zzah.ANALYTICS_STORAGE) && zzaiVar.zzi(zzah.AD_STORAGE)) || this.zzt.zzt().zzM();
        if (z != this.zzt.zzK()) {
            this.zzt.zzG(z);
            zzew zzm = this.zzt.zzm();
            zzfr zzfrVar = zzm.zzt;
            zzm.zzg();
            Boolean valueOf = zzm.zza().contains("measurement_enabled_from_api") ? Boolean.valueOf(zzm.zza().getBoolean("measurement_enabled_from_api", true)) : null;
            if (!z || valueOf == null || valueOf.booleanValue()) {
                zzaa(Boolean.valueOf(z), false);
            }
        }
    }

    public final void zzW(String str, String str2, Object obj, boolean z) {
        zzX("auto", "_ldl", obj, true, this.zzt.zzav().currentTimeMillis());
    }

    /* JADX WARN: Removed duplicated region for block: B:22:0x0049  */
    /* JADX WARN: Removed duplicated region for block: B:27:0x006e  */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public final void zzX(String str, String str2, Object obj, boolean z, long j) {
        int i;
        String str3 = str == null ? "app" : str;
        int i2 = 6;
        if (z) {
            i2 = this.zzt.zzv().zzl(str2);
        } else {
            zzlb zzv = this.zzt.zzv();
            if (zzv.zzac("user property", str2)) {
                if (zzv.zzZ("user property", zzgq.zza, null, str2)) {
                    zzv.zzt.zzf();
                    if (zzv.zzY("user property", 24, str2)) {
                        i = 0;
                        if (i == 0) {
                            zzlb zzv2 = this.zzt.zzv();
                            this.zzt.zzf();
                            this.zzt.zzv().zzN(this.zzn, null, i, "_ev", zzv2.zzD(str2, 24, true), str2 != null ? str2.length() : 0);
                            return;
                        } else if (obj != null) {
                            int zzd = this.zzt.zzv().zzd(str2, obj);
                            if (zzd != 0) {
                                zzlb zzv3 = this.zzt.zzv();
                                this.zzt.zzf();
                                String zzD = zzv3.zzD(str2, 24, true);
                                if ((obj instanceof String) || (obj instanceof CharSequence)) {
                                    r4 = obj.toString().length();
                                }
                                this.zzt.zzv().zzN(this.zzn, null, zzd, "_ev", zzD, r4);
                                return;
                            }
                            Object zzB = this.zzt.zzv().zzB(str2, obj);
                            if (zzB != null) {
                                zzN(str3, str2, j, zzB);
                                return;
                            }
                            return;
                        } else {
                            zzN(str3, str2, j, null);
                            return;
                        }
                    }
                } else {
                    i2 = 15;
                }
            }
        }
        i = i2;
        if (i == 0) {
        }
    }

    /* JADX WARN: Removed duplicated region for block: B:22:0x006f  */
    /* JADX WARN: Removed duplicated region for block: B:24:0x0077  */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public final void zzY(String str, String str2, Object obj, long j) {
        String str3;
        Object obj2;
        Preconditions.checkNotEmpty(str);
        Preconditions.checkNotEmpty(str2);
        zzg();
        zza();
        if (FirebaseAnalytics.UserProperty.ALLOW_AD_PERSONALIZATION_SIGNALS.equals(str2)) {
            if (obj instanceof String) {
                String str4 = (String) obj;
                if (!TextUtils.isEmpty(str4)) {
                    Long valueOf = Long.valueOf(true != "false".equals(str4.toLowerCase(Locale.ENGLISH)) ? 0L : 1L);
                    this.zzt.zzm().zzh.zzb(valueOf.longValue() == 1 ? "true" : "false");
                    obj = valueOf;
                    obj2 = obj;
                    str3 = "_npa";
                    if (!this.zzt.zzJ()) {
                        a.F(this.zzt, "User property not set since app measurement is disabled");
                        return;
                    } else if (this.zzt.zzM()) {
                        this.zzt.zzt().zzK(new zzkw(str3, j, obj2, str));
                        return;
                    } else {
                        return;
                    }
                }
            }
            if (obj == null) {
                this.zzt.zzm().zzh.zzb("unset");
                obj2 = obj;
                str3 = "_npa";
                if (!this.zzt.zzJ()) {
                }
            }
        }
        str3 = str2;
        obj2 = obj;
        if (!this.zzt.zzJ()) {
        }
    }

    public final void zzZ(zzgs zzgsVar) {
        zza();
        Preconditions.checkNotNull(zzgsVar);
        if (this.zze.remove(zzgsVar)) {
            return;
        }
        a.G(this.zzt, "OnEventListener had not been registered");
    }

    @Override // com.google.android.gms.measurement.internal.zzf
    public final boolean zzf() {
        return false;
    }

    public final int zzh(String str) {
        Preconditions.checkNotEmpty(str);
        this.zzt.zzf();
        return 25;
    }

    public final Boolean zzi() {
        AtomicReference atomicReference = new AtomicReference();
        return (Boolean) this.zzt.zzaz().zzd(atomicReference, 15000L, "boolean test flag value", new zzhi(this, atomicReference));
    }

    public final Double zzj() {
        AtomicReference atomicReference = new AtomicReference();
        return (Double) this.zzt.zzaz().zzd(atomicReference, 15000L, "double test flag value", new zzhp(this, atomicReference));
    }

    public final Integer zzl() {
        AtomicReference atomicReference = new AtomicReference();
        return (Integer) this.zzt.zzaz().zzd(atomicReference, 15000L, "int test flag value", new zzho(this, atomicReference));
    }

    public final Long zzm() {
        AtomicReference atomicReference = new AtomicReference();
        return (Long) this.zzt.zzaz().zzd(atomicReference, 15000L, "long test flag value", new zzhn(this, atomicReference));
    }

    public final String zzo() {
        return (String) this.zzg.get();
    }

    public final String zzp() {
        zzie zzi = this.zzt.zzs().zzi();
        if (zzi != null) {
            return zzi.zzb;
        }
        return null;
    }

    public final String zzq() {
        zzie zzi = this.zzt.zzs().zzi();
        if (zzi != null) {
            return zzi.zza;
        }
        return null;
    }

    public final String zzr() {
        AtomicReference atomicReference = new AtomicReference();
        return (String) this.zzt.zzaz().zzd(atomicReference, 15000L, "String test flag value", new zzhm(this, atomicReference));
    }

    public final ArrayList zzs(String str, String str2) {
        if (this.zzt.zzaz().zzs()) {
            this.zzt.zzay().zzd().zza("Cannot get conditional user properties from analytics worker thread");
            return new ArrayList(0);
        }
        this.zzt.zzaw();
        if (zzab.zza()) {
            this.zzt.zzay().zzd().zza("Cannot get conditional user properties from main thread");
            return new ArrayList(0);
        }
        AtomicReference atomicReference = new AtomicReference();
        this.zzt.zzaz().zzd(atomicReference, 5000L, "get conditional user properties", new zzhh(this, atomicReference, null, str, str2));
        List list = (List) atomicReference.get();
        if (list == null) {
            this.zzt.zzay().zzd().zzb("Timed out waiting for get conditional user properties", null);
            return new ArrayList();
        }
        return zzlb.zzH(list);
    }

    public final List zzt(boolean z) {
        zza();
        a.F(this.zzt, "Getting user properties (FE)");
        if (!this.zzt.zzaz().zzs()) {
            this.zzt.zzaw();
            if (zzab.zza()) {
                this.zzt.zzay().zzd().zza("Cannot get all user properties from main thread");
                return Collections.emptyList();
            }
            AtomicReference atomicReference = new AtomicReference();
            this.zzt.zzaz().zzd(atomicReference, 5000L, "get user properties", new zzhd(this, atomicReference, z));
            List list = (List) atomicReference.get();
            if (list == null) {
                this.zzt.zzay().zzd().zzb("Timed out waiting for get user properties, includeInternal", Boolean.valueOf(z));
                return Collections.emptyList();
            }
            return list;
        }
        this.zzt.zzay().zzd().zza("Cannot get all user properties from analytics worker thread");
        return Collections.emptyList();
    }

    public final Map zzu(String str, String str2, boolean z) {
        if (this.zzt.zzaz().zzs()) {
            this.zzt.zzay().zzd().zza("Cannot get user properties from analytics worker thread");
            return Collections.emptyMap();
        }
        this.zzt.zzaw();
        if (zzab.zza()) {
            this.zzt.zzay().zzd().zza("Cannot get user properties from main thread");
            return Collections.emptyMap();
        }
        AtomicReference atomicReference = new AtomicReference();
        this.zzt.zzaz().zzd(atomicReference, 5000L, "get user properties", new zzhj(this, atomicReference, null, str, str2, z));
        List<zzkw> list = (List) atomicReference.get();
        if (list == null) {
            this.zzt.zzay().zzd().zzb("Timed out waiting for handle get user properties, includeInternal", Boolean.valueOf(z));
            return Collections.emptyMap();
        }
        b.f.a aVar = new b.f.a(list.size());
        for (zzkw zzkwVar : list) {
            Object zza = zzkwVar.zza();
            if (zza != null) {
                aVar.put(zzkwVar.zzb, zza);
            }
        }
        return aVar;
    }

    public final void zzz() {
        zzg();
        zza();
        if (this.zzt.zzM()) {
            if (this.zzt.zzf().zzs(null, zzdu.zzX)) {
                zzag zzf = this.zzt.zzf();
                zzf.zzt.zzaw();
                Boolean zzk = zzf.zzk("google_analytics_deferred_deep_link_enabled");
                if (zzk != null && zzk.booleanValue()) {
                    this.zzt.zzay().zzc().zza("Deferred Deep Link feature enabled.");
                    this.zzt.zzaz().zzp(new Runnable() { // from class: com.google.android.gms.measurement.internal.zzgy
                        @Override // java.lang.Runnable
                        public final void run() {
                            zzhx zzhxVar = zzhx.this;
                            zzhxVar.zzg();
                            if (!zzhxVar.zzt.zzm().zzn.zzb()) {
                                long zza = zzhxVar.zzt.zzm().zzo.zza();
                                zzhxVar.zzt.zzm().zzo.zzb(1 + zza);
                                zzhxVar.zzt.zzf();
                                if (zza >= 5) {
                                    a.G(zzhxVar.zzt, "Permanently failed to retrieve Deferred Deep Link. Reached maximum retries.");
                                    zzhxVar.zzt.zzm().zzn.zza(true);
                                    return;
                                }
                                zzhxVar.zzt.zzE();
                                return;
                            }
                            zzhxVar.zzt.zzay().zzc().zza("Deferred Deep Link already retrieved. Not fetching again.");
                        }
                    });
                }
            }
            this.zzt.zzt().zzq();
            this.zzc = false;
            zzew zzm = this.zzt.zzm();
            zzm.zzg();
            String string = zzm.zza().getString("previous_os_version", null);
            zzm.zzt.zzg().zzu();
            String str = Build.VERSION.RELEASE;
            if (!TextUtils.isEmpty(str) && !str.equals(string)) {
                SharedPreferences.Editor edit = zzm.zza().edit();
                edit.putString("previous_os_version", str);
                edit.apply();
            }
            if (TextUtils.isEmpty(string)) {
                return;
            }
            this.zzt.zzg().zzu();
            if (string.equals(str)) {
                return;
            }
            Bundle bundle = new Bundle();
            bundle.putString("_po", string);
            zzG("auto", "_ou", bundle);
        }
    }
}