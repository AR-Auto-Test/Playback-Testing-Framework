package com.google.android.gms.measurement.internal;

import android.app.Activity;
import android.os.Bundle;
import com.google.android.gms.common.internal.Preconditions;
import com.google.android.gms.common.util.VisibleForTesting;
import com.google.firebase.analytics.FirebaseAnalytics;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/* compiled from: com.google.android.gms:play-services-measurement-impl@@21.2.0 */
/* loaded from: classes.dex */
public final class zzim extends zzf {
    @VisibleForTesting
    public zzie zza;
    private volatile zzie zzb;
    private volatile zzie zzc;
    private final Map zzd;
    private Activity zze;
    private volatile boolean zzf;
    private volatile zzie zzg;
    private zzie zzh;
    private boolean zzi;
    private final Object zzj;

    public zzim(zzfr zzfrVar) {
        super(zzfrVar);
        this.zzj = new Object();
        this.zzd = new ConcurrentHashMap();
    }

    /* JADX INFO: Access modifiers changed from: private */
    public final void zzA(zzie zzieVar, zzie zzieVar2, long j, boolean z, Bundle bundle) {
        Bundle bundle2;
        long j2;
        long j3;
        zzg();
        boolean z2 = false;
        boolean z3 = (zzieVar2 != null && zzieVar2.zzc == zzieVar.zzc && zzif.zza(zzieVar2.zzb, zzieVar.zzb) && zzif.zza(zzieVar2.zza, zzieVar.zza)) ? false : true;
        if (z && this.zza != null) {
            z2 = true;
        }
        if (z3) {
            if (bundle != null) {
                bundle2 = new Bundle(bundle);
            } else {
                bundle2 = new Bundle();
            }
            Bundle bundle3 = bundle2;
            zzlb.zzK(zzieVar, bundle3, true);
            if (zzieVar2 != null) {
                String str = zzieVar2.zza;
                if (str != null) {
                    bundle3.putString("_pn", str);
                }
                String str2 = zzieVar2.zzb;
                if (str2 != null) {
                    bundle3.putString("_pc", str2);
                }
                bundle3.putLong("_pi", zzieVar2.zzc);
            }
            if (z2) {
                zzka zzkaVar = this.zzt.zzu().zzb;
                long j4 = j - zzkaVar.zzb;
                zzkaVar.zzb = j;
                if (j4 > 0) {
                    this.zzt.zzv().zzI(bundle3, j4);
                }
            }
            if (!this.zzt.zzf().zzu()) {
                bundle3.putLong("_mst", 1L);
            }
            String str3 = true != zzieVar.zze ? "auto" : "app";
            long currentTimeMillis = this.zzt.zzav().currentTimeMillis();
            if (zzieVar.zze) {
                j2 = currentTimeMillis;
                long j5 = zzieVar.zzf;
                if (j5 != 0) {
                    j3 = j5;
                    this.zzt.zzq().zzH(str3, "_vs", j3, bundle3);
                }
            } else {
                j2 = currentTimeMillis;
            }
            j3 = j2;
            this.zzt.zzq().zzH(str3, "_vs", j3, bundle3);
        }
        if (z2) {
            zzB(this.zza, true, j);
        }
        this.zza = zzieVar;
        if (zzieVar.zze) {
            this.zzh = zzieVar;
        }
        this.zzt.zzt().zzG(zzieVar);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public final void zzB(zzie zzieVar, boolean z, long j) {
        this.zzt.zzd().zzf(this.zzt.zzav().elapsedRealtime());
        if (!this.zzt.zzu().zzb.zzd(zzieVar != null && zzieVar.zzd, z, j) || zzieVar == null) {
            return;
        }
        zzieVar.zzd = false;
    }

    public static /* bridge */ /* synthetic */ void zzp(zzim zzimVar, Bundle bundle, zzie zzieVar, zzie zzieVar2, long j) {
        bundle.remove(FirebaseAnalytics.Param.SCREEN_NAME);
        bundle.remove(FirebaseAnalytics.Param.SCREEN_CLASS);
        zzimVar.zzA(zzieVar, zzieVar2, j, true, zzimVar.zzt.zzv().zzy(null, FirebaseAnalytics.Event.SCREEN_VIEW, bundle, null, false));
    }

    private final zzie zzy(Activity activity) {
        Preconditions.checkNotNull(activity);
        zzie zzieVar = (zzie) this.zzd.get(activity);
        if (zzieVar == null) {
            zzie zzieVar2 = new zzie(null, zzl(activity.getClass(), "Activity"), this.zzt.zzv().zzq());
            this.zzd.put(activity, zzieVar2);
            zzieVar = zzieVar2;
        }
        return this.zzg != null ? this.zzg : zzieVar;
    }

    private final void zzz(Activity activity, zzie zzieVar, boolean z) {
        zzie zzieVar2;
        zzie zzieVar3 = this.zzb == null ? this.zzc : this.zzb;
        if (zzieVar.zzb == null) {
            zzieVar2 = new zzie(zzieVar.zza, activity != null ? zzl(activity.getClass(), "Activity") : null, zzieVar.zzc, zzieVar.zze, zzieVar.zzf);
        } else {
            zzieVar2 = zzieVar;
        }
        this.zzc = this.zzb;
        this.zzb = zzieVar2;
        this.zzt.zzaz().zzp(new zzih(this, zzieVar2, zzieVar3, this.zzt.zzav().elapsedRealtime(), z));
    }

    @Override // com.google.android.gms.measurement.internal.zzf
    public final boolean zzf() {
        return false;
    }

    public final zzie zzi() {
        return this.zzb;
    }

    public final zzie zzj(boolean z) {
        zza();
        zzg();
        if (z) {
            zzie zzieVar = this.zza;
            return zzieVar != null ? zzieVar : this.zzh;
        }
        return this.zza;
    }

    @VisibleForTesting
    public final String zzl(Class cls, String str) {
        String canonicalName = cls.getCanonicalName();
        if (canonicalName == null) {
            return "Activity";
        }
        String[] split = canonicalName.split("\\.");
        int length = split.length;
        String str2 = length > 0 ? split[length - 1] : "";
        int length2 = str2.length();
        this.zzt.zzf();
        if (length2 > 100) {
            this.zzt.zzf();
            return str2.substring(0, 100);
        }
        return str2;
    }

    public final void zzr(Activity activity, Bundle bundle) {
        Bundle bundle2;
        if (!this.zzt.zzf().zzu() || bundle == null || (bundle2 = bundle.getBundle("com.google.app_measurement.screen_service")) == null) {
            return;
        }
        this.zzd.put(activity, new zzie(bundle2.getString("name"), bundle2.getString("referrer_name"), bundle2.getLong("id")));
    }

    public final void zzs(Activity activity) {
        synchronized (this.zzj) {
            if (activity == this.zze) {
                this.zze = null;
            }
        }
        if (this.zzt.zzf().zzu()) {
            this.zzd.remove(activity);
        }
    }

    public final void zzt(Activity activity) {
        synchronized (this.zzj) {
            this.zzi = false;
            this.zzf = true;
        }
        long elapsedRealtime = this.zzt.zzav().elapsedRealtime();
        if (!this.zzt.zzf().zzu()) {
            this.zzb = null;
            this.zzt.zzaz().zzp(new zzij(this, elapsedRealtime));
            return;
        }
        zzie zzy = zzy(activity);
        this.zzc = this.zzb;
        this.zzb = null;
        this.zzt.zzaz().zzp(new zzik(this, zzy, elapsedRealtime));
    }

    public final void zzu(Activity activity) {
        synchronized (this.zzj) {
            this.zzi = true;
            if (activity != this.zze) {
                synchronized (this.zzj) {
                    this.zze = activity;
                    this.zzf = false;
                }
                if (this.zzt.zzf().zzu()) {
                    this.zzg = null;
                    this.zzt.zzaz().zzp(new zzil(this));
                }
            }
        }
        if (!this.zzt.zzf().zzu()) {
            this.zzb = this.zzg;
            this.zzt.zzaz().zzp(new zzii(this));
            return;
        }
        zzz(activity, zzy(activity), false);
        zzd zzd = this.zzt.zzd();
        zzd.zzt.zzaz().zzp(new zzc(zzd, zzd.zzt.zzav().elapsedRealtime()));
    }

    public final void zzv(Activity activity, Bundle bundle) {
        zzie zzieVar;
        if (!this.zzt.zzf().zzu() || bundle == null || (zzieVar = (zzie) this.zzd.get(activity)) == null) {
            return;
        }
        Bundle bundle2 = new Bundle();
        bundle2.putLong("id", zzieVar.zzc);
        bundle2.putString("name", zzieVar.zza);
        bundle2.putString("referrer_name", zzieVar.zzb);
        bundle.putBundle("com.google.app_measurement.screen_service", bundle2);
    }

    /* JADX WARN: Code restructure failed: missing block: B:27:0x0088, code lost:
        if (r5.length() <= 100) goto L29;
     */
    /* JADX WARN: Code restructure failed: missing block: B:35:0x00b4, code lost:
        if (r6.length() <= 100) goto L36;
     */
    @Deprecated
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public final void zzw(Activity activity, String str, String str2) {
        if (!this.zzt.zzf().zzu()) {
            this.zzt.zzay().zzl().zza("setCurrentScreen cannot be called while screen reporting is disabled.");
            return;
        }
        zzie zzieVar = this.zzb;
        if (zzieVar == null) {
            this.zzt.zzay().zzl().zza("setCurrentScreen cannot be called while no activity active");
        } else if (this.zzd.get(activity) == null) {
            this.zzt.zzay().zzl().zza("setCurrentScreen must be called with an activity in the activity lifecycle");
        } else {
            if (str2 == null) {
                str2 = zzl(activity.getClass(), "Activity");
            }
            boolean zza = zzif.zza(zzieVar.zzb, str2);
            boolean zza2 = zzif.zza(zzieVar.zza, str);
            if (zza && zza2) {
                this.zzt.zzay().zzl().zza("setCurrentScreen cannot be called with the same class and name");
                return;
            }
            if (str != null) {
                if (str.length() > 0) {
                    this.zzt.zzf();
                }
                this.zzt.zzay().zzl().zzb("Invalid screen name length in setCurrentScreen. Length", Integer.valueOf(str.length()));
                return;
            }
            if (str2 != null) {
                if (str2.length() > 0) {
                    this.zzt.zzf();
                }
                this.zzt.zzay().zzl().zzb("Invalid class name length in setCurrentScreen. Length", Integer.valueOf(str2.length()));
                return;
            }
            this.zzt.zzay().zzj().zzc("Setting current screen to name, class", str == null ? "null" : str, str2);
            zzie zzieVar2 = new zzie(str, str2, this.zzt.zzv().zzq());
            this.zzd.put(activity, zzieVar2);
            zzz(activity, zzieVar2, true);
        }
    }

    /* JADX WARN: Code restructure failed: missing block: B:14:0x0031, code lost:
        if (r2 > 100) goto L16;
     */
    /* JADX WARN: Code restructure failed: missing block: B:23:0x0063, code lost:
        if (r4 > 100) goto L25;
     */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public final void zzx(Bundle bundle, long j) {
        String str;
        synchronized (this.zzj) {
            if (!this.zzi) {
                this.zzt.zzay().zzl().zza("Cannot log screen view event when the app is in the background.");
                return;
            }
            String string = bundle.getString(FirebaseAnalytics.Param.SCREEN_NAME);
            if (string != null) {
                if (string.length() > 0) {
                    int length = string.length();
                    this.zzt.zzf();
                }
                this.zzt.zzay().zzl().zzb("Invalid screen name length for screen view. Length", Integer.valueOf(string.length()));
                return;
            }
            String string2 = bundle.getString(FirebaseAnalytics.Param.SCREEN_CLASS);
            if (string2 != null) {
                if (string2.length() > 0) {
                    int length2 = string2.length();
                    this.zzt.zzf();
                }
                this.zzt.zzay().zzl().zzb("Invalid screen class length for screen view. Length", Integer.valueOf(string2.length()));
                return;
            }
            if (string2 == null) {
                Activity activity = this.zze;
                str = activity != null ? zzl(activity.getClass(), "Activity") : "Activity";
            } else {
                str = string2;
            }
            zzie zzieVar = this.zzb;
            if (this.zzf && zzieVar != null) {
                this.zzf = false;
                boolean zza = zzif.zza(zzieVar.zzb, str);
                boolean zza2 = zzif.zza(zzieVar.zza, string);
                if (zza && zza2) {
                    this.zzt.zzay().zzl().zza("Ignoring call to log screen view event with duplicate parameters.");
                    return;
                }
            }
            this.zzt.zzay().zzj().zzc("Logging screen view with name, class", string == null ? "null" : string, str == null ? "null" : str);
            zzie zzieVar2 = this.zzb == null ? this.zzc : this.zzb;
            zzie zzieVar3 = new zzie(string, str, this.zzt.zzv().zzq(), true, j);
            this.zzb = zzieVar3;
            this.zzc = zzieVar2;
            this.zzg = zzieVar3;
            this.zzt.zzaz().zzp(new zzig(this, bundle, zzieVar3, zzieVar2, this.zzt.zzav().elapsedRealtime()));
        }
    }
}