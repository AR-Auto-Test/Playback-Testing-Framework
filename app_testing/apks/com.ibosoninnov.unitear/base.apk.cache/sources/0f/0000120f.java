package com.google.android.gms.internal.measurement;

import android.net.Uri;

/* compiled from: com.google.android.gms:play-services-measurement-impl@@21.2.0 */
/* loaded from: classes.dex */
public final class zzhy {
    public final String zza;
    public final Uri zzb;
    public final String zzc;
    public final String zzd;
    public final boolean zze;
    public final boolean zzf;
    public final boolean zzg;
    public final boolean zzh;
    public final zzif zzi;

    public zzhy(Uri uri) {
        this(null, uri, "", "", false, false, false, false, null);
    }

    private zzhy(String str, Uri uri, String str2, String str3, boolean z, boolean z2, boolean z3, boolean z4, zzif zzifVar) {
        this.zza = null;
        this.zzb = uri;
        this.zzc = "";
        this.zzd = "";
        this.zze = z;
        this.zzf = false;
        this.zzg = z3;
        this.zzh = false;
        this.zzi = null;
    }

    public final zzhy zza() {
        return new zzhy(null, this.zzb, this.zzc, this.zzd, this.zze, false, true, false, null);
    }

    public final zzhy zzb() {
        if (this.zzc.isEmpty()) {
            return new zzhy(null, this.zzb, this.zzc, this.zzd, true, false, this.zzg, false, null);
        }
        throw new IllegalStateException("Cannot set GServices prefix and skip GServices");
    }

    public final zzib zzc(String str, double d2) {
        return new zzhw(this, "measurement.test.double_flag", Double.valueOf(-3.0d), true);
    }

    public final zzib zzd(String str, long j) {
        return new zzhu(this, str, Long.valueOf(j), true);
    }

    public final zzib zze(String str, String str2) {
        return new zzhx(this, str, str2, true);
    }

    public final zzib zzf(String str, boolean z) {
        return new zzhv(this, str, Boolean.valueOf(z), true);
    }
}