package com.google.android.gms.internal.vision;

import android.content.Context;
import android.net.Uri;

/* compiled from: com.google.android.gms:play-services-vision-common@@19.1.3 */
/* loaded from: classes.dex */
public final class zzbo {
    public final String zza;
    public final Uri zzb;
    public final String zzc;
    public final String zzd;
    public final boolean zze;
    public final boolean zzf;
    public final boolean zzg;
    public final boolean zzh;
    public final zzcw<Context, Boolean> zzi;

    public zzbo(Uri uri) {
        this(null, uri, "", "", false, false, false, false, null);
    }

    public final zzbo zza(String str) {
        boolean z = this.zze;
        if (!z) {
            return new zzbo(this.zza, this.zzb, str, this.zzd, z, this.zzf, this.zzg, this.zzh, this.zzi);
        }
        throw new IllegalStateException("Cannot set GServices prefix and skip GServices");
    }

    private zzbo(String str, Uri uri, String str2, String str3, boolean z, boolean z2, boolean z3, boolean z4, zzcw<Context, Boolean> zzcwVar) {
        this.zza = str;
        this.zzb = uri;
        this.zzc = str2;
        this.zzd = str3;
        this.zze = z;
        this.zzf = z2;
        this.zzg = z3;
        this.zzh = z4;
        this.zzi = zzcwVar;
    }

    public final <T> zzbi<T> zza(String str, T t, zzbp<T> zzbpVar) {
        zzbi<T> zzb;
        zzb = zzbi.zzb(this, str, t, zzbpVar, true);
        return zzb;
    }
}