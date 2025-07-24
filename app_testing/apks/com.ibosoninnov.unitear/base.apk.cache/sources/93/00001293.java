package com.google.android.gms.internal.measurement;

import sun.misc.Unsafe;

/* compiled from: com.google.android.gms:play-services-measurement-base@@21.2.0 */
/* loaded from: classes.dex */
public final class zzmv extends zzmx {
    public zzmv(Unsafe unsafe) {
        super(unsafe);
    }

    @Override // com.google.android.gms.internal.measurement.zzmx
    public final double zza(Object obj, long j) {
        return Double.longBitsToDouble(zzk(obj, j));
    }

    @Override // com.google.android.gms.internal.measurement.zzmx
    public final float zzb(Object obj, long j) {
        return Float.intBitsToFloat(zzj(obj, j));
    }

    @Override // com.google.android.gms.internal.measurement.zzmx
    public final void zzc(Object obj, long j, boolean z) {
        if (zzmy.zzb) {
            zzmy.zzD(obj, j, r3 ? (byte) 1 : (byte) 0);
        } else {
            zzmy.zzE(obj, j, r3 ? (byte) 1 : (byte) 0);
        }
    }

    @Override // com.google.android.gms.internal.measurement.zzmx
    public final void zzd(Object obj, long j, byte b2) {
        if (zzmy.zzb) {
            zzmy.zzD(obj, j, b2);
        } else {
            zzmy.zzE(obj, j, b2);
        }
    }

    @Override // com.google.android.gms.internal.measurement.zzmx
    public final void zze(Object obj, long j, double d2) {
        zzo(obj, j, Double.doubleToLongBits(d2));
    }

    @Override // com.google.android.gms.internal.measurement.zzmx
    public final void zzf(Object obj, long j, float f2) {
        zzn(obj, j, Float.floatToIntBits(f2));
    }

    @Override // com.google.android.gms.internal.measurement.zzmx
    public final boolean zzg(Object obj, long j) {
        if (zzmy.zzb) {
            return zzmy.zzt(obj, j);
        }
        return zzmy.zzu(obj, j);
    }
}