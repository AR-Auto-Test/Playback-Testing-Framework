package com.google.android.gms.internal.vision;

/* compiled from: com.google.android.gms:play-services-vision-common@@19.1.3 */
/* loaded from: classes.dex */
public abstract class zzlu<T, B> {
    public abstract B zza();

    public abstract T zza(B b2);

    public abstract void zza(B b2, int i, int i2);

    public abstract void zza(B b2, int i, long j);

    public abstract void zza(B b2, int i, zzht zzhtVar);

    public abstract void zza(B b2, int i, T t);

    public abstract void zza(T t, zzmr zzmrVar);

    public abstract void zza(Object obj, T t);

    public abstract boolean zza(zzld zzldVar);

    public final boolean zza(B b2, zzld zzldVar) {
        int zzb = zzldVar.zzb();
        int i = zzb >>> 3;
        int i2 = zzb & 7;
        if (i2 == 0) {
            zza((zzlu<T, B>) b2, i, zzldVar.zzg());
            return true;
        } else if (i2 == 1) {
            zzb(b2, i, zzldVar.zzi());
            return true;
        } else if (i2 == 2) {
            zza((zzlu<T, B>) b2, i, zzldVar.zzn());
            return true;
        } else if (i2 != 3) {
            if (i2 != 4) {
                if (i2 == 5) {
                    zza((zzlu<T, B>) b2, i, zzldVar.zzj());
                    return true;
                }
                throw zzjk.zzf();
            }
            return false;
        } else {
            B zza = zza();
            int i3 = 4 | (i << 3);
            while (zzldVar.zza() != Integer.MAX_VALUE && zza((zzlu<T, B>) zza, zzldVar)) {
            }
            if (i3 == zzldVar.zzb()) {
                zza((zzlu<T, B>) b2, i, (int) zza((zzlu<T, B>) zza));
                return true;
            }
            throw zzjk.zze();
        }
    }

    public abstract T zzb(Object obj);

    public abstract void zzb(B b2, int i, long j);

    public abstract void zzb(T t, zzmr zzmrVar);

    public abstract void zzb(Object obj, B b2);

    public abstract B zzc(Object obj);

    public abstract T zzc(T t, T t2);

    public abstract void zzd(Object obj);

    public abstract int zze(T t);

    public abstract int zzf(T t);
}