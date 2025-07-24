package com.google.android.gms.flags;

import android.os.RemoteException;
import com.google.android.gms.common.annotation.KeepForSdk;

@KeepForSdk
@Deprecated
/* loaded from: classes.dex */
public abstract class Flag<T> {
    private final String mKey;
    private final int zze;
    private final T zzf;

    @Deprecated
    /* loaded from: classes.dex */
    public static class BooleanFlag extends Flag<Boolean> {
        public BooleanFlag(int i, String str, Boolean bool) {
            super(i, str, bool);
        }

        /* JADX DEBUG: Method merged with bridge method */
        /* JADX INFO: Access modifiers changed from: private */
        @Override // com.google.android.gms.flags.Flag
        /* renamed from: zzb */
        public final Boolean zza(zzc zzcVar) {
            try {
                return Boolean.valueOf(zzcVar.getBooleanFlagValue(getKey(), zzb().booleanValue(), getSource()));
            } catch (RemoteException unused) {
                return zzb();
            }
        }
    }

    @Deprecated
    @KeepForSdk
    /* loaded from: classes.dex */
    public static class IntegerFlag extends Flag<Integer> {
        public IntegerFlag(int i, String str, Integer num) {
            super(i, str, num);
        }

        /* JADX DEBUG: Method merged with bridge method */
        /* JADX INFO: Access modifiers changed from: private */
        @Override // com.google.android.gms.flags.Flag
        /* renamed from: zzc */
        public final Integer zza(zzc zzcVar) {
            try {
                return Integer.valueOf(zzcVar.getIntFlagValue(getKey(), zzb().intValue(), getSource()));
            } catch (RemoteException unused) {
                return zzb();
            }
        }
    }

    @Deprecated
    @KeepForSdk
    /* loaded from: classes.dex */
    public static class LongFlag extends Flag<Long> {
        public LongFlag(int i, String str, Long l) {
            super(i, str, l);
        }

        /* JADX DEBUG: Method merged with bridge method */
        /* JADX INFO: Access modifiers changed from: private */
        @Override // com.google.android.gms.flags.Flag
        /* renamed from: zzd */
        public final Long zza(zzc zzcVar) {
            try {
                return Long.valueOf(zzcVar.getLongFlagValue(getKey(), zzb().longValue(), getSource()));
            } catch (RemoteException unused) {
                return zzb();
            }
        }
    }

    @Deprecated
    @KeepForSdk
    /* loaded from: classes.dex */
    public static class StringFlag extends Flag<String> {
        public StringFlag(int i, String str, String str2) {
            super(i, str, str2);
        }

        /* JADX DEBUG: Method merged with bridge method */
        /* JADX INFO: Access modifiers changed from: private */
        @Override // com.google.android.gms.flags.Flag
        /* renamed from: zze */
        public final String zza(zzc zzcVar) {
            try {
                return zzcVar.getStringFlagValue(getKey(), zzb(), getSource());
            } catch (RemoteException unused) {
                return zzb();
            }
        }
    }

    private Flag(int i, String str, T t) {
        this.zze = i;
        this.mKey = str;
        this.zzf = t;
        Singletons.flagRegistry().zza(this);
    }

    @KeepForSdk
    @Deprecated
    public static BooleanFlag define(int i, String str, Boolean bool) {
        return new BooleanFlag(i, str, bool);
    }

    @KeepForSdk
    public T get() {
        return (T) Singletons.zzd().zzb(this);
    }

    public final String getKey() {
        return this.mKey;
    }

    @Deprecated
    public final int getSource() {
        return this.zze;
    }

    public abstract T zza(zzc zzcVar);

    public final T zzb() {
        return this.zzf;
    }

    @KeepForSdk
    @Deprecated
    public static IntegerFlag define(int i, String str, int i2) {
        return new IntegerFlag(i, str, Integer.valueOf(i2));
    }

    @KeepForSdk
    @Deprecated
    public static LongFlag define(int i, String str, long j) {
        return new LongFlag(i, str, Long.valueOf(j));
    }

    @KeepForSdk
    @Deprecated
    public static StringFlag define(int i, String str, String str2) {
        return new StringFlag(i, str, str2);
    }
}