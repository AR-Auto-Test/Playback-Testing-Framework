package com.google.android.gms.internal.clearcut;

/* loaded from: classes.dex */
public abstract class zzcy {
    private static final zzcy zzlt = new zzda();
    private static final zzcy zzlu = new zzdb();

    private zzcy() {
    }

    public static zzcy zzbv() {
        return zzlt;
    }

    public static zzcy zzbw() {
        return zzlu;
    }

    public abstract void zza(Object obj, long j);

    public abstract <L> void zza(Object obj, Object obj2, long j);
}