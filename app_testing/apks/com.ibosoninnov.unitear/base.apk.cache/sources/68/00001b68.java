package com.google.android.play.core.splitinstall;

import java.util.concurrent.atomic.AtomicReference;

/* compiled from: com.google.android.play:core@@1.10.3 */
/* loaded from: classes.dex */
public final class zzr {
    private static final AtomicReference zza = new AtomicReference(null);

    public static zzq zza() {
        return (zzq) zza.get();
    }

    public static void zzb(zzq zzqVar) {
        zza.compareAndSet(null, zzqVar);
    }
}