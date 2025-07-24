package com.google.android.play.core.missingsplits;

import android.content.Context;
import java.util.concurrent.atomic.AtomicReference;

/* compiled from: com.google.android.play:core@@1.10.3 */
@Deprecated
/* loaded from: classes.dex */
public final class MissingSplitsManagerFactory {
    private static final AtomicReference zza = new AtomicReference(null);

    private MissingSplitsManagerFactory() {
    }

    @Deprecated
    public static MissingSplitsManager create(Context context) {
        return new zzb(context, Runtime.getRuntime(), new zza(context, context.getPackageManager()), zza);
    }
}