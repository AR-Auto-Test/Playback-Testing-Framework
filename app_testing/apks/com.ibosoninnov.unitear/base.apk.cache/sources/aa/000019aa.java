package com.google.android.play.core.appupdate;

import android.content.Context;
import com.google.android.play.core.internal.zzce;

/* compiled from: com.google.android.play:core@@1.10.3 */
/* loaded from: classes.dex */
public final class zza {
    private static zzaa zza;

    public static synchronized zzaa zza(Context context) {
        zzaa zzaaVar;
        synchronized (zza.class) {
            if (zza == null) {
                zzy zzyVar = new zzy(null);
                zzyVar.zza(new zzh(zzce.zza(context)));
                zza = zzyVar.zzb();
            }
            zzaaVar = zza;
        }
        return zzaaVar;
    }
}