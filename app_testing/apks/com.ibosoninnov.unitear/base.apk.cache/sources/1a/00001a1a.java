package com.google.android.play.core.assetpacks;

import com.google.android.material.shadow.ShadowDrawableWrapper;
import java.util.HashMap;
import java.util.Map;

/* compiled from: com.google.android.play:core@@1.10.3 */
/* loaded from: classes.dex */
public final class zzco {
    private final Map zza = new HashMap();

    public final synchronized double zza(String str) {
        Double d2 = (Double) this.zza.get(str);
        if (d2 == null) {
            return ShadowDrawableWrapper.COS_45;
        }
        return d2.doubleValue();
    }

    public final synchronized double zzb(String str, zzdg zzdgVar) {
        double d2;
        d2 = (((zzce) zzdgVar).zzf + 1.0d) / ((zzce) zzdgVar).zzg;
        this.zza.put(str, Double.valueOf(d2));
        return d2;
    }

    public final synchronized void zzc(String str) {
        this.zza.put(str, Double.valueOf((double) ShadowDrawableWrapper.COS_45));
    }
}