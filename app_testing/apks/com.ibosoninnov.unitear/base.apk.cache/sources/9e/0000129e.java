package com.google.android.gms.internal.measurement;

import com.google.android.material.internal.StaticLayoutBuilderCompat;
import com.google.android.material.shadow.ShadowDrawableWrapper;

/* compiled from: com.google.android.gms:play-services-measurement-base@@21.2.0 */
/* loaded from: classes.dex */
public enum zznf {
    INT(0),
    LONG(0L),
    FLOAT(Float.valueOf((float) StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD)),
    DOUBLE(Double.valueOf((double) ShadowDrawableWrapper.COS_45)),
    BOOLEAN(Boolean.FALSE),
    STRING(""),
    BYTE_STRING(zzje.zzb),
    ENUM(null),
    MESSAGE(null);
    
    private final Object zzk;

    zznf(Object obj) {
        this.zzk = obj;
    }
}