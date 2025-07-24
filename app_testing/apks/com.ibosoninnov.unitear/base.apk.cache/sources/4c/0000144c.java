package com.google.android.gms.internal.vision;

import c.b.a.a.a;

/* compiled from: com.google.android.gms:play-services-vision-common@@19.1.3 */
/* loaded from: classes.dex */
public final class zzmg extends IllegalArgumentException {
    public zzmg(int i, int i2) {
        super(a.h(54, "Unpaired surrogate at index ", i, " of ", i2));
    }
}