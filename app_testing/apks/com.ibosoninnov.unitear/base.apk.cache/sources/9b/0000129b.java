package com.google.android.gms.internal.measurement;

import c.b.a.a.a;

/* compiled from: com.google.android.gms:play-services-measurement-base@@21.2.0 */
/* loaded from: classes.dex */
public final class zznc extends IllegalArgumentException {
    public zznc(int i, int i2) {
        super(a.k("Unpaired surrogate at index ", i, " of ", i2));
    }
}