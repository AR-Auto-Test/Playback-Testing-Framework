package com.google.android.gms.internal.measurement;

import android.util.Log;
import c.b.a.a.a;

/* compiled from: com.google.android.gms:play-services-measurement-impl@@21.2.0 */
/* loaded from: classes.dex */
public final class zzhu extends zzib {
    public zzhu(zzhy zzhyVar, String str, Long l, boolean z) {
        super(zzhyVar, str, l, true, null);
    }

    @Override // com.google.android.gms.internal.measurement.zzib
    public final /* bridge */ /* synthetic */ Object zza(Object obj) {
        try {
            return Long.valueOf(Long.parseLong((String) obj));
        } catch (NumberFormatException unused) {
            StringBuilder B = a.B("Invalid long value for ", super.zzc(), ": ");
            B.append((String) obj);
            Log.e("PhenotypeFlag", B.toString());
            return null;
        }
    }
}