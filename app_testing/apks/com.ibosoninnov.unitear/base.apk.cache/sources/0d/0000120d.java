package com.google.android.gms.internal.measurement;

import android.util.Log;
import c.b.a.a.a;

/* compiled from: com.google.android.gms:play-services-measurement-impl@@21.2.0 */
/* loaded from: classes.dex */
public final class zzhw extends zzib {
    public zzhw(zzhy zzhyVar, String str, Double d2, boolean z) {
        super(zzhyVar, "measurement.test.double_flag", d2, true, null);
    }

    @Override // com.google.android.gms.internal.measurement.zzib
    public final /* bridge */ /* synthetic */ Object zza(Object obj) {
        try {
            return Double.valueOf(Double.parseDouble((String) obj));
        } catch (NumberFormatException unused) {
            StringBuilder B = a.B("Invalid double value for ", super.zzc(), ": ");
            B.append((String) obj);
            Log.e("PhenotypeFlag", B.toString());
            return null;
        }
    }
}