package com.google.android.gms.internal.measurement;

import android.content.Context;
import java.util.Objects;

/* compiled from: com.google.android.gms:play-services-measurement-impl@@21.2.0 */
/* loaded from: classes.dex */
public final class zzhc extends zzhz {
    private final Context zza;
    private final zzii zzb;

    public zzhc(Context context, zzii zziiVar) {
        Objects.requireNonNull(context, "Null context");
        this.zza = context;
        this.zzb = zziiVar;
    }

    public final boolean equals(Object obj) {
        zzii zziiVar;
        if (obj == this) {
            return true;
        }
        if (obj instanceof zzhz) {
            zzhz zzhzVar = (zzhz) obj;
            if (this.zza.equals(zzhzVar.zza()) && ((zziiVar = this.zzb) != null ? zziiVar.equals(zzhzVar.zzb()) : zzhzVar.zzb() == null)) {
                return true;
            }
        }
        return false;
    }

    public final int hashCode() {
        int hashCode = (this.zza.hashCode() ^ 1000003) * 1000003;
        zzii zziiVar = this.zzb;
        return hashCode ^ (zziiVar == null ? 0 : zziiVar.hashCode());
    }

    public final String toString() {
        String obj = this.zza.toString();
        String valueOf = String.valueOf(this.zzb);
        return "FlagsContext{context=" + obj + ", hermeticFileOverrides=" + valueOf + "}";
    }

    @Override // com.google.android.gms.internal.measurement.zzhz
    public final Context zza() {
        return this.zza;
    }

    @Override // com.google.android.gms.internal.measurement.zzhz
    public final zzii zzb() {
        return this.zzb;
    }
}