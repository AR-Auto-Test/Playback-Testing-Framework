package com.google.android.gms.internal.measurement;

import c.b.a.a.a;
import java.io.Serializable;
import java.util.Objects;

/* compiled from: com.google.android.gms:play-services-measurement-impl@@21.2.0 */
/* loaded from: classes.dex */
public final class zzij implements Serializable, zzii {
    public final zzii zza;
    public volatile transient boolean zzb;
    public transient Object zzc;

    public zzij(zzii zziiVar) {
        Objects.requireNonNull(zziiVar);
        this.zza = zziiVar;
    }

    public final String toString() {
        return a.u(a.x("Suppliers.memoize("), this.zzb ? a.u(a.x("<supplier that returned "), this.zzc, ">") : this.zza, ")");
    }

    @Override // com.google.android.gms.internal.measurement.zzii
    public final Object zza() {
        if (!this.zzb) {
            synchronized (this) {
                if (!this.zzb) {
                    Object zza = this.zza.zza();
                    this.zzc = zza;
                    this.zzb = true;
                    return zza;
                }
            }
        }
        return this.zzc;
    }
}