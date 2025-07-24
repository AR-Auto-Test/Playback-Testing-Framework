package com.google.android.gms.internal.measurement;

import c.b.a.a.a;
import java.util.Objects;

/* compiled from: com.google.android.gms:play-services-measurement-impl@@21.2.0 */
/* loaded from: classes.dex */
public final class zzik implements zzii {
    public volatile zzii zza;
    public volatile boolean zzb;
    public Object zzc;

    public zzik(zzii zziiVar) {
        Objects.requireNonNull(zziiVar);
        this.zza = zziiVar;
    }

    public final String toString() {
        Object obj = this.zza;
        StringBuilder x = a.x("Suppliers.memoize(");
        if (obj == null) {
            obj = a.u(a.x("<supplier that returned "), this.zzc, ">");
        }
        return a.u(x, obj, ")");
    }

    @Override // com.google.android.gms.internal.measurement.zzii
    public final Object zza() {
        if (!this.zzb) {
            synchronized (this) {
                if (!this.zzb) {
                    zzii zziiVar = this.zza;
                    zziiVar.getClass();
                    Object zza = zziiVar.zza();
                    this.zzc = zza;
                    this.zzb = true;
                    this.zza = null;
                    return zza;
                }
            }
        }
        return this.zzc;
    }
}