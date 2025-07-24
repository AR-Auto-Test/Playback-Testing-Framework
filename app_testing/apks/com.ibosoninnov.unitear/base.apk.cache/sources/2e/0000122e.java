package com.google.android.gms.internal.measurement;

import c.b.a.a.a;
import java.nio.charset.Charset;
import java.util.Objects;

/* compiled from: com.google.android.gms:play-services-measurement-base@@21.2.0 */
/* loaded from: classes.dex */
public class zzjb extends zzja {
    public final byte[] zza;

    public zzjb(byte[] bArr) {
        Objects.requireNonNull(bArr);
        this.zza = bArr;
    }

    @Override // com.google.android.gms.internal.measurement.zzje
    public final boolean equals(Object obj) {
        if (obj == this) {
            return true;
        }
        if ((obj instanceof zzje) && zzd() == ((zzje) obj).zzd()) {
            if (zzd() == 0) {
                return true;
            }
            if (obj instanceof zzjb) {
                zzjb zzjbVar = (zzjb) obj;
                int zzk = zzk();
                int zzk2 = zzjbVar.zzk();
                if (zzk == 0 || zzk2 == 0 || zzk == zzk2) {
                    int zzd = zzd();
                    if (zzd <= zzjbVar.zzd()) {
                        if (zzd <= zzjbVar.zzd()) {
                            byte[] bArr = this.zza;
                            byte[] bArr2 = zzjbVar.zza;
                            zzjbVar.zzc();
                            int i = 0;
                            int i2 = 0;
                            while (i < zzd) {
                                if (bArr[i] != bArr2[i2]) {
                                    return false;
                                }
                                i++;
                                i2++;
                            }
                            return true;
                        }
                        throw new IllegalArgumentException(a.k("Ran off end of other: 0, ", zzd, ", ", zzjbVar.zzd()));
                    }
                    throw new IllegalArgumentException("Length too large: " + zzd + zzd());
                }
                return false;
            }
            return obj.equals(this);
        }
        return false;
    }

    @Override // com.google.android.gms.internal.measurement.zzje
    public byte zza(int i) {
        return this.zza[i];
    }

    @Override // com.google.android.gms.internal.measurement.zzje
    public byte zzb(int i) {
        return this.zza[i];
    }

    public int zzc() {
        return 0;
    }

    @Override // com.google.android.gms.internal.measurement.zzje
    public int zzd() {
        return this.zza.length;
    }

    @Override // com.google.android.gms.internal.measurement.zzje
    public final int zze(int i, int i2, int i3) {
        return zzkn.zzd(i, this.zza, 0, i3);
    }

    @Override // com.google.android.gms.internal.measurement.zzje
    public final zzje zzf(int i, int i2) {
        int zzj = zzje.zzj(0, i2, zzd());
        return zzj == 0 ? zzje.zzb : new zziy(this.zza, 0, zzj);
    }

    @Override // com.google.android.gms.internal.measurement.zzje
    public final String zzg(Charset charset) {
        return new String(this.zza, 0, zzd(), charset);
    }

    @Override // com.google.android.gms.internal.measurement.zzje
    public final void zzh(zziu zziuVar) {
        ((zzjj) zziuVar).zzc(this.zza, 0, zzd());
    }

    @Override // com.google.android.gms.internal.measurement.zzje
    public final boolean zzi() {
        return zznd.zzf(this.zza, 0, zzd());
    }
}