package com.google.android.gms.internal.clearcut;

import c.b.a.a.a;
import java.nio.charset.Charset;

/* loaded from: classes.dex */
public class zzbi extends zzbh {
    public final byte[] zzfp;

    public zzbi(byte[] bArr) {
        this.zzfp = bArr;
    }

    @Override // com.google.android.gms.internal.clearcut.zzbb
    public final boolean equals(Object obj) {
        if (obj == this) {
            return true;
        }
        if ((obj instanceof zzbb) && size() == ((zzbb) obj).size()) {
            if (size() == 0) {
                return true;
            }
            if (obj instanceof zzbi) {
                zzbi zzbiVar = (zzbi) obj;
                int zzab = zzab();
                int zzab2 = zzbiVar.zzab();
                if (zzab == 0 || zzab2 == 0 || zzab == zzab2) {
                    return zza(zzbiVar, 0, size());
                }
                return false;
            }
            return obj.equals(this);
        }
        return false;
    }

    @Override // com.google.android.gms.internal.clearcut.zzbb
    public int size() {
        return this.zzfp.length;
    }

    @Override // com.google.android.gms.internal.clearcut.zzbb
    public final int zza(int i, int i2, int i3) {
        return zzci.zza(i, this.zzfp, zzac(), i3);
    }

    @Override // com.google.android.gms.internal.clearcut.zzbb
    public final zzbb zza(int i, int i2) {
        int zzb = zzbb.zzb(0, i2, size());
        return zzb == 0 ? zzbb.zzfi : new zzbe(this.zzfp, zzac(), zzb);
    }

    @Override // com.google.android.gms.internal.clearcut.zzbb
    public final String zza(Charset charset) {
        return new String(this.zzfp, zzac(), size(), charset);
    }

    @Override // com.google.android.gms.internal.clearcut.zzbb
    public final void zza(zzba zzbaVar) {
        zzbaVar.zza(this.zzfp, zzac(), size());
    }

    @Override // com.google.android.gms.internal.clearcut.zzbh
    public final boolean zza(zzbb zzbbVar, int i, int i2) {
        if (i2 > zzbbVar.size()) {
            int size = size();
            StringBuilder sb = new StringBuilder(40);
            sb.append("Length too large: ");
            sb.append(i2);
            sb.append(size);
            throw new IllegalArgumentException(sb.toString());
        } else if (i2 <= zzbbVar.size()) {
            if (zzbbVar instanceof zzbi) {
                zzbi zzbiVar = (zzbi) zzbbVar;
                byte[] bArr = this.zzfp;
                byte[] bArr2 = zzbiVar.zzfp;
                int zzac = zzac() + i2;
                int zzac2 = zzac();
                int zzac3 = zzbiVar.zzac();
                while (zzac2 < zzac) {
                    if (bArr[zzac2] != bArr2[zzac3]) {
                        return false;
                    }
                    zzac2++;
                    zzac3++;
                }
                return true;
            }
            return zzbbVar.zza(0, i2).equals(zza(0, i2));
        } else {
            throw new IllegalArgumentException(a.h(59, "Ran off end of other: 0, ", i2, ", ", zzbbVar.size()));
        }
    }

    @Override // com.google.android.gms.internal.clearcut.zzbb
    public final boolean zzaa() {
        int zzac = zzac();
        return zzff.zze(this.zzfp, zzac, size() + zzac);
    }

    public int zzac() {
        return 0;
    }

    @Override // com.google.android.gms.internal.clearcut.zzbb
    public byte zzj(int i) {
        return this.zzfp[i];
    }
}