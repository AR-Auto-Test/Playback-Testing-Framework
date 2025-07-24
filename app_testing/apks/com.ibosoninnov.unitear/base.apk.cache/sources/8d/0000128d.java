package com.google.android.gms.internal.measurement;

import java.util.Arrays;

/* compiled from: com.google.android.gms:play-services-measurement-base@@21.2.0 */
/* loaded from: classes.dex */
public final class zzmp {
    private static final zzmp zza = new zzmp(0, new int[0], new Object[0], false);
    private int zzb;
    private int[] zzc;
    private Object[] zzd;
    private int zze;
    private boolean zzf;

    private zzmp() {
        this(0, new int[8], new Object[8], true);
    }

    private zzmp(int i, int[] iArr, Object[] objArr, boolean z) {
        this.zze = -1;
        this.zzb = i;
        this.zzc = iArr;
        this.zzd = objArr;
        this.zzf = z;
    }

    public static zzmp zzc() {
        return zza;
    }

    public static zzmp zze(zzmp zzmpVar, zzmp zzmpVar2) {
        int i = zzmpVar.zzb + zzmpVar2.zzb;
        int[] copyOf = Arrays.copyOf(zzmpVar.zzc, i);
        System.arraycopy(zzmpVar2.zzc, 0, copyOf, zzmpVar.zzb, zzmpVar2.zzb);
        Object[] copyOf2 = Arrays.copyOf(zzmpVar.zzd, i);
        System.arraycopy(zzmpVar2.zzd, 0, copyOf2, zzmpVar.zzb, zzmpVar2.zzb);
        return new zzmp(i, copyOf, copyOf2, true);
    }

    public static zzmp zzf() {
        return new zzmp(0, new int[8], new Object[8], true);
    }

    private final void zzl(int i) {
        int[] iArr = this.zzc;
        if (i > iArr.length) {
            int i2 = this.zzb;
            int i3 = (i2 / 2) + i2;
            if (i3 >= i) {
                i = i3;
            }
            if (i < 8) {
                i = 8;
            }
            this.zzc = Arrays.copyOf(iArr, i);
            this.zzd = Arrays.copyOf(this.zzd, i);
        }
    }

    public final boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }
        if (obj != null && (obj instanceof zzmp)) {
            zzmp zzmpVar = (zzmp) obj;
            int i = this.zzb;
            if (i == zzmpVar.zzb) {
                int[] iArr = this.zzc;
                int[] iArr2 = zzmpVar.zzc;
                int i2 = 0;
                while (true) {
                    if (i2 >= i) {
                        Object[] objArr = this.zzd;
                        Object[] objArr2 = zzmpVar.zzd;
                        int i3 = this.zzb;
                        for (int i4 = 0; i4 < i3; i4++) {
                            if (objArr[i4].equals(objArr2[i4])) {
                            }
                        }
                        return true;
                    } else if (iArr[i2] != iArr2[i2]) {
                        break;
                    } else {
                        i2++;
                    }
                }
            }
            return false;
        }
        return false;
    }

    public final int hashCode() {
        int i = this.zzb;
        int i2 = (i + 527) * 31;
        int[] iArr = this.zzc;
        int i3 = 17;
        int i4 = 17;
        for (int i5 = 0; i5 < i; i5++) {
            i4 = (i4 * 31) + iArr[i5];
        }
        int i6 = (i2 + i4) * 31;
        Object[] objArr = this.zzd;
        int i7 = this.zzb;
        for (int i8 = 0; i8 < i7; i8++) {
            i3 = (i3 * 31) + objArr[i8].hashCode();
        }
        return i6 + i3;
    }

    public final int zza() {
        int zzA;
        int zzB;
        int zzA2;
        int i = this.zze;
        if (i == -1) {
            int i2 = 0;
            for (int i3 = 0; i3 < this.zzb; i3++) {
                int i4 = this.zzc[i3];
                int i5 = i4 >>> 3;
                int i6 = i4 & 7;
                if (i6 != 0) {
                    if (i6 == 1) {
                        ((Long) this.zzd[i3]).longValue();
                        zzA2 = zzjm.zzA(i5 << 3) + 8;
                    } else if (i6 == 2) {
                        int zzA3 = zzjm.zzA(i5 << 3);
                        int zzd = ((zzje) this.zzd[i3]).zzd();
                        i2 = zzjm.zzA(zzd) + zzd + zzA3 + i2;
                    } else if (i6 == 3) {
                        int zzz = zzjm.zzz(i5);
                        zzA = zzz + zzz;
                        zzB = ((zzmp) this.zzd[i3]).zza();
                    } else if (i6 == 5) {
                        ((Integer) this.zzd[i3]).intValue();
                        zzA2 = zzjm.zzA(i5 << 3) + 4;
                    } else {
                        throw new IllegalStateException(zzkp.zza());
                    }
                    i2 = zzA2 + i2;
                } else {
                    long longValue = ((Long) this.zzd[i3]).longValue();
                    zzA = zzjm.zzA(i5 << 3);
                    zzB = zzjm.zzB(longValue);
                }
                i2 = zzB + zzA + i2;
            }
            this.zze = i2;
            return i2;
        }
        return i;
    }

    public final int zzb() {
        int i = this.zze;
        if (i == -1) {
            int i2 = 0;
            for (int i3 = 0; i3 < this.zzb; i3++) {
                int i4 = this.zzc[i3];
                int zzA = zzjm.zzA(8);
                int zzd = ((zzje) this.zzd[i3]).zzd();
                i2 += zzjm.zzA(zzd) + zzd + zzjm.zzA(24) + zzjm.zzA(i4 >>> 3) + zzjm.zzA(16) + zzA + zzA;
            }
            this.zze = i2;
            return i2;
        }
        return i;
    }

    public final zzmp zzd(zzmp zzmpVar) {
        if (zzmpVar.equals(zza)) {
            return this;
        }
        zzg();
        int i = this.zzb + zzmpVar.zzb;
        zzl(i);
        System.arraycopy(zzmpVar.zzc, 0, this.zzc, this.zzb, zzmpVar.zzb);
        System.arraycopy(zzmpVar.zzd, 0, this.zzd, this.zzb, zzmpVar.zzb);
        this.zzb = i;
        return this;
    }

    public final void zzg() {
        if (!this.zzf) {
            throw new UnsupportedOperationException();
        }
    }

    public final void zzh() {
        this.zzf = false;
    }

    public final void zzi(StringBuilder sb, int i) {
        for (int i2 = 0; i2 < this.zzb; i2++) {
            zzlo.zzb(sb, i, String.valueOf(this.zzc[i2] >>> 3), this.zzd[i2]);
        }
    }

    public final void zzj(int i, Object obj) {
        zzg();
        zzl(this.zzb + 1);
        int[] iArr = this.zzc;
        int i2 = this.zzb;
        iArr[i2] = i;
        this.zzd[i2] = obj;
        this.zzb = i2 + 1;
    }

    public final void zzk(zzng zzngVar) {
        if (this.zzb != 0) {
            for (int i = 0; i < this.zzb; i++) {
                int i2 = this.zzc[i];
                Object obj = this.zzd[i];
                int i3 = i2 >>> 3;
                int i4 = i2 & 7;
                if (i4 == 0) {
                    zzngVar.zzt(i3, ((Long) obj).longValue());
                } else if (i4 == 1) {
                    zzngVar.zzm(i3, ((Long) obj).longValue());
                } else if (i4 == 2) {
                    zzngVar.zzd(i3, (zzje) obj);
                } else if (i4 == 3) {
                    zzngVar.zzE(i3);
                    ((zzmp) obj).zzk(zzngVar);
                    zzngVar.zzh(i3);
                } else if (i4 == 5) {
                    zzngVar.zzk(i3, ((Integer) obj).intValue());
                } else {
                    throw new RuntimeException(zzkp.zza());
                }
            }
        }
    }
}