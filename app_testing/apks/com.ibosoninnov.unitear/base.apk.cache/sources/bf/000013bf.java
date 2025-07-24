package com.google.android.gms.internal.vision;

import com.google.common.base.Ascii;
import com.google.common.primitives.UnsignedBytes;
import java.nio.ByteBuffer;
import java.util.List;
import java.util.Map;

/* compiled from: com.google.android.gms:play-services-vision-common@@19.1.3 */
/* loaded from: classes.dex */
public final class zzho extends zzhm {
    private final boolean zza;
    private final byte[] zzb;
    private int zzc;
    private final int zzd;
    private int zze;
    private int zzf;
    private int zzg;

    public zzho(ByteBuffer byteBuffer, boolean z) {
        super(null);
        this.zza = true;
        this.zzb = byteBuffer.array();
        int position = byteBuffer.position() + byteBuffer.arrayOffset();
        this.zzc = position;
        this.zzd = position;
        this.zze = byteBuffer.limit() + byteBuffer.arrayOffset();
    }

    private final long zzaa() {
        zzb(8);
        return zzac();
    }

    private final int zzab() {
        int i = this.zzc;
        byte[] bArr = this.zzb;
        this.zzc = i + 4;
        return ((bArr[i + 3] & UnsignedBytes.MAX_VALUE) << 24) | (bArr[i] & UnsignedBytes.MAX_VALUE) | ((bArr[i + 1] & UnsignedBytes.MAX_VALUE) << 8) | ((bArr[i + 2] & UnsignedBytes.MAX_VALUE) << 16);
    }

    private final long zzac() {
        int i = this.zzc;
        byte[] bArr = this.zzb;
        this.zzc = i + 8;
        return ((bArr[i + 7] & 255) << 56) | (bArr[i] & 255) | ((bArr[i + 1] & 255) << 8) | ((bArr[i + 2] & 255) << 16) | ((bArr[i + 3] & 255) << 24) | ((bArr[i + 4] & 255) << 32) | ((bArr[i + 5] & 255) << 40) | ((bArr[i + 6] & 255) << 48);
    }

    private final boolean zzu() {
        return this.zzc == this.zze;
    }

    private final int zzv() {
        int i;
        int i2 = this.zzc;
        int i3 = this.zze;
        if (i3 != i2) {
            byte[] bArr = this.zzb;
            int i4 = i2 + 1;
            byte b2 = bArr[i2];
            if (b2 >= 0) {
                this.zzc = i4;
                return b2;
            } else if (i3 - i4 < 9) {
                return (int) zzx();
            } else {
                int i5 = i4 + 1;
                int i6 = b2 ^ (bArr[i4] << 7);
                if (i6 < 0) {
                    i = i6 ^ (-128);
                } else {
                    int i7 = i5 + 1;
                    int i8 = i6 ^ (bArr[i5] << 14);
                    if (i8 >= 0) {
                        i = i8 ^ 16256;
                    } else {
                        i5 = i7 + 1;
                        int i9 = i8 ^ (bArr[i7] << Ascii.NAK);
                        if (i9 < 0) {
                            i = i9 ^ (-2080896);
                        } else {
                            i7 = i5 + 1;
                            byte b3 = bArr[i5];
                            i = (i9 ^ (b3 << Ascii.FS)) ^ 266354560;
                            if (b3 < 0) {
                                i5 = i7 + 1;
                                if (bArr[i7] < 0) {
                                    i7 = i5 + 1;
                                    if (bArr[i5] < 0) {
                                        i5 = i7 + 1;
                                        if (bArr[i7] < 0) {
                                            i7 = i5 + 1;
                                            if (bArr[i5] < 0) {
                                                i5 = i7 + 1;
                                                if (bArr[i7] < 0) {
                                                    throw zzjk.zzc();
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    i5 = i7;
                }
                this.zzc = i5;
                return i;
            }
        }
        throw zzjk.zza();
    }

    private final long zzw() {
        long j;
        long j2;
        long j3;
        int i;
        int i2 = this.zzc;
        int i3 = this.zze;
        if (i3 != i2) {
            byte[] bArr = this.zzb;
            int i4 = i2 + 1;
            byte b2 = bArr[i2];
            if (b2 >= 0) {
                this.zzc = i4;
                return b2;
            } else if (i3 - i4 < 9) {
                return zzx();
            } else {
                int i5 = i4 + 1;
                int i6 = b2 ^ (bArr[i4] << 7);
                if (i6 >= 0) {
                    int i7 = i5 + 1;
                    int i8 = i6 ^ (bArr[i5] << 14);
                    if (i8 >= 0) {
                        i5 = i7;
                        j = i8 ^ 16256;
                    } else {
                        i5 = i7 + 1;
                        int i9 = i8 ^ (bArr[i7] << Ascii.NAK);
                        if (i9 < 0) {
                            i = i9 ^ (-2080896);
                        } else {
                            long j4 = i9;
                            int i10 = i5 + 1;
                            long j5 = j4 ^ (bArr[i5] << 28);
                            if (j5 >= 0) {
                                j3 = 266354560;
                            } else {
                                i5 = i10 + 1;
                                long j6 = j5 ^ (bArr[i10] << 35);
                                if (j6 < 0) {
                                    j2 = -34093383808L;
                                } else {
                                    i10 = i5 + 1;
                                    j5 = j6 ^ (bArr[i5] << 42);
                                    if (j5 >= 0) {
                                        j3 = 4363953127296L;
                                    } else {
                                        i5 = i10 + 1;
                                        j6 = j5 ^ (bArr[i10] << 49);
                                        if (j6 < 0) {
                                            j2 = -558586000294016L;
                                        } else {
                                            int i11 = i5 + 1;
                                            long j7 = (j6 ^ (bArr[i5] << 56)) ^ 71499008037633920L;
                                            if (j7 < 0) {
                                                i5 = i11 + 1;
                                                if (bArr[i11] < 0) {
                                                    throw zzjk.zzc();
                                                }
                                            } else {
                                                i5 = i11;
                                            }
                                            j = j7;
                                        }
                                    }
                                }
                                j = j6 ^ j2;
                            }
                            j = j5 ^ j3;
                            i5 = i10;
                        }
                    }
                    this.zzc = i5;
                    return j;
                }
                i = i6 ^ (-128);
                j = i;
                this.zzc = i5;
                return j;
            }
        }
        throw zzjk.zza();
    }

    private final long zzx() {
        long j = 0;
        for (int i = 0; i < 64; i += 7) {
            byte zzy = zzy();
            j |= (zzy & Ascii.DEL) << i;
            if ((zzy & UnsignedBytes.MAX_POWER_OF_TWO) == 0) {
                return j;
            }
        }
        throw zzjk.zzc();
    }

    private final byte zzy() {
        int i = this.zzc;
        if (i != this.zze) {
            byte[] bArr = this.zzb;
            this.zzc = i + 1;
            return bArr[i];
        }
        throw zzjk.zza();
    }

    private final int zzz() {
        zzb(4);
        return zzab();
    }

    @Override // com.google.android.gms.internal.vision.zzld
    public final int zza() {
        if (zzu()) {
            return Integer.MAX_VALUE;
        }
        int zzv = zzv();
        this.zzf = zzv;
        if (zzv == this.zzg) {
            return Integer.MAX_VALUE;
        }
        return zzv >>> 3;
    }

    @Override // com.google.android.gms.internal.vision.zzld
    public final int zzb() {
        return this.zzf;
    }

    @Override // com.google.android.gms.internal.vision.zzld
    public final boolean zzc() {
        int i;
        int i2;
        if (zzu() || (i = this.zzf) == (i2 = this.zzg)) {
            return false;
        }
        int i3 = i & 7;
        if (i3 == 0) {
            int i4 = this.zze;
            int i5 = this.zzc;
            if (i4 - i5 >= 10) {
                byte[] bArr = this.zzb;
                int i6 = 0;
                while (i6 < 10) {
                    int i7 = i5 + 1;
                    if (bArr[i5] >= 0) {
                        this.zzc = i7;
                        break;
                    }
                    i6++;
                    i5 = i7;
                }
            }
            for (int i8 = 0; i8 < 10; i8++) {
                if (zzy() >= 0) {
                    return true;
                }
            }
            throw zzjk.zzc();
        } else if (i3 == 1) {
            zza(8);
            return true;
        } else if (i3 == 2) {
            zza(zzv());
            return true;
        } else if (i3 != 3) {
            if (i3 == 5) {
                zza(4);
                return true;
            }
            throw zzjk.zzf();
        } else {
            this.zzg = ((i >>> 3) << 3) | 4;
            while (zza() != Integer.MAX_VALUE && zzc()) {
            }
            if (this.zzf == this.zzg) {
                this.zzg = i2;
                return true;
            }
            throw zzjk.zzg();
        }
    }

    @Override // com.google.android.gms.internal.vision.zzld
    public final double zzd() {
        zzc(1);
        return Double.longBitsToDouble(zzaa());
    }

    @Override // com.google.android.gms.internal.vision.zzld
    public final float zze() {
        zzc(5);
        return Float.intBitsToFloat(zzz());
    }

    @Override // com.google.android.gms.internal.vision.zzld
    public final long zzf() {
        zzc(0);
        return zzw();
    }

    @Override // com.google.android.gms.internal.vision.zzld
    public final long zzg() {
        zzc(0);
        return zzw();
    }

    @Override // com.google.android.gms.internal.vision.zzld
    public final int zzh() {
        zzc(0);
        return zzv();
    }

    @Override // com.google.android.gms.internal.vision.zzld
    public final long zzi() {
        zzc(1);
        return zzaa();
    }

    @Override // com.google.android.gms.internal.vision.zzld
    public final int zzj() {
        zzc(5);
        return zzz();
    }

    @Override // com.google.android.gms.internal.vision.zzld
    public final boolean zzk() {
        zzc(0);
        return zzv() != 0;
    }

    @Override // com.google.android.gms.internal.vision.zzld
    public final String zzl() {
        return zza(false);
    }

    @Override // com.google.android.gms.internal.vision.zzld
    public final String zzm() {
        return zza(true);
    }

    @Override // com.google.android.gms.internal.vision.zzld
    public final zzht zzn() {
        zzht zza;
        zzc(2);
        int zzv = zzv();
        if (zzv == 0) {
            return zzht.zza;
        }
        zzb(zzv);
        if (this.zza) {
            zza = zzht.zzb(this.zzb, this.zzc, zzv);
        } else {
            zza = zzht.zza(this.zzb, this.zzc, zzv);
        }
        this.zzc += zzv;
        return zza;
    }

    @Override // com.google.android.gms.internal.vision.zzld
    public final int zzo() {
        zzc(0);
        return zzv();
    }

    @Override // com.google.android.gms.internal.vision.zzld
    public final int zzp() {
        zzc(0);
        return zzv();
    }

    @Override // com.google.android.gms.internal.vision.zzld
    public final int zzq() {
        zzc(5);
        return zzz();
    }

    @Override // com.google.android.gms.internal.vision.zzld
    public final long zzr() {
        zzc(1);
        return zzaa();
    }

    @Override // com.google.android.gms.internal.vision.zzld
    public final int zzs() {
        zzc(0);
        return zzif.zze(zzv());
    }

    @Override // com.google.android.gms.internal.vision.zzld
    public final long zzt() {
        zzc(0);
        return zzif.zza(zzw());
    }

    @Override // com.google.android.gms.internal.vision.zzld
    public final <T> T zzb(Class<T> cls, zzio zzioVar) {
        zzc(3);
        return (T) zzd(zzky.zza().zza((Class) cls), zzioVar);
    }

    @Override // com.google.android.gms.internal.vision.zzld
    public final void zzl(List<Integer> list) {
        int i;
        int i2;
        if (list instanceof zzjd) {
            zzjd zzjdVar = (zzjd) list;
            int i3 = this.zzf & 7;
            if (i3 != 0) {
                if (i3 == 2) {
                    int zzv = this.zzc + zzv();
                    while (this.zzc < zzv) {
                        zzjdVar.zzc(zzv());
                    }
                    return;
                }
                throw zzjk.zzf();
            }
            do {
                zzjdVar.zzc(zzo());
                if (zzu()) {
                    return;
                }
                i2 = this.zzc;
            } while (zzv() == this.zzf);
            this.zzc = i2;
            return;
        }
        int i4 = this.zzf & 7;
        if (i4 != 0) {
            if (i4 == 2) {
                int zzv2 = this.zzc + zzv();
                while (this.zzc < zzv2) {
                    list.add(Integer.valueOf(zzv()));
                }
                return;
            }
            throw zzjk.zzf();
        }
        do {
            list.add(Integer.valueOf(zzo()));
            if (zzu()) {
                return;
            }
            i = this.zzc;
        } while (zzv() == this.zzf);
        this.zzc = i;
    }

    @Override // com.google.android.gms.internal.vision.zzld
    public final void zzm(List<Integer> list) {
        int i;
        int i2;
        if (list instanceof zzjd) {
            zzjd zzjdVar = (zzjd) list;
            int i3 = this.zzf & 7;
            if (i3 != 0) {
                if (i3 == 2) {
                    int zzv = this.zzc + zzv();
                    while (this.zzc < zzv) {
                        zzjdVar.zzc(zzv());
                    }
                    return;
                }
                throw zzjk.zzf();
            }
            do {
                zzjdVar.zzc(zzp());
                if (zzu()) {
                    return;
                }
                i2 = this.zzc;
            } while (zzv() == this.zzf);
            this.zzc = i2;
            return;
        }
        int i4 = this.zzf & 7;
        if (i4 != 0) {
            if (i4 == 2) {
                int zzv2 = this.zzc + zzv();
                while (this.zzc < zzv2) {
                    list.add(Integer.valueOf(zzv()));
                }
                return;
            }
            throw zzjk.zzf();
        }
        do {
            list.add(Integer.valueOf(zzp()));
            if (zzu()) {
                return;
            }
            i = this.zzc;
        } while (zzv() == this.zzf);
        this.zzc = i;
    }

    private final <T> T zzd(zzlc<T> zzlcVar, zzio zzioVar) {
        int i = this.zzg;
        this.zzg = ((this.zzf >>> 3) << 3) | 4;
        try {
            T zza = zzlcVar.zza();
            zzlcVar.zza(zza, this, zzioVar);
            zzlcVar.zzc(zza);
            if (this.zzf == this.zzg) {
                return zza;
            }
            throw zzjk.zzg();
        } finally {
            this.zzg = i;
        }
    }

    @Override // com.google.android.gms.internal.vision.zzld
    public final void zze(List<Integer> list) {
        int i;
        int i2;
        if (list instanceof zzjd) {
            zzjd zzjdVar = (zzjd) list;
            int i3 = this.zzf & 7;
            if (i3 == 0) {
                do {
                    zzjdVar.zzc(zzh());
                    if (zzu()) {
                        return;
                    }
                    i2 = this.zzc;
                } while (zzv() == this.zzf);
                this.zzc = i2;
                return;
            } else if (i3 == 2) {
                int zzv = this.zzc + zzv();
                while (this.zzc < zzv) {
                    zzjdVar.zzc(zzv());
                }
                zzf(zzv);
                return;
            } else {
                throw zzjk.zzf();
            }
        }
        int i4 = this.zzf & 7;
        if (i4 == 0) {
            do {
                list.add(Integer.valueOf(zzh()));
                if (zzu()) {
                    return;
                }
                i = this.zzc;
            } while (zzv() == this.zzf);
            this.zzc = i;
        } else if (i4 == 2) {
            int zzv2 = this.zzc + zzv();
            while (this.zzc < zzv2) {
                list.add(Integer.valueOf(zzv()));
            }
            zzf(zzv2);
        } else {
            throw zzjk.zzf();
        }
    }

    @Override // com.google.android.gms.internal.vision.zzld
    public final void zzf(List<Long> list) {
        int i;
        int i2;
        if (list instanceof zzjy) {
            zzjy zzjyVar = (zzjy) list;
            int i3 = this.zzf & 7;
            if (i3 == 1) {
                do {
                    zzjyVar.zza(zzi());
                    if (zzu()) {
                        return;
                    }
                    i2 = this.zzc;
                } while (zzv() == this.zzf);
                this.zzc = i2;
                return;
            } else if (i3 == 2) {
                int zzv = zzv();
                zzd(zzv);
                int i4 = this.zzc + zzv;
                while (this.zzc < i4) {
                    zzjyVar.zza(zzac());
                }
                return;
            } else {
                throw zzjk.zzf();
            }
        }
        int i5 = this.zzf & 7;
        if (i5 == 1) {
            do {
                list.add(Long.valueOf(zzi()));
                if (zzu()) {
                    return;
                }
                i = this.zzc;
            } while (zzv() == this.zzf);
            this.zzc = i;
        } else if (i5 == 2) {
            int zzv2 = zzv();
            zzd(zzv2);
            int i6 = this.zzc + zzv2;
            while (this.zzc < i6) {
                list.add(Long.valueOf(zzac()));
            }
        } else {
            throw zzjk.zzf();
        }
    }

    @Override // com.google.android.gms.internal.vision.zzld
    public final void zzg(List<Integer> list) {
        int i;
        int i2;
        if (list instanceof zzjd) {
            zzjd zzjdVar = (zzjd) list;
            int i3 = this.zzf & 7;
            if (i3 == 2) {
                int zzv = zzv();
                zze(zzv);
                int i4 = this.zzc + zzv;
                while (this.zzc < i4) {
                    zzjdVar.zzc(zzab());
                }
                return;
            } else if (i3 == 5) {
                do {
                    zzjdVar.zzc(zzj());
                    if (zzu()) {
                        return;
                    }
                    i2 = this.zzc;
                } while (zzv() == this.zzf);
                this.zzc = i2;
                return;
            } else {
                throw zzjk.zzf();
            }
        }
        int i5 = this.zzf & 7;
        if (i5 == 2) {
            int zzv2 = zzv();
            zze(zzv2);
            int i6 = this.zzc + zzv2;
            while (this.zzc < i6) {
                list.add(Integer.valueOf(zzab()));
            }
        } else if (i5 == 5) {
            do {
                list.add(Integer.valueOf(zzj()));
                if (zzu()) {
                    return;
                }
                i = this.zzc;
            } while (zzv() == this.zzf);
            this.zzc = i;
        } else {
            throw zzjk.zzf();
        }
    }

    @Override // com.google.android.gms.internal.vision.zzld
    public final void zzh(List<Boolean> list) {
        int i;
        int i2;
        if (list instanceof zzhr) {
            zzhr zzhrVar = (zzhr) list;
            int i3 = this.zzf & 7;
            if (i3 != 0) {
                if (i3 == 2) {
                    int zzv = this.zzc + zzv();
                    while (this.zzc < zzv) {
                        zzhrVar.zza(zzv() != 0);
                    }
                    zzf(zzv);
                    return;
                }
                throw zzjk.zzf();
            }
            do {
                zzhrVar.zza(zzk());
                if (zzu()) {
                    return;
                }
                i2 = this.zzc;
            } while (zzv() == this.zzf);
            this.zzc = i2;
            return;
        }
        int i4 = this.zzf & 7;
        if (i4 != 0) {
            if (i4 == 2) {
                int zzv2 = this.zzc + zzv();
                while (this.zzc < zzv2) {
                    list.add(Boolean.valueOf(zzv() != 0));
                }
                zzf(zzv2);
                return;
            }
            throw zzjk.zzf();
        }
        do {
            list.add(Boolean.valueOf(zzk()));
            if (zzu()) {
                return;
            }
            i = this.zzc;
        } while (zzv() == this.zzf);
        this.zzc = i;
    }

    @Override // com.google.android.gms.internal.vision.zzld
    public final void zzi(List<String> list) {
        zza(list, false);
    }

    @Override // com.google.android.gms.internal.vision.zzld
    public final void zzj(List<String> list) {
        zza(list, true);
    }

    @Override // com.google.android.gms.internal.vision.zzld
    public final void zzk(List<zzht> list) {
        int i;
        if ((this.zzf & 7) == 2) {
            do {
                list.add(zzn());
                if (zzu()) {
                    return;
                }
                i = this.zzc;
            } while (zzv() == this.zzf);
            this.zzc = i;
            return;
        }
        throw zzjk.zzf();
    }

    @Override // com.google.android.gms.internal.vision.zzld
    public final void zzo(List<Long> list) {
        int i;
        int i2;
        if (list instanceof zzjy) {
            zzjy zzjyVar = (zzjy) list;
            int i3 = this.zzf & 7;
            if (i3 == 1) {
                do {
                    zzjyVar.zza(zzr());
                    if (zzu()) {
                        return;
                    }
                    i2 = this.zzc;
                } while (zzv() == this.zzf);
                this.zzc = i2;
                return;
            } else if (i3 == 2) {
                int zzv = zzv();
                zzd(zzv);
                int i4 = this.zzc + zzv;
                while (this.zzc < i4) {
                    zzjyVar.zza(zzac());
                }
                return;
            } else {
                throw zzjk.zzf();
            }
        }
        int i5 = this.zzf & 7;
        if (i5 == 1) {
            do {
                list.add(Long.valueOf(zzr()));
                if (zzu()) {
                    return;
                }
                i = this.zzc;
            } while (zzv() == this.zzf);
            this.zzc = i;
        } else if (i5 == 2) {
            int zzv2 = zzv();
            zzd(zzv2);
            int i6 = this.zzc + zzv2;
            while (this.zzc < i6) {
                list.add(Long.valueOf(zzac()));
            }
        } else {
            throw zzjk.zzf();
        }
    }

    @Override // com.google.android.gms.internal.vision.zzld
    public final void zzp(List<Integer> list) {
        int i;
        int i2;
        if (list instanceof zzjd) {
            zzjd zzjdVar = (zzjd) list;
            int i3 = this.zzf & 7;
            if (i3 != 0) {
                if (i3 == 2) {
                    int zzv = this.zzc + zzv();
                    while (this.zzc < zzv) {
                        zzjdVar.zzc(zzif.zze(zzv()));
                    }
                    return;
                }
                throw zzjk.zzf();
            }
            do {
                zzjdVar.zzc(zzs());
                if (zzu()) {
                    return;
                }
                i2 = this.zzc;
            } while (zzv() == this.zzf);
            this.zzc = i2;
            return;
        }
        int i4 = this.zzf & 7;
        if (i4 != 0) {
            if (i4 == 2) {
                int zzv2 = this.zzc + zzv();
                while (this.zzc < zzv2) {
                    list.add(Integer.valueOf(zzif.zze(zzv())));
                }
                return;
            }
            throw zzjk.zzf();
        }
        do {
            list.add(Integer.valueOf(zzs()));
            if (zzu()) {
                return;
            }
            i = this.zzc;
        } while (zzv() == this.zzf);
        this.zzc = i;
    }

    @Override // com.google.android.gms.internal.vision.zzld
    public final void zzq(List<Long> list) {
        int i;
        int i2;
        if (list instanceof zzjy) {
            zzjy zzjyVar = (zzjy) list;
            int i3 = this.zzf & 7;
            if (i3 != 0) {
                if (i3 == 2) {
                    int zzv = this.zzc + zzv();
                    while (this.zzc < zzv) {
                        zzjyVar.zza(zzif.zza(zzw()));
                    }
                    return;
                }
                throw zzjk.zzf();
            }
            do {
                zzjyVar.zza(zzt());
                if (zzu()) {
                    return;
                }
                i2 = this.zzc;
            } while (zzv() == this.zzf);
            this.zzc = i2;
            return;
        }
        int i4 = this.zzf & 7;
        if (i4 != 0) {
            if (i4 == 2) {
                int zzv2 = this.zzc + zzv();
                while (this.zzc < zzv2) {
                    list.add(Long.valueOf(zzif.zza(zzw())));
                }
                return;
            }
            throw zzjk.zzf();
        }
        do {
            list.add(Long.valueOf(zzt()));
            if (zzu()) {
                return;
            }
            i = this.zzc;
        } while (zzv() == this.zzf);
        this.zzc = i;
    }

    private final String zza(boolean z) {
        zzc(2);
        int zzv = zzv();
        if (zzv == 0) {
            return "";
        }
        zzb(zzv);
        if (z) {
            byte[] bArr = this.zzb;
            int i = this.zzc;
            if (!zzmd.zza(bArr, i, i + zzv)) {
                throw zzjk.zzh();
            }
        }
        String str = new String(this.zzb, this.zzc, zzv, zzjf.zza);
        this.zzc += zzv;
        return str;
    }

    @Override // com.google.android.gms.internal.vision.zzld
    public final <T> T zzb(zzlc<T> zzlcVar, zzio zzioVar) {
        zzc(3);
        return (T) zzd(zzlcVar, zzioVar);
    }

    @Override // com.google.android.gms.internal.vision.zzld
    public final void zzb(List<Float> list) {
        int i;
        int i2;
        if (list instanceof zzja) {
            zzja zzjaVar = (zzja) list;
            int i3 = this.zzf & 7;
            if (i3 == 2) {
                int zzv = zzv();
                zze(zzv);
                int i4 = this.zzc + zzv;
                while (this.zzc < i4) {
                    zzjaVar.zza(Float.intBitsToFloat(zzab()));
                }
                return;
            } else if (i3 == 5) {
                do {
                    zzjaVar.zza(zze());
                    if (zzu()) {
                        return;
                    }
                    i2 = this.zzc;
                } while (zzv() == this.zzf);
                this.zzc = i2;
                return;
            } else {
                throw zzjk.zzf();
            }
        }
        int i5 = this.zzf & 7;
        if (i5 == 2) {
            int zzv2 = zzv();
            zze(zzv2);
            int i6 = this.zzc + zzv2;
            while (this.zzc < i6) {
                list.add(Float.valueOf(Float.intBitsToFloat(zzab())));
            }
        } else if (i5 == 5) {
            do {
                list.add(Float.valueOf(zze()));
                if (zzu()) {
                    return;
                }
                i = this.zzc;
            } while (zzv() == this.zzf);
            this.zzc = i;
        } else {
            throw zzjk.zzf();
        }
    }

    @Override // com.google.android.gms.internal.vision.zzld
    public final void zzn(List<Integer> list) {
        int i;
        int i2;
        if (list instanceof zzjd) {
            zzjd zzjdVar = (zzjd) list;
            int i3 = this.zzf & 7;
            if (i3 == 2) {
                int zzv = zzv();
                zze(zzv);
                int i4 = this.zzc + zzv;
                while (this.zzc < i4) {
                    zzjdVar.zzc(zzab());
                }
                return;
            } else if (i3 == 5) {
                do {
                    zzjdVar.zzc(zzq());
                    if (zzu()) {
                        return;
                    }
                    i2 = this.zzc;
                } while (zzv() == this.zzf);
                this.zzc = i2;
                return;
            } else {
                throw zzjk.zzf();
            }
        }
        int i5 = this.zzf & 7;
        if (i5 == 2) {
            int zzv2 = zzv();
            zze(zzv2);
            int i6 = this.zzc + zzv2;
            while (this.zzc < i6) {
                list.add(Integer.valueOf(zzab()));
            }
        } else if (i5 == 5) {
            do {
                list.add(Integer.valueOf(zzq()));
                if (zzu()) {
                    return;
                }
                i = this.zzc;
            } while (zzv() == this.zzf);
            this.zzc = i;
        } else {
            throw zzjk.zzf();
        }
    }

    @Override // com.google.android.gms.internal.vision.zzld
    public final <T> T zza(Class<T> cls, zzio zzioVar) {
        zzc(2);
        return (T) zzc(zzky.zza().zza((Class) cls), zzioVar);
    }

    @Override // com.google.android.gms.internal.vision.zzld
    public final <T> T zza(zzlc<T> zzlcVar, zzio zzioVar) {
        zzc(2);
        return (T) zzc(zzlcVar, zzioVar);
    }

    @Override // com.google.android.gms.internal.vision.zzld
    public final void zzd(List<Long> list) {
        int i;
        int i2;
        if (list instanceof zzjy) {
            zzjy zzjyVar = (zzjy) list;
            int i3 = this.zzf & 7;
            if (i3 == 0) {
                do {
                    zzjyVar.zza(zzg());
                    if (zzu()) {
                        return;
                    }
                    i2 = this.zzc;
                } while (zzv() == this.zzf);
                this.zzc = i2;
                return;
            } else if (i3 == 2) {
                int zzv = this.zzc + zzv();
                while (this.zzc < zzv) {
                    zzjyVar.zza(zzw());
                }
                zzf(zzv);
                return;
            } else {
                throw zzjk.zzf();
            }
        }
        int i4 = this.zzf & 7;
        if (i4 == 0) {
            do {
                list.add(Long.valueOf(zzg()));
                if (zzu()) {
                    return;
                }
                i = this.zzc;
            } while (zzv() == this.zzf);
            this.zzc = i;
        } else if (i4 == 2) {
            int zzv2 = this.zzc + zzv();
            while (this.zzc < zzv2) {
                list.add(Long.valueOf(zzw()));
            }
            zzf(zzv2);
        } else {
            throw zzjk.zzf();
        }
    }

    @Override // com.google.android.gms.internal.vision.zzld
    public final void zza(List<Double> list) {
        int i;
        int i2;
        if (list instanceof zzin) {
            zzin zzinVar = (zzin) list;
            int i3 = this.zzf & 7;
            if (i3 == 1) {
                do {
                    zzinVar.zza(zzd());
                    if (zzu()) {
                        return;
                    }
                    i2 = this.zzc;
                } while (zzv() == this.zzf);
                this.zzc = i2;
                return;
            } else if (i3 == 2) {
                int zzv = zzv();
                zzd(zzv);
                int i4 = this.zzc + zzv;
                while (this.zzc < i4) {
                    zzinVar.zza(Double.longBitsToDouble(zzac()));
                }
                return;
            } else {
                throw zzjk.zzf();
            }
        }
        int i5 = this.zzf & 7;
        if (i5 == 1) {
            do {
                list.add(Double.valueOf(zzd()));
                if (zzu()) {
                    return;
                }
                i = this.zzc;
            } while (zzv() == this.zzf);
            this.zzc = i;
        } else if (i5 == 2) {
            int zzv2 = zzv();
            zzd(zzv2);
            int i6 = this.zzc + zzv2;
            while (this.zzc < i6) {
                list.add(Double.valueOf(Double.longBitsToDouble(zzac())));
            }
        } else {
            throw zzjk.zzf();
        }
    }

    private final <T> T zzc(zzlc<T> zzlcVar, zzio zzioVar) {
        int zzv = zzv();
        zzb(zzv);
        int i = this.zze;
        int i2 = this.zzc + zzv;
        this.zze = i2;
        try {
            T zza = zzlcVar.zza();
            zzlcVar.zza(zza, this, zzioVar);
            zzlcVar.zzc(zza);
            if (this.zzc == i2) {
                return zza;
            }
            throw zzjk.zzg();
        } finally {
            this.zze = i;
        }
    }

    @Override // com.google.android.gms.internal.vision.zzld
    public final void zzc(List<Long> list) {
        int i;
        int i2;
        if (list instanceof zzjy) {
            zzjy zzjyVar = (zzjy) list;
            int i3 = this.zzf & 7;
            if (i3 == 0) {
                do {
                    zzjyVar.zza(zzf());
                    if (zzu()) {
                        return;
                    }
                    i2 = this.zzc;
                } while (zzv() == this.zzf);
                this.zzc = i2;
                return;
            } else if (i3 == 2) {
                int zzv = this.zzc + zzv();
                while (this.zzc < zzv) {
                    zzjyVar.zza(zzw());
                }
                zzf(zzv);
                return;
            } else {
                throw zzjk.zzf();
            }
        }
        int i4 = this.zzf & 7;
        if (i4 == 0) {
            do {
                list.add(Long.valueOf(zzf()));
                if (zzu()) {
                    return;
                }
                i = this.zzc;
            } while (zzv() == this.zzf);
            this.zzc = i;
        } else if (i4 == 2) {
            int zzv2 = this.zzc + zzv();
            while (this.zzc < zzv2) {
                list.add(Long.valueOf(zzw()));
            }
            zzf(zzv2);
        } else {
            throw zzjk.zzf();
        }
    }

    private final void zze(int i) {
        zzb(i);
        if ((i & 3) != 0) {
            throw zzjk.zzg();
        }
    }

    private final void zzf(int i) {
        if (this.zzc != i) {
            throw zzjk.zza();
        }
    }

    /* JADX DEBUG: Multi-variable search result rejected for r4v0, resolved type: java.util.List<T> */
    /* JADX WARN: Multi-variable type inference failed */
    @Override // com.google.android.gms.internal.vision.zzld
    public final <T> void zzb(List<T> list, zzlc<T> zzlcVar, zzio zzioVar) {
        int i;
        int i2 = this.zzf;
        if ((i2 & 7) == 3) {
            do {
                list.add(zzd(zzlcVar, zzioVar));
                if (zzu()) {
                    return;
                }
                i = this.zzc;
            } while (zzv() == i2);
            this.zzc = i;
            return;
        }
        throw zzjk.zzf();
    }

    private final void zzb(int i) {
        if (i < 0 || i > this.zze - this.zzc) {
            throw zzjk.zza();
        }
    }

    private final void zzd(int i) {
        zzb(i);
        if ((i & 7) != 0) {
            throw zzjk.zzg();
        }
    }

    private final void zza(List<String> list, boolean z) {
        int i;
        int i2;
        if ((this.zzf & 7) == 2) {
            if ((list instanceof zzjv) && !z) {
                zzjv zzjvVar = (zzjv) list;
                do {
                    zzjvVar.zza(zzn());
                    if (zzu()) {
                        return;
                    }
                    i2 = this.zzc;
                } while (zzv() == this.zzf);
                this.zzc = i2;
                return;
            }
            do {
                list.add(zza(z));
                if (zzu()) {
                    return;
                }
                i = this.zzc;
            } while (zzv() == this.zzf);
            this.zzc = i;
            return;
        }
        throw zzjk.zzf();
    }

    private final void zzc(int i) {
        if ((this.zzf & 7) != i) {
            throw zzjk.zzf();
        }
    }

    /* JADX DEBUG: Multi-variable search result rejected for r4v0, resolved type: java.util.List<T> */
    /* JADX WARN: Multi-variable type inference failed */
    @Override // com.google.android.gms.internal.vision.zzld
    public final <T> void zza(List<T> list, zzlc<T> zzlcVar, zzio zzioVar) {
        int i;
        int i2 = this.zzf;
        if ((i2 & 7) == 2) {
            do {
                list.add(zzc(zzlcVar, zzioVar));
                if (zzu()) {
                    return;
                }
                i = this.zzc;
            } while (zzv() == i2);
            this.zzc = i;
            return;
        }
        throw zzjk.zzf();
    }

    /* JADX DEBUG: Multi-variable search result rejected for r8v0, resolved type: java.util.Map<K, V> */
    /* JADX WARN: Multi-variable type inference failed */
    @Override // com.google.android.gms.internal.vision.zzld
    public final <K, V> void zza(Map<K, V> map, zzkf<K, V> zzkfVar, zzio zzioVar) {
        zzc(2);
        int zzv = zzv();
        zzb(zzv);
        int i = this.zze;
        this.zze = this.zzc + zzv;
        try {
            Object obj = zzkfVar.zzb;
            Object obj2 = zzkfVar.zzd;
            while (true) {
                int zza = zza();
                if (zza == Integer.MAX_VALUE) {
                    map.put(obj, obj2);
                    return;
                } else if (zza == 1) {
                    obj = zza(zzkfVar.zza, (Class<?>) null, (zzio) null);
                } else if (zza != 2) {
                    try {
                        if (!zzc()) {
                            throw new zzjk("Unable to parse map entry.");
                            break;
                        }
                    } catch (zzjn unused) {
                        if (!zzc()) {
                            throw new zzjk("Unable to parse map entry.");
                        }
                    }
                } else {
                    obj2 = zza(zzkfVar.zzc, zzkfVar.zzd.getClass(), zzioVar);
                }
            }
        } finally {
            this.zze = i;
        }
    }

    private final Object zza(zzml zzmlVar, Class<?> cls, zzio zzioVar) {
        switch (zzhp.zza[zzmlVar.ordinal()]) {
            case 1:
                return Boolean.valueOf(zzk());
            case 2:
                return zzn();
            case 3:
                return Double.valueOf(zzd());
            case 4:
                return Integer.valueOf(zzp());
            case 5:
                return Integer.valueOf(zzj());
            case 6:
                return Long.valueOf(zzi());
            case 7:
                return Float.valueOf(zze());
            case 8:
                return Integer.valueOf(zzh());
            case 9:
                return Long.valueOf(zzg());
            case 10:
                return zza(cls, zzioVar);
            case 11:
                return Integer.valueOf(zzq());
            case 12:
                return Long.valueOf(zzr());
            case 13:
                return Integer.valueOf(zzs());
            case 14:
                return Long.valueOf(zzt());
            case 15:
                return zza(true);
            case 16:
                return Integer.valueOf(zzo());
            case 17:
                return Long.valueOf(zzf());
            default:
                throw new RuntimeException("unsupported field type.");
        }
    }

    private final void zza(int i) {
        zzb(i);
        this.zzc += i;
    }
}