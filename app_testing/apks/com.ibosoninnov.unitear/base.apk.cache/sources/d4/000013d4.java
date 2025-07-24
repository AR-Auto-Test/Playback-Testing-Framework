package com.google.android.gms.internal.vision;

import java.io.IOException;
import java.util.Objects;
import java.util.logging.Level;
import java.util.logging.Logger;

/* compiled from: com.google.android.gms:play-services-vision-common@@19.1.3 */
/* loaded from: classes.dex */
public abstract class zzii extends zzhq {
    private static final Logger zzb = Logger.getLogger(zzii.class.getName());
    private static final boolean zzc = zzma.zza();
    public zzil zza;

    /* compiled from: com.google.android.gms:play-services-vision-common@@19.1.3 */
    /* loaded from: classes.dex */
    public static class zza extends zzii {
        private final byte[] zzb;
        private final int zzc;
        private final int zzd;
        private int zze;

        public zza(byte[] bArr, int i, int i2) {
            super();
            Objects.requireNonNull(bArr, "buffer");
            if ((i2 | 0 | (bArr.length - i2)) >= 0) {
                this.zzb = bArr;
                this.zzc = 0;
                this.zze = 0;
                this.zzd = i2;
                return;
            }
            throw new IllegalArgumentException(String.format("Array range is invalid. Buffer.length=%d, offset=%d, length=%d", Integer.valueOf(bArr.length), 0, Integer.valueOf(i2)));
        }

        @Override // com.google.android.gms.internal.vision.zzii
        public final void zza(int i, int i2) {
            zzb((i << 3) | i2);
        }

        @Override // com.google.android.gms.internal.vision.zzii
        public final void zzb(int i, int i2) {
            zza(i, 0);
            zza(i2);
        }

        @Override // com.google.android.gms.internal.vision.zzii
        public final void zzc(int i, int i2) {
            zza(i, 0);
            zzb(i2);
        }

        @Override // com.google.android.gms.internal.vision.zzii
        public final void zzd(int i) {
            try {
                byte[] bArr = this.zzb;
                int i2 = this.zze;
                int i3 = i2 + 1;
                this.zze = i3;
                bArr[i2] = (byte) i;
                int i4 = i3 + 1;
                this.zze = i4;
                bArr[i3] = (byte) (i >> 8);
                int i5 = i4 + 1;
                this.zze = i5;
                bArr[i4] = (byte) (i >> 16);
                this.zze = i5 + 1;
                bArr[i5] = (byte) (i >>> 24);
            } catch (IndexOutOfBoundsException e2) {
                throw new zzb(String.format("Pos: %d, limit: %d, len: %d", Integer.valueOf(this.zze), Integer.valueOf(this.zzd), 1), e2);
            }
        }

        @Override // com.google.android.gms.internal.vision.zzii
        public final void zze(int i, int i2) {
            zza(i, 5);
            zzd(i2);
        }

        @Override // com.google.android.gms.internal.vision.zzii
        public final void zza(int i, long j) {
            zza(i, 0);
            zza(j);
        }

        @Override // com.google.android.gms.internal.vision.zzii
        public final void zzb(byte[] bArr, int i, int i2) {
            zzb(i2);
            zzc(bArr, 0, i2);
        }

        @Override // com.google.android.gms.internal.vision.zzii
        public final void zzc(int i, long j) {
            zza(i, 1);
            zzc(j);
        }

        @Override // com.google.android.gms.internal.vision.zzii
        public final void zza(int i, boolean z) {
            zza(i, 0);
            zza(z ? (byte) 1 : (byte) 0);
        }

        @Override // com.google.android.gms.internal.vision.zzii
        public final void zzb(int i, zzht zzhtVar) {
            zza(1, 3);
            zzc(2, i);
            zza(3, zzhtVar);
            zza(1, 4);
        }

        @Override // com.google.android.gms.internal.vision.zzii
        public final void zzc(long j) {
            try {
                byte[] bArr = this.zzb;
                int i = this.zze;
                int i2 = i + 1;
                this.zze = i2;
                bArr[i] = (byte) j;
                int i3 = i2 + 1;
                this.zze = i3;
                bArr[i2] = (byte) (j >> 8);
                int i4 = i3 + 1;
                this.zze = i4;
                bArr[i3] = (byte) (j >> 16);
                int i5 = i4 + 1;
                this.zze = i5;
                bArr[i4] = (byte) (j >> 24);
                int i6 = i5 + 1;
                this.zze = i6;
                bArr[i5] = (byte) (j >> 32);
                int i7 = i6 + 1;
                this.zze = i7;
                bArr[i6] = (byte) (j >> 40);
                int i8 = i7 + 1;
                this.zze = i8;
                bArr[i7] = (byte) (j >> 48);
                this.zze = i8 + 1;
                bArr[i8] = (byte) (j >> 56);
            } catch (IndexOutOfBoundsException e2) {
                throw new zzb(String.format("Pos: %d, limit: %d, len: %d", Integer.valueOf(this.zze), Integer.valueOf(this.zzd), 1), e2);
            }
        }

        @Override // com.google.android.gms.internal.vision.zzii
        public final void zza(int i, String str) {
            zza(i, 2);
            zza(str);
        }

        @Override // com.google.android.gms.internal.vision.zzii
        public final void zza(int i, zzht zzhtVar) {
            zza(i, 2);
            zza(zzhtVar);
        }

        @Override // com.google.android.gms.internal.vision.zzii
        public final void zzb(int i) {
            if (!zzii.zzc || zzhi.zza() || zza() < 5) {
                while ((i & (-128)) != 0) {
                    try {
                        byte[] bArr = this.zzb;
                        int i2 = this.zze;
                        this.zze = i2 + 1;
                        bArr[i2] = (byte) ((i & 127) | 128);
                        i >>>= 7;
                    } catch (IndexOutOfBoundsException e2) {
                        throw new zzb(String.format("Pos: %d, limit: %d, len: %d", Integer.valueOf(this.zze), Integer.valueOf(this.zzd), 1), e2);
                    }
                }
                byte[] bArr2 = this.zzb;
                int i3 = this.zze;
                this.zze = i3 + 1;
                bArr2[i3] = (byte) i;
            } else if ((i & (-128)) == 0) {
                byte[] bArr3 = this.zzb;
                int i4 = this.zze;
                this.zze = i4 + 1;
                zzma.zza(bArr3, i4, (byte) i);
            } else {
                byte[] bArr4 = this.zzb;
                int i5 = this.zze;
                this.zze = i5 + 1;
                zzma.zza(bArr4, i5, (byte) (i | 128));
                int i6 = i >>> 7;
                if ((i6 & (-128)) == 0) {
                    byte[] bArr5 = this.zzb;
                    int i7 = this.zze;
                    this.zze = i7 + 1;
                    zzma.zza(bArr5, i7, (byte) i6);
                    return;
                }
                byte[] bArr6 = this.zzb;
                int i8 = this.zze;
                this.zze = i8 + 1;
                zzma.zza(bArr6, i8, (byte) (i6 | 128));
                int i9 = i6 >>> 7;
                if ((i9 & (-128)) == 0) {
                    byte[] bArr7 = this.zzb;
                    int i10 = this.zze;
                    this.zze = i10 + 1;
                    zzma.zza(bArr7, i10, (byte) i9);
                    return;
                }
                byte[] bArr8 = this.zzb;
                int i11 = this.zze;
                this.zze = i11 + 1;
                zzma.zza(bArr8, i11, (byte) (i9 | 128));
                int i12 = i9 >>> 7;
                if ((i12 & (-128)) == 0) {
                    byte[] bArr9 = this.zzb;
                    int i13 = this.zze;
                    this.zze = i13 + 1;
                    zzma.zza(bArr9, i13, (byte) i12);
                    return;
                }
                byte[] bArr10 = this.zzb;
                int i14 = this.zze;
                this.zze = i14 + 1;
                zzma.zza(bArr10, i14, (byte) (i12 | 128));
                byte[] bArr11 = this.zzb;
                int i15 = this.zze;
                this.zze = i15 + 1;
                zzma.zza(bArr11, i15, (byte) (i12 >>> 7));
            }
        }

        @Override // com.google.android.gms.internal.vision.zzii
        public final void zza(zzht zzhtVar) {
            zzb(zzhtVar.zza());
            zzhtVar.zza(this);
        }

        @Override // com.google.android.gms.internal.vision.zzii
        public final void zza(int i, zzkk zzkkVar, zzlc zzlcVar) {
            zza(i, 2);
            zzhf zzhfVar = (zzhf) zzkkVar;
            int zzi = zzhfVar.zzi();
            if (zzi == -1) {
                zzi = zzlcVar.zzb(zzhfVar);
                zzhfVar.zzb(zzi);
            }
            zzb(zzi);
            zzlcVar.zza((zzlc) zzkkVar, (zzmr) this.zza);
        }

        private final void zzc(byte[] bArr, int i, int i2) {
            try {
                System.arraycopy(bArr, i, this.zzb, this.zze, i2);
                this.zze += i2;
            } catch (IndexOutOfBoundsException e2) {
                throw new zzb(String.format("Pos: %d, limit: %d, len: %d", Integer.valueOf(this.zze), Integer.valueOf(this.zzd), Integer.valueOf(i2)), e2);
            }
        }

        @Override // com.google.android.gms.internal.vision.zzii
        public final void zza(int i, zzkk zzkkVar) {
            zza(1, 3);
            zzc(2, i);
            zza(3, 2);
            zza(zzkkVar);
            zza(1, 4);
        }

        @Override // com.google.android.gms.internal.vision.zzii
        public final void zza(zzkk zzkkVar) {
            zzb(zzkkVar.zzm());
            zzkkVar.zza(this);
        }

        @Override // com.google.android.gms.internal.vision.zzii
        public final void zza(byte b2) {
            try {
                byte[] bArr = this.zzb;
                int i = this.zze;
                this.zze = i + 1;
                bArr[i] = b2;
            } catch (IndexOutOfBoundsException e2) {
                throw new zzb(String.format("Pos: %d, limit: %d, len: %d", Integer.valueOf(this.zze), Integer.valueOf(this.zzd), 1), e2);
            }
        }

        @Override // com.google.android.gms.internal.vision.zzii
        public final void zza(int i) {
            if (i >= 0) {
                zzb(i);
            } else {
                zza(i);
            }
        }

        @Override // com.google.android.gms.internal.vision.zzii
        public final void zza(long j) {
            if (zzii.zzc && zza() >= 10) {
                while ((j & (-128)) != 0) {
                    byte[] bArr = this.zzb;
                    int i = this.zze;
                    this.zze = i + 1;
                    zzma.zza(bArr, i, (byte) ((((int) j) & 127) | 128));
                    j >>>= 7;
                }
                byte[] bArr2 = this.zzb;
                int i2 = this.zze;
                this.zze = i2 + 1;
                zzma.zza(bArr2, i2, (byte) j);
                return;
            }
            while ((j & (-128)) != 0) {
                try {
                    byte[] bArr3 = this.zzb;
                    int i3 = this.zze;
                    this.zze = i3 + 1;
                    bArr3[i3] = (byte) ((((int) j) & 127) | 128);
                    j >>>= 7;
                } catch (IndexOutOfBoundsException e2) {
                    throw new zzb(String.format("Pos: %d, limit: %d, len: %d", Integer.valueOf(this.zze), Integer.valueOf(this.zzd), 1), e2);
                }
            }
            byte[] bArr4 = this.zzb;
            int i4 = this.zze;
            this.zze = i4 + 1;
            bArr4[i4] = (byte) j;
        }

        @Override // com.google.android.gms.internal.vision.zzhq
        public final void zza(byte[] bArr, int i, int i2) {
            zzc(bArr, i, i2);
        }

        @Override // com.google.android.gms.internal.vision.zzii
        public final void zza(String str) {
            int i = this.zze;
            try {
                int zzg = zzii.zzg(str.length() * 3);
                int zzg2 = zzii.zzg(str.length());
                if (zzg2 == zzg) {
                    int i2 = i + zzg2;
                    this.zze = i2;
                    int zza = zzmd.zza(str, this.zzb, i2, zza());
                    this.zze = i;
                    zzb((zza - i) - zzg2);
                    this.zze = zza;
                    return;
                }
                zzb(zzmd.zza(str));
                this.zze = zzmd.zza(str, this.zzb, this.zze, zza());
            } catch (zzmg e2) {
                this.zze = i;
                zza(str, e2);
            } catch (IndexOutOfBoundsException e3) {
                throw new zzb(e3);
            }
        }

        @Override // com.google.android.gms.internal.vision.zzii
        public final int zza() {
            return this.zzd - this.zze;
        }
    }

    /* compiled from: com.google.android.gms:play-services-vision-common@@19.1.3 */
    /* loaded from: classes.dex */
    public static class zzb extends IOException {
        public zzb() {
            super("CodedOutputStream was writing to a flat byte array and ran out of space.");
        }

        public zzb(Throwable th) {
            super("CodedOutputStream was writing to a flat byte array and ran out of space.", th);
        }

        /* JADX WARN: Illegal instructions before constructor call */
        /*
            Code decompiled incorrectly, please refer to instructions dump.
        */
        public zzb(String str, Throwable th) {
            super(r3.length() != 0 ? "CodedOutputStream was writing to a flat byte array and ran out of space.: ".concat(r3) : new String("CodedOutputStream was writing to a flat byte array and ran out of space.: "), th);
            String valueOf = String.valueOf(str);
        }
    }

    private zzii() {
    }

    public static zzii zza(byte[] bArr) {
        return new zza(bArr, 0, bArr.length);
    }

    public static int zzb(double d2) {
        return 8;
    }

    public static int zzb(float f2) {
        return 4;
    }

    public static int zzb(boolean z) {
        return 1;
    }

    public static int zze(int i, long j) {
        return zze(j) + zzg(i << 3);
    }

    public static int zze(long j) {
        int i;
        if (((-128) & j) == 0) {
            return 1;
        }
        if (j < 0) {
            return 10;
        }
        if (((-34359738368L) & j) != 0) {
            i = 6;
            j >>>= 28;
        } else {
            i = 2;
        }
        if (((-2097152) & j) != 0) {
            i += 2;
            j >>>= 14;
        }
        return (j & (-16384)) != 0 ? i + 1 : i;
    }

    public static int zzf(int i, int i2) {
        return zzf(i2) + zzg(i << 3);
    }

    public static int zzg(int i) {
        if ((i & (-128)) == 0) {
            return 1;
        }
        if ((i & (-16384)) == 0) {
            return 2;
        }
        if (((-2097152) & i) == 0) {
            return 3;
        }
        return (i & (-268435456)) == 0 ? 4 : 5;
    }

    public static int zzg(int i, int i2) {
        return zzg(i2) + zzg(i << 3);
    }

    public static int zzg(long j) {
        return 8;
    }

    public static int zzh(int i, int i2) {
        return zzg(zzm(i2)) + zzg(i << 3);
    }

    public static int zzh(long j) {
        return 8;
    }

    public static int zzi(int i) {
        return 4;
    }

    public static int zzi(int i, int i2) {
        return zzg(i << 3) + 4;
    }

    private static long zzi(long j) {
        return (j >> 63) ^ (j << 1);
    }

    public static int zzj(int i) {
        return 4;
    }

    public static int zzj(int i, int i2) {
        return zzg(i << 3) + 4;
    }

    public static int zzk(int i, int i2) {
        return zzf(i2) + zzg(i << 3);
    }

    @Deprecated
    public static int zzl(int i) {
        return zzg(i);
    }

    private static int zzm(int i) {
        return (i >> 31) ^ (i << 1);
    }

    public abstract int zza();

    public abstract void zza(byte b2);

    public abstract void zza(int i);

    public abstract void zza(int i, int i2);

    public abstract void zza(int i, long j);

    public abstract void zza(int i, zzht zzhtVar);

    public abstract void zza(int i, zzkk zzkkVar);

    public abstract void zza(int i, zzkk zzkkVar, zzlc zzlcVar);

    public abstract void zza(int i, String str);

    public abstract void zza(int i, boolean z);

    public abstract void zza(long j);

    public abstract void zza(zzht zzhtVar);

    public abstract void zza(zzkk zzkkVar);

    public abstract void zza(String str);

    public abstract void zzb(int i);

    public abstract void zzb(int i, int i2);

    public final void zzb(int i, long j) {
        zza(i, zzi(j));
    }

    public abstract void zzb(int i, zzht zzhtVar);

    public abstract void zzb(byte[] bArr, int i, int i2);

    public final void zzc(int i) {
        zzb(zzm(i));
    }

    public abstract void zzc(int i, int i2);

    public abstract void zzc(int i, long j);

    public abstract void zzc(long j);

    public abstract void zzd(int i);

    public final void zzd(int i, int i2) {
        zzc(i, zzm(i2));
    }

    public abstract void zze(int i, int i2);

    public static int zzc(int i, zzht zzhtVar) {
        int zzg = zzg(i << 3);
        int zza2 = zzhtVar.zza();
        return zzg(zza2) + zza2 + zzg;
    }

    public static int zzd(int i, long j) {
        return zze(j) + zzg(i << 3);
    }

    public final void zzb(long j) {
        zza(zzi(j));
    }

    public static int zzb(int i, float f2) {
        return zzg(i << 3) + 4;
    }

    public static int zze(int i) {
        return zzg(i << 3);
    }

    public static int zzf(int i, long j) {
        return zze(zzi(j)) + zzg(i << 3);
    }

    public static int zzg(int i, long j) {
        return zzg(i << 3) + 8;
    }

    public static int zzh(int i, long j) {
        return zzg(i << 3) + 8;
    }

    public static int zzk(int i) {
        return zzf(i);
    }

    public final void zza(int i, float f2) {
        zze(i, Float.floatToRawIntBits(f2));
    }

    public static int zzb(int i, double d2) {
        return zzg(i << 3) + 8;
    }

    public static int zzd(int i, zzht zzhtVar) {
        int zzg = zzg(2, i);
        return zzc(3, zzhtVar) + zzg + (zzg(8) << 1);
    }

    public static int zzh(int i) {
        return zzg(zzm(i));
    }

    public final void zza(int i, double d2) {
        zzc(i, Double.doubleToRawLongBits(d2));
    }

    public static int zzb(int i, boolean z) {
        return zzg(i << 3) + 1;
    }

    @Deprecated
    public static int zzc(int i, zzkk zzkkVar, zzlc zzlcVar) {
        int zzg = zzg(i << 3) << 1;
        zzhf zzhfVar = (zzhf) zzkkVar;
        int zzi = zzhfVar.zzi();
        if (zzi == -1) {
            zzi = zzlcVar.zzb(zzhfVar);
            zzhfVar.zzb(zzi);
        }
        return zzg + zzi;
    }

    public static int zzf(int i) {
        if (i >= 0) {
            return zzg(i);
        }
        return 10;
    }

    public final void zza(float f2) {
        zzd(Float.floatToRawIntBits(f2));
    }

    public static int zzb(int i, String str) {
        return zzb(str) + zzg(i << 3);
    }

    public static int zzf(long j) {
        return zze(zzi(j));
    }

    public final void zza(double d2) {
        zzc(Double.doubleToRawLongBits(d2));
    }

    public static int zzd(long j) {
        return zze(j);
    }

    public final void zza(boolean z) {
        zza(z ? (byte) 1 : (byte) 0);
    }

    public static int zza(int i, zzjt zzjtVar) {
        int zzg = zzg(i << 3);
        int zzb2 = zzjtVar.zzb();
        return zzg(zzb2) + zzb2 + zzg;
    }

    public static int zzb(int i, zzkk zzkkVar, zzlc zzlcVar) {
        return zza(zzkkVar, zzlcVar) + zzg(i << 3);
    }

    public static int zzb(int i, zzkk zzkkVar) {
        return zzb(zzkkVar) + zzg(24) + zzg(2, i) + (zzg(8) << 1);
    }

    @Deprecated
    public static int zzc(zzkk zzkkVar) {
        return zzkkVar.zzm();
    }

    public static int zza(zzjt zzjtVar) {
        int zzb2 = zzjtVar.zzb();
        return zzg(zzb2) + zzb2;
    }

    public static int zza(zzkk zzkkVar, zzlc zzlcVar) {
        zzhf zzhfVar = (zzhf) zzkkVar;
        int zzi = zzhfVar.zzi();
        if (zzi == -1) {
            zzi = zzlcVar.zzb(zzhfVar);
            zzhfVar.zzb(zzi);
        }
        return zzg(zzi) + zzi;
    }

    public static int zzb(int i, zzjt zzjtVar) {
        int zzg = zzg(2, i);
        return zza(3, zzjtVar) + zzg + (zzg(8) << 1);
    }

    public static int zzb(String str) {
        int length;
        try {
            length = zzmd.zza(str);
        } catch (zzmg unused) {
            length = str.getBytes(zzjf.zza).length;
        }
        return zzg(length) + length;
    }

    public final void zza(String str, zzmg zzmgVar) {
        zzb.logp(Level.WARNING, "com.google.protobuf.CodedOutputStream", "inefficientWriteStringNoTag", "Converting ill-formed UTF-16. Your Protocol Buffer will not round trip correctly!", (Throwable) zzmgVar);
        byte[] bytes = str.getBytes(zzjf.zza);
        try {
            zzb(bytes.length);
            zza(bytes, 0, bytes.length);
        } catch (zzb e2) {
            throw e2;
        } catch (IndexOutOfBoundsException e3) {
            throw new zzb(e3);
        }
    }

    public static int zzb(zzht zzhtVar) {
        int zza2 = zzhtVar.zza();
        return zzg(zza2) + zza2;
    }

    public static int zzb(byte[] bArr) {
        int length = bArr.length;
        return zzg(length) + length;
    }

    public static int zzb(zzkk zzkkVar) {
        int zzm = zzkkVar.zzm();
        return zzg(zzm) + zzm;
    }

    public final void zzb() {
        if (zza() != 0) {
            throw new IllegalStateException("Did not write as much data as expected.");
        }
    }
}