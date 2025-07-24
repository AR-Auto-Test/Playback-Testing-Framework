package com.google.android.gms.internal.vision;

import com.google.common.base.Ascii;
import com.google.common.primitives.UnsignedBytes;

/* compiled from: com.google.android.gms:play-services-vision-common@@19.1.3 */
/* loaded from: classes.dex */
public final class zzhl {
    public static int zza(byte[] bArr, int i, zzhn zzhnVar) {
        int i2 = i + 1;
        byte b2 = bArr[i];
        if (b2 >= 0) {
            zzhnVar.zza = b2;
            return i2;
        }
        return zza(b2, bArr, i2, zzhnVar);
    }

    public static int zzb(byte[] bArr, int i, zzhn zzhnVar) {
        byte b2;
        int i2 = i + 1;
        long j = bArr[i];
        if (j >= 0) {
            zzhnVar.zzb = j;
            return i2;
        }
        int i3 = i2 + 1;
        byte b3 = bArr[i2];
        long j2 = (j & 127) | ((b3 & Ascii.DEL) << 7);
        int i4 = 7;
        while (b3 < 0) {
            int i5 = i3 + 1;
            i4 += 7;
            j2 |= (b2 & Ascii.DEL) << i4;
            b3 = bArr[i3];
            i3 = i5;
        }
        zzhnVar.zzb = j2;
        return i3;
    }

    public static double zzc(byte[] bArr, int i) {
        return Double.longBitsToDouble(zzb(bArr, i));
    }

    public static float zzd(byte[] bArr, int i) {
        return Float.intBitsToFloat(zza(bArr, i));
    }

    public static int zze(byte[] bArr, int i, zzhn zzhnVar) {
        int zza = zza(bArr, i, zzhnVar);
        int i2 = zzhnVar.zza;
        if (i2 >= 0) {
            if (i2 <= bArr.length - zza) {
                if (i2 == 0) {
                    zzhnVar.zzc = zzht.zza;
                    return zza;
                }
                zzhnVar.zzc = zzht.zza(bArr, zza, i2);
                return zza + i2;
            }
            throw zzjk.zza();
        }
        throw zzjk.zzb();
    }

    public static int zzc(byte[] bArr, int i, zzhn zzhnVar) {
        int zza = zza(bArr, i, zzhnVar);
        int i2 = zzhnVar.zza;
        if (i2 >= 0) {
            if (i2 == 0) {
                zzhnVar.zzc = "";
                return zza;
            }
            zzhnVar.zzc = new String(bArr, zza, i2, zzjf.zza);
            return zza + i2;
        }
        throw zzjk.zzb();
    }

    public static int zzd(byte[] bArr, int i, zzhn zzhnVar) {
        int zza = zza(bArr, i, zzhnVar);
        int i2 = zzhnVar.zza;
        if (i2 >= 0) {
            if (i2 == 0) {
                zzhnVar.zzc = "";
                return zza;
            }
            zzhnVar.zzc = zzmd.zzb(bArr, zza, i2);
            return zza + i2;
        }
        throw zzjk.zzb();
    }

    public static int zza(int i, byte[] bArr, int i2, zzhn zzhnVar) {
        int i3 = i & 127;
        int i4 = i2 + 1;
        byte b2 = bArr[i2];
        if (b2 >= 0) {
            zzhnVar.zza = i3 | (b2 << 7);
            return i4;
        }
        int i5 = i3 | ((b2 & Ascii.DEL) << 7);
        int i6 = i4 + 1;
        byte b3 = bArr[i4];
        if (b3 >= 0) {
            zzhnVar.zza = i5 | (b3 << 14);
            return i6;
        }
        int i7 = i5 | ((b3 & Ascii.DEL) << 14);
        int i8 = i6 + 1;
        byte b4 = bArr[i6];
        if (b4 >= 0) {
            zzhnVar.zza = i7 | (b4 << Ascii.NAK);
            return i8;
        }
        int i9 = i7 | ((b4 & Ascii.DEL) << 21);
        int i10 = i8 + 1;
        byte b5 = bArr[i8];
        if (b5 >= 0) {
            zzhnVar.zza = i9 | (b5 << Ascii.FS);
            return i10;
        }
        int i11 = i9 | ((b5 & Ascii.DEL) << 28);
        while (true) {
            int i12 = i10 + 1;
            if (bArr[i10] >= 0) {
                zzhnVar.zza = i11;
                return i12;
            }
            i10 = i12;
        }
    }

    public static long zzb(byte[] bArr, int i) {
        return ((bArr[i + 7] & 255) << 56) | (bArr[i] & 255) | ((bArr[i + 1] & 255) << 8) | ((bArr[i + 2] & 255) << 16) | ((bArr[i + 3] & 255) << 24) | ((bArr[i + 4] & 255) << 32) | ((bArr[i + 5] & 255) << 40) | ((bArr[i + 6] & 255) << 48);
    }

    public static int zza(byte[] bArr, int i) {
        return ((bArr[i + 3] & UnsignedBytes.MAX_VALUE) << 24) | (bArr[i] & UnsignedBytes.MAX_VALUE) | ((bArr[i + 1] & UnsignedBytes.MAX_VALUE) << 8) | ((bArr[i + 2] & UnsignedBytes.MAX_VALUE) << 16);
    }

    public static int zza(zzlc zzlcVar, byte[] bArr, int i, int i2, zzhn zzhnVar) {
        int i3 = i + 1;
        int i4 = bArr[i];
        if (i4 < 0) {
            i3 = zza(i4, bArr, i3, zzhnVar);
            i4 = zzhnVar.zza;
        }
        int i5 = i3;
        if (i4 >= 0 && i4 <= i2 - i5) {
            Object zza = zzlcVar.zza();
            int i6 = i4 + i5;
            zzlcVar.zza(zza, bArr, i5, i6, zzhnVar);
            zzlcVar.zzc(zza);
            zzhnVar.zzc = zza;
            return i6;
        }
        throw zzjk.zza();
    }

    public static int zza(zzlc zzlcVar, byte[] bArr, int i, int i2, int i3, zzhn zzhnVar) {
        zzko zzkoVar = (zzko) zzlcVar;
        Object zza = zzkoVar.zza();
        int zza2 = zzkoVar.zza((zzko) zza, bArr, i, i2, i3, zzhnVar);
        zzkoVar.zzc((zzko) zza);
        zzhnVar.zzc = zza;
        return zza2;
    }

    public static int zza(int i, byte[] bArr, int i2, int i3, zzjl<?> zzjlVar, zzhn zzhnVar) {
        zzjd zzjdVar = (zzjd) zzjlVar;
        int zza = zza(bArr, i2, zzhnVar);
        zzjdVar.zzc(zzhnVar.zza);
        while (zza < i3) {
            int zza2 = zza(bArr, zza, zzhnVar);
            if (i != zzhnVar.zza) {
                break;
            }
            zza = zza(bArr, zza2, zzhnVar);
            zzjdVar.zzc(zzhnVar.zza);
        }
        return zza;
    }

    public static int zza(byte[] bArr, int i, zzjl<?> zzjlVar, zzhn zzhnVar) {
        zzjd zzjdVar = (zzjd) zzjlVar;
        int zza = zza(bArr, i, zzhnVar);
        int i2 = zzhnVar.zza + zza;
        while (zza < i2) {
            zza = zza(bArr, zza, zzhnVar);
            zzjdVar.zzc(zzhnVar.zza);
        }
        if (zza == i2) {
            return zza;
        }
        throw zzjk.zza();
    }

    public static int zza(zzlc<?> zzlcVar, int i, byte[] bArr, int i2, int i3, zzjl<?> zzjlVar, zzhn zzhnVar) {
        int zza = zza(zzlcVar, bArr, i2, i3, zzhnVar);
        zzjlVar.add(zzhnVar.zzc);
        while (zza < i3) {
            int zza2 = zza(bArr, zza, zzhnVar);
            if (i != zzhnVar.zza) {
                break;
            }
            zza = zza(zzlcVar, bArr, zza2, i3, zzhnVar);
            zzjlVar.add(zzhnVar.zzc);
        }
        return zza;
    }

    public static int zza(int i, byte[] bArr, int i2, int i3, zzlx zzlxVar, zzhn zzhnVar) {
        if ((i >>> 3) != 0) {
            int i4 = i & 7;
            if (i4 == 0) {
                int zzb = zzb(bArr, i2, zzhnVar);
                zzlxVar.zza(i, Long.valueOf(zzhnVar.zzb));
                return zzb;
            } else if (i4 == 1) {
                zzlxVar.zza(i, Long.valueOf(zzb(bArr, i2)));
                return i2 + 8;
            } else if (i4 == 2) {
                int zza = zza(bArr, i2, zzhnVar);
                int i5 = zzhnVar.zza;
                if (i5 >= 0) {
                    if (i5 <= bArr.length - zza) {
                        if (i5 == 0) {
                            zzlxVar.zza(i, zzht.zza);
                        } else {
                            zzlxVar.zza(i, zzht.zza(bArr, zza, i5));
                        }
                        return zza + i5;
                    }
                    throw zzjk.zza();
                }
                throw zzjk.zzb();
            } else if (i4 != 3) {
                if (i4 == 5) {
                    zzlxVar.zza(i, Integer.valueOf(zza(bArr, i2)));
                    return i2 + 4;
                }
                throw zzjk.zzd();
            } else {
                zzlx zzb2 = zzlx.zzb();
                int i6 = (i & (-8)) | 4;
                int i7 = 0;
                while (true) {
                    if (i2 >= i3) {
                        break;
                    }
                    int zza2 = zza(bArr, i2, zzhnVar);
                    int i8 = zzhnVar.zza;
                    i7 = i8;
                    if (i8 == i6) {
                        i2 = zza2;
                        break;
                    }
                    int zza3 = zza(i7, bArr, zza2, i3, zzb2, zzhnVar);
                    i7 = i8;
                    i2 = zza3;
                }
                if (i2 <= i3 && i7 == i6) {
                    zzlxVar.zza(i, zzb2);
                    return i2;
                }
                throw zzjk.zzg();
            }
        }
        throw zzjk.zzd();
    }

    public static int zza(int i, byte[] bArr, int i2, int i3, zzhn zzhnVar) {
        if ((i >>> 3) != 0) {
            int i4 = i & 7;
            if (i4 != 0) {
                if (i4 != 1) {
                    if (i4 != 2) {
                        if (i4 != 3) {
                            if (i4 == 5) {
                                return i2 + 4;
                            }
                            throw zzjk.zzd();
                        }
                        int i5 = (i & (-8)) | 4;
                        int i6 = 0;
                        while (i2 < i3) {
                            i2 = zza(bArr, i2, zzhnVar);
                            i6 = zzhnVar.zza;
                            if (i6 == i5) {
                                break;
                            }
                            i2 = zza(i6, bArr, i2, i3, zzhnVar);
                        }
                        if (i2 > i3 || i6 != i5) {
                            throw zzjk.zzg();
                        }
                        return i2;
                    }
                    return zza(bArr, i2, zzhnVar) + zzhnVar.zza;
                }
                return i2 + 8;
            }
            return zzb(bArr, i2, zzhnVar);
        }
        throw zzjk.zzd();
    }
}