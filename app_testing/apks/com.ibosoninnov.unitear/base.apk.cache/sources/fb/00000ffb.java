package com.google.android.gms.internal.clearcut;

import com.google.common.base.Ascii;
import com.google.common.primitives.UnsignedBytes;

/* loaded from: classes.dex */
public final class zzax {
    public static int zza(int i, byte[] bArr, int i2, int i3, zzay zzayVar) {
        if ((i >>> 3) != 0) {
            int i4 = i & 7;
            if (i4 != 0) {
                if (i4 != 1) {
                    if (i4 != 2) {
                        if (i4 != 3) {
                            if (i4 == 5) {
                                return i2 + 4;
                            }
                            throw zzco.zzbm();
                        }
                        int i5 = (i & (-8)) | 4;
                        int i6 = 0;
                        while (i2 < i3) {
                            i2 = zza(bArr, i2, zzayVar);
                            i6 = zzayVar.zzfd;
                            if (i6 == i5) {
                                break;
                            }
                            i2 = zza(i6, bArr, i2, i3, zzayVar);
                        }
                        if (i2 > i3 || i6 != i5) {
                            throw zzco.zzbo();
                        }
                        return i2;
                    }
                    return zza(bArr, i2, zzayVar) + zzayVar.zzfd;
                }
                return i2 + 8;
            }
            return zzb(bArr, i2, zzayVar);
        }
        throw zzco.zzbm();
    }

    public static int zza(int i, byte[] bArr, int i2, int i3, zzcn<?> zzcnVar, zzay zzayVar) {
        zzch zzchVar = (zzch) zzcnVar;
        int zza = zza(bArr, i2, zzayVar);
        while (true) {
            zzchVar.zzac(zzayVar.zzfd);
            if (zza >= i3) {
                break;
            }
            int zza2 = zza(bArr, zza, zzayVar);
            if (i != zzayVar.zzfd) {
                break;
            }
            zza = zza(bArr, zza2, zzayVar);
        }
        return zza;
    }

    public static int zza(int i, byte[] bArr, int i2, int i3, zzey zzeyVar, zzay zzayVar) {
        if ((i >>> 3) != 0) {
            int i4 = i & 7;
            if (i4 == 0) {
                int zzb = zzb(bArr, i2, zzayVar);
                zzeyVar.zzb(i, Long.valueOf(zzayVar.zzfe));
                return zzb;
            } else if (i4 == 1) {
                zzeyVar.zzb(i, Long.valueOf(zzd(bArr, i2)));
                return i2 + 8;
            } else if (i4 == 2) {
                int zza = zza(bArr, i2, zzayVar);
                int i5 = zzayVar.zzfd;
                zzeyVar.zzb(i, i5 == 0 ? zzbb.zzfi : zzbb.zzb(bArr, zza, i5));
                return zza + i5;
            } else if (i4 != 3) {
                if (i4 == 5) {
                    zzeyVar.zzb(i, Integer.valueOf(zzc(bArr, i2)));
                    return i2 + 4;
                }
                throw zzco.zzbm();
            } else {
                zzey zzeb = zzey.zzeb();
                int i6 = (i & (-8)) | 4;
                int i7 = 0;
                while (true) {
                    if (i2 >= i3) {
                        break;
                    }
                    int zza2 = zza(bArr, i2, zzayVar);
                    int i8 = zzayVar.zzfd;
                    i7 = i8;
                    if (i8 == i6) {
                        i2 = zza2;
                        break;
                    }
                    int zza3 = zza(i7, bArr, zza2, i3, zzeb, zzayVar);
                    i7 = i8;
                    i2 = zza3;
                }
                if (i2 > i3 || i7 != i6) {
                    throw zzco.zzbo();
                }
                zzeyVar.zzb(i, zzeb);
                return i2;
            }
        }
        throw zzco.zzbm();
    }

    public static int zza(int i, byte[] bArr, int i2, zzay zzayVar) {
        int i3;
        int i4;
        int i5 = i & 127;
        int i6 = i2 + 1;
        byte b2 = bArr[i2];
        if (b2 < 0) {
            int i7 = i5 | ((b2 & Ascii.DEL) << 7);
            int i8 = i6 + 1;
            byte b3 = bArr[i6];
            if (b3 >= 0) {
                i3 = b3 << 14;
            } else {
                i5 = i7 | ((b3 & Ascii.DEL) << 14);
                i6 = i8 + 1;
                byte b4 = bArr[i8];
                if (b4 >= 0) {
                    i4 = b4 << Ascii.NAK;
                } else {
                    i7 = i5 | ((b4 & Ascii.DEL) << 21);
                    i8 = i6 + 1;
                    byte b5 = bArr[i6];
                    if (b5 >= 0) {
                        i3 = b5 << Ascii.FS;
                    } else {
                        int i9 = i7 | ((b5 & Ascii.DEL) << 28);
                        while (true) {
                            int i10 = i8 + 1;
                            if (bArr[i8] >= 0) {
                                zzayVar.zzfd = i9;
                                return i10;
                            }
                            i8 = i10;
                        }
                    }
                }
            }
            zzayVar.zzfd = i7 | i3;
            return i8;
        }
        i4 = b2 << 7;
        zzayVar.zzfd = i5 | i4;
        return i6;
    }

    public static int zza(byte[] bArr, int i, zzay zzayVar) {
        int i2 = i + 1;
        byte b2 = bArr[i];
        if (b2 >= 0) {
            zzayVar.zzfd = b2;
            return i2;
        }
        return zza(b2, bArr, i2, zzayVar);
    }

    public static int zza(byte[] bArr, int i, zzcn<?> zzcnVar, zzay zzayVar) {
        zzch zzchVar = (zzch) zzcnVar;
        int zza = zza(bArr, i, zzayVar);
        int i2 = zzayVar.zzfd + zza;
        while (zza < i2) {
            zza = zza(bArr, zza, zzayVar);
            zzchVar.zzac(zzayVar.zzfd);
        }
        if (zza == i2) {
            return zza;
        }
        throw zzco.zzbl();
    }

    public static int zzb(byte[] bArr, int i, zzay zzayVar) {
        byte b2;
        int i2 = i + 1;
        long j = bArr[i];
        if (j >= 0) {
            zzayVar.zzfe = j;
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
        zzayVar.zzfe = j2;
        return i3;
    }

    public static int zzc(byte[] bArr, int i) {
        return ((bArr[i + 3] & UnsignedBytes.MAX_VALUE) << 24) | (bArr[i] & UnsignedBytes.MAX_VALUE) | ((bArr[i + 1] & UnsignedBytes.MAX_VALUE) << 8) | ((bArr[i + 2] & UnsignedBytes.MAX_VALUE) << 16);
    }

    public static int zzc(byte[] bArr, int i, zzay zzayVar) {
        int zza = zza(bArr, i, zzayVar);
        int i2 = zzayVar.zzfd;
        if (i2 == 0) {
            zzayVar.zzff = "";
            return zza;
        }
        zzayVar.zzff = new String(bArr, zza, i2, zzci.UTF_8);
        return zza + i2;
    }

    public static int zzd(byte[] bArr, int i, zzay zzayVar) {
        int zza = zza(bArr, i, zzayVar);
        int i2 = zzayVar.zzfd;
        if (i2 == 0) {
            zzayVar.zzff = "";
            return zza;
        }
        int i3 = zza + i2;
        if (zzff.zze(bArr, zza, i3)) {
            zzayVar.zzff = new String(bArr, zza, i2, zzci.UTF_8);
            return i3;
        }
        throw zzco.zzbp();
    }

    public static long zzd(byte[] bArr, int i) {
        return ((bArr[i + 7] & 255) << 56) | (bArr[i] & 255) | ((bArr[i + 1] & 255) << 8) | ((bArr[i + 2] & 255) << 16) | ((bArr[i + 3] & 255) << 24) | ((bArr[i + 4] & 255) << 32) | ((bArr[i + 5] & 255) << 40) | ((bArr[i + 6] & 255) << 48);
    }

    public static double zze(byte[] bArr, int i) {
        return Double.longBitsToDouble(zzd(bArr, i));
    }

    public static int zze(byte[] bArr, int i, zzay zzayVar) {
        int zza = zza(bArr, i, zzayVar);
        int i2 = zzayVar.zzfd;
        if (i2 == 0) {
            zzayVar.zzff = zzbb.zzfi;
            return zza;
        }
        zzayVar.zzff = zzbb.zzb(bArr, zza, i2);
        return zza + i2;
    }

    public static float zzf(byte[] bArr, int i) {
        return Float.intBitsToFloat(zzc(bArr, i));
    }
}