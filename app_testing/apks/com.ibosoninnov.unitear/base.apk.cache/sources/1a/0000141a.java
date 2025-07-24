package com.google.android.gms.internal.vision;

import c.b.a.a.a;
import com.google.android.gms.internal.vision.zzjb;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import com.google.android.material.shadow.ShadowDrawableWrapper;
import java.io.IOException;
import java.lang.reflect.Field;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import sun.misc.Unsafe;

/* compiled from: com.google.android.gms:play-services-vision-common@@19.1.3 */
/* loaded from: classes.dex */
public final class zzko<T> implements zzlc<T> {
    private static final int[] zza = new int[0];
    private static final Unsafe zzb = zzma.zzc();
    private final int[] zzc;
    private final Object[] zzd;
    private final int zze;
    private final int zzf;
    private final zzkk zzg;
    private final boolean zzh;
    private final boolean zzi;
    private final boolean zzj;
    private final boolean zzk;
    private final int[] zzl;
    private final int zzm;
    private final int zzn;
    private final zzks zzo;
    private final zzju zzp;
    private final zzlu<?, ?> zzq;
    private final zziq<?> zzr;
    private final zzkh zzs;

    private zzko(int[] iArr, Object[] objArr, int i, int i2, zzkk zzkkVar, boolean z, boolean z2, int[] iArr2, int i3, int i4, zzks zzksVar, zzju zzjuVar, zzlu<?, ?> zzluVar, zziq<?> zziqVar, zzkh zzkhVar) {
        this.zzc = iArr;
        this.zzd = objArr;
        this.zze = i;
        this.zzf = i2;
        this.zzi = zzkkVar instanceof zzjb;
        this.zzj = z;
        this.zzh = zziqVar != null && zziqVar.zza(zzkkVar);
        this.zzk = false;
        this.zzl = iArr2;
        this.zzm = i3;
        this.zzn = i4;
        this.zzo = zzksVar;
        this.zzp = zzjuVar;
        this.zzq = zzluVar;
        this.zzr = zziqVar;
        this.zzg = zzkkVar;
        this.zzs = zzkhVar;
    }

    /* JADX WARN: Removed duplicated region for block: B:163:0x0338  */
    /* JADX WARN: Removed duplicated region for block: B:180:0x0397  */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public static <T> zzko<T> zza(Class<T> cls, zzki zzkiVar, zzks zzksVar, zzju zzjuVar, zzlu<?, ?> zzluVar, zziq<?> zziqVar, zzkh zzkhVar) {
        int i;
        int charAt;
        int charAt2;
        int charAt3;
        int charAt4;
        int charAt5;
        int i2;
        int[] iArr;
        int i3;
        char charAt6;
        int i4;
        char charAt7;
        int i5;
        char charAt8;
        int i6;
        char charAt9;
        int i7;
        char charAt10;
        int i8;
        char charAt11;
        int i9;
        char charAt12;
        int i10;
        char charAt13;
        int i11;
        int i12;
        int i13;
        int i14;
        int i15;
        int objectFieldOffset;
        String str;
        boolean z;
        boolean z2;
        int i16;
        int i17;
        int i18;
        Field zza2;
        int i19;
        char charAt14;
        int i20;
        int i21;
        Field zza3;
        Field zza4;
        int i22;
        char charAt15;
        int i23;
        char charAt16;
        int i24;
        char charAt17;
        int i25;
        char charAt18;
        if (zzkiVar instanceof zzla) {
            zzla zzlaVar = (zzla) zzkiVar;
            int i26 = 0;
            boolean z3 = zzlaVar.zza() == zzkz.zzb;
            String zzd = zzlaVar.zzd();
            int length = zzd.length();
            if (zzd.charAt(0) >= 55296) {
                int i27 = 1;
                while (true) {
                    i = i27 + 1;
                    if (zzd.charAt(i27) < 55296) {
                        break;
                    }
                    i27 = i;
                }
            } else {
                i = 1;
            }
            int i28 = i + 1;
            int charAt19 = zzd.charAt(i);
            if (charAt19 >= 55296) {
                int i29 = charAt19 & 8191;
                int i30 = 13;
                while (true) {
                    i25 = i28 + 1;
                    charAt18 = zzd.charAt(i28);
                    if (charAt18 < 55296) {
                        break;
                    }
                    i29 |= (charAt18 & 8191) << i30;
                    i30 += 13;
                    i28 = i25;
                }
                charAt19 = i29 | (charAt18 << i30);
                i28 = i25;
            }
            if (charAt19 == 0) {
                charAt = 0;
                charAt2 = 0;
                charAt3 = 0;
                charAt4 = 0;
                charAt5 = 0;
                iArr = zza;
                i2 = 0;
            } else {
                int i31 = i28 + 1;
                int charAt20 = zzd.charAt(i28);
                if (charAt20 >= 55296) {
                    int i32 = charAt20 & 8191;
                    int i33 = 13;
                    while (true) {
                        i10 = i31 + 1;
                        charAt13 = zzd.charAt(i31);
                        if (charAt13 < 55296) {
                            break;
                        }
                        i32 |= (charAt13 & 8191) << i33;
                        i33 += 13;
                        i31 = i10;
                    }
                    charAt20 = i32 | (charAt13 << i33);
                    i31 = i10;
                }
                int i34 = i31 + 1;
                int charAt21 = zzd.charAt(i31);
                if (charAt21 >= 55296) {
                    int i35 = charAt21 & 8191;
                    int i36 = 13;
                    while (true) {
                        i9 = i34 + 1;
                        charAt12 = zzd.charAt(i34);
                        if (charAt12 < 55296) {
                            break;
                        }
                        i35 |= (charAt12 & 8191) << i36;
                        i36 += 13;
                        i34 = i9;
                    }
                    charAt21 = i35 | (charAt12 << i36);
                    i34 = i9;
                }
                int i37 = i34 + 1;
                charAt = zzd.charAt(i34);
                if (charAt >= 55296) {
                    int i38 = charAt & 8191;
                    int i39 = 13;
                    while (true) {
                        i8 = i37 + 1;
                        charAt11 = zzd.charAt(i37);
                        if (charAt11 < 55296) {
                            break;
                        }
                        i38 |= (charAt11 & 8191) << i39;
                        i39 += 13;
                        i37 = i8;
                    }
                    charAt = i38 | (charAt11 << i39);
                    i37 = i8;
                }
                int i40 = i37 + 1;
                charAt2 = zzd.charAt(i37);
                if (charAt2 >= 55296) {
                    int i41 = charAt2 & 8191;
                    int i42 = 13;
                    while (true) {
                        i7 = i40 + 1;
                        charAt10 = zzd.charAt(i40);
                        if (charAt10 < 55296) {
                            break;
                        }
                        i41 |= (charAt10 & 8191) << i42;
                        i42 += 13;
                        i40 = i7;
                    }
                    charAt2 = i41 | (charAt10 << i42);
                    i40 = i7;
                }
                int i43 = i40 + 1;
                charAt3 = zzd.charAt(i40);
                if (charAt3 >= 55296) {
                    int i44 = charAt3 & 8191;
                    int i45 = 13;
                    while (true) {
                        i6 = i43 + 1;
                        charAt9 = zzd.charAt(i43);
                        if (charAt9 < 55296) {
                            break;
                        }
                        i44 |= (charAt9 & 8191) << i45;
                        i45 += 13;
                        i43 = i6;
                    }
                    charAt3 = i44 | (charAt9 << i45);
                    i43 = i6;
                }
                int i46 = i43 + 1;
                charAt4 = zzd.charAt(i43);
                if (charAt4 >= 55296) {
                    int i47 = charAt4 & 8191;
                    int i48 = 13;
                    while (true) {
                        i5 = i46 + 1;
                        charAt8 = zzd.charAt(i46);
                        if (charAt8 < 55296) {
                            break;
                        }
                        i47 |= (charAt8 & 8191) << i48;
                        i48 += 13;
                        i46 = i5;
                    }
                    charAt4 = i47 | (charAt8 << i48);
                    i46 = i5;
                }
                int i49 = i46 + 1;
                int charAt22 = zzd.charAt(i46);
                if (charAt22 >= 55296) {
                    int i50 = charAt22 & 8191;
                    int i51 = 13;
                    while (true) {
                        i4 = i49 + 1;
                        charAt7 = zzd.charAt(i49);
                        if (charAt7 < 55296) {
                            break;
                        }
                        i50 |= (charAt7 & 8191) << i51;
                        i51 += 13;
                        i49 = i4;
                    }
                    charAt22 = i50 | (charAt7 << i51);
                    i49 = i4;
                }
                int i52 = i49 + 1;
                charAt5 = zzd.charAt(i49);
                if (charAt5 >= 55296) {
                    int i53 = charAt5 & 8191;
                    int i54 = i52;
                    int i55 = 13;
                    while (true) {
                        i3 = i54 + 1;
                        charAt6 = zzd.charAt(i54);
                        if (charAt6 < 55296) {
                            break;
                        }
                        i53 |= (charAt6 & 8191) << i55;
                        i55 += 13;
                        i54 = i3;
                    }
                    charAt5 = i53 | (charAt6 << i55);
                    i52 = i3;
                }
                i2 = (charAt20 << 1) + charAt21;
                iArr = new int[charAt5 + charAt4 + charAt22];
                i26 = charAt20;
                i28 = i52;
            }
            Unsafe unsafe = zzb;
            Object[] zze = zzlaVar.zze();
            Class<?> cls2 = zzlaVar.zzc().getClass();
            int i56 = i28;
            int[] iArr2 = new int[charAt3 * 3];
            Object[] objArr = new Object[charAt3 << 1];
            int i57 = charAt5 + charAt4;
            int i58 = i2;
            int i59 = charAt5;
            int i60 = i56;
            int i61 = i57;
            int i62 = 0;
            int i63 = 0;
            while (i60 < length) {
                int i64 = i60 + 1;
                int charAt23 = zzd.charAt(i60);
                if (charAt23 >= 55296) {
                    int i65 = charAt23 & 8191;
                    int i66 = i64;
                    int i67 = 13;
                    while (true) {
                        i24 = i66 + 1;
                        charAt17 = zzd.charAt(i66);
                        i11 = length;
                        if (charAt17 < 55296) {
                            break;
                        }
                        i65 |= (charAt17 & 8191) << i67;
                        i67 += 13;
                        i66 = i24;
                        length = i11;
                    }
                    charAt23 = i65 | (charAt17 << i67);
                    i12 = i24;
                } else {
                    i11 = length;
                    i12 = i64;
                }
                int i68 = i12 + 1;
                int charAt24 = zzd.charAt(i12);
                if (charAt24 >= 55296) {
                    int i69 = charAt24 & 8191;
                    int i70 = i68;
                    int i71 = 13;
                    while (true) {
                        i23 = i70 + 1;
                        charAt16 = zzd.charAt(i70);
                        i13 = charAt5;
                        if (charAt16 < 55296) {
                            break;
                        }
                        i69 |= (charAt16 & 8191) << i71;
                        i71 += 13;
                        i70 = i23;
                        charAt5 = i13;
                    }
                    charAt24 = i69 | (charAt16 << i71);
                    i14 = i23;
                } else {
                    i13 = charAt5;
                    i14 = i68;
                }
                int i72 = charAt24 & 255;
                int i73 = charAt2;
                if ((charAt24 & 1024) != 0) {
                    iArr[i62] = i63;
                    i62++;
                }
                int i74 = charAt;
                if (i72 >= 51) {
                    int i75 = i14 + 1;
                    int charAt25 = zzd.charAt(i14);
                    char c2 = 55296;
                    if (charAt25 >= 55296) {
                        int i76 = charAt25 & 8191;
                        int i77 = 13;
                        while (true) {
                            i22 = i75 + 1;
                            charAt15 = zzd.charAt(i75);
                            if (charAt15 < c2) {
                                break;
                            }
                            i76 |= (charAt15 & 8191) << i77;
                            i77 += 13;
                            i75 = i22;
                            c2 = 55296;
                        }
                        charAt25 = i76 | (charAt15 << i77);
                        i75 = i22;
                    }
                    int i78 = i72 - 51;
                    int i79 = i75;
                    if (i78 == 9 || i78 == 17) {
                        i21 = 1;
                        objArr[((i63 / 3) << 1) + 1] = zze[i58];
                        i58++;
                    } else {
                        if (i78 == 12 && !z3) {
                            objArr[((i63 / 3) << 1) + 1] = zze[i58];
                            i58++;
                        }
                        i21 = 1;
                    }
                    int i80 = charAt25 << i21;
                    Object obj = zze[i80];
                    if (obj instanceof Field) {
                        zza3 = (Field) obj;
                    } else {
                        zza3 = zza(cls2, (String) obj);
                        zze[i80] = zza3;
                    }
                    int objectFieldOffset2 = (int) unsafe.objectFieldOffset(zza3);
                    int i81 = i80 + 1;
                    Object obj2 = zze[i81];
                    if (obj2 instanceof Field) {
                        zza4 = (Field) obj2;
                    } else {
                        zza4 = zza(cls2, (String) obj2);
                        zze[i81] = zza4;
                    }
                    str = zzd;
                    i16 = (int) unsafe.objectFieldOffset(zza4);
                    z = z3;
                    objectFieldOffset = objectFieldOffset2;
                    i17 = i79;
                    i18 = 0;
                    z2 = true;
                } else {
                    int i82 = i58 + 1;
                    Field zza5 = zza(cls2, (String) zze[i58]);
                    if (i72 == 9 || i72 == 17) {
                        objArr[((i63 / 3) << 1) + 1] = zza5.getType();
                    } else {
                        if (i72 == 27 || i72 == 49) {
                            i20 = i82 + 1;
                            objArr[((i63 / 3) << 1) + 1] = zze[i82];
                        } else {
                            if (i72 == 12 || i72 == 30 || i72 == 44) {
                                if (!z3) {
                                    i20 = i82 + 1;
                                    objArr[((i63 / 3) << 1) + 1] = zze[i82];
                                }
                            } else if (i72 == 50) {
                                int i83 = i59 + 1;
                                iArr[i59] = i63;
                                int i84 = (i63 / 3) << 1;
                                i20 = i82 + 1;
                                objArr[i84] = zze[i82];
                                if ((charAt24 & 2048) != 0) {
                                    i82 = i20 + 1;
                                    objArr[i84 + 1] = zze[i20];
                                    i59 = i83;
                                } else {
                                    i59 = i83;
                                }
                            }
                            objectFieldOffset = (int) unsafe.objectFieldOffset(zza5);
                            int i85 = i15;
                            if ((charAt24 & 4096) == 4096 || i72 > 17) {
                                str = zzd;
                                z = z3;
                                z2 = true;
                                i16 = 1048575;
                                i17 = i14;
                                i18 = 0;
                            } else {
                                int i86 = i14 + 1;
                                int charAt26 = zzd.charAt(i14);
                                if (charAt26 >= 55296) {
                                    int i87 = charAt26 & 8191;
                                    int i88 = 13;
                                    while (true) {
                                        i19 = i86 + 1;
                                        charAt14 = zzd.charAt(i86);
                                        if (charAt14 < 55296) {
                                            break;
                                        }
                                        i87 |= (charAt14 & 8191) << i88;
                                        i88 += 13;
                                        i86 = i19;
                                    }
                                    charAt26 = i87 | (charAt14 << i88);
                                    i86 = i19;
                                }
                                z2 = true;
                                int i89 = (charAt26 / 32) + (i26 << 1);
                                Object obj3 = zze[i89];
                                str = zzd;
                                if (obj3 instanceof Field) {
                                    zza2 = (Field) obj3;
                                } else {
                                    zza2 = zza(cls2, (String) obj3);
                                    zze[i89] = zza2;
                                }
                                i17 = i86;
                                z = z3;
                                i16 = (int) unsafe.objectFieldOffset(zza2);
                                i18 = charAt26 % 32;
                            }
                            if (i72 >= 18 && i72 <= 49) {
                                iArr[i61] = objectFieldOffset;
                                i61++;
                            }
                            i58 = i85;
                        }
                        i15 = i20;
                        objectFieldOffset = (int) unsafe.objectFieldOffset(zza5);
                        int i852 = i15;
                        if ((charAt24 & 4096) == 4096) {
                        }
                        str = zzd;
                        z = z3;
                        z2 = true;
                        i16 = 1048575;
                        i17 = i14;
                        i18 = 0;
                        if (i72 >= 18) {
                            iArr[i61] = objectFieldOffset;
                            i61++;
                        }
                        i58 = i852;
                    }
                    i15 = i82;
                    objectFieldOffset = (int) unsafe.objectFieldOffset(zza5);
                    int i8522 = i15;
                    if ((charAt24 & 4096) == 4096) {
                    }
                    str = zzd;
                    z = z3;
                    z2 = true;
                    i16 = 1048575;
                    i17 = i14;
                    i18 = 0;
                    if (i72 >= 18) {
                    }
                    i58 = i8522;
                }
                int i90 = i63 + 1;
                iArr2[i63] = charAt23;
                int i91 = i90 + 1;
                int i92 = i26;
                iArr2[i90] = ((charAt24 & 256) != 0 ? 268435456 : 0) | ((charAt24 & 512) != 0 ? 536870912 : 0) | (i72 << 20) | objectFieldOffset;
                int i93 = i91 + 1;
                iArr2[i91] = (i18 << 20) | i16;
                z3 = z;
                i26 = i92;
                charAt2 = i73;
                charAt5 = i13;
                i60 = i17;
                charAt = i74;
                zzd = str;
                i63 = i93;
                length = i11;
            }
            return new zzko<>(iArr2, objArr, charAt, charAt2, zzlaVar.zzc(), z3, false, iArr, charAt5, i57, zzksVar, zzjuVar, zzluVar, zziqVar, zzkhVar);
        }
        ((zzlr) zzkiVar).zza();
        int i94 = zzkz.zzb;
        throw new NoSuchMethodError();
    }

    private final zzjg zzc(int i) {
        return (zzjg) this.zzd[((i / 3) << 1) + 1];
    }

    private static zzlx zze(Object obj) {
        zzjb zzjbVar = (zzjb) obj;
        zzlx zzlxVar = zzjbVar.zzb;
        if (zzlxVar == zzlx.zza()) {
            zzlx zzb2 = zzlx.zzb();
            zzjbVar.zzb = zzb2;
            return zzb2;
        }
        return zzlxVar;
    }

    private static boolean zzf(int i) {
        return (i & 536870912) != 0;
    }

    private static <T> boolean zzf(T t, long j) {
        return ((Boolean) zzma.zzf(t, j)).booleanValue();
    }

    private final int zzg(int i) {
        if (i < this.zze || i > this.zzf) {
            return -1;
        }
        return zzb(i, 0);
    }

    @Override // com.google.android.gms.internal.vision.zzlc
    public final void zzb(T t, T t2) {
        Objects.requireNonNull(t2);
        for (int i = 0; i < this.zzc.length; i += 3) {
            int zzd = zzd(i);
            long j = 1048575 & zzd;
            int i2 = this.zzc[i];
            switch ((zzd & 267386880) >>> 20) {
                case 0:
                    if (zza((zzko<T>) t2, i)) {
                        zzma.zza(t, j, zzma.zze(t2, j));
                        zzb((zzko<T>) t, i);
                        break;
                    } else {
                        break;
                    }
                case 1:
                    if (zza((zzko<T>) t2, i)) {
                        zzma.zza((Object) t, j, zzma.zzd(t2, j));
                        zzb((zzko<T>) t, i);
                        break;
                    } else {
                        break;
                    }
                case 2:
                    if (zza((zzko<T>) t2, i)) {
                        zzma.zza((Object) t, j, zzma.zzb(t2, j));
                        zzb((zzko<T>) t, i);
                        break;
                    } else {
                        break;
                    }
                case 3:
                    if (zza((zzko<T>) t2, i)) {
                        zzma.zza((Object) t, j, zzma.zzb(t2, j));
                        zzb((zzko<T>) t, i);
                        break;
                    } else {
                        break;
                    }
                case 4:
                    if (zza((zzko<T>) t2, i)) {
                        zzma.zza((Object) t, j, zzma.zza(t2, j));
                        zzb((zzko<T>) t, i);
                        break;
                    } else {
                        break;
                    }
                case 5:
                    if (zza((zzko<T>) t2, i)) {
                        zzma.zza((Object) t, j, zzma.zzb(t2, j));
                        zzb((zzko<T>) t, i);
                        break;
                    } else {
                        break;
                    }
                case 6:
                    if (zza((zzko<T>) t2, i)) {
                        zzma.zza((Object) t, j, zzma.zza(t2, j));
                        zzb((zzko<T>) t, i);
                        break;
                    } else {
                        break;
                    }
                case 7:
                    if (zza((zzko<T>) t2, i)) {
                        zzma.zza(t, j, zzma.zzc(t2, j));
                        zzb((zzko<T>) t, i);
                        break;
                    } else {
                        break;
                    }
                case 8:
                    if (zza((zzko<T>) t2, i)) {
                        zzma.zza(t, j, zzma.zzf(t2, j));
                        zzb((zzko<T>) t, i);
                        break;
                    } else {
                        break;
                    }
                case 9:
                    zza(t, t2, i);
                    break;
                case 10:
                    if (zza((zzko<T>) t2, i)) {
                        zzma.zza(t, j, zzma.zzf(t2, j));
                        zzb((zzko<T>) t, i);
                        break;
                    } else {
                        break;
                    }
                case 11:
                    if (zza((zzko<T>) t2, i)) {
                        zzma.zza((Object) t, j, zzma.zza(t2, j));
                        zzb((zzko<T>) t, i);
                        break;
                    } else {
                        break;
                    }
                case 12:
                    if (zza((zzko<T>) t2, i)) {
                        zzma.zza((Object) t, j, zzma.zza(t2, j));
                        zzb((zzko<T>) t, i);
                        break;
                    } else {
                        break;
                    }
                case 13:
                    if (zza((zzko<T>) t2, i)) {
                        zzma.zza((Object) t, j, zzma.zza(t2, j));
                        zzb((zzko<T>) t, i);
                        break;
                    } else {
                        break;
                    }
                case 14:
                    if (zza((zzko<T>) t2, i)) {
                        zzma.zza((Object) t, j, zzma.zzb(t2, j));
                        zzb((zzko<T>) t, i);
                        break;
                    } else {
                        break;
                    }
                case 15:
                    if (zza((zzko<T>) t2, i)) {
                        zzma.zza((Object) t, j, zzma.zza(t2, j));
                        zzb((zzko<T>) t, i);
                        break;
                    } else {
                        break;
                    }
                case 16:
                    if (zza((zzko<T>) t2, i)) {
                        zzma.zza((Object) t, j, zzma.zzb(t2, j));
                        zzb((zzko<T>) t, i);
                        break;
                    } else {
                        break;
                    }
                case 17:
                    zza(t, t2, i);
                    break;
                case 18:
                case 19:
                case 20:
                case 21:
                case 22:
                case 23:
                case 24:
                case 25:
                case 26:
                case 27:
                case 28:
                case 29:
                case 30:
                case 31:
                case 32:
                case 33:
                case 34:
                case 35:
                case 36:
                case 37:
                case 38:
                case 39:
                case 40:
                case 41:
                case 42:
                case 43:
                case 44:
                case 45:
                case 46:
                case 47:
                case 48:
                case 49:
                    this.zzp.zza(t, t2, j);
                    break;
                case 50:
                    zzle.zza(this.zzs, t, t2, j);
                    break;
                case 51:
                case 52:
                case 53:
                case 54:
                case 55:
                case 56:
                case 57:
                case 58:
                case 59:
                    if (zza((zzko<T>) t2, i2, i)) {
                        zzma.zza(t, j, zzma.zzf(t2, j));
                        zzb((zzko<T>) t, i2, i);
                        break;
                    } else {
                        break;
                    }
                case 60:
                    zzb(t, t2, i);
                    break;
                case 61:
                case 62:
                case 63:
                case 64:
                case 65:
                case 66:
                case 67:
                    if (zza((zzko<T>) t2, i2, i)) {
                        zzma.zza(t, j, zzma.zzf(t2, j));
                        zzb((zzko<T>) t, i2, i);
                        break;
                    } else {
                        break;
                    }
                case 68:
                    zzb(t, t2, i);
                    break;
            }
        }
        zzle.zza(this.zzq, t, t2);
        if (this.zzh) {
            zzle.zza(this.zzr, t, t2);
        }
    }

    /* JADX WARN: Multi-variable type inference failed */
    /* JADX WARN: Type inference failed for: r1v21 */
    /* JADX WARN: Type inference failed for: r1v23, types: [com.google.android.gms.internal.vision.zzlc] */
    /* JADX WARN: Type inference failed for: r1v30 */
    /* JADX WARN: Type inference failed for: r1v8, types: [com.google.android.gms.internal.vision.zzlc] */
    @Override // com.google.android.gms.internal.vision.zzlc
    public final boolean zzd(T t) {
        int i;
        int i2;
        int i3 = 1048575;
        int i4 = 0;
        int i5 = 0;
        while (true) {
            boolean z = true;
            if (i5 >= this.zzm) {
                return !this.zzh || this.zzr.zza(t).zzf();
            }
            int i6 = this.zzl[i5];
            int i7 = this.zzc[i6];
            int zzd = zzd(i6);
            int i8 = this.zzc[i6 + 2];
            int i9 = i8 & 1048575;
            int i10 = 1 << (i8 >>> 20);
            if (i9 != i3) {
                if (i9 != 1048575) {
                    i4 = zzb.getInt(t, i9);
                }
                i2 = i4;
                i = i9;
            } else {
                i = i3;
                i2 = i4;
            }
            if (((268435456 & zzd) != 0) && !zza((zzko<T>) t, i6, i, i2, i10)) {
                return false;
            }
            int i11 = (267386880 & zzd) >>> 20;
            if (i11 != 9 && i11 != 17) {
                if (i11 != 27) {
                    if (i11 == 60 || i11 == 68) {
                        if (zza((zzko<T>) t, i7, i6) && !zza(t, zzd, zza(i6))) {
                            return false;
                        }
                    } else if (i11 != 49) {
                        if (i11 != 50) {
                            continue;
                        } else {
                            Map<?, ?> zzc = this.zzs.zzc(zzma.zzf(t, zzd & 1048575));
                            if (!zzc.isEmpty()) {
                                if (this.zzs.zzb(zzb(i6)).zzc.zza() == zzmo.MESSAGE) {
                                    zzlc<T> zzlcVar = 0;
                                    Iterator<?> it = zzc.values().iterator();
                                    while (true) {
                                        if (!it.hasNext()) {
                                            break;
                                        }
                                        Object next = it.next();
                                        if (zzlcVar == null) {
                                            zzlcVar = zzky.zza().zza((Class) next.getClass());
                                        }
                                        boolean zzd2 = zzlcVar.zzd(next);
                                        zzlcVar = zzlcVar;
                                        if (!zzd2) {
                                            z = false;
                                            break;
                                        }
                                    }
                                }
                            }
                            if (!z) {
                                return false;
                            }
                        }
                    }
                }
                List list = (List) zzma.zzf(t, zzd & 1048575);
                if (!list.isEmpty()) {
                    ?? zza2 = zza(i6);
                    int i12 = 0;
                    while (true) {
                        if (i12 >= list.size()) {
                            break;
                        } else if (!zza2.zzd(list.get(i12))) {
                            z = false;
                            break;
                        } else {
                            i12++;
                        }
                    }
                }
                if (!z) {
                    return false;
                }
            } else if (zza((zzko<T>) t, i6, i, i2, i10) && !zza(t, zzd, zza(i6))) {
                return false;
            }
            i5++;
            i3 = i;
            i4 = i2;
        }
    }

    @Override // com.google.android.gms.internal.vision.zzlc
    public final void zzc(T t) {
        int i;
        int i2 = this.zzm;
        while (true) {
            i = this.zzn;
            if (i2 >= i) {
                break;
            }
            long zzd = zzd(this.zzl[i2]) & 1048575;
            Object zzf = zzma.zzf(t, zzd);
            if (zzf != null) {
                zzma.zza(t, zzd, this.zzs.zze(zzf));
            }
            i2++;
        }
        int length = this.zzl.length;
        while (i < length) {
            this.zzp.zzb(t, this.zzl[i]);
            i++;
        }
        this.zzq.zzd(t);
        if (this.zzh) {
            this.zzr.zzc(t);
        }
    }

    private final int zze(int i) {
        return this.zzc[i + 2];
    }

    private static <T> long zze(T t, long j) {
        return ((Long) zzma.zzf(t, j)).longValue();
    }

    private static <T> float zzc(T t, long j) {
        return ((Float) zzma.zzf(t, j)).floatValue();
    }

    private final boolean zzc(T t, T t2, int i) {
        return zza((zzko<T>) t, i) == zza((zzko<T>) t2, i);
    }

    private final int zzd(int i) {
        return this.zzc[i + 1];
    }

    private static <T> int zzd(T t, long j) {
        return ((Integer) zzma.zzf(t, j)).intValue();
    }

    private final void zzb(T t, T t2, int i) {
        int zzd = zzd(i);
        int i2 = this.zzc[i];
        long j = zzd & 1048575;
        if (zza((zzko<T>) t2, i2, i)) {
            Object zzf = zza((zzko<T>) t, i2, i) ? zzma.zzf(t, j) : null;
            Object zzf2 = zzma.zzf(t2, j);
            if (zzf != null && zzf2 != null) {
                zzma.zza(t, j, zzjf.zza(zzf, zzf2));
                zzb((zzko<T>) t, i2, i);
            } else if (zzf2 != null) {
                zzma.zza(t, j, zzf2);
                zzb((zzko<T>) t, i2, i);
            }
        }
    }

    /* JADX WARN: Can't fix incorrect switch cases order, some code will duplicate */
    @Override // com.google.android.gms.internal.vision.zzlc
    public final int zzb(T t) {
        int i;
        int i2;
        int i3;
        int i4;
        int i5;
        int i6;
        int zzd;
        int i7;
        int zzb2;
        int i8;
        int zzh;
        int zzi;
        int zze;
        int zzg;
        int zzb3;
        int zzi2;
        int zze2;
        int zzg2;
        int i9 = 267386880;
        int i10 = 1048575;
        int i11 = 1;
        int i12 = 0;
        if (this.zzj) {
            Unsafe unsafe = zzb;
            int i13 = 0;
            int i14 = 0;
            while (i13 < this.zzc.length) {
                int zzd2 = zzd(i13);
                int i15 = (i9 & zzd2) >>> 20;
                int i16 = this.zzc[i13];
                long j = zzd2 & 1048575;
                if (i15 >= zziv.zza.zza() && i15 <= zziv.zzb.zza()) {
                    int i17 = this.zzc[i13 + 2];
                }
                switch (i15) {
                    case 0:
                        if (zza((zzko<T>) t, i13)) {
                            zzb3 = zzii.zzb(i16, (double) ShadowDrawableWrapper.COS_45);
                            i14 += zzb3;
                            break;
                        } else {
                            break;
                        }
                    case 1:
                        if (zza((zzko<T>) t, i13)) {
                            zzb3 = zzii.zzb(i16, (float) StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
                            i14 += zzb3;
                            break;
                        } else {
                            break;
                        }
                    case 2:
                        if (zza((zzko<T>) t, i13)) {
                            zzb3 = zzii.zzd(i16, zzma.zzb(t, j));
                            i14 += zzb3;
                            break;
                        } else {
                            break;
                        }
                    case 3:
                        if (zza((zzko<T>) t, i13)) {
                            zzb3 = zzii.zze(i16, zzma.zzb(t, j));
                            i14 += zzb3;
                            break;
                        } else {
                            break;
                        }
                    case 4:
                        if (zza((zzko<T>) t, i13)) {
                            zzb3 = zzii.zzf(i16, zzma.zza(t, j));
                            i14 += zzb3;
                            break;
                        } else {
                            break;
                        }
                    case 5:
                        if (zza((zzko<T>) t, i13)) {
                            zzb3 = zzii.zzg(i16, 0L);
                            i14 += zzb3;
                            break;
                        } else {
                            break;
                        }
                    case 6:
                        if (zza((zzko<T>) t, i13)) {
                            zzb3 = zzii.zzi(i16, 0);
                            i14 += zzb3;
                            break;
                        } else {
                            break;
                        }
                    case 7:
                        if (zza((zzko<T>) t, i13)) {
                            zzb3 = zzii.zzb(i16, true);
                            i14 += zzb3;
                            break;
                        } else {
                            break;
                        }
                    case 8:
                        if (zza((zzko<T>) t, i13)) {
                            Object zzf = zzma.zzf(t, j);
                            if (zzf instanceof zzht) {
                                zzb3 = zzii.zzc(i16, (zzht) zzf);
                            } else {
                                zzb3 = zzii.zzb(i16, (String) zzf);
                            }
                            i14 += zzb3;
                            break;
                        } else {
                            break;
                        }
                    case 9:
                        if (zza((zzko<T>) t, i13)) {
                            zzb3 = zzle.zza(i16, zzma.zzf(t, j), zza(i13));
                            i14 += zzb3;
                            break;
                        } else {
                            break;
                        }
                    case 10:
                        if (zza((zzko<T>) t, i13)) {
                            zzb3 = zzii.zzc(i16, (zzht) zzma.zzf(t, j));
                            i14 += zzb3;
                            break;
                        } else {
                            break;
                        }
                    case 11:
                        if (zza((zzko<T>) t, i13)) {
                            zzb3 = zzii.zzg(i16, zzma.zza(t, j));
                            i14 += zzb3;
                            break;
                        } else {
                            break;
                        }
                    case 12:
                        if (zza((zzko<T>) t, i13)) {
                            zzb3 = zzii.zzk(i16, zzma.zza(t, j));
                            i14 += zzb3;
                            break;
                        } else {
                            break;
                        }
                    case 13:
                        if (zza((zzko<T>) t, i13)) {
                            zzb3 = zzii.zzj(i16, 0);
                            i14 += zzb3;
                            break;
                        } else {
                            break;
                        }
                    case 14:
                        if (zza((zzko<T>) t, i13)) {
                            zzb3 = zzii.zzh(i16, 0L);
                            i14 += zzb3;
                            break;
                        } else {
                            break;
                        }
                    case 15:
                        if (zza((zzko<T>) t, i13)) {
                            zzb3 = zzii.zzh(i16, zzma.zza(t, j));
                            i14 += zzb3;
                            break;
                        } else {
                            break;
                        }
                    case 16:
                        if (zza((zzko<T>) t, i13)) {
                            zzb3 = zzii.zzf(i16, zzma.zzb(t, j));
                            i14 += zzb3;
                            break;
                        } else {
                            break;
                        }
                    case 17:
                        if (zza((zzko<T>) t, i13)) {
                            zzb3 = zzii.zzc(i16, (zzkk) zzma.zzf(t, j), zza(i13));
                            i14 += zzb3;
                            break;
                        } else {
                            break;
                        }
                    case 18:
                        zzb3 = zzle.zzi(i16, zza(t, j), false);
                        i14 += zzb3;
                        break;
                    case 19:
                        zzb3 = zzle.zzh(i16, zza(t, j), false);
                        i14 += zzb3;
                        break;
                    case 20:
                        zzb3 = zzle.zza(i16, (List<Long>) zza(t, j), false);
                        i14 += zzb3;
                        break;
                    case 21:
                        zzb3 = zzle.zzb(i16, (List<Long>) zza(t, j), false);
                        i14 += zzb3;
                        break;
                    case 22:
                        zzb3 = zzle.zze(i16, zza(t, j), false);
                        i14 += zzb3;
                        break;
                    case 23:
                        zzb3 = zzle.zzi(i16, zza(t, j), false);
                        i14 += zzb3;
                        break;
                    case 24:
                        zzb3 = zzle.zzh(i16, zza(t, j), false);
                        i14 += zzb3;
                        break;
                    case 25:
                        zzb3 = zzle.zzj(i16, zza(t, j), false);
                        i14 += zzb3;
                        break;
                    case 26:
                        zzb3 = zzle.zza(i16, zza(t, j));
                        i14 += zzb3;
                        break;
                    case 27:
                        zzb3 = zzle.zza(i16, zza(t, j), zza(i13));
                        i14 += zzb3;
                        break;
                    case 28:
                        zzb3 = zzle.zzb(i16, zza(t, j));
                        i14 += zzb3;
                        break;
                    case 29:
                        zzb3 = zzle.zzf(i16, zza(t, j), false);
                        i14 += zzb3;
                        break;
                    case 30:
                        zzb3 = zzle.zzd(i16, zza(t, j), false);
                        i14 += zzb3;
                        break;
                    case 31:
                        zzb3 = zzle.zzh(i16, zza(t, j), false);
                        i14 += zzb3;
                        break;
                    case 32:
                        zzb3 = zzle.zzi(i16, zza(t, j), false);
                        i14 += zzb3;
                        break;
                    case 33:
                        zzb3 = zzle.zzg(i16, zza(t, j), false);
                        i14 += zzb3;
                        break;
                    case 34:
                        zzb3 = zzle.zzc(i16, zza(t, j), false);
                        i14 += zzb3;
                        break;
                    case 35:
                        zzi2 = zzle.zzi((List) unsafe.getObject(t, j));
                        if (zzi2 > 0) {
                            zze2 = zzii.zze(i16);
                            zzg2 = zzii.zzg(zzi2);
                            i14 += zzg2 + zze2 + zzi2;
                            break;
                        } else {
                            break;
                        }
                    case 36:
                        zzi2 = zzle.zzh((List) unsafe.getObject(t, j));
                        if (zzi2 > 0) {
                            zze2 = zzii.zze(i16);
                            zzg2 = zzii.zzg(zzi2);
                            i14 += zzg2 + zze2 + zzi2;
                            break;
                        } else {
                            break;
                        }
                    case 37:
                        zzi2 = zzle.zza((List) unsafe.getObject(t, j));
                        if (zzi2 > 0) {
                            zze2 = zzii.zze(i16);
                            zzg2 = zzii.zzg(zzi2);
                            i14 += zzg2 + zze2 + zzi2;
                            break;
                        } else {
                            break;
                        }
                    case 38:
                        zzi2 = zzle.zzb((List) unsafe.getObject(t, j));
                        if (zzi2 > 0) {
                            zze2 = zzii.zze(i16);
                            zzg2 = zzii.zzg(zzi2);
                            i14 += zzg2 + zze2 + zzi2;
                            break;
                        } else {
                            break;
                        }
                    case 39:
                        zzi2 = zzle.zze((List) unsafe.getObject(t, j));
                        if (zzi2 > 0) {
                            zze2 = zzii.zze(i16);
                            zzg2 = zzii.zzg(zzi2);
                            i14 += zzg2 + zze2 + zzi2;
                            break;
                        } else {
                            break;
                        }
                    case 40:
                        zzi2 = zzle.zzi((List) unsafe.getObject(t, j));
                        if (zzi2 > 0) {
                            zze2 = zzii.zze(i16);
                            zzg2 = zzii.zzg(zzi2);
                            i14 += zzg2 + zze2 + zzi2;
                            break;
                        } else {
                            break;
                        }
                    case 41:
                        zzi2 = zzle.zzh((List) unsafe.getObject(t, j));
                        if (zzi2 > 0) {
                            zze2 = zzii.zze(i16);
                            zzg2 = zzii.zzg(zzi2);
                            i14 += zzg2 + zze2 + zzi2;
                            break;
                        } else {
                            break;
                        }
                    case 42:
                        zzi2 = zzle.zzj((List) unsafe.getObject(t, j));
                        if (zzi2 > 0) {
                            zze2 = zzii.zze(i16);
                            zzg2 = zzii.zzg(zzi2);
                            i14 += zzg2 + zze2 + zzi2;
                            break;
                        } else {
                            break;
                        }
                    case 43:
                        zzi2 = zzle.zzf((List) unsafe.getObject(t, j));
                        if (zzi2 > 0) {
                            zze2 = zzii.zze(i16);
                            zzg2 = zzii.zzg(zzi2);
                            i14 += zzg2 + zze2 + zzi2;
                            break;
                        } else {
                            break;
                        }
                    case 44:
                        zzi2 = zzle.zzd((List) unsafe.getObject(t, j));
                        if (zzi2 > 0) {
                            zze2 = zzii.zze(i16);
                            zzg2 = zzii.zzg(zzi2);
                            i14 += zzg2 + zze2 + zzi2;
                            break;
                        } else {
                            break;
                        }
                    case 45:
                        zzi2 = zzle.zzh((List) unsafe.getObject(t, j));
                        if (zzi2 > 0) {
                            zze2 = zzii.zze(i16);
                            zzg2 = zzii.zzg(zzi2);
                            i14 += zzg2 + zze2 + zzi2;
                            break;
                        } else {
                            break;
                        }
                    case 46:
                        zzi2 = zzle.zzi((List) unsafe.getObject(t, j));
                        if (zzi2 > 0) {
                            zze2 = zzii.zze(i16);
                            zzg2 = zzii.zzg(zzi2);
                            i14 += zzg2 + zze2 + zzi2;
                            break;
                        } else {
                            break;
                        }
                    case 47:
                        zzi2 = zzle.zzg((List) unsafe.getObject(t, j));
                        if (zzi2 > 0) {
                            zze2 = zzii.zze(i16);
                            zzg2 = zzii.zzg(zzi2);
                            i14 += zzg2 + zze2 + zzi2;
                            break;
                        } else {
                            break;
                        }
                    case 48:
                        zzi2 = zzle.zzc((List) unsafe.getObject(t, j));
                        if (zzi2 > 0) {
                            zze2 = zzii.zze(i16);
                            zzg2 = zzii.zzg(zzi2);
                            i14 += zzg2 + zze2 + zzi2;
                            break;
                        } else {
                            break;
                        }
                    case 49:
                        zzb3 = zzle.zzb(i16, (List<zzkk>) zza(t, j), zza(i13));
                        i14 += zzb3;
                        break;
                    case 50:
                        zzb3 = this.zzs.zza(i16, zzma.zzf(t, j), zzb(i13));
                        i14 += zzb3;
                        break;
                    case 51:
                        if (zza((zzko<T>) t, i16, i13)) {
                            zzb3 = zzii.zzb(i16, (double) ShadowDrawableWrapper.COS_45);
                            i14 += zzb3;
                            break;
                        } else {
                            break;
                        }
                    case 52:
                        if (zza((zzko<T>) t, i16, i13)) {
                            zzb3 = zzii.zzb(i16, (float) StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
                            i14 += zzb3;
                            break;
                        } else {
                            break;
                        }
                    case 53:
                        if (zza((zzko<T>) t, i16, i13)) {
                            zzb3 = zzii.zzd(i16, zze(t, j));
                            i14 += zzb3;
                            break;
                        } else {
                            break;
                        }
                    case 54:
                        if (zza((zzko<T>) t, i16, i13)) {
                            zzb3 = zzii.zze(i16, zze(t, j));
                            i14 += zzb3;
                            break;
                        } else {
                            break;
                        }
                    case 55:
                        if (zza((zzko<T>) t, i16, i13)) {
                            zzb3 = zzii.zzf(i16, zzd(t, j));
                            i14 += zzb3;
                            break;
                        } else {
                            break;
                        }
                    case 56:
                        if (zza((zzko<T>) t, i16, i13)) {
                            zzb3 = zzii.zzg(i16, 0L);
                            i14 += zzb3;
                            break;
                        } else {
                            break;
                        }
                    case 57:
                        if (zza((zzko<T>) t, i16, i13)) {
                            zzb3 = zzii.zzi(i16, 0);
                            i14 += zzb3;
                            break;
                        } else {
                            break;
                        }
                    case 58:
                        if (zza((zzko<T>) t, i16, i13)) {
                            zzb3 = zzii.zzb(i16, true);
                            i14 += zzb3;
                            break;
                        } else {
                            break;
                        }
                    case 59:
                        if (zza((zzko<T>) t, i16, i13)) {
                            Object zzf2 = zzma.zzf(t, j);
                            if (zzf2 instanceof zzht) {
                                zzb3 = zzii.zzc(i16, (zzht) zzf2);
                            } else {
                                zzb3 = zzii.zzb(i16, (String) zzf2);
                            }
                            i14 += zzb3;
                            break;
                        } else {
                            break;
                        }
                    case 60:
                        if (zza((zzko<T>) t, i16, i13)) {
                            zzb3 = zzle.zza(i16, zzma.zzf(t, j), zza(i13));
                            i14 += zzb3;
                            break;
                        } else {
                            break;
                        }
                    case 61:
                        if (zza((zzko<T>) t, i16, i13)) {
                            zzb3 = zzii.zzc(i16, (zzht) zzma.zzf(t, j));
                            i14 += zzb3;
                            break;
                        } else {
                            break;
                        }
                    case 62:
                        if (zza((zzko<T>) t, i16, i13)) {
                            zzb3 = zzii.zzg(i16, zzd(t, j));
                            i14 += zzb3;
                            break;
                        } else {
                            break;
                        }
                    case 63:
                        if (zza((zzko<T>) t, i16, i13)) {
                            zzb3 = zzii.zzk(i16, zzd(t, j));
                            i14 += zzb3;
                            break;
                        } else {
                            break;
                        }
                    case 64:
                        if (zza((zzko<T>) t, i16, i13)) {
                            zzb3 = zzii.zzj(i16, 0);
                            i14 += zzb3;
                            break;
                        } else {
                            break;
                        }
                    case 65:
                        if (zza((zzko<T>) t, i16, i13)) {
                            zzb3 = zzii.zzh(i16, 0L);
                            i14 += zzb3;
                            break;
                        } else {
                            break;
                        }
                    case 66:
                        if (zza((zzko<T>) t, i16, i13)) {
                            zzb3 = zzii.zzh(i16, zzd(t, j));
                            i14 += zzb3;
                            break;
                        } else {
                            break;
                        }
                    case 67:
                        if (zza((zzko<T>) t, i16, i13)) {
                            zzb3 = zzii.zzf(i16, zze(t, j));
                            i14 += zzb3;
                            break;
                        } else {
                            break;
                        }
                    case 68:
                        if (zza((zzko<T>) t, i16, i13)) {
                            zzb3 = zzii.zzc(i16, (zzkk) zzma.zzf(t, j), zza(i13));
                            i14 += zzb3;
                            break;
                        } else {
                            break;
                        }
                }
                i13 += 3;
                i9 = 267386880;
            }
            return i14 + zza((zzlu) this.zzq, (Object) t);
        }
        Unsafe unsafe2 = zzb;
        int i18 = 1048575;
        int i19 = 0;
        int i20 = 0;
        int i21 = 0;
        while (i12 < this.zzc.length) {
            int zzd3 = zzd(i12);
            int[] iArr = this.zzc;
            int i22 = iArr[i12];
            int i23 = (267386880 & zzd3) >>> 20;
            if (i23 <= 17) {
                int i24 = iArr[i12 + 2];
                int i25 = i24 & i10;
                i = i11 << (i24 >>> 20);
                if (i25 != i18) {
                    i21 = unsafe2.getInt(t, i25);
                    i18 = i25;
                }
            } else {
                i = 0;
            }
            long j2 = zzd3 & i10;
            switch (i23) {
                case 0:
                    i2 = 1;
                    i4 = 0;
                    if ((i & i21) != 0) {
                        i20 = zzii.zzb(i22, (double) ShadowDrawableWrapper.COS_45) + i20;
                        break;
                    }
                    break;
                case 1:
                    i2 = 1;
                    i4 = 0;
                    if ((i & i21) != 0) {
                        i20 += zzii.zzb(i22, (float) StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
                        break;
                    }
                    break;
                case 2:
                    i5 = 1;
                    i6 = 0;
                    if ((i & i21) != 0) {
                        zzd = zzii.zzd(i22, unsafe2.getLong(t, j2));
                        i2 = i5;
                        i4 = i6;
                        i20 += zzd;
                        break;
                    }
                    i2 = i5;
                    i4 = i6;
                    break;
                case 3:
                    i5 = 1;
                    i6 = 0;
                    if ((i & i21) != 0) {
                        zzd = zzii.zze(i22, unsafe2.getLong(t, j2));
                        i2 = i5;
                        i4 = i6;
                        i20 += zzd;
                        break;
                    }
                    i2 = i5;
                    i4 = i6;
                    break;
                case 4:
                    i5 = 1;
                    i6 = 0;
                    if ((i & i21) != 0) {
                        zzd = zzii.zzf(i22, unsafe2.getInt(t, j2));
                        i2 = i5;
                        i4 = i6;
                        i20 += zzd;
                        break;
                    }
                    i2 = i5;
                    i4 = i6;
                    break;
                case 5:
                    i2 = 1;
                    i4 = 0;
                    if ((i & i21) != 0) {
                        zzd = zzii.zzg(i22, 0L);
                        i20 += zzd;
                        break;
                    }
                    break;
                case 6:
                    i2 = 1;
                    int i26 = i21 & i;
                    i7 = 0;
                    if (i26 != 0) {
                        i20 += zzii.zzi(i22, 0);
                    }
                    i4 = i7;
                    break;
                case 7:
                    if ((i21 & i) != 0) {
                        i2 = 1;
                        i3 = zzii.zzb(i22, true) + i20;
                        i7 = 0;
                        i20 = i3;
                        i4 = i7;
                        break;
                    }
                    i2 = 1;
                    i3 = i20;
                    i7 = 0;
                    i20 = i3;
                    i4 = i7;
                case 8:
                    if ((i & i21) != 0) {
                        Object object = unsafe2.getObject(t, j2);
                        if (object instanceof zzht) {
                            zzb2 = zzii.zzc(i22, (zzht) object);
                        } else {
                            zzb2 = zzii.zzb(i22, (String) object);
                        }
                        i20 += zzb2;
                    }
                    i2 = 1;
                    i3 = i20;
                    i7 = 0;
                    i20 = i3;
                    i4 = i7;
                    break;
                case 9:
                    if ((i & i21) != 0) {
                        zzb2 = zzle.zza(i22, unsafe2.getObject(t, j2), zza(i12));
                        i20 += zzb2;
                    }
                    i2 = 1;
                    i3 = i20;
                    i7 = 0;
                    i20 = i3;
                    i4 = i7;
                    break;
                case 10:
                    if ((i & i21) != 0) {
                        zzb2 = zzii.zzc(i22, (zzht) unsafe2.getObject(t, j2));
                        i20 += zzb2;
                    }
                    i2 = 1;
                    i3 = i20;
                    i7 = 0;
                    i20 = i3;
                    i4 = i7;
                    break;
                case 11:
                    if ((i & i21) != 0) {
                        zzb2 = zzii.zzg(i22, unsafe2.getInt(t, j2));
                        i20 += zzb2;
                    }
                    i2 = 1;
                    i3 = i20;
                    i7 = 0;
                    i20 = i3;
                    i4 = i7;
                    break;
                case 12:
                    if ((i & i21) != 0) {
                        zzb2 = zzii.zzk(i22, unsafe2.getInt(t, j2));
                        i20 += zzb2;
                    }
                    i2 = 1;
                    i3 = i20;
                    i7 = 0;
                    i20 = i3;
                    i4 = i7;
                    break;
                case 13:
                    if ((i21 & i) != 0) {
                        zzb2 = zzii.zzj(i22, 0);
                        i20 += zzb2;
                    }
                    i2 = 1;
                    i3 = i20;
                    i7 = 0;
                    i20 = i3;
                    i4 = i7;
                    break;
                case 14:
                    if ((i21 & i) != 0) {
                        zzb2 = zzii.zzh(i22, 0L);
                        i20 += zzb2;
                    }
                    i2 = 1;
                    i3 = i20;
                    i7 = 0;
                    i20 = i3;
                    i4 = i7;
                    break;
                case 15:
                    if ((i & i21) != 0) {
                        zzb2 = zzii.zzh(i22, unsafe2.getInt(t, j2));
                        i20 += zzb2;
                    }
                    i2 = 1;
                    i3 = i20;
                    i7 = 0;
                    i20 = i3;
                    i4 = i7;
                    break;
                case 16:
                    if ((i & i21) != 0) {
                        zzb2 = zzii.zzf(i22, unsafe2.getLong(t, j2));
                        i20 += zzb2;
                    }
                    i2 = 1;
                    i3 = i20;
                    i7 = 0;
                    i20 = i3;
                    i4 = i7;
                    break;
                case 17:
                    if ((i & i21) != 0) {
                        zzb2 = zzii.zzc(i22, (zzkk) unsafe2.getObject(t, j2), zza(i12));
                        i20 += zzb2;
                    }
                    i2 = 1;
                    i3 = i20;
                    i7 = 0;
                    i20 = i3;
                    i4 = i7;
                    break;
                case 18:
                    zzb2 = zzle.zzi(i22, (List) unsafe2.getObject(t, j2), false);
                    i20 += zzb2;
                    i2 = 1;
                    i3 = i20;
                    i7 = 0;
                    i20 = i3;
                    i4 = i7;
                    break;
                case 19:
                    i8 = 0;
                    zzh = zzle.zzh(i22, (List) unsafe2.getObject(t, j2), false);
                    i4 = i8;
                    i20 += zzh;
                    i2 = 1;
                    break;
                case 20:
                    i8 = 0;
                    zzh = zzle.zza(i22, (List<Long>) unsafe2.getObject(t, j2), false);
                    i4 = i8;
                    i20 += zzh;
                    i2 = 1;
                    break;
                case 21:
                    i8 = 0;
                    zzh = zzle.zzb(i22, (List<Long>) unsafe2.getObject(t, j2), false);
                    i4 = i8;
                    i20 += zzh;
                    i2 = 1;
                    break;
                case 22:
                    i8 = 0;
                    zzh = zzle.zze(i22, (List) unsafe2.getObject(t, j2), false);
                    i4 = i8;
                    i20 += zzh;
                    i2 = 1;
                    break;
                case 23:
                    i8 = 0;
                    zzh = zzle.zzi(i22, (List) unsafe2.getObject(t, j2), false);
                    i4 = i8;
                    i20 += zzh;
                    i2 = 1;
                    break;
                case 24:
                    i8 = 0;
                    zzh = zzle.zzh(i22, (List) unsafe2.getObject(t, j2), false);
                    i4 = i8;
                    i20 += zzh;
                    i2 = 1;
                    break;
                case 25:
                    i4 = 0;
                    zzh = zzle.zzj(i22, (List) unsafe2.getObject(t, j2), false);
                    i20 += zzh;
                    i2 = 1;
                    break;
                case 26:
                    zzb2 = zzle.zza(i22, (List) unsafe2.getObject(t, j2));
                    i20 += zzb2;
                    i2 = 1;
                    i3 = i20;
                    i7 = 0;
                    i20 = i3;
                    i4 = i7;
                    break;
                case 27:
                    zzb2 = zzle.zza(i22, (List<?>) unsafe2.getObject(t, j2), zza(i12));
                    i20 += zzb2;
                    i2 = 1;
                    i3 = i20;
                    i7 = 0;
                    i20 = i3;
                    i4 = i7;
                    break;
                case 28:
                    zzb2 = zzle.zzb(i22, (List) unsafe2.getObject(t, j2));
                    i20 += zzb2;
                    i2 = 1;
                    i3 = i20;
                    i7 = 0;
                    i20 = i3;
                    i4 = i7;
                    break;
                case 29:
                    zzb2 = zzle.zzf(i22, (List) unsafe2.getObject(t, j2), false);
                    i20 += zzb2;
                    i2 = 1;
                    i3 = i20;
                    i7 = 0;
                    i20 = i3;
                    i4 = i7;
                    break;
                case 30:
                    i8 = 0;
                    zzh = zzle.zzd(i22, (List) unsafe2.getObject(t, j2), false);
                    i4 = i8;
                    i20 += zzh;
                    i2 = 1;
                    break;
                case 31:
                    i8 = 0;
                    zzh = zzle.zzh(i22, (List) unsafe2.getObject(t, j2), false);
                    i4 = i8;
                    i20 += zzh;
                    i2 = 1;
                    break;
                case 32:
                    i8 = 0;
                    zzh = zzle.zzi(i22, (List) unsafe2.getObject(t, j2), false);
                    i4 = i8;
                    i20 += zzh;
                    i2 = 1;
                    break;
                case 33:
                    i8 = 0;
                    zzh = zzle.zzg(i22, (List) unsafe2.getObject(t, j2), false);
                    i4 = i8;
                    i20 += zzh;
                    i2 = 1;
                    break;
                case 34:
                    i4 = 0;
                    zzh = zzle.zzc(i22, (List) unsafe2.getObject(t, j2), false);
                    i20 += zzh;
                    i2 = 1;
                    break;
                case 35:
                    zzi = zzle.zzi((List) unsafe2.getObject(t, j2));
                    if (zzi > 0) {
                        zze = zzii.zze(i22);
                        zzg = zzii.zzg(zzi);
                        i20 += zzg + zze + zzi;
                    }
                    i2 = 1;
                    i3 = i20;
                    i7 = 0;
                    i20 = i3;
                    i4 = i7;
                    break;
                case 36:
                    zzi = zzle.zzh((List) unsafe2.getObject(t, j2));
                    if (zzi > 0) {
                        zze = zzii.zze(i22);
                        zzg = zzii.zzg(zzi);
                        i20 += zzg + zze + zzi;
                    }
                    i2 = 1;
                    i3 = i20;
                    i7 = 0;
                    i20 = i3;
                    i4 = i7;
                    break;
                case 37:
                    zzi = zzle.zza((List) unsafe2.getObject(t, j2));
                    if (zzi > 0) {
                        zze = zzii.zze(i22);
                        zzg = zzii.zzg(zzi);
                        i20 += zzg + zze + zzi;
                    }
                    i2 = 1;
                    i3 = i20;
                    i7 = 0;
                    i20 = i3;
                    i4 = i7;
                    break;
                case 38:
                    zzi = zzle.zzb((List) unsafe2.getObject(t, j2));
                    if (zzi > 0) {
                        zze = zzii.zze(i22);
                        zzg = zzii.zzg(zzi);
                        i20 += zzg + zze + zzi;
                    }
                    i2 = 1;
                    i3 = i20;
                    i7 = 0;
                    i20 = i3;
                    i4 = i7;
                    break;
                case 39:
                    zzi = zzle.zze((List) unsafe2.getObject(t, j2));
                    if (zzi > 0) {
                        zze = zzii.zze(i22);
                        zzg = zzii.zzg(zzi);
                        i20 += zzg + zze + zzi;
                    }
                    i2 = 1;
                    i3 = i20;
                    i7 = 0;
                    i20 = i3;
                    i4 = i7;
                    break;
                case 40:
                    zzi = zzle.zzi((List) unsafe2.getObject(t, j2));
                    if (zzi > 0) {
                        zze = zzii.zze(i22);
                        zzg = zzii.zzg(zzi);
                        i20 += zzg + zze + zzi;
                    }
                    i2 = 1;
                    i3 = i20;
                    i7 = 0;
                    i20 = i3;
                    i4 = i7;
                    break;
                case 41:
                    zzi = zzle.zzh((List) unsafe2.getObject(t, j2));
                    if (zzi > 0) {
                        zze = zzii.zze(i22);
                        zzg = zzii.zzg(zzi);
                        i20 += zzg + zze + zzi;
                    }
                    i2 = 1;
                    i3 = i20;
                    i7 = 0;
                    i20 = i3;
                    i4 = i7;
                    break;
                case 42:
                    zzi = zzle.zzj((List) unsafe2.getObject(t, j2));
                    if (zzi > 0) {
                        zze = zzii.zze(i22);
                        zzg = zzii.zzg(zzi);
                        i20 += zzg + zze + zzi;
                    }
                    i2 = 1;
                    i3 = i20;
                    i7 = 0;
                    i20 = i3;
                    i4 = i7;
                    break;
                case 43:
                    zzi = zzle.zzf((List) unsafe2.getObject(t, j2));
                    if (zzi > 0) {
                        zze = zzii.zze(i22);
                        zzg = zzii.zzg(zzi);
                        i20 += zzg + zze + zzi;
                    }
                    i2 = 1;
                    i3 = i20;
                    i7 = 0;
                    i20 = i3;
                    i4 = i7;
                    break;
                case 44:
                    zzi = zzle.zzd((List) unsafe2.getObject(t, j2));
                    if (zzi > 0) {
                        zze = zzii.zze(i22);
                        zzg = zzii.zzg(zzi);
                        i20 += zzg + zze + zzi;
                    }
                    i2 = 1;
                    i3 = i20;
                    i7 = 0;
                    i20 = i3;
                    i4 = i7;
                    break;
                case 45:
                    zzi = zzle.zzh((List) unsafe2.getObject(t, j2));
                    if (zzi > 0) {
                        zze = zzii.zze(i22);
                        zzg = zzii.zzg(zzi);
                        i20 += zzg + zze + zzi;
                    }
                    i2 = 1;
                    i3 = i20;
                    i7 = 0;
                    i20 = i3;
                    i4 = i7;
                    break;
                case 46:
                    zzi = zzle.zzi((List) unsafe2.getObject(t, j2));
                    if (zzi > 0) {
                        zze = zzii.zze(i22);
                        zzg = zzii.zzg(zzi);
                        i20 += zzg + zze + zzi;
                    }
                    i2 = 1;
                    i3 = i20;
                    i7 = 0;
                    i20 = i3;
                    i4 = i7;
                    break;
                case 47:
                    zzi = zzle.zzg((List) unsafe2.getObject(t, j2));
                    if (zzi > 0) {
                        zze = zzii.zze(i22);
                        zzg = zzii.zzg(zzi);
                        i20 += zzg + zze + zzi;
                    }
                    i2 = 1;
                    i3 = i20;
                    i7 = 0;
                    i20 = i3;
                    i4 = i7;
                    break;
                case 48:
                    zzi = zzle.zzc((List) unsafe2.getObject(t, j2));
                    if (zzi > 0) {
                        zze = zzii.zze(i22);
                        zzg = zzii.zzg(zzi);
                        i20 += zzg + zze + zzi;
                    }
                    i2 = 1;
                    i3 = i20;
                    i7 = 0;
                    i20 = i3;
                    i4 = i7;
                    break;
                case 49:
                    zzb2 = zzle.zzb(i22, (List) unsafe2.getObject(t, j2), zza(i12));
                    i20 += zzb2;
                    i2 = 1;
                    i3 = i20;
                    i7 = 0;
                    i20 = i3;
                    i4 = i7;
                    break;
                case 50:
                    zzb2 = this.zzs.zza(i22, unsafe2.getObject(t, j2), zzb(i12));
                    i20 += zzb2;
                    i2 = 1;
                    i3 = i20;
                    i7 = 0;
                    i20 = i3;
                    i4 = i7;
                    break;
                case 51:
                    if (zza((zzko<T>) t, i22, i12)) {
                        zzb2 = zzii.zzb(i22, (double) ShadowDrawableWrapper.COS_45);
                        i20 += zzb2;
                    }
                    i2 = 1;
                    i3 = i20;
                    i7 = 0;
                    i20 = i3;
                    i4 = i7;
                    break;
                case 52:
                    if (zza((zzko<T>) t, i22, i12)) {
                        zzb2 = zzii.zzb(i22, (float) StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
                        i20 += zzb2;
                    }
                    i2 = 1;
                    i3 = i20;
                    i7 = 0;
                    i20 = i3;
                    i4 = i7;
                    break;
                case 53:
                    if (zza((zzko<T>) t, i22, i12)) {
                        zzb2 = zzii.zzd(i22, zze(t, j2));
                        i20 += zzb2;
                    }
                    i2 = 1;
                    i3 = i20;
                    i7 = 0;
                    i20 = i3;
                    i4 = i7;
                    break;
                case 54:
                    if (zza((zzko<T>) t, i22, i12)) {
                        zzb2 = zzii.zze(i22, zze(t, j2));
                        i20 += zzb2;
                    }
                    i2 = 1;
                    i3 = i20;
                    i7 = 0;
                    i20 = i3;
                    i4 = i7;
                    break;
                case 55:
                    if (zza((zzko<T>) t, i22, i12)) {
                        zzb2 = zzii.zzf(i22, zzd(t, j2));
                        i20 += zzb2;
                    }
                    i2 = 1;
                    i3 = i20;
                    i7 = 0;
                    i20 = i3;
                    i4 = i7;
                    break;
                case 56:
                    if (zza((zzko<T>) t, i22, i12)) {
                        zzb2 = zzii.zzg(i22, 0L);
                        i20 += zzb2;
                    }
                    i2 = 1;
                    i3 = i20;
                    i7 = 0;
                    i20 = i3;
                    i4 = i7;
                    break;
                case 57:
                    if (zza((zzko<T>) t, i22, i12)) {
                        zzb2 = zzii.zzi(i22, 0);
                        i20 += zzb2;
                    }
                    i2 = 1;
                    i3 = i20;
                    i7 = 0;
                    i20 = i3;
                    i4 = i7;
                    break;
                case 58:
                    if (zza((zzko<T>) t, i22, i12)) {
                        zzb2 = zzii.zzb(i22, true);
                        i20 += zzb2;
                    }
                    i2 = 1;
                    i3 = i20;
                    i7 = 0;
                    i20 = i3;
                    i4 = i7;
                    break;
                case 59:
                    if (zza((zzko<T>) t, i22, i12)) {
                        Object object2 = unsafe2.getObject(t, j2);
                        if (object2 instanceof zzht) {
                            zzb2 = zzii.zzc(i22, (zzht) object2);
                        } else {
                            zzb2 = zzii.zzb(i22, (String) object2);
                        }
                        i20 += zzb2;
                    }
                    i2 = 1;
                    i3 = i20;
                    i7 = 0;
                    i20 = i3;
                    i4 = i7;
                    break;
                case 60:
                    if (zza((zzko<T>) t, i22, i12)) {
                        zzb2 = zzle.zza(i22, unsafe2.getObject(t, j2), zza(i12));
                        i20 += zzb2;
                    }
                    i2 = 1;
                    i3 = i20;
                    i7 = 0;
                    i20 = i3;
                    i4 = i7;
                    break;
                case 61:
                    if (zza((zzko<T>) t, i22, i12)) {
                        zzb2 = zzii.zzc(i22, (zzht) unsafe2.getObject(t, j2));
                        i20 += zzb2;
                    }
                    i2 = 1;
                    i3 = i20;
                    i7 = 0;
                    i20 = i3;
                    i4 = i7;
                    break;
                case 62:
                    if (zza((zzko<T>) t, i22, i12)) {
                        zzb2 = zzii.zzg(i22, zzd(t, j2));
                        i20 += zzb2;
                    }
                    i2 = 1;
                    i3 = i20;
                    i7 = 0;
                    i20 = i3;
                    i4 = i7;
                    break;
                case 63:
                    if (zza((zzko<T>) t, i22, i12)) {
                        zzb2 = zzii.zzk(i22, zzd(t, j2));
                        i20 += zzb2;
                    }
                    i2 = 1;
                    i3 = i20;
                    i7 = 0;
                    i20 = i3;
                    i4 = i7;
                    break;
                case 64:
                    if (zza((zzko<T>) t, i22, i12)) {
                        zzb2 = zzii.zzj(i22, 0);
                        i20 += zzb2;
                    }
                    i2 = 1;
                    i3 = i20;
                    i7 = 0;
                    i20 = i3;
                    i4 = i7;
                    break;
                case 65:
                    if (zza((zzko<T>) t, i22, i12)) {
                        zzb2 = zzii.zzh(i22, 0L);
                        i20 += zzb2;
                    }
                    i2 = 1;
                    i3 = i20;
                    i7 = 0;
                    i20 = i3;
                    i4 = i7;
                    break;
                case 66:
                    if (zza((zzko<T>) t, i22, i12)) {
                        zzb2 = zzii.zzh(i22, zzd(t, j2));
                        i20 += zzb2;
                    }
                    i2 = 1;
                    i3 = i20;
                    i7 = 0;
                    i20 = i3;
                    i4 = i7;
                    break;
                case 67:
                    if (zza((zzko<T>) t, i22, i12)) {
                        zzb2 = zzii.zzf(i22, zze(t, j2));
                        i20 += zzb2;
                    }
                    i2 = 1;
                    i3 = i20;
                    i7 = 0;
                    i20 = i3;
                    i4 = i7;
                    break;
                case 68:
                    if (zza((zzko<T>) t, i22, i12)) {
                        zzb2 = zzii.zzc(i22, (zzkk) unsafe2.getObject(t, j2), zza(i12));
                        i20 += zzb2;
                    }
                    i2 = 1;
                    i3 = i20;
                    i7 = 0;
                    i20 = i3;
                    i4 = i7;
                    break;
                default:
                    i2 = 1;
                    i3 = i20;
                    i7 = 0;
                    i20 = i3;
                    i4 = i7;
                    break;
            }
            i11 = i2;
            i19 = i4;
            i12 += 3;
            i10 = 1048575;
        }
        int zza2 = i20 + zza((zzlu) this.zzq, (Object) t);
        if (this.zzh) {
            zziu<?> zza3 = this.zzr.zza(t);
            int i27 = i19;
            while (i19 < zza3.zza.zzc()) {
                Map.Entry<?, Object> zzb4 = zza3.zza.zzb(i19);
                i27 += zziu.zzc((zziw) zzb4.getKey(), zzb4.getValue());
                i19++;
            }
            for (Map.Entry<?, Object> entry : zza3.zza.zzd()) {
                i27 += zziu.zzc((zziw) entry.getKey(), entry.getValue());
            }
            return zza2 + i27;
        }
        return zza2;
    }

    private static Field zza(Class<?> cls, String str) {
        try {
            return cls.getDeclaredField(str);
        } catch (NoSuchFieldException unused) {
            Field[] declaredFields = cls.getDeclaredFields();
            for (Field field : declaredFields) {
                if (str.equals(field.getName())) {
                    return field;
                }
            }
            String name = cls.getName();
            String arrays = Arrays.toString(declaredFields);
            StringBuilder sb = new StringBuilder(String.valueOf(arrays).length() + name.length() + String.valueOf(str).length() + 40);
            sb.append("Field ");
            sb.append(str);
            sb.append(" for ");
            sb.append(name);
            throw new RuntimeException(a.v(sb, " not found. Known fields are ", arrays));
        }
    }

    @Override // com.google.android.gms.internal.vision.zzlc
    public final T zza() {
        return (T) this.zzo.zza(this.zzg);
    }

    /* JADX WARN: Code restructure failed: missing block: B:103:0x01bf, code lost:
        if (java.lang.Double.doubleToLongBits(com.google.android.gms.internal.vision.zzma.zze(r10, r6)) == java.lang.Double.doubleToLongBits(com.google.android.gms.internal.vision.zzma.zze(r11, r6))) goto L85;
     */
    /* JADX WARN: Code restructure failed: missing block: B:11:0x0038, code lost:
        if (com.google.android.gms.internal.vision.zzle.zza(com.google.android.gms.internal.vision.zzma.zzf(r10, r6), com.google.android.gms.internal.vision.zzma.zzf(r11, r6)) != false) goto L85;
     */
    /* JADX WARN: Code restructure failed: missing block: B:18:0x006a, code lost:
        if (com.google.android.gms.internal.vision.zzle.zza(com.google.android.gms.internal.vision.zzma.zzf(r10, r6), com.google.android.gms.internal.vision.zzma.zzf(r11, r6)) != false) goto L85;
     */
    /* JADX WARN: Code restructure failed: missing block: B:23:0x007e, code lost:
        if (com.google.android.gms.internal.vision.zzma.zzb(r10, r6) == com.google.android.gms.internal.vision.zzma.zzb(r11, r6)) goto L85;
     */
    /* JADX WARN: Code restructure failed: missing block: B:28:0x0090, code lost:
        if (com.google.android.gms.internal.vision.zzma.zza(r10, r6) == com.google.android.gms.internal.vision.zzma.zza(r11, r6)) goto L85;
     */
    /* JADX WARN: Code restructure failed: missing block: B:33:0x00a4, code lost:
        if (com.google.android.gms.internal.vision.zzma.zzb(r10, r6) == com.google.android.gms.internal.vision.zzma.zzb(r11, r6)) goto L85;
     */
    /* JADX WARN: Code restructure failed: missing block: B:38:0x00b6, code lost:
        if (com.google.android.gms.internal.vision.zzma.zza(r10, r6) == com.google.android.gms.internal.vision.zzma.zza(r11, r6)) goto L85;
     */
    /* JADX WARN: Code restructure failed: missing block: B:43:0x00c8, code lost:
        if (com.google.android.gms.internal.vision.zzma.zza(r10, r6) == com.google.android.gms.internal.vision.zzma.zza(r11, r6)) goto L85;
     */
    /* JADX WARN: Code restructure failed: missing block: B:48:0x00da, code lost:
        if (com.google.android.gms.internal.vision.zzma.zza(r10, r6) == com.google.android.gms.internal.vision.zzma.zza(r11, r6)) goto L85;
     */
    /* JADX WARN: Code restructure failed: missing block: B:53:0x00f0, code lost:
        if (com.google.android.gms.internal.vision.zzle.zza(com.google.android.gms.internal.vision.zzma.zzf(r10, r6), com.google.android.gms.internal.vision.zzma.zzf(r11, r6)) != false) goto L85;
     */
    /* JADX WARN: Code restructure failed: missing block: B:58:0x0106, code lost:
        if (com.google.android.gms.internal.vision.zzle.zza(com.google.android.gms.internal.vision.zzma.zzf(r10, r6), com.google.android.gms.internal.vision.zzma.zzf(r11, r6)) != false) goto L85;
     */
    /* JADX WARN: Code restructure failed: missing block: B:63:0x011c, code lost:
        if (com.google.android.gms.internal.vision.zzle.zza(com.google.android.gms.internal.vision.zzma.zzf(r10, r6), com.google.android.gms.internal.vision.zzma.zzf(r11, r6)) != false) goto L85;
     */
    /* JADX WARN: Code restructure failed: missing block: B:68:0x012e, code lost:
        if (com.google.android.gms.internal.vision.zzma.zzc(r10, r6) == com.google.android.gms.internal.vision.zzma.zzc(r11, r6)) goto L85;
     */
    /* JADX WARN: Code restructure failed: missing block: B:73:0x0140, code lost:
        if (com.google.android.gms.internal.vision.zzma.zza(r10, r6) == com.google.android.gms.internal.vision.zzma.zza(r11, r6)) goto L85;
     */
    /* JADX WARN: Code restructure failed: missing block: B:78:0x0154, code lost:
        if (com.google.android.gms.internal.vision.zzma.zzb(r10, r6) == com.google.android.gms.internal.vision.zzma.zzb(r11, r6)) goto L85;
     */
    /* JADX WARN: Code restructure failed: missing block: B:83:0x0165, code lost:
        if (com.google.android.gms.internal.vision.zzma.zza(r10, r6) == com.google.android.gms.internal.vision.zzma.zza(r11, r6)) goto L85;
     */
    /* JADX WARN: Code restructure failed: missing block: B:88:0x0178, code lost:
        if (com.google.android.gms.internal.vision.zzma.zzb(r10, r6) == com.google.android.gms.internal.vision.zzma.zzb(r11, r6)) goto L85;
     */
    /* JADX WARN: Code restructure failed: missing block: B:93:0x018b, code lost:
        if (com.google.android.gms.internal.vision.zzma.zzb(r10, r6) == com.google.android.gms.internal.vision.zzma.zzb(r11, r6)) goto L85;
     */
    /* JADX WARN: Code restructure failed: missing block: B:98:0x01a4, code lost:
        if (java.lang.Float.floatToIntBits(com.google.android.gms.internal.vision.zzma.zzd(r10, r6)) == java.lang.Float.floatToIntBits(com.google.android.gms.internal.vision.zzma.zzd(r11, r6))) goto L85;
     */
    @Override // com.google.android.gms.internal.vision.zzlc
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public final boolean zza(T t, T t2) {
        int length = this.zzc.length;
        int i = 0;
        while (true) {
            boolean z = true;
            if (i < length) {
                int zzd = zzd(i);
                long j = zzd & 1048575;
                switch ((zzd & 267386880) >>> 20) {
                    case 0:
                        if (zzc(t, t2, i)) {
                            break;
                        }
                        z = false;
                        break;
                    case 1:
                        if (zzc(t, t2, i)) {
                            break;
                        }
                        z = false;
                        break;
                    case 2:
                        if (zzc(t, t2, i)) {
                            break;
                        }
                        z = false;
                        break;
                    case 3:
                        if (zzc(t, t2, i)) {
                            break;
                        }
                        z = false;
                        break;
                    case 4:
                        if (zzc(t, t2, i)) {
                            break;
                        }
                        z = false;
                        break;
                    case 5:
                        if (zzc(t, t2, i)) {
                            break;
                        }
                        z = false;
                        break;
                    case 6:
                        if (zzc(t, t2, i)) {
                            break;
                        }
                        z = false;
                        break;
                    case 7:
                        if (zzc(t, t2, i)) {
                            break;
                        }
                        z = false;
                        break;
                    case 8:
                        if (zzc(t, t2, i)) {
                            break;
                        }
                        z = false;
                        break;
                    case 9:
                        if (zzc(t, t2, i)) {
                            break;
                        }
                        z = false;
                        break;
                    case 10:
                        if (zzc(t, t2, i)) {
                            break;
                        }
                        z = false;
                        break;
                    case 11:
                        if (zzc(t, t2, i)) {
                            break;
                        }
                        z = false;
                        break;
                    case 12:
                        if (zzc(t, t2, i)) {
                            break;
                        }
                        z = false;
                        break;
                    case 13:
                        if (zzc(t, t2, i)) {
                            break;
                        }
                        z = false;
                        break;
                    case 14:
                        if (zzc(t, t2, i)) {
                            break;
                        }
                        z = false;
                        break;
                    case 15:
                        if (zzc(t, t2, i)) {
                            break;
                        }
                        z = false;
                        break;
                    case 16:
                        if (zzc(t, t2, i)) {
                            break;
                        }
                        z = false;
                        break;
                    case 17:
                        if (zzc(t, t2, i)) {
                            break;
                        }
                        z = false;
                        break;
                    case 18:
                    case 19:
                    case 20:
                    case 21:
                    case 22:
                    case 23:
                    case 24:
                    case 25:
                    case 26:
                    case 27:
                    case 28:
                    case 29:
                    case 30:
                    case 31:
                    case 32:
                    case 33:
                    case 34:
                    case 35:
                    case 36:
                    case 37:
                    case 38:
                    case 39:
                    case 40:
                    case 41:
                    case 42:
                    case 43:
                    case 44:
                    case 45:
                    case 46:
                    case 47:
                    case 48:
                    case 49:
                        z = zzle.zza(zzma.zzf(t, j), zzma.zzf(t2, j));
                        break;
                    case 50:
                        z = zzle.zza(zzma.zzf(t, j), zzma.zzf(t2, j));
                        break;
                    case 51:
                    case 52:
                    case 53:
                    case 54:
                    case 55:
                    case 56:
                    case 57:
                    case 58:
                    case 59:
                    case 60:
                    case 61:
                    case 62:
                    case 63:
                    case 64:
                    case 65:
                    case 66:
                    case 67:
                    case 68:
                        long zze = zze(i) & 1048575;
                        if (zzma.zza(t, zze) == zzma.zza(t2, zze)) {
                            break;
                        }
                        z = false;
                        break;
                }
                if (!z) {
                    return false;
                }
                i += 3;
            } else if (this.zzq.zzb(t).equals(this.zzq.zzb(t2))) {
                if (this.zzh) {
                    return this.zzr.zza(t).equals(this.zzr.zza(t2));
                }
                return true;
            } else {
                return false;
            }
        }
    }

    @Override // com.google.android.gms.internal.vision.zzlc
    public final int zza(T t) {
        int i;
        int zza2;
        int i2;
        int zzd;
        int length = this.zzc.length;
        int i3 = 0;
        for (int i4 = 0; i4 < length; i4 += 3) {
            int zzd2 = zzd(i4);
            int i5 = this.zzc[i4];
            long j = 1048575 & zzd2;
            int i6 = 37;
            switch ((zzd2 & 267386880) >>> 20) {
                case 0:
                    i = i3 * 53;
                    zza2 = zzjf.zza(Double.doubleToLongBits(zzma.zze(t, j)));
                    i3 = zza2 + i;
                    break;
                case 1:
                    i = i3 * 53;
                    zza2 = Float.floatToIntBits(zzma.zzd(t, j));
                    i3 = zza2 + i;
                    break;
                case 2:
                    i = i3 * 53;
                    zza2 = zzjf.zza(zzma.zzb(t, j));
                    i3 = zza2 + i;
                    break;
                case 3:
                    i = i3 * 53;
                    zza2 = zzjf.zza(zzma.zzb(t, j));
                    i3 = zza2 + i;
                    break;
                case 4:
                    i = i3 * 53;
                    zza2 = zzma.zza(t, j);
                    i3 = zza2 + i;
                    break;
                case 5:
                    i = i3 * 53;
                    zza2 = zzjf.zza(zzma.zzb(t, j));
                    i3 = zza2 + i;
                    break;
                case 6:
                    i = i3 * 53;
                    zza2 = zzma.zza(t, j);
                    i3 = zza2 + i;
                    break;
                case 7:
                    i = i3 * 53;
                    zza2 = zzjf.zza(zzma.zzc(t, j));
                    i3 = zza2 + i;
                    break;
                case 8:
                    i = i3 * 53;
                    zza2 = ((String) zzma.zzf(t, j)).hashCode();
                    i3 = zza2 + i;
                    break;
                case 9:
                    Object zzf = zzma.zzf(t, j);
                    if (zzf != null) {
                        i6 = zzf.hashCode();
                    }
                    i3 = (i3 * 53) + i6;
                    break;
                case 10:
                    i = i3 * 53;
                    zza2 = zzma.zzf(t, j).hashCode();
                    i3 = zza2 + i;
                    break;
                case 11:
                    i = i3 * 53;
                    zza2 = zzma.zza(t, j);
                    i3 = zza2 + i;
                    break;
                case 12:
                    i = i3 * 53;
                    zza2 = zzma.zza(t, j);
                    i3 = zza2 + i;
                    break;
                case 13:
                    i = i3 * 53;
                    zza2 = zzma.zza(t, j);
                    i3 = zza2 + i;
                    break;
                case 14:
                    i = i3 * 53;
                    zza2 = zzjf.zza(zzma.zzb(t, j));
                    i3 = zza2 + i;
                    break;
                case 15:
                    i = i3 * 53;
                    zza2 = zzma.zza(t, j);
                    i3 = zza2 + i;
                    break;
                case 16:
                    i = i3 * 53;
                    zza2 = zzjf.zza(zzma.zzb(t, j));
                    i3 = zza2 + i;
                    break;
                case 17:
                    Object zzf2 = zzma.zzf(t, j);
                    if (zzf2 != null) {
                        i6 = zzf2.hashCode();
                    }
                    i3 = (i3 * 53) + i6;
                    break;
                case 18:
                case 19:
                case 20:
                case 21:
                case 22:
                case 23:
                case 24:
                case 25:
                case 26:
                case 27:
                case 28:
                case 29:
                case 30:
                case 31:
                case 32:
                case 33:
                case 34:
                case 35:
                case 36:
                case 37:
                case 38:
                case 39:
                case 40:
                case 41:
                case 42:
                case 43:
                case 44:
                case 45:
                case 46:
                case 47:
                case 48:
                case 49:
                    i = i3 * 53;
                    zza2 = zzma.zzf(t, j).hashCode();
                    i3 = zza2 + i;
                    break;
                case 50:
                    i = i3 * 53;
                    zza2 = zzma.zzf(t, j).hashCode();
                    i3 = zza2 + i;
                    break;
                case 51:
                    if (zza((zzko<T>) t, i5, i4)) {
                        i = i3 * 53;
                        zza2 = zzjf.zza(Double.doubleToLongBits(zzb(t, j)));
                        i3 = zza2 + i;
                        break;
                    } else {
                        break;
                    }
                case 52:
                    if (zza((zzko<T>) t, i5, i4)) {
                        i = i3 * 53;
                        zza2 = Float.floatToIntBits(zzc(t, j));
                        i3 = zza2 + i;
                        break;
                    } else {
                        break;
                    }
                case 53:
                    if (zza((zzko<T>) t, i5, i4)) {
                        i = i3 * 53;
                        zza2 = zzjf.zza(zze(t, j));
                        i3 = zza2 + i;
                        break;
                    } else {
                        break;
                    }
                case 54:
                    if (zza((zzko<T>) t, i5, i4)) {
                        i = i3 * 53;
                        zza2 = zzjf.zza(zze(t, j));
                        i3 = zza2 + i;
                        break;
                    } else {
                        break;
                    }
                case 55:
                    if (zza((zzko<T>) t, i5, i4)) {
                        i2 = i3 * 53;
                        zzd = zzd(t, j);
                        i3 = i2 + zzd;
                        break;
                    } else {
                        break;
                    }
                case 56:
                    if (zza((zzko<T>) t, i5, i4)) {
                        i = i3 * 53;
                        zza2 = zzjf.zza(zze(t, j));
                        i3 = zza2 + i;
                        break;
                    } else {
                        break;
                    }
                case 57:
                    if (zza((zzko<T>) t, i5, i4)) {
                        i2 = i3 * 53;
                        zzd = zzd(t, j);
                        i3 = i2 + zzd;
                        break;
                    } else {
                        break;
                    }
                case 58:
                    if (zza((zzko<T>) t, i5, i4)) {
                        i = i3 * 53;
                        zza2 = zzjf.zza(zzf(t, j));
                        i3 = zza2 + i;
                        break;
                    } else {
                        break;
                    }
                case 59:
                    if (zza((zzko<T>) t, i5, i4)) {
                        i = i3 * 53;
                        zza2 = ((String) zzma.zzf(t, j)).hashCode();
                        i3 = zza2 + i;
                        break;
                    } else {
                        break;
                    }
                case 60:
                    if (zza((zzko<T>) t, i5, i4)) {
                        i = i3 * 53;
                        zza2 = zzma.zzf(t, j).hashCode();
                        i3 = zza2 + i;
                        break;
                    } else {
                        break;
                    }
                case 61:
                    if (zza((zzko<T>) t, i5, i4)) {
                        i = i3 * 53;
                        zza2 = zzma.zzf(t, j).hashCode();
                        i3 = zza2 + i;
                        break;
                    } else {
                        break;
                    }
                case 62:
                    if (zza((zzko<T>) t, i5, i4)) {
                        i2 = i3 * 53;
                        zzd = zzd(t, j);
                        i3 = i2 + zzd;
                        break;
                    } else {
                        break;
                    }
                case 63:
                    if (zza((zzko<T>) t, i5, i4)) {
                        i2 = i3 * 53;
                        zzd = zzd(t, j);
                        i3 = i2 + zzd;
                        break;
                    } else {
                        break;
                    }
                case 64:
                    if (zza((zzko<T>) t, i5, i4)) {
                        i2 = i3 * 53;
                        zzd = zzd(t, j);
                        i3 = i2 + zzd;
                        break;
                    } else {
                        break;
                    }
                case 65:
                    if (zza((zzko<T>) t, i5, i4)) {
                        i = i3 * 53;
                        zza2 = zzjf.zza(zze(t, j));
                        i3 = zza2 + i;
                        break;
                    } else {
                        break;
                    }
                case 66:
                    if (zza((zzko<T>) t, i5, i4)) {
                        i2 = i3 * 53;
                        zzd = zzd(t, j);
                        i3 = i2 + zzd;
                        break;
                    } else {
                        break;
                    }
                case 67:
                    if (zza((zzko<T>) t, i5, i4)) {
                        i = i3 * 53;
                        zza2 = zzjf.zza(zze(t, j));
                        i3 = zza2 + i;
                        break;
                    } else {
                        break;
                    }
                case 68:
                    if (zza((zzko<T>) t, i5, i4)) {
                        i = i3 * 53;
                        zza2 = zzma.zzf(t, j).hashCode();
                        i3 = zza2 + i;
                        break;
                    } else {
                        break;
                    }
            }
        }
        int hashCode = this.zzq.zzb(t).hashCode() + (i3 * 53);
        return this.zzh ? (hashCode * 53) + this.zzr.zza(t).hashCode() : hashCode;
    }

    private final void zza(T t, T t2, int i) {
        long zzd = zzd(i) & 1048575;
        if (zza((zzko<T>) t2, i)) {
            Object zzf = zzma.zzf(t, zzd);
            Object zzf2 = zzma.zzf(t2, zzd);
            if (zzf != null && zzf2 != null) {
                zzma.zza(t, zzd, zzjf.zza(zzf, zzf2));
                zzb((zzko<T>) t, i);
            } else if (zzf2 != null) {
                zzma.zza(t, zzd, zzf2);
                zzb((zzko<T>) t, i);
            }
        }
    }

    private static <UT, UB> int zza(zzlu<UT, UB> zzluVar, T t) {
        return zzluVar.zzf(zzluVar.zzb(t));
    }

    private static List<?> zza(Object obj, long j) {
        return (List) zzma.zzf(obj, j);
    }

    /* JADX WARN: Removed duplicated region for block: B:12:0x003b  */
    /* JADX WARN: Removed duplicated region for block: B:166:0x0513  */
    /* JADX WARN: Removed duplicated region for block: B:181:0x0552  */
    /* JADX WARN: Removed duplicated region for block: B:335:0x0a2a  */
    @Override // com.google.android.gms.internal.vision.zzlc
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public final void zza(T t, zzmr zzmrVar) {
        Iterator<Map.Entry<?, Object>> it;
        Map.Entry<?, Object> entry;
        int length;
        int i;
        Iterator<Map.Entry<?, Object>> it2;
        Map.Entry<?, Object> entry2;
        int length2;
        if (zzmrVar.zza() == zzmq.zzb) {
            zza(this.zzq, t, zzmrVar);
            if (this.zzh) {
                zziu<?> zza2 = this.zzr.zza(t);
                if (!zza2.zza.isEmpty()) {
                    it2 = zza2.zze();
                    entry2 = it2.next();
                    for (length2 = this.zzc.length - 3; length2 >= 0; length2 -= 3) {
                        int zzd = zzd(length2);
                        int i2 = this.zzc[length2];
                        while (entry2 != null && this.zzr.zza(entry2) > i2) {
                            this.zzr.zza(zzmrVar, entry2);
                            entry2 = it2.hasNext() ? it2.next() : null;
                        }
                        switch ((zzd & 267386880) >>> 20) {
                            case 0:
                                if (zza((zzko<T>) t, length2)) {
                                    zzmrVar.zza(i2, zzma.zze(t, zzd & 1048575));
                                    break;
                                } else {
                                    break;
                                }
                            case 1:
                                if (zza((zzko<T>) t, length2)) {
                                    zzmrVar.zza(i2, zzma.zzd(t, zzd & 1048575));
                                    break;
                                } else {
                                    break;
                                }
                            case 2:
                                if (zza((zzko<T>) t, length2)) {
                                    zzmrVar.zza(i2, zzma.zzb(t, zzd & 1048575));
                                    break;
                                } else {
                                    break;
                                }
                            case 3:
                                if (zza((zzko<T>) t, length2)) {
                                    zzmrVar.zzc(i2, zzma.zzb(t, zzd & 1048575));
                                    break;
                                } else {
                                    break;
                                }
                            case 4:
                                if (zza((zzko<T>) t, length2)) {
                                    zzmrVar.zzc(i2, zzma.zza(t, zzd & 1048575));
                                    break;
                                } else {
                                    break;
                                }
                            case 5:
                                if (zza((zzko<T>) t, length2)) {
                                    zzmrVar.zzd(i2, zzma.zzb(t, zzd & 1048575));
                                    break;
                                } else {
                                    break;
                                }
                            case 6:
                                if (zza((zzko<T>) t, length2)) {
                                    zzmrVar.zzd(i2, zzma.zza(t, zzd & 1048575));
                                    break;
                                } else {
                                    break;
                                }
                            case 7:
                                if (zza((zzko<T>) t, length2)) {
                                    zzmrVar.zza(i2, zzma.zzc(t, zzd & 1048575));
                                    break;
                                } else {
                                    break;
                                }
                            case 8:
                                if (zza((zzko<T>) t, length2)) {
                                    zza(i2, zzma.zzf(t, zzd & 1048575), zzmrVar);
                                    break;
                                } else {
                                    break;
                                }
                            case 9:
                                if (zza((zzko<T>) t, length2)) {
                                    zzmrVar.zza(i2, zzma.zzf(t, zzd & 1048575), zza(length2));
                                    break;
                                } else {
                                    break;
                                }
                            case 10:
                                if (zza((zzko<T>) t, length2)) {
                                    zzmrVar.zza(i2, (zzht) zzma.zzf(t, zzd & 1048575));
                                    break;
                                } else {
                                    break;
                                }
                            case 11:
                                if (zza((zzko<T>) t, length2)) {
                                    zzmrVar.zze(i2, zzma.zza(t, zzd & 1048575));
                                    break;
                                } else {
                                    break;
                                }
                            case 12:
                                if (zza((zzko<T>) t, length2)) {
                                    zzmrVar.zzb(i2, zzma.zza(t, zzd & 1048575));
                                    break;
                                } else {
                                    break;
                                }
                            case 13:
                                if (zza((zzko<T>) t, length2)) {
                                    zzmrVar.zza(i2, zzma.zza(t, zzd & 1048575));
                                    break;
                                } else {
                                    break;
                                }
                            case 14:
                                if (zza((zzko<T>) t, length2)) {
                                    zzmrVar.zzb(i2, zzma.zzb(t, zzd & 1048575));
                                    break;
                                } else {
                                    break;
                                }
                            case 15:
                                if (zza((zzko<T>) t, length2)) {
                                    zzmrVar.zzf(i2, zzma.zza(t, zzd & 1048575));
                                    break;
                                } else {
                                    break;
                                }
                            case 16:
                                if (zza((zzko<T>) t, length2)) {
                                    zzmrVar.zze(i2, zzma.zzb(t, zzd & 1048575));
                                    break;
                                } else {
                                    break;
                                }
                            case 17:
                                if (zza((zzko<T>) t, length2)) {
                                    zzmrVar.zzb(i2, zzma.zzf(t, zzd & 1048575), zza(length2));
                                    break;
                                } else {
                                    break;
                                }
                            case 18:
                                zzle.zza(this.zzc[length2], (List<Double>) zzma.zzf(t, zzd & 1048575), zzmrVar, false);
                                break;
                            case 19:
                                zzle.zzb(this.zzc[length2], (List<Float>) zzma.zzf(t, zzd & 1048575), zzmrVar, false);
                                break;
                            case 20:
                                zzle.zzc(this.zzc[length2], (List) zzma.zzf(t, zzd & 1048575), zzmrVar, false);
                                break;
                            case 21:
                                zzle.zzd(this.zzc[length2], (List) zzma.zzf(t, zzd & 1048575), zzmrVar, false);
                                break;
                            case 22:
                                zzle.zzh(this.zzc[length2], (List) zzma.zzf(t, zzd & 1048575), zzmrVar, false);
                                break;
                            case 23:
                                zzle.zzf(this.zzc[length2], (List) zzma.zzf(t, zzd & 1048575), zzmrVar, false);
                                break;
                            case 24:
                                zzle.zzk(this.zzc[length2], (List) zzma.zzf(t, zzd & 1048575), zzmrVar, false);
                                break;
                            case 25:
                                zzle.zzn(this.zzc[length2], (List) zzma.zzf(t, zzd & 1048575), zzmrVar, false);
                                break;
                            case 26:
                                zzle.zza(this.zzc[length2], (List) zzma.zzf(t, zzd & 1048575), zzmrVar);
                                break;
                            case 27:
                                zzle.zza(this.zzc[length2], (List) zzma.zzf(t, zzd & 1048575), zzmrVar, zza(length2));
                                break;
                            case 28:
                                zzle.zzb(this.zzc[length2], (List) zzma.zzf(t, zzd & 1048575), zzmrVar);
                                break;
                            case 29:
                                zzle.zzi(this.zzc[length2], (List) zzma.zzf(t, zzd & 1048575), zzmrVar, false);
                                break;
                            case 30:
                                zzle.zzm(this.zzc[length2], (List) zzma.zzf(t, zzd & 1048575), zzmrVar, false);
                                break;
                            case 31:
                                zzle.zzl(this.zzc[length2], (List) zzma.zzf(t, zzd & 1048575), zzmrVar, false);
                                break;
                            case 32:
                                zzle.zzg(this.zzc[length2], (List) zzma.zzf(t, zzd & 1048575), zzmrVar, false);
                                break;
                            case 33:
                                zzle.zzj(this.zzc[length2], (List) zzma.zzf(t, zzd & 1048575), zzmrVar, false);
                                break;
                            case 34:
                                zzle.zze(this.zzc[length2], (List) zzma.zzf(t, zzd & 1048575), zzmrVar, false);
                                break;
                            case 35:
                                zzle.zza(this.zzc[length2], (List<Double>) zzma.zzf(t, zzd & 1048575), zzmrVar, true);
                                break;
                            case 36:
                                zzle.zzb(this.zzc[length2], (List<Float>) zzma.zzf(t, zzd & 1048575), zzmrVar, true);
                                break;
                            case 37:
                                zzle.zzc(this.zzc[length2], (List) zzma.zzf(t, zzd & 1048575), zzmrVar, true);
                                break;
                            case 38:
                                zzle.zzd(this.zzc[length2], (List) zzma.zzf(t, zzd & 1048575), zzmrVar, true);
                                break;
                            case 39:
                                zzle.zzh(this.zzc[length2], (List) zzma.zzf(t, zzd & 1048575), zzmrVar, true);
                                break;
                            case 40:
                                zzle.zzf(this.zzc[length2], (List) zzma.zzf(t, zzd & 1048575), zzmrVar, true);
                                break;
                            case 41:
                                zzle.zzk(this.zzc[length2], (List) zzma.zzf(t, zzd & 1048575), zzmrVar, true);
                                break;
                            case 42:
                                zzle.zzn(this.zzc[length2], (List) zzma.zzf(t, zzd & 1048575), zzmrVar, true);
                                break;
                            case 43:
                                zzle.zzi(this.zzc[length2], (List) zzma.zzf(t, zzd & 1048575), zzmrVar, true);
                                break;
                            case 44:
                                zzle.zzm(this.zzc[length2], (List) zzma.zzf(t, zzd & 1048575), zzmrVar, true);
                                break;
                            case 45:
                                zzle.zzl(this.zzc[length2], (List) zzma.zzf(t, zzd & 1048575), zzmrVar, true);
                                break;
                            case 46:
                                zzle.zzg(this.zzc[length2], (List) zzma.zzf(t, zzd & 1048575), zzmrVar, true);
                                break;
                            case 47:
                                zzle.zzj(this.zzc[length2], (List) zzma.zzf(t, zzd & 1048575), zzmrVar, true);
                                break;
                            case 48:
                                zzle.zze(this.zzc[length2], (List) zzma.zzf(t, zzd & 1048575), zzmrVar, true);
                                break;
                            case 49:
                                zzle.zzb(this.zzc[length2], (List) zzma.zzf(t, zzd & 1048575), zzmrVar, zza(length2));
                                break;
                            case 50:
                                zza(zzmrVar, i2, zzma.zzf(t, zzd & 1048575), length2);
                                break;
                            case 51:
                                if (zza((zzko<T>) t, i2, length2)) {
                                    zzmrVar.zza(i2, zzb(t, zzd & 1048575));
                                    break;
                                } else {
                                    break;
                                }
                            case 52:
                                if (zza((zzko<T>) t, i2, length2)) {
                                    zzmrVar.zza(i2, zzc(t, zzd & 1048575));
                                    break;
                                } else {
                                    break;
                                }
                            case 53:
                                if (zza((zzko<T>) t, i2, length2)) {
                                    zzmrVar.zza(i2, zze(t, zzd & 1048575));
                                    break;
                                } else {
                                    break;
                                }
                            case 54:
                                if (zza((zzko<T>) t, i2, length2)) {
                                    zzmrVar.zzc(i2, zze(t, zzd & 1048575));
                                    break;
                                } else {
                                    break;
                                }
                            case 55:
                                if (zza((zzko<T>) t, i2, length2)) {
                                    zzmrVar.zzc(i2, zzd(t, zzd & 1048575));
                                    break;
                                } else {
                                    break;
                                }
                            case 56:
                                if (zza((zzko<T>) t, i2, length2)) {
                                    zzmrVar.zzd(i2, zze(t, zzd & 1048575));
                                    break;
                                } else {
                                    break;
                                }
                            case 57:
                                if (zza((zzko<T>) t, i2, length2)) {
                                    zzmrVar.zzd(i2, zzd(t, zzd & 1048575));
                                    break;
                                } else {
                                    break;
                                }
                            case 58:
                                if (zza((zzko<T>) t, i2, length2)) {
                                    zzmrVar.zza(i2, zzf(t, zzd & 1048575));
                                    break;
                                } else {
                                    break;
                                }
                            case 59:
                                if (zza((zzko<T>) t, i2, length2)) {
                                    zza(i2, zzma.zzf(t, zzd & 1048575), zzmrVar);
                                    break;
                                } else {
                                    break;
                                }
                            case 60:
                                if (zza((zzko<T>) t, i2, length2)) {
                                    zzmrVar.zza(i2, zzma.zzf(t, zzd & 1048575), zza(length2));
                                    break;
                                } else {
                                    break;
                                }
                            case 61:
                                if (zza((zzko<T>) t, i2, length2)) {
                                    zzmrVar.zza(i2, (zzht) zzma.zzf(t, zzd & 1048575));
                                    break;
                                } else {
                                    break;
                                }
                            case 62:
                                if (zza((zzko<T>) t, i2, length2)) {
                                    zzmrVar.zze(i2, zzd(t, zzd & 1048575));
                                    break;
                                } else {
                                    break;
                                }
                            case 63:
                                if (zza((zzko<T>) t, i2, length2)) {
                                    zzmrVar.zzb(i2, zzd(t, zzd & 1048575));
                                    break;
                                } else {
                                    break;
                                }
                            case 64:
                                if (zza((zzko<T>) t, i2, length2)) {
                                    zzmrVar.zza(i2, zzd(t, zzd & 1048575));
                                    break;
                                } else {
                                    break;
                                }
                            case 65:
                                if (zza((zzko<T>) t, i2, length2)) {
                                    zzmrVar.zzb(i2, zze(t, zzd & 1048575));
                                    break;
                                } else {
                                    break;
                                }
                            case 66:
                                if (zza((zzko<T>) t, i2, length2)) {
                                    zzmrVar.zzf(i2, zzd(t, zzd & 1048575));
                                    break;
                                } else {
                                    break;
                                }
                            case 67:
                                if (zza((zzko<T>) t, i2, length2)) {
                                    zzmrVar.zze(i2, zze(t, zzd & 1048575));
                                    break;
                                } else {
                                    break;
                                }
                            case 68:
                                if (zza((zzko<T>) t, i2, length2)) {
                                    zzmrVar.zzb(i2, zzma.zzf(t, zzd & 1048575), zza(length2));
                                    break;
                                } else {
                                    break;
                                }
                        }
                    }
                    while (entry2 != null) {
                        this.zzr.zza(zzmrVar, entry2);
                        entry2 = it2.hasNext() ? it2.next() : null;
                    }
                }
            }
            it2 = null;
            entry2 = null;
            while (length2 >= 0) {
            }
            while (entry2 != null) {
            }
        } else if (this.zzj) {
            if (this.zzh) {
                zziu<?> zza3 = this.zzr.zza(t);
                if (!zza3.zza.isEmpty()) {
                    it = zza3.zzd();
                    entry = it.next();
                    length = this.zzc.length;
                    for (i = 0; i < length; i += 3) {
                        int zzd2 = zzd(i);
                        int i3 = this.zzc[i];
                        while (entry != null && this.zzr.zza(entry) <= i3) {
                            this.zzr.zza(zzmrVar, entry);
                            entry = it.hasNext() ? it.next() : null;
                        }
                        switch ((zzd2 & 267386880) >>> 20) {
                            case 0:
                                if (zza((zzko<T>) t, i)) {
                                    zzmrVar.zza(i3, zzma.zze(t, zzd2 & 1048575));
                                    break;
                                } else {
                                    break;
                                }
                            case 1:
                                if (zza((zzko<T>) t, i)) {
                                    zzmrVar.zza(i3, zzma.zzd(t, zzd2 & 1048575));
                                    break;
                                } else {
                                    break;
                                }
                            case 2:
                                if (zza((zzko<T>) t, i)) {
                                    zzmrVar.zza(i3, zzma.zzb(t, zzd2 & 1048575));
                                    break;
                                } else {
                                    break;
                                }
                            case 3:
                                if (zza((zzko<T>) t, i)) {
                                    zzmrVar.zzc(i3, zzma.zzb(t, zzd2 & 1048575));
                                    break;
                                } else {
                                    break;
                                }
                            case 4:
                                if (zza((zzko<T>) t, i)) {
                                    zzmrVar.zzc(i3, zzma.zza(t, zzd2 & 1048575));
                                    break;
                                } else {
                                    break;
                                }
                            case 5:
                                if (zza((zzko<T>) t, i)) {
                                    zzmrVar.zzd(i3, zzma.zzb(t, zzd2 & 1048575));
                                    break;
                                } else {
                                    break;
                                }
                            case 6:
                                if (zza((zzko<T>) t, i)) {
                                    zzmrVar.zzd(i3, zzma.zza(t, zzd2 & 1048575));
                                    break;
                                } else {
                                    break;
                                }
                            case 7:
                                if (zza((zzko<T>) t, i)) {
                                    zzmrVar.zza(i3, zzma.zzc(t, zzd2 & 1048575));
                                    break;
                                } else {
                                    break;
                                }
                            case 8:
                                if (zza((zzko<T>) t, i)) {
                                    zza(i3, zzma.zzf(t, zzd2 & 1048575), zzmrVar);
                                    break;
                                } else {
                                    break;
                                }
                            case 9:
                                if (zza((zzko<T>) t, i)) {
                                    zzmrVar.zza(i3, zzma.zzf(t, zzd2 & 1048575), zza(i));
                                    break;
                                } else {
                                    break;
                                }
                            case 10:
                                if (zza((zzko<T>) t, i)) {
                                    zzmrVar.zza(i3, (zzht) zzma.zzf(t, zzd2 & 1048575));
                                    break;
                                } else {
                                    break;
                                }
                            case 11:
                                if (zza((zzko<T>) t, i)) {
                                    zzmrVar.zze(i3, zzma.zza(t, zzd2 & 1048575));
                                    break;
                                } else {
                                    break;
                                }
                            case 12:
                                if (zza((zzko<T>) t, i)) {
                                    zzmrVar.zzb(i3, zzma.zza(t, zzd2 & 1048575));
                                    break;
                                } else {
                                    break;
                                }
                            case 13:
                                if (zza((zzko<T>) t, i)) {
                                    zzmrVar.zza(i3, zzma.zza(t, zzd2 & 1048575));
                                    break;
                                } else {
                                    break;
                                }
                            case 14:
                                if (zza((zzko<T>) t, i)) {
                                    zzmrVar.zzb(i3, zzma.zzb(t, zzd2 & 1048575));
                                    break;
                                } else {
                                    break;
                                }
                            case 15:
                                if (zza((zzko<T>) t, i)) {
                                    zzmrVar.zzf(i3, zzma.zza(t, zzd2 & 1048575));
                                    break;
                                } else {
                                    break;
                                }
                            case 16:
                                if (zza((zzko<T>) t, i)) {
                                    zzmrVar.zze(i3, zzma.zzb(t, zzd2 & 1048575));
                                    break;
                                } else {
                                    break;
                                }
                            case 17:
                                if (zza((zzko<T>) t, i)) {
                                    zzmrVar.zzb(i3, zzma.zzf(t, zzd2 & 1048575), zza(i));
                                    break;
                                } else {
                                    break;
                                }
                            case 18:
                                zzle.zza(this.zzc[i], (List<Double>) zzma.zzf(t, zzd2 & 1048575), zzmrVar, false);
                                break;
                            case 19:
                                zzle.zzb(this.zzc[i], (List<Float>) zzma.zzf(t, zzd2 & 1048575), zzmrVar, false);
                                break;
                            case 20:
                                zzle.zzc(this.zzc[i], (List) zzma.zzf(t, zzd2 & 1048575), zzmrVar, false);
                                break;
                            case 21:
                                zzle.zzd(this.zzc[i], (List) zzma.zzf(t, zzd2 & 1048575), zzmrVar, false);
                                break;
                            case 22:
                                zzle.zzh(this.zzc[i], (List) zzma.zzf(t, zzd2 & 1048575), zzmrVar, false);
                                break;
                            case 23:
                                zzle.zzf(this.zzc[i], (List) zzma.zzf(t, zzd2 & 1048575), zzmrVar, false);
                                break;
                            case 24:
                                zzle.zzk(this.zzc[i], (List) zzma.zzf(t, zzd2 & 1048575), zzmrVar, false);
                                break;
                            case 25:
                                zzle.zzn(this.zzc[i], (List) zzma.zzf(t, zzd2 & 1048575), zzmrVar, false);
                                break;
                            case 26:
                                zzle.zza(this.zzc[i], (List) zzma.zzf(t, zzd2 & 1048575), zzmrVar);
                                break;
                            case 27:
                                zzle.zza(this.zzc[i], (List) zzma.zzf(t, zzd2 & 1048575), zzmrVar, zza(i));
                                break;
                            case 28:
                                zzle.zzb(this.zzc[i], (List) zzma.zzf(t, zzd2 & 1048575), zzmrVar);
                                break;
                            case 29:
                                zzle.zzi(this.zzc[i], (List) zzma.zzf(t, zzd2 & 1048575), zzmrVar, false);
                                break;
                            case 30:
                                zzle.zzm(this.zzc[i], (List) zzma.zzf(t, zzd2 & 1048575), zzmrVar, false);
                                break;
                            case 31:
                                zzle.zzl(this.zzc[i], (List) zzma.zzf(t, zzd2 & 1048575), zzmrVar, false);
                                break;
                            case 32:
                                zzle.zzg(this.zzc[i], (List) zzma.zzf(t, zzd2 & 1048575), zzmrVar, false);
                                break;
                            case 33:
                                zzle.zzj(this.zzc[i], (List) zzma.zzf(t, zzd2 & 1048575), zzmrVar, false);
                                break;
                            case 34:
                                zzle.zze(this.zzc[i], (List) zzma.zzf(t, zzd2 & 1048575), zzmrVar, false);
                                break;
                            case 35:
                                zzle.zza(this.zzc[i], (List<Double>) zzma.zzf(t, zzd2 & 1048575), zzmrVar, true);
                                break;
                            case 36:
                                zzle.zzb(this.zzc[i], (List<Float>) zzma.zzf(t, zzd2 & 1048575), zzmrVar, true);
                                break;
                            case 37:
                                zzle.zzc(this.zzc[i], (List) zzma.zzf(t, zzd2 & 1048575), zzmrVar, true);
                                break;
                            case 38:
                                zzle.zzd(this.zzc[i], (List) zzma.zzf(t, zzd2 & 1048575), zzmrVar, true);
                                break;
                            case 39:
                                zzle.zzh(this.zzc[i], (List) zzma.zzf(t, zzd2 & 1048575), zzmrVar, true);
                                break;
                            case 40:
                                zzle.zzf(this.zzc[i], (List) zzma.zzf(t, zzd2 & 1048575), zzmrVar, true);
                                break;
                            case 41:
                                zzle.zzk(this.zzc[i], (List) zzma.zzf(t, zzd2 & 1048575), zzmrVar, true);
                                break;
                            case 42:
                                zzle.zzn(this.zzc[i], (List) zzma.zzf(t, zzd2 & 1048575), zzmrVar, true);
                                break;
                            case 43:
                                zzle.zzi(this.zzc[i], (List) zzma.zzf(t, zzd2 & 1048575), zzmrVar, true);
                                break;
                            case 44:
                                zzle.zzm(this.zzc[i], (List) zzma.zzf(t, zzd2 & 1048575), zzmrVar, true);
                                break;
                            case 45:
                                zzle.zzl(this.zzc[i], (List) zzma.zzf(t, zzd2 & 1048575), zzmrVar, true);
                                break;
                            case 46:
                                zzle.zzg(this.zzc[i], (List) zzma.zzf(t, zzd2 & 1048575), zzmrVar, true);
                                break;
                            case 47:
                                zzle.zzj(this.zzc[i], (List) zzma.zzf(t, zzd2 & 1048575), zzmrVar, true);
                                break;
                            case 48:
                                zzle.zze(this.zzc[i], (List) zzma.zzf(t, zzd2 & 1048575), zzmrVar, true);
                                break;
                            case 49:
                                zzle.zzb(this.zzc[i], (List) zzma.zzf(t, zzd2 & 1048575), zzmrVar, zza(i));
                                break;
                            case 50:
                                zza(zzmrVar, i3, zzma.zzf(t, zzd2 & 1048575), i);
                                break;
                            case 51:
                                if (zza((zzko<T>) t, i3, i)) {
                                    zzmrVar.zza(i3, zzb(t, zzd2 & 1048575));
                                    break;
                                } else {
                                    break;
                                }
                            case 52:
                                if (zza((zzko<T>) t, i3, i)) {
                                    zzmrVar.zza(i3, zzc(t, zzd2 & 1048575));
                                    break;
                                } else {
                                    break;
                                }
                            case 53:
                                if (zza((zzko<T>) t, i3, i)) {
                                    zzmrVar.zza(i3, zze(t, zzd2 & 1048575));
                                    break;
                                } else {
                                    break;
                                }
                            case 54:
                                if (zza((zzko<T>) t, i3, i)) {
                                    zzmrVar.zzc(i3, zze(t, zzd2 & 1048575));
                                    break;
                                } else {
                                    break;
                                }
                            case 55:
                                if (zza((zzko<T>) t, i3, i)) {
                                    zzmrVar.zzc(i3, zzd(t, zzd2 & 1048575));
                                    break;
                                } else {
                                    break;
                                }
                            case 56:
                                if (zza((zzko<T>) t, i3, i)) {
                                    zzmrVar.zzd(i3, zze(t, zzd2 & 1048575));
                                    break;
                                } else {
                                    break;
                                }
                            case 57:
                                if (zza((zzko<T>) t, i3, i)) {
                                    zzmrVar.zzd(i3, zzd(t, zzd2 & 1048575));
                                    break;
                                } else {
                                    break;
                                }
                            case 58:
                                if (zza((zzko<T>) t, i3, i)) {
                                    zzmrVar.zza(i3, zzf(t, zzd2 & 1048575));
                                    break;
                                } else {
                                    break;
                                }
                            case 59:
                                if (zza((zzko<T>) t, i3, i)) {
                                    zza(i3, zzma.zzf(t, zzd2 & 1048575), zzmrVar);
                                    break;
                                } else {
                                    break;
                                }
                            case 60:
                                if (zza((zzko<T>) t, i3, i)) {
                                    zzmrVar.zza(i3, zzma.zzf(t, zzd2 & 1048575), zza(i));
                                    break;
                                } else {
                                    break;
                                }
                            case 61:
                                if (zza((zzko<T>) t, i3, i)) {
                                    zzmrVar.zza(i3, (zzht) zzma.zzf(t, zzd2 & 1048575));
                                    break;
                                } else {
                                    break;
                                }
                            case 62:
                                if (zza((zzko<T>) t, i3, i)) {
                                    zzmrVar.zze(i3, zzd(t, zzd2 & 1048575));
                                    break;
                                } else {
                                    break;
                                }
                            case 63:
                                if (zza((zzko<T>) t, i3, i)) {
                                    zzmrVar.zzb(i3, zzd(t, zzd2 & 1048575));
                                    break;
                                } else {
                                    break;
                                }
                            case 64:
                                if (zza((zzko<T>) t, i3, i)) {
                                    zzmrVar.zza(i3, zzd(t, zzd2 & 1048575));
                                    break;
                                } else {
                                    break;
                                }
                            case 65:
                                if (zza((zzko<T>) t, i3, i)) {
                                    zzmrVar.zzb(i3, zze(t, zzd2 & 1048575));
                                    break;
                                } else {
                                    break;
                                }
                            case 66:
                                if (zza((zzko<T>) t, i3, i)) {
                                    zzmrVar.zzf(i3, zzd(t, zzd2 & 1048575));
                                    break;
                                } else {
                                    break;
                                }
                            case 67:
                                if (zza((zzko<T>) t, i3, i)) {
                                    zzmrVar.zze(i3, zze(t, zzd2 & 1048575));
                                    break;
                                } else {
                                    break;
                                }
                            case 68:
                                if (zza((zzko<T>) t, i3, i)) {
                                    zzmrVar.zzb(i3, zzma.zzf(t, zzd2 & 1048575), zza(i));
                                    break;
                                } else {
                                    break;
                                }
                        }
                    }
                    while (entry != null) {
                        this.zzr.zza(zzmrVar, entry);
                        entry = it.hasNext() ? it.next() : null;
                    }
                    zza(this.zzq, t, zzmrVar);
                }
            }
            it = null;
            entry = null;
            length = this.zzc.length;
            while (i < length) {
            }
            while (entry != null) {
            }
            zza(this.zzq, t, zzmrVar);
        } else {
            zzb((zzko<T>) t, zzmrVar);
        }
    }

    /* JADX WARN: Can't fix incorrect switch cases order, some code will duplicate */
    /* JADX WARN: Removed duplicated region for block: B:10:0x0031  */
    /* JADX WARN: Removed duplicated region for block: B:171:0x048b  */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    private final void zzb(T t, zzmr zzmrVar) {
        Iterator<Map.Entry<?, Object>> it;
        Map.Entry<?, Object> entry;
        int length;
        int i;
        int i2;
        boolean z;
        if (this.zzh) {
            zziu<?> zza2 = this.zzr.zza(t);
            if (!zza2.zza.isEmpty()) {
                it = zza2.zzd();
                entry = it.next();
                length = this.zzc.length;
                Unsafe unsafe = zzb;
                int i3 = 1048575;
                int i4 = 0;
                for (i = 0; i < length; i += 3) {
                    int zzd = zzd(i);
                    int[] iArr = this.zzc;
                    int i5 = iArr[i];
                    int i6 = (zzd & 267386880) >>> 20;
                    if (i6 <= 17) {
                        int i7 = iArr[i + 2];
                        int i8 = i7 & 1048575;
                        if (i8 != i3) {
                            i4 = unsafe.getInt(t, i8);
                            i3 = i8;
                        }
                        i2 = 1 << (i7 >>> 20);
                    } else {
                        i2 = 0;
                    }
                    while (entry != null && this.zzr.zza(entry) <= i5) {
                        this.zzr.zza(zzmrVar, entry);
                        entry = it.hasNext() ? it.next() : null;
                    }
                    long j = zzd & 1048575;
                    switch (i6) {
                        case 0:
                            if ((i2 & i4) != 0) {
                                zzmrVar.zza(i5, zzma.zze(t, j));
                                break;
                            } else {
                                break;
                            }
                        case 1:
                            if ((i2 & i4) != 0) {
                                zzmrVar.zza(i5, zzma.zzd(t, j));
                                break;
                            } else {
                                break;
                            }
                        case 2:
                            if ((i2 & i4) != 0) {
                                zzmrVar.zza(i5, unsafe.getLong(t, j));
                                break;
                            } else {
                                break;
                            }
                        case 3:
                            if ((i2 & i4) != 0) {
                                zzmrVar.zzc(i5, unsafe.getLong(t, j));
                                break;
                            } else {
                                break;
                            }
                        case 4:
                            if ((i2 & i4) != 0) {
                                zzmrVar.zzc(i5, unsafe.getInt(t, j));
                                break;
                            } else {
                                break;
                            }
                        case 5:
                            if ((i2 & i4) != 0) {
                                zzmrVar.zzd(i5, unsafe.getLong(t, j));
                                break;
                            } else {
                                break;
                            }
                        case 6:
                            if ((i2 & i4) != 0) {
                                zzmrVar.zzd(i5, unsafe.getInt(t, j));
                                break;
                            } else {
                                break;
                            }
                        case 7:
                            if ((i2 & i4) != 0) {
                                zzmrVar.zza(i5, zzma.zzc(t, j));
                                break;
                            } else {
                                break;
                            }
                        case 8:
                            if ((i2 & i4) != 0) {
                                zza(i5, unsafe.getObject(t, j), zzmrVar);
                                break;
                            } else {
                                break;
                            }
                        case 9:
                            if ((i2 & i4) != 0) {
                                zzmrVar.zza(i5, unsafe.getObject(t, j), zza(i));
                                break;
                            } else {
                                break;
                            }
                        case 10:
                            if ((i2 & i4) != 0) {
                                zzmrVar.zza(i5, (zzht) unsafe.getObject(t, j));
                                break;
                            } else {
                                break;
                            }
                        case 11:
                            if ((i2 & i4) != 0) {
                                zzmrVar.zze(i5, unsafe.getInt(t, j));
                                break;
                            } else {
                                break;
                            }
                        case 12:
                            if ((i2 & i4) != 0) {
                                zzmrVar.zzb(i5, unsafe.getInt(t, j));
                                break;
                            } else {
                                break;
                            }
                        case 13:
                            if ((i2 & i4) != 0) {
                                zzmrVar.zza(i5, unsafe.getInt(t, j));
                                break;
                            } else {
                                break;
                            }
                        case 14:
                            if ((i2 & i4) != 0) {
                                zzmrVar.zzb(i5, unsafe.getLong(t, j));
                                break;
                            } else {
                                break;
                            }
                        case 15:
                            if ((i2 & i4) != 0) {
                                zzmrVar.zzf(i5, unsafe.getInt(t, j));
                                break;
                            } else {
                                break;
                            }
                        case 16:
                            if ((i2 & i4) != 0) {
                                zzmrVar.zze(i5, unsafe.getLong(t, j));
                                break;
                            } else {
                                break;
                            }
                        case 17:
                            if ((i2 & i4) != 0) {
                                zzmrVar.zzb(i5, unsafe.getObject(t, j), zza(i));
                                break;
                            } else {
                                break;
                            }
                        case 18:
                            z = false;
                            zzle.zza(this.zzc[i], (List<Double>) unsafe.getObject(t, j), zzmrVar, false);
                            break;
                        case 19:
                            z = false;
                            zzle.zzb(this.zzc[i], (List<Float>) unsafe.getObject(t, j), zzmrVar, false);
                            break;
                        case 20:
                            z = false;
                            zzle.zzc(this.zzc[i], (List) unsafe.getObject(t, j), zzmrVar, false);
                            break;
                        case 21:
                            z = false;
                            zzle.zzd(this.zzc[i], (List) unsafe.getObject(t, j), zzmrVar, false);
                            break;
                        case 22:
                            z = false;
                            zzle.zzh(this.zzc[i], (List) unsafe.getObject(t, j), zzmrVar, false);
                            break;
                        case 23:
                            z = false;
                            zzle.zzf(this.zzc[i], (List) unsafe.getObject(t, j), zzmrVar, false);
                            break;
                        case 24:
                            z = false;
                            zzle.zzk(this.zzc[i], (List) unsafe.getObject(t, j), zzmrVar, false);
                            break;
                        case 25:
                            z = false;
                            zzle.zzn(this.zzc[i], (List) unsafe.getObject(t, j), zzmrVar, false);
                            break;
                        case 26:
                            zzle.zza(this.zzc[i], (List) unsafe.getObject(t, j), zzmrVar);
                            break;
                        case 27:
                            zzle.zza(this.zzc[i], (List) unsafe.getObject(t, j), zzmrVar, zza(i));
                            break;
                        case 28:
                            zzle.zzb(this.zzc[i], (List) unsafe.getObject(t, j), zzmrVar);
                            break;
                        case 29:
                            z = false;
                            zzle.zzi(this.zzc[i], (List) unsafe.getObject(t, j), zzmrVar, false);
                            break;
                        case 30:
                            z = false;
                            zzle.zzm(this.zzc[i], (List) unsafe.getObject(t, j), zzmrVar, false);
                            break;
                        case 31:
                            z = false;
                            zzle.zzl(this.zzc[i], (List) unsafe.getObject(t, j), zzmrVar, false);
                            break;
                        case 32:
                            z = false;
                            zzle.zzg(this.zzc[i], (List) unsafe.getObject(t, j), zzmrVar, false);
                            break;
                        case 33:
                            z = false;
                            zzle.zzj(this.zzc[i], (List) unsafe.getObject(t, j), zzmrVar, false);
                            break;
                        case 34:
                            z = false;
                            zzle.zze(this.zzc[i], (List) unsafe.getObject(t, j), zzmrVar, false);
                            break;
                        case 35:
                            zzle.zza(this.zzc[i], (List<Double>) unsafe.getObject(t, j), zzmrVar, true);
                            break;
                        case 36:
                            zzle.zzb(this.zzc[i], (List<Float>) unsafe.getObject(t, j), zzmrVar, true);
                            break;
                        case 37:
                            zzle.zzc(this.zzc[i], (List) unsafe.getObject(t, j), zzmrVar, true);
                            break;
                        case 38:
                            zzle.zzd(this.zzc[i], (List) unsafe.getObject(t, j), zzmrVar, true);
                            break;
                        case 39:
                            zzle.zzh(this.zzc[i], (List) unsafe.getObject(t, j), zzmrVar, true);
                            break;
                        case 40:
                            zzle.zzf(this.zzc[i], (List) unsafe.getObject(t, j), zzmrVar, true);
                            break;
                        case 41:
                            zzle.zzk(this.zzc[i], (List) unsafe.getObject(t, j), zzmrVar, true);
                            break;
                        case 42:
                            zzle.zzn(this.zzc[i], (List) unsafe.getObject(t, j), zzmrVar, true);
                            break;
                        case 43:
                            zzle.zzi(this.zzc[i], (List) unsafe.getObject(t, j), zzmrVar, true);
                            break;
                        case 44:
                            zzle.zzm(this.zzc[i], (List) unsafe.getObject(t, j), zzmrVar, true);
                            break;
                        case 45:
                            zzle.zzl(this.zzc[i], (List) unsafe.getObject(t, j), zzmrVar, true);
                            break;
                        case 46:
                            zzle.zzg(this.zzc[i], (List) unsafe.getObject(t, j), zzmrVar, true);
                            break;
                        case 47:
                            zzle.zzj(this.zzc[i], (List) unsafe.getObject(t, j), zzmrVar, true);
                            break;
                        case 48:
                            zzle.zze(this.zzc[i], (List) unsafe.getObject(t, j), zzmrVar, true);
                            break;
                        case 49:
                            zzle.zzb(this.zzc[i], (List) unsafe.getObject(t, j), zzmrVar, zza(i));
                            break;
                        case 50:
                            zza(zzmrVar, i5, unsafe.getObject(t, j), i);
                            break;
                        case 51:
                            if (zza((zzko<T>) t, i5, i)) {
                                zzmrVar.zza(i5, zzb(t, j));
                            }
                            break;
                        case 52:
                            if (zza((zzko<T>) t, i5, i)) {
                                zzmrVar.zza(i5, zzc(t, j));
                            }
                            break;
                        case 53:
                            if (zza((zzko<T>) t, i5, i)) {
                                zzmrVar.zza(i5, zze(t, j));
                            }
                            break;
                        case 54:
                            if (zza((zzko<T>) t, i5, i)) {
                                zzmrVar.zzc(i5, zze(t, j));
                            }
                            break;
                        case 55:
                            if (zza((zzko<T>) t, i5, i)) {
                                zzmrVar.zzc(i5, zzd(t, j));
                            }
                            break;
                        case 56:
                            if (zza((zzko<T>) t, i5, i)) {
                                zzmrVar.zzd(i5, zze(t, j));
                            }
                            break;
                        case 57:
                            if (zza((zzko<T>) t, i5, i)) {
                                zzmrVar.zzd(i5, zzd(t, j));
                            }
                            break;
                        case 58:
                            if (zza((zzko<T>) t, i5, i)) {
                                zzmrVar.zza(i5, zzf(t, j));
                            }
                            break;
                        case 59:
                            if (zza((zzko<T>) t, i5, i)) {
                                zza(i5, unsafe.getObject(t, j), zzmrVar);
                            }
                            break;
                        case 60:
                            if (zza((zzko<T>) t, i5, i)) {
                                zzmrVar.zza(i5, unsafe.getObject(t, j), zza(i));
                            }
                            break;
                        case 61:
                            if (zza((zzko<T>) t, i5, i)) {
                                zzmrVar.zza(i5, (zzht) unsafe.getObject(t, j));
                            }
                            break;
                        case 62:
                            if (zza((zzko<T>) t, i5, i)) {
                                zzmrVar.zze(i5, zzd(t, j));
                            }
                            break;
                        case 63:
                            if (zza((zzko<T>) t, i5, i)) {
                                zzmrVar.zzb(i5, zzd(t, j));
                            }
                            break;
                        case 64:
                            if (zza((zzko<T>) t, i5, i)) {
                                zzmrVar.zza(i5, zzd(t, j));
                            }
                            break;
                        case 65:
                            if (zza((zzko<T>) t, i5, i)) {
                                zzmrVar.zzb(i5, zze(t, j));
                            }
                            break;
                        case 66:
                            if (zza((zzko<T>) t, i5, i)) {
                                zzmrVar.zzf(i5, zzd(t, j));
                            }
                            break;
                        case 67:
                            if (zza((zzko<T>) t, i5, i)) {
                                zzmrVar.zze(i5, zze(t, j));
                            }
                            break;
                        case 68:
                            if (zza((zzko<T>) t, i5, i)) {
                                zzmrVar.zzb(i5, unsafe.getObject(t, j), zza(i));
                            }
                            break;
                    }
                }
                while (entry != null) {
                    this.zzr.zza(zzmrVar, entry);
                    entry = it.hasNext() ? it.next() : null;
                }
                zza(this.zzq, t, zzmrVar);
            }
        }
        it = null;
        entry = null;
        length = this.zzc.length;
        Unsafe unsafe2 = zzb;
        int i32 = 1048575;
        int i42 = 0;
        while (i < length) {
        }
        while (entry != null) {
        }
        zza(this.zzq, t, zzmrVar);
    }

    private final Object zzb(int i) {
        return this.zzd[(i / 3) << 1];
    }

    private static <T> double zzb(T t, long j) {
        return ((Double) zzma.zzf(t, j)).doubleValue();
    }

    private final void zzb(T t, int i) {
        int zze = zze(i);
        long j = 1048575 & zze;
        if (j == 1048575) {
            return;
        }
        zzma.zza((Object) t, j, (1 << (zze >>> 20)) | zzma.zza(t, j));
    }

    private final void zzb(T t, int i, int i2) {
        zzma.zza((Object) t, zze(i2) & 1048575, i);
    }

    private final int zzb(int i, int i2) {
        int length = (this.zzc.length / 3) - 1;
        while (i2 <= length) {
            int i3 = (length + i2) >>> 1;
            int i4 = i3 * 3;
            int i5 = this.zzc[i4];
            if (i == i5) {
                return i4;
            }
            if (i < i5) {
                length = i3 - 1;
            } else {
                i2 = i3 + 1;
            }
        }
        return -1;
    }

    private final <K, V> void zza(zzmr zzmrVar, int i, Object obj, int i2) {
        if (obj != null) {
            zzmrVar.zza(i, this.zzs.zzb(zzb(i2)), this.zzs.zzc(obj));
        }
    }

    private static <UT, UB> void zza(zzlu<UT, UB> zzluVar, T t, zzmr zzmrVar) {
        zzluVar.zza((zzlu<UT, UB>) zzluVar.zzb(t), zzmrVar);
    }

    /* JADX DEBUG: Another duplicated slice has different insns count: {[IGET]}, finally: {[IGET, IGET, AGET, INVOKE, ARITH, INVOKE, IF, IF, IGET] complete} */
    @Override // com.google.android.gms.internal.vision.zzlc
    public final void zza(T t, zzld zzldVar, zzio zzioVar) {
        Objects.requireNonNull(zzioVar);
        zzlu zzluVar = this.zzq;
        zziq<?> zziqVar = this.zzr;
        zziu<?> zziuVar = null;
        Object obj = null;
        while (true) {
            try {
                int zza2 = zzldVar.zza();
                int zzg = zzg(zza2);
                if (zzg >= 0) {
                    int zzd = zzd(zzg);
                    switch ((267386880 & zzd) >>> 20) {
                        case 0:
                            zzma.zza(t, zzd & 1048575, zzldVar.zzd());
                            zzb((zzko<T>) t, zzg);
                            continue;
                        case 1:
                            zzma.zza((Object) t, zzd & 1048575, zzldVar.zze());
                            zzb((zzko<T>) t, zzg);
                            continue;
                        case 2:
                            zzma.zza((Object) t, zzd & 1048575, zzldVar.zzg());
                            zzb((zzko<T>) t, zzg);
                            continue;
                        case 3:
                            zzma.zza((Object) t, zzd & 1048575, zzldVar.zzf());
                            zzb((zzko<T>) t, zzg);
                            continue;
                        case 4:
                            zzma.zza((Object) t, zzd & 1048575, zzldVar.zzh());
                            zzb((zzko<T>) t, zzg);
                            continue;
                        case 5:
                            zzma.zza((Object) t, zzd & 1048575, zzldVar.zzi());
                            zzb((zzko<T>) t, zzg);
                            continue;
                        case 6:
                            zzma.zza((Object) t, zzd & 1048575, zzldVar.zzj());
                            zzb((zzko<T>) t, zzg);
                            continue;
                        case 7:
                            zzma.zza(t, zzd & 1048575, zzldVar.zzk());
                            zzb((zzko<T>) t, zzg);
                            continue;
                        case 8:
                            zza(t, zzd, zzldVar);
                            zzb((zzko<T>) t, zzg);
                            continue;
                        case 9:
                            if (zza((zzko<T>) t, zzg)) {
                                long j = zzd & 1048575;
                                zzma.zza(t, j, zzjf.zza(zzma.zzf(t, j), zzldVar.zza(zza(zzg), zzioVar)));
                                break;
                            } else {
                                zzma.zza(t, zzd & 1048575, zzldVar.zza(zza(zzg), zzioVar));
                                zzb((zzko<T>) t, zzg);
                                continue;
                            }
                        case 10:
                            zzma.zza(t, zzd & 1048575, zzldVar.zzn());
                            zzb((zzko<T>) t, zzg);
                            continue;
                        case 11:
                            zzma.zza((Object) t, zzd & 1048575, zzldVar.zzo());
                            zzb((zzko<T>) t, zzg);
                            continue;
                        case 12:
                            int zzp = zzldVar.zzp();
                            zzjg zzc = zzc(zzg);
                            if (zzc != null && !zzc.zza(zzp)) {
                                obj = zzle.zza(zza2, zzp, obj, zzluVar);
                                break;
                            }
                            zzma.zza((Object) t, zzd & 1048575, zzp);
                            zzb((zzko<T>) t, zzg);
                            continue;
                        case 13:
                            zzma.zza((Object) t, zzd & 1048575, zzldVar.zzq());
                            zzb((zzko<T>) t, zzg);
                            continue;
                        case 14:
                            zzma.zza((Object) t, zzd & 1048575, zzldVar.zzr());
                            zzb((zzko<T>) t, zzg);
                            continue;
                        case 15:
                            zzma.zza((Object) t, zzd & 1048575, zzldVar.zzs());
                            zzb((zzko<T>) t, zzg);
                            continue;
                        case 16:
                            zzma.zza((Object) t, zzd & 1048575, zzldVar.zzt());
                            zzb((zzko<T>) t, zzg);
                            continue;
                        case 17:
                            if (zza((zzko<T>) t, zzg)) {
                                long j2 = zzd & 1048575;
                                zzma.zza(t, j2, zzjf.zza(zzma.zzf(t, j2), zzldVar.zzb(zza(zzg), zzioVar)));
                                break;
                            } else {
                                zzma.zza(t, zzd & 1048575, zzldVar.zzb(zza(zzg), zzioVar));
                                zzb((zzko<T>) t, zzg);
                                continue;
                            }
                        case 18:
                            zzldVar.zza(this.zzp.zza(t, zzd & 1048575));
                            continue;
                        case 19:
                            zzldVar.zzb(this.zzp.zza(t, zzd & 1048575));
                            continue;
                        case 20:
                            zzldVar.zzd(this.zzp.zza(t, zzd & 1048575));
                            continue;
                        case 21:
                            zzldVar.zzc(this.zzp.zza(t, zzd & 1048575));
                            continue;
                        case 22:
                            zzldVar.zze(this.zzp.zza(t, zzd & 1048575));
                            continue;
                        case 23:
                            zzldVar.zzf(this.zzp.zza(t, zzd & 1048575));
                            continue;
                        case 24:
                            zzldVar.zzg(this.zzp.zza(t, zzd & 1048575));
                            continue;
                        case 25:
                            zzldVar.zzh(this.zzp.zza(t, zzd & 1048575));
                            continue;
                        case 26:
                            if (zzf(zzd)) {
                                zzldVar.zzj(this.zzp.zza(t, zzd & 1048575));
                                break;
                            } else {
                                zzldVar.zzi(this.zzp.zza(t, zzd & 1048575));
                                continue;
                            }
                        case 27:
                            zzldVar.zza(this.zzp.zza(t, zzd & 1048575), zza(zzg), zzioVar);
                            continue;
                        case 28:
                            zzldVar.zzk(this.zzp.zza(t, zzd & 1048575));
                            continue;
                        case 29:
                            zzldVar.zzl(this.zzp.zza(t, zzd & 1048575));
                            continue;
                        case 30:
                            List<Integer> zza3 = this.zzp.zza(t, zzd & 1048575);
                            zzldVar.zzm(zza3);
                            obj = zzle.zza(zza2, zza3, zzc(zzg), obj, zzluVar);
                            continue;
                        case 31:
                            zzldVar.zzn(this.zzp.zza(t, zzd & 1048575));
                            continue;
                        case 32:
                            zzldVar.zzo(this.zzp.zza(t, zzd & 1048575));
                            continue;
                        case 33:
                            zzldVar.zzp(this.zzp.zza(t, zzd & 1048575));
                            continue;
                        case 34:
                            zzldVar.zzq(this.zzp.zza(t, zzd & 1048575));
                            continue;
                        case 35:
                            zzldVar.zza(this.zzp.zza(t, zzd & 1048575));
                            continue;
                        case 36:
                            zzldVar.zzb(this.zzp.zza(t, zzd & 1048575));
                            continue;
                        case 37:
                            zzldVar.zzd(this.zzp.zza(t, zzd & 1048575));
                            continue;
                        case 38:
                            zzldVar.zzc(this.zzp.zza(t, zzd & 1048575));
                            continue;
                        case 39:
                            zzldVar.zze(this.zzp.zza(t, zzd & 1048575));
                            continue;
                        case 40:
                            zzldVar.zzf(this.zzp.zza(t, zzd & 1048575));
                            continue;
                        case 41:
                            zzldVar.zzg(this.zzp.zza(t, zzd & 1048575));
                            continue;
                        case 42:
                            zzldVar.zzh(this.zzp.zza(t, zzd & 1048575));
                            continue;
                        case 43:
                            zzldVar.zzl(this.zzp.zza(t, zzd & 1048575));
                            continue;
                        case 44:
                            List<Integer> zza4 = this.zzp.zza(t, zzd & 1048575);
                            zzldVar.zzm(zza4);
                            obj = zzle.zza(zza2, zza4, zzc(zzg), obj, zzluVar);
                            continue;
                        case 45:
                            zzldVar.zzn(this.zzp.zza(t, zzd & 1048575));
                            continue;
                        case 46:
                            zzldVar.zzo(this.zzp.zza(t, zzd & 1048575));
                            continue;
                        case 47:
                            zzldVar.zzp(this.zzp.zza(t, zzd & 1048575));
                            continue;
                        case 48:
                            zzldVar.zzq(this.zzp.zza(t, zzd & 1048575));
                            continue;
                        case 49:
                            zzldVar.zzb(this.zzp.zza(t, zzd & 1048575), zza(zzg), zzioVar);
                            continue;
                        case 50:
                            Object zzb2 = zzb(zzg);
                            long zzd2 = zzd(zzg) & 1048575;
                            Object zzf = zzma.zzf(t, zzd2);
                            if (zzf == null) {
                                zzf = this.zzs.zzf(zzb2);
                                zzma.zza(t, zzd2, zzf);
                            } else if (this.zzs.zzd(zzf)) {
                                Object zzf2 = this.zzs.zzf(zzb2);
                                this.zzs.zza(zzf2, zzf);
                                zzma.zza(t, zzd2, zzf2);
                                zzf = zzf2;
                            }
                            zzldVar.zza(this.zzs.zza(zzf), this.zzs.zzb(zzb2), zzioVar);
                            continue;
                        case 51:
                            zzma.zza(t, zzd & 1048575, Double.valueOf(zzldVar.zzd()));
                            zzb((zzko<T>) t, zza2, zzg);
                            continue;
                        case 52:
                            zzma.zza(t, zzd & 1048575, Float.valueOf(zzldVar.zze()));
                            zzb((zzko<T>) t, zza2, zzg);
                            continue;
                        case 53:
                            zzma.zza(t, zzd & 1048575, Long.valueOf(zzldVar.zzg()));
                            zzb((zzko<T>) t, zza2, zzg);
                            continue;
                        case 54:
                            zzma.zza(t, zzd & 1048575, Long.valueOf(zzldVar.zzf()));
                            zzb((zzko<T>) t, zza2, zzg);
                            continue;
                        case 55:
                            zzma.zza(t, zzd & 1048575, Integer.valueOf(zzldVar.zzh()));
                            zzb((zzko<T>) t, zza2, zzg);
                            continue;
                        case 56:
                            zzma.zza(t, zzd & 1048575, Long.valueOf(zzldVar.zzi()));
                            zzb((zzko<T>) t, zza2, zzg);
                            continue;
                        case 57:
                            zzma.zza(t, zzd & 1048575, Integer.valueOf(zzldVar.zzj()));
                            zzb((zzko<T>) t, zza2, zzg);
                            continue;
                        case 58:
                            zzma.zza(t, zzd & 1048575, Boolean.valueOf(zzldVar.zzk()));
                            zzb((zzko<T>) t, zza2, zzg);
                            continue;
                        case 59:
                            zza(t, zzd, zzldVar);
                            zzb((zzko<T>) t, zza2, zzg);
                            continue;
                        case 60:
                            if (zza((zzko<T>) t, zza2, zzg)) {
                                long j3 = zzd & 1048575;
                                zzma.zza(t, j3, zzjf.zza(zzma.zzf(t, j3), zzldVar.zza(zza(zzg), zzioVar)));
                            } else {
                                zzma.zza(t, zzd & 1048575, zzldVar.zza(zza(zzg), zzioVar));
                                zzb((zzko<T>) t, zzg);
                            }
                            zzb((zzko<T>) t, zza2, zzg);
                            continue;
                        case 61:
                            zzma.zza(t, zzd & 1048575, zzldVar.zzn());
                            zzb((zzko<T>) t, zza2, zzg);
                            continue;
                        case 62:
                            zzma.zza(t, zzd & 1048575, Integer.valueOf(zzldVar.zzo()));
                            zzb((zzko<T>) t, zza2, zzg);
                            continue;
                        case 63:
                            int zzp2 = zzldVar.zzp();
                            zzjg zzc2 = zzc(zzg);
                            if (zzc2 != null && !zzc2.zza(zzp2)) {
                                obj = zzle.zza(zza2, zzp2, obj, zzluVar);
                                break;
                            }
                            zzma.zza(t, zzd & 1048575, Integer.valueOf(zzp2));
                            zzb((zzko<T>) t, zza2, zzg);
                            continue;
                        case 64:
                            zzma.zza(t, zzd & 1048575, Integer.valueOf(zzldVar.zzq()));
                            zzb((zzko<T>) t, zza2, zzg);
                            continue;
                        case 65:
                            zzma.zza(t, zzd & 1048575, Long.valueOf(zzldVar.zzr()));
                            zzb((zzko<T>) t, zza2, zzg);
                            continue;
                        case 66:
                            zzma.zza(t, zzd & 1048575, Integer.valueOf(zzldVar.zzs()));
                            zzb((zzko<T>) t, zza2, zzg);
                            continue;
                        case 67:
                            zzma.zza(t, zzd & 1048575, Long.valueOf(zzldVar.zzt()));
                            zzb((zzko<T>) t, zza2, zzg);
                            continue;
                        case 68:
                            zzma.zza(t, zzd & 1048575, zzldVar.zzb(zza(zzg), zzioVar));
                            zzb((zzko<T>) t, zza2, zzg);
                            continue;
                        default:
                            if (obj == null) {
                                try {
                                    obj = zzluVar.zza();
                                } catch (zzjn unused) {
                                    zzluVar.zza(zzldVar);
                                    if (obj == null) {
                                        obj = zzluVar.zzc(t);
                                    }
                                    if (!zzluVar.zza((zzlu) obj, zzldVar)) {
                                        for (int i = this.zzm; i < this.zzn; i++) {
                                            obj = zza((Object) t, this.zzl[i], (int) obj, (zzlu<UT, int>) zzluVar);
                                        }
                                        if (obj != null) {
                                            zzluVar.zzb((Object) t, (T) obj);
                                            return;
                                        }
                                        return;
                                    }
                                    break;
                                }
                            }
                            if (!zzluVar.zza((zzlu) obj, zzldVar)) {
                                for (int i2 = this.zzm; i2 < this.zzn; i2++) {
                                    obj = zza((Object) t, this.zzl[i2], (int) obj, (zzlu<UT, int>) zzluVar);
                                }
                                if (obj != null) {
                                    zzluVar.zzb((Object) t, (T) obj);
                                    return;
                                }
                                return;
                            }
                            continue;
                    }
                } else if (zza2 == Integer.MAX_VALUE) {
                    for (int i3 = this.zzm; i3 < this.zzn; i3++) {
                        obj = zza((Object) t, this.zzl[i3], (int) obj, (zzlu<UT, int>) zzluVar);
                    }
                    if (obj != null) {
                        zzluVar.zzb((Object) t, (T) obj);
                        return;
                    }
                    return;
                } else {
                    Object zza5 = !this.zzh ? null : zziqVar.zza(zzioVar, this.zzg, zza2);
                    if (zza5 != null) {
                        if (zziuVar == null) {
                            zziuVar = zziqVar.zzb(t);
                        }
                        zziu<?> zziuVar2 = zziuVar;
                        obj = zziqVar.zza(zzldVar, zza5, zzioVar, zziuVar2, obj, zzluVar);
                        zziuVar = zziuVar2;
                    } else {
                        zzluVar.zza(zzldVar);
                        if (obj == null) {
                            obj = zzluVar.zzc(t);
                        }
                        if (!zzluVar.zza((zzlu) obj, zzldVar)) {
                            for (int i4 = this.zzm; i4 < this.zzn; i4++) {
                                obj = zza((Object) t, this.zzl[i4], (int) obj, (zzlu<UT, int>) zzluVar);
                            }
                            if (obj != null) {
                                zzluVar.zzb((Object) t, (T) obj);
                                return;
                            }
                            return;
                        }
                    }
                }
            } catch (Throwable th) {
                for (int i5 = this.zzm; i5 < this.zzn; i5++) {
                    obj = zza((Object) t, this.zzl[i5], (int) obj, (zzlu<UT, int>) zzluVar);
                }
                if (obj != null) {
                    zzluVar.zzb((Object) t, (T) obj);
                }
                throw th;
            }
        }
    }

    private static int zza(byte[] bArr, int i, int i2, zzml zzmlVar, Class<?> cls, zzhn zzhnVar) {
        switch (zzkr.zza[zzmlVar.ordinal()]) {
            case 1:
                int zzb2 = zzhl.zzb(bArr, i, zzhnVar);
                zzhnVar.zzc = Boolean.valueOf(zzhnVar.zzb != 0);
                return zzb2;
            case 2:
                return zzhl.zze(bArr, i, zzhnVar);
            case 3:
                zzhnVar.zzc = Double.valueOf(zzhl.zzc(bArr, i));
                return i + 8;
            case 4:
            case 5:
                zzhnVar.zzc = Integer.valueOf(zzhl.zza(bArr, i));
                return i + 4;
            case 6:
            case 7:
                zzhnVar.zzc = Long.valueOf(zzhl.zzb(bArr, i));
                return i + 8;
            case 8:
                zzhnVar.zzc = Float.valueOf(zzhl.zzd(bArr, i));
                return i + 4;
            case 9:
            case 10:
            case 11:
                int zza2 = zzhl.zza(bArr, i, zzhnVar);
                zzhnVar.zzc = Integer.valueOf(zzhnVar.zza);
                return zza2;
            case 12:
            case 13:
                int zzb3 = zzhl.zzb(bArr, i, zzhnVar);
                zzhnVar.zzc = Long.valueOf(zzhnVar.zzb);
                return zzb3;
            case 14:
                return zzhl.zza(zzky.zza().zza((Class) cls), bArr, i, i2, zzhnVar);
            case 15:
                int zza3 = zzhl.zza(bArr, i, zzhnVar);
                zzhnVar.zzc = Integer.valueOf(zzif.zze(zzhnVar.zza));
                return zza3;
            case 16:
                int zzb4 = zzhl.zzb(bArr, i, zzhnVar);
                zzhnVar.zzc = Long.valueOf(zzif.zza(zzhnVar.zzb));
                return zzb4;
            case 17:
                return zzhl.zzd(bArr, i, zzhnVar);
            default:
                throw new RuntimeException("unsupported field type.");
        }
    }

    /* JADX WARN: Removed duplicated region for block: B:117:0x0236  */
    /* JADX WARN: Removed duplicated region for block: B:68:0x016e  */
    /* JADX WARN: Removed duplicated region for block: B:97:0x01e8  */
    /* JADX WARN: Unsupported multi-entry loop pattern (BACK_EDGE: B:115:0x0233 -> B:116:0x0234). Please submit an issue!!! */
    /* JADX WARN: Unsupported multi-entry loop pattern (BACK_EDGE: B:66:0x016b -> B:67:0x016c). Please submit an issue!!! */
    /* JADX WARN: Unsupported multi-entry loop pattern (BACK_EDGE: B:95:0x01e5 -> B:96:0x01e6). Please submit an issue!!! */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    private final int zza(T t, byte[] bArr, int i, int i2, int i3, int i4, int i5, int i6, long j, int i7, long j2, zzhn zzhnVar) {
        int zza2;
        int i8;
        int i9 = i;
        Unsafe unsafe = zzb;
        zzjl zzjlVar = (zzjl) unsafe.getObject(t, j2);
        if (!zzjlVar.zza()) {
            int size = zzjlVar.size();
            zzjlVar = zzjlVar.zza(size == 0 ? 10 : size << 1);
            unsafe.putObject(t, j2, zzjlVar);
        }
        switch (i7) {
            case 18:
            case 35:
                if (i5 == 2) {
                    zzin zzinVar = (zzin) zzjlVar;
                    int zza3 = zzhl.zza(bArr, i9, zzhnVar);
                    int i10 = zzhnVar.zza + zza3;
                    while (zza3 < i10) {
                        zzinVar.zza(zzhl.zzc(bArr, zza3));
                        zza3 += 8;
                    }
                    if (zza3 == i10) {
                        return zza3;
                    }
                    throw zzjk.zza();
                }
                if (i5 == 1) {
                    zzin zzinVar2 = (zzin) zzjlVar;
                    zzinVar2.zza(zzhl.zzc(bArr, i));
                    while (true) {
                        int i11 = i9 + 8;
                        if (i11 >= i2) {
                            return i11;
                        }
                        i9 = zzhl.zza(bArr, i11, zzhnVar);
                        if (i3 != zzhnVar.zza) {
                            return i11;
                        }
                        zzinVar2.zza(zzhl.zzc(bArr, i9));
                    }
                }
                return i9;
            case 19:
            case 36:
                if (i5 == 2) {
                    zzja zzjaVar = (zzja) zzjlVar;
                    int zza4 = zzhl.zza(bArr, i9, zzhnVar);
                    int i12 = zzhnVar.zza + zza4;
                    while (zza4 < i12) {
                        zzjaVar.zza(zzhl.zzd(bArr, zza4));
                        zza4 += 4;
                    }
                    if (zza4 == i12) {
                        return zza4;
                    }
                    throw zzjk.zza();
                }
                if (i5 == 5) {
                    zzja zzjaVar2 = (zzja) zzjlVar;
                    zzjaVar2.zza(zzhl.zzd(bArr, i));
                    while (true) {
                        int i13 = i9 + 4;
                        if (i13 >= i2) {
                            return i13;
                        }
                        i9 = zzhl.zza(bArr, i13, zzhnVar);
                        if (i3 != zzhnVar.zza) {
                            return i13;
                        }
                        zzjaVar2.zza(zzhl.zzd(bArr, i9));
                    }
                }
                return i9;
            case 20:
            case 21:
            case 37:
            case 38:
                if (i5 == 2) {
                    zzjy zzjyVar = (zzjy) zzjlVar;
                    int zza5 = zzhl.zza(bArr, i9, zzhnVar);
                    int i14 = zzhnVar.zza + zza5;
                    while (zza5 < i14) {
                        zza5 = zzhl.zzb(bArr, zza5, zzhnVar);
                        zzjyVar.zza(zzhnVar.zzb);
                    }
                    if (zza5 == i14) {
                        return zza5;
                    }
                    throw zzjk.zza();
                }
                if (i5 == 0) {
                    zzjy zzjyVar2 = (zzjy) zzjlVar;
                    int zzb2 = zzhl.zzb(bArr, i9, zzhnVar);
                    zzjyVar2.zza(zzhnVar.zzb);
                    while (zzb2 < i2) {
                        int zza6 = zzhl.zza(bArr, zzb2, zzhnVar);
                        if (i3 != zzhnVar.zza) {
                            return zzb2;
                        }
                        zzb2 = zzhl.zzb(bArr, zza6, zzhnVar);
                        zzjyVar2.zza(zzhnVar.zzb);
                    }
                    return zzb2;
                }
                return i9;
            case 22:
            case 29:
            case 39:
            case 43:
                if (i5 == 2) {
                    return zzhl.zza(bArr, i9, zzjlVar, zzhnVar);
                }
                if (i5 == 0) {
                    return zzhl.zza(i3, bArr, i, i2, zzjlVar, zzhnVar);
                }
                return i9;
            case 23:
            case 32:
            case 40:
            case 46:
                if (i5 == 2) {
                    zzjy zzjyVar3 = (zzjy) zzjlVar;
                    int zza7 = zzhl.zza(bArr, i9, zzhnVar);
                    int i15 = zzhnVar.zza + zza7;
                    while (zza7 < i15) {
                        zzjyVar3.zza(zzhl.zzb(bArr, zza7));
                        zza7 += 8;
                    }
                    if (zza7 == i15) {
                        return zza7;
                    }
                    throw zzjk.zza();
                }
                if (i5 == 1) {
                    zzjy zzjyVar4 = (zzjy) zzjlVar;
                    zzjyVar4.zza(zzhl.zzb(bArr, i));
                    while (true) {
                        int i16 = i9 + 8;
                        if (i16 >= i2) {
                            return i16;
                        }
                        i9 = zzhl.zza(bArr, i16, zzhnVar);
                        if (i3 != zzhnVar.zza) {
                            return i16;
                        }
                        zzjyVar4.zza(zzhl.zzb(bArr, i9));
                    }
                }
                return i9;
            case 24:
            case 31:
            case 41:
            case 45:
                if (i5 == 2) {
                    zzjd zzjdVar = (zzjd) zzjlVar;
                    int zza8 = zzhl.zza(bArr, i9, zzhnVar);
                    int i17 = zzhnVar.zza + zza8;
                    while (zza8 < i17) {
                        zzjdVar.zzc(zzhl.zza(bArr, zza8));
                        zza8 += 4;
                    }
                    if (zza8 == i17) {
                        return zza8;
                    }
                    throw zzjk.zza();
                }
                if (i5 == 5) {
                    zzjd zzjdVar2 = (zzjd) zzjlVar;
                    zzjdVar2.zzc(zzhl.zza(bArr, i));
                    while (true) {
                        int i18 = i9 + 4;
                        if (i18 >= i2) {
                            return i18;
                        }
                        i9 = zzhl.zza(bArr, i18, zzhnVar);
                        if (i3 != zzhnVar.zza) {
                            return i18;
                        }
                        zzjdVar2.zzc(zzhl.zza(bArr, i9));
                    }
                }
                return i9;
            case 25:
            case 42:
                if (i5 == 2) {
                    zzhr zzhrVar = (zzhr) zzjlVar;
                    zza2 = zzhl.zza(bArr, i9, zzhnVar);
                    int i19 = zzhnVar.zza + zza2;
                    while (zza2 < i19) {
                        zza2 = zzhl.zzb(bArr, zza2, zzhnVar);
                        zzhrVar.zza(zzhnVar.zzb != 0);
                    }
                    if (zza2 != i19) {
                        throw zzjk.zza();
                    }
                    return zza2;
                }
                if (i5 == 0) {
                    zzhr zzhrVar2 = (zzhr) zzjlVar;
                    i9 = zzhl.zzb(bArr, i9, zzhnVar);
                    zzhrVar2.zza(zzhnVar.zzb != 0);
                    while (i9 < i2) {
                        int zza9 = zzhl.zza(bArr, i9, zzhnVar);
                        if (i3 == zzhnVar.zza) {
                            i9 = zzhl.zzb(bArr, zza9, zzhnVar);
                            zzhrVar2.zza(zzhnVar.zzb != 0);
                        }
                    }
                }
                return i9;
            case 26:
                if (i5 == 2) {
                    if ((j & 536870912) == 0) {
                        int zza10 = zzhl.zza(bArr, i9, zzhnVar);
                        int i20 = zzhnVar.zza;
                        if (i20 >= 0) {
                            if (i20 == 0) {
                                zzjlVar.add("");
                                while (zza10 < i2) {
                                    int zza11 = zzhl.zza(bArr, zza10, zzhnVar);
                                    if (i3 != zzhnVar.zza) {
                                        return zza10;
                                    }
                                    zza10 = zzhl.zza(bArr, zza11, zzhnVar);
                                    i20 = zzhnVar.zza;
                                    if (i20 < 0) {
                                        throw zzjk.zzb();
                                    }
                                    if (i20 == 0) {
                                        zzjlVar.add("");
                                    } else {
                                        zzjlVar.add(new String(bArr, zza10, i20, zzjf.zza));
                                        zza10 += i20;
                                        while (zza10 < i2) {
                                        }
                                    }
                                }
                                return zza10;
                            }
                            zzjlVar.add(new String(bArr, zza10, i20, zzjf.zza));
                            zza10 += i20;
                            while (zza10 < i2) {
                            }
                            return zza10;
                        }
                        throw zzjk.zzb();
                    }
                    int zza12 = zzhl.zza(bArr, i9, zzhnVar);
                    int i21 = zzhnVar.zza;
                    if (i21 >= 0) {
                        if (i21 == 0) {
                            zzjlVar.add("");
                            while (zza12 < i2) {
                                int zza13 = zzhl.zza(bArr, zza12, zzhnVar);
                                if (i3 != zzhnVar.zza) {
                                    return zza12;
                                }
                                zza12 = zzhl.zza(bArr, zza13, zzhnVar);
                                int i22 = zzhnVar.zza;
                                if (i22 < 0) {
                                    throw zzjk.zzb();
                                }
                                if (i22 == 0) {
                                    zzjlVar.add("");
                                } else {
                                    i8 = zza12 + i22;
                                    if (zzmd.zza(bArr, zza12, i8)) {
                                        zzjlVar.add(new String(bArr, zza12, i22, zzjf.zza));
                                        zza12 = i8;
                                        while (zza12 < i2) {
                                        }
                                    } else {
                                        throw zzjk.zzh();
                                    }
                                }
                            }
                            return zza12;
                        }
                        i8 = zza12 + i21;
                        if (zzmd.zza(bArr, zza12, i8)) {
                            zzjlVar.add(new String(bArr, zza12, i21, zzjf.zza));
                            zza12 = i8;
                            while (zza12 < i2) {
                            }
                            return zza12;
                        }
                        throw zzjk.zzh();
                    }
                    throw zzjk.zzb();
                }
                return i9;
            case 27:
                if (i5 == 2) {
                    return zzhl.zza(zza(i6), i3, bArr, i, i2, zzjlVar, zzhnVar);
                }
                return i9;
            case 28:
                if (i5 == 2) {
                    int zza14 = zzhl.zza(bArr, i9, zzhnVar);
                    int i23 = zzhnVar.zza;
                    if (i23 >= 0) {
                        if (i23 <= bArr.length - zza14) {
                            if (i23 == 0) {
                                zzjlVar.add(zzht.zza);
                                while (zza14 < i2) {
                                    int zza15 = zzhl.zza(bArr, zza14, zzhnVar);
                                    if (i3 != zzhnVar.zza) {
                                        return zza14;
                                    }
                                    zza14 = zzhl.zza(bArr, zza15, zzhnVar);
                                    i23 = zzhnVar.zza;
                                    if (i23 >= 0) {
                                        if (i23 > bArr.length - zza14) {
                                            throw zzjk.zza();
                                        }
                                        if (i23 == 0) {
                                            zzjlVar.add(zzht.zza);
                                        } else {
                                            zzjlVar.add(zzht.zza(bArr, zza14, i23));
                                            zza14 += i23;
                                            while (zza14 < i2) {
                                            }
                                        }
                                    } else {
                                        throw zzjk.zzb();
                                    }
                                }
                                return zza14;
                            }
                            zzjlVar.add(zzht.zza(bArr, zza14, i23));
                            zza14 += i23;
                            while (zza14 < i2) {
                            }
                            return zza14;
                        }
                        throw zzjk.zza();
                    }
                    throw zzjk.zzb();
                }
                return i9;
            case 30:
            case 44:
                if (i5 != 2) {
                    if (i5 == 0) {
                        zza2 = zzhl.zza(i3, bArr, i, i2, zzjlVar, zzhnVar);
                    }
                    return i9;
                }
                zza2 = zzhl.zza(bArr, i9, zzjlVar, zzhnVar);
                zzjb zzjbVar = (zzjb) t;
                zzlx zzlxVar = zzjbVar.zzb;
                if (zzlxVar == zzlx.zza()) {
                    zzlxVar = null;
                }
                zzlx zzlxVar2 = (zzlx) zzle.zza(i4, zzjlVar, zzc(i6), zzlxVar, this.zzq);
                if (zzlxVar2 != null) {
                    zzjbVar.zzb = zzlxVar2;
                }
                return zza2;
            case 33:
            case 47:
                if (i5 == 2) {
                    zzjd zzjdVar3 = (zzjd) zzjlVar;
                    int zza16 = zzhl.zza(bArr, i9, zzhnVar);
                    int i24 = zzhnVar.zza + zza16;
                    while (zza16 < i24) {
                        zza16 = zzhl.zza(bArr, zza16, zzhnVar);
                        zzjdVar3.zzc(zzif.zze(zzhnVar.zza));
                    }
                    if (zza16 == i24) {
                        return zza16;
                    }
                    throw zzjk.zza();
                }
                if (i5 == 0) {
                    zzjd zzjdVar4 = (zzjd) zzjlVar;
                    int zza17 = zzhl.zza(bArr, i9, zzhnVar);
                    zzjdVar4.zzc(zzif.zze(zzhnVar.zza));
                    while (zza17 < i2) {
                        int zza18 = zzhl.zza(bArr, zza17, zzhnVar);
                        if (i3 != zzhnVar.zza) {
                            return zza17;
                        }
                        zza17 = zzhl.zza(bArr, zza18, zzhnVar);
                        zzjdVar4.zzc(zzif.zze(zzhnVar.zza));
                    }
                    return zza17;
                }
                return i9;
            case 34:
            case 48:
                if (i5 == 2) {
                    zzjy zzjyVar5 = (zzjy) zzjlVar;
                    int zza19 = zzhl.zza(bArr, i9, zzhnVar);
                    int i25 = zzhnVar.zza + zza19;
                    while (zza19 < i25) {
                        zza19 = zzhl.zzb(bArr, zza19, zzhnVar);
                        zzjyVar5.zza(zzif.zza(zzhnVar.zzb));
                    }
                    if (zza19 == i25) {
                        return zza19;
                    }
                    throw zzjk.zza();
                }
                if (i5 == 0) {
                    zzjy zzjyVar6 = (zzjy) zzjlVar;
                    int zzb3 = zzhl.zzb(bArr, i9, zzhnVar);
                    zzjyVar6.zza(zzif.zza(zzhnVar.zzb));
                    while (zzb3 < i2) {
                        int zza20 = zzhl.zza(bArr, zzb3, zzhnVar);
                        if (i3 != zzhnVar.zza) {
                            return zzb3;
                        }
                        zzb3 = zzhl.zzb(bArr, zza20, zzhnVar);
                        zzjyVar6.zza(zzif.zza(zzhnVar.zzb));
                    }
                    return zzb3;
                }
                return i9;
            case 49:
                if (i5 == 3) {
                    zzlc zza21 = zza(i6);
                    int i26 = (i3 & (-8)) | 4;
                    i9 = zzhl.zza(zza21, bArr, i, i2, i26, zzhnVar);
                    zzjlVar.add(zzhnVar.zzc);
                    while (i9 < i2) {
                        int zza22 = zzhl.zza(bArr, i9, zzhnVar);
                        if (i3 == zzhnVar.zza) {
                            i9 = zzhl.zza(zza21, bArr, zza22, i2, i26, zzhnVar);
                            zzjlVar.add(zzhnVar.zzc);
                        }
                    }
                }
                return i9;
            default:
                return i9;
        }
    }

    /* JADX DEBUG: Failed to insert an additional move for type inference into block B:41:0x003e */
    /* JADX DEBUG: Failed to insert an additional move for type inference into block B:44:0x003e */
    /* JADX DEBUG: Type inference failed for r0v1. Raw type applied. Possible types: V, ? */
    /* JADX DEBUG: Type inference failed for r14v1. Raw type applied. Possible types: K, ? */
    /* JADX WARN: Multi-variable type inference failed */
    private final <K, V> int zza(T t, byte[] bArr, int i, int i2, int i3, long j, zzhn zzhnVar) {
        Unsafe unsafe = zzb;
        Object zzb2 = zzb(i3);
        Object object = unsafe.getObject(t, j);
        if (this.zzs.zzd(object)) {
            Object zzf = this.zzs.zzf(zzb2);
            this.zzs.zza(zzf, object);
            unsafe.putObject(t, j, zzf);
            object = zzf;
        }
        zzkf<?, ?> zzb3 = this.zzs.zzb(zzb2);
        Map<?, ?> zza2 = this.zzs.zza(object);
        int zza3 = zzhl.zza(bArr, i, zzhnVar);
        int i4 = zzhnVar.zza;
        if (i4 >= 0 && i4 <= i2 - zza3) {
            int i5 = i4 + zza3;
            Object obj = (K) zzb3.zzb;
            Object obj2 = (V) zzb3.zzd;
            while (zza3 < i5) {
                int i6 = zza3 + 1;
                int i7 = bArr[zza3];
                if (i7 < 0) {
                    i6 = zzhl.zza(i7, bArr, i6, zzhnVar);
                    i7 = zzhnVar.zza;
                }
                int i8 = i6;
                int i9 = i7 >>> 3;
                int i10 = i7 & 7;
                if (i9 != 1) {
                    if (i9 == 2 && i10 == zzb3.zzc.zzb()) {
                        zza3 = zza(bArr, i8, i2, zzb3.zzc, zzb3.zzd.getClass(), zzhnVar);
                        obj2 = zzhnVar.zzc;
                    }
                    zza3 = zzhl.zza(i7, bArr, i8, i2, zzhnVar);
                } else if (i10 == zzb3.zza.zzb()) {
                    zza3 = zza(bArr, i8, i2, zzb3.zza, (Class<?>) null, zzhnVar);
                    obj = (K) zzhnVar.zzc;
                } else {
                    zza3 = zzhl.zza(i7, bArr, i8, i2, zzhnVar);
                }
            }
            if (zza3 == i5) {
                zza2.put(obj, obj2);
                return i5;
            }
            throw zzjk.zzg();
        }
        throw zzjk.zza();
    }

    private final int zza(T t, byte[] bArr, int i, int i2, int i3, int i4, int i5, int i6, int i7, long j, int i8, zzhn zzhnVar) {
        int zzb2;
        Unsafe unsafe = zzb;
        long j2 = this.zzc[i8 + 2] & 1048575;
        switch (i7) {
            case 51:
                if (i5 == 1) {
                    unsafe.putObject(t, j, Double.valueOf(zzhl.zzc(bArr, i)));
                    zzb2 = i + 8;
                    unsafe.putInt(t, j2, i4);
                    return zzb2;
                }
                return i;
            case 52:
                if (i5 == 5) {
                    unsafe.putObject(t, j, Float.valueOf(zzhl.zzd(bArr, i)));
                    zzb2 = i + 4;
                    unsafe.putInt(t, j2, i4);
                    return zzb2;
                }
                return i;
            case 53:
            case 54:
                if (i5 == 0) {
                    zzb2 = zzhl.zzb(bArr, i, zzhnVar);
                    unsafe.putObject(t, j, Long.valueOf(zzhnVar.zzb));
                    unsafe.putInt(t, j2, i4);
                    return zzb2;
                }
                return i;
            case 55:
            case 62:
                if (i5 == 0) {
                    zzb2 = zzhl.zza(bArr, i, zzhnVar);
                    unsafe.putObject(t, j, Integer.valueOf(zzhnVar.zza));
                    unsafe.putInt(t, j2, i4);
                    return zzb2;
                }
                return i;
            case 56:
            case 65:
                if (i5 == 1) {
                    unsafe.putObject(t, j, Long.valueOf(zzhl.zzb(bArr, i)));
                    zzb2 = i + 8;
                    unsafe.putInt(t, j2, i4);
                    return zzb2;
                }
                return i;
            case 57:
            case 64:
                if (i5 == 5) {
                    unsafe.putObject(t, j, Integer.valueOf(zzhl.zza(bArr, i)));
                    zzb2 = i + 4;
                    unsafe.putInt(t, j2, i4);
                    return zzb2;
                }
                return i;
            case 58:
                if (i5 == 0) {
                    zzb2 = zzhl.zzb(bArr, i, zzhnVar);
                    unsafe.putObject(t, j, Boolean.valueOf(zzhnVar.zzb != 0));
                    unsafe.putInt(t, j2, i4);
                    return zzb2;
                }
                return i;
            case 59:
                if (i5 == 2) {
                    int zza2 = zzhl.zza(bArr, i, zzhnVar);
                    int i9 = zzhnVar.zza;
                    if (i9 == 0) {
                        unsafe.putObject(t, j, "");
                    } else if ((i6 & 536870912) != 0 && !zzmd.zza(bArr, zza2, zza2 + i9)) {
                        throw zzjk.zzh();
                    } else {
                        unsafe.putObject(t, j, new String(bArr, zza2, i9, zzjf.zza));
                        zza2 += i9;
                    }
                    unsafe.putInt(t, j2, i4);
                    return zza2;
                }
                return i;
            case 60:
                if (i5 == 2) {
                    int zza3 = zzhl.zza(zza(i8), bArr, i, i2, zzhnVar);
                    Object object = unsafe.getInt(t, j2) == i4 ? unsafe.getObject(t, j) : null;
                    if (object == null) {
                        unsafe.putObject(t, j, zzhnVar.zzc);
                    } else {
                        unsafe.putObject(t, j, zzjf.zza(object, zzhnVar.zzc));
                    }
                    unsafe.putInt(t, j2, i4);
                    return zza3;
                }
                return i;
            case 61:
                if (i5 == 2) {
                    zzb2 = zzhl.zze(bArr, i, zzhnVar);
                    unsafe.putObject(t, j, zzhnVar.zzc);
                    unsafe.putInt(t, j2, i4);
                    return zzb2;
                }
                return i;
            case 63:
                if (i5 == 0) {
                    int zza4 = zzhl.zza(bArr, i, zzhnVar);
                    int i10 = zzhnVar.zza;
                    zzjg zzc = zzc(i8);
                    if (zzc != null && !zzc.zza(i10)) {
                        zze(t).zza(i3, Long.valueOf(i10));
                        return zza4;
                    }
                    unsafe.putObject(t, j, Integer.valueOf(i10));
                    zzb2 = zza4;
                    unsafe.putInt(t, j2, i4);
                    return zzb2;
                }
                return i;
            case 66:
                if (i5 == 0) {
                    zzb2 = zzhl.zza(bArr, i, zzhnVar);
                    unsafe.putObject(t, j, Integer.valueOf(zzif.zze(zzhnVar.zza)));
                    unsafe.putInt(t, j2, i4);
                    return zzb2;
                }
                return i;
            case 67:
                if (i5 == 0) {
                    zzb2 = zzhl.zzb(bArr, i, zzhnVar);
                    unsafe.putObject(t, j, Long.valueOf(zzif.zza(zzhnVar.zzb)));
                    unsafe.putInt(t, j2, i4);
                    return zzb2;
                }
                return i;
            case 68:
                if (i5 == 3) {
                    zzb2 = zzhl.zza(zza(i8), bArr, i, i2, (i3 & (-8)) | 4, zzhnVar);
                    Object object2 = unsafe.getInt(t, j2) == i4 ? unsafe.getObject(t, j) : null;
                    if (object2 == null) {
                        unsafe.putObject(t, j, zzhnVar.zzc);
                    } else {
                        unsafe.putObject(t, j, zzjf.zza(object2, zzhnVar.zzc));
                    }
                    unsafe.putInt(t, j2, i4);
                    return zzb2;
                }
                return i;
            default:
                return i;
        }
    }

    private final zzlc zza(int i) {
        int i2 = (i / 3) << 1;
        zzlc zzlcVar = (zzlc) this.zzd[i2];
        if (zzlcVar != null) {
            return zzlcVar;
        }
        zzlc<T> zza2 = zzky.zza().zza((Class) ((Class) this.zzd[i2 + 1]));
        this.zzd[i2] = zza2;
        return zza2;
    }

    /* JADX DEBUG: Type inference failed for r5v3. Raw type applied. Possible types: com.google.android.gms.internal.vision.zzlu<?, ?>, com.google.android.gms.internal.vision.zzlu<UT, UB> */
    public final int zza(T t, byte[] bArr, int i, int i2, int i3, zzhn zzhnVar) {
        Unsafe unsafe;
        int i4;
        int i5;
        T t2;
        zzko<T> zzkoVar;
        int i6;
        int i7;
        byte b2;
        int zzg;
        int i8;
        int i9;
        int i10;
        int i11;
        int i12;
        int i13;
        boolean z;
        T t3;
        byte[] bArr2;
        zzhn zzhnVar2;
        int i14;
        int i15;
        Object obj;
        Object zza2;
        long j;
        int i16;
        int i17;
        int i18;
        int i19;
        boolean z2;
        int zzb2;
        int zze;
        T t4;
        int i20;
        int i21;
        zzko<T> zzkoVar2 = this;
        T t5 = t;
        byte[] bArr3 = bArr;
        int i22 = i2;
        int i23 = i3;
        zzhn zzhnVar3 = zzhnVar;
        Unsafe unsafe2 = zzb;
        int i24 = i;
        int i25 = 0;
        int i26 = 0;
        int i27 = 0;
        int i28 = -1;
        int i29 = 1048575;
        while (true) {
            Object obj2 = null;
            if (i24 < i22) {
                int i30 = i24 + 1;
                byte b3 = bArr3[i24];
                if (b3 < 0) {
                    int zza3 = zzhl.zza(b3, bArr3, i30, zzhnVar3);
                    b2 = zzhnVar3.zza;
                    i30 = zza3;
                } else {
                    b2 = b3;
                }
                int i31 = b2 >>> 3;
                int i32 = b2 & 7;
                if (i31 > i28) {
                    zzg = zzkoVar2.zza(i31, i25 / 3);
                } else {
                    zzg = zzkoVar2.zzg(i31);
                }
                int i33 = zzg;
                if (i33 == -1) {
                    i8 = i31;
                    i9 = i30;
                    i10 = b2;
                    i11 = i27;
                    unsafe = unsafe2;
                    i12 = i23;
                    i13 = 0;
                    z = true;
                } else {
                    int[] iArr = zzkoVar2.zzc;
                    int i34 = iArr[i33 + 1];
                    int i35 = (i34 & 267386880) >>> 20;
                    int i36 = b2;
                    long j2 = i34 & 1048575;
                    if (i35 <= 17) {
                        int i37 = iArr[i33 + 2];
                        int i38 = 1 << (i37 >>> 20);
                        int i39 = i37 & 1048575;
                        if (i39 != i29) {
                            if (i29 != 1048575) {
                                long j3 = i29;
                                t4 = t;
                                j = j2;
                                unsafe2.putInt(t4, j3, i27);
                            } else {
                                t4 = t;
                                j = j2;
                            }
                            i27 = unsafe2.getInt(t4, i39);
                            t5 = t4;
                        } else {
                            t5 = t;
                            j = j2;
                            i39 = i29;
                        }
                        int i40 = i27;
                        switch (i35) {
                            case 0:
                                i16 = i31;
                                i17 = i33;
                                i11 = i40;
                                i18 = i39;
                                i19 = i36;
                                long j4 = j;
                                z2 = true;
                                if (i32 == 1) {
                                    zzma.zza(t5, j4, zzhl.zzc(bArr3, i30));
                                    i24 = i30 + 8;
                                    i27 = i11 | i38;
                                    i29 = i18;
                                    i26 = i19;
                                    i25 = i17;
                                    i28 = i16;
                                    i22 = i2;
                                    i23 = i3;
                                    break;
                                } else {
                                    i29 = i18;
                                    i12 = i3;
                                    z = z2;
                                    i9 = i30;
                                    i10 = i19;
                                    unsafe = unsafe2;
                                    i13 = i17;
                                    i8 = i16;
                                    break;
                                }
                            case 1:
                                i16 = i31;
                                i17 = i33;
                                i11 = i40;
                                i18 = i39;
                                i19 = i36;
                                long j5 = j;
                                if (i32 == 5) {
                                    zzma.zza((Object) t5, j5, zzhl.zzd(bArr3, i30));
                                    i24 = i30 + 4;
                                    i27 = i11 | i38;
                                    i29 = i18;
                                    i26 = i19;
                                    i25 = i17;
                                    i28 = i16;
                                    i22 = i2;
                                    i23 = i3;
                                    break;
                                } else {
                                    z2 = true;
                                    i29 = i18;
                                    i12 = i3;
                                    z = z2;
                                    i9 = i30;
                                    i10 = i19;
                                    unsafe = unsafe2;
                                    i13 = i17;
                                    i8 = i16;
                                    break;
                                }
                            case 2:
                            case 3:
                                i16 = i31;
                                i17 = i33;
                                i11 = i40;
                                i18 = i39;
                                i19 = i36;
                                long j6 = j;
                                if (i32 == 0) {
                                    zzb2 = zzhl.zzb(bArr3, i30, zzhnVar3);
                                    unsafe2.putLong(t, j6, zzhnVar3.zzb);
                                    i27 = i11 | i38;
                                    i29 = i18;
                                    i24 = zzb2;
                                    i26 = i19;
                                    i25 = i17;
                                    i28 = i16;
                                    i22 = i2;
                                    i23 = i3;
                                    break;
                                } else {
                                    z2 = true;
                                    i29 = i18;
                                    i12 = i3;
                                    z = z2;
                                    i9 = i30;
                                    i10 = i19;
                                    unsafe = unsafe2;
                                    i13 = i17;
                                    i8 = i16;
                                    break;
                                }
                            case 4:
                            case 11:
                                i16 = i31;
                                i17 = i33;
                                i11 = i40;
                                i18 = i39;
                                i19 = i36;
                                long j7 = j;
                                if (i32 == 0) {
                                    i24 = zzhl.zza(bArr3, i30, zzhnVar3);
                                    unsafe2.putInt(t5, j7, zzhnVar3.zza);
                                    i27 = i11 | i38;
                                    i29 = i18;
                                    i26 = i19;
                                    i25 = i17;
                                    i28 = i16;
                                    i22 = i2;
                                    i23 = i3;
                                    break;
                                } else {
                                    z2 = true;
                                    i29 = i18;
                                    i12 = i3;
                                    z = z2;
                                    i9 = i30;
                                    i10 = i19;
                                    unsafe = unsafe2;
                                    i13 = i17;
                                    i8 = i16;
                                    break;
                                }
                            case 5:
                            case 14:
                                i16 = i31;
                                i17 = i33;
                                i11 = i40;
                                i18 = i39;
                                i19 = i36;
                                long j8 = j;
                                if (i32 == 1) {
                                    unsafe2.putLong(t, j8, zzhl.zzb(bArr3, i30));
                                    i24 = i30 + 8;
                                    i27 = i11 | i38;
                                    i29 = i18;
                                    i26 = i19;
                                    i25 = i17;
                                    i28 = i16;
                                    i22 = i2;
                                    i23 = i3;
                                    break;
                                } else {
                                    z2 = true;
                                    i29 = i18;
                                    i12 = i3;
                                    z = z2;
                                    i9 = i30;
                                    i10 = i19;
                                    unsafe = unsafe2;
                                    i13 = i17;
                                    i8 = i16;
                                    break;
                                }
                            case 6:
                            case 13:
                                i16 = i31;
                                i17 = i33;
                                i11 = i40;
                                i18 = i39;
                                i19 = i36;
                                long j9 = j;
                                if (i32 == 5) {
                                    unsafe2.putInt(t5, j9, zzhl.zza(bArr3, i30));
                                    i24 = i30 + 4;
                                    i27 = i11 | i38;
                                    i29 = i18;
                                    i26 = i19;
                                    i25 = i17;
                                    i28 = i16;
                                    i22 = i2;
                                    i23 = i3;
                                    break;
                                } else {
                                    z2 = true;
                                    i29 = i18;
                                    i12 = i3;
                                    z = z2;
                                    i9 = i30;
                                    i10 = i19;
                                    unsafe = unsafe2;
                                    i13 = i17;
                                    i8 = i16;
                                    break;
                                }
                            case 7:
                                i16 = i31;
                                i17 = i33;
                                i11 = i40;
                                i18 = i39;
                                i19 = i36;
                                long j10 = j;
                                if (i32 == 0) {
                                    i24 = zzhl.zzb(bArr3, i30, zzhnVar3);
                                    zzma.zza(t5, j10, zzhnVar3.zzb != 0);
                                    i27 = i11 | i38;
                                    i29 = i18;
                                    i26 = i19;
                                    i25 = i17;
                                    i28 = i16;
                                    i22 = i2;
                                    i23 = i3;
                                    break;
                                } else {
                                    z2 = true;
                                    i29 = i18;
                                    i12 = i3;
                                    z = z2;
                                    i9 = i30;
                                    i10 = i19;
                                    unsafe = unsafe2;
                                    i13 = i17;
                                    i8 = i16;
                                    break;
                                }
                            case 8:
                                i16 = i31;
                                i17 = i33;
                                i11 = i40;
                                i18 = i39;
                                i19 = i36;
                                long j11 = j;
                                if (i32 == 2) {
                                    if ((536870912 & i34) == 0) {
                                        i24 = zzhl.zzc(bArr3, i30, zzhnVar3);
                                    } else {
                                        i24 = zzhl.zzd(bArr3, i30, zzhnVar3);
                                    }
                                    unsafe2.putObject(t5, j11, zzhnVar3.zzc);
                                    i27 = i11 | i38;
                                    i29 = i18;
                                    i26 = i19;
                                    i25 = i17;
                                    i28 = i16;
                                    i22 = i2;
                                    i23 = i3;
                                    break;
                                } else {
                                    z2 = true;
                                    i29 = i18;
                                    i12 = i3;
                                    z = z2;
                                    i9 = i30;
                                    i10 = i19;
                                    unsafe = unsafe2;
                                    i13 = i17;
                                    i8 = i16;
                                    break;
                                }
                            case 9:
                                i16 = i31;
                                i17 = i33;
                                i18 = i39;
                                i19 = i36;
                                long j12 = j;
                                if (i32 == 2) {
                                    int zza4 = zzhl.zza(zzkoVar2.zza(i17), bArr3, i30, i2, zzhnVar3);
                                    if ((i40 & i38) == 0) {
                                        unsafe2.putObject(t5, j12, zzhnVar3.zzc);
                                    } else {
                                        unsafe2.putObject(t5, j12, zzjf.zza(unsafe2.getObject(t5, j12), zzhnVar3.zzc));
                                    }
                                    int i41 = i40 | i38;
                                    i29 = i18;
                                    i26 = i19;
                                    i28 = i16;
                                    i22 = i2;
                                    i27 = i41;
                                    i24 = zza4;
                                    i25 = i17;
                                    i23 = i3;
                                    break;
                                } else {
                                    i11 = i40;
                                    z2 = true;
                                    i29 = i18;
                                    i12 = i3;
                                    z = z2;
                                    i9 = i30;
                                    i10 = i19;
                                    unsafe = unsafe2;
                                    i13 = i17;
                                    i8 = i16;
                                    break;
                                }
                            case 10:
                                i16 = i31;
                                i17 = i33;
                                i18 = i39;
                                i19 = i36;
                                long j13 = j;
                                if (i32 == 2) {
                                    zze = zzhl.zze(bArr3, i30, zzhnVar3);
                                    unsafe2.putObject(t5, j13, zzhnVar3.zzc);
                                    i27 = i40 | i38;
                                    i29 = i18;
                                    i24 = zze;
                                    i26 = i19;
                                    i25 = i17;
                                    i28 = i16;
                                    i22 = i2;
                                    i23 = i3;
                                    break;
                                } else {
                                    i11 = i40;
                                    z2 = true;
                                    i29 = i18;
                                    i12 = i3;
                                    z = z2;
                                    i9 = i30;
                                    i10 = i19;
                                    unsafe = unsafe2;
                                    i13 = i17;
                                    i8 = i16;
                                    break;
                                }
                            case 12:
                                i16 = i31;
                                i17 = i33;
                                i18 = i39;
                                i19 = i36;
                                long j14 = j;
                                if (i32 == 0) {
                                    zze = zzhl.zza(bArr3, i30, zzhnVar3);
                                    int i42 = zzhnVar3.zza;
                                    zzjg zzc = zzkoVar2.zzc(i17);
                                    if (zzc != null && !zzc.zza(i42)) {
                                        zze(t).zza(i19, Long.valueOf(i42));
                                        i24 = zze;
                                        i27 = i40;
                                        i26 = i19;
                                        i25 = i17;
                                        i28 = i16;
                                        i29 = i18;
                                        i22 = i2;
                                        i23 = i3;
                                    } else {
                                        unsafe2.putInt(t5, j14, i42);
                                        i27 = i40 | i38;
                                        i29 = i18;
                                        i24 = zze;
                                        i26 = i19;
                                        i25 = i17;
                                        i28 = i16;
                                        i22 = i2;
                                        i23 = i3;
                                        break;
                                    }
                                } else {
                                    i11 = i40;
                                    z2 = true;
                                    i29 = i18;
                                    i12 = i3;
                                    z = z2;
                                    i9 = i30;
                                    i10 = i19;
                                    unsafe = unsafe2;
                                    i13 = i17;
                                    i8 = i16;
                                    break;
                                }
                                break;
                            case 15:
                                i16 = i31;
                                i17 = i33;
                                i18 = i39;
                                i19 = i36;
                                long j15 = j;
                                if (i32 == 0) {
                                    zze = zzhl.zza(bArr3, i30, zzhnVar3);
                                    unsafe2.putInt(t5, j15, zzif.zze(zzhnVar3.zza));
                                    i27 = i40 | i38;
                                    i29 = i18;
                                    i24 = zze;
                                    i26 = i19;
                                    i25 = i17;
                                    i28 = i16;
                                    i22 = i2;
                                    i23 = i3;
                                    break;
                                } else {
                                    i11 = i40;
                                    z2 = true;
                                    i29 = i18;
                                    i12 = i3;
                                    z = z2;
                                    i9 = i30;
                                    i10 = i19;
                                    unsafe = unsafe2;
                                    i13 = i17;
                                    i8 = i16;
                                    break;
                                }
                            case 16:
                                i16 = i31;
                                i17 = i33;
                                long j16 = j;
                                if (i32 == 0) {
                                    zzb2 = zzhl.zzb(bArr3, i30, zzhnVar3);
                                    i18 = i39;
                                    i19 = i36;
                                    unsafe2.putLong(t, j16, zzif.zza(zzhnVar3.zzb));
                                    i27 = i40 | i38;
                                    i29 = i18;
                                    i24 = zzb2;
                                    i26 = i19;
                                    i25 = i17;
                                    i28 = i16;
                                    i22 = i2;
                                    i23 = i3;
                                    break;
                                } else {
                                    i18 = i39;
                                    i19 = i36;
                                    i11 = i40;
                                    z2 = true;
                                    i29 = i18;
                                    i12 = i3;
                                    z = z2;
                                    i9 = i30;
                                    i10 = i19;
                                    unsafe = unsafe2;
                                    i13 = i17;
                                    i8 = i16;
                                    break;
                                }
                            case 17:
                                if (i32 == 3) {
                                    i16 = i31;
                                    i17 = i33;
                                    i24 = zzhl.zza(zzkoVar2.zza(i33), bArr, i30, i2, (i31 << 3) | 4, zzhnVar);
                                    if ((i40 & i38) == 0) {
                                        unsafe2.putObject(t5, j, zzhnVar3.zzc);
                                    } else {
                                        long j17 = j;
                                        unsafe2.putObject(t5, j17, zzjf.zza(unsafe2.getObject(t5, j17), zzhnVar3.zzc));
                                    }
                                    i27 = i40 | i38;
                                    i26 = i36;
                                    i29 = i39;
                                    i25 = i17;
                                    i28 = i16;
                                    i22 = i2;
                                    i23 = i3;
                                    break;
                                } else {
                                    i16 = i31;
                                    i17 = i33;
                                    i11 = i40;
                                    i18 = i39;
                                    i19 = i36;
                                    z2 = true;
                                    i29 = i18;
                                    i12 = i3;
                                    z = z2;
                                    i9 = i30;
                                    i10 = i19;
                                    unsafe = unsafe2;
                                    i13 = i17;
                                    i8 = i16;
                                    break;
                                }
                            default:
                                i16 = i31;
                                i17 = i33;
                                i11 = i40;
                                i18 = i39;
                                i19 = i36;
                                z2 = true;
                                i29 = i18;
                                i12 = i3;
                                z = z2;
                                i9 = i30;
                                i10 = i19;
                                unsafe = unsafe2;
                                i13 = i17;
                                i8 = i16;
                                break;
                        }
                    } else {
                        i11 = i27;
                        int i43 = i29;
                        t5 = t;
                        if (i35 != 27) {
                            i13 = i33;
                            if (i35 <= 49) {
                                int i44 = i30;
                                i21 = i36;
                                z = true;
                                unsafe = unsafe2;
                                i12 = i3;
                                i8 = i31;
                                i24 = zza((zzko<T>) t, bArr, i30, i2, i36, i31, i32, i13, i34, i35, j2, zzhnVar);
                                if (i24 == i44) {
                                    i9 = i24;
                                } else {
                                    t5 = t;
                                    bArr3 = bArr;
                                    i22 = i2;
                                    zzhnVar3 = zzhnVar;
                                    i23 = i12;
                                    i26 = i21;
                                    i29 = i43;
                                    i27 = i11;
                                    i25 = i13;
                                    i28 = i8;
                                    unsafe2 = unsafe;
                                    zzkoVar2 = this;
                                }
                            } else {
                                i12 = i3;
                                i20 = i30;
                                i21 = i36;
                                unsafe = unsafe2;
                                i8 = i31;
                                z = true;
                                if (i35 != 50) {
                                    i24 = zza((zzko<T>) t, bArr, i20, i2, i21, i8, i32, i34, i35, j2, i13, zzhnVar);
                                    if (i24 != i20) {
                                        t5 = t;
                                        bArr3 = bArr;
                                        i22 = i2;
                                        zzhnVar3 = zzhnVar;
                                        i26 = i21;
                                        i23 = i12;
                                        i29 = i43;
                                        i27 = i11;
                                        i25 = i13;
                                        i28 = i8;
                                        unsafe2 = unsafe;
                                        zzkoVar2 = this;
                                    }
                                } else if (i32 == 2) {
                                    i24 = zza((zzko<T>) t, bArr, i20, i2, i13, j2, zzhnVar);
                                    if (i24 != i20) {
                                        t5 = t;
                                        bArr3 = bArr;
                                        i22 = i2;
                                        zzhnVar3 = zzhnVar;
                                        i23 = i12;
                                        i26 = i21;
                                        i29 = i43;
                                        i27 = i11;
                                        i25 = i13;
                                        i28 = i8;
                                        unsafe2 = unsafe;
                                        zzkoVar2 = this;
                                    }
                                } else {
                                    i9 = i20;
                                }
                                i9 = i24;
                            }
                        } else if (i32 == 2) {
                            zzjl zzjlVar = (zzjl) unsafe2.getObject(t5, j2);
                            if (!zzjlVar.zza()) {
                                int size = zzjlVar.size();
                                zzjlVar = zzjlVar.zza(size == 0 ? 10 : size << 1);
                                unsafe2.putObject(t5, j2, zzjlVar);
                            }
                            i24 = zzhl.zza(zzkoVar2.zza(i33), i36, bArr, i30, i2, zzjlVar, zzhnVar);
                            i23 = i3;
                            i26 = i36;
                            i28 = i31;
                            i29 = i43;
                            i27 = i11;
                            i25 = i33;
                            i22 = i2;
                        } else {
                            i13 = i33;
                            i12 = i3;
                            i20 = i30;
                            i21 = i36;
                            unsafe = unsafe2;
                            i8 = i31;
                            z = true;
                            i9 = i20;
                        }
                        i10 = i21;
                        i29 = i43;
                    }
                }
                if (i10 != i12 || i12 == 0) {
                    int i45 = i12;
                    if (this.zzh) {
                        zzhnVar2 = zzhnVar;
                        if (zzhnVar2.zzd != zzio.zzb()) {
                            int i46 = i8;
                            zzjb.zze zza5 = zzhnVar2.zzd.zza(this.zzg, i46);
                            if (zza5 == null) {
                                i24 = zzhl.zza(i10, bArr, i9, i2, zze(t), zzhnVar);
                                t3 = t;
                                i14 = i29;
                                i8 = i46;
                                bArr2 = bArr;
                                i15 = i2;
                            } else {
                                t3 = t;
                                zzjb.zzc zzcVar = (zzjb.zzc) t3;
                                zzcVar.zza();
                                zziu<zzjb.zzf> zziuVar = zzcVar.zzc;
                                zzjb.zzf zzfVar = zza5.zzd;
                                boolean z3 = zzfVar.zzd;
                                zzml zzmlVar = zzfVar.zzc;
                                if (zzmlVar != zzml.zzn) {
                                    int[] iArr2 = zzhk.zza;
                                    switch (iArr2[zzmlVar.ordinal()]) {
                                        case 1:
                                            i14 = i29;
                                            i8 = i46;
                                            bArr2 = bArr;
                                            i15 = i2;
                                            obj2 = Double.valueOf(zzhl.zzc(bArr2, i9));
                                            i9 += 8;
                                            obj = obj2;
                                            break;
                                        case 2:
                                            i14 = i29;
                                            i8 = i46;
                                            bArr2 = bArr;
                                            i15 = i2;
                                            obj2 = Float.valueOf(zzhl.zzd(bArr2, i9));
                                            i9 += 4;
                                            obj = obj2;
                                            break;
                                        case 3:
                                        case 4:
                                            i14 = i29;
                                            i8 = i46;
                                            bArr2 = bArr;
                                            i15 = i2;
                                            i9 = zzhl.zzb(bArr2, i9, zzhnVar2);
                                            obj2 = Long.valueOf(zzhnVar2.zzb);
                                            obj = obj2;
                                            break;
                                        case 5:
                                        case 6:
                                            i14 = i29;
                                            i8 = i46;
                                            bArr2 = bArr;
                                            i15 = i2;
                                            i9 = zzhl.zza(bArr2, i9, zzhnVar2);
                                            obj2 = Integer.valueOf(zzhnVar2.zza);
                                            obj = obj2;
                                            break;
                                        case 7:
                                        case 8:
                                            i14 = i29;
                                            i8 = i46;
                                            bArr2 = bArr;
                                            i15 = i2;
                                            obj2 = Long.valueOf(zzhl.zzb(bArr2, i9));
                                            i9 += 8;
                                            obj = obj2;
                                            break;
                                        case 9:
                                        case 10:
                                            i14 = i29;
                                            i8 = i46;
                                            bArr2 = bArr;
                                            i15 = i2;
                                            obj2 = Integer.valueOf(zzhl.zza(bArr2, i9));
                                            i9 += 4;
                                            obj = obj2;
                                            break;
                                        case 11:
                                            i14 = i29;
                                            i8 = i46;
                                            bArr2 = bArr;
                                            i15 = i2;
                                            i9 = zzhl.zzb(bArr2, i9, zzhnVar2);
                                            if (zzhnVar2.zzb == 0) {
                                                z = false;
                                            }
                                            obj2 = Boolean.valueOf(z);
                                            obj = obj2;
                                            break;
                                        case 12:
                                            i14 = i29;
                                            i8 = i46;
                                            bArr2 = bArr;
                                            i15 = i2;
                                            i9 = zzhl.zza(bArr2, i9, zzhnVar2);
                                            obj2 = Integer.valueOf(zzif.zze(zzhnVar2.zza));
                                            obj = obj2;
                                            break;
                                        case 13:
                                            i14 = i29;
                                            i8 = i46;
                                            bArr2 = bArr;
                                            i15 = i2;
                                            i9 = zzhl.zzb(bArr2, i9, zzhnVar2);
                                            obj2 = Long.valueOf(zzif.zza(zzhnVar2.zzb));
                                            obj = obj2;
                                            break;
                                        case 14:
                                            throw new IllegalStateException("Shouldn't reach here.");
                                        case 15:
                                            i14 = i29;
                                            i8 = i46;
                                            bArr2 = bArr;
                                            i15 = i2;
                                            i9 = zzhl.zze(bArr2, i9, zzhnVar2);
                                            obj = zzhnVar2.zzc;
                                            break;
                                        case 16:
                                            i14 = i29;
                                            i8 = i46;
                                            bArr2 = bArr;
                                            i15 = i2;
                                            i9 = zzhl.zzc(bArr2, i9, zzhnVar2);
                                            obj = zzhnVar2.zzc;
                                            break;
                                        case 17:
                                            int i47 = (i46 << 3) | 4;
                                            i14 = i29;
                                            i15 = i2;
                                            i8 = i46;
                                            bArr2 = bArr;
                                            i9 = zzhl.zza(zzky.zza().zza((Class) zza5.zzc.getClass()), bArr, i9, i2, i47, zzhnVar);
                                            obj = zzhnVar2.zzc;
                                            break;
                                        case 18:
                                            i9 = zzhl.zza(zzky.zza().zza((Class) zza5.zzc.getClass()), bArr, i9, i2, zzhnVar2);
                                            obj = zzhnVar2.zzc;
                                            i14 = i29;
                                            i8 = i46;
                                            i15 = i2;
                                            bArr2 = bArr;
                                            break;
                                        default:
                                            i14 = i29;
                                            i8 = i46;
                                            bArr2 = bArr;
                                            i15 = i2;
                                            obj = obj2;
                                            break;
                                    }
                                    zzjb.zzf zzfVar2 = zza5.zzd;
                                    if (zzfVar2.zzd) {
                                        zziuVar.zzb(zzfVar2, obj);
                                    } else {
                                        int i48 = iArr2[zzfVar2.zzc.ordinal()];
                                        if ((i48 == 17 || i48 == 18) && (zza2 = zziuVar.zza((zziu<zzjb.zzf>) zza5.zzd)) != null) {
                                            obj = zzjf.zza(zza2, obj);
                                        }
                                        zziuVar.zza((zziu<zzjb.zzf>) zza5.zzd, obj);
                                    }
                                    i24 = i9;
                                } else {
                                    zzhl.zza(bArr, i9, zzhnVar2);
                                    throw null;
                                }
                            }
                            i26 = i10;
                            zzkoVar2 = this;
                            bArr3 = bArr2;
                            t5 = t3;
                            i27 = i11;
                            i25 = i13;
                            i28 = i8;
                            i22 = i15;
                            i23 = i45;
                            zzhnVar3 = zzhnVar2;
                            unsafe2 = unsafe;
                            i29 = i14;
                        } else {
                            t3 = t;
                            bArr2 = bArr;
                        }
                    } else {
                        t3 = t;
                        bArr2 = bArr;
                        zzhnVar2 = zzhnVar;
                    }
                    i14 = i29;
                    i15 = i2;
                    i24 = zzhl.zza(i10, bArr, i9, i2, zze(t), zzhnVar);
                    i26 = i10;
                    zzkoVar2 = this;
                    bArr3 = bArr2;
                    t5 = t3;
                    i27 = i11;
                    i25 = i13;
                    i28 = i8;
                    i22 = i15;
                    i23 = i45;
                    zzhnVar3 = zzhnVar2;
                    unsafe2 = unsafe;
                    i29 = i14;
                } else {
                    zzkoVar = this;
                    t2 = t;
                    i24 = i9;
                    i6 = i29;
                    i26 = i10;
                    i4 = i12;
                    i27 = i11;
                    i7 = 1048575;
                    i5 = i2;
                }
            } else {
                int i49 = i29;
                unsafe = unsafe2;
                i4 = i23;
                i5 = i22;
                t2 = t5;
                zzkoVar = zzkoVar2;
                i6 = i49;
                i7 = 1048575;
            }
        }
        if (i6 != i7) {
            unsafe.putInt(t2, i6, i27);
        }
        zzlx zzlxVar = null;
        for (int i50 = zzkoVar.zzm; i50 < zzkoVar.zzn; i50++) {
            zzlxVar = (zzlx) zzkoVar.zza((Object) t2, zzkoVar.zzl[i50], (int) zzlxVar, (zzlu<UT, int>) zzkoVar.zzq);
        }
        if (zzlxVar != null) {
            zzkoVar.zzq.zzb((Object) t2, (T) zzlxVar);
        }
        if (i4 == 0) {
            if (i24 != i5) {
                throw zzjk.zzg();
            }
        } else if (i24 > i5 || i26 != i4) {
            throw zzjk.zzg();
        }
        return i24;
    }

    /* JADX WARN: Code restructure failed: missing block: B:101:0x02d1, code lost:
        if (r0 == r5) goto L136;
     */
    /* JADX WARN: Code restructure failed: missing block: B:103:0x02d5, code lost:
        r15 = r30;
        r14 = r31;
        r12 = r32;
        r13 = r34;
        r11 = r35;
        r2 = r18;
        r10 = r20;
        r1 = r25;
        r6 = r27;
        r7 = r28;
     */
    /* JADX WARN: Code restructure failed: missing block: B:109:0x031a, code lost:
        if (r0 == r15) goto L136;
     */
    /* JADX WARN: Code restructure failed: missing block: B:114:0x033d, code lost:
        if (r0 == r15) goto L136;
     */
    /* JADX WARN: Code restructure failed: missing block: B:115:0x033f, code lost:
        r2 = r0;
     */
    /* JADX WARN: Multi-variable type inference failed */
    /* JADX WARN: Type inference failed for: r3v11, types: [int] */
    @Override // com.google.android.gms.internal.vision.zzlc
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public final void zza(T t, byte[] bArr, int i, int i2, zzhn zzhnVar) {
        byte b2;
        int i3;
        int zzg;
        int i4;
        int i5;
        Unsafe unsafe;
        int i6;
        int i7;
        int i8;
        int i9;
        int i10;
        int i11;
        int i12;
        Unsafe unsafe2;
        int i13;
        int i14;
        Unsafe unsafe3;
        Unsafe unsafe4;
        zzko<T> zzkoVar = this;
        T t2 = t;
        byte[] bArr2 = bArr;
        int i15 = i2;
        zzhn zzhnVar2 = zzhnVar;
        if (zzkoVar.zzj) {
            Unsafe unsafe5 = zzb;
            int i16 = -1;
            int i17 = 1048575;
            int i18 = i;
            int i19 = 1048575;
            int i20 = -1;
            int i21 = 0;
            int i22 = 0;
            while (i18 < i15) {
                int i23 = i18 + 1;
                byte b3 = bArr2[i18];
                if (b3 < 0) {
                    i3 = zzhl.zza(b3, bArr2, i23, zzhnVar2);
                    b2 = zzhnVar2.zza;
                } else {
                    b2 = b3;
                    i3 = i23;
                }
                int i24 = b2 >>> 3;
                int i25 = b2 & 7;
                if (i24 > i20) {
                    zzg = zzkoVar.zza(i24, i21 / 3);
                } else {
                    zzg = zzkoVar.zzg(i24);
                }
                int i26 = zzg;
                if (i26 == i16) {
                    i4 = i3;
                    i5 = i24;
                    unsafe = unsafe5;
                    i6 = i16;
                    i7 = 0;
                } else {
                    int[] iArr = zzkoVar.zzc;
                    int i27 = iArr[i26 + 1];
                    int i28 = (i27 & 267386880) >>> 20;
                    Unsafe unsafe6 = unsafe5;
                    long j = i27 & i17;
                    if (i28 <= 17) {
                        int i29 = iArr[i26 + 2];
                        int i30 = 1 << (i29 >>> 20);
                        int i31 = i29 & 1048575;
                        if (i31 != i19) {
                            if (i19 != 1048575) {
                                long j2 = i19;
                                unsafe4 = unsafe6;
                                unsafe4.putInt(t2, j2, i22);
                            } else {
                                unsafe4 = unsafe6;
                            }
                            if (i31 != 1048575) {
                                i22 = unsafe4.getInt(t2, i31);
                            }
                            unsafe2 = unsafe4;
                            i19 = i31;
                        } else {
                            unsafe2 = unsafe6;
                        }
                        switch (i28) {
                            case 0:
                                i5 = i24;
                                i9 = 1048575;
                                i13 = i26;
                                i14 = i3;
                                i8 = i19;
                                unsafe3 = unsafe2;
                                if (i25 != 1) {
                                    i4 = i14;
                                    unsafe = unsafe3;
                                    i7 = i13;
                                    i19 = i8;
                                    i6 = -1;
                                    break;
                                } else {
                                    zzma.zza(t2, j, zzhl.zzc(bArr2, i14));
                                    i18 = i14 + 8;
                                    i22 |= i30;
                                    unsafe5 = unsafe3;
                                    i21 = i13;
                                    break;
                                }
                            case 1:
                                i5 = i24;
                                i9 = 1048575;
                                i13 = i26;
                                i14 = i3;
                                i8 = i19;
                                unsafe3 = unsafe2;
                                if (i25 != 5) {
                                    i4 = i14;
                                    unsafe = unsafe3;
                                    i7 = i13;
                                    i19 = i8;
                                    i6 = -1;
                                    break;
                                } else {
                                    zzma.zza((Object) t2, j, zzhl.zzd(bArr2, i14));
                                    i18 = i14 + 4;
                                    i22 |= i30;
                                    unsafe5 = unsafe3;
                                    i21 = i13;
                                    break;
                                }
                            case 2:
                            case 3:
                                i5 = i24;
                                i9 = 1048575;
                                i13 = i26;
                                i14 = i3;
                                i8 = i19;
                                unsafe3 = unsafe2;
                                if (i25 != 0) {
                                    i4 = i14;
                                    unsafe = unsafe3;
                                    i7 = i13;
                                    i19 = i8;
                                    i6 = -1;
                                    break;
                                } else {
                                    int zzb2 = zzhl.zzb(bArr2, i14, zzhnVar2);
                                    unsafe3.putLong(t, j, zzhnVar2.zzb);
                                    i22 |= i30;
                                    unsafe5 = unsafe3;
                                    i21 = i13;
                                    i18 = zzb2;
                                    break;
                                }
                            case 4:
                            case 11:
                                i5 = i24;
                                i9 = 1048575;
                                i13 = i26;
                                i14 = i3;
                                i8 = i19;
                                unsafe3 = unsafe2;
                                if (i25 != 0) {
                                    i4 = i14;
                                    unsafe = unsafe3;
                                    i7 = i13;
                                    i19 = i8;
                                    i6 = -1;
                                    break;
                                } else {
                                    i18 = zzhl.zza(bArr2, i14, zzhnVar2);
                                    unsafe3.putInt(t2, j, zzhnVar2.zza);
                                    i22 |= i30;
                                    unsafe5 = unsafe3;
                                    i21 = i13;
                                    break;
                                }
                            case 5:
                            case 14:
                                i5 = i24;
                                i9 = 1048575;
                                i13 = i26;
                                i8 = i19;
                                unsafe3 = unsafe2;
                                if (i25 != 1) {
                                    i14 = i3;
                                    i4 = i14;
                                    unsafe = unsafe3;
                                    i7 = i13;
                                    i19 = i8;
                                    i6 = -1;
                                    break;
                                } else {
                                    unsafe3.putLong(t, j, zzhl.zzb(bArr2, i3));
                                    i18 = i3 + 8;
                                    i22 |= i30;
                                    unsafe5 = unsafe3;
                                    i21 = i13;
                                    break;
                                }
                            case 6:
                            case 13:
                                i5 = i24;
                                i9 = 1048575;
                                i13 = i26;
                                i8 = i19;
                                unsafe3 = unsafe2;
                                if (i25 != 5) {
                                    i14 = i3;
                                    i4 = i14;
                                    unsafe = unsafe3;
                                    i7 = i13;
                                    i19 = i8;
                                    i6 = -1;
                                    break;
                                } else {
                                    unsafe3.putInt(t2, j, zzhl.zza(bArr2, i3));
                                    i18 = i3 + 4;
                                    i22 |= i30;
                                    unsafe5 = unsafe3;
                                    i21 = i13;
                                    break;
                                }
                            case 7:
                                i5 = i24;
                                i9 = 1048575;
                                i13 = i26;
                                i8 = i19;
                                unsafe3 = unsafe2;
                                if (i25 != 0) {
                                    i14 = i3;
                                    i4 = i14;
                                    unsafe = unsafe3;
                                    i7 = i13;
                                    i19 = i8;
                                    i6 = -1;
                                    break;
                                } else {
                                    i18 = zzhl.zzb(bArr2, i3, zzhnVar2);
                                    zzma.zza(t2, j, zzhnVar2.zzb != 0);
                                    i22 |= i30;
                                    unsafe5 = unsafe3;
                                    i21 = i13;
                                    break;
                                }
                            case 8:
                                i5 = i24;
                                i9 = 1048575;
                                i13 = i26;
                                i8 = i19;
                                unsafe3 = unsafe2;
                                if (i25 != 2) {
                                    i14 = i3;
                                    i4 = i14;
                                    unsafe = unsafe3;
                                    i7 = i13;
                                    i19 = i8;
                                    i6 = -1;
                                    break;
                                } else {
                                    if ((i27 & 536870912) == 0) {
                                        i18 = zzhl.zzc(bArr2, i3, zzhnVar2);
                                    } else {
                                        i18 = zzhl.zzd(bArr2, i3, zzhnVar2);
                                    }
                                    unsafe3.putObject(t2, j, zzhnVar2.zzc);
                                    i22 |= i30;
                                    unsafe5 = unsafe3;
                                    i21 = i13;
                                    break;
                                }
                            case 9:
                                i5 = i24;
                                i9 = 1048575;
                                i13 = i26;
                                i8 = i19;
                                unsafe3 = unsafe2;
                                if (i25 != 2) {
                                    i14 = i3;
                                    i4 = i14;
                                    unsafe = unsafe3;
                                    i7 = i13;
                                    i19 = i8;
                                    i6 = -1;
                                    break;
                                } else {
                                    i18 = zzhl.zza(zzkoVar.zza(i13), bArr2, i3, i15, zzhnVar2);
                                    Object object = unsafe3.getObject(t2, j);
                                    if (object == null) {
                                        unsafe3.putObject(t2, j, zzhnVar2.zzc);
                                    } else {
                                        unsafe3.putObject(t2, j, zzjf.zza(object, zzhnVar2.zzc));
                                    }
                                    i22 |= i30;
                                    unsafe5 = unsafe3;
                                    i21 = i13;
                                    break;
                                }
                            case 10:
                                i5 = i24;
                                i9 = 1048575;
                                i13 = i26;
                                i8 = i19;
                                unsafe3 = unsafe2;
                                if (i25 != 2) {
                                    i14 = i3;
                                    i4 = i14;
                                    unsafe = unsafe3;
                                    i7 = i13;
                                    i19 = i8;
                                    i6 = -1;
                                    break;
                                } else {
                                    i18 = zzhl.zze(bArr2, i3, zzhnVar2);
                                    unsafe3.putObject(t2, j, zzhnVar2.zzc);
                                    i22 |= i30;
                                    unsafe5 = unsafe3;
                                    i21 = i13;
                                    break;
                                }
                            case 12:
                                i5 = i24;
                                i9 = 1048575;
                                i13 = i26;
                                i8 = i19;
                                unsafe3 = unsafe2;
                                if (i25 != 0) {
                                    i14 = i3;
                                    i4 = i14;
                                    unsafe = unsafe3;
                                    i7 = i13;
                                    i19 = i8;
                                    i6 = -1;
                                    break;
                                } else {
                                    i18 = zzhl.zza(bArr2, i3, zzhnVar2);
                                    unsafe3.putInt(t2, j, zzhnVar2.zza);
                                    i22 |= i30;
                                    unsafe5 = unsafe3;
                                    i21 = i13;
                                    break;
                                }
                            case 15:
                                i5 = i24;
                                i9 = 1048575;
                                i13 = i26;
                                i8 = i19;
                                unsafe3 = unsafe2;
                                if (i25 != 0) {
                                    i14 = i3;
                                    i4 = i14;
                                    unsafe = unsafe3;
                                    i7 = i13;
                                    i19 = i8;
                                    i6 = -1;
                                    break;
                                } else {
                                    i18 = zzhl.zza(bArr2, i3, zzhnVar2);
                                    unsafe3.putInt(t2, j, zzif.zze(zzhnVar2.zza));
                                    i22 |= i30;
                                    unsafe5 = unsafe3;
                                    i21 = i13;
                                    break;
                                }
                            case 16:
                                if (i25 != 0) {
                                    i5 = i24;
                                    i8 = i19;
                                    unsafe3 = unsafe2;
                                    i14 = i3;
                                    i13 = i26;
                                    i4 = i14;
                                    unsafe = unsafe3;
                                    i7 = i13;
                                    i19 = i8;
                                    i6 = -1;
                                    break;
                                } else {
                                    int zzb3 = zzhl.zzb(bArr2, i3, zzhnVar2);
                                    i8 = i19;
                                    i5 = i24;
                                    i9 = 1048575;
                                    unsafe2.putLong(t, j, zzif.zza(zzhnVar2.zzb));
                                    i22 |= i30;
                                    unsafe5 = unsafe2;
                                    i21 = i26;
                                    i18 = zzb3;
                                    break;
                                }
                            default:
                                i5 = i24;
                                i13 = i26;
                                i14 = i3;
                                i8 = i19;
                                unsafe3 = unsafe2;
                                i4 = i14;
                                unsafe = unsafe3;
                                i7 = i13;
                                i19 = i8;
                                i6 = -1;
                                break;
                        }
                        i19 = i8;
                        i20 = i5;
                        i17 = i9;
                        i16 = -1;
                    } else {
                        i5 = i24;
                        int i32 = i3;
                        i8 = i19;
                        i9 = 1048575;
                        if (i28 != 27) {
                            i7 = i26;
                            if (i28 <= 49) {
                                i11 = i22;
                                i12 = i8;
                                unsafe = unsafe6;
                                i6 = -1;
                                i18 = zza((zzko<T>) t, bArr, i32, i2, b2, i5, i25, i7, i27, i28, j, zzhnVar);
                            } else {
                                i10 = i32;
                                i11 = i22;
                                unsafe = unsafe6;
                                i12 = i8;
                                i6 = -1;
                                if (i28 != 50) {
                                    i18 = zza((zzko<T>) t, bArr, i10, i2, b2, i5, i25, i27, i28, j, i7, zzhnVar);
                                } else if (i25 == 2) {
                                    i18 = zza((zzko<T>) t, bArr, i10, i2, i7, j, zzhnVar);
                                }
                            }
                            unsafe5 = unsafe;
                            i17 = 1048575;
                        } else if (i25 == 2) {
                            zzjl zzjlVar = (zzjl) unsafe6.getObject(t2, j);
                            if (!zzjlVar.zza()) {
                                int size = zzjlVar.size();
                                zzjlVar = zzjlVar.zza(size == 0 ? 10 : size << 1);
                                unsafe6.putObject(t2, j, zzjlVar);
                            }
                            i18 = zzhl.zza(zzkoVar.zza(i26), b2, bArr, i32, i2, zzjlVar, zzhnVar);
                            unsafe5 = unsafe6;
                            i22 = i22;
                            i21 = i26;
                            i19 = i8;
                            i20 = i5;
                            i17 = i9;
                            i16 = -1;
                        } else {
                            i7 = i26;
                            i10 = i32;
                            i11 = i22;
                            unsafe = unsafe6;
                            i12 = i8;
                            i6 = -1;
                        }
                        i4 = i10;
                        i22 = i11;
                        i19 = i12;
                    }
                }
                i18 = zzhl.zza(b2, bArr, i4, i2, zze(t), zzhnVar);
                zzkoVar = this;
                t2 = t;
                bArr2 = bArr;
                i15 = i2;
                zzhnVar2 = zzhnVar;
                i21 = i7;
                i16 = i6;
                i20 = i5;
                unsafe5 = unsafe;
                i17 = 1048575;
            }
            int i33 = i22;
            Unsafe unsafe7 = unsafe5;
            if (i19 != i17) {
                unsafe7.putInt(t, i19, i33);
            }
            if (i18 != i2) {
                throw zzjk.zzg();
            }
            return;
        }
        zza((zzko<T>) t, bArr, i, i2, 0, zzhnVar);
    }

    /* JADX DEBUG: Type inference failed for r4v0. Raw type applied. Possible types: java.util.Map<?, ?>, java.util.Map<K, V> */
    private final <UT, UB> UB zza(Object obj, int i, UB ub, zzlu<UT, UB> zzluVar) {
        zzjg zzc;
        int i2 = this.zzc[i];
        Object zzf = zzma.zzf(obj, zzd(i) & 1048575);
        return (zzf == null || (zzc = zzc(i)) == null) ? ub : (UB) zza(i, i2, this.zzs.zza(zzf), zzc, (zzjg) ub, (zzlu<UT, zzjg>) zzluVar);
    }

    private final <K, V, UT, UB> UB zza(int i, int i2, Map<K, V> map, zzjg zzjgVar, UB ub, zzlu<UT, UB> zzluVar) {
        zzkf<?, ?> zzb2 = this.zzs.zzb(zzb(i));
        Iterator<Map.Entry<K, V>> it = map.entrySet().iterator();
        while (it.hasNext()) {
            Map.Entry<K, V> next = it.next();
            if (!zzjgVar.zza(((Integer) next.getValue()).intValue())) {
                if (ub == null) {
                    ub = zzluVar.zza();
                }
                zzib zzc = zzht.zzc(zzkc.zza(zzb2, next.getKey(), next.getValue()));
                try {
                    zzkc.zza(zzc.zzb(), zzb2, next.getKey(), next.getValue());
                    zzluVar.zza((zzlu<UT, UB>) ub, i2, zzc.zza());
                    it.remove();
                } catch (IOException e2) {
                    throw new RuntimeException(e2);
                }
            }
        }
        return ub;
    }

    /* JADX DEBUG: Multi-variable search result rejected for r4v0, resolved type: com.google.android.gms.internal.vision.zzlc */
    /* JADX WARN: Multi-variable type inference failed */
    private static boolean zza(Object obj, int i, zzlc zzlcVar) {
        return zzlcVar.zzd(zzma.zzf(obj, i & 1048575));
    }

    private static void zza(int i, Object obj, zzmr zzmrVar) {
        if (obj instanceof String) {
            zzmrVar.zza(i, (String) obj);
        } else {
            zzmrVar.zza(i, (zzht) obj);
        }
    }

    private final void zza(Object obj, int i, zzld zzldVar) {
        if (zzf(i)) {
            zzma.zza(obj, i & 1048575, zzldVar.zzm());
        } else if (this.zzi) {
            zzma.zza(obj, i & 1048575, zzldVar.zzl());
        } else {
            zzma.zza(obj, i & 1048575, zzldVar.zzn());
        }
    }

    private final boolean zza(T t, int i, int i2, int i3, int i4) {
        if (i2 == 1048575) {
            return zza((zzko<T>) t, i);
        }
        return (i3 & i4) != 0;
    }

    private final boolean zza(T t, int i) {
        int zze = zze(i);
        long j = zze & 1048575;
        if (j != 1048575) {
            return (zzma.zza(t, j) & (1 << (zze >>> 20))) != 0;
        }
        int zzd = zzd(i);
        long j2 = zzd & 1048575;
        switch ((zzd & 267386880) >>> 20) {
            case 0:
                return zzma.zze(t, j2) != ShadowDrawableWrapper.COS_45;
            case 1:
                return zzma.zzd(t, j2) != StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
            case 2:
                return zzma.zzb(t, j2) != 0;
            case 3:
                return zzma.zzb(t, j2) != 0;
            case 4:
                return zzma.zza(t, j2) != 0;
            case 5:
                return zzma.zzb(t, j2) != 0;
            case 6:
                return zzma.zza(t, j2) != 0;
            case 7:
                return zzma.zzc(t, j2);
            case 8:
                Object zzf = zzma.zzf(t, j2);
                if (zzf instanceof String) {
                    return !((String) zzf).isEmpty();
                } else if (zzf instanceof zzht) {
                    return !zzht.zza.equals(zzf);
                } else {
                    throw new IllegalArgumentException();
                }
            case 9:
                return zzma.zzf(t, j2) != null;
            case 10:
                return !zzht.zza.equals(zzma.zzf(t, j2));
            case 11:
                return zzma.zza(t, j2) != 0;
            case 12:
                return zzma.zza(t, j2) != 0;
            case 13:
                return zzma.zza(t, j2) != 0;
            case 14:
                return zzma.zzb(t, j2) != 0;
            case 15:
                return zzma.zza(t, j2) != 0;
            case 16:
                return zzma.zzb(t, j2) != 0;
            case 17:
                return zzma.zzf(t, j2) != null;
            default:
                throw new IllegalArgumentException();
        }
    }

    private final boolean zza(T t, int i, int i2) {
        return zzma.zza(t, (long) (zze(i2) & 1048575)) == i;
    }

    private final int zza(int i, int i2) {
        if (i < this.zze || i > this.zzf) {
            return -1;
        }
        return zzb(i, i2);
    }
}