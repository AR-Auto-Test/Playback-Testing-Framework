package com.google.android.gms.internal.measurement;

import java.lang.reflect.Field;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import sun.misc.Unsafe;

/* compiled from: com.google.android.gms:play-services-measurement-base@@21.2.0 */
/* loaded from: classes.dex */
public final class zzlp<T> implements zzlx<T> {
    private static final int[] zza = new int[0];
    private static final Unsafe zzb = zzmy.zzg();
    private final int[] zzc;
    private final Object[] zzd;
    private final int zze;
    private final int zzf;
    private final zzlm zzg;
    private final boolean zzh;
    private final boolean zzi;
    private final int[] zzj;
    private final int zzk;
    private final int zzl;
    private final zzla zzm;
    private final zzmo zzn;
    private final zzjs zzo;
    private final zzlr zzp;
    private final zzlh zzq;

    private zzlp(int[] iArr, Object[] objArr, int i, int i2, zzlm zzlmVar, boolean z, boolean z2, int[] iArr2, int i3, int i4, zzlr zzlrVar, zzla zzlaVar, zzmo zzmoVar, zzjs zzjsVar, zzlh zzlhVar, byte[] bArr) {
        this.zzc = iArr;
        this.zzd = objArr;
        this.zze = i;
        this.zzf = i2;
        this.zzi = z;
        boolean z3 = false;
        if (zzjsVar != null && zzjsVar.zzc(zzlmVar)) {
            z3 = true;
        }
        this.zzh = z3;
        this.zzj = iArr2;
        this.zzk = i3;
        this.zzl = i4;
        this.zzp = zzlrVar;
        this.zzm = zzlaVar;
        this.zzn = zzmoVar;
        this.zzo = zzjsVar;
        this.zzg = zzlmVar;
        this.zzq = zzlhVar;
    }

    private static int zzA(int i) {
        return (i >>> 20) & 255;
    }

    private final int zzB(int i) {
        return this.zzc[i + 1];
    }

    private static long zzC(Object obj, long j) {
        return ((Long) zzmy.zzf(obj, j)).longValue();
    }

    private final zzkj zzD(int i) {
        int i2 = i / 3;
        return (zzkj) this.zzd[i2 + i2 + 1];
    }

    private final zzlx zzE(int i) {
        int i2 = i / 3;
        int i3 = i2 + i2;
        zzlx zzlxVar = (zzlx) this.zzd[i3];
        if (zzlxVar != null) {
            return zzlxVar;
        }
        zzlx zzb2 = zzlu.zza().zzb((Class) this.zzd[i3 + 1]);
        this.zzd[i3] = zzb2;
        return zzb2;
    }

    private final Object zzF(int i) {
        int i2 = i / 3;
        return this.zzd[i2 + i2];
    }

    private final Object zzG(Object obj, int i) {
        zzlx zzE = zzE(i);
        long zzB = zzB(i) & 1048575;
        if (!zzT(obj, i)) {
            return zzE.zze();
        }
        Object object = zzb.getObject(obj, zzB);
        if (zzW(object)) {
            return object;
        }
        Object zze = zzE.zze();
        if (object != null) {
            zzE.zzg(zze, object);
        }
        return zze;
    }

    private final Object zzH(Object obj, int i, int i2) {
        zzlx zzE = zzE(i2);
        if (!zzX(obj, i, i2)) {
            return zzE.zze();
        }
        Object object = zzb.getObject(obj, zzB(i2) & 1048575);
        if (zzW(object)) {
            return object;
        }
        Object zze = zzE.zze();
        if (object != null) {
            zzE.zzg(zze, object);
        }
        return zze;
    }

    private static Field zzI(Class cls, String str) {
        try {
            return cls.getDeclaredField(str);
        } catch (NoSuchFieldException unused) {
            Field[] declaredFields = cls.getDeclaredFields();
            for (Field field : declaredFields) {
                if (str.equals(field.getName())) {
                    return field;
                }
            }
            throw new RuntimeException("Field " + str + " for " + cls.getName() + " not found. Known fields are " + Arrays.toString(declaredFields));
        }
    }

    private static void zzJ(Object obj) {
        if (!zzW(obj)) {
            throw new IllegalArgumentException("Mutating immutable message: ".concat(String.valueOf(obj)));
        }
    }

    private final void zzK(Object obj, Object obj2, int i) {
        if (zzT(obj2, i)) {
            long zzB = zzB(i) & 1048575;
            Unsafe unsafe = zzb;
            Object object = unsafe.getObject(obj2, zzB);
            if (object != null) {
                zzlx zzE = zzE(i);
                if (!zzT(obj, i)) {
                    if (!zzW(object)) {
                        unsafe.putObject(obj, zzB, object);
                    } else {
                        Object zze = zzE.zze();
                        zzE.zzg(zze, object);
                        unsafe.putObject(obj, zzB, zze);
                    }
                    zzM(obj, i);
                    return;
                }
                Object object2 = unsafe.getObject(obj, zzB);
                if (!zzW(object2)) {
                    Object zze2 = zzE.zze();
                    zzE.zzg(zze2, object2);
                    unsafe.putObject(obj, zzB, zze2);
                    object2 = zze2;
                }
                zzE.zzg(object2, object);
                return;
            }
            throw new IllegalStateException("Source subfield " + this.zzc[i] + " is present but null: " + obj2.toString());
        }
    }

    private final void zzL(Object obj, Object obj2, int i) {
        int i2 = this.zzc[i];
        if (zzX(obj2, i2, i)) {
            long zzB = zzB(i) & 1048575;
            Unsafe unsafe = zzb;
            Object object = unsafe.getObject(obj2, zzB);
            if (object != null) {
                zzlx zzE = zzE(i);
                if (!zzX(obj, i2, i)) {
                    if (!zzW(object)) {
                        unsafe.putObject(obj, zzB, object);
                    } else {
                        Object zze = zzE.zze();
                        zzE.zzg(zze, object);
                        unsafe.putObject(obj, zzB, zze);
                    }
                    zzN(obj, i2, i);
                    return;
                }
                Object object2 = unsafe.getObject(obj, zzB);
                if (!zzW(object2)) {
                    Object zze2 = zzE.zze();
                    zzE.zzg(zze2, object2);
                    unsafe.putObject(obj, zzB, zze2);
                    object2 = zze2;
                }
                zzE.zzg(object2, object);
                return;
            }
            throw new IllegalStateException("Source subfield " + this.zzc[i] + " is present but null: " + obj2.toString());
        }
    }

    private final void zzM(Object obj, int i) {
        int zzy = zzy(i);
        long j = 1048575 & zzy;
        if (j == 1048575) {
            return;
        }
        zzmy.zzq(obj, j, (1 << (zzy >>> 20)) | zzmy.zzc(obj, j));
    }

    private final void zzN(Object obj, int i, int i2) {
        zzmy.zzq(obj, zzy(i2) & 1048575, i);
    }

    private final void zzO(Object obj, int i, Object obj2) {
        zzb.putObject(obj, zzB(i) & 1048575, obj2);
        zzM(obj, i);
    }

    private final void zzP(Object obj, int i, int i2, Object obj2) {
        zzb.putObject(obj, zzB(i2) & 1048575, obj2);
        zzN(obj, i, i2);
    }

    /* JADX WARN: Can't fix incorrect switch cases order, some code will duplicate */
    private final void zzQ(Object obj, zzng zzngVar) {
        int i;
        boolean z;
        if (!this.zzh) {
            int length = this.zzc.length;
            Unsafe unsafe = zzb;
            int i2 = 1048575;
            int i3 = 1048575;
            int i4 = 0;
            int i5 = 0;
            while (i4 < length) {
                int zzB = zzB(i4);
                int[] iArr = this.zzc;
                int i6 = iArr[i4];
                int zzA = zzA(zzB);
                if (zzA <= 17) {
                    int i7 = iArr[i4 + 2];
                    int i8 = i7 & i2;
                    if (i8 != i3) {
                        i5 = unsafe.getInt(obj, i8);
                        i3 = i8;
                    }
                    i = 1 << (i7 >>> 20);
                } else {
                    i = 0;
                }
                long j = zzB & i2;
                switch (zzA) {
                    case 0:
                        if ((i5 & i) == 0) {
                            break;
                        } else {
                            zzngVar.zzf(i6, zzmy.zza(obj, j));
                            break;
                        }
                    case 1:
                        if ((i5 & i) == 0) {
                            break;
                        } else {
                            zzngVar.zzo(i6, zzmy.zzb(obj, j));
                            break;
                        }
                    case 2:
                        if ((i5 & i) == 0) {
                            break;
                        } else {
                            zzngVar.zzt(i6, unsafe.getLong(obj, j));
                            break;
                        }
                    case 3:
                        if ((i5 & i) == 0) {
                            break;
                        } else {
                            zzngVar.zzJ(i6, unsafe.getLong(obj, j));
                            break;
                        }
                    case 4:
                        if ((i5 & i) == 0) {
                            break;
                        } else {
                            zzngVar.zzr(i6, unsafe.getInt(obj, j));
                            break;
                        }
                    case 5:
                        if ((i5 & i) == 0) {
                            break;
                        } else {
                            zzngVar.zzm(i6, unsafe.getLong(obj, j));
                            break;
                        }
                    case 6:
                        if ((i5 & i) == 0) {
                            break;
                        } else {
                            zzngVar.zzk(i6, unsafe.getInt(obj, j));
                            break;
                        }
                    case 7:
                        if ((i5 & i) == 0) {
                            break;
                        } else {
                            zzngVar.zzb(i6, zzmy.zzw(obj, j));
                            break;
                        }
                    case 8:
                        if ((i5 & i) == 0) {
                            break;
                        } else {
                            zzZ(i6, unsafe.getObject(obj, j), zzngVar);
                            break;
                        }
                    case 9:
                        if ((i5 & i) == 0) {
                            break;
                        } else {
                            zzngVar.zzv(i6, unsafe.getObject(obj, j), zzE(i4));
                            break;
                        }
                    case 10:
                        if ((i5 & i) == 0) {
                            break;
                        } else {
                            zzngVar.zzd(i6, (zzje) unsafe.getObject(obj, j));
                            break;
                        }
                    case 11:
                        if ((i5 & i) == 0) {
                            break;
                        } else {
                            zzngVar.zzH(i6, unsafe.getInt(obj, j));
                            break;
                        }
                    case 12:
                        if ((i5 & i) == 0) {
                            break;
                        } else {
                            zzngVar.zzi(i6, unsafe.getInt(obj, j));
                            break;
                        }
                    case 13:
                        if ((i5 & i) == 0) {
                            break;
                        } else {
                            zzngVar.zzw(i6, unsafe.getInt(obj, j));
                            break;
                        }
                    case 14:
                        if ((i5 & i) == 0) {
                            break;
                        } else {
                            zzngVar.zzy(i6, unsafe.getLong(obj, j));
                            break;
                        }
                    case 15:
                        if ((i5 & i) == 0) {
                            break;
                        } else {
                            zzngVar.zzA(i6, unsafe.getInt(obj, j));
                            break;
                        }
                    case 16:
                        if ((i5 & i) == 0) {
                            break;
                        } else {
                            zzngVar.zzC(i6, unsafe.getLong(obj, j));
                            break;
                        }
                    case 17:
                        if ((i5 & i) == 0) {
                            break;
                        } else {
                            zzngVar.zzq(i6, unsafe.getObject(obj, j), zzE(i4));
                            break;
                        }
                    case 18:
                        zzlz.zzJ(this.zzc[i4], (List) unsafe.getObject(obj, j), zzngVar, false);
                        break;
                    case 19:
                        zzlz.zzN(this.zzc[i4], (List) unsafe.getObject(obj, j), zzngVar, false);
                        break;
                    case 20:
                        zzlz.zzQ(this.zzc[i4], (List) unsafe.getObject(obj, j), zzngVar, false);
                        break;
                    case 21:
                        zzlz.zzY(this.zzc[i4], (List) unsafe.getObject(obj, j), zzngVar, false);
                        break;
                    case 22:
                        zzlz.zzP(this.zzc[i4], (List) unsafe.getObject(obj, j), zzngVar, false);
                        break;
                    case 23:
                        zzlz.zzM(this.zzc[i4], (List) unsafe.getObject(obj, j), zzngVar, false);
                        break;
                    case 24:
                        zzlz.zzL(this.zzc[i4], (List) unsafe.getObject(obj, j), zzngVar, false);
                        break;
                    case 25:
                        zzlz.zzH(this.zzc[i4], (List) unsafe.getObject(obj, j), zzngVar, false);
                        break;
                    case 26:
                        zzlz.zzW(this.zzc[i4], (List) unsafe.getObject(obj, j), zzngVar);
                        break;
                    case 27:
                        zzlz.zzR(this.zzc[i4], (List) unsafe.getObject(obj, j), zzngVar, zzE(i4));
                        break;
                    case 28:
                        zzlz.zzI(this.zzc[i4], (List) unsafe.getObject(obj, j), zzngVar);
                        break;
                    case 29:
                        z = false;
                        zzlz.zzX(this.zzc[i4], (List) unsafe.getObject(obj, j), zzngVar, false);
                        break;
                    case 30:
                        z = false;
                        zzlz.zzK(this.zzc[i4], (List) unsafe.getObject(obj, j), zzngVar, false);
                        break;
                    case 31:
                        z = false;
                        zzlz.zzS(this.zzc[i4], (List) unsafe.getObject(obj, j), zzngVar, false);
                        break;
                    case 32:
                        z = false;
                        zzlz.zzT(this.zzc[i4], (List) unsafe.getObject(obj, j), zzngVar, false);
                        break;
                    case 33:
                        z = false;
                        zzlz.zzU(this.zzc[i4], (List) unsafe.getObject(obj, j), zzngVar, false);
                        break;
                    case 34:
                        z = false;
                        zzlz.zzV(this.zzc[i4], (List) unsafe.getObject(obj, j), zzngVar, false);
                        break;
                    case 35:
                        zzlz.zzJ(this.zzc[i4], (List) unsafe.getObject(obj, j), zzngVar, true);
                        break;
                    case 36:
                        zzlz.zzN(this.zzc[i4], (List) unsafe.getObject(obj, j), zzngVar, true);
                        break;
                    case 37:
                        zzlz.zzQ(this.zzc[i4], (List) unsafe.getObject(obj, j), zzngVar, true);
                        break;
                    case 38:
                        zzlz.zzY(this.zzc[i4], (List) unsafe.getObject(obj, j), zzngVar, true);
                        break;
                    case 39:
                        zzlz.zzP(this.zzc[i4], (List) unsafe.getObject(obj, j), zzngVar, true);
                        break;
                    case 40:
                        zzlz.zzM(this.zzc[i4], (List) unsafe.getObject(obj, j), zzngVar, true);
                        break;
                    case 41:
                        zzlz.zzL(this.zzc[i4], (List) unsafe.getObject(obj, j), zzngVar, true);
                        break;
                    case 42:
                        zzlz.zzH(this.zzc[i4], (List) unsafe.getObject(obj, j), zzngVar, true);
                        break;
                    case 43:
                        zzlz.zzX(this.zzc[i4], (List) unsafe.getObject(obj, j), zzngVar, true);
                        break;
                    case 44:
                        zzlz.zzK(this.zzc[i4], (List) unsafe.getObject(obj, j), zzngVar, true);
                        break;
                    case 45:
                        zzlz.zzS(this.zzc[i4], (List) unsafe.getObject(obj, j), zzngVar, true);
                        break;
                    case 46:
                        zzlz.zzT(this.zzc[i4], (List) unsafe.getObject(obj, j), zzngVar, true);
                        break;
                    case 47:
                        zzlz.zzU(this.zzc[i4], (List) unsafe.getObject(obj, j), zzngVar, true);
                        break;
                    case 48:
                        zzlz.zzV(this.zzc[i4], (List) unsafe.getObject(obj, j), zzngVar, true);
                        break;
                    case 49:
                        zzlz.zzO(this.zzc[i4], (List) unsafe.getObject(obj, j), zzngVar, zzE(i4));
                        break;
                    case 50:
                        zzR(zzngVar, i6, unsafe.getObject(obj, j), i4);
                        break;
                    case 51:
                        if (zzX(obj, i6, i4)) {
                            zzngVar.zzf(i6, zzn(obj, j));
                        }
                        break;
                    case 52:
                        if (zzX(obj, i6, i4)) {
                            zzngVar.zzo(i6, zzo(obj, j));
                        }
                        break;
                    case 53:
                        if (zzX(obj, i6, i4)) {
                            zzngVar.zzt(i6, zzC(obj, j));
                        }
                        break;
                    case 54:
                        if (zzX(obj, i6, i4)) {
                            zzngVar.zzJ(i6, zzC(obj, j));
                        }
                        break;
                    case 55:
                        if (zzX(obj, i6, i4)) {
                            zzngVar.zzr(i6, zzr(obj, j));
                        }
                        break;
                    case 56:
                        if (zzX(obj, i6, i4)) {
                            zzngVar.zzm(i6, zzC(obj, j));
                        }
                        break;
                    case 57:
                        if (zzX(obj, i6, i4)) {
                            zzngVar.zzk(i6, zzr(obj, j));
                        }
                        break;
                    case 58:
                        if (zzX(obj, i6, i4)) {
                            zzngVar.zzb(i6, zzY(obj, j));
                        }
                        break;
                    case 59:
                        if (zzX(obj, i6, i4)) {
                            zzZ(i6, unsafe.getObject(obj, j), zzngVar);
                        }
                        break;
                    case 60:
                        if (zzX(obj, i6, i4)) {
                            zzngVar.zzv(i6, unsafe.getObject(obj, j), zzE(i4));
                        }
                        break;
                    case 61:
                        if (zzX(obj, i6, i4)) {
                            zzngVar.zzd(i6, (zzje) unsafe.getObject(obj, j));
                        }
                        break;
                    case 62:
                        if (zzX(obj, i6, i4)) {
                            zzngVar.zzH(i6, zzr(obj, j));
                        }
                        break;
                    case 63:
                        if (zzX(obj, i6, i4)) {
                            zzngVar.zzi(i6, zzr(obj, j));
                        }
                        break;
                    case 64:
                        if (zzX(obj, i6, i4)) {
                            zzngVar.zzw(i6, zzr(obj, j));
                        }
                        break;
                    case 65:
                        if (zzX(obj, i6, i4)) {
                            zzngVar.zzy(i6, zzC(obj, j));
                        }
                        break;
                    case 66:
                        if (zzX(obj, i6, i4)) {
                            zzngVar.zzA(i6, zzr(obj, j));
                        }
                        break;
                    case 67:
                        if (zzX(obj, i6, i4)) {
                            zzngVar.zzC(i6, zzC(obj, j));
                        }
                        break;
                    case 68:
                        if (zzX(obj, i6, i4)) {
                            zzngVar.zzq(i6, unsafe.getObject(obj, j), zzE(i4));
                        }
                        break;
                }
                i4 += 3;
                i2 = 1048575;
            }
            zzmo zzmoVar = this.zzn;
            zzmoVar.zzi(zzmoVar.zzd(obj), zzngVar);
            return;
        }
        this.zzo.zza(obj);
        throw null;
    }

    private final void zzR(zzng zzngVar, int i, Object obj, int i2) {
        if (obj == null) {
            return;
        }
        zzlf zzlfVar = (zzlf) zzF(i2);
        throw null;
    }

    private final boolean zzS(Object obj, Object obj2, int i) {
        return zzT(obj, i) == zzT(obj2, i);
    }

    private final boolean zzT(Object obj, int i) {
        int zzy = zzy(i);
        long j = zzy & 1048575;
        if (j != 1048575) {
            return (zzmy.zzc(obj, j) & (1 << (zzy >>> 20))) != 0;
        }
        int zzB = zzB(i);
        long j2 = zzB & 1048575;
        switch (zzA(zzB)) {
            case 0:
                return Double.doubleToRawLongBits(zzmy.zza(obj, j2)) != 0;
            case 1:
                return Float.floatToRawIntBits(zzmy.zzb(obj, j2)) != 0;
            case 2:
                return zzmy.zzd(obj, j2) != 0;
            case 3:
                return zzmy.zzd(obj, j2) != 0;
            case 4:
                return zzmy.zzc(obj, j2) != 0;
            case 5:
                return zzmy.zzd(obj, j2) != 0;
            case 6:
                return zzmy.zzc(obj, j2) != 0;
            case 7:
                return zzmy.zzw(obj, j2);
            case 8:
                Object zzf = zzmy.zzf(obj, j2);
                if (zzf instanceof String) {
                    return !((String) zzf).isEmpty();
                } else if (zzf instanceof zzje) {
                    return !zzje.zzb.equals(zzf);
                } else {
                    throw new IllegalArgumentException();
                }
            case 9:
                return zzmy.zzf(obj, j2) != null;
            case 10:
                return !zzje.zzb.equals(zzmy.zzf(obj, j2));
            case 11:
                return zzmy.zzc(obj, j2) != 0;
            case 12:
                return zzmy.zzc(obj, j2) != 0;
            case 13:
                return zzmy.zzc(obj, j2) != 0;
            case 14:
                return zzmy.zzd(obj, j2) != 0;
            case 15:
                return zzmy.zzc(obj, j2) != 0;
            case 16:
                return zzmy.zzd(obj, j2) != 0;
            case 17:
                return zzmy.zzf(obj, j2) != null;
            default:
                throw new IllegalArgumentException();
        }
    }

    private final boolean zzU(Object obj, int i, int i2, int i3, int i4) {
        if (i2 == 1048575) {
            return zzT(obj, i);
        }
        return (i3 & i4) != 0;
    }

    private static boolean zzV(Object obj, int i, zzlx zzlxVar) {
        return zzlxVar.zzk(zzmy.zzf(obj, i & 1048575));
    }

    private static boolean zzW(Object obj) {
        if (obj == null) {
            return false;
        }
        if (obj instanceof zzkf) {
            return ((zzkf) obj).zzbO();
        }
        return true;
    }

    private final boolean zzX(Object obj, int i, int i2) {
        return zzmy.zzc(obj, (long) (zzy(i2) & 1048575)) == i;
    }

    private static boolean zzY(Object obj, long j) {
        return ((Boolean) zzmy.zzf(obj, j)).booleanValue();
    }

    private static final void zzZ(int i, Object obj, zzng zzngVar) {
        if (obj instanceof String) {
            zzngVar.zzF(i, (String) obj);
        } else {
            zzngVar.zzd(i, (zzje) obj);
        }
    }

    public static zzmp zzd(Object obj) {
        zzkf zzkfVar = (zzkf) obj;
        zzmp zzmpVar = zzkfVar.zzc;
        if (zzmpVar == zzmp.zzc()) {
            zzmp zzf = zzmp.zzf();
            zzkfVar.zzc = zzf;
            return zzf;
        }
        return zzmpVar;
    }

    public static zzlp zzl(Class cls, zzlj zzljVar, zzlr zzlrVar, zzla zzlaVar, zzmo zzmoVar, zzjs zzjsVar, zzlh zzlhVar) {
        if (zzljVar instanceof zzlw) {
            return zzm((zzlw) zzljVar, zzlrVar, zzlaVar, zzmoVar, zzjsVar, zzlhVar);
        }
        zzml zzmlVar = (zzml) zzljVar;
        throw null;
    }

    /* JADX WARN: Removed duplicated region for block: B:123:0x025e  */
    /* JADX WARN: Removed duplicated region for block: B:124:0x0261  */
    /* JADX WARN: Removed duplicated region for block: B:127:0x0279  */
    /* JADX WARN: Removed duplicated region for block: B:128:0x027c  */
    /* JADX WARN: Removed duplicated region for block: B:162:0x032c  */
    /* JADX WARN: Removed duplicated region for block: B:180:0x0385  */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public static zzlp zzm(zzlw zzlwVar, zzlr zzlrVar, zzla zzlaVar, zzmo zzmoVar, zzjs zzjsVar, zzlh zzlhVar) {
        int i;
        int charAt;
        int charAt2;
        int charAt3;
        int[] iArr;
        int i2;
        int i3;
        int i4;
        int i5;
        int i6;
        char charAt4;
        int i7;
        char charAt5;
        int i8;
        char charAt6;
        int i9;
        char charAt7;
        int i10;
        char charAt8;
        int i11;
        char charAt9;
        int i12;
        char charAt10;
        int i13;
        char charAt11;
        int i14;
        int i15;
        int i16;
        int[] iArr2;
        int i17;
        int i18;
        int i19;
        int objectFieldOffset;
        Object[] objArr;
        String str;
        Class<?> cls;
        int i20;
        int i21;
        int i22;
        Field zzI;
        char charAt12;
        int i23;
        int i24;
        int i25;
        Object obj;
        Field zzI2;
        Object obj2;
        Field zzI3;
        int i26;
        char charAt13;
        int i27;
        char charAt14;
        int i28;
        char charAt15;
        int i29;
        char charAt16;
        boolean z = zzlwVar.zzc() == 2;
        String zzd = zzlwVar.zzd();
        int length = zzd.length();
        char c2 = 55296;
        if (zzd.charAt(0) >= 55296) {
            int i30 = 1;
            while (true) {
                i = i30 + 1;
                if (zzd.charAt(i30) < 55296) {
                    break;
                }
                i30 = i;
            }
        } else {
            i = 1;
        }
        int i31 = i + 1;
        int charAt17 = zzd.charAt(i);
        if (charAt17 >= 55296) {
            int i32 = charAt17 & 8191;
            int i33 = 13;
            while (true) {
                i29 = i31 + 1;
                charAt16 = zzd.charAt(i31);
                if (charAt16 < 55296) {
                    break;
                }
                i32 |= (charAt16 & 8191) << i33;
                i33 += 13;
                i31 = i29;
            }
            charAt17 = i32 | (charAt16 << i33);
            i31 = i29;
        }
        if (charAt17 == 0) {
            charAt = 0;
            i5 = 0;
            charAt2 = 0;
            i4 = 0;
            charAt3 = 0;
            i2 = 0;
            iArr = zza;
            i3 = 0;
        } else {
            int i34 = i31 + 1;
            int charAt18 = zzd.charAt(i31);
            if (charAt18 >= 55296) {
                int i35 = charAt18 & 8191;
                int i36 = 13;
                while (true) {
                    i13 = i34 + 1;
                    charAt11 = zzd.charAt(i34);
                    if (charAt11 < 55296) {
                        break;
                    }
                    i35 |= (charAt11 & 8191) << i36;
                    i36 += 13;
                    i34 = i13;
                }
                charAt18 = i35 | (charAt11 << i36);
                i34 = i13;
            }
            int i37 = i34 + 1;
            int charAt19 = zzd.charAt(i34);
            if (charAt19 >= 55296) {
                int i38 = charAt19 & 8191;
                int i39 = 13;
                while (true) {
                    i12 = i37 + 1;
                    charAt10 = zzd.charAt(i37);
                    if (charAt10 < 55296) {
                        break;
                    }
                    i38 |= (charAt10 & 8191) << i39;
                    i39 += 13;
                    i37 = i12;
                }
                charAt19 = i38 | (charAt10 << i39);
                i37 = i12;
            }
            int i40 = i37 + 1;
            charAt = zzd.charAt(i37);
            if (charAt >= 55296) {
                int i41 = charAt & 8191;
                int i42 = 13;
                while (true) {
                    i11 = i40 + 1;
                    charAt9 = zzd.charAt(i40);
                    if (charAt9 < 55296) {
                        break;
                    }
                    i41 |= (charAt9 & 8191) << i42;
                    i42 += 13;
                    i40 = i11;
                }
                charAt = i41 | (charAt9 << i42);
                i40 = i11;
            }
            int i43 = i40 + 1;
            int charAt20 = zzd.charAt(i40);
            if (charAt20 >= 55296) {
                int i44 = charAt20 & 8191;
                int i45 = 13;
                while (true) {
                    i10 = i43 + 1;
                    charAt8 = zzd.charAt(i43);
                    if (charAt8 < 55296) {
                        break;
                    }
                    i44 |= (charAt8 & 8191) << i45;
                    i45 += 13;
                    i43 = i10;
                }
                charAt20 = i44 | (charAt8 << i45);
                i43 = i10;
            }
            int i46 = i43 + 1;
            charAt2 = zzd.charAt(i43);
            if (charAt2 >= 55296) {
                int i47 = charAt2 & 8191;
                int i48 = 13;
                while (true) {
                    i9 = i46 + 1;
                    charAt7 = zzd.charAt(i46);
                    if (charAt7 < 55296) {
                        break;
                    }
                    i47 |= (charAt7 & 8191) << i48;
                    i48 += 13;
                    i46 = i9;
                }
                charAt2 = i47 | (charAt7 << i48);
                i46 = i9;
            }
            int i49 = i46 + 1;
            int charAt21 = zzd.charAt(i46);
            if (charAt21 >= 55296) {
                int i50 = charAt21 & 8191;
                int i51 = 13;
                while (true) {
                    i8 = i49 + 1;
                    charAt6 = zzd.charAt(i49);
                    if (charAt6 < 55296) {
                        break;
                    }
                    i50 |= (charAt6 & 8191) << i51;
                    i51 += 13;
                    i49 = i8;
                }
                charAt21 = i50 | (charAt6 << i51);
                i49 = i8;
            }
            int i52 = i49 + 1;
            int charAt22 = zzd.charAt(i49);
            if (charAt22 >= 55296) {
                int i53 = charAt22 & 8191;
                int i54 = 13;
                while (true) {
                    i7 = i52 + 1;
                    charAt5 = zzd.charAt(i52);
                    if (charAt5 < 55296) {
                        break;
                    }
                    i53 |= (charAt5 & 8191) << i54;
                    i54 += 13;
                    i52 = i7;
                }
                charAt22 = i53 | (charAt5 << i54);
                i52 = i7;
            }
            int i55 = i52 + 1;
            charAt3 = zzd.charAt(i52);
            if (charAt3 >= 55296) {
                int i56 = charAt3 & 8191;
                int i57 = 13;
                while (true) {
                    i6 = i55 + 1;
                    charAt4 = zzd.charAt(i55);
                    if (charAt4 < 55296) {
                        break;
                    }
                    i56 |= (charAt4 & 8191) << i57;
                    i57 += 13;
                    i55 = i6;
                }
                charAt3 = i56 | (charAt4 << i57);
                i55 = i6;
            }
            iArr = new int[charAt3 + charAt21 + charAt22];
            i2 = charAt18 + charAt18 + charAt19;
            i3 = charAt18;
            i31 = i55;
            int i58 = charAt21;
            i4 = charAt20;
            i5 = i58;
        }
        Unsafe unsafe = zzb;
        Object[] zze = zzlwVar.zze();
        Class<?> cls2 = zzlwVar.zza().getClass();
        int[] iArr3 = new int[charAt2 * 3];
        Object[] objArr2 = new Object[charAt2 + charAt2];
        int i59 = charAt3 + i5;
        int i60 = charAt3;
        int i61 = i59;
        int i62 = 0;
        int i63 = 0;
        while (i31 < length) {
            int i64 = i31 + 1;
            int charAt23 = zzd.charAt(i31);
            if (charAt23 >= c2) {
                int i65 = charAt23 & 8191;
                int i66 = i64;
                int i67 = 13;
                while (true) {
                    i28 = i66 + 1;
                    charAt15 = zzd.charAt(i66);
                    if (charAt15 < c2) {
                        break;
                    }
                    i65 |= (charAt15 & 8191) << i67;
                    i67 += 13;
                    i66 = i28;
                }
                charAt23 = i65 | (charAt15 << i67);
                i14 = i28;
            } else {
                i14 = i64;
            }
            int i68 = i14 + 1;
            int charAt24 = zzd.charAt(i14);
            if (charAt24 >= c2) {
                int i69 = charAt24 & 8191;
                int i70 = i68;
                int i71 = 13;
                while (true) {
                    i27 = i70 + 1;
                    charAt14 = zzd.charAt(i70);
                    i15 = length;
                    if (charAt14 < 55296) {
                        break;
                    }
                    i69 |= (charAt14 & 8191) << i71;
                    i71 += 13;
                    i70 = i27;
                    length = i15;
                }
                charAt24 = i69 | (charAt14 << i71);
                i16 = i27;
            } else {
                i15 = length;
                i16 = i68;
            }
            int i72 = charAt24 & 255;
            int i73 = charAt3;
            if ((charAt24 & 1024) != 0) {
                iArr[i63] = i62;
                i63++;
            }
            if (i72 >= 51) {
                int i74 = i16 + 1;
                int charAt25 = zzd.charAt(i16);
                if (charAt25 >= 55296) {
                    int i75 = charAt25 & 8191;
                    int i76 = i74;
                    int i77 = 13;
                    while (true) {
                        i26 = i76 + 1;
                        charAt13 = zzd.charAt(i76);
                        i18 = i4;
                        if (charAt13 < 55296) {
                            break;
                        }
                        i75 |= (charAt13 & 8191) << i77;
                        i77 += 13;
                        i76 = i26;
                        i4 = i18;
                    }
                    charAt25 = i75 | (charAt13 << i77);
                    i24 = i26;
                } else {
                    i18 = i4;
                    i24 = i74;
                }
                int i78 = i72 - 51;
                i21 = i24;
                if (i78 == 9 || i78 == 17) {
                    int i79 = i62 / 3;
                    i25 = i2 + 1;
                    objArr2[i79 + i79 + 1] = zze[i2];
                } else {
                    if (i78 == 12 && !z) {
                        int i80 = i62 / 3;
                        i25 = i2 + 1;
                        objArr2[i80 + i80 + 1] = zze[i2];
                    }
                    int i81 = charAt25 + charAt25;
                    obj = zze[i81];
                    if (!(obj instanceof Field)) {
                        zzI2 = (Field) obj;
                    } else {
                        zzI2 = zzI(cls2, (String) obj);
                        zze[i81] = zzI2;
                    }
                    iArr2 = iArr3;
                    i17 = charAt;
                    int objectFieldOffset2 = (int) unsafe.objectFieldOffset(zzI2);
                    int i82 = i81 + 1;
                    obj2 = zze[i82];
                    if (!(obj2 instanceof Field)) {
                        zzI3 = (Field) obj2;
                    } else {
                        zzI3 = zzI(cls2, (String) obj2);
                        zze[i82] = zzI3;
                    }
                    cls = cls2;
                    objArr = objArr2;
                    str = zzd;
                    i20 = (int) unsafe.objectFieldOffset(zzI3);
                    objectFieldOffset = objectFieldOffset2;
                    i22 = 0;
                }
                i2 = i25;
                int i812 = charAt25 + charAt25;
                obj = zze[i812];
                if (!(obj instanceof Field)) {
                }
                iArr2 = iArr3;
                i17 = charAt;
                int objectFieldOffset22 = (int) unsafe.objectFieldOffset(zzI2);
                int i822 = i812 + 1;
                obj2 = zze[i822];
                if (!(obj2 instanceof Field)) {
                }
                cls = cls2;
                objArr = objArr2;
                str = zzd;
                i20 = (int) unsafe.objectFieldOffset(zzI3);
                objectFieldOffset = objectFieldOffset22;
                i22 = 0;
            } else {
                iArr2 = iArr3;
                i17 = charAt;
                i18 = i4;
                int i83 = i2 + 1;
                Field zzI4 = zzI(cls2, (String) zze[i2]);
                if (i72 == 9 || i72 == 17) {
                    int i84 = i62 / 3;
                    objArr2[i84 + i84 + 1] = zzI4.getType();
                } else {
                    if (i72 == 27 || i72 == 49) {
                        int i85 = i62 / 3;
                        i23 = i83 + 1;
                        objArr2[i85 + i85 + 1] = zze[i83];
                    } else if (i72 == 12 || i72 == 30 || i72 == 44) {
                        if (!z) {
                            int i86 = i62 / 3;
                            i23 = i83 + 1;
                            objArr2[i86 + i86 + 1] = zze[i83];
                        }
                    } else if (i72 == 50) {
                        int i87 = i60 + 1;
                        iArr[i60] = i62;
                        int i88 = i62 / 3;
                        int i89 = i88 + i88;
                        int i90 = i83 + 1;
                        objArr2[i89] = zze[i83];
                        if ((charAt24 & 2048) != 0) {
                            i83 = i90 + 1;
                            objArr2[i89 + 1] = zze[i90];
                            i60 = i87;
                        } else {
                            i60 = i87;
                            i19 = i90;
                            objectFieldOffset = (int) unsafe.objectFieldOffset(zzI4);
                            objArr = objArr2;
                            if ((charAt24 & 4096) == 4096 || i72 > 17) {
                                str = zzd;
                                cls = cls2;
                                i20 = 1048575;
                                i21 = i16;
                                i22 = 0;
                            } else {
                                int i91 = i16 + 1;
                                int charAt26 = zzd.charAt(i16);
                                if (charAt26 >= 55296) {
                                    int i92 = charAt26 & 8191;
                                    int i93 = 13;
                                    while (true) {
                                        i21 = i91 + 1;
                                        charAt12 = zzd.charAt(i91);
                                        if (charAt12 < 55296) {
                                            break;
                                        }
                                        i92 |= (charAt12 & 8191) << i93;
                                        i93 += 13;
                                        i91 = i21;
                                    }
                                    charAt26 = i92 | (charAt12 << i93);
                                } else {
                                    i21 = i91;
                                }
                                int i94 = (charAt26 / 32) + i3 + i3;
                                Object obj3 = zze[i94];
                                if (obj3 instanceof Field) {
                                    zzI = (Field) obj3;
                                } else {
                                    zzI = zzI(cls2, (String) obj3);
                                    zze[i94] = zzI;
                                }
                                str = zzd;
                                cls = cls2;
                                i20 = (int) unsafe.objectFieldOffset(zzI);
                                i22 = charAt26 % 32;
                            }
                            if (i72 >= 18 && i72 <= 49) {
                                iArr[i61] = objectFieldOffset;
                                i61++;
                            }
                            i2 = i19;
                        }
                    }
                    i19 = i23;
                    objectFieldOffset = (int) unsafe.objectFieldOffset(zzI4);
                    objArr = objArr2;
                    if ((charAt24 & 4096) == 4096) {
                    }
                    str = zzd;
                    cls = cls2;
                    i20 = 1048575;
                    i21 = i16;
                    i22 = 0;
                    if (i72 >= 18) {
                        iArr[i61] = objectFieldOffset;
                        i61++;
                    }
                    i2 = i19;
                }
                i19 = i83;
                objectFieldOffset = (int) unsafe.objectFieldOffset(zzI4);
                objArr = objArr2;
                if ((charAt24 & 4096) == 4096) {
                }
                str = zzd;
                cls = cls2;
                i20 = 1048575;
                i21 = i16;
                i22 = 0;
                if (i72 >= 18) {
                }
                i2 = i19;
            }
            int i95 = i62 + 1;
            iArr2[i62] = charAt23;
            int i96 = i95 + 1;
            iArr2[i95] = ((charAt24 & 256) != 0 ? 268435456 : 0) | ((charAt24 & 512) != 0 ? 536870912 : 0) | (i72 << 20) | objectFieldOffset;
            i62 = i96 + 1;
            iArr2[i96] = i20 | (i22 << 20);
            zzd = str;
            charAt = i17;
            charAt3 = i73;
            cls2 = cls;
            i31 = i21;
            length = i15;
            objArr2 = objArr;
            iArr3 = iArr2;
            i4 = i18;
            c2 = 55296;
        }
        return new zzlp(iArr3, objArr2, charAt, i4, zzlwVar.zza(), z, false, iArr, charAt3, i59, zzlrVar, zzlaVar, zzmoVar, zzjsVar, zzlhVar, null);
    }

    private static double zzn(Object obj, long j) {
        return ((Double) zzmy.zzf(obj, j)).doubleValue();
    }

    private static float zzo(Object obj, long j) {
        return ((Float) zzmy.zzf(obj, j)).floatValue();
    }

    private final int zzp(Object obj) {
        int i;
        int zzA;
        int zzA2;
        int zzA3;
        int zzB;
        int zzA4;
        int zzv;
        int zzA5;
        int zzA6;
        int zzd;
        int zzA7;
        int zzo;
        int zzA8;
        int zzB2;
        int zzi;
        int zzz;
        int zzA9;
        int i2;
        int zzA10;
        int zzd2;
        int zzA11;
        Unsafe unsafe = zzb;
        int i3 = 1048575;
        int i4 = 0;
        int i5 = 0;
        int i6 = 0;
        int i7 = 1048575;
        while (i6 < this.zzc.length) {
            int zzB3 = zzB(i6);
            int[] iArr = this.zzc;
            int i8 = iArr[i6];
            int zzA12 = zzA(zzB3);
            if (zzA12 <= 17) {
                int i9 = iArr[i6 + 2];
                int i10 = i9 & i3;
                i = 1 << (i9 >>> 20);
                if (i10 != i7) {
                    i4 = unsafe.getInt(obj, i10);
                    i7 = i10;
                }
            } else {
                i = 0;
            }
            long j = i3 & zzB3;
            switch (zzA12) {
                case 0:
                    if ((i4 & i) == 0) {
                        break;
                    } else {
                        zzA = zzjm.zzA(i8 << 3);
                        zzo = zzA + 8;
                        i5 += zzo;
                        break;
                    }
                case 1:
                    if ((i4 & i) == 0) {
                        break;
                    } else {
                        zzA2 = zzjm.zzA(i8 << 3);
                        zzo = zzA2 + 4;
                        i5 += zzo;
                        break;
                    }
                case 2:
                    if ((i4 & i) == 0) {
                        break;
                    } else {
                        long j2 = unsafe.getLong(obj, j);
                        zzA3 = zzjm.zzA(i8 << 3);
                        zzB = zzjm.zzB(j2);
                        zzo = zzA3 + zzB;
                        i5 += zzo;
                        break;
                    }
                case 3:
                    if ((i4 & i) == 0) {
                        break;
                    } else {
                        long j3 = unsafe.getLong(obj, j);
                        zzA3 = zzjm.zzA(i8 << 3);
                        zzB = zzjm.zzB(j3);
                        zzo = zzA3 + zzB;
                        i5 += zzo;
                        break;
                    }
                case 4:
                    if ((i4 & i) == 0) {
                        break;
                    } else {
                        int i11 = unsafe.getInt(obj, j);
                        zzA4 = zzjm.zzA(i8 << 3);
                        zzv = zzjm.zzv(i11);
                        zzo = zzv + zzA4;
                        i5 += zzo;
                        break;
                    }
                case 5:
                    if ((i4 & i) == 0) {
                        break;
                    } else {
                        zzA = zzjm.zzA(i8 << 3);
                        zzo = zzA + 8;
                        i5 += zzo;
                        break;
                    }
                case 6:
                    if ((i4 & i) == 0) {
                        break;
                    } else {
                        zzA2 = zzjm.zzA(i8 << 3);
                        zzo = zzA2 + 4;
                        i5 += zzo;
                        break;
                    }
                case 7:
                    if ((i4 & i) == 0) {
                        break;
                    } else {
                        zzA5 = zzjm.zzA(i8 << 3);
                        zzo = zzA5 + 1;
                        i5 += zzo;
                        break;
                    }
                case 8:
                    if ((i4 & i) == 0) {
                        break;
                    } else {
                        Object object = unsafe.getObject(obj, j);
                        if (object instanceof zzje) {
                            zzA6 = zzjm.zzA(i8 << 3);
                            zzd = ((zzje) object).zzd();
                            zzA7 = zzjm.zzA(zzd);
                            i5 = zzA7 + zzd + zzA6 + i5;
                            break;
                        } else {
                            zzA4 = zzjm.zzA(i8 << 3);
                            zzv = zzjm.zzy((String) object);
                            zzo = zzv + zzA4;
                            i5 += zzo;
                            break;
                        }
                    }
                case 9:
                    if ((i4 & i) == 0) {
                        break;
                    } else {
                        zzo = zzlz.zzo(i8, unsafe.getObject(obj, j), zzE(i6));
                        i5 += zzo;
                        break;
                    }
                case 10:
                    if ((i4 & i) == 0) {
                        break;
                    } else {
                        zzA6 = zzjm.zzA(i8 << 3);
                        zzd = ((zzje) unsafe.getObject(obj, j)).zzd();
                        zzA7 = zzjm.zzA(zzd);
                        i5 = zzA7 + zzd + zzA6 + i5;
                        break;
                    }
                case 11:
                    if ((i4 & i) == 0) {
                        break;
                    } else {
                        int i12 = unsafe.getInt(obj, j);
                        zzA4 = zzjm.zzA(i8 << 3);
                        zzv = zzjm.zzA(i12);
                        zzo = zzv + zzA4;
                        i5 += zzo;
                        break;
                    }
                case 12:
                    if ((i4 & i) == 0) {
                        break;
                    } else {
                        int i13 = unsafe.getInt(obj, j);
                        zzA4 = zzjm.zzA(i8 << 3);
                        zzv = zzjm.zzv(i13);
                        zzo = zzv + zzA4;
                        i5 += zzo;
                        break;
                    }
                case 13:
                    if ((i4 & i) == 0) {
                        break;
                    } else {
                        zzA2 = zzjm.zzA(i8 << 3);
                        zzo = zzA2 + 4;
                        i5 += zzo;
                        break;
                    }
                case 14:
                    if ((i4 & i) == 0) {
                        break;
                    } else {
                        zzA = zzjm.zzA(i8 << 3);
                        zzo = zzA + 8;
                        i5 += zzo;
                        break;
                    }
                case 15:
                    if ((i4 & i) == 0) {
                        break;
                    } else {
                        int i14 = unsafe.getInt(obj, j);
                        zzA4 = zzjm.zzA(i8 << 3);
                        zzv = zzjm.zzA((i14 >> 31) ^ (i14 + i14));
                        zzo = zzv + zzA4;
                        i5 += zzo;
                        break;
                    }
                case 16:
                    if ((i & i4) == 0) {
                        break;
                    } else {
                        long j4 = unsafe.getLong(obj, j);
                        zzA8 = zzjm.zzA(i8 << 3);
                        zzB2 = zzjm.zzB((j4 >> 63) ^ (j4 + j4));
                        zzo = zzB2 + zzA8;
                        i5 += zzo;
                        break;
                    }
                case 17:
                    if ((i4 & i) == 0) {
                        break;
                    } else {
                        zzo = zzjm.zzu(i8, (zzlm) unsafe.getObject(obj, j), zzE(i6));
                        i5 += zzo;
                        break;
                    }
                case 18:
                    zzo = zzlz.zzh(i8, (List) unsafe.getObject(obj, j), false);
                    i5 += zzo;
                    break;
                case 19:
                    zzo = zzlz.zzf(i8, (List) unsafe.getObject(obj, j), false);
                    i5 += zzo;
                    break;
                case 20:
                    zzo = zzlz.zzm(i8, (List) unsafe.getObject(obj, j), false);
                    i5 += zzo;
                    break;
                case 21:
                    zzo = zzlz.zzx(i8, (List) unsafe.getObject(obj, j), false);
                    i5 += zzo;
                    break;
                case 22:
                    zzo = zzlz.zzk(i8, (List) unsafe.getObject(obj, j), false);
                    i5 += zzo;
                    break;
                case 23:
                    zzo = zzlz.zzh(i8, (List) unsafe.getObject(obj, j), false);
                    i5 += zzo;
                    break;
                case 24:
                    zzo = zzlz.zzf(i8, (List) unsafe.getObject(obj, j), false);
                    i5 += zzo;
                    break;
                case 25:
                    zzo = zzlz.zza(i8, (List) unsafe.getObject(obj, j), false);
                    i5 += zzo;
                    break;
                case 26:
                    zzo = zzlz.zzu(i8, (List) unsafe.getObject(obj, j));
                    i5 += zzo;
                    break;
                case 27:
                    zzo = zzlz.zzp(i8, (List) unsafe.getObject(obj, j), zzE(i6));
                    i5 += zzo;
                    break;
                case 28:
                    zzo = zzlz.zzc(i8, (List) unsafe.getObject(obj, j));
                    i5 += zzo;
                    break;
                case 29:
                    zzo = zzlz.zzv(i8, (List) unsafe.getObject(obj, j), false);
                    i5 += zzo;
                    break;
                case 30:
                    zzo = zzlz.zzd(i8, (List) unsafe.getObject(obj, j), false);
                    i5 += zzo;
                    break;
                case 31:
                    zzo = zzlz.zzf(i8, (List) unsafe.getObject(obj, j), false);
                    i5 += zzo;
                    break;
                case 32:
                    zzo = zzlz.zzh(i8, (List) unsafe.getObject(obj, j), false);
                    i5 += zzo;
                    break;
                case 33:
                    zzo = zzlz.zzq(i8, (List) unsafe.getObject(obj, j), false);
                    i5 += zzo;
                    break;
                case 34:
                    zzo = zzlz.zzs(i8, (List) unsafe.getObject(obj, j), false);
                    i5 += zzo;
                    break;
                case 35:
                    zzi = zzlz.zzi((List) unsafe.getObject(obj, j));
                    if (zzi <= 0) {
                        break;
                    } else {
                        zzz = zzjm.zzz(i8);
                        zzA9 = zzjm.zzA(zzi);
                        i2 = zzA9 + zzz + zzi;
                        i5 += i2;
                        break;
                    }
                case 36:
                    zzi = zzlz.zzg((List) unsafe.getObject(obj, j));
                    if (zzi <= 0) {
                        break;
                    } else {
                        zzz = zzjm.zzz(i8);
                        zzA9 = zzjm.zzA(zzi);
                        i2 = zzA9 + zzz + zzi;
                        i5 += i2;
                        break;
                    }
                case 37:
                    zzi = zzlz.zzn((List) unsafe.getObject(obj, j));
                    if (zzi <= 0) {
                        break;
                    } else {
                        zzz = zzjm.zzz(i8);
                        zzA9 = zzjm.zzA(zzi);
                        i2 = zzA9 + zzz + zzi;
                        i5 += i2;
                        break;
                    }
                case 38:
                    zzi = zzlz.zzy((List) unsafe.getObject(obj, j));
                    if (zzi <= 0) {
                        break;
                    } else {
                        zzz = zzjm.zzz(i8);
                        zzA9 = zzjm.zzA(zzi);
                        i2 = zzA9 + zzz + zzi;
                        i5 += i2;
                        break;
                    }
                case 39:
                    zzi = zzlz.zzl((List) unsafe.getObject(obj, j));
                    if (zzi <= 0) {
                        break;
                    } else {
                        zzz = zzjm.zzz(i8);
                        zzA9 = zzjm.zzA(zzi);
                        i2 = zzA9 + zzz + zzi;
                        i5 += i2;
                        break;
                    }
                case 40:
                    zzi = zzlz.zzi((List) unsafe.getObject(obj, j));
                    if (zzi <= 0) {
                        break;
                    } else {
                        zzz = zzjm.zzz(i8);
                        zzA9 = zzjm.zzA(zzi);
                        i2 = zzA9 + zzz + zzi;
                        i5 += i2;
                        break;
                    }
                case 41:
                    zzi = zzlz.zzg((List) unsafe.getObject(obj, j));
                    if (zzi <= 0) {
                        break;
                    } else {
                        zzz = zzjm.zzz(i8);
                        zzA9 = zzjm.zzA(zzi);
                        i2 = zzA9 + zzz + zzi;
                        i5 += i2;
                        break;
                    }
                case 42:
                    zzi = zzlz.zzb((List) unsafe.getObject(obj, j));
                    if (zzi <= 0) {
                        break;
                    } else {
                        zzz = zzjm.zzz(i8);
                        zzA9 = zzjm.zzA(zzi);
                        i2 = zzA9 + zzz + zzi;
                        i5 += i2;
                        break;
                    }
                case 43:
                    zzi = zzlz.zzw((List) unsafe.getObject(obj, j));
                    if (zzi <= 0) {
                        break;
                    } else {
                        zzz = zzjm.zzz(i8);
                        zzA9 = zzjm.zzA(zzi);
                        i2 = zzA9 + zzz + zzi;
                        i5 += i2;
                        break;
                    }
                case 44:
                    zzi = zzlz.zze((List) unsafe.getObject(obj, j));
                    if (zzi <= 0) {
                        break;
                    } else {
                        zzz = zzjm.zzz(i8);
                        zzA9 = zzjm.zzA(zzi);
                        i2 = zzA9 + zzz + zzi;
                        i5 += i2;
                        break;
                    }
                case 45:
                    zzi = zzlz.zzg((List) unsafe.getObject(obj, j));
                    if (zzi <= 0) {
                        break;
                    } else {
                        zzz = zzjm.zzz(i8);
                        zzA9 = zzjm.zzA(zzi);
                        i2 = zzA9 + zzz + zzi;
                        i5 += i2;
                        break;
                    }
                case 46:
                    zzi = zzlz.zzi((List) unsafe.getObject(obj, j));
                    if (zzi <= 0) {
                        break;
                    } else {
                        zzz = zzjm.zzz(i8);
                        zzA9 = zzjm.zzA(zzi);
                        i2 = zzA9 + zzz + zzi;
                        i5 += i2;
                        break;
                    }
                case 47:
                    zzi = zzlz.zzr((List) unsafe.getObject(obj, j));
                    if (zzi <= 0) {
                        break;
                    } else {
                        zzz = zzjm.zzz(i8);
                        zzA9 = zzjm.zzA(zzi);
                        i2 = zzA9 + zzz + zzi;
                        i5 += i2;
                        break;
                    }
                case 48:
                    zzi = zzlz.zzt((List) unsafe.getObject(obj, j));
                    if (zzi <= 0) {
                        break;
                    } else {
                        zzz = zzjm.zzz(i8);
                        zzA9 = zzjm.zzA(zzi);
                        i2 = zzA9 + zzz + zzi;
                        i5 += i2;
                        break;
                    }
                case 49:
                    zzo = zzlz.zzj(i8, (List) unsafe.getObject(obj, j), zzE(i6));
                    i5 += zzo;
                    break;
                case 50:
                    zzlh.zza(i8, unsafe.getObject(obj, j), zzF(i6));
                    break;
                case 51:
                    if (!zzX(obj, i8, i6)) {
                        break;
                    } else {
                        zzA = zzjm.zzA(i8 << 3);
                        zzo = zzA + 8;
                        i5 += zzo;
                        break;
                    }
                case 52:
                    if (!zzX(obj, i8, i6)) {
                        break;
                    } else {
                        zzA2 = zzjm.zzA(i8 << 3);
                        zzo = zzA2 + 4;
                        i5 += zzo;
                        break;
                    }
                case 53:
                    if (!zzX(obj, i8, i6)) {
                        break;
                    } else {
                        long zzC = zzC(obj, j);
                        zzA3 = zzjm.zzA(i8 << 3);
                        zzB = zzjm.zzB(zzC);
                        zzo = zzA3 + zzB;
                        i5 += zzo;
                        break;
                    }
                case 54:
                    if (!zzX(obj, i8, i6)) {
                        break;
                    } else {
                        long zzC2 = zzC(obj, j);
                        zzA3 = zzjm.zzA(i8 << 3);
                        zzB = zzjm.zzB(zzC2);
                        zzo = zzA3 + zzB;
                        i5 += zzo;
                        break;
                    }
                case 55:
                    if (!zzX(obj, i8, i6)) {
                        break;
                    } else {
                        int zzr = zzr(obj, j);
                        zzA4 = zzjm.zzA(i8 << 3);
                        zzv = zzjm.zzv(zzr);
                        zzo = zzv + zzA4;
                        i5 += zzo;
                        break;
                    }
                case 56:
                    if (!zzX(obj, i8, i6)) {
                        break;
                    } else {
                        zzA = zzjm.zzA(i8 << 3);
                        zzo = zzA + 8;
                        i5 += zzo;
                        break;
                    }
                case 57:
                    if (!zzX(obj, i8, i6)) {
                        break;
                    } else {
                        zzA2 = zzjm.zzA(i8 << 3);
                        zzo = zzA2 + 4;
                        i5 += zzo;
                        break;
                    }
                case 58:
                    if (!zzX(obj, i8, i6)) {
                        break;
                    } else {
                        zzA5 = zzjm.zzA(i8 << 3);
                        zzo = zzA5 + 1;
                        i5 += zzo;
                        break;
                    }
                case 59:
                    if (!zzX(obj, i8, i6)) {
                        break;
                    } else {
                        Object object2 = unsafe.getObject(obj, j);
                        if (object2 instanceof zzje) {
                            zzA10 = zzjm.zzA(i8 << 3);
                            zzd2 = ((zzje) object2).zzd();
                            zzA11 = zzjm.zzA(zzd2);
                            i2 = zzA11 + zzd2 + zzA10;
                            i5 += i2;
                            break;
                        } else {
                            zzA4 = zzjm.zzA(i8 << 3);
                            zzv = zzjm.zzy((String) object2);
                            zzo = zzv + zzA4;
                            i5 += zzo;
                            break;
                        }
                    }
                case 60:
                    if (!zzX(obj, i8, i6)) {
                        break;
                    } else {
                        zzo = zzlz.zzo(i8, unsafe.getObject(obj, j), zzE(i6));
                        i5 += zzo;
                        break;
                    }
                case 61:
                    if (!zzX(obj, i8, i6)) {
                        break;
                    } else {
                        zzA10 = zzjm.zzA(i8 << 3);
                        zzd2 = ((zzje) unsafe.getObject(obj, j)).zzd();
                        zzA11 = zzjm.zzA(zzd2);
                        i2 = zzA11 + zzd2 + zzA10;
                        i5 += i2;
                        break;
                    }
                case 62:
                    if (!zzX(obj, i8, i6)) {
                        break;
                    } else {
                        int zzr2 = zzr(obj, j);
                        zzA4 = zzjm.zzA(i8 << 3);
                        zzv = zzjm.zzA(zzr2);
                        zzo = zzv + zzA4;
                        i5 += zzo;
                        break;
                    }
                case 63:
                    if (!zzX(obj, i8, i6)) {
                        break;
                    } else {
                        int zzr3 = zzr(obj, j);
                        zzA4 = zzjm.zzA(i8 << 3);
                        zzv = zzjm.zzv(zzr3);
                        zzo = zzv + zzA4;
                        i5 += zzo;
                        break;
                    }
                case 64:
                    if (!zzX(obj, i8, i6)) {
                        break;
                    } else {
                        zzA2 = zzjm.zzA(i8 << 3);
                        zzo = zzA2 + 4;
                        i5 += zzo;
                        break;
                    }
                case 65:
                    if (!zzX(obj, i8, i6)) {
                        break;
                    } else {
                        zzA = zzjm.zzA(i8 << 3);
                        zzo = zzA + 8;
                        i5 += zzo;
                        break;
                    }
                case 66:
                    if (!zzX(obj, i8, i6)) {
                        break;
                    } else {
                        int zzr4 = zzr(obj, j);
                        zzA4 = zzjm.zzA(i8 << 3);
                        zzv = zzjm.zzA((zzr4 >> 31) ^ (zzr4 + zzr4));
                        zzo = zzv + zzA4;
                        i5 += zzo;
                        break;
                    }
                case 67:
                    if (!zzX(obj, i8, i6)) {
                        break;
                    } else {
                        long zzC3 = zzC(obj, j);
                        zzA8 = zzjm.zzA(i8 << 3);
                        zzB2 = zzjm.zzB((zzC3 >> 63) ^ (zzC3 + zzC3));
                        zzo = zzB2 + zzA8;
                        i5 += zzo;
                        break;
                    }
                case 68:
                    if (!zzX(obj, i8, i6)) {
                        break;
                    } else {
                        zzo = zzjm.zzu(i8, (zzlm) unsafe.getObject(obj, j), zzE(i6));
                        i5 += zzo;
                        break;
                    }
            }
            i6 += 3;
            i3 = 1048575;
        }
        zzmo zzmoVar = this.zzn;
        int zza2 = zzmoVar.zza(zzmoVar.zzd(obj)) + i5;
        if (this.zzh) {
            this.zzo.zza(obj);
            throw null;
        }
        return zza2;
    }

    private final int zzq(Object obj) {
        int zzA;
        int zzA2;
        int zzA3;
        int zzB;
        int zzA4;
        int zzv;
        int zzA5;
        int zzA6;
        int zzd;
        int zzA7;
        int zzo;
        int zzi;
        int zzz;
        int zzA8;
        int i;
        Unsafe unsafe = zzb;
        int i2 = 0;
        for (int i3 = 0; i3 < this.zzc.length; i3 += 3) {
            int zzB2 = zzB(i3);
            int zzA9 = zzA(zzB2);
            int i4 = this.zzc[i3];
            long j = zzB2 & 1048575;
            if (zzA9 >= zzjx.zzJ.zza() && zzA9 <= zzjx.zzW.zza()) {
                int i5 = this.zzc[i3 + 2];
            }
            switch (zzA9) {
                case 0:
                    if (zzT(obj, i3)) {
                        zzA = zzjm.zzA(i4 << 3);
                        zzo = zzA + 8;
                        break;
                    } else {
                        continue;
                    }
                case 1:
                    if (zzT(obj, i3)) {
                        zzA2 = zzjm.zzA(i4 << 3);
                        zzo = zzA2 + 4;
                        break;
                    } else {
                        continue;
                    }
                case 2:
                    if (zzT(obj, i3)) {
                        long zzd2 = zzmy.zzd(obj, j);
                        zzA3 = zzjm.zzA(i4 << 3);
                        zzB = zzjm.zzB(zzd2);
                        zzo = zzB + zzA3;
                        break;
                    } else {
                        continue;
                    }
                case 3:
                    if (zzT(obj, i3)) {
                        long zzd3 = zzmy.zzd(obj, j);
                        zzA3 = zzjm.zzA(i4 << 3);
                        zzB = zzjm.zzB(zzd3);
                        zzo = zzB + zzA3;
                        break;
                    } else {
                        continue;
                    }
                case 4:
                    if (zzT(obj, i3)) {
                        int zzc = zzmy.zzc(obj, j);
                        zzA4 = zzjm.zzA(i4 << 3);
                        zzv = zzjm.zzv(zzc);
                        zzo = zzv + zzA4;
                        break;
                    } else {
                        continue;
                    }
                case 5:
                    if (zzT(obj, i3)) {
                        zzA = zzjm.zzA(i4 << 3);
                        zzo = zzA + 8;
                        break;
                    } else {
                        continue;
                    }
                case 6:
                    if (zzT(obj, i3)) {
                        zzA2 = zzjm.zzA(i4 << 3);
                        zzo = zzA2 + 4;
                        break;
                    } else {
                        continue;
                    }
                case 7:
                    if (zzT(obj, i3)) {
                        zzA5 = zzjm.zzA(i4 << 3);
                        zzo = zzA5 + 1;
                        break;
                    } else {
                        continue;
                    }
                case 8:
                    if (zzT(obj, i3)) {
                        Object zzf = zzmy.zzf(obj, j);
                        if (zzf instanceof zzje) {
                            zzA6 = zzjm.zzA(i4 << 3);
                            zzd = ((zzje) zzf).zzd();
                            zzA7 = zzjm.zzA(zzd);
                            i = zzA7 + zzd + zzA6;
                            i2 += i;
                        } else {
                            zzA4 = zzjm.zzA(i4 << 3);
                            zzv = zzjm.zzy((String) zzf);
                            zzo = zzv + zzA4;
                            break;
                        }
                    } else {
                        continue;
                    }
                case 9:
                    if (zzT(obj, i3)) {
                        zzo = zzlz.zzo(i4, zzmy.zzf(obj, j), zzE(i3));
                        break;
                    } else {
                        continue;
                    }
                case 10:
                    if (zzT(obj, i3)) {
                        zzA6 = zzjm.zzA(i4 << 3);
                        zzd = ((zzje) zzmy.zzf(obj, j)).zzd();
                        zzA7 = zzjm.zzA(zzd);
                        i = zzA7 + zzd + zzA6;
                        i2 += i;
                    } else {
                        continue;
                    }
                case 11:
                    if (zzT(obj, i3)) {
                        int zzc2 = zzmy.zzc(obj, j);
                        zzA4 = zzjm.zzA(i4 << 3);
                        zzv = zzjm.zzA(zzc2);
                        zzo = zzv + zzA4;
                        break;
                    } else {
                        continue;
                    }
                case 12:
                    if (zzT(obj, i3)) {
                        int zzc3 = zzmy.zzc(obj, j);
                        zzA4 = zzjm.zzA(i4 << 3);
                        zzv = zzjm.zzv(zzc3);
                        zzo = zzv + zzA4;
                        break;
                    } else {
                        continue;
                    }
                case 13:
                    if (zzT(obj, i3)) {
                        zzA2 = zzjm.zzA(i4 << 3);
                        zzo = zzA2 + 4;
                        break;
                    } else {
                        continue;
                    }
                case 14:
                    if (zzT(obj, i3)) {
                        zzA = zzjm.zzA(i4 << 3);
                        zzo = zzA + 8;
                        break;
                    } else {
                        continue;
                    }
                case 15:
                    if (zzT(obj, i3)) {
                        int zzc4 = zzmy.zzc(obj, j);
                        zzA4 = zzjm.zzA(i4 << 3);
                        zzv = zzjm.zzA((zzc4 >> 31) ^ (zzc4 + zzc4));
                        zzo = zzv + zzA4;
                        break;
                    } else {
                        continue;
                    }
                case 16:
                    if (zzT(obj, i3)) {
                        long zzd4 = zzmy.zzd(obj, j);
                        zzA4 = zzjm.zzA(i4 << 3);
                        zzv = zzjm.zzB((zzd4 >> 63) ^ (zzd4 + zzd4));
                        zzo = zzv + zzA4;
                        break;
                    } else {
                        continue;
                    }
                case 17:
                    if (zzT(obj, i3)) {
                        zzo = zzjm.zzu(i4, (zzlm) zzmy.zzf(obj, j), zzE(i3));
                        break;
                    } else {
                        continue;
                    }
                case 18:
                    zzo = zzlz.zzh(i4, (List) zzmy.zzf(obj, j), false);
                    break;
                case 19:
                    zzo = zzlz.zzf(i4, (List) zzmy.zzf(obj, j), false);
                    break;
                case 20:
                    zzo = zzlz.zzm(i4, (List) zzmy.zzf(obj, j), false);
                    break;
                case 21:
                    zzo = zzlz.zzx(i4, (List) zzmy.zzf(obj, j), false);
                    break;
                case 22:
                    zzo = zzlz.zzk(i4, (List) zzmy.zzf(obj, j), false);
                    break;
                case 23:
                    zzo = zzlz.zzh(i4, (List) zzmy.zzf(obj, j), false);
                    break;
                case 24:
                    zzo = zzlz.zzf(i4, (List) zzmy.zzf(obj, j), false);
                    break;
                case 25:
                    zzo = zzlz.zza(i4, (List) zzmy.zzf(obj, j), false);
                    break;
                case 26:
                    zzo = zzlz.zzu(i4, (List) zzmy.zzf(obj, j));
                    break;
                case 27:
                    zzo = zzlz.zzp(i4, (List) zzmy.zzf(obj, j), zzE(i3));
                    break;
                case 28:
                    zzo = zzlz.zzc(i4, (List) zzmy.zzf(obj, j));
                    break;
                case 29:
                    zzo = zzlz.zzv(i4, (List) zzmy.zzf(obj, j), false);
                    break;
                case 30:
                    zzo = zzlz.zzd(i4, (List) zzmy.zzf(obj, j), false);
                    break;
                case 31:
                    zzo = zzlz.zzf(i4, (List) zzmy.zzf(obj, j), false);
                    break;
                case 32:
                    zzo = zzlz.zzh(i4, (List) zzmy.zzf(obj, j), false);
                    break;
                case 33:
                    zzo = zzlz.zzq(i4, (List) zzmy.zzf(obj, j), false);
                    break;
                case 34:
                    zzo = zzlz.zzs(i4, (List) zzmy.zzf(obj, j), false);
                    break;
                case 35:
                    zzi = zzlz.zzi((List) unsafe.getObject(obj, j));
                    if (zzi > 0) {
                        zzz = zzjm.zzz(i4);
                        zzA8 = zzjm.zzA(zzi);
                        i = zzA8 + zzz + zzi;
                        i2 += i;
                    } else {
                        continue;
                    }
                case 36:
                    zzi = zzlz.zzg((List) unsafe.getObject(obj, j));
                    if (zzi > 0) {
                        zzz = zzjm.zzz(i4);
                        zzA8 = zzjm.zzA(zzi);
                        i = zzA8 + zzz + zzi;
                        i2 += i;
                    } else {
                        continue;
                    }
                case 37:
                    zzi = zzlz.zzn((List) unsafe.getObject(obj, j));
                    if (zzi > 0) {
                        zzz = zzjm.zzz(i4);
                        zzA8 = zzjm.zzA(zzi);
                        i = zzA8 + zzz + zzi;
                        i2 += i;
                    } else {
                        continue;
                    }
                case 38:
                    zzi = zzlz.zzy((List) unsafe.getObject(obj, j));
                    if (zzi > 0) {
                        zzz = zzjm.zzz(i4);
                        zzA8 = zzjm.zzA(zzi);
                        i = zzA8 + zzz + zzi;
                        i2 += i;
                    } else {
                        continue;
                    }
                case 39:
                    zzi = zzlz.zzl((List) unsafe.getObject(obj, j));
                    if (zzi > 0) {
                        zzz = zzjm.zzz(i4);
                        zzA8 = zzjm.zzA(zzi);
                        i = zzA8 + zzz + zzi;
                        i2 += i;
                    } else {
                        continue;
                    }
                case 40:
                    zzi = zzlz.zzi((List) unsafe.getObject(obj, j));
                    if (zzi > 0) {
                        zzz = zzjm.zzz(i4);
                        zzA8 = zzjm.zzA(zzi);
                        i = zzA8 + zzz + zzi;
                        i2 += i;
                    } else {
                        continue;
                    }
                case 41:
                    zzi = zzlz.zzg((List) unsafe.getObject(obj, j));
                    if (zzi > 0) {
                        zzz = zzjm.zzz(i4);
                        zzA8 = zzjm.zzA(zzi);
                        i = zzA8 + zzz + zzi;
                        i2 += i;
                    } else {
                        continue;
                    }
                case 42:
                    zzi = zzlz.zzb((List) unsafe.getObject(obj, j));
                    if (zzi > 0) {
                        zzz = zzjm.zzz(i4);
                        zzA8 = zzjm.zzA(zzi);
                        i = zzA8 + zzz + zzi;
                        i2 += i;
                    } else {
                        continue;
                    }
                case 43:
                    zzi = zzlz.zzw((List) unsafe.getObject(obj, j));
                    if (zzi > 0) {
                        zzz = zzjm.zzz(i4);
                        zzA8 = zzjm.zzA(zzi);
                        i = zzA8 + zzz + zzi;
                        i2 += i;
                    } else {
                        continue;
                    }
                case 44:
                    zzi = zzlz.zze((List) unsafe.getObject(obj, j));
                    if (zzi > 0) {
                        zzz = zzjm.zzz(i4);
                        zzA8 = zzjm.zzA(zzi);
                        i = zzA8 + zzz + zzi;
                        i2 += i;
                    } else {
                        continue;
                    }
                case 45:
                    zzi = zzlz.zzg((List) unsafe.getObject(obj, j));
                    if (zzi > 0) {
                        zzz = zzjm.zzz(i4);
                        zzA8 = zzjm.zzA(zzi);
                        i = zzA8 + zzz + zzi;
                        i2 += i;
                    } else {
                        continue;
                    }
                case 46:
                    zzi = zzlz.zzi((List) unsafe.getObject(obj, j));
                    if (zzi > 0) {
                        zzz = zzjm.zzz(i4);
                        zzA8 = zzjm.zzA(zzi);
                        i = zzA8 + zzz + zzi;
                        i2 += i;
                    } else {
                        continue;
                    }
                case 47:
                    zzi = zzlz.zzr((List) unsafe.getObject(obj, j));
                    if (zzi > 0) {
                        zzz = zzjm.zzz(i4);
                        zzA8 = zzjm.zzA(zzi);
                        i = zzA8 + zzz + zzi;
                        i2 += i;
                    } else {
                        continue;
                    }
                case 48:
                    zzi = zzlz.zzt((List) unsafe.getObject(obj, j));
                    if (zzi > 0) {
                        zzz = zzjm.zzz(i4);
                        zzA8 = zzjm.zzA(zzi);
                        i = zzA8 + zzz + zzi;
                        i2 += i;
                    } else {
                        continue;
                    }
                case 49:
                    zzo = zzlz.zzj(i4, (List) zzmy.zzf(obj, j), zzE(i3));
                    break;
                case 50:
                    zzlh.zza(i4, zzmy.zzf(obj, j), zzF(i3));
                    continue;
                case 51:
                    if (zzX(obj, i4, i3)) {
                        zzA = zzjm.zzA(i4 << 3);
                        zzo = zzA + 8;
                        break;
                    } else {
                        continue;
                    }
                case 52:
                    if (zzX(obj, i4, i3)) {
                        zzA2 = zzjm.zzA(i4 << 3);
                        zzo = zzA2 + 4;
                        break;
                    } else {
                        continue;
                    }
                case 53:
                    if (zzX(obj, i4, i3)) {
                        long zzC = zzC(obj, j);
                        zzA3 = zzjm.zzA(i4 << 3);
                        zzB = zzjm.zzB(zzC);
                        zzo = zzB + zzA3;
                        break;
                    } else {
                        continue;
                    }
                case 54:
                    if (zzX(obj, i4, i3)) {
                        long zzC2 = zzC(obj, j);
                        zzA3 = zzjm.zzA(i4 << 3);
                        zzB = zzjm.zzB(zzC2);
                        zzo = zzB + zzA3;
                        break;
                    } else {
                        continue;
                    }
                case 55:
                    if (zzX(obj, i4, i3)) {
                        int zzr = zzr(obj, j);
                        zzA4 = zzjm.zzA(i4 << 3);
                        zzv = zzjm.zzv(zzr);
                        zzo = zzv + zzA4;
                        break;
                    } else {
                        continue;
                    }
                case 56:
                    if (zzX(obj, i4, i3)) {
                        zzA = zzjm.zzA(i4 << 3);
                        zzo = zzA + 8;
                        break;
                    } else {
                        continue;
                    }
                case 57:
                    if (zzX(obj, i4, i3)) {
                        zzA2 = zzjm.zzA(i4 << 3);
                        zzo = zzA2 + 4;
                        break;
                    } else {
                        continue;
                    }
                case 58:
                    if (zzX(obj, i4, i3)) {
                        zzA5 = zzjm.zzA(i4 << 3);
                        zzo = zzA5 + 1;
                        break;
                    } else {
                        continue;
                    }
                case 59:
                    if (zzX(obj, i4, i3)) {
                        Object zzf2 = zzmy.zzf(obj, j);
                        if (zzf2 instanceof zzje) {
                            zzA6 = zzjm.zzA(i4 << 3);
                            zzd = ((zzje) zzf2).zzd();
                            zzA7 = zzjm.zzA(zzd);
                            i = zzA7 + zzd + zzA6;
                            i2 += i;
                        } else {
                            zzA4 = zzjm.zzA(i4 << 3);
                            zzv = zzjm.zzy((String) zzf2);
                            zzo = zzv + zzA4;
                            break;
                        }
                    } else {
                        continue;
                    }
                case 60:
                    if (zzX(obj, i4, i3)) {
                        zzo = zzlz.zzo(i4, zzmy.zzf(obj, j), zzE(i3));
                        break;
                    } else {
                        continue;
                    }
                case 61:
                    if (zzX(obj, i4, i3)) {
                        zzA6 = zzjm.zzA(i4 << 3);
                        zzd = ((zzje) zzmy.zzf(obj, j)).zzd();
                        zzA7 = zzjm.zzA(zzd);
                        i = zzA7 + zzd + zzA6;
                        i2 += i;
                    } else {
                        continue;
                    }
                case 62:
                    if (zzX(obj, i4, i3)) {
                        int zzr2 = zzr(obj, j);
                        zzA4 = zzjm.zzA(i4 << 3);
                        zzv = zzjm.zzA(zzr2);
                        zzo = zzv + zzA4;
                        break;
                    } else {
                        continue;
                    }
                case 63:
                    if (zzX(obj, i4, i3)) {
                        int zzr3 = zzr(obj, j);
                        zzA4 = zzjm.zzA(i4 << 3);
                        zzv = zzjm.zzv(zzr3);
                        zzo = zzv + zzA4;
                        break;
                    } else {
                        continue;
                    }
                case 64:
                    if (zzX(obj, i4, i3)) {
                        zzA2 = zzjm.zzA(i4 << 3);
                        zzo = zzA2 + 4;
                        break;
                    } else {
                        continue;
                    }
                case 65:
                    if (zzX(obj, i4, i3)) {
                        zzA = zzjm.zzA(i4 << 3);
                        zzo = zzA + 8;
                        break;
                    } else {
                        continue;
                    }
                case 66:
                    if (zzX(obj, i4, i3)) {
                        int zzr4 = zzr(obj, j);
                        zzA4 = zzjm.zzA(i4 << 3);
                        zzv = zzjm.zzA((zzr4 >> 31) ^ (zzr4 + zzr4));
                        zzo = zzv + zzA4;
                        break;
                    } else {
                        continue;
                    }
                case 67:
                    if (zzX(obj, i4, i3)) {
                        long zzC3 = zzC(obj, j);
                        zzA4 = zzjm.zzA(i4 << 3);
                        zzv = zzjm.zzB((zzC3 >> 63) ^ (zzC3 + zzC3));
                        zzo = zzv + zzA4;
                        break;
                    } else {
                        continue;
                    }
                case 68:
                    if (zzX(obj, i4, i3)) {
                        zzo = zzjm.zzu(i4, (zzlm) zzmy.zzf(obj, j), zzE(i3));
                        break;
                    } else {
                        continue;
                    }
                default:
            }
            i2 += zzo;
        }
        zzmo zzmoVar = this.zzn;
        return zzmoVar.zza(zzmoVar.zzd(obj)) + i2;
    }

    private static int zzr(Object obj, long j) {
        return ((Integer) zzmy.zzf(obj, j)).intValue();
    }

    private final int zzs(Object obj, byte[] bArr, int i, int i2, int i3, long j, zzir zzirVar) {
        Unsafe unsafe = zzb;
        Object zzF = zzF(i3);
        Object object = unsafe.getObject(obj, j);
        if (!((zzlg) object).zze()) {
            zzlg zzb2 = zzlg.zza().zzb();
            zzlh.zzb(zzb2, object);
            unsafe.putObject(obj, j, zzb2);
        }
        zzlf zzlfVar = (zzlf) zzF;
        throw null;
    }

    private final int zzt(Object obj, byte[] bArr, int i, int i2, int i3, int i4, int i5, int i6, int i7, long j, int i8, zzir zzirVar) {
        Unsafe unsafe = zzb;
        long j2 = this.zzc[i8 + 2] & 1048575;
        switch (i7) {
            case 51:
                if (i5 == 1) {
                    unsafe.putObject(obj, j, Double.valueOf(Double.longBitsToDouble(zzis.zzp(bArr, i))));
                    unsafe.putInt(obj, j2, i4);
                    return i + 8;
                }
                break;
            case 52:
                if (i5 == 5) {
                    unsafe.putObject(obj, j, Float.valueOf(Float.intBitsToFloat(zzis.zzb(bArr, i))));
                    unsafe.putInt(obj, j2, i4);
                    return i + 4;
                }
                break;
            case 53:
            case 54:
                if (i5 == 0) {
                    int zzm = zzis.zzm(bArr, i, zzirVar);
                    unsafe.putObject(obj, j, Long.valueOf(zzirVar.zzb));
                    unsafe.putInt(obj, j2, i4);
                    return zzm;
                }
                break;
            case 55:
            case 62:
                if (i5 == 0) {
                    int zzj = zzis.zzj(bArr, i, zzirVar);
                    unsafe.putObject(obj, j, Integer.valueOf(zzirVar.zza));
                    unsafe.putInt(obj, j2, i4);
                    return zzj;
                }
                break;
            case 56:
            case 65:
                if (i5 == 1) {
                    unsafe.putObject(obj, j, Long.valueOf(zzis.zzp(bArr, i)));
                    unsafe.putInt(obj, j2, i4);
                    return i + 8;
                }
                break;
            case 57:
            case 64:
                if (i5 == 5) {
                    unsafe.putObject(obj, j, Integer.valueOf(zzis.zzb(bArr, i)));
                    unsafe.putInt(obj, j2, i4);
                    return i + 4;
                }
                break;
            case 58:
                if (i5 == 0) {
                    int zzm2 = zzis.zzm(bArr, i, zzirVar);
                    unsafe.putObject(obj, j, Boolean.valueOf(zzirVar.zzb != 0));
                    unsafe.putInt(obj, j2, i4);
                    return zzm2;
                }
                break;
            case 59:
                if (i5 == 2) {
                    int zzj2 = zzis.zzj(bArr, i, zzirVar);
                    int i9 = zzirVar.zza;
                    if (i9 == 0) {
                        unsafe.putObject(obj, j, "");
                    } else if ((i6 & 536870912) != 0 && !zznd.zzf(bArr, zzj2, zzj2 + i9)) {
                        throw zzkp.zzc();
                    } else {
                        unsafe.putObject(obj, j, new String(bArr, zzj2, i9, zzkn.zzb));
                        zzj2 += i9;
                    }
                    unsafe.putInt(obj, j2, i4);
                    return zzj2;
                }
                break;
            case 60:
                if (i5 == 2) {
                    Object zzH = zzH(obj, i4, i8);
                    int zzo = zzis.zzo(zzH, zzE(i8), bArr, i, i2, zzirVar);
                    zzP(obj, i4, i8, zzH);
                    return zzo;
                }
                break;
            case 61:
                if (i5 == 2) {
                    int zza2 = zzis.zza(bArr, i, zzirVar);
                    unsafe.putObject(obj, j, zzirVar.zzc);
                    unsafe.putInt(obj, j2, i4);
                    return zza2;
                }
                break;
            case 63:
                if (i5 == 0) {
                    int zzj3 = zzis.zzj(bArr, i, zzirVar);
                    int i10 = zzirVar.zza;
                    zzkj zzD = zzD(i8);
                    if (zzD != null && !zzD.zza(i10)) {
                        zzd(obj).zzj(i3, Long.valueOf(i10));
                    } else {
                        unsafe.putObject(obj, j, Integer.valueOf(i10));
                        unsafe.putInt(obj, j2, i4);
                    }
                    return zzj3;
                }
                break;
            case 66:
                if (i5 == 0) {
                    int zzj4 = zzis.zzj(bArr, i, zzirVar);
                    unsafe.putObject(obj, j, Integer.valueOf(zzji.zzb(zzirVar.zza)));
                    unsafe.putInt(obj, j2, i4);
                    return zzj4;
                }
                break;
            case 67:
                if (i5 == 0) {
                    int zzm3 = zzis.zzm(bArr, i, zzirVar);
                    unsafe.putObject(obj, j, Long.valueOf(zzji.zzc(zzirVar.zzb)));
                    unsafe.putInt(obj, j2, i4);
                    return zzm3;
                }
                break;
            case 68:
                if (i5 == 3) {
                    Object zzH2 = zzH(obj, i4, i8);
                    int zzn = zzis.zzn(zzH2, zzE(i8), bArr, i, i2, (i3 & (-8)) | 4, zzirVar);
                    zzP(obj, i4, i8, zzH2);
                    return zzn;
                }
                break;
        }
        return i;
    }

    /* JADX WARN: Code restructure failed: missing block: B:100:0x02e7, code lost:
        if (r0 != r24) goto L42;
     */
    /* JADX WARN: Code restructure failed: missing block: B:101:0x02e9, code lost:
        r14 = r31;
        r12 = r32;
        r13 = r34;
        r11 = r35;
        r2 = r15;
        r10 = r18;
        r1 = r23;
        r6 = r25;
        r7 = r26;
     */
    /* JADX WARN: Code restructure failed: missing block: B:102:0x02fc, code lost:
        r2 = r0;
     */
    /* JADX WARN: Code restructure failed: missing block: B:108:0x0328, code lost:
        if (r0 != r14) goto L42;
     */
    /* JADX WARN: Code restructure failed: missing block: B:113:0x034b, code lost:
        if (r0 != r14) goto L42;
     */
    /* JADX WARN: Multi-variable type inference failed */
    /* JADX WARN: Type inference failed for: r3v10, types: [int] */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    private final int zzu(Object obj, byte[] bArr, int i, int i2, zzir zzirVar) {
        byte b2;
        int i3;
        int zzw;
        int i4;
        int i5;
        Unsafe unsafe;
        int i6;
        int i7;
        int i8;
        int i9;
        Unsafe unsafe2;
        int i10;
        Unsafe unsafe3;
        zzlp<T> zzlpVar;
        int i11;
        Unsafe unsafe4;
        int i12;
        int i13;
        int i14;
        zzlp<T> zzlpVar2 = this;
        Object obj2 = obj;
        byte[] bArr2 = bArr;
        int i15 = i2;
        zzir zzirVar2 = zzirVar;
        zzJ(obj);
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
                i3 = zzis.zzk(b3, bArr2, i23, zzirVar2);
                b2 = zzirVar2.zza;
            } else {
                b2 = b3;
                i3 = i23;
            }
            int i24 = b2 >>> 3;
            int i25 = b2 & 7;
            if (i24 > i20) {
                zzw = zzlpVar2.zzx(i24, i21 / 3);
            } else {
                zzw = zzlpVar2.zzw(i24);
            }
            int i26 = zzw;
            if (i26 == i16) {
                i4 = i3;
                i5 = i24;
                unsafe = unsafe5;
                i6 = i16;
                i7 = 0;
            } else {
                int[] iArr = zzlpVar2.zzc;
                int i27 = iArr[i26 + 1];
                int zzA = zzA(i27);
                Unsafe unsafe6 = unsafe5;
                long j = i27 & i17;
                if (zzA <= 17) {
                    int i28 = iArr[i26 + 2];
                    int i29 = 1 << (i28 >>> 20);
                    int i30 = i28 & 1048575;
                    if (i30 != i19) {
                        i8 = i27;
                        i9 = i26;
                        if (i19 != 1048575) {
                            long j2 = i19;
                            unsafe4 = unsafe6;
                            unsafe4.putInt(obj2, j2, i22);
                        } else {
                            unsafe4 = unsafe6;
                        }
                        if (i30 != 1048575) {
                            i22 = unsafe4.getInt(obj2, i30);
                        }
                        unsafe2 = unsafe4;
                        i19 = i30;
                    } else {
                        i8 = i27;
                        i9 = i26;
                        unsafe2 = unsafe6;
                    }
                    switch (zzA) {
                        case 0:
                            zzlpVar = this;
                            i5 = i24;
                            i11 = 1048575;
                            i7 = i9;
                            i10 = i19;
                            unsafe3 = unsafe2;
                            if (i25 != 1) {
                                i4 = i3;
                                unsafe = unsafe3;
                                i19 = i10;
                                i6 = -1;
                                break;
                            } else {
                                zzmy.zzo(obj2, j, Double.longBitsToDouble(zzis.zzp(bArr2, i3)));
                                i18 = i3 + 8;
                                i22 |= i29;
                                unsafe5 = unsafe3;
                                i21 = i7;
                                i17 = i11;
                                i19 = i10;
                                i20 = i5;
                                i16 = -1;
                                zzlpVar2 = zzlpVar;
                                i15 = i2;
                                break;
                            }
                        case 1:
                            zzlpVar = this;
                            i5 = i24;
                            i11 = 1048575;
                            i7 = i9;
                            i10 = i19;
                            unsafe3 = unsafe2;
                            if (i25 != 5) {
                                i4 = i3;
                                unsafe = unsafe3;
                                i19 = i10;
                                i6 = -1;
                                break;
                            } else {
                                zzmy.zzp(obj2, j, Float.intBitsToFloat(zzis.zzb(bArr2, i3)));
                                i18 = i3 + 4;
                                i22 |= i29;
                                unsafe5 = unsafe3;
                                i21 = i7;
                                i17 = i11;
                                i19 = i10;
                                i20 = i5;
                                i16 = -1;
                                zzlpVar2 = zzlpVar;
                                i15 = i2;
                                break;
                            }
                        case 2:
                        case 3:
                            zzlpVar = this;
                            i5 = i24;
                            i11 = 1048575;
                            i7 = i9;
                            i10 = i19;
                            unsafe3 = unsafe2;
                            if (i25 != 0) {
                                i4 = i3;
                                unsafe = unsafe3;
                                i19 = i10;
                                i6 = -1;
                                break;
                            } else {
                                int zzm = zzis.zzm(bArr2, i3, zzirVar2);
                                unsafe3.putLong(obj, j, zzirVar2.zzb);
                                i22 |= i29;
                                unsafe5 = unsafe3;
                                i21 = i7;
                                i18 = zzm;
                                i17 = i11;
                                i19 = i10;
                                i20 = i5;
                                i16 = -1;
                                zzlpVar2 = zzlpVar;
                                i15 = i2;
                                break;
                            }
                        case 4:
                        case 11:
                            zzlpVar = this;
                            i5 = i24;
                            i11 = 1048575;
                            i7 = i9;
                            i10 = i19;
                            unsafe3 = unsafe2;
                            if (i25 != 0) {
                                i4 = i3;
                                unsafe = unsafe3;
                                i19 = i10;
                                i6 = -1;
                                break;
                            } else {
                                i18 = zzis.zzj(bArr2, i3, zzirVar2);
                                unsafe3.putInt(obj2, j, zzirVar2.zza);
                                i22 |= i29;
                                unsafe5 = unsafe3;
                                i21 = i7;
                                i17 = i11;
                                i19 = i10;
                                i20 = i5;
                                i16 = -1;
                                zzlpVar2 = zzlpVar;
                                i15 = i2;
                                break;
                            }
                        case 5:
                        case 14:
                            zzlpVar = this;
                            i5 = i24;
                            i11 = 1048575;
                            i7 = i9;
                            i10 = i19;
                            unsafe3 = unsafe2;
                            if (i25 != 1) {
                                i4 = i3;
                                unsafe = unsafe3;
                                i19 = i10;
                                i6 = -1;
                                break;
                            } else {
                                unsafe3.putLong(obj, j, zzis.zzp(bArr2, i3));
                                i18 = i3 + 8;
                                i22 |= i29;
                                unsafe5 = unsafe3;
                                i21 = i7;
                                i17 = i11;
                                i19 = i10;
                                i20 = i5;
                                i16 = -1;
                                zzlpVar2 = zzlpVar;
                                i15 = i2;
                                break;
                            }
                        case 6:
                        case 13:
                            zzlpVar = this;
                            i5 = i24;
                            i11 = 1048575;
                            i7 = i9;
                            i10 = i19;
                            unsafe3 = unsafe2;
                            if (i25 != 5) {
                                i4 = i3;
                                unsafe = unsafe3;
                                i19 = i10;
                                i6 = -1;
                                break;
                            } else {
                                unsafe3.putInt(obj2, j, zzis.zzb(bArr2, i3));
                                i18 = i3 + 4;
                                i22 |= i29;
                                unsafe5 = unsafe3;
                                i21 = i7;
                                i17 = i11;
                                i19 = i10;
                                i20 = i5;
                                i16 = -1;
                                zzlpVar2 = zzlpVar;
                                i15 = i2;
                                break;
                            }
                        case 7:
                            zzlpVar = this;
                            i5 = i24;
                            i11 = 1048575;
                            i7 = i9;
                            i10 = i19;
                            unsafe3 = unsafe2;
                            if (i25 != 0) {
                                i4 = i3;
                                unsafe = unsafe3;
                                i19 = i10;
                                i6 = -1;
                                break;
                            } else {
                                i18 = zzis.zzm(bArr2, i3, zzirVar2);
                                zzmy.zzm(obj2, j, zzirVar2.zzb != 0);
                                i22 |= i29;
                                unsafe5 = unsafe3;
                                i21 = i7;
                                i17 = i11;
                                i19 = i10;
                                i20 = i5;
                                i16 = -1;
                                zzlpVar2 = zzlpVar;
                                i15 = i2;
                                break;
                            }
                        case 8:
                            zzlpVar = this;
                            i5 = i24;
                            i11 = 1048575;
                            i7 = i9;
                            i10 = i19;
                            unsafe3 = unsafe2;
                            if (i25 != 2) {
                                i4 = i3;
                                unsafe = unsafe3;
                                i19 = i10;
                                i6 = -1;
                                break;
                            } else {
                                if ((i8 & 536870912) == 0) {
                                    i18 = zzis.zzg(bArr2, i3, zzirVar2);
                                } else {
                                    i18 = zzis.zzh(bArr2, i3, zzirVar2);
                                }
                                unsafe3.putObject(obj2, j, zzirVar2.zzc);
                                i22 |= i29;
                                unsafe5 = unsafe3;
                                i21 = i7;
                                i17 = i11;
                                i19 = i10;
                                i20 = i5;
                                i16 = -1;
                                zzlpVar2 = zzlpVar;
                                i15 = i2;
                                break;
                            }
                        case 9:
                            i5 = i24;
                            i7 = i9;
                            i10 = i19;
                            unsafe3 = unsafe2;
                            if (i25 != 2) {
                                i4 = i3;
                                unsafe = unsafe3;
                                i19 = i10;
                                i6 = -1;
                                break;
                            } else {
                                zzlpVar = this;
                                i11 = 1048575;
                                Object zzG = zzlpVar.zzG(obj2, i7);
                                i18 = zzis.zzo(zzG, zzlpVar.zzE(i7), bArr, i3, i2, zzirVar);
                                zzlpVar.zzO(obj2, i7, zzG);
                                i22 |= i29;
                                unsafe5 = unsafe3;
                                i21 = i7;
                                i17 = i11;
                                i19 = i10;
                                i20 = i5;
                                i16 = -1;
                                zzlpVar2 = zzlpVar;
                                i15 = i2;
                                break;
                            }
                        case 10:
                            i5 = i24;
                            i7 = i9;
                            i10 = i19;
                            unsafe3 = unsafe2;
                            if (i25 == 2) {
                                i18 = zzis.zza(bArr2, i3, zzirVar2);
                                unsafe3.putObject(obj2, j, zzirVar2.zzc);
                                i22 |= i29;
                                i15 = i2;
                                unsafe5 = unsafe3;
                                i21 = i7;
                                i19 = i10;
                                i20 = i5;
                                i17 = 1048575;
                                i16 = -1;
                                zzlpVar2 = this;
                                break;
                            }
                            i4 = i3;
                            unsafe = unsafe3;
                            i19 = i10;
                            i6 = -1;
                            break;
                        case 12:
                            i5 = i24;
                            i7 = i9;
                            i10 = i19;
                            unsafe3 = unsafe2;
                            if (i25 == 0) {
                                i18 = zzis.zzj(bArr2, i3, zzirVar2);
                                unsafe3.putInt(obj2, j, zzirVar2.zza);
                                i22 |= i29;
                                i15 = i2;
                                unsafe5 = unsafe3;
                                i21 = i7;
                                i19 = i10;
                                i20 = i5;
                                i17 = 1048575;
                                i16 = -1;
                                zzlpVar2 = this;
                                break;
                            }
                            i4 = i3;
                            unsafe = unsafe3;
                            i19 = i10;
                            i6 = -1;
                            break;
                        case 15:
                            i5 = i24;
                            i7 = i9;
                            i10 = i19;
                            unsafe3 = unsafe2;
                            if (i25 == 0) {
                                i18 = zzis.zzj(bArr2, i3, zzirVar2);
                                unsafe3.putInt(obj2, j, zzji.zzb(zzirVar2.zza));
                                i22 |= i29;
                                i15 = i2;
                                unsafe5 = unsafe3;
                                i21 = i7;
                                i19 = i10;
                                i20 = i5;
                                i17 = 1048575;
                                i16 = -1;
                                zzlpVar2 = this;
                                break;
                            }
                            i4 = i3;
                            unsafe = unsafe3;
                            i19 = i10;
                            i6 = -1;
                            break;
                        case 16:
                            if (i25 != 0) {
                                i5 = i24;
                                i7 = i9;
                                i10 = i19;
                                unsafe3 = unsafe2;
                                i4 = i3;
                                unsafe = unsafe3;
                                i19 = i10;
                                i6 = -1;
                                break;
                            } else {
                                int zzm2 = zzis.zzm(bArr2, i3, zzirVar2);
                                unsafe2.putLong(obj, j, zzji.zzc(zzirVar2.zzb));
                                i22 |= i29;
                                unsafe5 = unsafe2;
                                i18 = zzm2;
                                i21 = i9;
                                i19 = i19;
                                i20 = i24;
                                i17 = 1048575;
                                i16 = -1;
                                zzlpVar2 = this;
                                i15 = i2;
                                break;
                            }
                        default:
                            i5 = i24;
                            i7 = i9;
                            i10 = i19;
                            unsafe3 = unsafe2;
                            i4 = i3;
                            unsafe = unsafe3;
                            i19 = i10;
                            i6 = -1;
                            break;
                    }
                } else {
                    i5 = i24;
                    int i31 = i19;
                    zzlp<T> zzlpVar3 = zzlpVar2;
                    i7 = i26;
                    if (zzA != 27) {
                        if (zzA <= 49) {
                            int i32 = i3;
                            i13 = i22;
                            i14 = i31;
                            unsafe = unsafe6;
                            i6 = -1;
                            i18 = zzv(obj, bArr, i3, i2, b2, i5, i25, i7, i27, zzA, j, zzirVar);
                        } else {
                            i12 = i3;
                            i13 = i22;
                            unsafe = unsafe6;
                            i14 = i31;
                            i6 = -1;
                            if (zzA != 50) {
                                i18 = zzt(obj, bArr, i12, i2, b2, i5, i25, i27, zzA, j, i7, zzirVar);
                            } else if (i25 == 2) {
                                i18 = zzs(obj, bArr, i12, i2, i7, j, zzirVar);
                            }
                        }
                        zzlpVar2 = this;
                    } else if (i25 == 2) {
                        zzkm zzkmVar = (zzkm) unsafe6.getObject(obj2, j);
                        if (!zzkmVar.zzc()) {
                            int size = zzkmVar.size();
                            zzkmVar = zzkmVar.zzd(size == 0 ? 10 : size + size);
                            unsafe6.putObject(obj2, j, zzkmVar);
                        }
                        i18 = zzis.zze(zzlpVar3.zzE(i7), b2, bArr, i3, i2, zzkmVar, zzirVar);
                        i15 = i2;
                        unsafe5 = unsafe6;
                        i22 = i22;
                        i21 = i7;
                        i17 = 1048575;
                        i19 = i31;
                        i20 = i5;
                        zzlpVar2 = zzlpVar3;
                        i16 = -1;
                    } else {
                        i12 = i3;
                        i13 = i22;
                        unsafe = unsafe6;
                        i14 = i31;
                        i6 = -1;
                    }
                    i4 = i12;
                    i22 = i13;
                    i19 = i14;
                }
                unsafe5 = unsafe;
                i17 = 1048575;
                zzlpVar2 = this;
            }
            i18 = zzis.zzi(b2, bArr, i4, i2, zzd(obj), zzirVar);
            obj2 = obj;
            bArr2 = bArr;
            i15 = i2;
            zzirVar2 = zzirVar;
            i21 = i7;
            i16 = i6;
            i20 = i5;
            unsafe5 = unsafe;
            i17 = 1048575;
            zzlpVar2 = this;
        }
        int i33 = i22;
        Unsafe unsafe7 = unsafe5;
        if (i19 != i17) {
            unsafe7.putInt(obj, i19, i33);
        }
        if (i18 == i2) {
            return i18;
        }
        throw zzkp.zze();
    }

    /* JADX WARN: Removed duplicated region for block: B:115:0x0216  */
    /* JADX WARN: Removed duplicated region for block: B:62:0x014b  */
    /* JADX WARN: Removed duplicated region for block: B:95:0x01c8  */
    /* JADX WARN: Unsupported multi-entry loop pattern (BACK_EDGE: B:113:0x0213 -> B:114:0x0214). Please submit an issue!!! */
    /* JADX WARN: Unsupported multi-entry loop pattern (BACK_EDGE: B:60:0x0148 -> B:61:0x0149). Please submit an issue!!! */
    /* JADX WARN: Unsupported multi-entry loop pattern (BACK_EDGE: B:93:0x01c5 -> B:94:0x01c6). Please submit an issue!!! */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    private final int zzv(Object obj, byte[] bArr, int i, int i2, int i3, int i4, int i5, int i6, long j, int i7, long j2, zzir zzirVar) {
        int i8;
        int i9;
        int i10;
        int i11;
        int zzj;
        int i12;
        int i13 = i;
        Unsafe unsafe = zzb;
        zzkm zzkmVar = (zzkm) unsafe.getObject(obj, j2);
        if (!zzkmVar.zzc()) {
            int size = zzkmVar.size();
            zzkmVar = zzkmVar.zzd(size == 0 ? 10 : size + size);
            unsafe.putObject(obj, j2, zzkmVar);
        }
        switch (i7) {
            case 18:
            case 35:
                if (i5 == 2) {
                    zzjo zzjoVar = (zzjo) zzkmVar;
                    int zzj2 = zzis.zzj(bArr, i13, zzirVar);
                    int i14 = zzirVar.zza + zzj2;
                    while (zzj2 < i14) {
                        zzjoVar.zze(Double.longBitsToDouble(zzis.zzp(bArr, zzj2)));
                        zzj2 += 8;
                    }
                    if (zzj2 == i14) {
                        return zzj2;
                    }
                    throw zzkp.zzf();
                }
                if (i5 == 1) {
                    zzjo zzjoVar2 = (zzjo) zzkmVar;
                    zzjoVar2.zze(Double.longBitsToDouble(zzis.zzp(bArr, i)));
                    while (true) {
                        i8 = i13 + 8;
                        if (i8 < i2) {
                            i13 = zzis.zzj(bArr, i8, zzirVar);
                            if (i3 == zzirVar.zza) {
                                zzjoVar2.zze(Double.longBitsToDouble(zzis.zzp(bArr, i13)));
                            }
                        }
                    }
                    return i8;
                }
                return i13;
            case 19:
            case 36:
                if (i5 == 2) {
                    zzjy zzjyVar = (zzjy) zzkmVar;
                    int zzj3 = zzis.zzj(bArr, i13, zzirVar);
                    int i15 = zzirVar.zza + zzj3;
                    while (zzj3 < i15) {
                        zzjyVar.zze(Float.intBitsToFloat(zzis.zzb(bArr, zzj3)));
                        zzj3 += 4;
                    }
                    if (zzj3 == i15) {
                        return zzj3;
                    }
                    throw zzkp.zzf();
                }
                if (i5 == 5) {
                    zzjy zzjyVar2 = (zzjy) zzkmVar;
                    zzjyVar2.zze(Float.intBitsToFloat(zzis.zzb(bArr, i)));
                    while (true) {
                        i9 = i13 + 4;
                        if (i9 < i2) {
                            i13 = zzis.zzj(bArr, i9, zzirVar);
                            if (i3 == zzirVar.zza) {
                                zzjyVar2.zze(Float.intBitsToFloat(zzis.zzb(bArr, i13)));
                            }
                        }
                    }
                    return i9;
                }
                return i13;
            case 20:
            case 21:
            case 37:
            case 38:
                if (i5 == 2) {
                    zzlb zzlbVar = (zzlb) zzkmVar;
                    int zzj4 = zzis.zzj(bArr, i13, zzirVar);
                    int i16 = zzirVar.zza + zzj4;
                    while (zzj4 < i16) {
                        zzj4 = zzis.zzm(bArr, zzj4, zzirVar);
                        zzlbVar.zzg(zzirVar.zzb);
                    }
                    if (zzj4 == i16) {
                        return zzj4;
                    }
                    throw zzkp.zzf();
                }
                if (i5 == 0) {
                    zzlb zzlbVar2 = (zzlb) zzkmVar;
                    int zzm = zzis.zzm(bArr, i13, zzirVar);
                    zzlbVar2.zzg(zzirVar.zzb);
                    while (zzm < i2) {
                        int zzj5 = zzis.zzj(bArr, zzm, zzirVar);
                        if (i3 != zzirVar.zza) {
                            return zzm;
                        }
                        zzm = zzis.zzm(bArr, zzj5, zzirVar);
                        zzlbVar2.zzg(zzirVar.zzb);
                    }
                    return zzm;
                }
                return i13;
            case 22:
            case 29:
            case 39:
            case 43:
                if (i5 == 2) {
                    return zzis.zzf(bArr, i13, zzkmVar, zzirVar);
                }
                if (i5 == 0) {
                    return zzis.zzl(i3, bArr, i, i2, zzkmVar, zzirVar);
                }
                return i13;
            case 23:
            case 32:
            case 40:
            case 46:
                if (i5 == 2) {
                    zzlb zzlbVar3 = (zzlb) zzkmVar;
                    int zzj6 = zzis.zzj(bArr, i13, zzirVar);
                    int i17 = zzirVar.zza + zzj6;
                    while (zzj6 < i17) {
                        zzlbVar3.zzg(zzis.zzp(bArr, zzj6));
                        zzj6 += 8;
                    }
                    if (zzj6 == i17) {
                        return zzj6;
                    }
                    throw zzkp.zzf();
                }
                if (i5 == 1) {
                    zzlb zzlbVar4 = (zzlb) zzkmVar;
                    zzlbVar4.zzg(zzis.zzp(bArr, i));
                    while (true) {
                        i10 = i13 + 8;
                        if (i10 < i2) {
                            i13 = zzis.zzj(bArr, i10, zzirVar);
                            if (i3 == zzirVar.zza) {
                                zzlbVar4.zzg(zzis.zzp(bArr, i13));
                            }
                        }
                    }
                    return i10;
                }
                return i13;
            case 24:
            case 31:
            case 41:
            case 45:
                if (i5 == 2) {
                    zzkg zzkgVar = (zzkg) zzkmVar;
                    int zzj7 = zzis.zzj(bArr, i13, zzirVar);
                    int i18 = zzirVar.zza + zzj7;
                    while (zzj7 < i18) {
                        zzkgVar.zzh(zzis.zzb(bArr, zzj7));
                        zzj7 += 4;
                    }
                    if (zzj7 == i18) {
                        return zzj7;
                    }
                    throw zzkp.zzf();
                }
                if (i5 == 5) {
                    zzkg zzkgVar2 = (zzkg) zzkmVar;
                    zzkgVar2.zzh(zzis.zzb(bArr, i));
                    while (true) {
                        i11 = i13 + 4;
                        if (i11 < i2) {
                            i13 = zzis.zzj(bArr, i11, zzirVar);
                            if (i3 == zzirVar.zza) {
                                zzkgVar2.zzh(zzis.zzb(bArr, i13));
                            }
                        }
                    }
                    return i11;
                }
                return i13;
            case 25:
            case 42:
                if (i5 == 2) {
                    zzit zzitVar = (zzit) zzkmVar;
                    zzj = zzis.zzj(bArr, i13, zzirVar);
                    int i19 = zzirVar.zza + zzj;
                    while (zzj < i19) {
                        zzj = zzis.zzm(bArr, zzj, zzirVar);
                        zzitVar.zze(zzirVar.zzb != 0);
                    }
                    if (zzj != i19) {
                        throw zzkp.zzf();
                    }
                    return zzj;
                }
                if (i5 == 0) {
                    zzit zzitVar2 = (zzit) zzkmVar;
                    int zzm2 = zzis.zzm(bArr, i13, zzirVar);
                    zzitVar2.zze(zzirVar.zzb != 0);
                    while (zzm2 < i2) {
                        int zzj8 = zzis.zzj(bArr, zzm2, zzirVar);
                        if (i3 != zzirVar.zza) {
                            return zzm2;
                        }
                        zzm2 = zzis.zzm(bArr, zzj8, zzirVar);
                        zzitVar2.zze(zzirVar.zzb != 0);
                    }
                    return zzm2;
                }
                return i13;
            case 26:
                if (i5 == 2) {
                    if ((j & 536870912) == 0) {
                        int zzj9 = zzis.zzj(bArr, i13, zzirVar);
                        int i20 = zzirVar.zza;
                        if (i20 >= 0) {
                            if (i20 == 0) {
                                zzkmVar.add("");
                                while (zzj9 < i2) {
                                    int zzj10 = zzis.zzj(bArr, zzj9, zzirVar);
                                    if (i3 != zzirVar.zza) {
                                        return zzj9;
                                    }
                                    zzj9 = zzis.zzj(bArr, zzj10, zzirVar);
                                    i20 = zzirVar.zza;
                                    if (i20 < 0) {
                                        throw zzkp.zzd();
                                    }
                                    if (i20 == 0) {
                                        zzkmVar.add("");
                                    } else {
                                        zzkmVar.add(new String(bArr, zzj9, i20, zzkn.zzb));
                                        zzj9 += i20;
                                        while (zzj9 < i2) {
                                        }
                                    }
                                }
                                return zzj9;
                            }
                            zzkmVar.add(new String(bArr, zzj9, i20, zzkn.zzb));
                            zzj9 += i20;
                            while (zzj9 < i2) {
                            }
                            return zzj9;
                        }
                        throw zzkp.zzd();
                    }
                    int zzj11 = zzis.zzj(bArr, i13, zzirVar);
                    int i21 = zzirVar.zza;
                    if (i21 >= 0) {
                        if (i21 == 0) {
                            zzkmVar.add("");
                            while (zzj11 < i2) {
                                int zzj12 = zzis.zzj(bArr, zzj11, zzirVar);
                                if (i3 != zzirVar.zza) {
                                    return zzj11;
                                }
                                zzj11 = zzis.zzj(bArr, zzj12, zzirVar);
                                int i22 = zzirVar.zza;
                                if (i22 < 0) {
                                    throw zzkp.zzd();
                                }
                                if (i22 == 0) {
                                    zzkmVar.add("");
                                } else {
                                    i12 = zzj11 + i22;
                                    if (zznd.zzf(bArr, zzj11, i12)) {
                                        zzkmVar.add(new String(bArr, zzj11, i22, zzkn.zzb));
                                        zzj11 = i12;
                                        while (zzj11 < i2) {
                                        }
                                    } else {
                                        throw zzkp.zzc();
                                    }
                                }
                            }
                            return zzj11;
                        }
                        i12 = zzj11 + i21;
                        if (zznd.zzf(bArr, zzj11, i12)) {
                            zzkmVar.add(new String(bArr, zzj11, i21, zzkn.zzb));
                            zzj11 = i12;
                            while (zzj11 < i2) {
                            }
                            return zzj11;
                        }
                        throw zzkp.zzc();
                    }
                    throw zzkp.zzd();
                }
                return i13;
            case 27:
                if (i5 == 2) {
                    return zzis.zze(zzE(i6), i3, bArr, i, i2, zzkmVar, zzirVar);
                }
                return i13;
            case 28:
                if (i5 == 2) {
                    int zzj13 = zzis.zzj(bArr, i13, zzirVar);
                    int i23 = zzirVar.zza;
                    if (i23 >= 0) {
                        if (i23 <= bArr.length - zzj13) {
                            if (i23 == 0) {
                                zzkmVar.add(zzje.zzb);
                                while (zzj13 < i2) {
                                    int zzj14 = zzis.zzj(bArr, zzj13, zzirVar);
                                    if (i3 != zzirVar.zza) {
                                        return zzj13;
                                    }
                                    zzj13 = zzis.zzj(bArr, zzj14, zzirVar);
                                    i23 = zzirVar.zza;
                                    if (i23 >= 0) {
                                        if (i23 > bArr.length - zzj13) {
                                            throw zzkp.zzf();
                                        }
                                        if (i23 == 0) {
                                            zzkmVar.add(zzje.zzb);
                                        } else {
                                            zzkmVar.add(zzje.zzl(bArr, zzj13, i23));
                                            zzj13 += i23;
                                            while (zzj13 < i2) {
                                            }
                                        }
                                    } else {
                                        throw zzkp.zzd();
                                    }
                                }
                                return zzj13;
                            }
                            zzkmVar.add(zzje.zzl(bArr, zzj13, i23));
                            zzj13 += i23;
                            while (zzj13 < i2) {
                            }
                            return zzj13;
                        }
                        throw zzkp.zzf();
                    }
                    throw zzkp.zzd();
                }
                return i13;
            case 30:
            case 44:
                if (i5 != 2) {
                    if (i5 == 0) {
                        zzj = zzis.zzl(i3, bArr, i, i2, zzkmVar, zzirVar);
                    }
                    return i13;
                }
                zzj = zzis.zzf(bArr, i13, zzkmVar, zzirVar);
                zzlz.zzC(obj, i4, zzkmVar, zzD(i6), null, this.zzn);
                return zzj;
            case 33:
            case 47:
                if (i5 == 2) {
                    zzkg zzkgVar3 = (zzkg) zzkmVar;
                    int zzj15 = zzis.zzj(bArr, i13, zzirVar);
                    int i24 = zzirVar.zza + zzj15;
                    while (zzj15 < i24) {
                        zzj15 = zzis.zzj(bArr, zzj15, zzirVar);
                        zzkgVar3.zzh(zzji.zzb(zzirVar.zza));
                    }
                    if (zzj15 == i24) {
                        return zzj15;
                    }
                    throw zzkp.zzf();
                }
                if (i5 == 0) {
                    zzkg zzkgVar4 = (zzkg) zzkmVar;
                    int zzj16 = zzis.zzj(bArr, i13, zzirVar);
                    zzkgVar4.zzh(zzji.zzb(zzirVar.zza));
                    while (zzj16 < i2) {
                        int zzj17 = zzis.zzj(bArr, zzj16, zzirVar);
                        if (i3 != zzirVar.zza) {
                            return zzj16;
                        }
                        zzj16 = zzis.zzj(bArr, zzj17, zzirVar);
                        zzkgVar4.zzh(zzji.zzb(zzirVar.zza));
                    }
                    return zzj16;
                }
                return i13;
            case 34:
            case 48:
                if (i5 == 2) {
                    zzlb zzlbVar5 = (zzlb) zzkmVar;
                    int zzj18 = zzis.zzj(bArr, i13, zzirVar);
                    int i25 = zzirVar.zza + zzj18;
                    while (zzj18 < i25) {
                        zzj18 = zzis.zzm(bArr, zzj18, zzirVar);
                        zzlbVar5.zzg(zzji.zzc(zzirVar.zzb));
                    }
                    if (zzj18 == i25) {
                        return zzj18;
                    }
                    throw zzkp.zzf();
                }
                if (i5 == 0) {
                    zzlb zzlbVar6 = (zzlb) zzkmVar;
                    int zzm3 = zzis.zzm(bArr, i13, zzirVar);
                    zzlbVar6.zzg(zzji.zzc(zzirVar.zzb));
                    while (zzm3 < i2) {
                        int zzj19 = zzis.zzj(bArr, zzm3, zzirVar);
                        if (i3 != zzirVar.zza) {
                            return zzm3;
                        }
                        zzm3 = zzis.zzm(bArr, zzj19, zzirVar);
                        zzlbVar6.zzg(zzji.zzc(zzirVar.zzb));
                    }
                    return zzm3;
                }
                return i13;
            default:
                if (i5 == 3) {
                    zzlx zzE = zzE(i6);
                    int i26 = (i3 & (-8)) | 4;
                    int zzc = zzis.zzc(zzE, bArr, i, i2, i26, zzirVar);
                    zzkmVar.add(zzirVar.zzc);
                    while (zzc < i2) {
                        int zzj20 = zzis.zzj(bArr, zzc, zzirVar);
                        if (i3 != zzirVar.zza) {
                            return zzc;
                        }
                        zzc = zzis.zzc(zzE, bArr, zzj20, i2, i26, zzirVar);
                        zzkmVar.add(zzirVar.zzc);
                    }
                    return zzc;
                }
                return i13;
        }
    }

    private final int zzw(int i) {
        if (i < this.zze || i > this.zzf) {
            return -1;
        }
        return zzz(i, 0);
    }

    private final int zzx(int i, int i2) {
        if (i < this.zze || i > this.zzf) {
            return -1;
        }
        return zzz(i, i2);
    }

    private final int zzy(int i) {
        return this.zzc[i + 2];
    }

    private final int zzz(int i, int i2) {
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

    @Override // com.google.android.gms.internal.measurement.zzlx
    public final int zza(Object obj) {
        return this.zzi ? zzq(obj) : zzp(obj);
    }

    @Override // com.google.android.gms.internal.measurement.zzlx
    public final int zzb(Object obj) {
        int i;
        int zzc;
        int i2;
        int zzr;
        int length = this.zzc.length;
        int i3 = 0;
        for (int i4 = 0; i4 < length; i4 += 3) {
            int zzB = zzB(i4);
            int i5 = this.zzc[i4];
            long j = 1048575 & zzB;
            int i6 = 37;
            switch (zzA(zzB)) {
                case 0:
                    i = i3 * 53;
                    zzc = zzkn.zzc(Double.doubleToLongBits(zzmy.zza(obj, j)));
                    i3 = zzc + i;
                    break;
                case 1:
                    i = i3 * 53;
                    zzc = Float.floatToIntBits(zzmy.zzb(obj, j));
                    i3 = zzc + i;
                    break;
                case 2:
                    i = i3 * 53;
                    zzc = zzkn.zzc(zzmy.zzd(obj, j));
                    i3 = zzc + i;
                    break;
                case 3:
                    i = i3 * 53;
                    zzc = zzkn.zzc(zzmy.zzd(obj, j));
                    i3 = zzc + i;
                    break;
                case 4:
                    i = i3 * 53;
                    zzc = zzmy.zzc(obj, j);
                    i3 = zzc + i;
                    break;
                case 5:
                    i = i3 * 53;
                    zzc = zzkn.zzc(zzmy.zzd(obj, j));
                    i3 = zzc + i;
                    break;
                case 6:
                    i = i3 * 53;
                    zzc = zzmy.zzc(obj, j);
                    i3 = zzc + i;
                    break;
                case 7:
                    i = i3 * 53;
                    zzc = zzkn.zza(zzmy.zzw(obj, j));
                    i3 = zzc + i;
                    break;
                case 8:
                    i = i3 * 53;
                    zzc = ((String) zzmy.zzf(obj, j)).hashCode();
                    i3 = zzc + i;
                    break;
                case 9:
                    Object zzf = zzmy.zzf(obj, j);
                    if (zzf != null) {
                        i6 = zzf.hashCode();
                    }
                    i3 = (i3 * 53) + i6;
                    break;
                case 10:
                    i = i3 * 53;
                    zzc = zzmy.zzf(obj, j).hashCode();
                    i3 = zzc + i;
                    break;
                case 11:
                    i = i3 * 53;
                    zzc = zzmy.zzc(obj, j);
                    i3 = zzc + i;
                    break;
                case 12:
                    i = i3 * 53;
                    zzc = zzmy.zzc(obj, j);
                    i3 = zzc + i;
                    break;
                case 13:
                    i = i3 * 53;
                    zzc = zzmy.zzc(obj, j);
                    i3 = zzc + i;
                    break;
                case 14:
                    i = i3 * 53;
                    zzc = zzkn.zzc(zzmy.zzd(obj, j));
                    i3 = zzc + i;
                    break;
                case 15:
                    i = i3 * 53;
                    zzc = zzmy.zzc(obj, j);
                    i3 = zzc + i;
                    break;
                case 16:
                    i = i3 * 53;
                    zzc = zzkn.zzc(zzmy.zzd(obj, j));
                    i3 = zzc + i;
                    break;
                case 17:
                    Object zzf2 = zzmy.zzf(obj, j);
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
                    zzc = zzmy.zzf(obj, j).hashCode();
                    i3 = zzc + i;
                    break;
                case 50:
                    i = i3 * 53;
                    zzc = zzmy.zzf(obj, j).hashCode();
                    i3 = zzc + i;
                    break;
                case 51:
                    if (zzX(obj, i5, i4)) {
                        i = i3 * 53;
                        zzc = zzkn.zzc(Double.doubleToLongBits(zzn(obj, j)));
                        i3 = zzc + i;
                        break;
                    } else {
                        break;
                    }
                case 52:
                    if (zzX(obj, i5, i4)) {
                        i = i3 * 53;
                        zzc = Float.floatToIntBits(zzo(obj, j));
                        i3 = zzc + i;
                        break;
                    } else {
                        break;
                    }
                case 53:
                    if (zzX(obj, i5, i4)) {
                        i = i3 * 53;
                        zzc = zzkn.zzc(zzC(obj, j));
                        i3 = zzc + i;
                        break;
                    } else {
                        break;
                    }
                case 54:
                    if (zzX(obj, i5, i4)) {
                        i = i3 * 53;
                        zzc = zzkn.zzc(zzC(obj, j));
                        i3 = zzc + i;
                        break;
                    } else {
                        break;
                    }
                case 55:
                    if (zzX(obj, i5, i4)) {
                        i2 = i3 * 53;
                        zzr = zzr(obj, j);
                        i3 = i2 + zzr;
                        break;
                    } else {
                        break;
                    }
                case 56:
                    if (zzX(obj, i5, i4)) {
                        i = i3 * 53;
                        zzc = zzkn.zzc(zzC(obj, j));
                        i3 = zzc + i;
                        break;
                    } else {
                        break;
                    }
                case 57:
                    if (zzX(obj, i5, i4)) {
                        i2 = i3 * 53;
                        zzr = zzr(obj, j);
                        i3 = i2 + zzr;
                        break;
                    } else {
                        break;
                    }
                case 58:
                    if (zzX(obj, i5, i4)) {
                        i = i3 * 53;
                        zzc = zzkn.zza(zzY(obj, j));
                        i3 = zzc + i;
                        break;
                    } else {
                        break;
                    }
                case 59:
                    if (zzX(obj, i5, i4)) {
                        i = i3 * 53;
                        zzc = ((String) zzmy.zzf(obj, j)).hashCode();
                        i3 = zzc + i;
                        break;
                    } else {
                        break;
                    }
                case 60:
                    if (zzX(obj, i5, i4)) {
                        i = i3 * 53;
                        zzc = zzmy.zzf(obj, j).hashCode();
                        i3 = zzc + i;
                        break;
                    } else {
                        break;
                    }
                case 61:
                    if (zzX(obj, i5, i4)) {
                        i = i3 * 53;
                        zzc = zzmy.zzf(obj, j).hashCode();
                        i3 = zzc + i;
                        break;
                    } else {
                        break;
                    }
                case 62:
                    if (zzX(obj, i5, i4)) {
                        i2 = i3 * 53;
                        zzr = zzr(obj, j);
                        i3 = i2 + zzr;
                        break;
                    } else {
                        break;
                    }
                case 63:
                    if (zzX(obj, i5, i4)) {
                        i2 = i3 * 53;
                        zzr = zzr(obj, j);
                        i3 = i2 + zzr;
                        break;
                    } else {
                        break;
                    }
                case 64:
                    if (zzX(obj, i5, i4)) {
                        i2 = i3 * 53;
                        zzr = zzr(obj, j);
                        i3 = i2 + zzr;
                        break;
                    } else {
                        break;
                    }
                case 65:
                    if (zzX(obj, i5, i4)) {
                        i = i3 * 53;
                        zzc = zzkn.zzc(zzC(obj, j));
                        i3 = zzc + i;
                        break;
                    } else {
                        break;
                    }
                case 66:
                    if (zzX(obj, i5, i4)) {
                        i2 = i3 * 53;
                        zzr = zzr(obj, j);
                        i3 = i2 + zzr;
                        break;
                    } else {
                        break;
                    }
                case 67:
                    if (zzX(obj, i5, i4)) {
                        i = i3 * 53;
                        zzc = zzkn.zzc(zzC(obj, j));
                        i3 = zzc + i;
                        break;
                    } else {
                        break;
                    }
                case 68:
                    if (zzX(obj, i5, i4)) {
                        i = i3 * 53;
                        zzc = zzmy.zzf(obj, j).hashCode();
                        i3 = zzc + i;
                        break;
                    } else {
                        break;
                    }
            }
        }
        int hashCode = this.zzn.zzd(obj).hashCode() + (i3 * 53);
        if (this.zzh) {
            this.zzo.zza(obj);
            throw null;
        }
        return hashCode;
    }

    /* JADX WARN: Code restructure failed: missing block: B:147:0x041e, code lost:
        if (r6 == 1048575) goto L36;
     */
    /* JADX WARN: Code restructure failed: missing block: B:148:0x0420, code lost:
        r28.putInt(r12, r6, r5);
     */
    /* JADX WARN: Code restructure failed: missing block: B:149:0x0426, code lost:
        r3 = r9.zzk;
     */
    /* JADX WARN: Code restructure failed: missing block: B:151:0x042a, code lost:
        if (r3 >= r9.zzl) goto L50;
     */
    /* JADX WARN: Code restructure failed: missing block: B:152:0x042c, code lost:
        r4 = r9.zzj[r3];
        r5 = r9.zzc[r4];
        r5 = com.google.android.gms.internal.measurement.zzmy.zzf(r12, r9.zzB(r4) & 1048575);
     */
    /* JADX WARN: Code restructure failed: missing block: B:153:0x043e, code lost:
        if (r5 != null) goto L41;
     */
    /* JADX WARN: Code restructure failed: missing block: B:156:0x0445, code lost:
        if (r9.zzD(r4) != null) goto L43;
     */
    /* JADX WARN: Code restructure failed: missing block: B:157:0x0447, code lost:
        r3 = r3 + 1;
     */
    /* JADX WARN: Code restructure failed: missing block: B:158:0x044a, code lost:
        r5 = (com.google.android.gms.internal.measurement.zzlg) r5;
        r0 = (com.google.android.gms.internal.measurement.zzlf) r9.zzF(r4);
     */
    /* JADX WARN: Code restructure failed: missing block: B:159:0x0452, code lost:
        throw null;
     */
    /* JADX WARN: Code restructure failed: missing block: B:160:0x0453, code lost:
        if (r7 != 0) goto L57;
     */
    /* JADX WARN: Code restructure failed: missing block: B:162:0x0457, code lost:
        if (r0 != r33) goto L54;
     */
    /* JADX WARN: Code restructure failed: missing block: B:165:0x045e, code lost:
        throw com.google.android.gms.internal.measurement.zzkp.zze();
     */
    /* JADX WARN: Code restructure failed: missing block: B:167:0x0461, code lost:
        if (r0 > r33) goto L60;
     */
    /* JADX WARN: Code restructure failed: missing block: B:168:0x0463, code lost:
        if (r1 != r7) goto L60;
     */
    /* JADX WARN: Code restructure failed: missing block: B:169:0x0465, code lost:
        return r0;
     */
    /* JADX WARN: Code restructure failed: missing block: B:171:0x046a, code lost:
        throw com.google.android.gms.internal.measurement.zzkp.zze();
     */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public final int zzc(Object obj, byte[] bArr, int i, int i2, int i3, zzir zzirVar) {
        Unsafe unsafe;
        int i4;
        Object obj2;
        zzlp<T> zzlpVar;
        byte b2;
        int zzw;
        int i5;
        int i6;
        int i7;
        int i8;
        int i9;
        Object obj3;
        zzir zzirVar2;
        int i10;
        int i11;
        int i12;
        int i13;
        int i14;
        int i15;
        int i16;
        int i17;
        int i18;
        int i19;
        int i20;
        zzlp<T> zzlpVar2 = this;
        Object obj4 = obj;
        byte[] bArr2 = bArr;
        int i21 = i2;
        int i22 = i3;
        zzir zzirVar3 = zzirVar;
        zzJ(obj);
        Unsafe unsafe2 = zzb;
        int i23 = i;
        int i24 = 0;
        int i25 = 0;
        int i26 = 0;
        int i27 = -1;
        int i28 = 1048575;
        while (true) {
            if (i23 < i21) {
                int i29 = i23 + 1;
                byte b3 = bArr2[i23];
                if (b3 < 0) {
                    int zzk = zzis.zzk(b3, bArr2, i29, zzirVar3);
                    b2 = zzirVar3.zza;
                    i29 = zzk;
                } else {
                    b2 = b3;
                }
                int i30 = b2 >>> 3;
                int i31 = b2 & 7;
                if (i30 > i27) {
                    zzw = zzlpVar2.zzx(i30, i25 / 3);
                } else {
                    zzw = zzlpVar2.zzw(i30);
                }
                if (zzw == -1) {
                    i5 = i30;
                    i6 = i29;
                    i7 = b2;
                    i8 = i26;
                    unsafe = unsafe2;
                    i4 = i22;
                    i9 = 0;
                } else {
                    int[] iArr = zzlpVar2.zzc;
                    int i32 = iArr[zzw + 1];
                    int zzA = zzA(i32);
                    int i33 = i29;
                    long j = i32 & 1048575;
                    if (zzA <= 17) {
                        int i34 = iArr[zzw + 2];
                        int i35 = 1 << (i34 >>> 20);
                        int i36 = i34 & 1048575;
                        if (i36 != i28) {
                            i11 = b2;
                            if (i28 != 1048575) {
                                unsafe2.putInt(obj4, i28, i26);
                            }
                            i12 = i36;
                            i8 = unsafe2.getInt(obj4, i36);
                        } else {
                            i11 = b2;
                            i8 = i26;
                            i12 = i28;
                        }
                        switch (zzA) {
                            case 0:
                                i13 = zzw;
                                i14 = i30;
                                i15 = i33;
                                if (i31 != 1) {
                                    i7 = i11;
                                    i4 = i3;
                                    i6 = i15;
                                    unsafe = unsafe2;
                                    i5 = i14;
                                    i9 = i13;
                                    i28 = i12;
                                    break;
                                } else {
                                    zzmy.zzo(obj4, j, Double.longBitsToDouble(zzis.zzp(bArr2, i15)));
                                    i23 = i15 + 8;
                                    i26 = i8 | i35;
                                    i27 = i14;
                                    i25 = i13;
                                    i24 = i11;
                                    break;
                                }
                            case 1:
                                i13 = zzw;
                                i14 = i30;
                                i15 = i33;
                                if (i31 != 5) {
                                    i7 = i11;
                                    i4 = i3;
                                    i6 = i15;
                                    unsafe = unsafe2;
                                    i5 = i14;
                                    i9 = i13;
                                    i28 = i12;
                                    break;
                                } else {
                                    zzmy.zzp(obj4, j, Float.intBitsToFloat(zzis.zzb(bArr2, i15)));
                                    i23 = i15 + 4;
                                    i26 = i8 | i35;
                                    i27 = i14;
                                    i25 = i13;
                                    i24 = i11;
                                    break;
                                }
                            case 2:
                            case 3:
                                i13 = zzw;
                                i14 = i30;
                                i15 = i33;
                                if (i31 != 0) {
                                    i7 = i11;
                                    i4 = i3;
                                    i6 = i15;
                                    unsafe = unsafe2;
                                    i5 = i14;
                                    i9 = i13;
                                    i28 = i12;
                                    break;
                                } else {
                                    int zzm = zzis.zzm(bArr2, i15, zzirVar3);
                                    unsafe2.putLong(obj, j, zzirVar3.zzb);
                                    i26 = i8 | i35;
                                    i23 = zzm;
                                    i27 = i14;
                                    i25 = i13;
                                    i24 = i11;
                                    break;
                                }
                            case 4:
                            case 11:
                                i13 = zzw;
                                i14 = i30;
                                i15 = i33;
                                if (i31 != 0) {
                                    i7 = i11;
                                    i4 = i3;
                                    i6 = i15;
                                    unsafe = unsafe2;
                                    i5 = i14;
                                    i9 = i13;
                                    i28 = i12;
                                    break;
                                } else {
                                    i23 = zzis.zzj(bArr2, i15, zzirVar3);
                                    unsafe2.putInt(obj4, j, zzirVar3.zza);
                                    i26 = i8 | i35;
                                    i27 = i14;
                                    i25 = i13;
                                    i24 = i11;
                                    break;
                                }
                            case 5:
                            case 14:
                                i13 = zzw;
                                int i37 = i11;
                                i14 = i30;
                                if (i31 != 1) {
                                    i11 = i37;
                                    i15 = i33;
                                    i7 = i11;
                                    i4 = i3;
                                    i6 = i15;
                                    unsafe = unsafe2;
                                    i5 = i14;
                                    i9 = i13;
                                    i28 = i12;
                                    break;
                                } else {
                                    i11 = i37;
                                    i15 = i33;
                                    unsafe2.putLong(obj, j, zzis.zzp(bArr2, i33));
                                    i23 = i15 + 8;
                                    i26 = i8 | i35;
                                    i27 = i14;
                                    i25 = i13;
                                    i24 = i11;
                                    break;
                                }
                            case 6:
                            case 13:
                                i13 = zzw;
                                i16 = i11;
                                i14 = i30;
                                i17 = i33;
                                if (i31 != 5) {
                                    i7 = i16;
                                    i15 = i17;
                                    i4 = i3;
                                    i6 = i15;
                                    unsafe = unsafe2;
                                    i5 = i14;
                                    i9 = i13;
                                    i28 = i12;
                                    break;
                                } else {
                                    unsafe2.putInt(obj4, j, zzis.zzb(bArr2, i17));
                                    i23 = i17 + 4;
                                    i26 = i8 | i35;
                                    i24 = i16;
                                    i27 = i14;
                                    i25 = i13;
                                    break;
                                }
                            case 7:
                                i13 = zzw;
                                i16 = i11;
                                i14 = i30;
                                i17 = i33;
                                if (i31 != 0) {
                                    i7 = i16;
                                    i15 = i17;
                                    i4 = i3;
                                    i6 = i15;
                                    unsafe = unsafe2;
                                    i5 = i14;
                                    i9 = i13;
                                    i28 = i12;
                                    break;
                                } else {
                                    i23 = zzis.zzm(bArr2, i17, zzirVar3);
                                    zzmy.zzm(obj4, j, zzirVar3.zzb != 0);
                                    i26 = i8 | i35;
                                    i24 = i16;
                                    i27 = i14;
                                    i25 = i13;
                                    break;
                                }
                            case 8:
                                i13 = zzw;
                                i16 = i11;
                                i14 = i30;
                                i17 = i33;
                                if (i31 != 2) {
                                    i7 = i16;
                                    i15 = i17;
                                    i4 = i3;
                                    i6 = i15;
                                    unsafe = unsafe2;
                                    i5 = i14;
                                    i9 = i13;
                                    i28 = i12;
                                    break;
                                } else {
                                    if ((536870912 & i32) == 0) {
                                        i23 = zzis.zzg(bArr2, i17, zzirVar3);
                                    } else {
                                        i23 = zzis.zzh(bArr2, i17, zzirVar3);
                                    }
                                    unsafe2.putObject(obj4, j, zzirVar3.zzc);
                                    i26 = i8 | i35;
                                    i24 = i16;
                                    i27 = i14;
                                    i25 = i13;
                                    break;
                                }
                            case 9:
                                i13 = zzw;
                                i7 = i11;
                                i14 = i30;
                                i18 = i33;
                                if (i31 != 2) {
                                    i15 = i18;
                                    i4 = i3;
                                    i6 = i15;
                                    unsafe = unsafe2;
                                    i5 = i14;
                                    i9 = i13;
                                    i28 = i12;
                                    break;
                                } else {
                                    Object zzG = zzlpVar2.zzG(obj4, i13);
                                    i23 = zzis.zzo(zzG, zzlpVar2.zzE(i13), bArr, i18, i2, zzirVar);
                                    zzlpVar2.zzO(obj4, i13, zzG);
                                    i26 = i8 | i35;
                                    i24 = i7;
                                    i27 = i14;
                                    i25 = i13;
                                    break;
                                }
                            case 10:
                                i13 = zzw;
                                i7 = i11;
                                i14 = i30;
                                i18 = i33;
                                if (i31 != 2) {
                                    i15 = i18;
                                    i4 = i3;
                                    i6 = i15;
                                    unsafe = unsafe2;
                                    i5 = i14;
                                    i9 = i13;
                                    i28 = i12;
                                    break;
                                } else {
                                    i23 = zzis.zza(bArr2, i18, zzirVar3);
                                    unsafe2.putObject(obj4, j, zzirVar3.zzc);
                                    i26 = i8 | i35;
                                    i24 = i7;
                                    i27 = i14;
                                    i25 = i13;
                                    break;
                                }
                            case 12:
                                i13 = zzw;
                                i7 = i11;
                                i14 = i30;
                                i18 = i33;
                                if (i31 != 0) {
                                    i15 = i18;
                                    i4 = i3;
                                    i6 = i15;
                                    unsafe = unsafe2;
                                    i5 = i14;
                                    i9 = i13;
                                    i28 = i12;
                                    break;
                                } else {
                                    i23 = zzis.zzj(bArr2, i18, zzirVar3);
                                    int i38 = zzirVar3.zza;
                                    zzkj zzD = zzlpVar2.zzD(i13);
                                    if (zzD != null && !zzD.zza(i38)) {
                                        zzd(obj).zzj(i7, Long.valueOf(i38));
                                        i24 = i7;
                                        i27 = i14;
                                        i25 = i13;
                                        i26 = i8;
                                        break;
                                    } else {
                                        unsafe2.putInt(obj4, j, i38);
                                        i26 = i8 | i35;
                                        i24 = i7;
                                        i27 = i14;
                                        i25 = i13;
                                        break;
                                    }
                                }
                                break;
                            case 15:
                                i13 = zzw;
                                i7 = i11;
                                i14 = i30;
                                i18 = i33;
                                if (i31 != 0) {
                                    i15 = i18;
                                    i4 = i3;
                                    i6 = i15;
                                    unsafe = unsafe2;
                                    i5 = i14;
                                    i9 = i13;
                                    i28 = i12;
                                    break;
                                } else {
                                    i23 = zzis.zzj(bArr2, i18, zzirVar3);
                                    unsafe2.putInt(obj4, j, zzji.zzb(zzirVar3.zza));
                                    i26 = i8 | i35;
                                    i24 = i7;
                                    i27 = i14;
                                    i25 = i13;
                                    break;
                                }
                            case 16:
                                if (i31 != 0) {
                                    i13 = zzw;
                                    i14 = i30;
                                    i7 = i11;
                                    i15 = i33;
                                    i4 = i3;
                                    i6 = i15;
                                    unsafe = unsafe2;
                                    i5 = i14;
                                    i9 = i13;
                                    i28 = i12;
                                    break;
                                } else {
                                    int zzm2 = zzis.zzm(bArr2, i33, zzirVar3);
                                    i14 = i30;
                                    i13 = zzw;
                                    i7 = i11;
                                    unsafe2.putLong(obj, j, zzji.zzc(zzirVar3.zzb));
                                    i26 = i8 | i35;
                                    i23 = zzm2;
                                    i24 = i7;
                                    i27 = i14;
                                    i25 = i13;
                                    break;
                                }
                            default:
                                i13 = zzw;
                                i14 = i30;
                                i15 = i33;
                                if (i31 != 3) {
                                    i7 = i11;
                                    i4 = i3;
                                    i6 = i15;
                                    unsafe = unsafe2;
                                    i5 = i14;
                                    i9 = i13;
                                    i28 = i12;
                                    break;
                                } else {
                                    Object zzG2 = zzlpVar2.zzG(obj4, i13);
                                    i7 = i11;
                                    i23 = zzis.zzn(zzG2, zzlpVar2.zzE(i13), bArr, i15, i2, (i14 << 3) | 4, zzirVar);
                                    zzlpVar2.zzO(obj4, i13, zzG2);
                                    i26 = i8 | i35;
                                    i24 = i7;
                                    i27 = i14;
                                    i25 = i13;
                                    break;
                                }
                        }
                        i28 = i12;
                        i21 = i2;
                        i22 = i3;
                    } else {
                        i13 = zzw;
                        int i39 = b2;
                        i12 = i28;
                        if (zzA != 27) {
                            i8 = i26;
                            if (zzA <= 49) {
                                i19 = i39;
                                i5 = i30;
                                unsafe = unsafe2;
                                i9 = i13;
                                i23 = zzv(obj, bArr, i33, i2, i39, i30, i31, i13, i32, zzA, j, zzirVar);
                                if (i23 != i33) {
                                    zzlpVar2 = this;
                                    obj4 = obj;
                                    bArr2 = bArr;
                                    i21 = i2;
                                    i22 = i3;
                                    zzirVar3 = zzirVar;
                                    i24 = i19;
                                    i27 = i5;
                                    i25 = i9;
                                    i26 = i8;
                                    i28 = i12;
                                    unsafe2 = unsafe;
                                } else {
                                    i4 = i3;
                                    i6 = i23;
                                    i7 = i19;
                                }
                            } else {
                                i5 = i30;
                                i19 = i39;
                                i20 = i33;
                                unsafe = unsafe2;
                                i9 = i13;
                                if (zzA != 50) {
                                    i23 = zzt(obj, bArr, i20, i2, i19, i5, i31, i32, zzA, j, i9, zzirVar);
                                    if (i23 != i20) {
                                        zzlpVar2 = this;
                                        obj4 = obj;
                                        bArr2 = bArr;
                                        i21 = i2;
                                        i22 = i3;
                                        zzirVar3 = zzirVar;
                                        i24 = i19;
                                        i27 = i5;
                                        i25 = i9;
                                        i26 = i8;
                                        i28 = i12;
                                        unsafe2 = unsafe;
                                    } else {
                                        i4 = i3;
                                        i6 = i23;
                                        i7 = i19;
                                    }
                                } else if (i31 == 2) {
                                    i23 = zzs(obj, bArr, i20, i2, i9, j, zzirVar);
                                    if (i23 != i20) {
                                        zzlpVar2 = this;
                                        obj4 = obj;
                                        bArr2 = bArr;
                                        i21 = i2;
                                        i22 = i3;
                                        zzirVar3 = zzirVar;
                                        i24 = i19;
                                        i27 = i5;
                                        i25 = i9;
                                        i26 = i8;
                                        i28 = i12;
                                        unsafe2 = unsafe;
                                    } else {
                                        i4 = i3;
                                        i6 = i23;
                                        i7 = i19;
                                    }
                                }
                            }
                            i28 = i12;
                        } else if (i31 == 2) {
                            zzkm zzkmVar = (zzkm) unsafe2.getObject(obj4, j);
                            if (!zzkmVar.zzc()) {
                                int size = zzkmVar.size();
                                zzkmVar = zzkmVar.zzd(size == 0 ? 10 : size + size);
                                unsafe2.putObject(obj4, j, zzkmVar);
                            }
                            i8 = i26;
                            i23 = zzis.zze(zzlpVar2.zzE(i13), i39, bArr, i33, i2, zzkmVar, zzirVar);
                            i27 = i30;
                            i24 = i39;
                            i25 = i13;
                            i26 = i8;
                            i28 = i12;
                            i21 = i2;
                            i22 = i3;
                        } else {
                            i8 = i26;
                            i5 = i30;
                            i19 = i39;
                            i20 = i33;
                            unsafe = unsafe2;
                            i9 = i13;
                        }
                        i4 = i3;
                        i6 = i20;
                        i7 = i19;
                        i28 = i12;
                    }
                }
                if (i7 != i4 || i4 == 0) {
                    if (this.zzh) {
                        zzirVar2 = zzirVar;
                        zzjr zzjrVar = zzirVar2.zzd;
                        if (zzjrVar != zzjr.zza) {
                            i10 = i5;
                            if (zzjrVar.zzb(this.zzg, i10) == null) {
                                i23 = zzis.zzi(i7, bArr, i6, i2, zzd(obj), zzirVar);
                                obj3 = obj;
                                i21 = i2;
                                i24 = i7;
                                zzlpVar2 = this;
                                zzirVar3 = zzirVar2;
                                i27 = i10;
                                obj4 = obj3;
                                i25 = i9;
                                i26 = i8;
                                unsafe2 = unsafe;
                                bArr2 = bArr;
                                i22 = i4;
                            } else {
                                zzkc zzkcVar = (zzkc) obj;
                                throw null;
                            }
                        } else {
                            obj3 = obj;
                        }
                    } else {
                        obj3 = obj;
                        zzirVar2 = zzirVar;
                    }
                    i10 = i5;
                    i23 = zzis.zzi(i7, bArr, i6, i2, zzd(obj), zzirVar);
                    i21 = i2;
                    i24 = i7;
                    zzlpVar2 = this;
                    zzirVar3 = zzirVar2;
                    i27 = i10;
                    obj4 = obj3;
                    i25 = i9;
                    i26 = i8;
                    unsafe2 = unsafe;
                    bArr2 = bArr;
                    i22 = i4;
                } else {
                    zzlpVar = this;
                    obj2 = obj;
                    i23 = i6;
                    i24 = i7;
                    i26 = i8;
                }
            } else {
                unsafe = unsafe2;
                i4 = i22;
                obj2 = obj4;
                zzlpVar = zzlpVar2;
            }
        }
    }

    @Override // com.google.android.gms.internal.measurement.zzlx
    public final Object zze() {
        return ((zzkf) this.zzg).zzbA();
    }

    @Override // com.google.android.gms.internal.measurement.zzlx
    public final void zzf(Object obj) {
        if (zzW(obj)) {
            if (obj instanceof zzkf) {
                zzkf zzkfVar = (zzkf) obj;
                zzkfVar.zzbM(Integer.MAX_VALUE);
                zzkfVar.zzb = 0;
                zzkfVar.zzbK();
            }
            int length = this.zzc.length;
            for (int i = 0; i < length; i += 3) {
                int zzB = zzB(i);
                long j = 1048575 & zzB;
                int zzA = zzA(zzB);
                if (zzA != 9) {
                    switch (zzA) {
                        case 17:
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
                            this.zzm.zza(obj, j);
                            continue;
                        case 50:
                            Unsafe unsafe = zzb;
                            Object object = unsafe.getObject(obj, j);
                            if (object != null) {
                                ((zzlg) object).zzc();
                                unsafe.putObject(obj, j, object);
                            } else {
                                continue;
                            }
                        default:
                    }
                }
                if (zzT(obj, i)) {
                    zzE(i).zzf(zzb.getObject(obj, j));
                }
            }
            this.zzn.zzg(obj);
            if (this.zzh) {
                this.zzo.zzb(obj);
            }
        }
    }

    @Override // com.google.android.gms.internal.measurement.zzlx
    public final void zzg(Object obj, Object obj2) {
        zzJ(obj);
        Objects.requireNonNull(obj2);
        for (int i = 0; i < this.zzc.length; i += 3) {
            int zzB = zzB(i);
            long j = 1048575 & zzB;
            int i2 = this.zzc[i];
            switch (zzA(zzB)) {
                case 0:
                    if (zzT(obj2, i)) {
                        zzmy.zzo(obj, j, zzmy.zza(obj2, j));
                        zzM(obj, i);
                        break;
                    } else {
                        break;
                    }
                case 1:
                    if (zzT(obj2, i)) {
                        zzmy.zzp(obj, j, zzmy.zzb(obj2, j));
                        zzM(obj, i);
                        break;
                    } else {
                        break;
                    }
                case 2:
                    if (zzT(obj2, i)) {
                        zzmy.zzr(obj, j, zzmy.zzd(obj2, j));
                        zzM(obj, i);
                        break;
                    } else {
                        break;
                    }
                case 3:
                    if (zzT(obj2, i)) {
                        zzmy.zzr(obj, j, zzmy.zzd(obj2, j));
                        zzM(obj, i);
                        break;
                    } else {
                        break;
                    }
                case 4:
                    if (zzT(obj2, i)) {
                        zzmy.zzq(obj, j, zzmy.zzc(obj2, j));
                        zzM(obj, i);
                        break;
                    } else {
                        break;
                    }
                case 5:
                    if (zzT(obj2, i)) {
                        zzmy.zzr(obj, j, zzmy.zzd(obj2, j));
                        zzM(obj, i);
                        break;
                    } else {
                        break;
                    }
                case 6:
                    if (zzT(obj2, i)) {
                        zzmy.zzq(obj, j, zzmy.zzc(obj2, j));
                        zzM(obj, i);
                        break;
                    } else {
                        break;
                    }
                case 7:
                    if (zzT(obj2, i)) {
                        zzmy.zzm(obj, j, zzmy.zzw(obj2, j));
                        zzM(obj, i);
                        break;
                    } else {
                        break;
                    }
                case 8:
                    if (zzT(obj2, i)) {
                        zzmy.zzs(obj, j, zzmy.zzf(obj2, j));
                        zzM(obj, i);
                        break;
                    } else {
                        break;
                    }
                case 9:
                    zzK(obj, obj2, i);
                    break;
                case 10:
                    if (zzT(obj2, i)) {
                        zzmy.zzs(obj, j, zzmy.zzf(obj2, j));
                        zzM(obj, i);
                        break;
                    } else {
                        break;
                    }
                case 11:
                    if (zzT(obj2, i)) {
                        zzmy.zzq(obj, j, zzmy.zzc(obj2, j));
                        zzM(obj, i);
                        break;
                    } else {
                        break;
                    }
                case 12:
                    if (zzT(obj2, i)) {
                        zzmy.zzq(obj, j, zzmy.zzc(obj2, j));
                        zzM(obj, i);
                        break;
                    } else {
                        break;
                    }
                case 13:
                    if (zzT(obj2, i)) {
                        zzmy.zzq(obj, j, zzmy.zzc(obj2, j));
                        zzM(obj, i);
                        break;
                    } else {
                        break;
                    }
                case 14:
                    if (zzT(obj2, i)) {
                        zzmy.zzr(obj, j, zzmy.zzd(obj2, j));
                        zzM(obj, i);
                        break;
                    } else {
                        break;
                    }
                case 15:
                    if (zzT(obj2, i)) {
                        zzmy.zzq(obj, j, zzmy.zzc(obj2, j));
                        zzM(obj, i);
                        break;
                    } else {
                        break;
                    }
                case 16:
                    if (zzT(obj2, i)) {
                        zzmy.zzr(obj, j, zzmy.zzd(obj2, j));
                        zzM(obj, i);
                        break;
                    } else {
                        break;
                    }
                case 17:
                    zzK(obj, obj2, i);
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
                    this.zzm.zzb(obj, obj2, j);
                    break;
                case 50:
                    zzlz.zzaa(this.zzq, obj, obj2, j);
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
                    if (zzX(obj2, i2, i)) {
                        zzmy.zzs(obj, j, zzmy.zzf(obj2, j));
                        zzN(obj, i2, i);
                        break;
                    } else {
                        break;
                    }
                case 60:
                    zzL(obj, obj2, i);
                    break;
                case 61:
                case 62:
                case 63:
                case 64:
                case 65:
                case 66:
                case 67:
                    if (zzX(obj2, i2, i)) {
                        zzmy.zzs(obj, j, zzmy.zzf(obj2, j));
                        zzN(obj, i2, i);
                        break;
                    } else {
                        break;
                    }
                case 68:
                    zzL(obj, obj2, i);
                    break;
            }
        }
        zzlz.zzF(this.zzn, obj, obj2);
        if (this.zzh) {
            zzlz.zzE(this.zzo, obj, obj2);
        }
    }

    @Override // com.google.android.gms.internal.measurement.zzlx
    public final void zzh(Object obj, byte[] bArr, int i, int i2, zzir zzirVar) {
        if (this.zzi) {
            zzu(obj, bArr, i, i2, zzirVar);
        } else {
            zzc(obj, bArr, i, i2, 0, zzirVar);
        }
    }

    @Override // com.google.android.gms.internal.measurement.zzlx
    public final void zzi(Object obj, zzng zzngVar) {
        if (!this.zzi) {
            zzQ(obj, zzngVar);
        } else if (!this.zzh) {
            int length = this.zzc.length;
            for (int i = 0; i < length; i += 3) {
                int zzB = zzB(i);
                int i2 = this.zzc[i];
                switch (zzA(zzB)) {
                    case 0:
                        if (zzT(obj, i)) {
                            zzngVar.zzf(i2, zzmy.zza(obj, zzB & 1048575));
                            break;
                        } else {
                            break;
                        }
                    case 1:
                        if (zzT(obj, i)) {
                            zzngVar.zzo(i2, zzmy.zzb(obj, zzB & 1048575));
                            break;
                        } else {
                            break;
                        }
                    case 2:
                        if (zzT(obj, i)) {
                            zzngVar.zzt(i2, zzmy.zzd(obj, zzB & 1048575));
                            break;
                        } else {
                            break;
                        }
                    case 3:
                        if (zzT(obj, i)) {
                            zzngVar.zzJ(i2, zzmy.zzd(obj, zzB & 1048575));
                            break;
                        } else {
                            break;
                        }
                    case 4:
                        if (zzT(obj, i)) {
                            zzngVar.zzr(i2, zzmy.zzc(obj, zzB & 1048575));
                            break;
                        } else {
                            break;
                        }
                    case 5:
                        if (zzT(obj, i)) {
                            zzngVar.zzm(i2, zzmy.zzd(obj, zzB & 1048575));
                            break;
                        } else {
                            break;
                        }
                    case 6:
                        if (zzT(obj, i)) {
                            zzngVar.zzk(i2, zzmy.zzc(obj, zzB & 1048575));
                            break;
                        } else {
                            break;
                        }
                    case 7:
                        if (zzT(obj, i)) {
                            zzngVar.zzb(i2, zzmy.zzw(obj, zzB & 1048575));
                            break;
                        } else {
                            break;
                        }
                    case 8:
                        if (zzT(obj, i)) {
                            zzZ(i2, zzmy.zzf(obj, zzB & 1048575), zzngVar);
                            break;
                        } else {
                            break;
                        }
                    case 9:
                        if (zzT(obj, i)) {
                            zzngVar.zzv(i2, zzmy.zzf(obj, zzB & 1048575), zzE(i));
                            break;
                        } else {
                            break;
                        }
                    case 10:
                        if (zzT(obj, i)) {
                            zzngVar.zzd(i2, (zzje) zzmy.zzf(obj, zzB & 1048575));
                            break;
                        } else {
                            break;
                        }
                    case 11:
                        if (zzT(obj, i)) {
                            zzngVar.zzH(i2, zzmy.zzc(obj, zzB & 1048575));
                            break;
                        } else {
                            break;
                        }
                    case 12:
                        if (zzT(obj, i)) {
                            zzngVar.zzi(i2, zzmy.zzc(obj, zzB & 1048575));
                            break;
                        } else {
                            break;
                        }
                    case 13:
                        if (zzT(obj, i)) {
                            zzngVar.zzw(i2, zzmy.zzc(obj, zzB & 1048575));
                            break;
                        } else {
                            break;
                        }
                    case 14:
                        if (zzT(obj, i)) {
                            zzngVar.zzy(i2, zzmy.zzd(obj, zzB & 1048575));
                            break;
                        } else {
                            break;
                        }
                    case 15:
                        if (zzT(obj, i)) {
                            zzngVar.zzA(i2, zzmy.zzc(obj, zzB & 1048575));
                            break;
                        } else {
                            break;
                        }
                    case 16:
                        if (zzT(obj, i)) {
                            zzngVar.zzC(i2, zzmy.zzd(obj, zzB & 1048575));
                            break;
                        } else {
                            break;
                        }
                    case 17:
                        if (zzT(obj, i)) {
                            zzngVar.zzq(i2, zzmy.zzf(obj, zzB & 1048575), zzE(i));
                            break;
                        } else {
                            break;
                        }
                    case 18:
                        zzlz.zzJ(i2, (List) zzmy.zzf(obj, zzB & 1048575), zzngVar, false);
                        break;
                    case 19:
                        zzlz.zzN(i2, (List) zzmy.zzf(obj, zzB & 1048575), zzngVar, false);
                        break;
                    case 20:
                        zzlz.zzQ(i2, (List) zzmy.zzf(obj, zzB & 1048575), zzngVar, false);
                        break;
                    case 21:
                        zzlz.zzY(i2, (List) zzmy.zzf(obj, zzB & 1048575), zzngVar, false);
                        break;
                    case 22:
                        zzlz.zzP(i2, (List) zzmy.zzf(obj, zzB & 1048575), zzngVar, false);
                        break;
                    case 23:
                        zzlz.zzM(i2, (List) zzmy.zzf(obj, zzB & 1048575), zzngVar, false);
                        break;
                    case 24:
                        zzlz.zzL(i2, (List) zzmy.zzf(obj, zzB & 1048575), zzngVar, false);
                        break;
                    case 25:
                        zzlz.zzH(i2, (List) zzmy.zzf(obj, zzB & 1048575), zzngVar, false);
                        break;
                    case 26:
                        zzlz.zzW(i2, (List) zzmy.zzf(obj, zzB & 1048575), zzngVar);
                        break;
                    case 27:
                        zzlz.zzR(i2, (List) zzmy.zzf(obj, zzB & 1048575), zzngVar, zzE(i));
                        break;
                    case 28:
                        zzlz.zzI(i2, (List) zzmy.zzf(obj, zzB & 1048575), zzngVar);
                        break;
                    case 29:
                        zzlz.zzX(i2, (List) zzmy.zzf(obj, zzB & 1048575), zzngVar, false);
                        break;
                    case 30:
                        zzlz.zzK(i2, (List) zzmy.zzf(obj, zzB & 1048575), zzngVar, false);
                        break;
                    case 31:
                        zzlz.zzS(i2, (List) zzmy.zzf(obj, zzB & 1048575), zzngVar, false);
                        break;
                    case 32:
                        zzlz.zzT(i2, (List) zzmy.zzf(obj, zzB & 1048575), zzngVar, false);
                        break;
                    case 33:
                        zzlz.zzU(i2, (List) zzmy.zzf(obj, zzB & 1048575), zzngVar, false);
                        break;
                    case 34:
                        zzlz.zzV(i2, (List) zzmy.zzf(obj, zzB & 1048575), zzngVar, false);
                        break;
                    case 35:
                        zzlz.zzJ(i2, (List) zzmy.zzf(obj, zzB & 1048575), zzngVar, true);
                        break;
                    case 36:
                        zzlz.zzN(i2, (List) zzmy.zzf(obj, zzB & 1048575), zzngVar, true);
                        break;
                    case 37:
                        zzlz.zzQ(i2, (List) zzmy.zzf(obj, zzB & 1048575), zzngVar, true);
                        break;
                    case 38:
                        zzlz.zzY(i2, (List) zzmy.zzf(obj, zzB & 1048575), zzngVar, true);
                        break;
                    case 39:
                        zzlz.zzP(i2, (List) zzmy.zzf(obj, zzB & 1048575), zzngVar, true);
                        break;
                    case 40:
                        zzlz.zzM(i2, (List) zzmy.zzf(obj, zzB & 1048575), zzngVar, true);
                        break;
                    case 41:
                        zzlz.zzL(i2, (List) zzmy.zzf(obj, zzB & 1048575), zzngVar, true);
                        break;
                    case 42:
                        zzlz.zzH(i2, (List) zzmy.zzf(obj, zzB & 1048575), zzngVar, true);
                        break;
                    case 43:
                        zzlz.zzX(i2, (List) zzmy.zzf(obj, zzB & 1048575), zzngVar, true);
                        break;
                    case 44:
                        zzlz.zzK(i2, (List) zzmy.zzf(obj, zzB & 1048575), zzngVar, true);
                        break;
                    case 45:
                        zzlz.zzS(i2, (List) zzmy.zzf(obj, zzB & 1048575), zzngVar, true);
                        break;
                    case 46:
                        zzlz.zzT(i2, (List) zzmy.zzf(obj, zzB & 1048575), zzngVar, true);
                        break;
                    case 47:
                        zzlz.zzU(i2, (List) zzmy.zzf(obj, zzB & 1048575), zzngVar, true);
                        break;
                    case 48:
                        zzlz.zzV(i2, (List) zzmy.zzf(obj, zzB & 1048575), zzngVar, true);
                        break;
                    case 49:
                        zzlz.zzO(i2, (List) zzmy.zzf(obj, zzB & 1048575), zzngVar, zzE(i));
                        break;
                    case 50:
                        zzR(zzngVar, i2, zzmy.zzf(obj, zzB & 1048575), i);
                        break;
                    case 51:
                        if (zzX(obj, i2, i)) {
                            zzngVar.zzf(i2, zzn(obj, zzB & 1048575));
                            break;
                        } else {
                            break;
                        }
                    case 52:
                        if (zzX(obj, i2, i)) {
                            zzngVar.zzo(i2, zzo(obj, zzB & 1048575));
                            break;
                        } else {
                            break;
                        }
                    case 53:
                        if (zzX(obj, i2, i)) {
                            zzngVar.zzt(i2, zzC(obj, zzB & 1048575));
                            break;
                        } else {
                            break;
                        }
                    case 54:
                        if (zzX(obj, i2, i)) {
                            zzngVar.zzJ(i2, zzC(obj, zzB & 1048575));
                            break;
                        } else {
                            break;
                        }
                    case 55:
                        if (zzX(obj, i2, i)) {
                            zzngVar.zzr(i2, zzr(obj, zzB & 1048575));
                            break;
                        } else {
                            break;
                        }
                    case 56:
                        if (zzX(obj, i2, i)) {
                            zzngVar.zzm(i2, zzC(obj, zzB & 1048575));
                            break;
                        } else {
                            break;
                        }
                    case 57:
                        if (zzX(obj, i2, i)) {
                            zzngVar.zzk(i2, zzr(obj, zzB & 1048575));
                            break;
                        } else {
                            break;
                        }
                    case 58:
                        if (zzX(obj, i2, i)) {
                            zzngVar.zzb(i2, zzY(obj, zzB & 1048575));
                            break;
                        } else {
                            break;
                        }
                    case 59:
                        if (zzX(obj, i2, i)) {
                            zzZ(i2, zzmy.zzf(obj, zzB & 1048575), zzngVar);
                            break;
                        } else {
                            break;
                        }
                    case 60:
                        if (zzX(obj, i2, i)) {
                            zzngVar.zzv(i2, zzmy.zzf(obj, zzB & 1048575), zzE(i));
                            break;
                        } else {
                            break;
                        }
                    case 61:
                        if (zzX(obj, i2, i)) {
                            zzngVar.zzd(i2, (zzje) zzmy.zzf(obj, zzB & 1048575));
                            break;
                        } else {
                            break;
                        }
                    case 62:
                        if (zzX(obj, i2, i)) {
                            zzngVar.zzH(i2, zzr(obj, zzB & 1048575));
                            break;
                        } else {
                            break;
                        }
                    case 63:
                        if (zzX(obj, i2, i)) {
                            zzngVar.zzi(i2, zzr(obj, zzB & 1048575));
                            break;
                        } else {
                            break;
                        }
                    case 64:
                        if (zzX(obj, i2, i)) {
                            zzngVar.zzw(i2, zzr(obj, zzB & 1048575));
                            break;
                        } else {
                            break;
                        }
                    case 65:
                        if (zzX(obj, i2, i)) {
                            zzngVar.zzy(i2, zzC(obj, zzB & 1048575));
                            break;
                        } else {
                            break;
                        }
                    case 66:
                        if (zzX(obj, i2, i)) {
                            zzngVar.zzA(i2, zzr(obj, zzB & 1048575));
                            break;
                        } else {
                            break;
                        }
                    case 67:
                        if (zzX(obj, i2, i)) {
                            zzngVar.zzC(i2, zzC(obj, zzB & 1048575));
                            break;
                        } else {
                            break;
                        }
                    case 68:
                        if (zzX(obj, i2, i)) {
                            zzngVar.zzq(i2, zzmy.zzf(obj, zzB & 1048575), zzE(i));
                            break;
                        } else {
                            break;
                        }
                }
            }
            zzmo zzmoVar = this.zzn;
            zzmoVar.zzi(zzmoVar.zzd(obj), zzngVar);
        } else {
            this.zzo.zza(obj);
            throw null;
        }
    }

    @Override // com.google.android.gms.internal.measurement.zzlx
    public final boolean zzj(Object obj, Object obj2) {
        boolean zzZ;
        int length = this.zzc.length;
        for (int i = 0; i < length; i += 3) {
            int zzB = zzB(i);
            long j = zzB & 1048575;
            switch (zzA(zzB)) {
                case 0:
                    if (zzS(obj, obj2, i) && Double.doubleToLongBits(zzmy.zza(obj, j)) == Double.doubleToLongBits(zzmy.zza(obj2, j))) {
                        continue;
                    }
                    return false;
                case 1:
                    if (zzS(obj, obj2, i) && Float.floatToIntBits(zzmy.zzb(obj, j)) == Float.floatToIntBits(zzmy.zzb(obj2, j))) {
                        continue;
                    }
                    return false;
                case 2:
                    if (zzS(obj, obj2, i) && zzmy.zzd(obj, j) == zzmy.zzd(obj2, j)) {
                        continue;
                    }
                    return false;
                case 3:
                    if (zzS(obj, obj2, i) && zzmy.zzd(obj, j) == zzmy.zzd(obj2, j)) {
                        continue;
                    }
                    return false;
                case 4:
                    if (zzS(obj, obj2, i) && zzmy.zzc(obj, j) == zzmy.zzc(obj2, j)) {
                        continue;
                    }
                    return false;
                case 5:
                    if (zzS(obj, obj2, i) && zzmy.zzd(obj, j) == zzmy.zzd(obj2, j)) {
                        continue;
                    }
                    return false;
                case 6:
                    if (zzS(obj, obj2, i) && zzmy.zzc(obj, j) == zzmy.zzc(obj2, j)) {
                        continue;
                    }
                    return false;
                case 7:
                    if (zzS(obj, obj2, i) && zzmy.zzw(obj, j) == zzmy.zzw(obj2, j)) {
                        continue;
                    }
                    return false;
                case 8:
                    if (zzS(obj, obj2, i) && zzlz.zzZ(zzmy.zzf(obj, j), zzmy.zzf(obj2, j))) {
                        continue;
                    }
                    return false;
                case 9:
                    if (zzS(obj, obj2, i) && zzlz.zzZ(zzmy.zzf(obj, j), zzmy.zzf(obj2, j))) {
                        continue;
                    }
                    return false;
                case 10:
                    if (zzS(obj, obj2, i) && zzlz.zzZ(zzmy.zzf(obj, j), zzmy.zzf(obj2, j))) {
                        continue;
                    }
                    return false;
                case 11:
                    if (zzS(obj, obj2, i) && zzmy.zzc(obj, j) == zzmy.zzc(obj2, j)) {
                        continue;
                    }
                    return false;
                case 12:
                    if (zzS(obj, obj2, i) && zzmy.zzc(obj, j) == zzmy.zzc(obj2, j)) {
                        continue;
                    }
                    return false;
                case 13:
                    if (zzS(obj, obj2, i) && zzmy.zzc(obj, j) == zzmy.zzc(obj2, j)) {
                        continue;
                    }
                    return false;
                case 14:
                    if (zzS(obj, obj2, i) && zzmy.zzd(obj, j) == zzmy.zzd(obj2, j)) {
                        continue;
                    }
                    return false;
                case 15:
                    if (zzS(obj, obj2, i) && zzmy.zzc(obj, j) == zzmy.zzc(obj2, j)) {
                        continue;
                    }
                    return false;
                case 16:
                    if (zzS(obj, obj2, i) && zzmy.zzd(obj, j) == zzmy.zzd(obj2, j)) {
                        continue;
                    }
                    return false;
                case 17:
                    if (zzS(obj, obj2, i) && zzlz.zzZ(zzmy.zzf(obj, j), zzmy.zzf(obj2, j))) {
                        continue;
                    }
                    return false;
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
                    zzZ = zzlz.zzZ(zzmy.zzf(obj, j), zzmy.zzf(obj2, j));
                    break;
                case 50:
                    zzZ = zzlz.zzZ(zzmy.zzf(obj, j), zzmy.zzf(obj2, j));
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
                    long zzy = zzy(i) & 1048575;
                    if (zzmy.zzc(obj, zzy) == zzmy.zzc(obj2, zzy) && zzlz.zzZ(zzmy.zzf(obj, j), zzmy.zzf(obj2, j))) {
                        continue;
                    }
                    return false;
                default:
            }
            if (!zzZ) {
                return false;
            }
        }
        if (this.zzn.zzd(obj).equals(this.zzn.zzd(obj2))) {
            if (this.zzh) {
                this.zzo.zza(obj);
                this.zzo.zza(obj2);
                throw null;
            }
            return true;
        }
        return false;
    }

    @Override // com.google.android.gms.internal.measurement.zzlx
    public final boolean zzk(Object obj) {
        int i;
        int i2;
        int i3 = 1048575;
        int i4 = 0;
        int i5 = 0;
        while (i5 < this.zzk) {
            int i6 = this.zzj[i5];
            int i7 = this.zzc[i6];
            int zzB = zzB(i6);
            int i8 = this.zzc[i6 + 2];
            int i9 = i8 & 1048575;
            int i10 = 1 << (i8 >>> 20);
            if (i9 != i3) {
                if (i9 != 1048575) {
                    i4 = zzb.getInt(obj, i9);
                }
                i2 = i4;
                i = i9;
            } else {
                i = i3;
                i2 = i4;
            }
            if ((268435456 & zzB) != 0 && !zzU(obj, i6, i, i2, i10)) {
                return false;
            }
            int zzA = zzA(zzB);
            if (zzA != 9 && zzA != 17) {
                if (zzA != 27) {
                    if (zzA == 60 || zzA == 68) {
                        if (zzX(obj, i7, i6) && !zzV(obj, zzB, zzE(i6))) {
                            return false;
                        }
                    } else if (zzA != 49) {
                        if (zzA == 50 && !((zzlg) zzmy.zzf(obj, zzB & 1048575)).isEmpty()) {
                            zzlf zzlfVar = (zzlf) zzF(i6);
                            throw null;
                        }
                    }
                }
                List list = (List) zzmy.zzf(obj, zzB & 1048575);
                if (list.isEmpty()) {
                    continue;
                } else {
                    zzlx zzE = zzE(i6);
                    for (int i11 = 0; i11 < list.size(); i11++) {
                        if (!zzE.zzk(list.get(i11))) {
                            return false;
                        }
                    }
                    continue;
                }
            } else if (zzU(obj, i6, i, i2, i10) && !zzV(obj, zzB, zzE(i6))) {
                return false;
            }
            i5++;
            i3 = i;
            i4 = i2;
        }
        if (this.zzh) {
            this.zzo.zza(obj);
            throw null;
        }
        return true;
    }
}