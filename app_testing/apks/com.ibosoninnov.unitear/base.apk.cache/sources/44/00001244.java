package com.google.android.gms.internal.measurement;

/* JADX WARN: Enum visitor error
jadx.core.utils.exceptions.JadxRuntimeException: Init of enum zza uses external variables
	at jadx.core.dex.visitors.EnumVisitor.createEnumFieldByConstructor(EnumVisitor.java:444)
	at jadx.core.dex.visitors.EnumVisitor.processEnumFieldByField(EnumVisitor.java:368)
	at jadx.core.dex.visitors.EnumVisitor.processEnumFieldByWrappedInsn(EnumVisitor.java:333)
	at jadx.core.dex.visitors.EnumVisitor.extractEnumFieldsFromFilledArray(EnumVisitor.java:318)
	at jadx.core.dex.visitors.EnumVisitor.extractEnumFieldsFromInsn(EnumVisitor.java:258)
	at jadx.core.dex.visitors.EnumVisitor.convertToEnum(EnumVisitor.java:151)
	at jadx.core.dex.visitors.EnumVisitor.visit(EnumVisitor.java:100)
 */
/* JADX WARN: Failed to restore enum class, 'enum' modifier and super class removed */
/* compiled from: com.google.android.gms:play-services-measurement-base@@21.2.0 */
/* loaded from: classes.dex */
public final class zzjx {
    public static final zzjx zzA;
    public static final zzjx zzB;
    public static final zzjx zzC;
    public static final zzjx zzD;
    public static final zzjx zzE;
    public static final zzjx zzF;
    public static final zzjx zzG;
    public static final zzjx zzH;
    public static final zzjx zzI;
    public static final zzjx zzJ;
    public static final zzjx zzK;
    public static final zzjx zzL;
    public static final zzjx zzM;
    public static final zzjx zzN;
    public static final zzjx zzO;
    public static final zzjx zzP;
    public static final zzjx zzQ;
    public static final zzjx zzR;
    public static final zzjx zzS;
    public static final zzjx zzT;
    public static final zzjx zzU;
    public static final zzjx zzV;
    public static final zzjx zzW;
    public static final zzjx zzX;
    public static final zzjx zzY;
    private static final zzjx[] zzZ;
    public static final zzjx zza;
    private static final /* synthetic */ zzjx[] zzaa;
    public static final zzjx zzb;
    public static final zzjx zzc;
    public static final zzjx zzd;
    public static final zzjx zze;
    public static final zzjx zzf;
    public static final zzjx zzg;
    public static final zzjx zzh;
    public static final zzjx zzi;
    public static final zzjx zzj;
    public static final zzjx zzk;
    public static final zzjx zzl;
    public static final zzjx zzm;
    public static final zzjx zzn;
    public static final zzjx zzo;
    public static final zzjx zzp;
    public static final zzjx zzq;
    public static final zzjx zzr;
    public static final zzjx zzs;
    public static final zzjx zzt;
    public static final zzjx zzu;
    public static final zzjx zzv;
    public static final zzjx zzw;
    public static final zzjx zzx;
    public static final zzjx zzy;
    public static final zzjx zzz;
    private final zzkq zzab;
    private final int zzac;
    private final Class zzad;

    static {
        zzkq zzkqVar = zzkq.zze;
        zza = new zzjx("DOUBLE", 0, 0, 1, zzkqVar);
        zzkq zzkqVar2 = zzkq.zzd;
        zzb = new zzjx("FLOAT", 1, 1, 1, zzkqVar2);
        zzkq zzkqVar3 = zzkq.zzc;
        zzc = new zzjx("INT64", 2, 2, 1, zzkqVar3);
        zzd = new zzjx("UINT64", 3, 3, 1, zzkqVar3);
        zzkq zzkqVar4 = zzkq.zzb;
        zze = new zzjx("INT32", 4, 4, 1, zzkqVar4);
        zzf = new zzjx("FIXED64", 5, 5, 1, zzkqVar3);
        zzg = new zzjx("FIXED32", 6, 6, 1, zzkqVar4);
        zzkq zzkqVar5 = zzkq.zzf;
        zzh = new zzjx("BOOL", 7, 7, 1, zzkqVar5);
        zzkq zzkqVar6 = zzkq.zzg;
        zzi = new zzjx("STRING", 8, 8, 1, zzkqVar6);
        zzkq zzkqVar7 = zzkq.zzj;
        zzj = new zzjx("MESSAGE", 9, 9, 1, zzkqVar7);
        zzkq zzkqVar8 = zzkq.zzh;
        zzk = new zzjx("BYTES", 10, 10, 1, zzkqVar8);
        zzl = new zzjx("UINT32", 11, 11, 1, zzkqVar4);
        zzkq zzkqVar9 = zzkq.zzi;
        zzm = new zzjx("ENUM", 12, 12, 1, zzkqVar9);
        zzn = new zzjx("SFIXED32", 13, 13, 1, zzkqVar4);
        zzo = new zzjx("SFIXED64", 14, 14, 1, zzkqVar3);
        zzp = new zzjx("SINT32", 15, 15, 1, zzkqVar4);
        zzq = new zzjx("SINT64", 16, 16, 1, zzkqVar3);
        zzr = new zzjx("GROUP", 17, 17, 1, zzkqVar7);
        zzs = new zzjx("DOUBLE_LIST", 18, 18, 2, zzkqVar);
        zzt = new zzjx("FLOAT_LIST", 19, 19, 2, zzkqVar2);
        zzu = new zzjx("INT64_LIST", 20, 20, 2, zzkqVar3);
        zzv = new zzjx("UINT64_LIST", 21, 21, 2, zzkqVar3);
        zzw = new zzjx("INT32_LIST", 22, 22, 2, zzkqVar4);
        zzx = new zzjx("FIXED64_LIST", 23, 23, 2, zzkqVar3);
        zzy = new zzjx("FIXED32_LIST", 24, 24, 2, zzkqVar4);
        zzz = new zzjx("BOOL_LIST", 25, 25, 2, zzkqVar5);
        zzA = new zzjx("STRING_LIST", 26, 26, 2, zzkqVar6);
        zzB = new zzjx("MESSAGE_LIST", 27, 27, 2, zzkqVar7);
        zzC = new zzjx("BYTES_LIST", 28, 28, 2, zzkqVar8);
        zzD = new zzjx("UINT32_LIST", 29, 29, 2, zzkqVar4);
        zzE = new zzjx("ENUM_LIST", 30, 30, 2, zzkqVar9);
        zzF = new zzjx("SFIXED32_LIST", 31, 31, 2, zzkqVar4);
        zzG = new zzjx("SFIXED64_LIST", 32, 32, 2, zzkqVar3);
        zzH = new zzjx("SINT32_LIST", 33, 33, 2, zzkqVar4);
        zzI = new zzjx("SINT64_LIST", 34, 34, 2, zzkqVar3);
        zzJ = new zzjx("DOUBLE_LIST_PACKED", 35, 35, 3, zzkqVar);
        zzK = new zzjx("FLOAT_LIST_PACKED", 36, 36, 3, zzkqVar2);
        zzL = new zzjx("INT64_LIST_PACKED", 37, 37, 3, zzkqVar3);
        zzM = new zzjx("UINT64_LIST_PACKED", 38, 38, 3, zzkqVar3);
        zzN = new zzjx("INT32_LIST_PACKED", 39, 39, 3, zzkqVar4);
        zzO = new zzjx("FIXED64_LIST_PACKED", 40, 40, 3, zzkqVar3);
        zzP = new zzjx("FIXED32_LIST_PACKED", 41, 41, 3, zzkqVar4);
        zzQ = new zzjx("BOOL_LIST_PACKED", 42, 42, 3, zzkqVar5);
        zzR = new zzjx("UINT32_LIST_PACKED", 43, 43, 3, zzkqVar4);
        zzS = new zzjx("ENUM_LIST_PACKED", 44, 44, 3, zzkqVar9);
        zzT = new zzjx("SFIXED32_LIST_PACKED", 45, 45, 3, zzkqVar4);
        zzkq zzkqVar10 = zzkq.zzc;
        zzU = new zzjx("SFIXED64_LIST_PACKED", 46, 46, 3, zzkqVar10);
        zzV = new zzjx("SINT32_LIST_PACKED", 47, 47, 3, zzkqVar4);
        zzW = new zzjx("SINT64_LIST_PACKED", 48, 48, 3, zzkqVar10);
        zzX = new zzjx("GROUP_LIST", 49, 49, 2, zzkqVar7);
        zzY = new zzjx("MAP", 50, 50, 4, zzkq.zza);
        zzaa = new zzjx[]{zza, zzb, zzc, zzd, zze, zzf, zzg, zzh, zzi, zzj, zzk, zzl, zzm, zzn, zzo, zzp, zzq, zzr, zzs, zzt, zzu, zzv, zzw, zzx, zzy, zzz, zzA, zzB, zzC, zzD, zzE, zzF, zzG, zzH, zzI, zzJ, zzK, zzL, zzM, zzN, zzO, zzP, zzQ, zzR, zzS, zzT, zzU, zzV, zzW, zzX, zzY};
        zzjx[] values = values();
        zzZ = new zzjx[51];
        for (int i = 0; i < 51; i++) {
            zzjx zzjxVar = values[i];
            zzZ[zzjxVar.zzac] = zzjxVar;
        }
    }

    private zzjx(String str, int i, int i2, int i3, zzkq zzkqVar) {
        this.zzac = i2;
        this.zzab = zzkqVar;
        zzkq zzkqVar2 = zzkq.zza;
        int i4 = i3 - 1;
        if (i4 == 1) {
            this.zzad = zzkqVar.zza();
        } else if (i4 != 3) {
            this.zzad = null;
        } else {
            this.zzad = zzkqVar.zza();
        }
        if (i3 == 1) {
            zzkqVar.ordinal();
        }
    }

    public static zzjx[] values() {
        return (zzjx[]) zzaa.clone();
    }

    public final int zza() {
        return this.zzac;
    }
}