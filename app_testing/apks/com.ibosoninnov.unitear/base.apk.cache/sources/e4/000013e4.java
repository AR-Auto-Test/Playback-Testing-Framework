package com.google.android.gms.internal.vision;

import java.lang.reflect.Type;

/* JADX WARN: Enum visitor error
jadx.core.utils.exceptions.JadxRuntimeException: Init of enum zzc uses external variables
	at jadx.core.dex.visitors.EnumVisitor.createEnumFieldByConstructor(EnumVisitor.java:444)
	at jadx.core.dex.visitors.EnumVisitor.processEnumFieldByField(EnumVisitor.java:368)
	at jadx.core.dex.visitors.EnumVisitor.processEnumFieldByWrappedInsn(EnumVisitor.java:333)
	at jadx.core.dex.visitors.EnumVisitor.extractEnumFieldsFromFilledArray(EnumVisitor.java:318)
	at jadx.core.dex.visitors.EnumVisitor.extractEnumFieldsFromInsn(EnumVisitor.java:258)
	at jadx.core.dex.visitors.EnumVisitor.convertToEnum(EnumVisitor.java:151)
	at jadx.core.dex.visitors.EnumVisitor.visit(EnumVisitor.java:100)
 */
/* JADX WARN: Failed to restore enum class, 'enum' modifier and super class removed */
/* compiled from: com.google.android.gms:play-services-vision-common@@19.1.3 */
/* loaded from: classes.dex */
public final class zziv {
    public static final zziv zza;
    private static final zziv zzaa;
    private static final zziv zzab;
    private static final zziv zzac;
    private static final zziv zzad;
    private static final zziv zzae;
    private static final zziv zzaf;
    private static final zziv zzag;
    private static final zziv zzah;
    private static final zziv zzai;
    private static final zziv zzaj;
    private static final zziv zzak;
    private static final zziv zzal;
    private static final zziv zzam;
    private static final zziv zzan;
    private static final zziv zzao;
    private static final zziv zzap;
    private static final zziv zzaq;
    private static final zziv zzar;
    private static final zziv zzas;
    private static final zziv zzat;
    private static final zziv zzau;
    private static final zziv zzav;
    private static final zziv zzaw;
    private static final zziv zzax;
    private static final zziv zzay;
    public static final zziv zzb;
    private static final zziv[] zzbe;
    private static final Type[] zzbf;
    private static final /* synthetic */ zziv[] zzbg;
    private static final zziv zzc;
    private static final zziv zzd;
    private static final zziv zze;
    private static final zziv zzf;
    private static final zziv zzg;
    private static final zziv zzh;
    private static final zziv zzi;
    private static final zziv zzj;
    private static final zziv zzk;
    private static final zziv zzl;
    private static final zziv zzm;
    private static final zziv zzn;
    private static final zziv zzo;
    private static final zziv zzp;
    private static final zziv zzq;
    private static final zziv zzr;
    private static final zziv zzs;
    private static final zziv zzt;
    private static final zziv zzu;
    private static final zziv zzv;
    private static final zziv zzw;
    private static final zziv zzx;
    private static final zziv zzy;
    private static final zziv zzz;
    private final zzjm zzaz;
    private final int zzba;
    private final zzix zzbb;
    private final Class<?> zzbc;
    private final boolean zzbd;

    static {
        zzix zzixVar = zzix.SCALAR;
        zzjm zzjmVar = zzjm.zze;
        zzc = new zziv("DOUBLE", 0, 0, zzixVar, zzjmVar);
        zzjm zzjmVar2 = zzjm.zzd;
        zzd = new zziv("FLOAT", 1, 1, zzixVar, zzjmVar2);
        zzjm zzjmVar3 = zzjm.zzc;
        zze = new zziv("INT64", 2, 2, zzixVar, zzjmVar3);
        zzf = new zziv("UINT64", 3, 3, zzixVar, zzjmVar3);
        zzjm zzjmVar4 = zzjm.zzb;
        zzg = new zziv("INT32", 4, 4, zzixVar, zzjmVar4);
        zzh = new zziv("FIXED64", 5, 5, zzixVar, zzjmVar3);
        zzi = new zziv("FIXED32", 6, 6, zzixVar, zzjmVar4);
        zzjm zzjmVar5 = zzjm.zzf;
        zzj = new zziv("BOOL", 7, 7, zzixVar, zzjmVar5);
        zzjm zzjmVar6 = zzjm.zzg;
        zzk = new zziv("STRING", 8, 8, zzixVar, zzjmVar6);
        zzjm zzjmVar7 = zzjm.zzj;
        zzl = new zziv("MESSAGE", 9, 9, zzixVar, zzjmVar7);
        zzjm zzjmVar8 = zzjm.zzh;
        zzm = new zziv("BYTES", 10, 10, zzixVar, zzjmVar8);
        zzn = new zziv("UINT32", 11, 11, zzixVar, zzjmVar4);
        zzjm zzjmVar9 = zzjm.zzi;
        zzo = new zziv("ENUM", 12, 12, zzixVar, zzjmVar9);
        zzp = new zziv("SFIXED32", 13, 13, zzixVar, zzjmVar4);
        zzq = new zziv("SFIXED64", 14, 14, zzixVar, zzjmVar3);
        zzr = new zziv("SINT32", 15, 15, zzixVar, zzjmVar4);
        zzs = new zziv("SINT64", 16, 16, zzixVar, zzjmVar3);
        zzt = new zziv("GROUP", 17, 17, zzixVar, zzjmVar7);
        zzix zzixVar2 = zzix.VECTOR;
        zzu = new zziv("DOUBLE_LIST", 18, 18, zzixVar2, zzjmVar);
        zzv = new zziv("FLOAT_LIST", 19, 19, zzixVar2, zzjmVar2);
        zzw = new zziv("INT64_LIST", 20, 20, zzixVar2, zzjmVar3);
        zzx = new zziv("UINT64_LIST", 21, 21, zzixVar2, zzjmVar3);
        zzy = new zziv("INT32_LIST", 22, 22, zzixVar2, zzjmVar4);
        zzz = new zziv("FIXED64_LIST", 23, 23, zzixVar2, zzjmVar3);
        zzaa = new zziv("FIXED32_LIST", 24, 24, zzixVar2, zzjmVar4);
        zzab = new zziv("BOOL_LIST", 25, 25, zzixVar2, zzjmVar5);
        zzac = new zziv("STRING_LIST", 26, 26, zzixVar2, zzjmVar6);
        zzad = new zziv("MESSAGE_LIST", 27, 27, zzixVar2, zzjmVar7);
        zzae = new zziv("BYTES_LIST", 28, 28, zzixVar2, zzjmVar8);
        zzaf = new zziv("UINT32_LIST", 29, 29, zzixVar2, zzjmVar4);
        zzag = new zziv("ENUM_LIST", 30, 30, zzixVar2, zzjmVar9);
        zzah = new zziv("SFIXED32_LIST", 31, 31, zzixVar2, zzjmVar4);
        zzai = new zziv("SFIXED64_LIST", 32, 32, zzixVar2, zzjmVar3);
        zzaj = new zziv("SINT32_LIST", 33, 33, zzixVar2, zzjmVar4);
        zzak = new zziv("SINT64_LIST", 34, 34, zzixVar2, zzjmVar3);
        zzix zzixVar3 = zzix.PACKED_VECTOR;
        zza = new zziv("DOUBLE_LIST_PACKED", 35, 35, zzixVar3, zzjmVar);
        zzal = new zziv("FLOAT_LIST_PACKED", 36, 36, zzixVar3, zzjmVar2);
        zzam = new zziv("INT64_LIST_PACKED", 37, 37, zzixVar3, zzjmVar3);
        zzan = new zziv("UINT64_LIST_PACKED", 38, 38, zzixVar3, zzjmVar3);
        zzao = new zziv("INT32_LIST_PACKED", 39, 39, zzixVar3, zzjmVar4);
        zzap = new zziv("FIXED64_LIST_PACKED", 40, 40, zzixVar3, zzjmVar3);
        zzaq = new zziv("FIXED32_LIST_PACKED", 41, 41, zzixVar3, zzjmVar4);
        zzar = new zziv("BOOL_LIST_PACKED", 42, 42, zzixVar3, zzjmVar5);
        zzas = new zziv("UINT32_LIST_PACKED", 43, 43, zzixVar3, zzjmVar4);
        zzat = new zziv("ENUM_LIST_PACKED", 44, 44, zzixVar3, zzjmVar9);
        zzau = new zziv("SFIXED32_LIST_PACKED", 45, 45, zzixVar3, zzjmVar4);
        zzjm zzjmVar10 = zzjm.zzc;
        zzav = new zziv("SFIXED64_LIST_PACKED", 46, 46, zzixVar3, zzjmVar10);
        zzaw = new zziv("SINT32_LIST_PACKED", 47, 47, zzixVar3, zzjm.zzb);
        zzb = new zziv("SINT64_LIST_PACKED", 48, 48, zzixVar3, zzjmVar10);
        zzax = new zziv("GROUP_LIST", 49, 49, zzixVar2, zzjmVar7);
        zzay = new zziv("MAP", 50, 50, zzix.MAP, zzjm.zza);
        zzbg = new zziv[]{zzc, zzd, zze, zzf, zzg, zzh, zzi, zzj, zzk, zzl, zzm, zzn, zzo, zzp, zzq, zzr, zzs, zzt, zzu, zzv, zzw, zzx, zzy, zzz, zzaa, zzab, zzac, zzad, zzae, zzaf, zzag, zzah, zzai, zzaj, zzak, zza, zzal, zzam, zzan, zzao, zzap, zzaq, zzar, zzas, zzat, zzau, zzav, zzaw, zzb, zzax, zzay};
        zzbf = new Type[0];
        zziv[] values = values();
        zzbe = new zziv[51];
        for (int i = 0; i < 51; i++) {
            zziv zzivVar = values[i];
            zzbe[zzivVar.zzba] = zzivVar;
        }
    }

    private zziv(String str, int i, int i2, zzix zzixVar, zzjm zzjmVar) {
        int i3;
        this.zzba = i2;
        this.zzbb = zzixVar;
        this.zzaz = zzjmVar;
        int i4 = zziy.zza[zzixVar.ordinal()];
        boolean z = true;
        if (i4 == 1) {
            this.zzbc = zzjmVar.zza();
        } else if (i4 != 2) {
            this.zzbc = null;
        } else {
            this.zzbc = zzjmVar.zza();
        }
        this.zzbd = (zzixVar != zzix.SCALAR || (i3 = zziy.zzb[zzjmVar.ordinal()]) == 1 || i3 == 2 || i3 == 3) ? false : z;
    }

    public static zziv[] values() {
        return (zziv[]) zzbg.clone();
    }

    public final int zza() {
        return this.zzba;
    }
}