package com.google.android.gms.internal.clearcut;

import java.lang.reflect.Type;

/* JADX WARN: Enum visitor error
jadx.core.utils.exceptions.JadxRuntimeException: Init of enum zzgy uses external variables
	at jadx.core.dex.visitors.EnumVisitor.createEnumFieldByConstructor(EnumVisitor.java:444)
	at jadx.core.dex.visitors.EnumVisitor.processEnumFieldByField(EnumVisitor.java:368)
	at jadx.core.dex.visitors.EnumVisitor.processEnumFieldByWrappedInsn(EnumVisitor.java:333)
	at jadx.core.dex.visitors.EnumVisitor.extractEnumFieldsFromFilledArray(EnumVisitor.java:318)
	at jadx.core.dex.visitors.EnumVisitor.extractEnumFieldsFromInsn(EnumVisitor.java:258)
	at jadx.core.dex.visitors.EnumVisitor.convertToEnum(EnumVisitor.java:151)
	at jadx.core.dex.visitors.EnumVisitor.visit(EnumVisitor.java:100)
 */
/* JADX WARN: Failed to restore enum class, 'enum' modifier and super class removed */
/* loaded from: classes.dex */
public final class zzcb {
    private static final zzcb zzgy;
    private static final zzcb zzgz;
    private static final zzcb zzha;
    private static final zzcb zzhb;
    private static final zzcb zzhc;
    private static final zzcb zzhd;
    private static final zzcb zzhe;
    private static final zzcb zzhf;
    private static final zzcb zzhg;
    public static final zzcb zzhh;
    private static final zzcb zzhi;
    private static final zzcb zzhj;
    public static final zzcb zzhk;
    private static final zzcb zzhl;
    private static final zzcb zzhm;
    private static final zzcb zzhn;
    private static final zzcb zzho;
    public static final zzcb zzhp;
    public static final zzcb zzhq;
    private static final zzcb zzhr;
    private static final zzcb zzhs;
    private static final zzcb zzht;
    private static final zzcb zzhu;
    private static final zzcb zzhv;
    private static final zzcb zzhw;
    private static final zzcb zzhx;
    private static final zzcb zzhy;
    public static final zzcb zzhz;
    private static final zzcb zzia;
    private static final zzcb zzib;
    public static final zzcb zzic;
    private static final zzcb zzid;
    private static final zzcb zzie;
    private static final zzcb zzif;
    private static final zzcb zzig;
    public static final zzcb zzih;
    private static final zzcb zzii;
    private static final zzcb zzij;
    private static final zzcb zzik;
    private static final zzcb zzil;
    private static final zzcb zzim;
    private static final zzcb zzin;
    private static final zzcb zzio;
    private static final zzcb zzip;
    public static final zzcb zziq;
    private static final zzcb zzir;
    private static final zzcb zzis;
    private static final zzcb zzit;
    public static final zzcb zziu;
    public static final zzcb zziv;
    public static final zzcb zziw;
    private static final zzcb[] zzjb;
    private static final Type[] zzjc;
    private static final /* synthetic */ zzcb[] zzjd;
    private final int id;
    private final zzcq zzix;
    private final zzcd zziy;
    private final Class<?> zziz;
    private final boolean zzja;

    static {
        zzcd zzcdVar = zzcd.SCALAR;
        zzcq zzcqVar = zzcq.zzlb;
        zzgy = new zzcb("DOUBLE", 0, 0, zzcdVar, zzcqVar);
        zzcq zzcqVar2 = zzcq.zzla;
        zzgz = new zzcb("FLOAT", 1, 1, zzcdVar, zzcqVar2);
        zzcq zzcqVar3 = zzcq.zzkz;
        zzha = new zzcb("INT64", 2, 2, zzcdVar, zzcqVar3);
        zzhb = new zzcb("UINT64", 3, 3, zzcdVar, zzcqVar3);
        zzcq zzcqVar4 = zzcq.zzky;
        zzhc = new zzcb("INT32", 4, 4, zzcdVar, zzcqVar4);
        zzhd = new zzcb("FIXED64", 5, 5, zzcdVar, zzcqVar3);
        zzhe = new zzcb("FIXED32", 6, 6, zzcdVar, zzcqVar4);
        zzcq zzcqVar5 = zzcq.zzlc;
        zzhf = new zzcb("BOOL", 7, 7, zzcdVar, zzcqVar5);
        zzcq zzcqVar6 = zzcq.zzld;
        zzhg = new zzcb("STRING", 8, 8, zzcdVar, zzcqVar6);
        zzcq zzcqVar7 = zzcq.zzlg;
        zzhh = new zzcb("MESSAGE", 9, 9, zzcdVar, zzcqVar7);
        zzcq zzcqVar8 = zzcq.zzle;
        zzhi = new zzcb("BYTES", 10, 10, zzcdVar, zzcqVar8);
        zzhj = new zzcb("UINT32", 11, 11, zzcdVar, zzcqVar4);
        zzcq zzcqVar9 = zzcq.zzlf;
        zzhk = new zzcb("ENUM", 12, 12, zzcdVar, zzcqVar9);
        zzhl = new zzcb("SFIXED32", 13, 13, zzcdVar, zzcqVar4);
        zzhm = new zzcb("SFIXED64", 14, 14, zzcdVar, zzcqVar3);
        zzhn = new zzcb("SINT32", 15, 15, zzcdVar, zzcqVar4);
        zzho = new zzcb("SINT64", 16, 16, zzcdVar, zzcqVar3);
        zzhp = new zzcb("GROUP", 17, 17, zzcdVar, zzcqVar7);
        zzcd zzcdVar2 = zzcd.VECTOR;
        zzhq = new zzcb("DOUBLE_LIST", 18, 18, zzcdVar2, zzcqVar);
        zzhr = new zzcb("FLOAT_LIST", 19, 19, zzcdVar2, zzcqVar2);
        zzhs = new zzcb("INT64_LIST", 20, 20, zzcdVar2, zzcqVar3);
        zzht = new zzcb("UINT64_LIST", 21, 21, zzcdVar2, zzcqVar3);
        zzhu = new zzcb("INT32_LIST", 22, 22, zzcdVar2, zzcqVar4);
        zzhv = new zzcb("FIXED64_LIST", 23, 23, zzcdVar2, zzcqVar3);
        zzhw = new zzcb("FIXED32_LIST", 24, 24, zzcdVar2, zzcqVar4);
        zzhx = new zzcb("BOOL_LIST", 25, 25, zzcdVar2, zzcqVar5);
        zzhy = new zzcb("STRING_LIST", 26, 26, zzcdVar2, zzcqVar6);
        zzhz = new zzcb("MESSAGE_LIST", 27, 27, zzcdVar2, zzcqVar7);
        zzia = new zzcb("BYTES_LIST", 28, 28, zzcdVar2, zzcqVar8);
        zzib = new zzcb("UINT32_LIST", 29, 29, zzcdVar2, zzcqVar4);
        zzic = new zzcb("ENUM_LIST", 30, 30, zzcdVar2, zzcqVar9);
        zzid = new zzcb("SFIXED32_LIST", 31, 31, zzcdVar2, zzcqVar4);
        zzie = new zzcb("SFIXED64_LIST", 32, 32, zzcdVar2, zzcqVar3);
        zzif = new zzcb("SINT32_LIST", 33, 33, zzcdVar2, zzcqVar4);
        zzig = new zzcb("SINT64_LIST", 34, 34, zzcdVar2, zzcqVar3);
        zzcd zzcdVar3 = zzcd.PACKED_VECTOR;
        zzih = new zzcb("DOUBLE_LIST_PACKED", 35, 35, zzcdVar3, zzcqVar);
        zzii = new zzcb("FLOAT_LIST_PACKED", 36, 36, zzcdVar3, zzcqVar2);
        zzij = new zzcb("INT64_LIST_PACKED", 37, 37, zzcdVar3, zzcqVar3);
        zzik = new zzcb("UINT64_LIST_PACKED", 38, 38, zzcdVar3, zzcqVar3);
        zzil = new zzcb("INT32_LIST_PACKED", 39, 39, zzcdVar3, zzcqVar4);
        zzim = new zzcb("FIXED64_LIST_PACKED", 40, 40, zzcdVar3, zzcqVar3);
        zzin = new zzcb("FIXED32_LIST_PACKED", 41, 41, zzcdVar3, zzcqVar4);
        zzio = new zzcb("BOOL_LIST_PACKED", 42, 42, zzcdVar3, zzcqVar5);
        zzip = new zzcb("UINT32_LIST_PACKED", 43, 43, zzcdVar3, zzcqVar4);
        zziq = new zzcb("ENUM_LIST_PACKED", 44, 44, zzcdVar3, zzcqVar9);
        zzir = new zzcb("SFIXED32_LIST_PACKED", 45, 45, zzcdVar3, zzcqVar4);
        zzcq zzcqVar10 = zzcq.zzkz;
        zzis = new zzcb("SFIXED64_LIST_PACKED", 46, 46, zzcdVar3, zzcqVar10);
        zzit = new zzcb("SINT32_LIST_PACKED", 47, 47, zzcdVar3, zzcq.zzky);
        zziu = new zzcb("SINT64_LIST_PACKED", 48, 48, zzcdVar3, zzcqVar10);
        zziv = new zzcb("GROUP_LIST", 49, 49, zzcdVar2, zzcqVar7);
        zziw = new zzcb("MAP", 50, 50, zzcd.MAP, zzcq.zzkx);
        zzjd = new zzcb[]{zzgy, zzgz, zzha, zzhb, zzhc, zzhd, zzhe, zzhf, zzhg, zzhh, zzhi, zzhj, zzhk, zzhl, zzhm, zzhn, zzho, zzhp, zzhq, zzhr, zzhs, zzht, zzhu, zzhv, zzhw, zzhx, zzhy, zzhz, zzia, zzib, zzic, zzid, zzie, zzif, zzig, zzih, zzii, zzij, zzik, zzil, zzim, zzin, zzio, zzip, zziq, zzir, zzis, zzit, zziu, zziv, zziw};
        zzjc = new Type[0];
        zzcb[] values = values();
        zzjb = new zzcb[51];
        for (int i = 0; i < 51; i++) {
            zzcb zzcbVar = values[i];
            zzjb[zzcbVar.id] = zzcbVar;
        }
    }

    private zzcb(String str, int i, int i2, zzcd zzcdVar, zzcq zzcqVar) {
        int i3;
        this.id = i2;
        this.zziy = zzcdVar;
        this.zzix = zzcqVar;
        int i4 = zzcc.zzje[zzcdVar.ordinal()];
        boolean z = true;
        this.zziz = (i4 == 1 || i4 == 2) ? zzcqVar.zzbq() : null;
        this.zzja = (zzcdVar != zzcd.SCALAR || (i3 = zzcc.zzjf[zzcqVar.ordinal()]) == 1 || i3 == 2 || i3 == 3) ? false : z;
    }

    public static zzcb[] values() {
        return (zzcb[]) zzjd.clone();
    }

    public final int id() {
        return this.id;
    }
}