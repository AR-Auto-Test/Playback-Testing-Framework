package com.google.android.gms.internal.clearcut;

/* loaded from: classes.dex */
public final /* synthetic */ class zzcc {
    public static final /* synthetic */ int[] zzje;
    public static final /* synthetic */ int[] zzjf;

    static {
        zzcq.values();
        int[] iArr = new int[10];
        zzjf = iArr;
        try {
            iArr[zzcq.zzle.ordinal()] = 1;
        } catch (NoSuchFieldError unused) {
        }
        try {
            zzjf[zzcq.zzlg.ordinal()] = 2;
        } catch (NoSuchFieldError unused2) {
        }
        try {
            zzjf[zzcq.zzld.ordinal()] = 3;
        } catch (NoSuchFieldError unused3) {
        }
        zzcd.values();
        int[] iArr2 = new int[4];
        zzje = iArr2;
        try {
            iArr2[zzcd.MAP.ordinal()] = 1;
        } catch (NoSuchFieldError unused4) {
        }
        try {
            zzje[zzcd.VECTOR.ordinal()] = 2;
        } catch (NoSuchFieldError unused5) {
        }
        try {
            zzje[zzcd.SCALAR.ordinal()] = 3;
        } catch (NoSuchFieldError unused6) {
        }
    }
}