package com.google.ar.schemas.motive;

/* loaded from: classes.dex */
public final class MatrixOpValueFb {
    public static final byte CompactSplineFb = 1;
    public static final byte CompactSplineFloatFb = 3;
    public static final byte ConstantOpFb = 2;
    public static final byte NONE = 0;
    public static final String[] names = {"NONE", "CompactSplineFb", "ConstantOpFb", "CompactSplineFloatFb"};

    private MatrixOpValueFb() {
    }

    public static String name(int i) {
        return names[i];
    }
}