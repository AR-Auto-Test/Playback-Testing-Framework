package com.google.ar.schemas.sceneform;

/* loaded from: classes.dex */
public final class SamplerCompareFunc {
    public static final short Always = 6;
    public static final short EqualTo = 4;
    public static final short GreaterEqual = 1;
    public static final short GreaterThan = 3;
    public static final short LessEqual = 0;
    public static final short LessThan = 2;
    public static final short Never = 7;
    public static final short NotEqual = 5;
    public static final String[] names = {"LessEqual", "GreaterEqual", "LessThan", "GreaterThan", "EqualTo", "NotEqual", "Always", "Never"};

    private SamplerCompareFunc() {
    }

    public static String name(int i) {
        return names[i];
    }
}