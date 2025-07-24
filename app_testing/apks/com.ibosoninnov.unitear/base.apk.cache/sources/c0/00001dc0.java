package com.google.ar.schemas.motive;

/* loaded from: classes.dex */
public final class AnimSourceUnion {
    public static final byte AnimSourceEmbedded = 2;
    public static final byte AnimSourceFileName = 1;
    public static final byte NONE = 0;
    public static final String[] names = {"NONE", "AnimSourceFileName", "AnimSourceEmbedded"};

    private AnimSourceUnion() {
    }

    public static String name(int i) {
        return names[i];
    }
}