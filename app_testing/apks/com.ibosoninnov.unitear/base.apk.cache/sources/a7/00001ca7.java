package com.google.ar.sceneform.lullmodel;

/* loaded from: classes.dex */
public final class TextureWrap {
    public static final short ClampToBorder = 0;
    public static final short ClampToEdge = 1;
    public static final short MirrorClampToEdge = 3;
    public static final short MirroredRepeat = 2;
    public static final short Repeat = 4;
    public static final String[] names = {"ClampToBorder", "ClampToEdge", "MirroredRepeat", "MirrorClampToEdge", "Repeat"};

    private TextureWrap() {
    }

    public static String name(int i) {
        return names[i];
    }
}