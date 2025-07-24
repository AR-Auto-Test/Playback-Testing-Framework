package com.google.ar.sceneform.lullmodel;

/* loaded from: classes.dex */
public final class TextureTargetType {
    public static final short CubeMap = 1;
    public static final short Standard2d = 0;
    public static final String[] names = {"Standard2d", "CubeMap"};

    private TextureTargetType() {
    }

    public static String name(int i) {
        return names[i];
    }
}