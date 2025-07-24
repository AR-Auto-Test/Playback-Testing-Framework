package com.google.ar.sceneform.lullmodel;

/* loaded from: classes.dex */
public final class LayoutHorizontalAlignment {
    public static final int Center = 1;
    public static final int Left = 0;
    public static final int Right = 2;
    public static final String[] names = {"Left", "Center", "Right"};

    private LayoutHorizontalAlignment() {
    }

    public static String name(int i) {
        return names[i];
    }
}