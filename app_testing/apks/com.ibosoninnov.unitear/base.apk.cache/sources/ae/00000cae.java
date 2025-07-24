package com.google.android.filament;

/* loaded from: classes.dex */
public class Filament {
    static {
        Platform.get();
        System.loadLibrary("filament-jni");
    }

    private Filament() {
    }

    public static void init() {
    }
}