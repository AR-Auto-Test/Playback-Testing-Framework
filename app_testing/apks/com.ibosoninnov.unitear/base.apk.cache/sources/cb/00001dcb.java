package com.google.ar.schemas.motive;

/* loaded from: classes.dex */
public final class MatrixOperationTypeFb {
    public static final byte kInvalidMatrixOperation = 0;
    public static final byte kQuaternionW = 11;
    public static final byte kQuaternionX = 12;
    public static final byte kQuaternionY = 13;
    public static final byte kQuaternionZ = 14;
    public static final byte kRotateAboutX = 1;
    public static final byte kRotateAboutY = 2;
    public static final byte kRotateAboutZ = 3;
    public static final byte kScaleUniformly = 10;
    public static final byte kScaleX = 7;
    public static final byte kScaleY = 8;
    public static final byte kScaleZ = 9;
    public static final byte kTranslateX = 4;
    public static final byte kTranslateY = 5;
    public static final byte kTranslateZ = 6;
    public static final String[] names = {"kInvalidMatrixOperation", "kRotateAboutX", "kRotateAboutY", "kRotateAboutZ", "kTranslateX", "kTranslateY", "kTranslateZ", "kScaleX", "kScaleY", "kScaleZ", "kScaleUniformly", "kQuaternionW", "kQuaternionX", "kQuaternionY", "kQuaternionZ"};

    private MatrixOperationTypeFb() {
    }

    public static String name(int i) {
        return names[i];
    }
}