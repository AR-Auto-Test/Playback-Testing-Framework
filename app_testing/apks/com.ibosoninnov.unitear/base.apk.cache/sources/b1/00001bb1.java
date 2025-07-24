package com.google.ar.core;

import com.google.ar.core.exceptions.FatalException;

/* loaded from: classes.dex */
public enum Coordinates2d {
    TEXTURE_TEXELS(0),
    TEXTURE_NORMALIZED(1),
    IMAGE_PIXELS(2),
    IMAGE_NORMALIZED(3),
    OPENGL_NORMALIZED_DEVICE_COORDINATES(6),
    VIEW(7),
    VIEW_NORMALIZED(8);
    
    public final int nativeCode;

    Coordinates2d(int i) {
        this.nativeCode = i;
    }

    public static Coordinates2d forNumber(int i) {
        Coordinates2d[] values = values();
        for (int i2 = 0; i2 < 7; i2++) {
            Coordinates2d coordinates2d = values[i2];
            if (coordinates2d.nativeCode == i) {
                return coordinates2d;
            }
        }
        throw new FatalException(c.b.a.a.a.g(60, "Unexpected value for native Coordinates2d, value=", i));
    }
}