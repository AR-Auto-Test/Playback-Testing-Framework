package com.google.android.filament;

import android.opengl.EGLContext;

/* loaded from: classes.dex */
public final class AndroidPlatform21 {
    public static long getSharedContextNativeHandle(Object obj) {
        return ((EGLContext) obj).getNativeHandle();
    }
}