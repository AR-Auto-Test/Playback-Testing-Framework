package com.google.android.filament;

/* loaded from: classes.dex */
public class SwapChain {
    public static final long CONFIG_DEFAULT = 0;
    public static final long CONFIG_READABLE = 2;
    public static final long CONFIG_TRANSPARENT = 1;
    private long mNativeObject;
    private final Object mSurface;

    public SwapChain(long j, Object obj) {
        this.mNativeObject = j;
        this.mSurface = obj;
    }

    public void clearNativeObject() {
        this.mNativeObject = 0L;
    }

    public long getNativeObject() {
        long j = this.mNativeObject;
        if (j != 0) {
            return j;
        }
        throw new IllegalStateException("Calling method on destroyed SwapChain");
    }

    public Object getNativeWindow() {
        return this.mSurface;
    }
}