package com.google.android.filament;

import com.google.android.filament.Texture;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import java.nio.Buffer;
import java.nio.BufferOverflowException;
import java.nio.ReadOnlyBufferException;

/* loaded from: classes.dex */
public class Renderer {
    public static final int MIRROR_FRAME_FLAG_CLEAR = 4;
    public static final int MIRROR_FRAME_FLAG_COMMIT = 1;
    public static final int MIRROR_FRAME_FLAG_SET_PRESENTATION_TIME = 2;
    private ClearOptions mClearOptions;
    private DisplayInfo mDisplayInfo;
    private final Engine mEngine;
    private FrameRateOptions mFrameRateOptions;
    private long mNativeObject;

    /* loaded from: classes.dex */
    public static class ClearOptions {
        public float[] clearColor = {StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD};
        public boolean clear = false;
        public boolean discard = true;
    }

    /* loaded from: classes.dex */
    public static class DisplayInfo {
        public float refreshRate = 60.0f;
        public long presentationDeadlineNanos = 0;
        public long vsyncOffsetNanos = 0;
    }

    /* loaded from: classes.dex */
    public static class FrameRateOptions {
        public float interval = 0.016666668f;
        public float headRoomRatio = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        public float scaleRate = 0.125f;
        public int history = 9;
    }

    public Renderer(Engine engine, long j) {
        this.mEngine = engine;
        this.mNativeObject = j;
    }

    private static native boolean nBeginFrame(long j, long j2, long j3);

    private static native void nCopyFrame(long j, long j2, int i, int i2, int i3, int i4, int i5, int i6, int i7, int i8, int i9);

    private static native void nEndFrame(long j);

    private static native double nGetUserTime(long j);

    private static native int nReadPixels(long j, long j2, int i, int i2, int i3, int i4, Buffer buffer, int i5, int i6, int i7, int i8, int i9, int i10, int i11, Object obj, Runnable runnable);

    private static native int nReadPixelsEx(long j, long j2, long j3, int i, int i2, int i3, int i4, Buffer buffer, int i5, int i6, int i7, int i8, int i9, int i10, int i11, Object obj, Runnable runnable);

    private static native void nRender(long j, long j2);

    private static native void nResetUserTime(long j);

    private static native void nSetClearOptions(long j, float f2, float f3, float f4, float f5, boolean z, boolean z2);

    private static native void nSetDisplayInfo(long j, float f2, long j2, long j3);

    private static native void nSetFrameRateOptions(long j, float f2, float f3, float f4, int i);

    public boolean beginFrame(SwapChain swapChain, long j) {
        return nBeginFrame(getNativeObject(), swapChain.getNativeObject(), j);
    }

    public void clearNativeObject() {
        this.mNativeObject = 0L;
    }

    public void copyFrame(SwapChain swapChain, Viewport viewport, Viewport viewport2, int i) {
        nCopyFrame(getNativeObject(), swapChain.getNativeObject(), viewport.left, viewport.bottom, viewport.width, viewport.height, viewport2.left, viewport2.bottom, viewport2.width, viewport2.height, i);
    }

    public void endFrame() {
        nEndFrame(getNativeObject());
    }

    public ClearOptions getClearOptions() {
        if (this.mClearOptions == null) {
            this.mClearOptions = new ClearOptions();
        }
        return this.mClearOptions;
    }

    public DisplayInfo getDisplayInfo() {
        if (this.mDisplayInfo == null) {
            this.mDisplayInfo = new DisplayInfo();
        }
        return this.mDisplayInfo;
    }

    public Engine getEngine() {
        return this.mEngine;
    }

    public FrameRateOptions getFrameRateOptions() {
        if (this.mFrameRateOptions == null) {
            this.mFrameRateOptions = new FrameRateOptions();
        }
        return this.mFrameRateOptions;
    }

    public long getNativeObject() {
        long j = this.mNativeObject;
        if (j != 0) {
            return j;
        }
        throw new IllegalStateException("Calling method on destroyed Renderer");
    }

    public double getUserTime() {
        return nGetUserTime(getNativeObject());
    }

    @Deprecated
    public void mirrorFrame(SwapChain swapChain, Viewport viewport, Viewport viewport2, int i) {
        copyFrame(swapChain, viewport, viewport2, i);
    }

    public void readPixels(int i, int i2, int i3, int i4, Texture.PixelBufferDescriptor pixelBufferDescriptor) {
        if (!pixelBufferDescriptor.storage.isReadOnly()) {
            long nativeObject = getNativeObject();
            long nativeObject2 = this.mEngine.getNativeObject();
            Buffer buffer = pixelBufferDescriptor.storage;
            if (nReadPixels(nativeObject, nativeObject2, i, i2, i3, i4, buffer, buffer.remaining(), pixelBufferDescriptor.left, pixelBufferDescriptor.top, pixelBufferDescriptor.type.ordinal(), pixelBufferDescriptor.alignment, pixelBufferDescriptor.stride, pixelBufferDescriptor.format.ordinal(), pixelBufferDescriptor.handler, pixelBufferDescriptor.callback) < 0) {
                throw new BufferOverflowException();
            }
            return;
        }
        throw new ReadOnlyBufferException();
    }

    public void render(View view) {
        nRender(getNativeObject(), view.getNativeObject());
    }

    public void resetUserTime() {
        nResetUserTime(getNativeObject());
    }

    public void setClearOptions(ClearOptions clearOptions) {
        this.mClearOptions = clearOptions;
        long nativeObject = getNativeObject();
        float[] fArr = clearOptions.clearColor;
        nSetClearOptions(nativeObject, fArr[0], fArr[1], fArr[2], fArr[3], clearOptions.clear, clearOptions.discard);
    }

    public void setDisplayInfo(DisplayInfo displayInfo) {
        this.mDisplayInfo = displayInfo;
        nSetDisplayInfo(getNativeObject(), displayInfo.refreshRate, displayInfo.presentationDeadlineNanos, displayInfo.vsyncOffsetNanos);
    }

    public void setFrameRateOptions(FrameRateOptions frameRateOptions) {
        this.mFrameRateOptions = frameRateOptions;
        nSetFrameRateOptions(getNativeObject(), frameRateOptions.interval, frameRateOptions.headRoomRatio, frameRateOptions.scaleRate, frameRateOptions.history);
    }

    public void readPixels(RenderTarget renderTarget, int i, int i2, int i3, int i4, Texture.PixelBufferDescriptor pixelBufferDescriptor) {
        if (!pixelBufferDescriptor.storage.isReadOnly()) {
            long nativeObject = getNativeObject();
            long nativeObject2 = this.mEngine.getNativeObject();
            long nativeObject3 = renderTarget.getNativeObject();
            Buffer buffer = pixelBufferDescriptor.storage;
            if (nReadPixelsEx(nativeObject, nativeObject2, nativeObject3, i, i2, i3, i4, buffer, buffer.remaining(), pixelBufferDescriptor.left, pixelBufferDescriptor.top, pixelBufferDescriptor.type.ordinal(), pixelBufferDescriptor.alignment, pixelBufferDescriptor.stride, pixelBufferDescriptor.format.ordinal(), pixelBufferDescriptor.handler, pixelBufferDescriptor.callback) < 0) {
                throw new BufferOverflowException();
            }
            return;
        }
        throw new ReadOnlyBufferException();
    }
}