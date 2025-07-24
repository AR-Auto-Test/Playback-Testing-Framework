package com.google.android.filament;

import c.b.a.a.a;
import com.google.android.filament.Texture;
import java.nio.Buffer;
import java.nio.BufferOverflowException;
import java.nio.ReadOnlyBufferException;

/* loaded from: classes.dex */
public class Stream {
    private long mNativeEngine;
    private long mNativeObject;

    /* loaded from: classes.dex */
    public enum StreamType {
        NATIVE,
        TEXTURE_ID,
        ACQUIRED
    }

    public Stream(long j, Engine engine) {
        this.mNativeObject = j;
        this.mNativeEngine = engine.getNativeObject();
    }

    /* JADX INFO: Access modifiers changed from: private */
    public static native long nBuilderBuild(long j, long j2);

    /* JADX INFO: Access modifiers changed from: private */
    public static native void nBuilderHeight(long j, int i);

    /* JADX INFO: Access modifiers changed from: private */
    public static native void nBuilderStream(long j, long j2);

    /* JADX INFO: Access modifiers changed from: private */
    public static native void nBuilderStreamSource(long j, Object obj);

    /* JADX INFO: Access modifiers changed from: private */
    public static native void nBuilderWidth(long j, int i);

    /* JADX INFO: Access modifiers changed from: private */
    public static native long nCreateBuilder();

    /* JADX INFO: Access modifiers changed from: private */
    public static native void nDestroyBuilder(long j);

    private static native int nGetStreamType(long j);

    private static native long nGetTimestamp(long j);

    private static native int nReadPixels(long j, long j2, int i, int i2, int i3, int i4, Buffer buffer, int i5, int i6, int i7, int i8, int i9, int i10, int i11, Object obj, Runnable runnable);

    private static native void nSetAcquiredImage(long j, long j2, Object obj, Object obj2, Runnable runnable);

    private static native void nSetDimensions(long j, int i, int i2);

    public void clearNativeObject() {
        this.mNativeObject = 0L;
    }

    public long getNativeObject() {
        long j = this.mNativeObject;
        if (j != 0) {
            return j;
        }
        throw new IllegalStateException("Calling method on destroyed Stream");
    }

    public StreamType getStreamType() {
        return StreamType.values()[nGetStreamType(getNativeObject())];
    }

    public long getTimestamp() {
        return nGetTimestamp(getNativeObject());
    }

    public void readPixels(int i, int i2, int i3, int i4, Texture.PixelBufferDescriptor pixelBufferDescriptor) {
        if (!pixelBufferDescriptor.storage.isReadOnly()) {
            long nativeObject = getNativeObject();
            long j = this.mNativeEngine;
            Buffer buffer = pixelBufferDescriptor.storage;
            if (nReadPixels(nativeObject, j, i, i2, i3, i4, buffer, buffer.remaining(), pixelBufferDescriptor.left, pixelBufferDescriptor.top, pixelBufferDescriptor.type.ordinal(), pixelBufferDescriptor.alignment, pixelBufferDescriptor.stride, pixelBufferDescriptor.format.ordinal(), pixelBufferDescriptor.handler, pixelBufferDescriptor.callback) < 0) {
                throw new BufferOverflowException();
            }
            return;
        }
        throw new ReadOnlyBufferException();
    }

    public void setAcquiredImage(Object obj, Object obj2, Runnable runnable) {
        nSetAcquiredImage(getNativeObject(), this.mNativeEngine, obj, obj2, runnable);
    }

    public void setDimensions(int i, int i2) {
        nSetDimensions(getNativeObject(), i, i2);
    }

    /* loaded from: classes.dex */
    public static class Builder {
        private final BuilderFinalizer mFinalizer;
        private final long mNativeBuilder;

        /* loaded from: classes.dex */
        public static class BuilderFinalizer {
            private final long mNativeObject;

            public BuilderFinalizer(long j) {
                this.mNativeObject = j;
            }

            public void finalize() {
                try {
                    super.finalize();
                } catch (Throwable unused) {
                }
                Stream.nDestroyBuilder(this.mNativeObject);
            }
        }

        public Builder() {
            long nCreateBuilder = Stream.nCreateBuilder();
            this.mNativeBuilder = nCreateBuilder;
            this.mFinalizer = new BuilderFinalizer(nCreateBuilder);
        }

        public Stream build(Engine engine) {
            long nBuilderBuild = Stream.nBuilderBuild(this.mNativeBuilder, engine.getNativeObject());
            if (nBuilderBuild != 0) {
                return new Stream(nBuilderBuild, engine);
            }
            throw new IllegalStateException("Couldn't create Stream");
        }

        public Builder height(int i) {
            Stream.nBuilderHeight(this.mNativeBuilder, i);
            return this;
        }

        public Builder stream(Object obj) {
            if (Platform.get().validateStreamSource(obj)) {
                Stream.nBuilderStreamSource(this.mNativeBuilder, obj);
                return this;
            }
            throw new IllegalArgumentException(a.p("Invalid stream source: ", obj));
        }

        public Builder width(int i) {
            Stream.nBuilderWidth(this.mNativeBuilder, i);
            return this;
        }

        public Builder stream(long j) {
            Stream.nBuilderStream(this.mNativeBuilder, j);
            return this;
        }
    }
}