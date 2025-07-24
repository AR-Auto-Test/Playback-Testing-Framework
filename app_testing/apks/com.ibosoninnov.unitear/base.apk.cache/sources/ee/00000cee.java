package com.google.android.filament;

import com.google.android.filament.proguard.UsedByReflection;
import java.nio.Buffer;
import java.nio.BufferOverflowException;
import java.nio.ByteBuffer;

/* loaded from: classes.dex */
public class Texture {
    public static final int BASE_LEVEL = 0;
    private long mNativeObject;

    /* renamed from: com.google.android.filament.Texture$1  reason: invalid class name */
    /* loaded from: classes.dex */
    public static /* synthetic */ class AnonymousClass1 {
        public static final /* synthetic */ int[] $SwitchMap$com$google$android$filament$Texture$Format;
        public static final /* synthetic */ int[] $SwitchMap$com$google$android$filament$Texture$Type;

        static {
            Type.values();
            int[] iArr = new int[11];
            $SwitchMap$com$google$android$filament$Texture$Type = iArr;
            try {
                iArr[Type.UBYTE.ordinal()] = 1;
            } catch (NoSuchFieldError unused) {
            }
            try {
                $SwitchMap$com$google$android$filament$Texture$Type[Type.BYTE.ordinal()] = 2;
            } catch (NoSuchFieldError unused2) {
            }
            try {
                $SwitchMap$com$google$android$filament$Texture$Type[Type.USHORT.ordinal()] = 3;
            } catch (NoSuchFieldError unused3) {
            }
            try {
                $SwitchMap$com$google$android$filament$Texture$Type[Type.SHORT.ordinal()] = 4;
            } catch (NoSuchFieldError unused4) {
            }
            try {
                $SwitchMap$com$google$android$filament$Texture$Type[Type.HALF.ordinal()] = 5;
            } catch (NoSuchFieldError unused5) {
            }
            try {
                $SwitchMap$com$google$android$filament$Texture$Type[Type.UINT.ordinal()] = 6;
            } catch (NoSuchFieldError unused6) {
            }
            try {
                $SwitchMap$com$google$android$filament$Texture$Type[Type.INT.ordinal()] = 7;
            } catch (NoSuchFieldError unused7) {
            }
            try {
                $SwitchMap$com$google$android$filament$Texture$Type[Type.FLOAT.ordinal()] = 8;
            } catch (NoSuchFieldError unused8) {
            }
            try {
                $SwitchMap$com$google$android$filament$Texture$Type[Type.UINT_10F_11F_11F_REV.ordinal()] = 9;
            } catch (NoSuchFieldError unused9) {
            }
            Format.values();
            int[] iArr2 = new int[13];
            $SwitchMap$com$google$android$filament$Texture$Format = iArr2;
            try {
                iArr2[Format.R.ordinal()] = 1;
            } catch (NoSuchFieldError unused10) {
            }
            try {
                $SwitchMap$com$google$android$filament$Texture$Format[Format.R_INTEGER.ordinal()] = 2;
            } catch (NoSuchFieldError unused11) {
            }
            try {
                $SwitchMap$com$google$android$filament$Texture$Format[Format.DEPTH_COMPONENT.ordinal()] = 3;
            } catch (NoSuchFieldError unused12) {
            }
            try {
                $SwitchMap$com$google$android$filament$Texture$Format[Format.ALPHA.ordinal()] = 4;
            } catch (NoSuchFieldError unused13) {
            }
            try {
                $SwitchMap$com$google$android$filament$Texture$Format[Format.RG.ordinal()] = 5;
            } catch (NoSuchFieldError unused14) {
            }
            try {
                $SwitchMap$com$google$android$filament$Texture$Format[Format.RG_INTEGER.ordinal()] = 6;
            } catch (NoSuchFieldError unused15) {
            }
            try {
                $SwitchMap$com$google$android$filament$Texture$Format[Format.DEPTH_STENCIL.ordinal()] = 7;
            } catch (NoSuchFieldError unused16) {
            }
            try {
                $SwitchMap$com$google$android$filament$Texture$Format[Format.STENCIL_INDEX.ordinal()] = 8;
            } catch (NoSuchFieldError unused17) {
            }
            try {
                $SwitchMap$com$google$android$filament$Texture$Format[Format.RGB.ordinal()] = 9;
            } catch (NoSuchFieldError unused18) {
            }
            try {
                $SwitchMap$com$google$android$filament$Texture$Format[Format.RGB_INTEGER.ordinal()] = 10;
            } catch (NoSuchFieldError unused19) {
            }
            try {
                $SwitchMap$com$google$android$filament$Texture$Format[Format.RGBA.ordinal()] = 11;
            } catch (NoSuchFieldError unused20) {
            }
            try {
                $SwitchMap$com$google$android$filament$Texture$Format[Format.RGBA_INTEGER.ordinal()] = 12;
            } catch (NoSuchFieldError unused21) {
            }
        }
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
                Texture.nDestroyBuilder(this.mNativeObject);
            }
        }

        public Builder() {
            long nCreateBuilder = Texture.nCreateBuilder();
            this.mNativeBuilder = nCreateBuilder;
            this.mFinalizer = new BuilderFinalizer(nCreateBuilder);
        }

        public Texture build(Engine engine) {
            long nBuilderBuild = Texture.nBuilderBuild(this.mNativeBuilder, engine.getNativeObject());
            if (nBuilderBuild != 0) {
                return new Texture(nBuilderBuild);
            }
            throw new IllegalStateException("Couldn't create Texture");
        }

        public Builder depth(int i) {
            Texture.nBuilderDepth(this.mNativeBuilder, i);
            return this;
        }

        public Builder format(InternalFormat internalFormat) {
            Texture.nBuilderFormat(this.mNativeBuilder, internalFormat.ordinal());
            return this;
        }

        public Builder height(int i) {
            Texture.nBuilderHeight(this.mNativeBuilder, i);
            return this;
        }

        public Builder levels(int i) {
            Texture.nBuilderLevels(this.mNativeBuilder, i);
            return this;
        }

        public Builder sampler(Sampler sampler) {
            Texture.nBuilderSampler(this.mNativeBuilder, sampler.ordinal());
            return this;
        }

        public Builder swizzle(Swizzle swizzle, Swizzle swizzle2, Swizzle swizzle3, Swizzle swizzle4) {
            Texture.nBuilderSwizzle(this.mNativeBuilder, swizzle.ordinal(), swizzle2.ordinal(), swizzle3.ordinal(), swizzle4.ordinal());
            return this;
        }

        public Builder usage(int i) {
            Texture.nBuilderUsage(this.mNativeBuilder, i);
            return this;
        }

        public Builder width(int i) {
            Texture.nBuilderWidth(this.mNativeBuilder, i);
            return this;
        }
    }

    /* loaded from: classes.dex */
    public enum CompressedFormat {
        EAC_R11,
        EAC_R11_SIGNED,
        EAC_RG11,
        EAC_RG11_SIGNED,
        ETC2_RGB8,
        ETC2_SRGB8,
        ETC2_RGB8_A1,
        ETC2_SRGB8_A1,
        ETC2_EAC_RGBA8,
        ETC2_EAC_SRGBA8,
        DXT1_RGB,
        DXT1_RGBA,
        DXT3_RGBA,
        DXT5_RGBA
    }

    /* loaded from: classes.dex */
    public enum CubemapFace {
        POSITIVE_X,
        NEGATIVE_X,
        POSITIVE_Y,
        NEGATIVE_Y,
        POSITIVE_Z,
        NEGATIVE_Z
    }

    /* loaded from: classes.dex */
    public enum Format {
        R,
        R_INTEGER,
        RG,
        RG_INTEGER,
        RGB,
        RGB_INTEGER,
        RGBA,
        RGBA_INTEGER,
        UNUSED,
        DEPTH_COMPONENT,
        DEPTH_STENCIL,
        STENCIL_INDEX,
        ALPHA
    }

    /* loaded from: classes.dex */
    public enum InternalFormat {
        R8,
        R8_SNORM,
        R8UI,
        R8I,
        STENCIL8,
        R16F,
        R16UI,
        R16I,
        RG8,
        RG8_SNORM,
        RG8UI,
        RG8I,
        RGB565,
        RGB9_E5,
        RGB5_A1,
        RGBA4,
        DEPTH16,
        RGB8,
        SRGB8,
        RGB8_SNORM,
        RGB8UI,
        RGB8I,
        DEPTH24,
        R32F,
        R32UI,
        R32I,
        RG16F,
        RG16UI,
        RG16I,
        R11F_G11F_B10F,
        RGBA8,
        SRGB8_A8,
        RGBA8_SNORM,
        UNUSED,
        RGB10_A2,
        RGBA8UI,
        RGBA8I,
        DEPTH32F,
        DEPTH24_STENCIL8,
        DEPTH32F_STENCIL8,
        RGB16F,
        RGB16UI,
        RGB16I,
        RG32F,
        RG32UI,
        RG32I,
        RGBA16F,
        RGBA16UI,
        RGBA16I,
        RGB32F,
        RGB32UI,
        RGB32I,
        RGBA32F,
        RGBA32UI,
        RGBA32I,
        EAC_R11,
        EAC_R11_SIGNED,
        EAC_RG11,
        EAC_RG11_SIGNED,
        ETC2_RGB8,
        ETC2_SRGB8,
        ETC2_RGB8_A1,
        ETC2_SRGB8_A1,
        ETC2_EAC_RGBA8,
        ETC2_EAC_SRGBA8,
        DXT1_RGB,
        DXT1_RGBA,
        DXT3_RGBA,
        DXT5_RGBA
    }

    /* loaded from: classes.dex */
    public static class PrefilterOptions {
        public int sampleCount = 8;
        public boolean mirror = true;
    }

    /* loaded from: classes.dex */
    public enum Sampler {
        SAMPLER_2D,
        SAMPLER_2D_ARRAY,
        SAMPLER_CUBEMAP,
        SAMPLER_EXTERNAL
    }

    /* loaded from: classes.dex */
    public enum Swizzle {
        SUBSTITUTE_ZERO,
        SUBSTITUTE_ONE,
        CHANNEL_0,
        CHANNEL_1,
        CHANNEL_2,
        CHANNEL_3
    }

    /* loaded from: classes.dex */
    public enum Type {
        UBYTE,
        BYTE,
        USHORT,
        SHORT,
        UINT,
        INT,
        HALF,
        FLOAT,
        COMPRESSED,
        UINT_10F_11F_11F_REV,
        USHORT_565
    }

    /* loaded from: classes.dex */
    public static class Usage {
        public static final int COLOR_ATTACHMENT = 1;
        public static final int DEFAULT = 24;
        public static final int DEPTH_ATTACHMENT = 2;
        public static final int SAMPLEABLE = 16;
        public static final int STENCIL_ATTACHMENT = 4;
        public static final int UPLOADABLE = 8;
    }

    public Texture(long j) {
        this.mNativeObject = j;
    }

    public static boolean isTextureFormatSupported(Engine engine, InternalFormat internalFormat) {
        return nIsTextureFormatSupported(engine.getNativeObject(), internalFormat.ordinal());
    }

    /* JADX INFO: Access modifiers changed from: private */
    public static native long nBuilderBuild(long j, long j2);

    /* JADX INFO: Access modifiers changed from: private */
    public static native void nBuilderDepth(long j, int i);

    /* JADX INFO: Access modifiers changed from: private */
    public static native void nBuilderFormat(long j, int i);

    /* JADX INFO: Access modifiers changed from: private */
    public static native void nBuilderHeight(long j, int i);

    /* JADX INFO: Access modifiers changed from: private */
    public static native void nBuilderLevels(long j, int i);

    /* JADX INFO: Access modifiers changed from: private */
    public static native void nBuilderSampler(long j, int i);

    /* JADX INFO: Access modifiers changed from: private */
    public static native void nBuilderSwizzle(long j, int i, int i2, int i3, int i4);

    /* JADX INFO: Access modifiers changed from: private */
    public static native void nBuilderUsage(long j, int i);

    /* JADX INFO: Access modifiers changed from: private */
    public static native void nBuilderWidth(long j, int i);

    /* JADX INFO: Access modifiers changed from: private */
    public static native long nCreateBuilder();

    /* JADX INFO: Access modifiers changed from: private */
    public static native void nDestroyBuilder(long j);

    private static native void nGenerateMipmaps(long j, long j2);

    private static native int nGeneratePrefilterMipmap(long j, long j2, int i, int i2, Buffer buffer, int i3, int i4, int i5, int i6, int i7, int i8, int i9, int[] iArr, Object obj, Runnable runnable, int i10, boolean z);

    private static native int nGetDepth(long j, int i);

    private static native int nGetHeight(long j, int i);

    private static native int nGetInternalFormat(long j);

    private static native int nGetLevels(long j);

    private static native int nGetTarget(long j);

    private static native int nGetWidth(long j, int i);

    private static native boolean nIsStreamValidForTexture(long j, long j2);

    private static native boolean nIsTextureFormatSupported(long j, int i);

    private static native void nSetExternalImage(long j, long j2, long j3);

    private static native void nSetExternalStream(long j, long j2, long j3);

    private static native int nSetImage(long j, long j2, int i, int i2, int i3, int i4, int i5, Buffer buffer, int i6, int i7, int i8, int i9, int i10, int i11, int i12, Object obj, Runnable runnable);

    private static native int nSetImageCompressed(long j, long j2, int i, int i2, int i3, int i4, int i5, Buffer buffer, int i6, int i7, int i8, int i9, int i10, int i11, int i12, Object obj, Runnable runnable);

    private static native int nSetImageCubemap(long j, long j2, int i, Buffer buffer, int i2, int i3, int i4, int i5, int i6, int i7, int i8, int[] iArr, Object obj, Runnable runnable);

    private static native int nSetImageCubemapCompressed(long j, long j2, int i, Buffer buffer, int i2, int i3, int i4, int i5, int i6, int i7, int i8, int[] iArr, Object obj, Runnable runnable);

    public void clearNativeObject() {
        this.mNativeObject = 0L;
    }

    public void generateMipmaps(Engine engine) {
        nGenerateMipmaps(getNativeObject(), engine.getNativeObject());
    }

    public void generatePrefilterMipmap(Engine engine, PixelBufferDescriptor pixelBufferDescriptor, int[] iArr, PrefilterOptions prefilterOptions) {
        int i;
        boolean z;
        int width = getWidth(0);
        int height = getHeight(0);
        if (prefilterOptions != null) {
            int i2 = prefilterOptions.sampleCount;
            z = prefilterOptions.mirror;
            i = i2;
        } else {
            i = 8;
            z = true;
        }
        long nativeObject = getNativeObject();
        long nativeObject2 = engine.getNativeObject();
        Buffer buffer = pixelBufferDescriptor.storage;
        if (nGeneratePrefilterMipmap(nativeObject, nativeObject2, width, height, buffer, buffer.remaining(), pixelBufferDescriptor.left, pixelBufferDescriptor.top, pixelBufferDescriptor.type.ordinal(), pixelBufferDescriptor.alignment, pixelBufferDescriptor.stride, pixelBufferDescriptor.format.ordinal(), iArr, pixelBufferDescriptor.handler, pixelBufferDescriptor.callback, i, z) < 0) {
            throw new BufferOverflowException();
        }
    }

    public int getDepth(int i) {
        return nGetDepth(getNativeObject(), i);
    }

    public InternalFormat getFormat() {
        return InternalFormat.values()[nGetInternalFormat(getNativeObject())];
    }

    public int getHeight(int i) {
        return nGetHeight(getNativeObject(), i);
    }

    public int getLevels() {
        return nGetLevels(getNativeObject());
    }

    @UsedByReflection("TextureHelper.java")
    public long getNativeObject() {
        long j = this.mNativeObject;
        if (j != 0) {
            return j;
        }
        throw new IllegalStateException("Calling method on destroyed Texture");
    }

    public Sampler getTarget() {
        return Sampler.values()[nGetTarget(getNativeObject())];
    }

    public int getWidth(int i) {
        return nGetWidth(getNativeObject(), i);
    }

    public void setExternalImage(Engine engine, long j) {
        nSetExternalImage(getNativeObject(), engine.getNativeObject(), j);
    }

    public void setExternalStream(Engine engine, Stream stream) {
        long nativeObject = getNativeObject();
        long nativeObject2 = stream.getNativeObject();
        if (nIsStreamValidForTexture(nativeObject, nativeObject2)) {
            nSetExternalStream(nativeObject, engine.getNativeObject(), nativeObject2);
            return;
        }
        throw new IllegalStateException("Invalid texture sampler: When used with a stream, a texture must use a SAMPLER_EXTERNAL");
    }

    public void setImage(Engine engine, int i, PixelBufferDescriptor pixelBufferDescriptor) {
        setImage(engine, i, 0, 0, getWidth(i), getHeight(i), pixelBufferDescriptor);
    }

    public void setImage(Engine engine, int i, int i2, int i3, int i4, int i5, PixelBufferDescriptor pixelBufferDescriptor) {
        int nSetImage;
        if (pixelBufferDescriptor.type == Type.COMPRESSED) {
            long nativeObject = getNativeObject();
            long nativeObject2 = engine.getNativeObject();
            Buffer buffer = pixelBufferDescriptor.storage;
            nSetImage = nSetImageCompressed(nativeObject, nativeObject2, i, i2, i3, i4, i5, buffer, buffer.remaining(), pixelBufferDescriptor.left, pixelBufferDescriptor.top, pixelBufferDescriptor.type.ordinal(), pixelBufferDescriptor.alignment, pixelBufferDescriptor.compressedSizeInBytes, pixelBufferDescriptor.compressedFormat.ordinal(), pixelBufferDescriptor.handler, pixelBufferDescriptor.callback);
        } else {
            long nativeObject3 = getNativeObject();
            long nativeObject4 = engine.getNativeObject();
            Buffer buffer2 = pixelBufferDescriptor.storage;
            nSetImage = nSetImage(nativeObject3, nativeObject4, i, i2, i3, i4, i5, buffer2, buffer2.remaining(), pixelBufferDescriptor.left, pixelBufferDescriptor.top, pixelBufferDescriptor.type.ordinal(), pixelBufferDescriptor.alignment, pixelBufferDescriptor.stride, pixelBufferDescriptor.format.ordinal(), pixelBufferDescriptor.handler, pixelBufferDescriptor.callback);
        }
        if (nSetImage < 0) {
            throw new BufferOverflowException();
        }
    }

    public Texture(Engine engine, long j) {
        this.mNativeObject = j;
    }

    /* loaded from: classes.dex */
    public static class PixelBufferDescriptor {
        public int alignment;
        public Runnable callback;
        public CompressedFormat compressedFormat;
        public int compressedSizeInBytes;
        public Format format;
        public Object handler;
        public int left;
        public Buffer storage;
        public int stride;
        public int top;
        public Type type;

        public PixelBufferDescriptor(Buffer buffer, Format format, Type type, int i, int i2, int i3, int i4, Object obj, Runnable runnable) {
            this.alignment = 1;
            this.left = 0;
            this.top = 0;
            this.stride = 0;
            this.storage = buffer;
            this.left = i2;
            this.top = i3;
            this.type = type;
            this.alignment = i;
            this.stride = i4;
            this.format = format;
            this.handler = obj;
            this.callback = runnable;
        }

        public static int computeDataSize(Format format, Type type, int i, int i2, int i3) {
            int i4 = 0;
            if (type == Type.COMPRESSED) {
                return 0;
            }
            int i5 = 4;
            switch (format.ordinal()) {
                case 0:
                case 1:
                case 9:
                case 12:
                    i4 = 1;
                    break;
                case 2:
                case 3:
                case 10:
                case 11:
                    i4 = 2;
                    break;
                case 4:
                case 5:
                    i4 = 3;
                    break;
                case 6:
                case 7:
                    i4 = 4;
                    break;
            }
            switch (type.ordinal()) {
                case 2:
                case 3:
                case 6:
                    i5 = i4 * 2;
                    break;
                case 4:
                case 5:
                case 7:
                    i5 = i4 * 4;
                    break;
                case 8:
                default:
                    i5 = i4;
                    break;
                case 9:
                    break;
            }
            return (((i3 - 1) + (i5 * i)) & (-i3)) * i2;
        }

        public void setCallback(Object obj, Runnable runnable) {
            this.handler = obj;
            this.callback = runnable;
        }

        public PixelBufferDescriptor(Buffer buffer, Format format, Type type) {
            this(buffer, format, type, 1, 0, 0, 0, null, null);
        }

        public PixelBufferDescriptor(Buffer buffer, Format format, Type type, int i) {
            this(buffer, format, type, i, 0, 0, 0, null, null);
        }

        public PixelBufferDescriptor(Buffer buffer, Format format, Type type, int i, int i2, int i3) {
            this(buffer, format, type, i, i2, i3, 0, null, null);
        }

        public PixelBufferDescriptor(ByteBuffer byteBuffer, CompressedFormat compressedFormat, int i) {
            this.alignment = 1;
            this.left = 0;
            this.top = 0;
            this.stride = 0;
            this.storage = byteBuffer;
            this.type = Type.COMPRESSED;
            this.alignment = 1;
            this.compressedFormat = compressedFormat;
            this.compressedSizeInBytes = i;
        }
    }

    public void setImage(Engine engine, int i, PixelBufferDescriptor pixelBufferDescriptor, int[] iArr) {
        int nSetImageCubemap;
        if (pixelBufferDescriptor.type == Type.COMPRESSED) {
            long nativeObject = getNativeObject();
            long nativeObject2 = engine.getNativeObject();
            Buffer buffer = pixelBufferDescriptor.storage;
            nSetImageCubemap = nSetImageCubemapCompressed(nativeObject, nativeObject2, i, buffer, buffer.remaining(), pixelBufferDescriptor.left, pixelBufferDescriptor.top, pixelBufferDescriptor.type.ordinal(), pixelBufferDescriptor.alignment, pixelBufferDescriptor.compressedSizeInBytes, pixelBufferDescriptor.compressedFormat.ordinal(), iArr, pixelBufferDescriptor.handler, pixelBufferDescriptor.callback);
        } else {
            long nativeObject3 = getNativeObject();
            long nativeObject4 = engine.getNativeObject();
            Buffer buffer2 = pixelBufferDescriptor.storage;
            nSetImageCubemap = nSetImageCubemap(nativeObject3, nativeObject4, i, buffer2, buffer2.remaining(), pixelBufferDescriptor.left, pixelBufferDescriptor.top, pixelBufferDescriptor.type.ordinal(), pixelBufferDescriptor.alignment, pixelBufferDescriptor.stride, pixelBufferDescriptor.format.ordinal(), iArr, pixelBufferDescriptor.handler, pixelBufferDescriptor.callback);
        }
        if (nSetImageCubemap < 0) {
            throw new BufferOverflowException();
        }
    }
}