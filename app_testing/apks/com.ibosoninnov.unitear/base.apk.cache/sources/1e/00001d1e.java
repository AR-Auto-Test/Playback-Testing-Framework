package com.google.ar.sceneform.rendering;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import c.b.a.a.a;
import com.google.android.filament.Texture;
import com.google.android.filament.android.TextureHelper;
import com.google.ar.core.annotations.UsedByNative;
import com.google.ar.sceneform.rendering.Texture;
import com.google.ar.sceneform.utilities.AndroidPreconditions;
import com.google.ar.sceneform.utilities.LoadHelper;
import com.google.ar.sceneform.utilities.Preconditions;
import java.io.InputStream;
import java.util.concurrent.Callable;
import java.util.concurrent.CompletableFuture;
import java.util.function.Function;
import java.util.function.Supplier;

@UsedByNative("material_java_wrappers.h")
/* loaded from: classes.dex */
public class Texture {
    private static final int MIP_LEVELS_TO_GENERATE = 255;
    private static final String TAG = "Texture";
    private final TextureInternalData textureData;

    /* renamed from: com.google.ar.sceneform.rendering.Texture$1  reason: invalid class name */
    /* loaded from: classes.dex */
    public static /* synthetic */ class AnonymousClass1 {
        public static final /* synthetic */ int[] $SwitchMap$com$google$ar$sceneform$rendering$Texture$Usage;

        static {
            Usage.values();
            int[] iArr = new int[3];
            $SwitchMap$com$google$ar$sceneform$rendering$Texture$Usage = iArr;
            try {
                iArr[Usage.COLOR.ordinal()] = 1;
            } catch (NoSuchFieldError unused) {
            }
            try {
                $SwitchMap$com$google$ar$sceneform$rendering$Texture$Usage[Usage.NORMAL.ordinal()] = 2;
            } catch (NoSuchFieldError unused2) {
            }
            try {
                $SwitchMap$com$google$ar$sceneform$rendering$Texture$Usage[Usage.DATA.ordinal()] = 3;
            } catch (NoSuchFieldError unused3) {
            }
        }
    }

    /* loaded from: classes.dex */
    public static final class Builder {
        private static final int MAX_BITMAP_SIZE = 4096;
        private Bitmap bitmap;
        private boolean inPremultiplied;
        private Callable<InputStream> inputStreamCreator;
        private Object registryId;
        private Sampler sampler;
        private TextureInternalData textureInternalData;
        private Usage usage;

        public /* synthetic */ Builder(AnonymousClass1 anonymousClass1) {
            this();
        }

        private static CompletableFuture<Bitmap> makeBitmap(final Callable<InputStream> callable, final boolean z) {
            return CompletableFuture.supplyAsync(new Supplier() { // from class: c.d.b.a.q.j0
                @Override // java.util.function.Supplier
                public final Object get() {
                    boolean z2 = z;
                    Callable callable2 = callable;
                    BitmapFactory.Options options = new BitmapFactory.Options();
                    options.inScaled = false;
                    options.inPremultiplied = z2;
                    try {
                        InputStream inputStream = (InputStream) callable2.call();
                        Bitmap decodeStream = BitmapFactory.decodeStream(inputStream, null, options);
                        if (inputStream != null) {
                            inputStream.close();
                        }
                        if (decodeStream != null) {
                            if (decodeStream.getConfig() == Bitmap.Config.ARGB_8888) {
                                return decodeStream;
                            }
                            throw new IllegalStateException("Texture must use ARGB8 format.");
                        }
                        throw new IllegalStateException("Failed to decode the texture bitmap. The InputStream was not a valid bitmap.");
                    } catch (Exception e2) {
                        throw new IllegalStateException(e2);
                    }
                }
            }, ThreadPools.getThreadPoolExecutor());
        }

        private static TextureInternalData makeTextureData(Bitmap bitmap, Sampler sampler, Usage usage, int i) {
            IEngine engine = EngineInstance.getEngine();
            com.google.android.filament.Texture build = new Texture.Builder().width(bitmap.getWidth()).height(bitmap.getHeight()).depth(1).levels(i).sampler(Texture.Sampler.SAMPLER_2D).format(Texture.getInternalFormatForUsage(usage)).build(engine.getFilamentEngine());
            TextureHelper.setBitmap(engine.getFilamentEngine(), build, 0, bitmap);
            if (i > 1) {
                build.generateMipmaps(engine.getFilamentEngine());
            }
            return new TextureInternalData(build, sampler);
        }

        public /* synthetic */ Texture a(Bitmap bitmap) {
            return new Texture(makeTextureData(bitmap, this.sampler, this.usage, 255), null);
        }

        /* JADX DEBUG: Type inference failed for r1v6. Raw type applied. Possible types: java.util.concurrent.CompletableFuture<U> */
        /* JADX DEBUG: Type inference failed for r2v0. Raw type applied. Possible types: java.util.function.Function, java.util.function.Function<? super android.graphics.Bitmap, ? extends U> */
        /* JADX WARN: Multi-variable type inference failed */
        public CompletableFuture<Texture> build() {
            CompletableFuture<Bitmap> completedFuture;
            CompletableFuture completableFuture;
            CompletableFuture<Texture> completableFuture2;
            AndroidPreconditions.checkUiThread();
            Object obj = this.registryId;
            if (obj == null || (completableFuture2 = ResourceManager.getInstance().getTextureRegistry().get(obj)) == null) {
                TextureInternalData textureInternalData = this.textureInternalData;
                if (textureInternalData == null || obj == null) {
                    if (textureInternalData != null) {
                        completableFuture = CompletableFuture.completedFuture(new Texture(textureInternalData, null));
                    } else {
                        Callable<InputStream> callable = this.inputStreamCreator;
                        if (callable != null) {
                            completedFuture = makeBitmap(callable, this.inPremultiplied);
                        } else {
                            Bitmap bitmap = this.bitmap;
                            if (bitmap != null) {
                                completedFuture = CompletableFuture.completedFuture(bitmap);
                            } else {
                                throw new IllegalStateException("Texture must have a source.");
                            }
                        }
                        completableFuture = completedFuture.thenApplyAsync((Function<? super Bitmap, ? extends U>) new Function() { // from class: c.d.b.a.q.i0
                            @Override // java.util.function.Function
                            public final Object apply(Object obj2) {
                                return Texture.Builder.this.a((Bitmap) obj2);
                            }
                        }, ThreadPools.getMainExecutor());
                    }
                    if (obj != null) {
                        ResourceManager.getInstance().getTextureRegistry().register(obj, completableFuture);
                    }
                    String str = Texture.TAG;
                    FutureHelper.logOnException(str, completableFuture, "Unable to load Texture registryId='" + obj + "'");
                    return completableFuture;
                }
                throw new IllegalStateException("Builder must not set both a bitmap and filament texture");
            }
            return completableFuture2;
        }

        public Builder setData(TextureInternalData textureInternalData) {
            this.textureInternalData = textureInternalData;
            return this;
        }

        public Builder setPremultiplied(boolean z) {
            this.inPremultiplied = z;
            return this;
        }

        public Builder setRegistryId(Object obj) {
            this.registryId = obj;
            return this;
        }

        public Builder setSampler(Sampler sampler) {
            this.sampler = sampler;
            return this;
        }

        public Builder setSource(Context context, Uri uri) {
            Preconditions.checkNotNull(uri, "Parameter \"sourceUri\" was null.");
            this.registryId = uri;
            setSource(LoadHelper.fromUri(context, uri));
            return this;
        }

        public Builder setUsage(Usage usage) {
            this.usage = usage;
            return this;
        }

        private Builder() {
            this.inputStreamCreator = null;
            this.bitmap = null;
            this.textureInternalData = null;
            this.usage = Usage.COLOR;
            this.registryId = null;
            this.inPremultiplied = true;
            this.sampler = Sampler.builder().build();
        }

        public Builder setSource(Callable<InputStream> callable) {
            Preconditions.checkNotNull(callable, "Parameter \"inputStreamCreator\" was null.");
            this.inputStreamCreator = callable;
            this.bitmap = null;
            return this;
        }

        public Builder setSource(Context context, int i) {
            setSource(LoadHelper.fromResource(context, i));
            this.registryId = context.getResources().getResourceName(i);
            return this;
        }

        public Builder setSource(Bitmap bitmap) {
            Preconditions.checkNotNull(bitmap, "Parameter \"bitmap\" was null.");
            if (bitmap.getConfig() == Bitmap.Config.ARGB_8888) {
                if (bitmap.hasAlpha() && !bitmap.isPremultiplied()) {
                    throw new IllegalArgumentException("Invalid Bitmap: Bitmap must be premultiplied.");
                }
                if (bitmap.getWidth() <= 4096 && bitmap.getHeight() <= 4096) {
                    this.bitmap = bitmap;
                    this.registryId = null;
                    this.inputStreamCreator = null;
                    return this;
                }
                StringBuilder x = a.x("Invalid Bitmap: Bitmap width and height must be smaller than 4096. Bitmap was ");
                x.append(bitmap.getWidth());
                x.append(" width and ");
                x.append(bitmap.getHeight());
                x.append(" height.");
                throw new IllegalArgumentException(x.toString());
            }
            StringBuilder x2 = a.x("Invalid Bitmap: Bitmap's configuration must be ARGB_8888, but it was ");
            x2.append(bitmap.getConfig());
            throw new IllegalArgumentException(x2.toString());
        }
    }

    /* loaded from: classes.dex */
    public static final class CleanupCallback implements Runnable {
        private final TextureInternalData textureData;

        public CleanupCallback(TextureInternalData textureInternalData) {
            this.textureData = textureInternalData;
        }

        @Override // java.lang.Runnable
        public void run() {
            AndroidPreconditions.checkUiThread();
            TextureInternalData textureInternalData = this.textureData;
            if (textureInternalData != null) {
                textureInternalData.release();
            }
        }
    }

    @UsedByNative("material_java_wrappers.h")
    /* loaded from: classes.dex */
    public static class Sampler {
        private final MagFilter magFilter;
        private final MinFilter minFilter;
        private final WrapMode wrapModeR;
        private final WrapMode wrapModeS;
        private final WrapMode wrapModeT;

        /* loaded from: classes.dex */
        public static class Builder {
            private MagFilter magFilter;
            private MinFilter minFilter;
            private WrapMode wrapModeR;
            private WrapMode wrapModeS;
            private WrapMode wrapModeT;

            public Sampler build() {
                return new Sampler(this, null);
            }

            public Builder setMagFilter(MagFilter magFilter) {
                this.magFilter = magFilter;
                return this;
            }

            public Builder setMinFilter(MinFilter minFilter) {
                this.minFilter = minFilter;
                return this;
            }

            public Builder setMinMagFilter(MagFilter magFilter) {
                return setMinFilter(MinFilter.values()[magFilter.ordinal()]).setMagFilter(magFilter);
            }

            public Builder setWrapMode(WrapMode wrapMode) {
                return setWrapModeS(wrapMode).setWrapModeT(wrapMode).setWrapModeR(wrapMode);
            }

            public Builder setWrapModeR(WrapMode wrapMode) {
                this.wrapModeR = wrapMode;
                return this;
            }

            public Builder setWrapModeS(WrapMode wrapMode) {
                this.wrapModeS = wrapMode;
                return this;
            }

            public Builder setWrapModeT(WrapMode wrapMode) {
                this.wrapModeT = wrapMode;
                return this;
            }
        }

        @UsedByNative("material_java_wrappers.h")
        /* loaded from: classes.dex */
        public enum MagFilter {
            NEAREST,
            LINEAR
        }

        @UsedByNative("material_java_wrappers.h")
        /* loaded from: classes.dex */
        public enum MinFilter {
            NEAREST,
            LINEAR,
            NEAREST_MIPMAP_NEAREST,
            LINEAR_MIPMAP_NEAREST,
            NEAREST_MIPMAP_LINEAR,
            LINEAR_MIPMAP_LINEAR
        }

        @UsedByNative("material_java_wrappers.h")
        /* loaded from: classes.dex */
        public enum WrapMode {
            CLAMP_TO_EDGE,
            REPEAT,
            MIRRORED_REPEAT
        }

        public /* synthetic */ Sampler(Builder builder, AnonymousClass1 anonymousClass1) {
            this(builder);
        }

        public static Builder builder() {
            return new Builder().setMinFilter(MinFilter.LINEAR_MIPMAP_LINEAR).setMagFilter(MagFilter.LINEAR).setWrapMode(WrapMode.CLAMP_TO_EDGE);
        }

        public MagFilter getMagFilter() {
            return this.magFilter;
        }

        public MinFilter getMinFilter() {
            return this.minFilter;
        }

        public WrapMode getWrapModeR() {
            return this.wrapModeR;
        }

        public WrapMode getWrapModeS() {
            return this.wrapModeS;
        }

        public WrapMode getWrapModeT() {
            return this.wrapModeT;
        }

        private Sampler(Builder builder) {
            this.minFilter = builder.minFilter;
            this.magFilter = builder.magFilter;
            this.wrapModeS = builder.wrapModeS;
            this.wrapModeT = builder.wrapModeT;
            this.wrapModeR = builder.wrapModeR;
        }
    }

    /* loaded from: classes.dex */
    public enum Usage {
        COLOR,
        NORMAL,
        DATA
    }

    public /* synthetic */ Texture(TextureInternalData textureInternalData, AnonymousClass1 anonymousClass1) {
        this(textureInternalData);
    }

    public static Builder builder() {
        AndroidPreconditions.checkMinAndroidApiLevel();
        return new Builder(null);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public static Texture.InternalFormat getInternalFormatForUsage(Usage usage) {
        if (usage.ordinal() != 0) {
            return Texture.InternalFormat.RGBA8;
        }
        return Texture.InternalFormat.SRGB8_A8;
    }

    public com.google.android.filament.Texture getFilamentTexture() {
        return ((TextureInternalData) Preconditions.checkNotNull(this.textureData)).getFilamentTexture();
    }

    public Sampler getSampler() {
        return ((TextureInternalData) Preconditions.checkNotNull(this.textureData)).getSampler();
    }

    @UsedByNative("material_java_wrappers.h")
    private Texture(TextureInternalData textureInternalData) {
        this.textureData = textureInternalData;
        textureInternalData.retain();
        ResourceManager.getInstance().getTextureCleanupRegistry().register(this, new CleanupCallback(textureInternalData));
    }
}