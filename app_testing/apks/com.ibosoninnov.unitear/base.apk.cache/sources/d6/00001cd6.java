package com.google.ar.sceneform.rendering;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.media.Image;
import android.net.Uri;
import android.util.Log;
import c.b.a.a.a;
import com.google.android.filament.IndirectLight;
import com.google.android.filament.Texture;
import com.google.ar.sceneform.math.Matrix;
import com.google.ar.sceneform.math.Quaternion;
import com.google.ar.sceneform.rendering.LightProbe;
import com.google.ar.sceneform.rendering.SceneformBundle;
import com.google.ar.sceneform.utilities.AndroidPreconditions;
import com.google.ar.sceneform.utilities.ChangeId;
import com.google.ar.sceneform.utilities.EnvironmentalHdrParameters;
import com.google.ar.sceneform.utilities.LoadHelper;
import com.google.ar.sceneform.utilities.Preconditions;
import com.google.ar.sceneform.utilities.SceneformBufferUtils;
import com.google.ar.schemas.lull.Vec3;
import com.google.ar.schemas.sceneform.LightingCubeDef;
import com.google.ar.schemas.sceneform.LightingCubeFaceDef;
import com.google.ar.schemas.sceneform.LightingDef;
import com.google.ar.schemas.sceneform.SceneformBundleDef;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.util.concurrent.Callable;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.function.Function;
import java.util.function.Supplier;

/* loaded from: classes.dex */
public class LightProbe {
    private static final int BYTES_PER_FLOAT16 = 2;
    private static final int CUBEMAP_FACE_COUNT = 6;
    private static final int CUBEMAP_MIN_WIDTH = 4;
    private static final int EXPECTED_SPHERICAL_HARMONICS_LENGTH = 27;
    private static final int FLOATS_PER_VECTOR = 3;
    private static final float LIGHT_ESTIMATE_OFFSET = 0.0f;
    private static final float LIGHT_ESTIMATE_SCALE = 1.8f;
    private static final int RGBA_BYTES_PER_PIXEL = 8;
    private static final int RGBM_BYTES_PER_PIXEL = 4;
    private static final int RGB_BYTES_PER_PIXEL = 6;
    private static final int RGB_CHANNEL_COUNT = 3;
    private static final int SH_ORDER = 3;
    private static final int SH_VECTORS_FOR_THIRD_ORDER = 9;
    private static final String TAG = "LightProbe";
    private final Color ambientColor;
    private ChangeId changeId;
    private final Color colorCorrection;
    private ByteBuffer cubemapBuffer;
    private float intensity;
    private float[] irradianceData;
    private float lightEstimate;
    private String name;
    private com.google.android.filament.Texture reflectCubemap;
    private Quaternion rotation;
    private static final int[] FACE_TO_FILAMENT_MAPPING = {3, 0, 4, 1, 5, 2};
    private static final float[] ENVIRONMENTAL_HDR_TO_FILAMENT_SH_COEFFIECIENTS = {0.282095f, -0.325735f, 0.325735f, -0.325735f, 0.273137f, -0.273137f, 0.078848f, -0.273137f, 0.136569f};
    private static final int[] ENVIRONMENTAL_HDR_TO_FILAMENT_SH_INDEX_MAP = {0, 1, 2, 3, 4, 5, 7, 6, 8};

    /* loaded from: classes.dex */
    public static final class Builder {
        private Callable<InputStream> inputStreamCreator;
        private float intensity;
        private String name;
        private Quaternion rotation;

        public CompletableFuture<LightProbe> build() {
            if (this.inputStreamCreator != null) {
                final LightProbe lightProbe = new LightProbe(this);
                CompletableFuture thenApplyAsync = lightProbe.loadInBackground(this.inputStreamCreator).thenApplyAsync(new Function() { // from class: c.d.b.a.q.f
                    @Override // java.util.function.Function
                    public final Object apply(Object obj) {
                        LightProbe lightProbe2 = LightProbe.this;
                        LightProbe.access$700(lightProbe2, (LightingDef) obj);
                        return lightProbe2;
                    }
                }, ThreadPools.getMainExecutor());
                if (thenApplyAsync != null) {
                    String str = LightProbe.TAG;
                    StringBuilder x = a.x("Unable to load LightProbe: name='");
                    x.append(this.name);
                    x.append("'");
                    return FutureHelper.logOnException(str, thenApplyAsync, x.toString());
                }
                throw new IllegalStateException("CompletableFuture result is null.");
            }
            throw new IllegalStateException("Light Probe source is NULL, this should never happen.");
        }

        public Builder setAssetName(String str) {
            this.name = str;
            return this;
        }

        public Builder setIntensity(float f2) {
            this.intensity = f2;
            return this;
        }

        public Builder setRotation(Quaternion quaternion) {
            this.rotation = quaternion;
            return this;
        }

        public Builder setSource(Context context, Uri uri) {
            Preconditions.checkNotNull(uri, "Parameter \"sourceUri\" was null.");
            setSource(LoadHelper.fromUri(context, uri));
            return this;
        }

        private Builder() {
            this.inputStreamCreator = null;
            this.intensity = 220.0f;
            this.name = null;
        }

        public Builder setSource(Context context, int i) {
            setSource(LoadHelper.fromResource(context, i));
            return this;
        }

        public Builder setSource(Callable<InputStream> callable) {
            Preconditions.checkNotNull(callable, "Parameter \"sourceInputStreamCallable\" was null.");
            this.inputStreamCreator = callable;
            return this;
        }
    }

    /* JADX DEBUG: Method not inlined, still used in: [c.d.b.a.q.f.apply(java.lang.Object):java.lang.Object] */
    public static /* synthetic */ void access$700(LightProbe lightProbe, LightingDef lightingDef) {
        lightProbe.buildFilamentResource(lightingDef);
    }

    public void buildFilamentResource(LightingDef lightingDef) {
        dispose();
        this.changeId.update();
        if (lightingDef != null) {
            com.google.android.filament.Texture loadReflectCubemapFromLightingDef = loadReflectCubemapFromLightingDef(lightingDef);
            if (loadReflectCubemapFromLightingDef != null) {
                setCubeMapFromTexture(loadReflectCubemapFromLightingDef);
                int shCoefficientsLength = lightingDef.shCoefficientsLength();
                if (shCoefficientsLength >= 9) {
                    int i = shCoefficientsLength * 3;
                    float[] fArr = this.irradianceData;
                    if (fArr == null || fArr.length != i) {
                        this.irradianceData = new float[i];
                    }
                    for (int i2 = 0; i2 < shCoefficientsLength; i2++) {
                        Vec3 shCoefficients = lightingDef.shCoefficients(i2);
                        int i3 = i2 * 3;
                        this.irradianceData[i3 + 0] = shCoefficients.x() / 3.1415927f;
                        this.irradianceData[i3 + 1] = shCoefficients.y() / 3.1415927f;
                        this.irradianceData[i3 + 2] = shCoefficients.z() / 3.1415927f;
                    }
                    Color color = this.ambientColor;
                    float[] fArr2 = this.irradianceData;
                    color.set(fArr2[0], fArr2[1], fArr2[2]);
                    return;
                }
                throw new IllegalStateException("Too few SH vectors for the current Order (3).");
            }
            throw new IllegalStateException("Load reflection cubemap failed.");
        }
        throw new IllegalStateException("buildFilamentResource called but no resource is available to load.");
    }

    public static Builder builder() {
        return new Builder();
    }

    public CompletableFuture<LightingDef> loadInBackground(final Callable<InputStream> callable) {
        return CompletableFuture.supplyAsync(new Supplier() { // from class: c.d.b.a.q.g
            @Override // java.util.function.Supplier
            public final Object get() {
                return LightProbe.this.a(callable);
            }
        }, ThreadPools.getThreadPoolExecutor());
    }

    private static com.google.android.filament.Texture loadReflectCubemapFromLightingDef(LightingDef lightingDef) {
        Preconditions.checkNotNull(lightingDef, "Parameter \"lightingDef\" was null.");
        IEngine engine = EngineInstance.getEngine();
        int cubeLevelsLength = lightingDef.cubeLevelsLength();
        if (cubeLevelsLength >= 1) {
            int i = 0;
            LightingCubeFaceDef faces = lightingDef.cubeLevels(0).faces(0);
            BitmapFactory.Options options = new BitmapFactory.Options();
            options.inPremultiplied = false;
            options.inScaled = false;
            options.inJustDecodeBounds = true;
            ByteBuffer dataAsByteBuffer = faces.dataAsByteBuffer();
            BitmapFactory.decodeByteArray(dataAsByteBuffer.array(), dataAsByteBuffer.position() + dataAsByteBuffer.arrayOffset(), dataAsByteBuffer.limit() - dataAsByteBuffer.position(), options);
            int i2 = options.outWidth;
            int i3 = options.outHeight;
            if (i2 >= 4 && i3 >= 4 && i2 == i3) {
                com.google.android.filament.Texture build = new Texture.Builder().width(i2).height(i3).levels(cubeLevelsLength).format(Texture.InternalFormat.R11F_G11F_B10F).sampler(Texture.Sampler.SAMPLER_CUBEMAP).build(engine.getFilamentEngine());
                int[] iArr = new int[6];
                options.inJustDecodeBounds = false;
                int i4 = i2 * i3 * 4;
                int i5 = 6;
                int i6 = i3;
                int i7 = 0;
                while (i < cubeLevelsLength) {
                    ByteBuffer allocateDirect = ByteBuffer.allocateDirect(i4 * 6);
                    LightingCubeDef cubeLevels = lightingDef.cubeLevels(i);
                    while (i7 < i5) {
                        LightingCubeFaceDef faces2 = cubeLevels.faces(FACE_TO_FILAMENT_MAPPING[i7]);
                        iArr[i7] = i4 * i7;
                        ByteBuffer dataAsByteBuffer2 = faces2.dataAsByteBuffer();
                        Bitmap decodeByteArray = BitmapFactory.decodeByteArray(dataAsByteBuffer2.array(), dataAsByteBuffer2.position() + dataAsByteBuffer2.arrayOffset(), dataAsByteBuffer2.limit() - dataAsByteBuffer2.position(), options);
                        if (decodeByteArray.getWidth() == i2 && decodeByteArray.getHeight() == i6) {
                            decodeByteArray.copyPixelsToBuffer(allocateDirect);
                            i7++;
                            i5 = 6;
                        } else {
                            throw new AssertionError("All cube map textures must have the same size");
                        }
                    }
                    allocateDirect.rewind();
                    build.setImage(engine.getFilamentEngine(), i, new Texture.PixelBufferDescriptor(allocateDirect, Texture.Format.RGB, Texture.Type.UINT_10F_11F_11F_REV), iArr);
                    i2 >>= 1;
                    i6 >>= 1;
                    i4 = i2 * i6 * 4;
                    i++;
                    i7 = 0;
                    i5 = 6;
                }
                return build;
            }
            throw new IllegalStateException(a.k("Lighting cubemap has invalid dimensions: ", i2, " x ", i3));
        }
        throw new IllegalStateException("Lighting cubemap has no image data.");
    }

    private static float[] quaternionToRotationMatrix(Quaternion quaternion) {
        Matrix matrix = new Matrix();
        matrix.makeRotation(quaternion);
        float[] fArr = matrix.data;
        return new float[]{fArr[0], fArr[1], fArr[2], fArr[4], fArr[5], fArr[6], fArr[8], fArr[9], fArr[10]};
    }

    private void setCubeMapFromTexture(com.google.android.filament.Texture texture) {
        com.google.android.filament.Texture texture2 = this.reflectCubemap;
        IEngine engine = EngineInstance.getEngine();
        if (texture2 != null && engine != null && engine.isValid()) {
            engine.destroyTexture(texture2);
        }
        this.reflectCubemap = texture;
    }

    public /* synthetic */ LightingDef a(Callable callable) {
        if (callable != null) {
            try {
                InputStream inputStream = (InputStream) callable.call();
                ByteBuffer readStream = SceneformBufferUtils.readStream(inputStream);
                if (inputStream != null) {
                    inputStream.close();
                }
                if (readStream != null) {
                    try {
                        SceneformBundleDef tryLoadSceneformBundle = SceneformBundle.tryLoadSceneformBundle(readStream);
                        if (tryLoadSceneformBundle != null) {
                            int lightingDefsLength = tryLoadSceneformBundle.lightingDefsLength();
                            if (lightingDefsLength >= 1) {
                                int i = -1;
                                int i2 = 0;
                                if (this.name != null) {
                                    while (true) {
                                        if (i2 >= lightingDefsLength) {
                                            break;
                                        } else if (tryLoadSceneformBundle.lightingDefs(i2).name().equals(this.name)) {
                                            i = i2;
                                            break;
                                        } else {
                                            i2++;
                                        }
                                    }
                                    if (i < 0) {
                                        throw new IllegalArgumentException(a.v(a.x("Light Probe asset \""), this.name, "\" not found in bundle."));
                                    }
                                    i2 = i;
                                }
                                LightingDef lightingDefs = tryLoadSceneformBundle.lightingDefs(i2);
                                if (lightingDefs != null) {
                                    return lightingDefs;
                                }
                                throw new IllegalStateException("LightingDef is invalid.");
                            }
                            throw new IllegalStateException("Content does not contain any Light Probe data.");
                        }
                        throw new AssertionError("The Sceneform bundle containing the Light Probe could not be loaded.");
                    } catch (SceneformBundle.VersionException e2) {
                        throw new CompletionException(e2);
                    }
                }
                throw new AssertionError("The Sceneform bundle containing the Light Probe could not be loaded.");
            } catch (Exception e3) {
                throw new CompletionException(e3);
            }
        }
        throw new IllegalArgumentException("Invalid source.");
    }

    public IndirectLight buildIndirectLight() {
        Preconditions.checkNotNull(this.irradianceData, "\"irradianceData\" was null.");
        Preconditions.checkState(this.irradianceData.length >= 3, "\"irradianceData\" does not have enough components to store a vector");
        if (this.reflectCubemap != null) {
            float[] fArr = this.irradianceData;
            Color color = this.ambientColor;
            float f2 = color.r;
            Color color2 = this.colorCorrection;
            fArr[0] = f2 * color2.r;
            fArr[1] = color.f5628g * color2.f5628g;
            fArr[2] = color.f5627b * color2.f5627b;
            IndirectLight build = new IndirectLight.Builder().reflections(this.reflectCubemap).irradiance(3, this.irradianceData).intensity(this.intensity * this.lightEstimate).build(EngineInstance.getEngine().getFilamentEngine());
            Quaternion quaternion = this.rotation;
            if (quaternion != null) {
                build.setRotation(quaternionToRotationMatrix(quaternion));
            }
            if (build != null) {
                return build;
            }
            throw new IllegalStateException("Light Probe is invalid.");
        }
        throw new IllegalStateException("reflectCubemap is null.");
    }

    public void dispose() {
        AndroidPreconditions.checkUiThread();
        setCubeMapFromTexture(null);
        this.changeId = new ChangeId();
    }

    public void finalize() {
        try {
            try {
                ThreadPools.getMainExecutor().execute(new Runnable() { // from class: c.d.b.a.q.e
                    @Override // java.lang.Runnable
                    public final void run() {
                        LightProbe.this.dispose();
                    }
                });
            } catch (Exception e2) {
                Log.e(TAG, "Error while Finalizing Light Probe.", e2);
            }
        } finally {
            super.finalize();
        }
    }

    public int getId() {
        return this.changeId.get();
    }

    public float getIntensity() {
        return this.intensity;
    }

    public Quaternion getRotation() {
        return this.rotation;
    }

    public boolean isReady() {
        return !this.changeId.isEmpty();
    }

    public void setCubeMap(Image[] imageArr) {
        if (imageArr.length == 6) {
            int width = imageArr[0].getWidth();
            int height = imageArr[0].getHeight();
            int i = width * height * 6 * 3 * 2;
            if (this.cubemapBuffer.capacity() < i) {
                this.cubemapBuffer = ByteBuffer.allocate(i);
            } else {
                this.cubemapBuffer.clear();
            }
            int[] iArr = new int[6];
            for (int i2 = 0; i2 < 6; i2++) {
                iArr[i2] = this.cubemapBuffer.position();
                Image.Plane[] planes = imageArr[i2].getPlanes();
                if (planes.length == 1) {
                    Image.Plane plane = planes[0];
                    if (plane.getPixelStride() == 8) {
                        int i3 = width * 8;
                        if (plane.getRowStride() == i3) {
                            ByteBuffer buffer = plane.getBuffer();
                            while (buffer.hasRemaining()) {
                                for (int i4 = 0; i4 < 8; i4++) {
                                    byte b2 = buffer.get();
                                    if (i4 < 6) {
                                        this.cubemapBuffer.put(b2);
                                    }
                                }
                            }
                        } else {
                            StringBuilder y = a.y("Unexpected row stride in cubemap data: expected ", i3, ", got ");
                            y.append(plane.getRowStride());
                            throw new IllegalArgumentException(y.toString());
                        }
                    } else {
                        StringBuilder x = a.x("Unexpected pixel stride in cubemap data: expected 8, got ");
                        x.append(plane.getPixelStride());
                        throw new IllegalArgumentException(x.toString());
                    }
                } else {
                    StringBuilder x2 = a.x("Unexpected number of Planes in cubemap Image array: ");
                    x2.append(planes.length);
                    throw new IllegalArgumentException(x2.toString());
                }
            }
            this.cubemapBuffer.flip();
            IEngine engine = EngineInstance.getEngine();
            com.google.android.filament.Texture build = new Texture.Builder().width(width).height(height).levels((int) ((Math.log(width) / Math.log(2.0d)) + 1.0d)).sampler(Texture.Sampler.SAMPLER_CUBEMAP).format(Texture.InternalFormat.R11F_G11F_B10F).build(engine.getFilamentEngine());
            Texture.PixelBufferDescriptor pixelBufferDescriptor = new Texture.PixelBufferDescriptor(this.cubemapBuffer, Texture.Format.RGB, Texture.Type.HALF);
            Texture.PrefilterOptions prefilterOptions = new Texture.PrefilterOptions();
            prefilterOptions.mirror = false;
            build.generatePrefilterMipmap(engine.getFilamentEngine(), pixelBufferDescriptor, iArr, prefilterOptions);
            setCubeMapFromTexture(build);
            return;
        }
        StringBuilder x3 = a.x("Unexpected cubemap array length: ");
        x3.append(imageArr.length);
        throw new IllegalArgumentException(x3.toString());
    }

    public void setEnvironmentalHdrSphericalHarmonics(float[] fArr, float f2, EnvironmentalHdrParameters environmentalHdrParameters) {
        float ambientShScaleForFilament = environmentalHdrParameters.getAmbientShScaleForFilament() / (environmentalHdrParameters.getReflectionScaleForFilament() * f2);
        if (fArr.length == 27) {
            float[] fArr2 = this.irradianceData;
            if (fArr2 == null || fArr2.length != fArr.length) {
                this.irradianceData = new float[27];
            }
            for (int i = 0; i < 9; i++) {
                int i2 = ENVIRONMENTAL_HDR_TO_FILAMENT_SH_INDEX_MAP[i];
                float[] fArr3 = this.irradianceData;
                int i3 = i2 * 3;
                int i4 = i * 3;
                float f3 = fArr[i4];
                float[] fArr4 = ENVIRONMENTAL_HDR_TO_FILAMENT_SH_COEFFIECIENTS;
                fArr3[i3] = f3 * fArr4[i2] * ambientShScaleForFilament;
                fArr3[i3 + 1] = fArr[i4 + 1] * fArr4[i2] * ambientShScaleForFilament;
                fArr3[i3 + 2] = fArr[i4 + 2] * fArr4[i2] * ambientShScaleForFilament;
            }
            Color color = this.ambientColor;
            float[] fArr5 = this.irradianceData;
            color.set(fArr5[0], fArr5[1], fArr5[2]);
            this.colorCorrection.set(new Color(1.0f, 1.0f, 1.0f));
            this.lightEstimate = environmentalHdrParameters.getReflectionScaleForFilament();
            this.intensity = 1.0f;
            return;
        }
        throw new RuntimeException("Expected 27 spherical Harmonics coefficients");
    }

    public void setIntensity(float f2) {
        this.intensity = f2;
    }

    public void setLightEstimate(Color color, float f2) {
        this.lightEstimate = Math.min((f2 * LIGHT_ESTIMATE_SCALE) + 0.0f, 1.0f);
        this.colorCorrection.set(color);
    }

    public void setRotation(Quaternion quaternion) {
        this.rotation = quaternion;
    }

    private LightProbe(Builder builder) {
        this.cubemapBuffer = ByteBuffer.allocate(10000);
        this.reflectCubemap = null;
        this.colorCorrection = new Color(1.0f, 1.0f, 1.0f);
        this.ambientColor = new Color();
        this.name = null;
        this.changeId = new ChangeId();
        this.lightEstimate = 1.0f;
        this.intensity = builder.intensity;
        this.rotation = builder.rotation;
        this.name = builder.name;
    }
}