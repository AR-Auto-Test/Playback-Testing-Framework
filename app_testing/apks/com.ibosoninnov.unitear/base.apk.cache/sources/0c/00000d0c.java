package com.google.android.filament;

import com.google.android.material.internal.StaticLayoutBuilderCompat;
import java.util.EnumSet;
import java.util.Iterator;

/* loaded from: classes.dex */
public class View {
    private AmbientOcclusionOptions mAmbientOcclusionOptions;
    private BlendMode mBlendMode;
    private BloomOptions mBloomOptions;
    private Camera mCamera;
    private DepthOfFieldOptions mDepthOfFieldOptions;
    private DynamicResolutionOptions mDynamicResolution;
    private FogOptions mFogOptions;
    private String mName;
    private long mNativeObject;
    private RenderQuality mRenderQuality;
    private RenderTarget mRenderTarget;
    private Scene mScene;
    private Viewport mViewport = new Viewport(0, 0, 0, 0);

    /* loaded from: classes.dex */
    public enum AmbientOcclusion {
        NONE,
        SSAO
    }

    /* loaded from: classes.dex */
    public static class AmbientOcclusionOptions {
        public float radius = 0.3f;
        public float bias = 5.0E-4f;
        public float power = 1.0f;
        public float resolution = 0.5f;
        public float intensity = 1.0f;
        public QualityLevel quality = QualityLevel.LOW;
    }

    /* loaded from: classes.dex */
    public enum AntiAliasing {
        NONE,
        FXAA
    }

    /* loaded from: classes.dex */
    public enum BlendMode {
        OPAQUE,
        TRANSLUCENT
    }

    /* loaded from: classes.dex */
    public static class BloomOptions {
        public Texture dirt = null;
        public float dirtStrength = 0.2f;
        public float strength = 0.1f;
        public int resolution = 360;
        public float anamorphism = 1.0f;
        public int levels = 6;
        public BlendingMode blendingMode = BlendingMode.ADD;
        public boolean threshold = true;
        public boolean enabled = false;

        /* loaded from: classes.dex */
        public enum BlendingMode {
            ADD,
            INTERPOLATE
        }
    }

    /* loaded from: classes.dex */
    public static class DepthOfFieldOptions {
        public float focusDistance = 10.0f;
        public float blurScale = 1.0f;
        public float maxApertureDiameter = 0.01f;
        public boolean enabled = false;
    }

    /* loaded from: classes.dex */
    public enum Dithering {
        NONE,
        TEMPORAL
    }

    /* loaded from: classes.dex */
    public static class DynamicResolutionOptions {
        public boolean enabled = false;
        public boolean homogeneousScaling = false;
        public float minScale = 0.5f;
        public float maxScale = 1.0f;
        public QualityLevel quality = QualityLevel.LOW;
    }

    /* loaded from: classes.dex */
    public static class FogOptions {
        public float distance = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        public float maximumOpacity = 1.0f;
        public float height = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        public float heightFalloff = 1.0f;
        public float[] color = {0.5f, 0.5f, 0.5f};
        public float density = 0.1f;
        public float inScatteringStart = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        public float inScatteringSize = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        public boolean fogColorFromIbl = false;
        public boolean enabled = false;
    }

    /* loaded from: classes.dex */
    public enum QualityLevel {
        LOW,
        MEDIUM,
        HIGH,
        ULTRA
    }

    /* loaded from: classes.dex */
    public static class RenderQuality {
        public QualityLevel hdrColorBuffer = QualityLevel.HIGH;
    }

    /* loaded from: classes.dex */
    public enum TargetBufferFlags {
        COLOR0(1),
        COLOR1(2),
        COLOR2(4),
        COLOR3(8),
        DEPTH(16),
        STENCIL(32);
        
        public static EnumSet<TargetBufferFlags> ALL;
        public static EnumSet<TargetBufferFlags> ALL_COLOR;
        public static EnumSet<TargetBufferFlags> DEPTH_STENCIL;
        public static EnumSet<TargetBufferFlags> NONE;
        private int mFlags;

        static {
            TargetBufferFlags targetBufferFlags = COLOR0;
            TargetBufferFlags targetBufferFlags2 = COLOR1;
            TargetBufferFlags targetBufferFlags3 = COLOR2;
            TargetBufferFlags targetBufferFlags4 = COLOR3;
            TargetBufferFlags targetBufferFlags5 = DEPTH;
            TargetBufferFlags targetBufferFlags6 = STENCIL;
            NONE = EnumSet.noneOf(TargetBufferFlags.class);
            ALL_COLOR = EnumSet.of(targetBufferFlags, targetBufferFlags2, targetBufferFlags3, targetBufferFlags4);
            DEPTH_STENCIL = EnumSet.of(targetBufferFlags5, targetBufferFlags6);
            ALL = EnumSet.range(targetBufferFlags, targetBufferFlags6);
        }

        TargetBufferFlags(int i) {
            this.mFlags = i;
        }

        public static int flags(EnumSet<TargetBufferFlags> enumSet) {
            Iterator it = enumSet.iterator();
            int i = 0;
            while (it.hasNext()) {
                i |= ((TargetBufferFlags) it.next()).mFlags;
            }
            return i;
        }
    }

    /* loaded from: classes.dex */
    public enum ToneMapping {
        LINEAR,
        ACES
    }

    public View(long j) {
        this.mNativeObject = j;
    }

    private static native int nGetAmbientOcclusion(long j);

    private static native int nGetAntiAliasing(long j);

    private static native int nGetDithering(long j);

    private static native int nGetSampleCount(long j);

    private static native int nGetToneMapping(long j);

    private static native boolean nIsFrontFaceWindingInverted(long j);

    private static native boolean nIsPostProcessingEnabled(long j);

    private static native void nSetAmbientOcclusion(long j, int i);

    private static native void nSetAmbientOcclusionOptions(long j, float f2, float f3, float f4, float f5, float f6, int i);

    private static native void nSetAntiAliasing(long j, int i);

    private static native void nSetBlendMode(long j, int i);

    private static native void nSetBloomOptions(long j, long j2, float f2, float f3, int i, float f4, int i2, int i3, boolean z, boolean z2);

    private static native void nSetCamera(long j, long j2);

    private static native void nSetDepthOfFieldOptions(long j, float f2, float f3, float f4, boolean z);

    private static native void nSetDithering(long j, int i);

    private static native void nSetDynamicLightingOptions(long j, float f2, float f3);

    private static native void nSetDynamicResolutionOptions(long j, boolean z, boolean z2, float f2, float f3, int i);

    private static native void nSetFogOptions(long j, float f2, float f3, float f4, float f5, float f6, float f7, float f8, float f9, float f10, float f11, boolean z, boolean z2);

    private static native void nSetFrontFaceWindingInverted(long j, boolean z);

    private static native void nSetName(long j, String str);

    private static native void nSetPostProcessingEnabled(long j, boolean z);

    private static native void nSetRenderQuality(long j, int i);

    private static native void nSetRenderTarget(long j, long j2);

    private static native void nSetSampleCount(long j, int i);

    private static native void nSetScene(long j, long j2);

    private static native void nSetShadowsEnabled(long j, boolean z);

    private static native void nSetToneMapping(long j, int i);

    private static native void nSetViewport(long j, int i, int i2, int i3, int i4);

    private static native void nSetVisibleLayers(long j, int i, int i2);

    public void clearNativeObject() {
        this.mNativeObject = 0L;
    }

    public AmbientOcclusion getAmbientOcclusion() {
        return AmbientOcclusion.values()[nGetAmbientOcclusion(getNativeObject())];
    }

    public AmbientOcclusionOptions getAmbientOcclusionOptions() {
        if (this.mAmbientOcclusionOptions == null) {
            this.mAmbientOcclusionOptions = new AmbientOcclusionOptions();
        }
        return this.mAmbientOcclusionOptions;
    }

    public AntiAliasing getAntiAliasing() {
        return AntiAliasing.values()[nGetAntiAliasing(getNativeObject())];
    }

    public BlendMode getBlendMode() {
        return this.mBlendMode;
    }

    public BloomOptions getBloomOptions() {
        if (this.mBloomOptions == null) {
            this.mBloomOptions = new BloomOptions();
        }
        return this.mBloomOptions;
    }

    public Camera getCamera() {
        return this.mCamera;
    }

    public DepthOfFieldOptions getDepthOfFieldOptions() {
        if (this.mDepthOfFieldOptions == null) {
            this.mDepthOfFieldOptions = new DepthOfFieldOptions();
        }
        return this.mDepthOfFieldOptions;
    }

    public Dithering getDithering() {
        return Dithering.values()[nGetDithering(getNativeObject())];
    }

    public DynamicResolutionOptions getDynamicResolutionOptions() {
        if (this.mDynamicResolution == null) {
            this.mDynamicResolution = new DynamicResolutionOptions();
        }
        return this.mDynamicResolution;
    }

    public FogOptions getFogOptions() {
        if (this.mFogOptions == null) {
            this.mFogOptions = new FogOptions();
        }
        return this.mFogOptions;
    }

    public String getName() {
        return this.mName;
    }

    public long getNativeObject() {
        long j = this.mNativeObject;
        if (j != 0) {
            return j;
        }
        throw new IllegalStateException("Calling method on destroyed View");
    }

    public RenderQuality getRenderQuality() {
        if (this.mRenderQuality == null) {
            this.mRenderQuality = new RenderQuality();
        }
        return this.mRenderQuality;
    }

    public RenderTarget getRenderTarget() {
        return this.mRenderTarget;
    }

    public int getSampleCount() {
        return nGetSampleCount(getNativeObject());
    }

    public Scene getScene() {
        return this.mScene;
    }

    public ToneMapping getToneMapping() {
        return ToneMapping.values()[nGetToneMapping(getNativeObject())];
    }

    public Viewport getViewport() {
        return this.mViewport;
    }

    public boolean isFrontFaceWindingInverted() {
        return nIsFrontFaceWindingInverted(getNativeObject());
    }

    public boolean isPostProcessingEnabled() {
        return nIsPostProcessingEnabled(getNativeObject());
    }

    public void setAmbientOcclusion(AmbientOcclusion ambientOcclusion) {
        nSetAmbientOcclusion(getNativeObject(), ambientOcclusion.ordinal());
    }

    public void setAmbientOcclusionOptions(AmbientOcclusionOptions ambientOcclusionOptions) {
        this.mAmbientOcclusionOptions = ambientOcclusionOptions;
        nSetAmbientOcclusionOptions(getNativeObject(), ambientOcclusionOptions.radius, ambientOcclusionOptions.bias, ambientOcclusionOptions.power, ambientOcclusionOptions.resolution, ambientOcclusionOptions.intensity, ambientOcclusionOptions.quality.ordinal());
    }

    public void setAntiAliasing(AntiAliasing antiAliasing) {
        nSetAntiAliasing(getNativeObject(), antiAliasing.ordinal());
    }

    public void setBlendMode(BlendMode blendMode) {
        this.mBlendMode = blendMode;
        nSetBlendMode(getNativeObject(), blendMode.ordinal());
    }

    public void setBloomOptions(BloomOptions bloomOptions) {
        this.mBloomOptions = bloomOptions;
        long nativeObject = getNativeObject();
        Texture texture = bloomOptions.dirt;
        nSetBloomOptions(nativeObject, texture != null ? texture.getNativeObject() : 0L, bloomOptions.dirtStrength, bloomOptions.strength, bloomOptions.resolution, bloomOptions.anamorphism, bloomOptions.levels, bloomOptions.blendingMode.ordinal(), bloomOptions.threshold, bloomOptions.enabled);
    }

    public void setCamera(Camera camera) {
        this.mCamera = camera;
        nSetCamera(getNativeObject(), camera == null ? 0L : camera.getNativeObject());
    }

    public void setDepthOfFieldOptions(DepthOfFieldOptions depthOfFieldOptions) {
        this.mDepthOfFieldOptions = depthOfFieldOptions;
        nSetDepthOfFieldOptions(getNativeObject(), depthOfFieldOptions.focusDistance, depthOfFieldOptions.blurScale, depthOfFieldOptions.maxApertureDiameter, depthOfFieldOptions.enabled);
    }

    public void setDithering(Dithering dithering) {
        nSetDithering(getNativeObject(), dithering.ordinal());
    }

    public void setDynamicLightingOptions(float f2, float f3) {
        nSetDynamicLightingOptions(getNativeObject(), f2, f3);
    }

    public void setDynamicResolutionOptions(DynamicResolutionOptions dynamicResolutionOptions) {
        this.mDynamicResolution = dynamicResolutionOptions;
        nSetDynamicResolutionOptions(getNativeObject(), dynamicResolutionOptions.enabled, dynamicResolutionOptions.homogeneousScaling, dynamicResolutionOptions.minScale, dynamicResolutionOptions.maxScale, dynamicResolutionOptions.quality.ordinal());
    }

    public void setFogOptions(FogOptions fogOptions) {
        this.mFogOptions = fogOptions;
        long nativeObject = getNativeObject();
        float f2 = fogOptions.distance;
        float f3 = fogOptions.maximumOpacity;
        float f4 = fogOptions.height;
        float f5 = fogOptions.heightFalloff;
        float[] fArr = fogOptions.color;
        nSetFogOptions(nativeObject, f2, f3, f4, f5, fArr[0], fArr[1], fArr[2], fogOptions.density, fogOptions.inScatteringStart, fogOptions.inScatteringSize, fogOptions.fogColorFromIbl, fogOptions.enabled);
    }

    public void setFrontFaceWindingInverted(boolean z) {
        nSetFrontFaceWindingInverted(getNativeObject(), z);
    }

    public void setName(String str) {
        this.mName = str;
        nSetName(getNativeObject(), str);
    }

    public void setPostProcessingEnabled(boolean z) {
        nSetPostProcessingEnabled(getNativeObject(), z);
    }

    public void setRenderQuality(RenderQuality renderQuality) {
        this.mRenderQuality = renderQuality;
        nSetRenderQuality(getNativeObject(), renderQuality.hdrColorBuffer.ordinal());
    }

    public void setRenderTarget(RenderTarget renderTarget) {
        this.mRenderTarget = renderTarget;
        nSetRenderTarget(getNativeObject(), renderTarget != null ? renderTarget.getNativeObject() : 0L);
    }

    public void setSampleCount(int i) {
        nSetSampleCount(getNativeObject(), i);
    }

    public void setScene(Scene scene) {
        this.mScene = scene;
        nSetScene(getNativeObject(), scene == null ? 0L : scene.getNativeObject());
    }

    public void setShadowsEnabled(boolean z) {
        nSetShadowsEnabled(getNativeObject(), z);
    }

    public void setToneMapping(ToneMapping toneMapping) {
        nSetToneMapping(getNativeObject(), toneMapping.ordinal());
    }

    public void setViewport(Viewport viewport) {
        this.mViewport = viewport;
        long nativeObject = getNativeObject();
        Viewport viewport2 = this.mViewport;
        nSetViewport(nativeObject, viewport2.left, viewport2.bottom, viewport2.width, viewport2.height);
    }

    public void setVisibleLayers(int i, int i2) {
        nSetVisibleLayers(getNativeObject(), i & 255, i2 & 255);
    }
}