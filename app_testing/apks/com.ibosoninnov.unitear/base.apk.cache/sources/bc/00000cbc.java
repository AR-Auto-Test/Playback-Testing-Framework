package com.google.android.filament;

import com.google.android.filament.Colors;
import com.google.android.filament.MaterialInstance;
import com.google.android.filament.VertexBuffer;
import com.google.android.filament.proguard.UsedByNative;
import java.nio.Buffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;

/* loaded from: classes.dex */
public class Material {
    private final MaterialInstance mDefaultInstance;
    private long mNativeObject;
    private Set<VertexBuffer.VertexAttribute> mRequiredAttributes;

    /* loaded from: classes.dex */
    public enum BlendingMode {
        OPAQUE,
        TRANSPARENT,
        ADD,
        MASKED,
        FADE,
        MULTIPLY,
        SCREEN
    }

    /* loaded from: classes.dex */
    public static class Builder {
        private Buffer mBuffer;
        private int mSize;

        public Material build(Engine engine) {
            long nBuilderBuild = Material.nBuilderBuild(engine.getNativeObject(), this.mBuffer, this.mSize);
            if (nBuilderBuild != 0) {
                return new Material(nBuilderBuild);
            }
            throw new IllegalStateException("Couldn't create Material");
        }

        public Builder payload(Buffer buffer, int i) {
            this.mBuffer = buffer;
            this.mSize = i;
            return this;
        }
    }

    /* loaded from: classes.dex */
    public enum CullingMode {
        NONE,
        FRONT,
        BACK,
        FRONT_AND_BACK
    }

    /* loaded from: classes.dex */
    public enum Interpolation {
        SMOOTH,
        FLAT
    }

    @UsedByNative("Material.cpp")
    /* loaded from: classes.dex */
    public static class Parameter {
        @UsedByNative("Material.cpp")
        private static final int SAMPLER_OFFSET = Type.MAT4.ordinal() + 1;
        public final int count;
        public final String name;
        public final Precision precision;
        public final Type type;

        /* loaded from: classes.dex */
        public enum Precision {
            LOW,
            MEDIUM,
            HIGH,
            DEFAULT
        }

        /* loaded from: classes.dex */
        public enum Type {
            BOOL,
            BOOL2,
            BOOL3,
            BOOL4,
            FLOAT,
            FLOAT2,
            FLOAT3,
            FLOAT4,
            INT,
            INT2,
            INT3,
            INT4,
            UINT,
            UINT2,
            UINT3,
            UINT4,
            MAT3,
            MAT4,
            SAMPLER_2D,
            SAMPLER_CUBEMAP,
            SAMPLER_EXTERNAL
        }

        private Parameter(String str, Type type, Precision precision, int i) {
            this.name = str;
            this.type = type;
            this.precision = precision;
            this.count = i;
        }

        @UsedByNative("Material.cpp")
        private static void add(List<Parameter> list, String str, int i, int i2, int i3) {
            list.add(new Parameter(str, Type.values()[i], Precision.values()[i2], i3));
        }
    }

    /* loaded from: classes.dex */
    public enum RefractionMode {
        NONE,
        CUBEMAP,
        SCREEN_SPACE
    }

    /* loaded from: classes.dex */
    public enum RefractionType {
        SOLID,
        THIN
    }

    /* loaded from: classes.dex */
    public enum Shading {
        UNLIT,
        LIT,
        SUBSURFACE,
        CLOTH,
        SPECULAR_GLOSSINESS
    }

    /* loaded from: classes.dex */
    public enum VertexDomain {
        OBJECT,
        WORLD,
        VIEW,
        DEVICE
    }

    public Material(long j) {
        this.mNativeObject = j;
        this.mDefaultInstance = new MaterialInstance(this, nGetDefaultInstance(j));
    }

    /* JADX INFO: Access modifiers changed from: private */
    public static native long nBuilderBuild(long j, Buffer buffer, int i);

    private static native long nCreateInstance(long j);

    private static native long nCreateInstanceWithName(long j, String str);

    private static native int nGetBlendingMode(long j);

    private static native int nGetCullingMode(long j);

    private static native long nGetDefaultInstance(long j);

    private static native int nGetInterpolation(long j);

    private static native float nGetMaskThreshold(long j);

    private static native String nGetName(long j);

    private static native int nGetParameterCount(long j);

    private static native void nGetParameters(long j, List<Parameter> list, int i);

    private static native int nGetRefractionMode(long j);

    private static native int nGetRefractionType(long j);

    private static native int nGetRequiredAttributes(long j);

    private static native int nGetShading(long j);

    private static native float nGetSpecularAntiAliasingThreshold(long j);

    private static native float nGetSpecularAntiAliasingVariance(long j);

    private static native int nGetVertexDomain(long j);

    private static native boolean nHasParameter(long j, String str);

    private static native boolean nIsColorWriteEnabled(long j);

    private static native boolean nIsDepthCullingEnabled(long j);

    private static native boolean nIsDepthWriteEnabled(long j);

    private static native boolean nIsDoubleSided(long j);

    public void clearNativeObject() {
        this.mNativeObject = 0L;
    }

    public MaterialInstance createInstance() {
        long nCreateInstance = nCreateInstance(getNativeObject());
        if (nCreateInstance != 0) {
            return new MaterialInstance(this, nCreateInstance);
        }
        throw new IllegalStateException("Couldn't create MaterialInstance");
    }

    public BlendingMode getBlendingMode() {
        return BlendingMode.values()[nGetBlendingMode(getNativeObject())];
    }

    public CullingMode getCullingMode() {
        return CullingMode.values()[nGetCullingMode(getNativeObject())];
    }

    public MaterialInstance getDefaultInstance() {
        return this.mDefaultInstance;
    }

    public Interpolation getInterpolation() {
        return Interpolation.values()[nGetInterpolation(getNativeObject())];
    }

    public float getMaskThreshold() {
        return nGetMaskThreshold(getNativeObject());
    }

    public String getName() {
        return nGetName(getNativeObject());
    }

    public long getNativeObject() {
        long j = this.mNativeObject;
        if (j != 0) {
            return j;
        }
        throw new IllegalStateException("Calling method on destroyed Material");
    }

    public int getParameterCount() {
        return nGetParameterCount(getNativeObject());
    }

    public List<Parameter> getParameters() {
        int parameterCount = getParameterCount();
        ArrayList arrayList = new ArrayList(parameterCount);
        if (parameterCount > 0) {
            nGetParameters(getNativeObject(), arrayList, parameterCount);
        }
        return arrayList;
    }

    public RefractionMode getRefractionMode() {
        return RefractionMode.values()[nGetRefractionMode(getNativeObject())];
    }

    public RefractionType getRefractionType() {
        return RefractionType.values()[nGetRefractionType(getNativeObject())];
    }

    public Set<VertexBuffer.VertexAttribute> getRequiredAttributes() {
        if (this.mRequiredAttributes == null) {
            int nGetRequiredAttributes = nGetRequiredAttributes(getNativeObject());
            this.mRequiredAttributes = EnumSet.noneOf(VertexBuffer.VertexAttribute.class);
            VertexBuffer.VertexAttribute[] values = VertexBuffer.VertexAttribute.values();
            for (int i = 0; i < 16; i++) {
                if (((1 << i) & nGetRequiredAttributes) != 0) {
                    this.mRequiredAttributes.add(values[i]);
                }
            }
            this.mRequiredAttributes = Collections.unmodifiableSet(this.mRequiredAttributes);
        }
        return this.mRequiredAttributes;
    }

    public int getRequiredAttributesAsInt() {
        return nGetRequiredAttributes(getNativeObject());
    }

    public Shading getShading() {
        return Shading.values()[nGetShading(getNativeObject())];
    }

    public float getSpecularAntiAliasingThreshold() {
        return nGetSpecularAntiAliasingThreshold(getNativeObject());
    }

    public float getSpecularAntiAliasingVariance() {
        return nGetSpecularAntiAliasingVariance(getNativeObject());
    }

    public VertexDomain getVertexDomain() {
        return VertexDomain.values()[nGetVertexDomain(getNativeObject())];
    }

    public boolean hasParameter(String str) {
        return nHasParameter(getNativeObject(), str);
    }

    public boolean isColorWriteEnabled() {
        return nIsColorWriteEnabled(getNativeObject());
    }

    public boolean isDepthCullingEnabled() {
        return nIsDepthCullingEnabled(getNativeObject());
    }

    public boolean isDepthWriteEnabled() {
        return nIsDepthWriteEnabled(getNativeObject());
    }

    public boolean isDoubleSided() {
        return nIsDoubleSided(getNativeObject());
    }

    public void setDefaultParameter(String str, boolean z) {
        this.mDefaultInstance.setParameter(str, z);
    }

    public void setDefaultParameter(String str, float f2) {
        this.mDefaultInstance.setParameter(str, f2);
    }

    public void setDefaultParameter(String str, int i) {
        this.mDefaultInstance.setParameter(str, i);
    }

    public MaterialInstance createInstance(String str) {
        long nCreateInstanceWithName = nCreateInstanceWithName(getNativeObject(), str);
        if (nCreateInstanceWithName != 0) {
            return new MaterialInstance(this, nCreateInstanceWithName);
        }
        throw new IllegalStateException("Couldn't create MaterialInstance");
    }

    public void setDefaultParameter(String str, boolean z, boolean z2) {
        this.mDefaultInstance.setParameter(str, z, z2);
    }

    public void setDefaultParameter(String str, float f2, float f3) {
        this.mDefaultInstance.setParameter(str, f2, f3);
    }

    public void setDefaultParameter(String str, int i, int i2) {
        this.mDefaultInstance.setParameter(str, i, i2);
    }

    public void setDefaultParameter(String str, boolean z, boolean z2, boolean z3) {
        this.mDefaultInstance.setParameter(str, z, z2, z3);
    }

    public void setDefaultParameter(String str, float f2, float f3, float f4) {
        this.mDefaultInstance.setParameter(str, f2, f3, f4);
    }

    public void setDefaultParameter(String str, int i, int i2, int i3) {
        this.mDefaultInstance.setParameter(str, i, i2, i3);
    }

    public void setDefaultParameter(String str, boolean z, boolean z2, boolean z3, boolean z4) {
        this.mDefaultInstance.setParameter(str, z, z2, z3, z4);
    }

    public void setDefaultParameter(String str, float f2, float f3, float f4, float f5) {
        this.mDefaultInstance.setParameter(str, f2, f3, f4, f5);
    }

    public void setDefaultParameter(String str, int i, int i2, int i3, int i4) {
        this.mDefaultInstance.setParameter(str, i, i2, i3, i4);
    }

    public void setDefaultParameter(String str, MaterialInstance.BooleanElement booleanElement, boolean[] zArr, int i, int i2) {
        this.mDefaultInstance.setParameter(str, booleanElement, zArr, i, i2);
    }

    public void setDefaultParameter(String str, MaterialInstance.IntElement intElement, int[] iArr, int i, int i2) {
        this.mDefaultInstance.setParameter(str, intElement, iArr, i, i2);
    }

    public void setDefaultParameter(String str, MaterialInstance.FloatElement floatElement, float[] fArr, int i, int i2) {
        this.mDefaultInstance.setParameter(str, floatElement, fArr, i, i2);
    }

    public void setDefaultParameter(String str, Colors.RgbType rgbType, float f2, float f3, float f4) {
        this.mDefaultInstance.setParameter(str, rgbType, f2, f3, f4);
    }

    public void setDefaultParameter(String str, Colors.RgbaType rgbaType, float f2, float f3, float f4, float f5) {
        this.mDefaultInstance.setParameter(str, rgbaType, f2, f3, f4, f5);
    }

    public void setDefaultParameter(String str, Texture texture, TextureSampler textureSampler) {
        this.mDefaultInstance.setParameter(str, texture, textureSampler);
    }
}