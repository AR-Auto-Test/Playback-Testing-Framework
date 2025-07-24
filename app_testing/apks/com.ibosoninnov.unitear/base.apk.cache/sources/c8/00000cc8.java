package com.google.android.filament;

import com.google.android.filament.Colors;
import com.google.android.filament.Material;

/* loaded from: classes.dex */
public class MaterialInstance {
    private Material mMaterial;
    private String mName;
    private long mNativeMaterial;
    private long mNativeObject;

    /* loaded from: classes.dex */
    public enum BooleanElement {
        BOOL,
        BOOL2,
        BOOL3,
        BOOL4
    }

    /* loaded from: classes.dex */
    public enum FloatElement {
        FLOAT,
        FLOAT2,
        FLOAT3,
        FLOAT4,
        MAT3,
        MAT4
    }

    /* loaded from: classes.dex */
    public enum IntElement {
        INT,
        INT2,
        INT3,
        INT4
    }

    public MaterialInstance(Engine engine, long j) {
        this.mNativeObject = j;
        this.mNativeMaterial = nGetMaterial(j);
    }

    private static native long nGetMaterial(long j);

    private static native String nGetName(long j);

    private static native void nSetBooleanParameterArray(long j, String str, int i, boolean[] zArr, int i2, int i3);

    private static native void nSetColorWrite(long j, boolean z);

    private static native void nSetCullingMode(long j, long j2);

    private static native void nSetDepthCulling(long j, boolean z);

    private static native void nSetDepthWrite(long j, boolean z);

    private static native void nSetDoubleSided(long j, boolean z);

    private static native void nSetFloatParameterArray(long j, String str, int i, float[] fArr, int i2, int i3);

    private static native void nSetIntParameterArray(long j, String str, int i, int[] iArr, int i2, int i3);

    private static native void nSetMaskThreshold(long j, float f2);

    private static native void nSetParameterBool(long j, String str, boolean z);

    private static native void nSetParameterBool2(long j, String str, boolean z, boolean z2);

    private static native void nSetParameterBool3(long j, String str, boolean z, boolean z2, boolean z3);

    private static native void nSetParameterBool4(long j, String str, boolean z, boolean z2, boolean z3, boolean z4);

    private static native void nSetParameterFloat(long j, String str, float f2);

    private static native void nSetParameterFloat2(long j, String str, float f2, float f3);

    private static native void nSetParameterFloat3(long j, String str, float f2, float f3, float f4);

    private static native void nSetParameterFloat4(long j, String str, float f2, float f3, float f4, float f5);

    private static native void nSetParameterInt(long j, String str, int i);

    private static native void nSetParameterInt2(long j, String str, int i, int i2);

    private static native void nSetParameterInt3(long j, String str, int i, int i2, int i3);

    private static native void nSetParameterInt4(long j, String str, int i, int i2, int i3, int i4);

    private static native void nSetParameterTexture(long j, String str, long j2, int i);

    private static native void nSetPolygonOffset(long j, float f2, float f3);

    private static native void nSetScissor(long j, int i, int i2, int i3, int i4);

    private static native void nSetSpecularAntiAliasingThreshold(long j, float f2);

    private static native void nSetSpecularAntiAliasingVariance(long j, float f2);

    private static native void nUnsetScissor(long j);

    public void clearNativeObject() {
        this.mNativeObject = 0L;
    }

    public Material getMaterial() {
        if (this.mMaterial == null) {
            this.mMaterial = new Material(this.mNativeMaterial);
        }
        return this.mMaterial;
    }

    public String getName() {
        if (this.mName == null) {
            this.mName = nGetName(getNativeObject());
        }
        return this.mName;
    }

    public long getNativeObject() {
        long j = this.mNativeObject;
        if (j != 0) {
            return j;
        }
        throw new IllegalStateException("Calling method on destroyed MaterialInstance");
    }

    public void setColorWrite(boolean z) {
        nSetColorWrite(getNativeObject(), z);
    }

    public void setCullingMode(Material.CullingMode cullingMode) {
        nSetCullingMode(getNativeObject(), cullingMode.ordinal());
    }

    public void setDepthCulling(boolean z) {
        nSetDepthCulling(getNativeObject(), z);
    }

    public void setDepthWrite(boolean z) {
        nSetDepthWrite(getNativeObject(), z);
    }

    public void setDoubleSided(boolean z) {
        nSetDoubleSided(getNativeObject(), z);
    }

    public void setMaskThreshold(float f2) {
        nSetMaskThreshold(getNativeObject(), f2);
    }

    public void setParameter(String str, boolean z) {
        nSetParameterBool(getNativeObject(), str, z);
    }

    public void setPolygonOffset(float f2, float f3) {
        nSetPolygonOffset(getNativeObject(), f2, f3);
    }

    public void setScissor(int i, int i2, int i3, int i4) {
        nSetScissor(getNativeObject(), i, i2, i3, i4);
    }

    public void setSpecularAntiAliasingThreshold(float f2) {
        nSetSpecularAntiAliasingThreshold(getNativeObject(), f2);
    }

    public void setSpecularAntiAliasingVariance(float f2) {
        nSetSpecularAntiAliasingVariance(getNativeObject(), f2);
    }

    public void unsetScissor() {
        nUnsetScissor(getNativeObject());
    }

    public void setParameter(String str, float f2) {
        nSetParameterFloat(getNativeObject(), str, f2);
    }

    public void setParameter(String str, int i) {
        nSetParameterInt(getNativeObject(), str, i);
    }

    public MaterialInstance(Material material, long j) {
        this.mMaterial = material;
        this.mNativeObject = j;
    }

    public void setParameter(String str, boolean z, boolean z2) {
        nSetParameterBool2(getNativeObject(), str, z, z2);
    }

    public void setParameter(String str, float f2, float f3) {
        nSetParameterFloat2(getNativeObject(), str, f2, f3);
    }

    public void setParameter(String str, int i, int i2) {
        nSetParameterInt2(getNativeObject(), str, i, i2);
    }

    public MaterialInstance(long j) {
        this.mNativeObject = j;
        this.mNativeMaterial = nGetMaterial(j);
    }

    public void setParameter(String str, boolean z, boolean z2, boolean z3) {
        nSetParameterBool3(getNativeObject(), str, z, z2, z3);
    }

    public void setParameter(String str, float f2, float f3, float f4) {
        nSetParameterFloat3(getNativeObject(), str, f2, f3, f4);
    }

    public void setParameter(String str, int i, int i2, int i3) {
        nSetParameterInt3(getNativeObject(), str, i, i2, i3);
    }

    public void setParameter(String str, boolean z, boolean z2, boolean z3, boolean z4) {
        nSetParameterBool4(getNativeObject(), str, z, z2, z3, z4);
    }

    public void setParameter(String str, float f2, float f3, float f4, float f5) {
        nSetParameterFloat4(getNativeObject(), str, f2, f3, f4, f5);
    }

    public void setParameter(String str, int i, int i2, int i3, int i4) {
        nSetParameterInt4(getNativeObject(), str, i, i2, i3, i4);
    }

    public void setParameter(String str, Texture texture, TextureSampler textureSampler) {
        nSetParameterTexture(getNativeObject(), str, texture.getNativeObject(), textureSampler.mSampler);
    }

    public void setParameter(String str, BooleanElement booleanElement, boolean[] zArr, int i, int i2) {
        nSetBooleanParameterArray(getNativeObject(), str, booleanElement.ordinal(), zArr, i, i2);
    }

    public void setParameter(String str, IntElement intElement, int[] iArr, int i, int i2) {
        nSetIntParameterArray(getNativeObject(), str, intElement.ordinal(), iArr, i, i2);
    }

    public void setParameter(String str, FloatElement floatElement, float[] fArr, int i, int i2) {
        nSetFloatParameterArray(getNativeObject(), str, floatElement.ordinal(), fArr, i, i2);
    }

    public void setParameter(String str, Colors.RgbType rgbType, float f2, float f3, float f4) {
        float[] linear = Colors.toLinear(rgbType, f2, f3, f4);
        nSetParameterFloat3(getNativeObject(), str, linear[0], linear[1], linear[2]);
    }

    public void setParameter(String str, Colors.RgbaType rgbaType, float f2, float f3, float f4, float f5) {
        float[] linear = Colors.toLinear(rgbaType, f2, f3, f4, f5);
        nSetParameterFloat4(getNativeObject(), str, linear[0], linear[1], linear[2], linear[3]);
    }
}