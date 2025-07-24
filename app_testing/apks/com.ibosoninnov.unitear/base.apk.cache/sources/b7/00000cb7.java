package com.google.android.filament;

import com.google.android.material.internal.StaticLayoutBuilderCompat;

/* loaded from: classes.dex */
public class LightManager {
    public static final float EFFICIENCY_FLUORESCENT = 0.0878f;
    public static final float EFFICIENCY_HALOGEN = 0.0707f;
    public static final float EFFICIENCY_INCANDESCENT = 0.022f;
    public static final float EFFICIENCY_LED = 0.1171f;
    private long mNativeObject;

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
                LightManager.nDestroyBuilder(this.mNativeObject);
            }
        }

        public Builder(Type type) {
            long nCreateBuilder = LightManager.nCreateBuilder(type.ordinal());
            this.mNativeBuilder = nCreateBuilder;
            this.mFinalizer = new BuilderFinalizer(nCreateBuilder);
        }

        public void build(Engine engine, @Entity int i) {
            if (LightManager.nBuilderBuild(this.mNativeBuilder, engine.getNativeObject(), i)) {
                return;
            }
            throw new IllegalStateException("Couldn't create Light component for entity " + i + ", see log.");
        }

        public Builder castLight(boolean z) {
            LightManager.nBuilderCastLight(this.mNativeBuilder, z);
            return this;
        }

        public Builder castShadows(boolean z) {
            LightManager.nBuilderCastShadows(this.mNativeBuilder, z);
            return this;
        }

        public Builder color(float f2, float f3, float f4) {
            LightManager.nBuilderColor(this.mNativeBuilder, f2, f3, f4);
            return this;
        }

        public Builder direction(float f2, float f3, float f4) {
            LightManager.nBuilderDirection(this.mNativeBuilder, f2, f3, f4);
            return this;
        }

        public Builder falloff(float f2) {
            LightManager.nBuilderFalloff(this.mNativeBuilder, f2);
            return this;
        }

        public Builder intensity(float f2) {
            LightManager.nBuilderIntensity(this.mNativeBuilder, f2);
            return this;
        }

        public Builder intensityCandela(float f2) {
            LightManager.nBuilderIntensityCandela(this.mNativeBuilder, f2);
            return this;
        }

        public Builder position(float f2, float f3, float f4) {
            LightManager.nBuilderPosition(this.mNativeBuilder, f2, f3, f4);
            return this;
        }

        public Builder shadowOptions(ShadowOptions shadowOptions) {
            LightManager.nBuilderShadowOptions(this.mNativeBuilder, shadowOptions.mapSize, shadowOptions.constantBias, shadowOptions.normalBias, shadowOptions.shadowFar, shadowOptions.shadowNearHint, shadowOptions.shadowFarHint, shadowOptions.stable, shadowOptions.screenSpaceContactShadows, shadowOptions.stepCount, shadowOptions.maxShadowDistance);
            return this;
        }

        public Builder spotLightCone(float f2, float f3) {
            LightManager.nBuilderSpotLightCone(this.mNativeBuilder, f2, f3);
            return this;
        }

        public Builder sunAngularRadius(float f2) {
            LightManager.nBuilderAngularRadius(this.mNativeBuilder, f2);
            return this;
        }

        public Builder sunHaloFalloff(float f2) {
            LightManager.nBuilderHaloFalloff(this.mNativeBuilder, f2);
            return this;
        }

        public Builder sunHaloSize(float f2) {
            LightManager.nBuilderHaloSize(this.mNativeBuilder, f2);
            return this;
        }

        public Builder intensity(float f2, float f3) {
            LightManager.nBuilderIntensity(this.mNativeBuilder, f2, f3);
            return this;
        }
    }

    /* loaded from: classes.dex */
    public static class ShadowOptions {
        public int mapSize = 1024;
        public float constantBias = 0.05f;
        public float normalBias = 0.4f;
        public float shadowFar = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        public float shadowNearHint = 1.0f;
        public float shadowFarHint = 100.0f;
        public boolean stable = true;
        public boolean screenSpaceContactShadows = false;
        public int stepCount = 8;
        public float maxShadowDistance = 0.3f;
    }

    /* loaded from: classes.dex */
    public enum Type {
        SUN,
        DIRECTIONAL,
        POINT,
        FOCUSED_SPOT,
        SPOT
    }

    public LightManager(long j) {
        this.mNativeObject = j;
    }

    /* JADX INFO: Access modifiers changed from: private */
    public static native void nBuilderAngularRadius(long j, float f2);

    /* JADX INFO: Access modifiers changed from: private */
    public static native boolean nBuilderBuild(long j, long j2, int i);

    /* JADX INFO: Access modifiers changed from: private */
    public static native void nBuilderCastLight(long j, boolean z);

    /* JADX INFO: Access modifiers changed from: private */
    public static native void nBuilderCastShadows(long j, boolean z);

    /* JADX INFO: Access modifiers changed from: private */
    public static native void nBuilderColor(long j, float f2, float f3, float f4);

    /* JADX INFO: Access modifiers changed from: private */
    public static native void nBuilderDirection(long j, float f2, float f3, float f4);

    /* JADX INFO: Access modifiers changed from: private */
    public static native void nBuilderFalloff(long j, float f2);

    /* JADX INFO: Access modifiers changed from: private */
    public static native void nBuilderHaloFalloff(long j, float f2);

    /* JADX INFO: Access modifiers changed from: private */
    public static native void nBuilderHaloSize(long j, float f2);

    /* JADX INFO: Access modifiers changed from: private */
    public static native void nBuilderIntensity(long j, float f2);

    /* JADX INFO: Access modifiers changed from: private */
    public static native void nBuilderIntensity(long j, float f2, float f3);

    /* JADX INFO: Access modifiers changed from: private */
    public static native void nBuilderIntensityCandela(long j, float f2);

    /* JADX INFO: Access modifiers changed from: private */
    public static native void nBuilderPosition(long j, float f2, float f3, float f4);

    /* JADX INFO: Access modifiers changed from: private */
    public static native void nBuilderShadowOptions(long j, int i, float f2, float f3, float f4, float f5, float f6, boolean z, boolean z2, int i2, float f7);

    /* JADX INFO: Access modifiers changed from: private */
    public static native void nBuilderSpotLightCone(long j, float f2, float f3);

    /* JADX INFO: Access modifiers changed from: private */
    public static native long nCreateBuilder(int i);

    private static native void nDestroy(long j, int i);

    /* JADX INFO: Access modifiers changed from: private */
    public static native void nDestroyBuilder(long j);

    private static native void nGetColor(long j, int i, float[] fArr);

    private static native int nGetComponentCount(long j);

    private static native void nGetDirection(long j, int i, float[] fArr);

    private static native float nGetFalloff(long j, int i);

    private static native int nGetInstance(long j, int i);

    private static native float nGetIntensity(long j, int i);

    private static native void nGetPosition(long j, int i, float[] fArr);

    private static native float nGetSunAngularRadius(long j, int i);

    private static native float nGetSunHaloFalloff(long j, int i);

    private static native float nGetSunHaloSize(long j, int i);

    private static native int nGetType(long j, int i);

    private static native boolean nHasComponent(long j, int i);

    private static native boolean nIsShadowCaster(long j, int i);

    private static native void nSetColor(long j, int i, float f2, float f3, float f4);

    private static native void nSetDirection(long j, int i, float f2, float f3, float f4);

    private static native void nSetFalloff(long j, int i, float f2);

    private static native void nSetIntensity(long j, int i, float f2);

    private static native void nSetIntensity(long j, int i, float f2, float f3);

    private static native void nSetIntensityCandela(long j, int i, float f2);

    private static native void nSetPosition(long j, int i, float f2, float f3, float f4);

    private static native void nSetShadowCaster(long j, int i, boolean z);

    private static native void nSetSpotLightCone(long j, int i, float f2, float f3);

    private static native void nSetSunAngularRadius(long j, int i, float f2);

    private static native void nSetSunHaloFalloff(long j, int i, float f2);

    private static native void nSetSunHaloSize(long j, int i, float f2);

    public void destroy(@Entity int i) {
        nDestroy(this.mNativeObject, i);
    }

    public float[] getColor(@EntityInstance int i, float[] fArr) {
        float[] assertFloat3 = Asserts.assertFloat3(fArr);
        nGetColor(this.mNativeObject, i, assertFloat3);
        return assertFloat3;
    }

    public int getComponentCount() {
        return nGetComponentCount(this.mNativeObject);
    }

    public float[] getDirection(@EntityInstance int i, float[] fArr) {
        float[] assertFloat3 = Asserts.assertFloat3(fArr);
        nGetDirection(this.mNativeObject, i, assertFloat3);
        return assertFloat3;
    }

    public float getFalloff(@EntityInstance int i) {
        return nGetFalloff(this.mNativeObject, i);
    }

    @EntityInstance
    public int getInstance(@Entity int i) {
        return nGetInstance(this.mNativeObject, i);
    }

    public float getIntensity(@EntityInstance int i) {
        return nGetIntensity(this.mNativeObject, i);
    }

    public long getNativeObject() {
        return this.mNativeObject;
    }

    public float[] getPosition(@EntityInstance int i, float[] fArr) {
        float[] assertFloat3 = Asserts.assertFloat3(fArr);
        nGetPosition(this.mNativeObject, i, assertFloat3);
        return assertFloat3;
    }

    public float getSunAngularRadius(@EntityInstance int i) {
        return nGetSunAngularRadius(this.mNativeObject, i);
    }

    public float getSunHaloFalloff(@EntityInstance int i) {
        return nGetSunHaloFalloff(this.mNativeObject, i);
    }

    public float getSunHaloSize(@EntityInstance int i) {
        return nGetSunHaloSize(this.mNativeObject, i);
    }

    public Type getType(@EntityInstance int i) {
        return Type.values()[nGetType(this.mNativeObject, i)];
    }

    public boolean hasComponent(@Entity int i) {
        return nHasComponent(this.mNativeObject, i);
    }

    public boolean isDirectional(@EntityInstance int i) {
        Type type = getType(i);
        return type == Type.DIRECTIONAL || type == Type.SUN;
    }

    public boolean isPointLight(@EntityInstance int i) {
        return getType(i) == Type.POINT;
    }

    public boolean isShadowCaster(@EntityInstance int i) {
        return nIsShadowCaster(this.mNativeObject, i);
    }

    public boolean isSpotLight(@EntityInstance int i) {
        Type type = getType(i);
        return type == Type.SPOT || type == Type.FOCUSED_SPOT;
    }

    public void setColor(@EntityInstance int i, float f2, float f3, float f4) {
        nSetColor(this.mNativeObject, i, f2, f3, f4);
    }

    public void setDirection(@EntityInstance int i, float f2, float f3, float f4) {
        nSetDirection(this.mNativeObject, i, f2, f3, f4);
    }

    public void setFalloff(@EntityInstance int i, float f2) {
        nSetFalloff(this.mNativeObject, i, f2);
    }

    public void setIntensity(@EntityInstance int i, float f2) {
        nSetIntensity(this.mNativeObject, i, f2);
    }

    public void setIntensityCandela(@EntityInstance int i, float f2) {
        nSetIntensityCandela(this.mNativeObject, i, f2);
    }

    public void setPosition(@EntityInstance int i, float f2, float f3, float f4) {
        nSetPosition(this.mNativeObject, i, f2, f3, f4);
    }

    public void setShadowCaster(@EntityInstance int i, boolean z) {
        nSetShadowCaster(this.mNativeObject, i, z);
    }

    public void setSpotLightCone(@EntityInstance int i, float f2, float f3) {
        nSetSpotLightCone(this.mNativeObject, i, f2, f3);
    }

    public void setSunAngularRadius(@EntityInstance int i, float f2) {
        nSetSunAngularRadius(this.mNativeObject, i, f2);
    }

    public void setSunHaloFalloff(@EntityInstance int i, float f2) {
        nSetSunHaloFalloff(this.mNativeObject, i, f2);
    }

    public void setSunHaloSize(@EntityInstance int i, float f2) {
        nSetSunHaloSize(this.mNativeObject, i, f2);
    }

    public void setIntensity(@EntityInstance int i, float f2, float f3) {
        nSetIntensity(this.mNativeObject, i, f2, f3);
    }
}