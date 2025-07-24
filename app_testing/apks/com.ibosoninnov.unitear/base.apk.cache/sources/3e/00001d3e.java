package com.google.ar.sceneform.utilities;

/* loaded from: classes.dex */
public class EnvironmentalHdrParameters {
    public static final float DEFAULT_AMBIENT_SH_SCALE_FOR_FILAMENT = 1.0f;
    public static final float DEFAULT_DIRECT_INTENSITY_FOR_FILAMENT = 1.0f;
    public static final float DEFAULT_REFLECTION_SCALE_FOR_FILAMENT = 1.0f;
    private final float ambientShScaleForFilament;
    private final float directIntensityForFilament;
    private final float reflectionScaleForFilament;

    /* loaded from: classes.dex */
    public static class Builder {
        private float ambientShScaleForFilament;
        private float directIntensityForFilament;
        private float reflectionScaleForFilament;

        public EnvironmentalHdrParameters build() {
            return new EnvironmentalHdrParameters(this);
        }

        public Builder setAmbientShScaleForFilament(float f2) {
            this.ambientShScaleForFilament = f2;
            return this;
        }

        public Builder setDirectIntensityForFilament(float f2) {
            this.directIntensityForFilament = f2;
            return this;
        }

        public Builder setReflectionScaleForFilament(float f2) {
            this.reflectionScaleForFilament = f2;
            return this;
        }
    }

    public static Builder builder() {
        return new Builder();
    }

    public static EnvironmentalHdrParameters makeDefault() {
        return builder().setAmbientShScaleForFilament(1.0f).setDirectIntensityForFilament(1.0f).setReflectionScaleForFilament(1.0f).build();
    }

    public float getAmbientShScaleForFilament() {
        return this.ambientShScaleForFilament;
    }

    public float getDirectIntensityForFilament() {
        return this.directIntensityForFilament;
    }

    public float getReflectionScaleForFilament() {
        return this.reflectionScaleForFilament;
    }

    private EnvironmentalHdrParameters(Builder builder) {
        this.ambientShScaleForFilament = builder.ambientShScaleForFilament;
        this.directIntensityForFilament = builder.directIntensityForFilament;
        this.reflectionScaleForFilament = builder.reflectionScaleForFilament;
    }
}