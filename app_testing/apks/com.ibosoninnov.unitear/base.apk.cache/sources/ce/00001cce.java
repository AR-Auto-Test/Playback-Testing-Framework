package com.google.ar.sceneform.rendering;

import com.google.android.filament.Colors;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import com.google.ar.sceneform.common.TransformProvider;
import com.google.ar.sceneform.math.Vector3;
import com.google.ar.sceneform.utilities.AndroidPreconditions;
import java.util.ArrayList;
import java.util.Iterator;

/* loaded from: classes.dex */
public class Light {
    private static final float MIN_LIGHT_INTENSITY = 1.0E-4f;
    private final ArrayList<LightChangedListener> changedListeners;
    private final Color color;
    private Vector3 direction;
    private final boolean enableShadows;
    private float falloffRadius;
    private float intensity;
    private Vector3 position;
    private float spotlightConeInner;
    private float spotlightConeOuter;
    private final Type type;

    /* loaded from: classes.dex */
    public static final class Builder {
        private static final float DEFAULT_DIRECTIONAL_INTENSITY = 420.0f;
        private Color color;
        private Vector3 direction;
        private boolean enableShadows;
        private float falloffRadius;
        private float intensity;
        private Vector3 position;
        private float spotlightConeInner;
        private float spotlightConeOuter;
        private final Type type;

        public Light build() {
            return new Light(this);
        }

        public Builder setColor(Color color) {
            this.color = color;
            return this;
        }

        public Builder setColorTemperature(float f2) {
            float[] cct = Colors.cct(f2);
            setColor(new Color(cct[0], cct[1], cct[2]));
            return this;
        }

        public Builder setFalloffRadius(float f2) {
            this.falloffRadius = f2;
            return this;
        }

        public Builder setInnerConeAngle(float f2) {
            this.spotlightConeInner = f2;
            return this;
        }

        public Builder setIntensity(float f2) {
            this.intensity = f2;
            return this;
        }

        public Builder setOuterConeAngle(float f2) {
            this.spotlightConeOuter = f2;
            return this;
        }

        public Builder setShadowCastingEnabled(boolean z) {
            this.enableShadows = z;
            return this;
        }

        private Builder(Type type) {
            this.enableShadows = false;
            this.position = new Vector3(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
            this.direction = new Vector3(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, -1.0f);
            this.color = new Color(1.0f, 1.0f, 1.0f);
            this.intensity = 2500.0f;
            this.falloffRadius = 10.0f;
            this.spotlightConeInner = 0.5f;
            this.spotlightConeOuter = 0.6f;
            this.type = type;
            if (type == Type.DIRECTIONAL) {
                this.intensity = DEFAULT_DIRECTIONAL_INTENSITY;
            }
        }
    }

    /* loaded from: classes.dex */
    public interface LightChangedListener {
        void onChange();
    }

    /* loaded from: classes.dex */
    public enum Type {
        POINT,
        DIRECTIONAL,
        SPOTLIGHT,
        FOCUSED_SPOTLIGHT
    }

    public static Builder builder(Type type) {
        AndroidPreconditions.checkMinAndroidApiLevel();
        return new Builder(type);
    }

    private void fireChangedListeners() {
        Iterator<LightChangedListener> it = this.changedListeners.iterator();
        while (it.hasNext()) {
            it.next().onChange();
        }
    }

    public void addChangedListener(LightChangedListener lightChangedListener) {
        this.changedListeners.add(lightChangedListener);
    }

    public LightInstance createInstance(TransformProvider transformProvider) {
        return new LightInstance(this, transformProvider);
    }

    public Color getColor() {
        return new Color(this.color);
    }

    public float getFalloffRadius() {
        return this.falloffRadius;
    }

    public float getInnerConeAngle() {
        return this.spotlightConeInner;
    }

    public float getIntensity() {
        return this.intensity;
    }

    public Vector3 getLocalDirection() {
        return new Vector3(this.direction);
    }

    public Vector3 getLocalPosition() {
        return new Vector3(this.position);
    }

    public float getOuterConeAngle() {
        return this.spotlightConeOuter;
    }

    public Type getType() {
        return this.type;
    }

    public boolean isShadowCastingEnabled() {
        return this.enableShadows;
    }

    public void removeChangedListener(LightChangedListener lightChangedListener) {
        this.changedListeners.remove(lightChangedListener);
    }

    public void setColor(Color color) {
        this.color.set(color);
        fireChangedListeners();
    }

    public void setColorTemperature(float f2) {
        float[] cct = Colors.cct(f2);
        setColor(new Color(cct[0], cct[1], cct[2]));
    }

    public void setFalloffRadius(float f2) {
        this.falloffRadius = f2;
        fireChangedListeners();
    }

    public void setInnerConeAngle(float f2) {
        this.spotlightConeInner = f2;
        fireChangedListeners();
    }

    public void setIntensity(float f2) {
        this.intensity = Math.max(f2, 1.0E-4f);
        fireChangedListeners();
    }

    public void setOuterConeAngle(float f2) {
        this.spotlightConeOuter = f2;
        fireChangedListeners();
    }

    private Light(Builder builder) {
        this.changedListeners = new ArrayList<>();
        this.type = builder.type;
        this.enableShadows = builder.enableShadows;
        this.position = builder.position;
        this.direction = builder.direction;
        this.color = builder.color;
        this.intensity = builder.intensity;
        this.falloffRadius = builder.falloffRadius;
        this.spotlightConeInner = builder.spotlightConeInner;
        this.spotlightConeOuter = builder.spotlightConeOuter;
    }
}