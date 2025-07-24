package com.google.ar.sceneform;

import com.google.ar.sceneform.math.Quaternion;
import com.google.ar.sceneform.math.Vector3;
import com.google.ar.sceneform.rendering.Color;
import com.google.ar.sceneform.rendering.Light;
import com.google.ar.sceneform.utilities.EnvironmentalHdrParameters;
import com.google.ar.sceneform.utilities.Preconditions;

/* loaded from: classes.dex */
public class Sun extends Node {
    public static final int DEFAULT_SUNLIGHT_COLOR = -863292;
    public static final Vector3 DEFAULT_SUNLIGHT_DIRECTION = new Vector3(0.7f, -1.0f, -0.8f);
    private static final float LIGHT_ESTIMATE_OFFSET = 0.0f;
    private static final float LIGHT_ESTIMATE_SCALE = 1.8f;
    private float baseIntensity = 0.0f;

    public Sun() {
    }

    private void setupDefaultLighting(SceneView sceneView) {
        Preconditions.checkNotNull(sceneView, "Parameter \"view\" was null.");
        Color color = new Color((int) DEFAULT_SUNLIGHT_COLOR);
        setLookDirection(DEFAULT_SUNLIGHT_DIRECTION.normalized());
        Light build = Light.builder(Light.Type.DIRECTIONAL).setColor(color).setShadowCastingEnabled(true).build();
        if (build != null) {
            setLight(build);
            return;
        }
        throw new AssertionError("Failed to create the default sunlight.");
    }

    public void setEnvironmentalHdrLightEstimate(float[] fArr, Color color, float f2, float f3, EnvironmentalHdrParameters environmentalHdrParameters) {
        Light light = getLight();
        if (light == null) {
            return;
        }
        light.setColor(color);
        light.setIntensity((environmentalHdrParameters.getDirectIntensityForFilament() * f2) / f3);
        setWorldRotation(Quaternion.rotationBetweenVectors(Vector3.forward(), new Vector3(-fArr[0], -Math.abs(fArr[1]), -fArr[2]).normalized()));
    }

    public void setLightEstimate(Color color, float f2) {
        Light light = getLight();
        if (light == null) {
            return;
        }
        if (this.baseIntensity == 0.0f) {
            this.baseIntensity = light.getIntensity();
        }
        float min = Math.min((f2 * LIGHT_ESTIMATE_SCALE) + 0.0f, 1.0f) * this.baseIntensity;
        Color color2 = new Color((int) DEFAULT_SUNLIGHT_COLOR);
        color2.r *= color.r;
        color2.f5628g *= color.f5628g;
        color2.f5627b *= color.f5627b;
        light.setColor(color2);
        light.setIntensity(min);
    }

    @Override // com.google.ar.sceneform.Node
    public void setParent(NodeParent nodeParent) {
        throw new UnsupportedOperationException("Sun's parent cannot be changed, it is always the scene.");
    }

    public Sun(Scene scene) {
        Preconditions.checkNotNull(scene, "Parameter \"scene\" was null.");
        super.setParent(scene);
        setupDefaultLighting(scene.getView());
    }
}