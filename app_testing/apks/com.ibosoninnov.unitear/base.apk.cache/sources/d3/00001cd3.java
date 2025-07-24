package com.google.ar.sceneform.rendering;

import android.util.Log;
import com.google.android.filament.Entity;
import com.google.android.filament.EntityManager;
import com.google.android.filament.LightManager;
import com.google.ar.sceneform.common.TransformProvider;
import com.google.ar.sceneform.math.Matrix;
import com.google.ar.sceneform.math.Vector3;
import com.google.ar.sceneform.rendering.Light;
import com.google.ar.sceneform.rendering.LightInstance;
import com.google.ar.sceneform.utilities.AndroidPreconditions;

/* loaded from: classes.dex */
public class LightInstance {
    private static final String TAG = "LightInstance";
    private LightInstanceChangeListener changeListener = new LightInstanceChangeListener();
    private boolean dirty = false;
    @Entity
    private final int entity;
    private final Light light;
    private Vector3 localDirection;
    private Vector3 localPosition;
    private Renderer renderer;
    private TransformProvider transformProvider;

    /* loaded from: classes.dex */
    public class LightInstanceChangeListener implements Light.LightChangedListener {
        private LightInstanceChangeListener() {
        }

        @Override // com.google.ar.sceneform.rendering.Light.LightChangedListener
        public void onChange() {
            LightInstance.this.dirty = true;
        }
    }

    public LightInstance(Light light, TransformProvider transformProvider) {
        this.transformProvider = null;
        this.light = light;
        this.transformProvider = transformProvider;
        this.localPosition = light.getLocalPosition();
        this.localDirection = light.getLocalDirection();
        light.addChangedListener(this.changeListener);
        int create = EntityManager.get().create();
        this.entity = create;
        IEngine engine = EngineInstance.getEngine();
        if (light.getType() == Light.Type.POINT) {
            new LightManager.Builder(LightManager.Type.POINT).position(light.getLocalPosition().x, light.getLocalPosition().y, light.getLocalPosition().z).color(light.getColor().r, light.getColor().f5628g, light.getColor().f5627b).intensity(light.getIntensity()).falloff(light.getFalloffRadius()).castShadows(light.isShadowCastingEnabled()).build(engine.getFilamentEngine(), create);
        } else if (light.getType() == Light.Type.DIRECTIONAL) {
            new LightManager.Builder(LightManager.Type.DIRECTIONAL).direction(light.getLocalDirection().x, light.getLocalDirection().y, light.getLocalDirection().z).color(light.getColor().r, light.getColor().f5628g, light.getColor().f5627b).intensity(light.getIntensity()).castShadows(light.isShadowCastingEnabled()).build(engine.getFilamentEngine(), create);
        } else if (light.getType() == Light.Type.SPOTLIGHT) {
            new LightManager.Builder(LightManager.Type.SPOT).position(light.getLocalPosition().x, light.getLocalPosition().y, light.getLocalPosition().z).direction(light.getLocalDirection().x, light.getLocalDirection().y, light.getLocalDirection().z).color(light.getColor().r, light.getColor().f5628g, light.getColor().f5627b).intensity(light.getIntensity()).spotLightCone(Math.min(light.getInnerConeAngle(), light.getOuterConeAngle()), light.getOuterConeAngle()).castShadows(light.isShadowCastingEnabled()).build(engine.getFilamentEngine(), create);
        } else if (light.getType() == Light.Type.FOCUSED_SPOTLIGHT) {
            new LightManager.Builder(LightManager.Type.FOCUSED_SPOT).position(light.getLocalPosition().x, light.getLocalPosition().y, light.getLocalPosition().z).direction(light.getLocalDirection().x, light.getLocalDirection().y, light.getLocalDirection().z).color(light.getColor().r, light.getColor().f5628g, light.getColor().f5627b).intensity(light.getIntensity()).spotLightCone(Math.min(light.getInnerConeAngle(), light.getOuterConeAngle()), light.getOuterConeAngle()).castShadows(light.isShadowCastingEnabled()).build(engine.getFilamentEngine(), create);
        } else {
            throw new UnsupportedOperationException("Unsupported light type.");
        }
    }

    private static boolean lightTypeRequiresDirection(Light.Type type) {
        return type == Light.Type.SPOTLIGHT || type == Light.Type.FOCUSED_SPOTLIGHT || type == Light.Type.DIRECTIONAL;
    }

    private static boolean lightTypeRequiresPosition(Light.Type type) {
        return type == Light.Type.POINT || type == Light.Type.SPOTLIGHT || type == Light.Type.FOCUSED_SPOTLIGHT;
    }

    private void updateProperties() {
        if (this.dirty) {
            this.dirty = false;
            LightManager lightManager = EngineInstance.getEngine().getLightManager();
            int lightManager2 = lightManager.getInstance(this.entity);
            this.localPosition = this.light.getLocalPosition();
            this.localDirection = this.light.getLocalDirection();
            if (this.transformProvider == null) {
                if (lightTypeRequiresPosition(this.light.getType())) {
                    Vector3 vector3 = this.localPosition;
                    lightManager.setPosition(lightManager2, vector3.x, vector3.y, vector3.z);
                }
                if (lightTypeRequiresDirection(this.light.getType())) {
                    Vector3 vector32 = this.localDirection;
                    lightManager.setDirection(lightManager2, vector32.x, vector32.y, vector32.z);
                }
            }
            lightManager.setColor(lightManager2, this.light.getColor().r, this.light.getColor().f5628g, this.light.getColor().f5627b);
            lightManager.setIntensity(lightManager2, this.light.getIntensity());
            if (this.light.getType() == Light.Type.POINT) {
                lightManager.setFalloff(lightManager2, this.light.getFalloffRadius());
            } else if (this.light.getType() == Light.Type.SPOTLIGHT || this.light.getType() == Light.Type.FOCUSED_SPOTLIGHT) {
                lightManager.setSpotLightCone(lightManager2, Math.min(this.light.getInnerConeAngle(), this.light.getOuterConeAngle()), this.light.getOuterConeAngle());
            }
        }
    }

    public void attachToRenderer(Renderer renderer) {
        renderer.addLight(this);
        this.renderer = renderer;
    }

    public void detachFromRenderer() {
        Renderer renderer = this.renderer;
        if (renderer != null) {
            renderer.removeLight(this);
        }
    }

    public void dispose() {
        AndroidPreconditions.checkUiThread();
        Light light = this.light;
        if (light != null) {
            light.removeChangedListener(this.changeListener);
            this.changeListener = null;
        }
        IEngine engine = EngineInstance.getEngine();
        if (engine == null || !engine.isValid()) {
            return;
        }
        engine.getLightManager().destroy(this.entity);
        EntityManager.get().destroy(this.entity);
    }

    public void finalize() {
        try {
            try {
                ThreadPools.getMainExecutor().execute(new Runnable() { // from class: c.d.b.a.q.d
                    @Override // java.lang.Runnable
                    public final void run() {
                        LightInstance.this.dispose();
                    }
                });
            } catch (Exception e2) {
                Log.e(TAG, "Error while Finalizing Light Instance.", e2);
            }
        } finally {
            super.finalize();
        }
    }

    @Entity
    public int getEntity() {
        return this.entity;
    }

    public Light getLight() {
        return this.light;
    }

    public void updateTransform() {
        updateProperties();
        if (this.transformProvider == null) {
            return;
        }
        LightManager lightManager = EngineInstance.getEngine().getLightManager();
        int lightManager2 = lightManager.getInstance(this.entity);
        Matrix worldModelMatrix = this.transformProvider.getWorldModelMatrix();
        if (lightTypeRequiresPosition(this.light.getType())) {
            Vector3 transformPoint = worldModelMatrix.transformPoint(this.localPosition);
            lightManager.setPosition(lightManager2, transformPoint.x, transformPoint.y, transformPoint.z);
        }
        if (lightTypeRequiresDirection(this.light.getType())) {
            Vector3 transformDirection = worldModelMatrix.transformDirection(this.localDirection);
            lightManager.setDirection(lightManager2, transformDirection.x, transformDirection.y, transformDirection.z);
        }
    }
}