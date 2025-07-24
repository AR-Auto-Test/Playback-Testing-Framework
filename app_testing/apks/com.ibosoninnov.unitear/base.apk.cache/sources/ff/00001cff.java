package com.google.ar.sceneform.rendering;

import c.d.b.a.q.a0;
import com.google.ar.core.Frame;
import com.google.ar.core.HitResult;
import com.google.ar.core.Plane;
import com.google.ar.core.Pose;
import com.google.ar.core.Trackable;
import com.google.ar.core.TrackingState;
import com.google.ar.sceneform.math.Vector3;
import com.google.ar.sceneform.rendering.Material;
import com.google.ar.sceneform.rendering.PlaneRenderer;
import com.google.ar.sceneform.rendering.RenderingResources;
import com.google.ar.sceneform.rendering.Texture;
import java.util.Collection;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionStage;
import java.util.function.BiFunction;
import java.util.function.Consumer;
import java.util.function.Function;

/* loaded from: classes.dex */
public class PlaneRenderer {
    private static final float BASE_UV_SCALE = 8.0f;
    private static final float DEFAULT_TEXTURE_HEIGHT = 513.0f;
    private static final float DEFAULT_TEXTURE_WIDTH = 293.0f;
    public static final String MATERIAL_COLOR = "color";
    private static final String MATERIAL_SPOTLIGHT_FOCUS_POINT = "focusPoint";
    public static final String MATERIAL_SPOTLIGHT_RADIUS = "radius";
    public static final String MATERIAL_TEXTURE = "texture";
    public static final String MATERIAL_UV_SCALE = "uvScale";
    private static final float SPOTLIGHT_RADIUS = 0.5f;
    private static final String TAG = "PlaneRenderer";
    private CompletableFuture<Material> planeMaterialFuture;
    private final Renderer renderer;
    private Material shadowMaterial;
    private final Map<Plane, PlaneVisualizer> visualizerMap = new HashMap();
    private boolean isEnabled = true;
    private boolean isVisible = true;
    private boolean isShadowReceiver = true;
    private final Map<Plane, Material> materialOverrides = new HashMap();
    private float lastPlaneHitDistance = 4.0f;

    public PlaneRenderer(Renderer renderer) {
        this.renderer = renderer;
        loadPlaneMaterial();
        loadShadowMaterial();
    }

    private Vector3 getFocusPoint(Frame frame, int i, int i2) {
        List<HitResult> hitTest = frame.hitTest(i / 2, i2 / 2);
        if (hitTest != null && !hitTest.isEmpty()) {
            for (HitResult hitResult : hitTest) {
                Trackable trackable = hitResult.getTrackable();
                Pose hitPose = hitResult.getHitPose();
                if ((trackable instanceof Plane) && ((Plane) trackable).isPoseInPolygon(hitPose)) {
                    Vector3 vector3 = new Vector3(hitPose.tx(), hitPose.ty(), hitPose.tz());
                    this.lastPlaneHitDistance = hitResult.getDistance();
                    return vector3;
                }
            }
        }
        Pose pose = frame.getCamera().getPose();
        Vector3 vector32 = new Vector3(pose.tx(), pose.ty(), pose.tz());
        float[] zAxis = pose.getZAxis();
        return Vector3.add(vector32, new Vector3(zAxis[0], zAxis[1], zAxis[2]).scaled(-this.lastPlaneHitDistance));
    }

    /* JADX DEBUG: Type inference failed for r0v6. Raw type applied. Possible types: java.util.concurrent.CompletableFuture<V>, java.util.concurrent.CompletableFuture<com.google.ar.sceneform.rendering.Material> */
    private void loadPlaneMaterial() {
        this.planeMaterialFuture = Material.builder().setSource(this.renderer.getContext(), RenderingResources.GetSceneformResource(this.renderer.getContext(), RenderingResources.Resource.PLANE_MATERIAL)).build().thenCombine((CompletionStage) Texture.builder().setSource(this.renderer.getContext(), RenderingResources.GetSceneformResource(this.renderer.getContext(), RenderingResources.Resource.PLANE)).setSampler(Texture.Sampler.builder().setMinMagFilter(Texture.Sampler.MagFilter.LINEAR).setWrapMode(Texture.Sampler.WrapMode.REPEAT).build()).build(), new BiFunction() { // from class: c.d.b.a.q.b0
            @Override // java.util.function.BiFunction
            public final Object apply(Object obj, Object obj2) {
                Material material = (Material) obj;
                PlaneRenderer.this.a(material, (Texture) obj2);
                return material;
            }
        });
    }

    private void loadShadowMaterial() {
        Material.builder().setSource(this.renderer.getContext(), RenderingResources.GetSceneformResource(this.renderer.getContext(), RenderingResources.Resource.PLANE_SHADOW_MATERIAL)).build().thenAccept(new Consumer() { // from class: c.d.b.a.q.c0
            @Override // java.util.function.Consumer
            public final void accept(Object obj) {
                PlaneRenderer.this.b((Material) obj);
            }
        }).exceptionally((Function<Throwable, ? extends Void>) a0.f4314a);
    }

    public /* synthetic */ Material a(Material material, Texture texture) {
        material.setTexture("texture", texture);
        material.setFloat3("color", 1.0f, 1.0f, 1.0f);
        material.setFloat2(MATERIAL_UV_SCALE, BASE_UV_SCALE, 4.569201f);
        for (Map.Entry<Plane, PlaneVisualizer> entry : this.visualizerMap.entrySet()) {
            if (!this.materialOverrides.containsKey(entry.getKey())) {
                entry.getValue().setPlaneMaterial(material);
            }
        }
        return material;
    }

    public /* synthetic */ void b(Material material) {
        this.shadowMaterial = material;
        for (PlaneVisualizer planeVisualizer : this.visualizerMap.values()) {
            planeVisualizer.setShadowMaterial(this.shadowMaterial);
        }
    }

    public CompletableFuture<Material> getMaterial() {
        return this.planeMaterialFuture;
    }

    public boolean isEnabled() {
        return this.isEnabled;
    }

    public boolean isShadowReceiver() {
        return this.isShadowReceiver;
    }

    public boolean isVisible() {
        return this.isVisible;
    }

    public void setEnabled(boolean z) {
        if (this.isEnabled != z) {
            this.isEnabled = z;
            for (PlaneVisualizer planeVisualizer : this.visualizerMap.values()) {
                planeVisualizer.setEnabled(this.isEnabled);
            }
        }
    }

    public void setShadowReceiver(boolean z) {
        if (this.isShadowReceiver != z) {
            this.isShadowReceiver = z;
            for (PlaneVisualizer planeVisualizer : this.visualizerMap.values()) {
                planeVisualizer.setShadowReceiver(this.isShadowReceiver);
            }
        }
    }

    public void setVisible(boolean z) {
        if (this.isVisible != z) {
            this.isVisible = z;
            for (PlaneVisualizer planeVisualizer : this.visualizerMap.values()) {
                planeVisualizer.setVisible(this.isVisible);
            }
        }
    }

    public void update(Frame frame, int i, int i2) {
        PlaneVisualizer planeVisualizer;
        Collection<Plane> updatedTrackables = frame.getUpdatedTrackables(Plane.class);
        Vector3 focusPoint = getFocusPoint(frame, i, i2);
        Material now = this.planeMaterialFuture.getNow(null);
        if (now != null) {
            now.setFloat3(MATERIAL_SPOTLIGHT_FOCUS_POINT, focusPoint);
            now.setFloat(MATERIAL_SPOTLIGHT_RADIUS, 0.5f);
        }
        for (Plane plane : updatedTrackables) {
            if (this.visualizerMap.containsKey(plane)) {
                planeVisualizer = this.visualizerMap.get(plane);
            } else {
                PlaneVisualizer planeVisualizer2 = new PlaneVisualizer(plane, this.renderer);
                Material material = this.materialOverrides.get(plane);
                if (material != null) {
                    planeVisualizer2.setPlaneMaterial(material);
                } else if (now != null) {
                    planeVisualizer2.setPlaneMaterial(now);
                }
                Material material2 = this.shadowMaterial;
                if (material2 != null) {
                    planeVisualizer2.setShadowMaterial(material2);
                }
                planeVisualizer2.setShadowReceiver(this.isShadowReceiver);
                planeVisualizer2.setVisible(this.isVisible);
                planeVisualizer2.setEnabled(this.isEnabled);
                this.visualizerMap.put(plane, planeVisualizer2);
                planeVisualizer = planeVisualizer2;
            }
            planeVisualizer.updatePlane();
        }
        Iterator<Map.Entry<Plane, PlaneVisualizer>> it = this.visualizerMap.entrySet().iterator();
        while (it.hasNext()) {
            Map.Entry<Plane, PlaneVisualizer> next = it.next();
            Plane key = next.getKey();
            PlaneVisualizer value = next.getValue();
            if (key.getSubsumedBy() != null || key.getTrackingState() == TrackingState.STOPPED) {
                value.release();
                it.remove();
            }
        }
    }
}