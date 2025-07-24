package com.google.ar.sceneform;

import android.media.Image;
import android.util.Log;
import android.view.MotionEvent;
import c.b.a.a.a;
import c.d.b.a.j;
import c.d.b.a.k;
import c.d.b.a.m;
import com.google.ar.sceneform.FrameTime;
import com.google.ar.sceneform.Node;
import com.google.ar.sceneform.Scene;
import com.google.ar.sceneform.collision.Collider;
import com.google.ar.sceneform.collision.CollisionSystem;
import com.google.ar.sceneform.collision.Ray;
import com.google.ar.sceneform.rendering.Color;
import com.google.ar.sceneform.rendering.LightProbe;
import com.google.ar.sceneform.rendering.Renderer;
import com.google.ar.sceneform.utilities.AndroidPreconditions;
import com.google.ar.sceneform.utilities.EnvironmentalHdrParameters;
import com.google.ar.sceneform.utilities.LoadHelper;
import com.google.ar.sceneform.utilities.Preconditions;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.function.Consumer;
import java.util.function.Function;

/* loaded from: classes.dex */
public class Scene extends NodeParent {
    private static final float DEFAULT_EXPOSURE = 1.0f;
    public static final EnvironmentalHdrParameters DEFAULT_HDR_PARAMETERS = EnvironmentalHdrParameters.makeDefault();
    private static final String DEFAULT_LIGHTPROBE_ASSET_NAME = "small_empty_house_2k";
    private static final String DEFAULT_LIGHTPROBE_RESOURCE_NAME = "sceneform_default_light_probe";
    private static final String TAG = "Scene";
    private final Camera camera;
    public final CollisionSystem collisionSystem;
    private boolean isUnderTesting;
    private LightProbe lightProbe;
    private boolean lightProbeSet;
    private final ArrayList<OnUpdateListener> onUpdateListeners;
    private final Sun sunlightNode;
    private final TouchEventSystem touchEventSystem;
    private final SceneView view;

    /* loaded from: classes.dex */
    public interface OnPeekTouchListener {
        void onPeekTouch(HitTestResult hitTestResult, MotionEvent motionEvent);
    }

    /* loaded from: classes.dex */
    public interface OnTouchListener {
        boolean onSceneTouch(HitTestResult hitTestResult, MotionEvent motionEvent);
    }

    /* loaded from: classes.dex */
    public interface OnUpdateListener {
        void onUpdate(FrameTime frameTime);
    }

    public Scene() {
        this.lightProbeSet = false;
        this.isUnderTesting = false;
        this.collisionSystem = new CollisionSystem();
        this.touchEventSystem = new TouchEventSystem();
        this.onUpdateListeners = new ArrayList<>();
        this.view = null;
        this.lightProbe = null;
        this.camera = new Camera(true);
        if (!AndroidPreconditions.isMinAndroidApiLevel()) {
            this.sunlightNode = null;
        } else {
            this.sunlightNode = new Sun();
        }
        this.isUnderTesting = true;
    }

    private void setupLightProbe(SceneView sceneView) {
        Preconditions.checkNotNull(sceneView, "Parameter \"view\" was null.");
        int rawResourceNameToIdentifier = LoadHelper.rawResourceNameToIdentifier(sceneView.getContext(), DEFAULT_LIGHTPROBE_RESOURCE_NAME);
        if (rawResourceNameToIdentifier == 0) {
            Log.w(TAG, "Unable to find the default Light Probe. The scene will not be lit unless a light probe is set.");
            return;
        }
        try {
            LightProbe.builder().setSource(sceneView.getContext(), rawResourceNameToIdentifier).setAssetName(DEFAULT_LIGHTPROBE_ASSET_NAME).build().thenAccept(new Consumer() { // from class: c.d.b.a.i
                @Override // java.util.function.Consumer
                public final void accept(Object obj) {
                    Scene.this.a((LightProbe) obj);
                }
            }).exceptionally((Function<Throwable, ? extends Void>) k.f4307a);
        } catch (Exception e2) {
            StringBuilder x = a.x("Failed to create the default Light Probe: ");
            x.append(e2.getLocalizedMessage());
            throw new IllegalStateException(x.toString());
        }
    }

    public /* synthetic */ void a(LightProbe lightProbe) {
        if (this.lightProbeSet) {
            return;
        }
        setLightProbe(lightProbe);
    }

    public void addOnPeekTouchListener(OnPeekTouchListener onPeekTouchListener) {
        this.touchEventSystem.addOnPeekTouchListener(onPeekTouchListener);
    }

    public void addOnUpdateListener(OnUpdateListener onUpdateListener) {
        Preconditions.checkNotNull(onUpdateListener, "Parameter 'onUpdateListener' was null.");
        if (this.onUpdateListeners.contains(onUpdateListener)) {
            return;
        }
        this.onUpdateListeners.add(onUpdateListener);
    }

    public void dispatchUpdate(final FrameTime frameTime) {
        Iterator<OnUpdateListener> it = this.onUpdateListeners.iterator();
        while (it.hasNext()) {
            it.next().onUpdate(frameTime);
        }
        callOnHierarchy(new Consumer() { // from class: c.d.b.a.h
            @Override // java.util.function.Consumer
            public final void accept(Object obj) {
                FrameTime frameTime2 = FrameTime.this;
                EnvironmentalHdrParameters environmentalHdrParameters = Scene.DEFAULT_HDR_PARAMETERS;
                ((Node) obj).dispatchUpdate(frameTime2);
            }
        });
    }

    public Camera getCamera() {
        return this.camera;
    }

    public LightProbe getLightProbe() {
        LightProbe lightProbe = this.lightProbe;
        if (lightProbe != null) {
            return lightProbe;
        }
        throw new IllegalStateException("Scene's lightProbe must not be null.");
    }

    public Node getSunlight() {
        return this.sunlightNode;
    }

    public SceneView getView() {
        SceneView sceneView = this.view;
        if (sceneView != null) {
            return sceneView;
        }
        throw new IllegalStateException("Scene's view must not be null.");
    }

    public HitTestResult hitTest(MotionEvent motionEvent) {
        Preconditions.checkNotNull(motionEvent, "Parameter \"motionEvent\" was null.");
        Camera camera = this.camera;
        if (camera == null) {
            return new HitTestResult();
        }
        return hitTest(camera.motionEventToRay(motionEvent));
    }

    public ArrayList<HitTestResult> hitTestAll(MotionEvent motionEvent) {
        Preconditions.checkNotNull(motionEvent, "Parameter \"motionEvent\" was null.");
        Camera camera = this.camera;
        if (camera == null) {
            return new ArrayList<>();
        }
        return hitTestAll(camera.motionEventToRay(motionEvent));
    }

    public boolean isUnderTesting() {
        return this.isUnderTesting;
    }

    @Override // com.google.ar.sceneform.NodeParent
    public void onAddChild(Node node) {
        super.onAddChild(node);
        node.setSceneRecursively(this);
    }

    @Override // com.google.ar.sceneform.NodeParent
    public void onRemoveChild(Node node) {
        super.onRemoveChild(node);
        node.setSceneRecursively(null);
    }

    public void onTouchEvent(MotionEvent motionEvent) {
        Preconditions.checkNotNull(motionEvent, "Parameter \"motionEvent\" was null.");
        this.touchEventSystem.onTouchEvent(hitTest(motionEvent), motionEvent);
    }

    public Node overlapTest(Node node) {
        Collider intersects;
        Preconditions.checkNotNull(node, "Parameter \"node\" was null.");
        Collider collider = node.getCollider();
        if (collider == null || (intersects = this.collisionSystem.intersects(collider)) == null) {
            return null;
        }
        return (Node) intersects.getTransformProvider();
    }

    public ArrayList<Node> overlapTestAll(Node node) {
        Preconditions.checkNotNull(node, "Parameter \"node\" was null.");
        final ArrayList<Node> arrayList = new ArrayList<>();
        Collider collider = node.getCollider();
        if (collider == null) {
            return arrayList;
        }
        this.collisionSystem.intersectsAll(collider, new Consumer() { // from class: c.d.b.a.l
            @Override // java.util.function.Consumer
            public final void accept(Object obj) {
                ArrayList arrayList2 = arrayList;
                EnvironmentalHdrParameters environmentalHdrParameters = Scene.DEFAULT_HDR_PARAMETERS;
                arrayList2.add((Node) ((Collider) obj).getTransformProvider());
            }
        });
        return arrayList;
    }

    public void removeOnPeekTouchListener(OnPeekTouchListener onPeekTouchListener) {
        this.touchEventSystem.removeOnPeekTouchListener(onPeekTouchListener);
    }

    public void removeOnUpdateListener(OnUpdateListener onUpdateListener) {
        Preconditions.checkNotNull(onUpdateListener, "Parameter 'onUpdateListener' was null.");
        this.onUpdateListeners.remove(onUpdateListener);
    }

    public void setEnvironmentalHdrLightEstimate(float[] fArr, float[] fArr2, Color color, float f2, Image[] imageArr) {
        EnvironmentalHdrParameters environmentalHdrParameters;
        float f3;
        SceneView sceneView = this.view;
        if (sceneView == null) {
            f3 = 1.0f;
            environmentalHdrParameters = DEFAULT_HDR_PARAMETERS;
        } else {
            Renderer renderer = (Renderer) Preconditions.checkNotNull(sceneView.getRenderer());
            float exposure = renderer.getExposure();
            environmentalHdrParameters = renderer.getEnvironmentalHdrParameters();
            f3 = exposure;
        }
        LightProbe lightProbe = this.lightProbe;
        if (lightProbe != null) {
            if (fArr != null) {
                lightProbe.setEnvironmentalHdrSphericalHarmonics(fArr, f3, environmentalHdrParameters);
            }
            if (imageArr != null) {
                this.lightProbe.setCubeMap(imageArr);
            }
            setLightProbe(this.lightProbe);
        }
        Sun sun = this.sunlightNode;
        if (sun == null || fArr2 == null) {
            return;
        }
        sun.setEnvironmentalHdrLightEstimate(fArr2, color, f2, f3, environmentalHdrParameters);
    }

    public void setLightEstimate(Color color, float f2) {
        LightProbe lightProbe = this.lightProbe;
        if (lightProbe != null) {
            lightProbe.setLightEstimate(color, f2);
            setLightProbe(this.lightProbe);
        }
        Sun sun = this.sunlightNode;
        if (sun != null) {
            sun.setLightEstimate(color, f2);
        }
    }

    public void setLightProbe(LightProbe lightProbe) {
        Preconditions.checkNotNull(lightProbe, "Parameter \"lightProbe\" was null.");
        this.lightProbe = lightProbe;
        this.lightProbeSet = true;
        SceneView sceneView = this.view;
        if (sceneView != null) {
            ((Renderer) Preconditions.checkNotNull(sceneView.getRenderer())).setLightProbe(lightProbe);
            return;
        }
        throw new IllegalStateException("Scene's view must not be null.");
    }

    public void setOnTouchListener(OnTouchListener onTouchListener) {
        this.touchEventSystem.setOnTouchListener(onTouchListener);
    }

    public void setUseHdrLightEstimate(boolean z) {
        SceneView sceneView = this.view;
        if (sceneView != null) {
            ((Renderer) Preconditions.checkNotNull(sceneView.getRenderer())).setUseHdrLightEstimate(z);
        }
    }

    public HitTestResult hitTest(Ray ray) {
        Preconditions.checkNotNull(ray, "Parameter \"ray\" was null.");
        HitTestResult hitTestResult = new HitTestResult();
        Collider raycast = this.collisionSystem.raycast(ray, hitTestResult);
        if (raycast != null) {
            hitTestResult.setNode((Node) raycast.getTransformProvider());
        }
        return hitTestResult;
    }

    public ArrayList<HitTestResult> hitTestAll(Ray ray) {
        Preconditions.checkNotNull(ray, "Parameter \"ray\" was null.");
        ArrayList<HitTestResult> arrayList = new ArrayList<>();
        this.collisionSystem.raycastAll(ray, arrayList, j.f4306a, m.f4309a);
        return arrayList;
    }

    public Scene(SceneView sceneView) {
        this.lightProbeSet = false;
        this.isUnderTesting = false;
        this.collisionSystem = new CollisionSystem();
        this.touchEventSystem = new TouchEventSystem();
        this.onUpdateListeners = new ArrayList<>();
        Preconditions.checkNotNull(sceneView, "Parameter \"view\" was null.");
        this.view = sceneView;
        this.camera = new Camera(this);
        if (!AndroidPreconditions.isMinAndroidApiLevel()) {
            this.sunlightNode = null;
            return;
        }
        this.sunlightNode = new Sun(this);
        setupLightProbe(sceneView);
    }
}