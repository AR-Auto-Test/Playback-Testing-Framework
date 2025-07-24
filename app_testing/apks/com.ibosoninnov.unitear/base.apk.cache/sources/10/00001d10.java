package com.google.ar.sceneform.rendering;

import android.content.Context;
import android.view.Surface;
import android.view.SurfaceView;
import com.google.android.filament.Camera;
import com.google.android.filament.IndirectLight;
import com.google.android.filament.Renderer;
import com.google.android.filament.Scene;
import com.google.android.filament.SwapChain;
import com.google.android.filament.TransformManager;
import com.google.android.filament.View;
import com.google.android.filament.Viewport;
import com.google.android.filament.android.UiHelper;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import com.google.ar.sceneform.utilities.AndroidPreconditions;
import com.google.ar.sceneform.utilities.EnvironmentalHdrParameters;
import com.google.ar.sceneform.utilities.Preconditions;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/* loaded from: classes.dex */
public class Renderer implements UiHelper.RendererCallback {
    private static final float ARCORE_HDR_LIGHTING_CAMERA_APERATURE = 1.0f;
    private static final float ARCORE_HDR_LIGHTING_CAMERA_ISO = 100.0f;
    private static final float ARCORE_HDR_LIGHTING_CAMERA_SHUTTER_SPEED = 1.2f;
    private static final float DEFAULT_CAMERA_APERATURE = 4.0f;
    private static final float DEFAULT_CAMERA_ISO = 320.0f;
    private static final float DEFAULT_CAMERA_SHUTTER_SPEED = 0.033333335f;
    private static final Color DEFAULT_CLEAR_COLOR = new Color(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 1.0f);
    private static final int MAXIMUM_RESOLUTION = 1080;
    private Camera camera;
    private float cameraAperature;
    private float cameraIso;
    private CameraProvider cameraProvider;
    private float cameraShutterSpeed;
    private View emptyView;
    private UiHelper filamentHelper;
    private IndirectLight indirectLight;
    private PreRenderCallback preRenderCallback;
    private boolean recreateSwapChain;
    private com.google.android.filament.Renderer renderer;
    private Scene scene;
    private Surface surface;
    private final SurfaceView surfaceView;
    private SwapChain swapChain;
    private View view;
    private final ViewAttachmentManager viewAttachmentManager;
    private final ArrayList<RenderableInstance> renderableInstances = new ArrayList<>();
    private final ArrayList<LightInstance> lightInstances = new ArrayList<>();
    private final double[] cameraProjectionMatrix = new double[16];
    private EnvironmentalHdrParameters environmentalHdrParameters = EnvironmentalHdrParameters.makeDefault();
    private final List<Mirror> mirrors = new ArrayList();
    private Runnable onFrameRenderDebugCallback = null;

    /* loaded from: classes.dex */
    public static class Mirror {
        public Surface surface;
        public SwapChain swapChain;
        public Viewport viewport;

        private Mirror() {
        }
    }

    /* loaded from: classes.dex */
    public interface PreRenderCallback {
        void preRender(com.google.android.filament.Renderer renderer, SwapChain swapChain, Camera camera);
    }

    public Renderer(SurfaceView surfaceView) {
        Preconditions.checkNotNull(surfaceView, "Parameter \"view\" was null.");
        AndroidPreconditions.checkMinAndroidApiLevel();
        this.surfaceView = surfaceView;
        this.viewAttachmentManager = new ViewAttachmentManager(getContext(), surfaceView);
        initialize();
    }

    private void addModelInstanceInternal(RenderableInstance renderableInstance) {
    }

    public static void destroyAllResources() {
        ResourceManager.getInstance().destroyAllResources();
        EngineInstance.destroyEngine();
    }

    private Viewport getLetterboxViewport(Viewport viewport, Viewport viewport2) {
        float f2;
        float f3;
        int i = viewport2.width;
        int i2 = viewport2.height;
        float f4 = i / i2;
        int i3 = viewport.width;
        int i4 = viewport.height;
        if (f4 > ((float) i3) / ((float) i4)) {
            f2 = i2;
            f3 = i4;
        } else {
            f2 = i;
            f3 = i3;
        }
        float f5 = f2 / f3;
        int i5 = (int) (i3 * f5);
        int i6 = (int) (i4 * f5);
        return new Viewport((i - i5) / 2, (i2 - i6) / 2, i5, i6);
    }

    private void initialize() {
        SurfaceView surfaceView = getSurfaceView();
        UiHelper uiHelper = new UiHelper(UiHelper.ContextErrorPolicy.DONT_CHECK);
        this.filamentHelper = uiHelper;
        uiHelper.setRenderCallback(this);
        this.filamentHelper.attachTo(surfaceView);
        IEngine engine = EngineInstance.getEngine();
        this.renderer = engine.createRenderer();
        this.scene = engine.createScene();
        this.view = engine.createView();
        this.emptyView = engine.createView();
        this.camera = engine.createCamera();
        setUseHdrLightEstimate(false);
        setDefaultClearColor();
        this.view.setCamera(this.camera);
        this.view.setScene(this.scene);
        setDynamicResolutionEnabled(true);
        this.emptyView.setCamera(engine.createCamera());
        this.emptyView.setScene(engine.createScene());
    }

    public static long reclaimReleasedResources() {
        return ResourceManager.getInstance().reclaimReleasedResources();
    }

    private void removeModelInstanceInternal(RenderableInstance renderableInstance) {
    }

    private void updateInstances() {
        TransformManager transformManager = EngineInstance.getEngine().getTransformManager();
        transformManager.openLocalTransformTransaction();
        Iterator<RenderableInstance> it = this.renderableInstances.iterator();
        while (it.hasNext()) {
            RenderableInstance next = it.next();
            next.prepareForDraw();
            next.setModelMatrix(transformManager, next.getWorldModelMatrix().data);
        }
        transformManager.commitLocalTransformTransaction();
    }

    private void updateLights() {
        Iterator<LightInstance> it = this.lightInstances.iterator();
        while (it.hasNext()) {
            it.next().updateTransform();
        }
    }

    public void addInstance(RenderableInstance renderableInstance) {
        this.scene.addEntity(renderableInstance.getRenderedEntity());
        addModelInstanceInternal(renderableInstance);
        this.renderableInstances.add(renderableInstance);
    }

    public void addLight(LightInstance lightInstance) {
        this.scene.addEntity(lightInstance.getEntity());
        this.lightInstances.add(lightInstance);
    }

    public void dispose() {
        this.filamentHelper.detach();
        IEngine engine = EngineInstance.getEngine();
        IndirectLight indirectLight = this.indirectLight;
        if (indirectLight != null) {
            engine.destroyIndirectLight(indirectLight);
        }
        engine.destroyRenderer(this.renderer);
        engine.destroyView(this.view);
        reclaimReleasedResources();
    }

    public void enablePerformanceMode() {
    }

    public Context getContext() {
        return getSurfaceView().getContext();
    }

    public int getDesiredHeight() {
        return this.filamentHelper.getDesiredHeight();
    }

    public int getDesiredWidth() {
        return this.filamentHelper.getDesiredWidth();
    }

    public EnvironmentalHdrParameters getEnvironmentalHdrParameters() {
        return this.environmentalHdrParameters;
    }

    public float getExposure() {
        float f2 = this.cameraAperature;
        return 1.0f / (((((f2 * f2) / this.cameraShutterSpeed) * ARCORE_HDR_LIGHTING_CAMERA_ISO) / this.cameraIso) * ARCORE_HDR_LIGHTING_CAMERA_SHUTTER_SPEED);
    }

    public com.google.android.filament.Renderer getFilamentRenderer() {
        return this.renderer;
    }

    public Scene getFilamentScene() {
        return this.scene;
    }

    public SurfaceView getSurfaceView() {
        return this.surfaceView;
    }

    public ViewAttachmentManager getViewAttachmentManager() {
        return this.viewAttachmentManager;
    }

    public boolean isFrontFaceWindingInverted() {
        return this.view.isFrontFaceWindingInverted();
    }

    @Override // com.google.android.filament.android.UiHelper.RendererCallback
    public void onDetachedFromSurface() {
        SwapChain swapChain = this.swapChain;
        if (swapChain != null) {
            IEngine engine = EngineInstance.getEngine();
            engine.destroySwapChain(swapChain);
            engine.flushAndWait();
            this.swapChain = null;
        }
    }

    @Override // com.google.android.filament.android.UiHelper.RendererCallback
    public void onNativeWindowChanged(Surface surface) {
        synchronized (this) {
            this.surface = surface;
            this.recreateSwapChain = true;
        }
    }

    public void onPause() {
        this.viewAttachmentManager.onPause();
    }

    @Override // com.google.android.filament.android.UiHelper.RendererCallback
    public void onResized(int i, int i2) {
        this.view.setViewport(new Viewport(0, 0, i, i2));
        this.emptyView.setViewport(new Viewport(0, 0, i, i2));
    }

    public void onResume() {
        this.viewAttachmentManager.onResume();
    }

    public void removeInstance(RenderableInstance renderableInstance) {
        removeModelInstanceInternal(renderableInstance);
        this.scene.remove(renderableInstance.getRenderedEntity());
        this.renderableInstances.remove(renderableInstance);
    }

    public void removeLight(LightInstance lightInstance) {
        this.scene.remove(lightInstance.getEntity());
        this.lightInstances.remove(lightInstance);
    }

    public void render(boolean z) {
        int i;
        synchronized (this) {
            if (this.recreateSwapChain) {
                IEngine engine = EngineInstance.getEngine();
                SwapChain swapChain = this.swapChain;
                if (swapChain != null) {
                    engine.destroySwapChain(swapChain);
                }
                this.swapChain = engine.createSwapChain(this.surface, 2L);
                this.recreateSwapChain = false;
            }
        }
        synchronized (this.mirrors) {
            Iterator<Mirror> it = this.mirrors.iterator();
            while (it.hasNext()) {
                Mirror next = it.next();
                if (next.surface == null) {
                    if (next.swapChain != null) {
                        EngineInstance.getEngine().destroySwapChain((SwapChain) Preconditions.checkNotNull(next.swapChain));
                    }
                    it.remove();
                } else if (next.swapChain == null) {
                    next.swapChain = EngineInstance.getEngine().createSwapChain(Preconditions.checkNotNull(next.surface));
                }
            }
        }
        if (this.filamentHelper.isReadyToRender() || EngineInstance.isHeadlessMode()) {
            updateInstances();
            updateLights();
            CameraProvider cameraProvider = this.cameraProvider;
            if (cameraProvider != null) {
                float[] fArr = cameraProvider.getProjectionMatrix().data;
                for (i = 0; i < 16; i++) {
                    this.cameraProjectionMatrix[i] = fArr[i];
                }
                this.camera.setModelMatrix(cameraProvider.getWorldModelMatrix().data);
                this.camera.setCustomProjection(this.cameraProjectionMatrix, cameraProvider.getNearClipPlane(), cameraProvider.getFarClipPlane());
                SwapChain swapChain2 = this.swapChain;
                if (swapChain2 != null) {
                    if (this.renderer.beginFrame(swapChain2, 0L)) {
                        PreRenderCallback preRenderCallback = this.preRenderCallback;
                        if (preRenderCallback != null) {
                            preRenderCallback.preRender(this.renderer, swapChain2, this.camera);
                        }
                        View view = cameraProvider.isActive() ? this.view : this.emptyView;
                        this.renderer.render(view);
                        synchronized (this.mirrors) {
                            for (Mirror mirror : this.mirrors) {
                                SwapChain swapChain3 = mirror.swapChain;
                                if (swapChain3 != null) {
                                    this.renderer.mirrorFrame(swapChain3, getLetterboxViewport(view.getViewport(), mirror.viewport), view.getViewport(), 7);
                                }
                            }
                        }
                        Runnable runnable = this.onFrameRenderDebugCallback;
                        if (runnable != null) {
                            runnable.run();
                        }
                        this.renderer.endFrame();
                    }
                    reclaimReleasedResources();
                    return;
                }
                throw new AssertionError("Internal Error: Failed to get swap chain");
            }
        }
    }

    public void setAntiAliasing(View.AntiAliasing antiAliasing) {
        this.view.setAntiAliasing(antiAliasing);
    }

    public void setCameraProvider(CameraProvider cameraProvider) {
        this.cameraProvider = cameraProvider;
    }

    public void setClearColor(Color color) {
        Renderer.ClearOptions clearOptions = new Renderer.ClearOptions();
        float[] fArr = clearOptions.clearColor;
        fArr[0] = color.r;
        fArr[1] = color.f5628g;
        fArr[2] = color.f5627b;
        fArr[3] = color.f5626a;
        this.renderer.setClearOptions(clearOptions);
    }

    public void setDefaultClearColor() {
        setClearColor(DEFAULT_CLEAR_COLOR);
    }

    public void setDesiredSize(int i, int i2) {
        int min = Math.min(i, i2);
        int max = Math.max(i, i2);
        if (min > MAXIMUM_RESOLUTION) {
            max = (max * MAXIMUM_RESOLUTION) / min;
            min = MAXIMUM_RESOLUTION;
        }
        if (i >= i2) {
            int i3 = max;
            max = min;
            min = i3;
        }
        this.filamentHelper.setDesiredSize(min, max);
    }

    public void setDithering(View.Dithering dithering) {
        this.view.setDithering(dithering);
    }

    public void setDynamicResolutionEnabled(boolean z) {
        View.DynamicResolutionOptions dynamicResolutionOptions = new View.DynamicResolutionOptions();
        dynamicResolutionOptions.enabled = z;
        this.view.setDynamicResolutionOptions(dynamicResolutionOptions);
    }

    public void setEnvironmentalHdrParameters(EnvironmentalHdrParameters environmentalHdrParameters) {
        this.environmentalHdrParameters = environmentalHdrParameters;
    }

    public void setFrameRenderDebugCallback(Runnable runnable) {
        this.onFrameRenderDebugCallback = runnable;
    }

    public void setFrontFaceWindingInverted(Boolean bool) {
        this.view.setFrontFaceWindingInverted(bool.booleanValue());
    }

    public void setLightProbe(LightProbe lightProbe) {
        if (lightProbe != null) {
            IndirectLight buildIndirectLight = lightProbe.buildIndirectLight();
            if (buildIndirectLight != null) {
                this.scene.setIndirectLight(buildIndirectLight);
                IndirectLight indirectLight = this.indirectLight;
                if (indirectLight != null && indirectLight != buildIndirectLight) {
                    EngineInstance.getEngine().destroyIndirectLight(this.indirectLight);
                }
                this.indirectLight = buildIndirectLight;
                return;
            }
            return;
        }
        throw new AssertionError("Passed in an invalid light probe.");
    }

    public void setPostProcessingEnabled(boolean z) {
    }

    public void setPreRenderCallback(PreRenderCallback preRenderCallback) {
        this.preRenderCallback = preRenderCallback;
    }

    public void setRenderQuality(View.RenderQuality renderQuality) {
    }

    public void setUseHdrLightEstimate(boolean z) {
        if (z) {
            this.cameraAperature = 1.0f;
            this.cameraShutterSpeed = ARCORE_HDR_LIGHTING_CAMERA_SHUTTER_SPEED;
            this.cameraIso = ARCORE_HDR_LIGHTING_CAMERA_ISO;
        } else {
            this.cameraAperature = DEFAULT_CAMERA_APERATURE;
            this.cameraShutterSpeed = DEFAULT_CAMERA_SHUTTER_SPEED;
            this.cameraIso = DEFAULT_CAMERA_ISO;
        }
        this.camera.setExposure(this.cameraAperature, this.cameraShutterSpeed, this.cameraIso);
    }

    public void startMirroring(Surface surface, int i, int i2, int i3, int i4) {
        Mirror mirror = new Mirror();
        mirror.surface = surface;
        mirror.viewport = new Viewport(i, i2, i3, i4);
        mirror.swapChain = null;
        synchronized (this.mirrors) {
            this.mirrors.add(mirror);
        }
    }

    public void stopMirroring(Surface surface) {
        synchronized (this.mirrors) {
            for (Mirror mirror : this.mirrors) {
                if (mirror.surface == surface) {
                    mirror.surface = null;
                }
            }
        }
    }
}