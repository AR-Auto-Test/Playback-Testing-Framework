package com.google.ar.sceneform;

import android.content.Context;
import android.util.AttributeSet;
import android.util.Log;
import android.view.Display;
import android.view.WindowManager;
import c.d.b.a.e;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import com.google.ar.core.Anchor;
import com.google.ar.core.ArImage;
import com.google.ar.core.CameraConfig;
import com.google.ar.core.Config;
import com.google.ar.core.Frame;
import com.google.ar.core.LightEstimate;
import com.google.ar.core.Pose;
import com.google.ar.core.Session;
import com.google.ar.core.TrackingState;
import com.google.ar.core.exceptions.CameraNotAvailableException;
import com.google.ar.core.exceptions.FatalException;
import com.google.ar.sceneform.ArSceneView;
import com.google.ar.sceneform.math.Vector3;
import com.google.ar.sceneform.rendering.CameraStream;
import com.google.ar.sceneform.rendering.Color;
import com.google.ar.sceneform.rendering.EnvironmentalHdrLightEstimate;
import com.google.ar.sceneform.rendering.GLHelper;
import com.google.ar.sceneform.rendering.PlaneRenderer;
import com.google.ar.sceneform.rendering.Renderer;
import com.google.ar.sceneform.rendering.ThreadPools;
import com.google.ar.sceneform.utilities.AndroidPreconditions;
import com.google.ar.sceneform.utilities.ArCoreVersion;
import com.google.ar.sceneform.utilities.Preconditions;
import java.lang.ref.WeakReference;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Executor;
import java.util.function.Consumer;

/* loaded from: classes.dex */
public class ArSceneView extends SceneView {
    private static final float DEFAULT_PIXEL_INTENSITY = 1.0f;
    private static final float RECREATE_LIGHTING_ANCHOR_DISTANCE = 0.5f;
    private static final String REPORTED_ENGINE_TYPE = "Sceneform";
    private static final String REPORTED_ENGINE_VERSION = "1.7";

    /* renamed from: c  reason: collision with root package name */
    public static final /* synthetic */ int f5624c = 0;
    private Config cachedConfig;
    private CameraStream cameraStream;
    private int cameraTextureId;
    private final float[] colorCorrectionPixelIntensity;
    private Frame currentFrame;
    private Display display;
    private boolean isAutofocusEnabled;
    private boolean isDepthSupported;
    private boolean isLightDirectionUpdateEnabled;
    private boolean isPausedAR;
    private final Color lastValidColorCorrection;
    private float[] lastValidEnvironmentalHdrAmbientSphericalHarmonics;
    private Anchor lastValidEnvironmentalHdrAnchor;
    private float[] lastValidEnvironmentalHdrMainLightDirection;
    private float[] lastValidEnvironmentalHdrMainLightIntensity;
    private float lastValidPixelIntensity;
    private boolean lightEstimationEnabled;
    private int minArCoreVersionCode;
    private Consumer<EnvironmentalHdrLightEstimate> onNextHdrLightingEstimate;
    private final SequentialTask pauseResumeTask;
    private PlaneRenderer planeRenderer;
    private Session session;
    private static final String TAG = ArSceneView.class.getSimpleName();
    private static final Color DEFAULT_COLOR_CORRECTION = new Color(1.0f, 1.0f, 1.0f);

    public ArSceneView(Context context) {
        super(context);
        this.lightEstimationEnabled = true;
        this.isLightDirectionUpdateEnabled = true;
        this.onNextHdrLightingEstimate = null;
        this.lastValidPixelIntensity = 1.0f;
        this.lastValidColorCorrection = new Color(DEFAULT_COLOR_CORRECTION);
        this.colorCorrectionPixelIntensity = new float[4];
        this.pauseResumeTask = new SequentialTask();
        this.isPausedAR = false;
        this.isAutofocusEnabled = false;
        this.isDepthSupported = false;
        ((Renderer) Preconditions.checkNotNull(getRenderer())).enablePerformanceMode();
        initializeAr();
    }

    private void ensureUpdateMode() {
        Session session = this.session;
        if (session != null && this.minArCoreVersionCode >= 180604036) {
            Config config = this.cachedConfig;
            if (config == null) {
                this.cachedConfig = session.getConfig();
            } else {
                session.getConfig(config);
            }
            setUpAutoFocus();
            Config.UpdateMode updateMode = this.cachedConfig.getUpdateMode();
            if (updateMode == Config.UpdateMode.LATEST_CAMERA_IMAGE) {
                return;
            }
            throw new RuntimeException("Invalid ARCore UpdateMode " + updateMode + ", Sceneform requires that the ARCore session is configured to the UpdateMode LATEST_CAMERA_IMAGE.");
        }
    }

    private void initializeAr() {
        this.minArCoreVersionCode = ArCoreVersion.getMinArCoreVersionCode(getContext());
        this.display = ((WindowManager) getContext().getSystemService(WindowManager.class)).getDefaultDisplay();
        initializePlaneRenderer();
        initializeCameraStream();
    }

    private void initializeCameraStream() {
        this.cameraTextureId = GLHelper.createCameraTexture();
        this.cameraStream = new CameraStream(this.cameraTextureId, (Renderer) Preconditions.checkNotNull(getRenderer()));
    }

    private void initializeFacingDirection(Session session) {
        if (session.getCameraConfig().getFacingDirection() == CameraConfig.FacingDirection.FRONT) {
            ((Renderer) Preconditions.checkNotNull(getRenderer())).setFrontFaceWindingInverted(Boolean.TRUE);
        }
    }

    private void initializePlaneRenderer() {
        this.planeRenderer = new PlaneRenderer((Renderer) Preconditions.checkNotNull(getRenderer()));
    }

    public static /* synthetic */ void lambda$pauseAsync$2(WeakReference weakReference) {
        ArSceneView arSceneView = (ArSceneView) weakReference.get();
        if (arSceneView == null) {
            return;
        }
        arSceneView.pauseScene();
    }

    public static /* synthetic */ void lambda$resumeAsync$1(WeakReference weakReference) {
        ArSceneView arSceneView = (ArSceneView) weakReference.get();
        if (arSceneView == null) {
            return;
        }
        arSceneView.resumeScene();
    }

    private static boolean loadUnifiedJni() {
        return false;
    }

    private static native void nativeReportEngineType(Session session, String str, String str2);

    private void pauseScene() {
        super.pause();
    }

    private void reportEngineType() {
    }

    private void resumeScene() {
        try {
            super.resume();
        } catch (CameraNotAvailableException e2) {
            throw new IllegalStateException(e2);
        }
    }

    private void setUpAutoFocus() {
        Session session = this.session;
        if (session == null || this.isAutofocusEnabled) {
            return;
        }
        this.isAutofocusEnabled = true;
        Config config = session.getConfig();
        config.setFocusMode(Config.FocusMode.AUTO);
        this.session.configure(config);
    }

    private boolean shouldRecalculateCameraUvs(Frame frame) {
        return frame.hasDisplayGeometryChanged();
    }

    private void updateLightEstimate(Frame frame) {
        if (!this.lightEstimationEnabled || getSession() == null) {
            return;
        }
        LightEstimate lightEstimate = frame.getLightEstimate();
        if (isEnvironmentalHdrLightingAvailable()) {
            if (frame.getCamera().getTrackingState() == TrackingState.TRACKING) {
                updateHdrLightEstimate(lightEstimate, (Session) Preconditions.checkNotNull(getSession()), frame.getCamera());
                return;
            }
            return;
        }
        updateNormalLightEstimate(lightEstimate);
    }

    private void updateNormalLightEstimate(LightEstimate lightEstimate) {
        getScene().setUseHdrLightEstimate(false);
        float f2 = this.lastValidPixelIntensity;
        if (lightEstimate.getState() == LightEstimate.State.VALID) {
            lightEstimate.getColorCorrection(this.colorCorrectionPixelIntensity, 0);
            f2 = Math.max(this.colorCorrectionPixelIntensity[3], (float) StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
            Color color = this.lastValidColorCorrection;
            float[] fArr = this.colorCorrectionPixelIntensity;
            color.set(fArr[0], fArr[1], fArr[2]);
        }
        getScene().setLightEstimate(this.lastValidColorCorrection, f2);
        this.lastValidPixelIntensity = f2;
    }

    public void captureLightingValues(Consumer<EnvironmentalHdrLightEstimate> consumer) {
        this.onNextHdrLightingEstimate = consumer;
    }

    public Frame getArFrame() {
        return this.currentFrame;
    }

    public PlaneRenderer getPlaneRenderer() {
        return this.planeRenderer;
    }

    public Session getSession() {
        return this.session;
    }

    public boolean isDepthSupported() {
        return this.isDepthSupported;
    }

    public boolean isEnvironmentalHdrLightingAvailable() {
        Config config = this.cachedConfig;
        return config != null && config.getLightEstimationMode() == Config.LightEstimationMode.ENVIRONMENTAL_HDR;
    }

    public boolean isLightDirectionUpdateEnabled() {
        return this.isLightDirectionUpdateEnabled;
    }

    public boolean isLightEstimationEnabled() {
        return this.lightEstimationEnabled;
    }

    @Override // com.google.ar.sceneform.SceneView
    public boolean onBeginFrame(long j) {
        Session session = this.session;
        if (session != null && this.pauseResumeTask.isDone()) {
            boolean z = true;
            if (this.isPausedAR) {
                return true;
            }
            ensureUpdateMode();
            try {
                Frame update = session.update();
                if (update == null) {
                    return false;
                }
                if (!this.cameraStream.isTextureInitialized()) {
                    this.cameraStream.initializeTexture(update);
                }
                if (shouldRecalculateCameraUvs(update)) {
                    this.cameraStream.recalculateCameraUvs(update);
                }
                Frame frame = this.currentFrame;
                if (frame != null && frame.getTimestamp() == update.getTimestamp()) {
                    z = false;
                }
                this.currentFrame = update;
                com.google.ar.core.Camera camera = update.getCamera();
                if (camera == null) {
                    getScene().setUseHdrLightEstimate(false);
                    return false;
                }
                if (z) {
                    getScene().getCamera().updateTrackedPose(camera);
                    Frame frame2 = this.currentFrame;
                    if (frame2 != null) {
                        updateLightEstimate(frame2);
                        this.planeRenderer.update(frame2, getWidth(), getHeight());
                    }
                }
                return z;
            } catch (CameraNotAvailableException e2) {
                Log.w(TAG, "Exception updating ARCore session", e2);
                return false;
            }
        }
        return false;
    }

    @Override // com.google.ar.sceneform.SceneView, android.view.View
    public void onLayout(boolean z, int i, int i2, int i3, int i4) {
        super.onLayout(z, i, i2, i3, i4);
        Session session = this.session;
        if (session != null) {
            session.setDisplayGeometry(this.display.getRotation(), i3 - i, i4 - i2);
        }
    }

    @Override // com.google.ar.sceneform.SceneView
    public void pause() {
        pauseScene();
        pauseSession();
    }

    public CompletableFuture<Void> pauseAsync(Executor executor) {
        final WeakReference weakReference = new WeakReference(this);
        this.pauseResumeTask.appendRunnable(new Runnable() { // from class: c.d.b.a.d
            @Override // java.lang.Runnable
            public final void run() {
                ArSceneView.lambda$pauseAsync$2(weakReference);
            }
        }, ThreadPools.getMainExecutor());
        return this.pauseResumeTask.appendRunnable(new Runnable() { // from class: c.d.b.a.b
            @Override // java.lang.Runnable
            public final void run() {
                WeakReference weakReference2 = weakReference;
                int i = ArSceneView.f5624c;
                ArSceneView arSceneView = (ArSceneView) weakReference2.get();
                if (arSceneView == null) {
                    return;
                }
                arSceneView.pauseSession();
            }
        }, executor).thenAcceptAsync((Consumer<? super Void>) e.f4300a, ThreadPools.getMainExecutor());
    }

    public void pauseSession() {
        Session session = this.session;
        if (session != null) {
            session.pause();
            this.isPausedAR = true;
        }
    }

    @Override // com.google.ar.sceneform.SceneView
    public void resume() {
        resumeSession();
        resumeScene();
    }

    public CompletableFuture<Void> resumeAsync(Executor executor) {
        final WeakReference weakReference = new WeakReference(this);
        this.pauseResumeTask.appendRunnable(new Runnable() { // from class: c.d.b.a.c
            @Override // java.lang.Runnable
            public final void run() {
                WeakReference weakReference2 = weakReference;
                int i = ArSceneView.f5624c;
                ArSceneView arSceneView = (ArSceneView) weakReference2.get();
                if (arSceneView == null) {
                    return;
                }
                try {
                    arSceneView.resumeSession();
                } catch (CameraNotAvailableException e2) {
                    throw new RuntimeException(e2);
                }
            }
        }, executor);
        return this.pauseResumeTask.appendRunnable(new Runnable() { // from class: c.d.b.a.a
            @Override // java.lang.Runnable
            public final void run() {
                ArSceneView.lambda$resumeAsync$1(weakReference);
            }
        }, ThreadPools.getMainExecutor());
    }

    public void resumeSession() {
        Session session = this.session;
        if (session != null) {
            reportEngineType();
            session.resume();
            this.isPausedAR = false;
        }
    }

    public void setLightDirectionUpdateEnabled(boolean z) {
        this.isLightDirectionUpdateEnabled = z;
    }

    public void setLightEstimationEnabled(boolean z) {
        this.lightEstimationEnabled = z;
        if (z) {
            return;
        }
        Scene scene = getScene();
        Color color = DEFAULT_COLOR_CORRECTION;
        scene.setLightEstimate(color, 1.0f);
        this.lastValidPixelIntensity = 1.0f;
        this.lastValidColorCorrection.set(color);
    }

    public void setupSession(Session session) {
        if (this.session != null) {
            Log.w(TAG, "The session has already been setup, cannot set it up again.");
            return;
        }
        AndroidPreconditions.checkMinAndroidApiLevel();
        this.session = session;
        Renderer renderer = (Renderer) Preconditions.checkNotNull(getRenderer());
        int desiredWidth = renderer.getDesiredWidth();
        int desiredHeight = renderer.getDesiredHeight();
        if (desiredWidth != 0 && desiredHeight != 0) {
            session.setDisplayGeometry(this.display.getRotation(), desiredWidth, desiredHeight);
        }
        initializeFacingDirection(session);
        session.setCameraTextureName(this.cameraTextureId);
    }

    /* JADX WARN: Removed duplicated region for block: B:20:0x006e  */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public void updateHdrLightEstimate(LightEstimate lightEstimate, Session session, com.google.ar.core.Camera camera) {
        boolean z;
        float[] environmentalHdrMainLightDirection;
        float[] fArr;
        if (lightEstimate.getState() != LightEstimate.State.VALID) {
            return;
        }
        getScene().setUseHdrLightEstimate(true);
        if (this.isLightDirectionUpdateEnabled || this.lastValidEnvironmentalHdrMainLightDirection == null) {
            Anchor anchor = this.lastValidEnvironmentalHdrAnchor;
            if (anchor != null && anchor.getTrackingState() == TrackingState.TRACKING) {
                Pose pose = camera.getPose();
                Vector3 vector3 = new Vector3(pose.tx(), pose.ty(), pose.tz());
                Pose pose2 = ((Anchor) Preconditions.checkNotNull(this.lastValidEnvironmentalHdrAnchor)).getPose();
                if (Vector3.subtract(vector3, new Vector3(pose2.tx(), pose2.ty(), pose2.tz())).length() <= 0.5f) {
                    z = false;
                    if (z) {
                        Anchor anchor2 = this.lastValidEnvironmentalHdrAnchor;
                        if (anchor2 != null) {
                            anchor2.detach();
                            this.lastValidEnvironmentalHdrAnchor = null;
                        }
                        this.lastValidEnvironmentalHdrMainLightDirection = null;
                        if (camera.getTrackingState() == TrackingState.TRACKING) {
                            try {
                                this.lastValidEnvironmentalHdrAnchor = session.createAnchor(camera.getPose());
                            } catch (FatalException e2) {
                                Log.e(TAG, "Error trying to create environmental hdr anchor", e2);
                            }
                        }
                    }
                    if (this.lastValidEnvironmentalHdrAnchor != null && (environmentalHdrMainLightDirection = lightEstimate.getEnvironmentalHdrMainLightDirection()) != null) {
                        this.lastValidEnvironmentalHdrMainLightDirection = ((Anchor) Preconditions.checkNotNull(this.lastValidEnvironmentalHdrAnchor)).getPose().inverse().rotateVector(environmentalHdrMainLightDirection);
                    }
                }
            }
            z = true;
            if (z) {
            }
            if (this.lastValidEnvironmentalHdrAnchor != null) {
                this.lastValidEnvironmentalHdrMainLightDirection = ((Anchor) Preconditions.checkNotNull(this.lastValidEnvironmentalHdrAnchor)).getPose().inverse().rotateVector(environmentalHdrMainLightDirection);
            }
        }
        float[] environmentalHdrAmbientSphericalHarmonics = lightEstimate.getEnvironmentalHdrAmbientSphericalHarmonics();
        if (environmentalHdrAmbientSphericalHarmonics != null) {
            this.lastValidEnvironmentalHdrAmbientSphericalHarmonics = environmentalHdrAmbientSphericalHarmonics;
        }
        float[] environmentalHdrMainLightIntensity = lightEstimate.getEnvironmentalHdrMainLightIntensity();
        if (environmentalHdrMainLightIntensity != null) {
            this.lastValidEnvironmentalHdrMainLightIntensity = environmentalHdrMainLightIntensity;
        }
        if (this.lastValidEnvironmentalHdrAnchor == null || (fArr = this.lastValidEnvironmentalHdrMainLightIntensity) == null || this.lastValidEnvironmentalHdrAmbientSphericalHarmonics == null || this.lastValidEnvironmentalHdrMainLightDirection == null) {
            return;
        }
        float max = Math.max(1.0f, Math.max(Math.max(fArr[0], fArr[1]), this.lastValidEnvironmentalHdrMainLightIntensity[2]));
        float[] fArr2 = this.lastValidEnvironmentalHdrMainLightIntensity;
        Color color = new Color(fArr2[0] / max, fArr2[1] / max, fArr2[2] / max);
        ArImage[] acquireEnvironmentalHdrCubeMap = lightEstimate.acquireEnvironmentalHdrCubeMap();
        float[] rotateVector = ((Anchor) Preconditions.checkNotNull(this.lastValidEnvironmentalHdrAnchor)).getPose().rotateVector((float[]) Preconditions.checkNotNull(this.lastValidEnvironmentalHdrMainLightDirection));
        if (this.onNextHdrLightingEstimate != null) {
            this.onNextHdrLightingEstimate.accept(new EnvironmentalHdrLightEstimate(this.lastValidEnvironmentalHdrAmbientSphericalHarmonics, rotateVector, color, max, acquireEnvironmentalHdrCubeMap));
            this.onNextHdrLightingEstimate = null;
        }
        getScene().setEnvironmentalHdrLightEstimate(this.lastValidEnvironmentalHdrAmbientSphericalHarmonics, rotateVector, color, max, acquireEnvironmentalHdrCubeMap);
        for (ArImage arImage : acquireEnvironmentalHdrCubeMap) {
            arImage.close();
        }
    }

    public ArSceneView(Context context, AttributeSet attributeSet) {
        super(context, attributeSet);
        this.lightEstimationEnabled = true;
        this.isLightDirectionUpdateEnabled = true;
        this.onNextHdrLightingEstimate = null;
        this.lastValidPixelIntensity = 1.0f;
        this.lastValidColorCorrection = new Color(DEFAULT_COLOR_CORRECTION);
        this.colorCorrectionPixelIntensity = new float[4];
        this.pauseResumeTask = new SequentialTask();
        this.isPausedAR = false;
        this.isAutofocusEnabled = false;
        this.isDepthSupported = false;
        ((Renderer) Preconditions.checkNotNull(getRenderer())).enablePerformanceMode();
        initializeAr();
    }
}