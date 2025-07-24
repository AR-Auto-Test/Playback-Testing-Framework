package com.google.ar.sceneform;

import android.content.Context;
import android.util.AttributeSet;
import android.util.Log;
import android.view.Display;
import android.view.WindowManager;
import c.b.a.a.a;
import com.google.ar.sceneform.SimpleSceneView;
import com.google.ar.sceneform.rendering.ExternalTexture;
import com.google.ar.sceneform.rendering.GLHelper;
import com.google.ar.sceneform.rendering.Renderer;
import com.google.ar.sceneform.rendering.SimpleCameraStream;
import com.google.ar.sceneform.rendering.ThreadPools;
import com.google.ar.sceneform.utilities.Preconditions;
import java.lang.ref.WeakReference;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Executor;

/* loaded from: classes.dex */
public class SimpleSceneView extends SceneView {
    private static final float RECREATE_LIGHTING_ANCHOR_DISTANCE = 0.5f;
    private static final String TAG = SimpleSceneView.class.getSimpleName();
    private SimpleCameraStream cameraStream;
    private int cameraTextureId;
    private Display display;
    private ExternalTexture externalTexture;
    private final SequentialTask pauseResumeTask;

    public SimpleSceneView(Context context) {
        super(context);
        this.externalTexture = null;
        this.pauseResumeTask = new SequentialTask();
        ((Renderer) Preconditions.checkNotNull(getRenderer())).enablePerformanceMode();
        initializeAr();
    }

    private void initializeAr() {
        this.display = ((WindowManager) getContext().getSystemService(WindowManager.class)).getDefaultDisplay();
        initializeCameraStream();
    }

    private void initializeCameraStream() {
        this.cameraTextureId = GLHelper.createCameraTexture();
        Renderer renderer = (Renderer) Preconditions.checkNotNull(getRenderer());
        renderer.setDesiredSize(getWidth(), getHeight());
        this.cameraStream = new SimpleCameraStream(this.cameraTextureId, renderer);
        String str = TAG;
        StringBuilder x = a.x("initializeCameraStream ");
        x.append(this.cameraTextureId);
        Log.d(str, x.toString());
    }

    public static /* synthetic */ void lambda$pauseAsync$0(WeakReference weakReference) {
        SimpleSceneView simpleSceneView = (SimpleSceneView) weakReference.get();
        if (simpleSceneView == null) {
            return;
        }
        simpleSceneView.pauseScene();
    }

    private static boolean loadUnifiedJni() {
        return false;
    }

    private void pauseScene() {
        super.pause();
    }

    private void reportEngineType() {
    }

    private void resumeScene() {
        try {
            super.resume();
        } catch (Exception e2) {
            throw new IllegalStateException(e2);
        }
    }

    public int getCameraTextureId() {
        return this.cameraTextureId;
    }

    @Override // com.google.ar.sceneform.SceneView
    public boolean onBeginFrame(long j) {
        if (this.pauseResumeTask.isDone()) {
            if (this.externalTexture == null || this.cameraStream.isTextureInitialized()) {
                return true;
            }
            this.cameraStream.initializeTexture(this.externalTexture);
            return true;
        }
        return false;
    }

    @Override // com.google.ar.sceneform.SceneView, android.view.View
    public void onLayout(boolean z, int i, int i2, int i3, int i4) {
        super.onLayout(z, i, i2, i3, i4);
    }

    @Override // com.google.ar.sceneform.SceneView
    public void pause() {
        pauseScene();
    }

    public CompletableFuture<Void> pauseAsync(Executor executor) {
        final WeakReference weakReference = new WeakReference(this);
        return this.pauseResumeTask.appendRunnable(new Runnable() { // from class: c.d.b.a.o
            @Override // java.lang.Runnable
            public final void run() {
                SimpleSceneView.lambda$pauseAsync$0(weakReference);
            }
        }, ThreadPools.getMainExecutor());
    }

    @Override // com.google.ar.sceneform.SceneView
    public void resume() {
        resumeScene();
    }

    public void setExternalTexture(ExternalTexture externalTexture) {
        this.externalTexture = externalTexture;
    }

    public SimpleSceneView(Context context, AttributeSet attributeSet) {
        super(context, attributeSet);
        this.externalTexture = null;
        this.pauseResumeTask = new SequentialTask();
        ((Renderer) Preconditions.checkNotNull(getRenderer())).enablePerformanceMode();
        initializeAr();
    }
}