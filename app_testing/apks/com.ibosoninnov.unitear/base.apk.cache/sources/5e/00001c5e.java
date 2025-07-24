package com.google.ar.sceneform;

import android.annotation.SuppressLint;
import android.content.Context;
import android.graphics.drawable.ColorDrawable;
import android.graphics.drawable.Drawable;
import android.util.AttributeSet;
import android.util.Log;
import android.view.Choreographer;
import android.view.MotionEvent;
import android.view.Surface;
import android.view.SurfaceView;
import c.b.a.a.a;
import c.d.b.a.n;
import com.google.ar.sceneform.rendering.Color;
import com.google.ar.sceneform.rendering.Renderer;
import com.google.ar.sceneform.utilities.AndroidPreconditions;
import com.google.ar.sceneform.utilities.MovingAverageMillisecondsTracker;
import com.google.ar.sceneform.utilities.Preconditions;

/* loaded from: classes.dex */
public class SceneView extends SurfaceView implements Choreographer.FrameCallback {
    private static final String TAG = SceneView.class.getSimpleName();

    /* renamed from: b  reason: collision with root package name */
    public static final /* synthetic */ int f5625b = 0;
    private AnimationTimeTransformer animationTimeTransformer;
    private Color backgroundColor;
    private volatile boolean debugEnabled;
    private final MovingAverageMillisecondsTracker frameRenderTracker;
    private final FrameTime frameTime;
    private final MovingAverageMillisecondsTracker frameTotalTracker;
    private final MovingAverageMillisecondsTracker frameUpdateTracker;
    private boolean isInitialized;
    private Renderer renderer;
    private Scene scene;

    /* loaded from: classes.dex */
    public interface AnimationTimeTransformer {
        long getAnimationTime(long j);
    }

    public SceneView(Context context) {
        super(context);
        this.renderer = null;
        this.frameTime = new FrameTime();
        this.debugEnabled = false;
        this.isInitialized = false;
        this.frameTotalTracker = new MovingAverageMillisecondsTracker();
        this.frameUpdateTracker = new MovingAverageMillisecondsTracker();
        this.frameRenderTracker = new MovingAverageMillisecondsTracker();
        this.animationTimeTransformer = n.f4310a;
        initialize();
    }

    public static void destroyAllResources() {
        Renderer.destroyAllResources();
    }

    private void doRender() {
        Renderer renderer = this.renderer;
        if (renderer == null) {
            return;
        }
        if (this.debugEnabled) {
            this.frameRenderTracker.beginSample();
        }
        renderer.render(this.debugEnabled);
        if (this.debugEnabled) {
            this.frameRenderTracker.endSample();
        }
    }

    private void doUpdate(long j) {
        if (this.debugEnabled) {
            this.frameUpdateTracker.beginSample();
        }
        this.frameTime.update(j);
        updateAnimation(j);
        this.scene.dispatchUpdate(this.frameTime);
        if (this.debugEnabled) {
            this.frameUpdateTracker.endSample();
        }
    }

    private void initialize() {
        if (this.isInitialized) {
            Log.w(TAG, "SceneView already initialized.");
            return;
        }
        if (!AndroidPreconditions.isMinAndroidApiLevel()) {
            Log.e(TAG, "Sceneform requires Android N or later");
            this.renderer = null;
        } else {
            Renderer renderer = new Renderer(this);
            this.renderer = renderer;
            Color color = this.backgroundColor;
            if (color != null) {
                renderer.setClearColor(color.inverseTonemap());
            }
            Scene scene = new Scene(this);
            this.scene = scene;
            this.renderer.setCameraProvider(scene.getCamera());
            initializeAnimation();
        }
        this.isInitialized = true;
    }

    private void initializeAnimation() {
    }

    public static long reclaimReleasedResources() {
        return Renderer.reclaimReleasedResources();
    }

    private void updateAnimation(long j) {
    }

    public void destroy() {
        Renderer renderer = this.renderer;
        if (renderer != null) {
            renderer.dispose();
            this.renderer = null;
        }
    }

    @Override // android.view.Choreographer.FrameCallback
    public void doFrame(long j) {
        Choreographer.getInstance().postFrameCallback(this);
        doFrameNoRepost(j);
    }

    public void doFrameNoRepost(long j) {
        if (this.debugEnabled) {
            this.frameTotalTracker.beginSample();
        }
        if (onBeginFrame(j)) {
            doUpdate(j);
            doRender();
        }
        if (this.debugEnabled) {
            this.frameTotalTracker.endSample();
            if ((System.currentTimeMillis() / 1000) % 60 == 0) {
                String str = TAG;
                StringBuilder x = a.x(" PERF COUNTER: frameRender: ");
                x.append(this.frameRenderTracker.getAverage());
                Log.d(str, x.toString());
                Log.d(str, " PERF COUNTER: frameTotal: " + this.frameTotalTracker.getAverage());
                Log.d(str, " PERF COUNTER: frameUpdate: " + this.frameUpdateTracker.getAverage());
            }
        }
    }

    public void enableDebug(boolean z) {
        this.debugEnabled = z;
    }

    public Renderer getRenderer() {
        return this.renderer;
    }

    public Scene getScene() {
        return this.scene;
    }

    public boolean isDebugEnabled() {
        return this.debugEnabled;
    }

    public boolean onBeginFrame(long j) {
        return true;
    }

    @Override // android.view.View
    public void onLayout(boolean z, int i, int i2, int i3, int i4) {
        super.onLayout(z, i, i2, i3, i4);
        ((Renderer) Preconditions.checkNotNull(this.renderer)).setDesiredSize(i3 - i, i4 - i2);
    }

    @Override // android.view.View
    @SuppressLint({"ClickableViewAccessibility"})
    public boolean onTouchEvent(MotionEvent motionEvent) {
        if (!super.onTouchEvent(motionEvent)) {
            this.scene.onTouchEvent(motionEvent);
        }
        return true;
    }

    public void pause() {
        Choreographer.getInstance().removeFrameCallback(this);
        Renderer renderer = this.renderer;
        if (renderer != null) {
            renderer.onPause();
        }
    }

    public void resume() {
        Renderer renderer = this.renderer;
        if (renderer != null) {
            renderer.onResume();
        }
        Choreographer.getInstance().removeFrameCallback(this);
        Choreographer.getInstance().postFrameCallback(this);
    }

    @Override // android.view.View
    public void setBackground(Drawable drawable) {
        if (drawable instanceof ColorDrawable) {
            Color color = new Color(((ColorDrawable) drawable).getColor());
            this.backgroundColor = color;
            Renderer renderer = this.renderer;
            if (renderer != null) {
                renderer.setClearColor(color.inverseTonemap());
                return;
            }
            return;
        }
        this.backgroundColor = null;
        Renderer renderer2 = this.renderer;
        if (renderer2 != null) {
            renderer2.setDefaultClearColor();
        }
        super.setBackground(drawable);
    }

    public void startMirroringToSurface(Surface surface, int i, int i2, int i3, int i4) {
        Renderer renderer = this.renderer;
        if (renderer != null) {
            renderer.startMirroring(surface, i, i2, i3, i4);
        }
    }

    public void stopMirroringToSurface(Surface surface) {
        Renderer renderer = this.renderer;
        if (renderer != null) {
            renderer.stopMirroring(surface);
        }
    }

    public SceneView(Context context, AttributeSet attributeSet) {
        super(context, attributeSet);
        this.renderer = null;
        this.frameTime = new FrameTime();
        this.debugEnabled = false;
        this.isInitialized = false;
        this.frameTotalTracker = new MovingAverageMillisecondsTracker();
        this.frameUpdateTracker = new MovingAverageMillisecondsTracker();
        this.frameRenderTracker = new MovingAverageMillisecondsTracker();
        this.animationTimeTransformer = n.f4310a;
        initialize();
    }
}