package com.google.ar.sceneform.ux;

import android.os.Bundle;
import android.util.Log;
import android.view.GestureDetector;
import android.view.LayoutInflater;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup;
import android.view.ViewTreeObserver;
import android.widget.FrameLayout;
import android.widget.Toast;
import androidx.fragment.app.Fragment;
import b.q.b.d;
import c.b.a.a.a;
import com.google.ar.sceneform.FrameTime;
import com.google.ar.sceneform.HitTestResult;
import com.google.ar.sceneform.Scene;
import com.google.ar.sceneform.SimpleSceneView;
import com.google.ar.sceneform.rendering.ModelRenderable;
import com.google.ar.sceneform.ux.BaseSimpleFragment;
import com.google.ar.sceneform.ux.FootprintSelectionVisualizer;
import java.util.function.Consumer;
import java.util.function.Function;

/* loaded from: classes.dex */
public abstract class BaseSimpleFragment extends Fragment implements Scene.OnPeekTouchListener, Scene.OnUpdateListener {
    private static final int RC_PERMISSIONS = 1010;
    private static final String TAG = BaseSimpleFragment.class.getSimpleName();

    /* renamed from: b  reason: collision with root package name */
    public static final /* synthetic */ int f5630b = 0;
    private SimpleSceneView arSceneView;
    private FrameLayout frameLayout;
    private GestureDetector gestureDetector;
    private boolean isStarted;
    private TransformationSystem transformationSystem;
    private boolean canRequestDangerousPermissions = true;
    private final ViewTreeObserver.OnWindowFocusChangeListener onFocusListener = new ViewTreeObserver.OnWindowFocusChangeListener() { // from class: c.d.b.a.t.e
        @Override // android.view.ViewTreeObserver.OnWindowFocusChangeListener
        public final void onWindowFocusChanged(boolean z) {
            BaseSimpleFragment.this.onWindowFocusChanged(z);
        }
    };

    /* JADX INFO: Access modifiers changed from: private */
    public void onSingleTap(MotionEvent motionEvent) {
        this.transformationSystem.selectNode(null);
    }

    private void start() {
        if (this.isStarted || getActivity() == null) {
            return;
        }
        this.isStarted = true;
        try {
            this.arSceneView.resume();
        } catch (Exception e2) {
            String str = TAG;
            StringBuilder x = a.x("Start ");
            x.append(e2.getMessage());
            Log.e(str, x.toString());
        }
    }

    private void stop() {
        if (this.isStarted) {
            this.isStarted = false;
            this.arSceneView.pause();
        }
    }

    public SimpleSceneView getArSceneView() {
        return this.arSceneView;
    }

    public int getCameraTextureId() {
        return this.arSceneView.getCameraTextureId();
    }

    public TransformationSystem getTransformationSystem() {
        return this.transformationSystem;
    }

    public TransformationSystem makeTransformationSystem() {
        return new TransformationSystem(getResources().getDisplayMetrics(), new FootprintSelectionVisualizer());
    }

    @Override // androidx.fragment.app.Fragment
    public View onCreateView(LayoutInflater layoutInflater, ViewGroup viewGroup, Bundle bundle) {
        FrameLayout frameLayout = (FrameLayout) layoutInflater.inflate(R.layout.simple_sceneform_ux_fragment_layout, viewGroup, false);
        this.frameLayout = frameLayout;
        this.arSceneView = (SimpleSceneView) frameLayout.findViewById(R.id.sceneform_ar_scene_view);
        this.transformationSystem = makeTransformationSystem();
        this.gestureDetector = new GestureDetector(getContext(), new GestureDetector.SimpleOnGestureListener() { // from class: com.google.ar.sceneform.ux.BaseSimpleFragment.1
            @Override // android.view.GestureDetector.SimpleOnGestureListener, android.view.GestureDetector.OnGestureListener
            public boolean onDown(MotionEvent motionEvent) {
                return true;
            }

            @Override // android.view.GestureDetector.SimpleOnGestureListener, android.view.GestureDetector.OnGestureListener
            public boolean onSingleTapUp(MotionEvent motionEvent) {
                BaseSimpleFragment.this.onSingleTap(motionEvent);
                return true;
            }
        });
        this.arSceneView.getScene().addOnPeekTouchListener(this);
        this.arSceneView.getScene().addOnUpdateListener(this);
        this.arSceneView.getViewTreeObserver().addOnWindowFocusChangeListener(this.onFocusListener);
        return this.frameLayout;
    }

    @Override // androidx.fragment.app.Fragment
    public void onDestroy() {
        stop();
        this.arSceneView.destroy();
        super.onDestroy();
    }

    @Override // androidx.fragment.app.Fragment
    public void onDestroyView() {
        super.onDestroyView();
        this.arSceneView.getViewTreeObserver().removeOnWindowFocusChangeListener(this.onFocusListener);
    }

    @Override // androidx.fragment.app.Fragment
    public void onPause() {
        super.onPause();
        stop();
    }

    @Override // com.google.ar.sceneform.Scene.OnPeekTouchListener
    public void onPeekTouch(HitTestResult hitTestResult, MotionEvent motionEvent) {
        this.transformationSystem.onTouch(hitTestResult, motionEvent);
        if (hitTestResult.getNode() == null) {
            this.gestureDetector.onTouchEvent(motionEvent);
        }
    }

    @Override // androidx.fragment.app.Fragment
    public void onResume() {
        super.onResume();
        start();
    }

    @Override // com.google.ar.sceneform.Scene.OnUpdateListener
    public void onUpdate(FrameTime frameTime) {
    }

    public void onWindowFocusChanged(boolean z) {
        d activity = getActivity();
        if (!z || activity == null) {
            return;
        }
        activity.getWindow().getDecorView().setSystemUiVisibility(5894);
        activity.getWindow().addFlags(128);
    }

    public void setupSelectionRenderable(final FootprintSelectionVisualizer footprintSelectionVisualizer) {
        ModelRenderable.builder().setSource(getActivity(), R.raw.sceneform_footprint).setIsFilamentGltf(true).build().thenAccept(new Consumer() { // from class: c.d.b.a.t.f
            @Override // java.util.function.Consumer
            public final void accept(Object obj) {
                FootprintSelectionVisualizer footprintSelectionVisualizer2 = FootprintSelectionVisualizer.this;
                ModelRenderable modelRenderable = (ModelRenderable) obj;
                int i = BaseSimpleFragment.f5630b;
                if (footprintSelectionVisualizer2.getFootprintRenderable() == null) {
                    footprintSelectionVisualizer2.setFootprintRenderable(modelRenderable);
                }
            }
        }).exceptionally(new Function() { // from class: c.d.b.a.t.d
            @Override // java.util.function.Function
            public final Object apply(Object obj) {
                Throwable th = (Throwable) obj;
                Toast makeText = Toast.makeText(BaseSimpleFragment.this.getContext(), "Unable to load footprint renderable", 1);
                makeText.setGravity(17, 0, 0);
                makeText.show();
                return null;
            }
        });
    }
}