package com.google.ar.sceneform.ux;

import android.app.AlertDialog;
import android.content.DialogInterface;
import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.view.GestureDetector;
import android.view.LayoutInflater;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup;
import android.view.ViewTreeObserver;
import android.widget.FrameLayout;
import android.widget.Toast;
import androidx.fragment.app.Fragment;
import b.j.c.a;
import b.q.b.d;
import com.google.ar.core.ArCoreApk;
import com.google.ar.core.Config;
import com.google.ar.core.Frame;
import com.google.ar.core.HitResult;
import com.google.ar.core.Plane;
import com.google.ar.core.Session;
import com.google.ar.core.Trackable;
import com.google.ar.core.TrackingState;
import com.google.ar.core.exceptions.CameraNotAvailableException;
import com.google.ar.core.exceptions.UnavailableException;
import com.google.ar.sceneform.ArSceneView;
import com.google.ar.sceneform.FrameTime;
import com.google.ar.sceneform.HitTestResult;
import com.google.ar.sceneform.Scene;
import com.google.ar.sceneform.rendering.ModelRenderable;
import com.google.ar.sceneform.ux.BaseArFragment;
import com.google.ar.sceneform.ux.FootprintSelectionVisualizer;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.Set;
import java.util.function.Consumer;
import java.util.function.Function;

/* loaded from: classes.dex */
public abstract class BaseArFragment extends Fragment implements Scene.OnPeekTouchListener, Scene.OnUpdateListener {
    private static final int RC_PERMISSIONS = 1010;
    private static final String TAG = BaseArFragment.class.getSimpleName();

    /* renamed from: b  reason: collision with root package name */
    public static final /* synthetic */ int f5629b = 0;
    private ArSceneView arSceneView;
    private FrameLayout frameLayout;
    private GestureDetector gestureDetector;
    private boolean installRequested;
    private boolean isStarted;
    private OnSessionInitializationListener onSessionInitializationListener;
    private OnTapArPlaneListener onTapArPlaneListener;
    private PlaneDiscoveryController planeDiscoveryController;
    private TransformationSystem transformationSystem;
    private boolean sessionInitializationFailed = false;
    private boolean canRequestDangerousPermissions = true;
    private final ViewTreeObserver.OnWindowFocusChangeListener onFocusListener = new ViewTreeObserver.OnWindowFocusChangeListener() { // from class: c.d.b.a.t.b
        @Override // android.view.ViewTreeObserver.OnWindowFocusChangeListener
        public final void onWindowFocusChanged(boolean z) {
            BaseArFragment.this.onWindowFocusChanged(z);
        }
    };

    /* renamed from: com.google.ar.sceneform.ux.BaseArFragment$4  reason: invalid class name */
    /* loaded from: classes.dex */
    public static /* synthetic */ class AnonymousClass4 {
        public static final /* synthetic */ int[] $SwitchMap$com$google$ar$core$ArCoreApk$InstallStatus;

        static {
            ArCoreApk.InstallStatus.values();
            int[] iArr = new int[2];
            $SwitchMap$com$google$ar$core$ArCoreApk$InstallStatus = iArr;
            try {
                iArr[ArCoreApk.InstallStatus.INSTALL_REQUESTED.ordinal()] = 1;
            } catch (NoSuchFieldError unused) {
            }
            try {
                $SwitchMap$com$google$ar$core$ArCoreApk$InstallStatus[ArCoreApk.InstallStatus.INSTALLED.ordinal()] = 2;
            } catch (NoSuchFieldError unused2) {
            }
        }
    }

    /* loaded from: classes.dex */
    public interface OnSessionInitializationListener {
        void onSessionInitialization(Session session);
    }

    /* loaded from: classes.dex */
    public interface OnTapArPlaneListener {
        void onTapPlane(HitResult hitResult, Plane plane, MotionEvent motionEvent);
    }

    private Session createSession() {
        Session createSessionWithFeatures = createSessionWithFeatures();
        return createSessionWithFeatures == null ? new Session(requireActivity()) : createSessionWithFeatures;
    }

    private View loadPlaneDiscoveryView(LayoutInflater layoutInflater, ViewGroup viewGroup) {
        return layoutInflater.inflate(R.layout.sceneform_plane_discovery_layout, viewGroup, false);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void onSingleTap(MotionEvent motionEvent) {
        Frame arFrame = this.arSceneView.getArFrame();
        this.transformationSystem.selectNode(null);
        OnTapArPlaneListener onTapArPlaneListener = this.onTapArPlaneListener;
        if (arFrame == null || onTapArPlaneListener == null || motionEvent == null || arFrame.getCamera().getTrackingState() != TrackingState.TRACKING) {
            return;
        }
        for (HitResult hitResult : arFrame.hitTest(motionEvent)) {
            Trackable trackable = hitResult.getTrackable();
            if (trackable instanceof Plane) {
                Plane plane = (Plane) trackable;
                if (plane.isPoseInPolygon(hitResult.getHitPose())) {
                    onTapArPlaneListener.onTapPlane(hitResult, plane, motionEvent);
                    return;
                }
            }
        }
    }

    private void start() {
        if (this.isStarted || getActivity() == null) {
            return;
        }
        this.isStarted = true;
        try {
            this.arSceneView.resume();
        } catch (CameraNotAvailableException unused) {
            this.sessionInitializationFailed = true;
        }
        if (this.sessionInitializationFailed) {
            return;
        }
        this.planeDiscoveryController.show();
    }

    private void stop() {
        if (this.isStarted) {
            this.isStarted = false;
            this.planeDiscoveryController.hide();
            this.arSceneView.pause();
        }
    }

    public Session createSessionWithFeatures() {
        if (ArFragment.isFrontCam) {
            return new Session(requireActivity(), EnumSet.of(Session.Feature.FRONT_CAMERA));
        }
        return new Session(requireActivity(), getSessionFeatures());
    }

    public abstract String[] getAdditionalPermissions();

    public ArSceneView getArSceneView() {
        return this.arSceneView;
    }

    public Boolean getCanRequestDangerousPermissions() {
        return Boolean.valueOf(this.canRequestDangerousPermissions);
    }

    public PlaneDiscoveryController getPlaneDiscoveryController() {
        return this.planeDiscoveryController;
    }

    public abstract Config getSessionConfiguration(Session session);

    public abstract Set<Session.Feature> getSessionFeatures();

    public TransformationSystem getTransformationSystem() {
        return this.transformationSystem;
    }

    public abstract void handleSessionException(UnavailableException unavailableException);

    public final void initializeSession() {
        UnavailableException unavailableException;
        if (this.sessionInitializationFailed) {
            return;
        }
        if (a.a(requireActivity(), "android.permission.CAMERA") == 0) {
            try {
                if (requestInstall()) {
                    return;
                }
                Session createSession = createSession();
                OnSessionInitializationListener onSessionInitializationListener = this.onSessionInitializationListener;
                if (onSessionInitializationListener != null) {
                    onSessionInitializationListener.onSessionInitialization(createSession);
                }
                Config sessionConfiguration = getSessionConfiguration(createSession);
                sessionConfiguration.setUpdateMode(Config.UpdateMode.LATEST_CAMERA_IMAGE);
                createSession.configure(sessionConfiguration);
                getArSceneView().setupSession(createSession);
                return;
            } catch (UnavailableException e2) {
                unavailableException = e2;
                this.sessionInitializationFailed = true;
                handleSessionException(unavailableException);
                return;
            } catch (Exception e3) {
                unavailableException = new UnavailableException();
                unavailableException.initCause(e3);
                this.sessionInitializationFailed = true;
                handleSessionException(unavailableException);
                return;
            }
        }
        requestDangerousPermissions();
    }

    public abstract boolean isArRequired();

    public TransformationSystem makeTransformationSystem() {
        return new TransformationSystem(getResources().getDisplayMetrics(), new FootprintSelectionVisualizer());
    }

    @Override // androidx.fragment.app.Fragment
    public View onCreateView(LayoutInflater layoutInflater, ViewGroup viewGroup, Bundle bundle) {
        FrameLayout frameLayout = (FrameLayout) layoutInflater.inflate(R.layout.sceneform_ux_fragment_layout, viewGroup, false);
        this.frameLayout = frameLayout;
        this.arSceneView = (ArSceneView) frameLayout.findViewById(R.id.sceneform_ar_scene_view);
        View loadPlaneDiscoveryView = loadPlaneDiscoveryView(layoutInflater, viewGroup);
        if (loadPlaneDiscoveryView != null) {
            this.frameLayout.addView(loadPlaneDiscoveryView);
        }
        this.planeDiscoveryController = new PlaneDiscoveryController(loadPlaneDiscoveryView);
        this.transformationSystem = makeTransformationSystem();
        this.gestureDetector = new GestureDetector(getContext(), new GestureDetector.SimpleOnGestureListener() { // from class: com.google.ar.sceneform.ux.BaseArFragment.1
            @Override // android.view.GestureDetector.SimpleOnGestureListener, android.view.GestureDetector.OnGestureListener
            public boolean onDown(MotionEvent motionEvent) {
                return true;
            }

            @Override // android.view.GestureDetector.SimpleOnGestureListener, android.view.GestureDetector.OnGestureListener
            public boolean onSingleTapUp(MotionEvent motionEvent) {
                BaseArFragment.this.onSingleTap(motionEvent);
                return true;
            }
        });
        this.arSceneView.getScene().addOnPeekTouchListener(this);
        this.arSceneView.getScene().addOnUpdateListener(this);
        if (isArRequired()) {
            requestDangerousPermissions();
        }
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
    public void onRequestPermissionsResult(int i, String[] strArr, int[] iArr) {
        if (a.a(requireActivity(), "android.permission.CAMERA") == 0) {
            return;
        }
        new AlertDialog.Builder(requireActivity(), 16974374).setTitle("Camera permission required").setMessage("Add camera permission via Settings?").setPositiveButton(17039370, new DialogInterface.OnClickListener() { // from class: com.google.ar.sceneform.ux.BaseArFragment.3
            @Override // android.content.DialogInterface.OnClickListener
            public void onClick(DialogInterface dialogInterface, int i2) {
                Intent intent = new Intent();
                intent.setAction("android.settings.APPLICATION_DETAILS_SETTINGS");
                intent.setData(Uri.fromParts("package", BaseArFragment.this.requireActivity().getPackageName(), null));
                BaseArFragment.this.requireActivity().startActivity(intent);
                BaseArFragment.this.setCanRequestDangerousPermissions(Boolean.TRUE);
            }
        }).setNegativeButton(17039360, (DialogInterface.OnClickListener) null).setIcon(17301543).setOnDismissListener(new DialogInterface.OnDismissListener() { // from class: com.google.ar.sceneform.ux.BaseArFragment.2
            @Override // android.content.DialogInterface.OnDismissListener
            public void onDismiss(DialogInterface dialogInterface) {
                if (BaseArFragment.this.getCanRequestDangerousPermissions().booleanValue()) {
                    return;
                }
                BaseArFragment.this.requireActivity().finish();
            }
        }).show();
    }

    @Override // androidx.fragment.app.Fragment
    public void onResume() {
        super.onResume();
        if (isArRequired() && this.arSceneView.getSession() == null) {
            initializeSession();
        }
        start();
    }

    @Override // com.google.ar.sceneform.Scene.OnUpdateListener
    public void onUpdate(FrameTime frameTime) {
        Frame arFrame = this.arSceneView.getArFrame();
        if (arFrame == null) {
            return;
        }
        for (Plane plane : arFrame.getUpdatedTrackables(Plane.class)) {
            if (plane.getTrackingState() == TrackingState.TRACKING) {
                this.planeDiscoveryController.hide();
            }
        }
    }

    public void onWindowFocusChanged(boolean z) {
        d activity = getActivity();
        if (!z || activity == null) {
            return;
        }
        activity.getWindow().getDecorView().setSystemUiVisibility(5894);
        activity.getWindow().addFlags(128);
    }

    public void requestDangerousPermissions() {
        if (this.canRequestDangerousPermissions) {
            this.canRequestDangerousPermissions = false;
            ArrayList arrayList = new ArrayList();
            String[] additionalPermissions = getAdditionalPermissions();
            int length = additionalPermissions != null ? additionalPermissions.length : 0;
            for (int i = 0; i < length; i++) {
                if (a.a(requireActivity(), additionalPermissions[i]) != 0) {
                    arrayList.add(additionalPermissions[i]);
                }
            }
            if (a.a(requireActivity(), "android.permission.CAMERA") != 0) {
                arrayList.add("android.permission.CAMERA");
            }
            if (arrayList.isEmpty()) {
                return;
            }
            requestPermissions((String[]) arrayList.toArray(new String[arrayList.size()]), 1010);
        }
    }

    public final boolean requestInstall() {
        if (ArCoreApk.getInstance().requestInstall(requireActivity(), !this.installRequested).ordinal() != 1) {
            return false;
        }
        this.installRequested = true;
        return true;
    }

    public void setCanRequestDangerousPermissions(Boolean bool) {
        this.canRequestDangerousPermissions = bool.booleanValue();
    }

    public void setOnSessionInitializationListener(OnSessionInitializationListener onSessionInitializationListener) {
        this.onSessionInitializationListener = onSessionInitializationListener;
    }

    public void setOnTapArPlaneListener(OnTapArPlaneListener onTapArPlaneListener) {
        this.onTapArPlaneListener = onTapArPlaneListener;
    }

    public void setupSelectionRenderable(final FootprintSelectionVisualizer footprintSelectionVisualizer) {
        ModelRenderable.builder().setSource(getActivity(), R.raw.sceneform_footprint).setIsFilamentGltf(true).build().thenAccept(new Consumer() { // from class: c.d.b.a.t.c
            @Override // java.util.function.Consumer
            public final void accept(Object obj) {
                FootprintSelectionVisualizer footprintSelectionVisualizer2 = FootprintSelectionVisualizer.this;
                ModelRenderable modelRenderable = (ModelRenderable) obj;
                int i = BaseArFragment.f5629b;
                if (footprintSelectionVisualizer2.getFootprintRenderable() == null) {
                    footprintSelectionVisualizer2.setFootprintRenderable(modelRenderable);
                }
            }
        }).exceptionally(new Function() { // from class: c.d.b.a.t.a
            @Override // java.util.function.Function
            public final Object apply(Object obj) {
                Throwable th = (Throwable) obj;
                Toast makeText = Toast.makeText(BaseArFragment.this.getContext(), "Unable to load footprint renderable", 1);
                makeText.setGravity(17, 0, 0);
                makeText.show();
                return null;
            }
        });
    }
}