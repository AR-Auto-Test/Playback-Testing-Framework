package com.google.ar.sceneform;

import android.view.MotionEvent;
import android.view.View;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import com.google.ar.core.Pose;
import com.google.ar.sceneform.Camera;
import com.google.ar.sceneform.collision.Ray;
import com.google.ar.sceneform.math.MathHelper;
import com.google.ar.sceneform.math.Matrix;
import com.google.ar.sceneform.math.Quaternion;
import com.google.ar.sceneform.math.Vector3;
import com.google.ar.sceneform.rendering.CameraProvider;
import com.google.ar.sceneform.rendering.EngineInstance;
import com.google.ar.sceneform.utilities.Preconditions;

/* loaded from: classes.dex */
public class Camera extends Node implements CameraProvider {
    private static final float DEFAULT_FAR_PLANE = 30.0f;
    private static final float DEFAULT_NEAR_PLANE = 0.01f;
    private static final float DEFAULT_VERTICAL_FOV_DEGREES = 90.0f;
    private static final int FALLBACK_VIEW_HEIGHT = 1080;
    private static final int FALLBACK_VIEW_WIDTH = 1920;
    private boolean areMatricesInitialized;
    private final boolean isArCamera;
    private final Matrix viewMatrix = new Matrix();
    private final Matrix projectionMatrix = new Matrix();
    private float nearPlane = DEFAULT_NEAR_PLANE;
    private float farPlane = DEFAULT_FAR_PLANE;
    private float verticalFov = DEFAULT_VERTICAL_FOV_DEGREES;

    public Camera(boolean z) {
        this.isArCamera = z;
    }

    private int getViewHeight() {
        Scene scene = getScene();
        return (scene == null || EngineInstance.isHeadlessMode()) ? FALLBACK_VIEW_HEIGHT : scene.getView().getHeight();
    }

    private int getViewWidth() {
        Scene scene = getScene();
        return (scene == null || EngineInstance.isHeadlessMode()) ? FALLBACK_VIEW_WIDTH : scene.getView().getWidth();
    }

    private void refreshProjectionMatrix() {
        if (this.isArCamera) {
            return;
        }
        int viewWidth = getViewWidth();
        int viewHeight = getViewHeight();
        if (viewWidth == 0 || viewHeight == 0) {
            return;
        }
        setPerspective(this.verticalFov, viewWidth / viewHeight, this.nearPlane, this.farPlane);
    }

    private void setPerspective(float f2, float f3, float f4, float f5) {
        if (f2 <= StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD || f2 >= 180.0f) {
            throw new IllegalArgumentException("Parameter \"verticalFovInDegrees\" is out of the valid range of (0, 180) degrees.");
        }
        if (f3 > StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
            float tan = ((float) Math.tan(Math.toRadians(f2) * 0.5d)) * f4;
            float f6 = tan * f3;
            setPerspective(-f6, f6, -tan, tan, f4, f5);
            return;
        }
        throw new IllegalArgumentException("Parameter \"aspect\" must be greater than zero.");
    }

    private boolean unproject(float f2, float f3, float f4, Vector3 vector3) {
        Preconditions.checkNotNull(vector3, "Parameter \"dest\" was null.");
        Matrix matrix = new Matrix();
        Matrix.multiply(this.projectionMatrix, this.viewMatrix, matrix);
        Matrix.invert(matrix, matrix);
        int viewWidth = getViewWidth();
        float viewHeight = getViewHeight();
        float f5 = ((f2 / viewWidth) * 2.0f) - 1.0f;
        float f6 = (((viewHeight - f3) / viewHeight) * 2.0f) - 1.0f;
        float f7 = (f4 * 2.0f) - 1.0f;
        float[] fArr = matrix.data;
        vector3.x = (fArr[12] * 1.0f) + (fArr[8] * f7) + (fArr[4] * f6) + (fArr[0] * f5);
        vector3.y = (fArr[13] * 1.0f) + (fArr[9] * f7) + (fArr[5] * f6) + (fArr[1] * f5);
        vector3.z = (fArr[14] * 1.0f) + (fArr[10] * f7) + (fArr[6] * f6) + (fArr[2] * f5);
        float f8 = (fArr[15] * 1.0f) + (f7 * fArr[11]) + (f6 * fArr[7]) + (f5 * fArr[3]);
        if (MathHelper.almostEqualRelativeAndAbs(f8, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD)) {
            vector3.set(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
            return false;
        }
        vector3.set(vector3.scaled(1.0f / f8));
        return true;
    }

    public /* synthetic */ void a(View view, int i, int i2, int i3, int i4, int i5, int i6, int i7, int i8) {
        refreshProjectionMatrix();
    }

    @Override // com.google.ar.sceneform.rendering.CameraProvider
    public float getFarClipPlane() {
        return this.farPlane;
    }

    @Override // com.google.ar.sceneform.rendering.CameraProvider
    public float getNearClipPlane() {
        return this.nearPlane;
    }

    @Override // com.google.ar.sceneform.rendering.CameraProvider
    public Matrix getProjectionMatrix() {
        return this.projectionMatrix;
    }

    public float getVerticalFovDegrees() {
        if (this.isArCamera) {
            if (this.areMatricesInitialized) {
                return (float) Math.toDegrees(Math.atan(1.0d / this.projectionMatrix.data[5]) * 2.0d);
            }
            throw new IllegalStateException("Cannot get the field of view for AR cameras until the first frame after ARCore has been resumed.");
        }
        return this.verticalFov;
    }

    @Override // com.google.ar.sceneform.rendering.CameraProvider
    public Matrix getViewMatrix() {
        return this.viewMatrix;
    }

    public Ray motionEventToRay(MotionEvent motionEvent) {
        Preconditions.checkNotNull(motionEvent, "Parameter \"motionEvent\" was null.");
        int actionIndex = motionEvent.getActionIndex();
        return screenPointToRay(motionEvent.getX(actionIndex), motionEvent.getY(actionIndex));
    }

    public Ray screenPointToRay(float f2, float f3) {
        Vector3 vector3 = new Vector3();
        Vector3 vector32 = new Vector3();
        unproject(f2, f3, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, vector3);
        unproject(f2, f3, 1.0f, vector32);
        return new Ray(vector3, Vector3.subtract(vector32, vector3));
    }

    public void setFarClipPlane(float f2) {
        this.farPlane = f2;
        if (this.isArCamera) {
            return;
        }
        refreshProjectionMatrix();
    }

    @Override // com.google.ar.sceneform.Node
    public void setLocalPosition(Vector3 vector3) {
        if (!this.isArCamera) {
            super.setLocalPosition(vector3);
            Matrix.invert(getWorldModelMatrix(), this.viewMatrix);
            return;
        }
        throw new UnsupportedOperationException("Camera's position cannot be changed, it is controller by the ARCore camera pose.");
    }

    @Override // com.google.ar.sceneform.Node
    public void setLocalRotation(Quaternion quaternion) {
        if (!this.isArCamera) {
            super.setLocalRotation(quaternion);
            Matrix.invert(getWorldModelMatrix(), this.viewMatrix);
            return;
        }
        throw new UnsupportedOperationException("Camera's rotation cannot be changed, it is controller by the ARCore camera pose.");
    }

    public void setNearClipPlane(float f2) {
        this.nearPlane = f2;
        if (this.isArCamera) {
            return;
        }
        refreshProjectionMatrix();
    }

    @Override // com.google.ar.sceneform.Node
    public void setParent(NodeParent nodeParent) {
        throw new UnsupportedOperationException("Camera's parent cannot be changed, it is always the scene.");
    }

    public void setProjectionMatrix(Matrix matrix) {
        this.projectionMatrix.set(matrix.data);
    }

    public void setVerticalFovDegrees(float f2) {
        this.verticalFov = f2;
        if (!this.isArCamera) {
            refreshProjectionMatrix();
            return;
        }
        throw new UnsupportedOperationException("Cannot set the field of view for AR cameras.");
    }

    @Override // com.google.ar.sceneform.Node
    public void setWorldPosition(Vector3 vector3) {
        if (!this.isArCamera) {
            super.setWorldPosition(vector3);
            Matrix.invert(getWorldModelMatrix(), this.viewMatrix);
            return;
        }
        throw new UnsupportedOperationException("Camera's position cannot be changed, it is controller by the ARCore camera pose.");
    }

    @Override // com.google.ar.sceneform.Node
    public void setWorldRotation(Quaternion quaternion) {
        if (!this.isArCamera) {
            super.setWorldRotation(quaternion);
            Matrix.invert(getWorldModelMatrix(), this.viewMatrix);
            return;
        }
        throw new UnsupportedOperationException("Camera's rotation cannot be changed, it is controller by the ARCore camera pose.");
    }

    @Override // com.google.ar.sceneform.rendering.CameraProvider
    public void updateTrackedPose(com.google.ar.core.Camera camera) {
        Preconditions.checkNotNull(camera, "Parameter \"camera\" was null.");
        camera.getProjectionMatrix(this.projectionMatrix.data, 0, this.nearPlane, this.farPlane);
        camera.getViewMatrix(this.viewMatrix.data, 0);
        Pose displayOrientedPose = camera.getDisplayOrientedPose();
        Vector3 extractPositionFromPose = ArHelpers.extractPositionFromPose(displayOrientedPose);
        Quaternion extractRotationFromPose = ArHelpers.extractRotationFromPose(displayOrientedPose);
        super.setWorldPosition(extractPositionFromPose);
        super.setWorldRotation(extractRotationFromPose);
        this.areMatricesInitialized = true;
    }

    public Vector3 worldToScreenPoint(Vector3 vector3) {
        Matrix matrix = new Matrix();
        Matrix.multiply(this.projectionMatrix, this.viewMatrix, matrix);
        int viewWidth = getViewWidth();
        int viewHeight = getViewHeight();
        float f2 = vector3.x;
        float f3 = vector3.y;
        float f4 = vector3.z;
        Vector3 vector32 = new Vector3();
        float[] fArr = matrix.data;
        float f5 = (fArr[12] * 1.0f) + (fArr[8] * f4) + (fArr[4] * f3) + (fArr[0] * f2);
        vector32.x = f5;
        float f6 = (fArr[13] * 1.0f) + (fArr[9] * f4) + (fArr[5] * f3) + (fArr[1] * f2);
        vector32.y = f6;
        float f7 = (fArr[15] * 1.0f) + (f4 * fArr[11]) + (f3 * fArr[7]) + (f2 * fArr[3]);
        float f8 = ((f5 / f7) + 1.0f) * 0.5f;
        vector32.x = f8;
        float f9 = ((f6 / f7) + 1.0f) * 0.5f;
        vector32.y = f9;
        vector32.x = f8 * viewWidth;
        float f10 = viewHeight;
        float f11 = f9 * f10;
        vector32.y = f11;
        vector32.y = f10 - f11;
        return vector32;
    }

    private void setPerspective(float f2, float f3, float f4, float f5, float f6, float f7) {
        float[] fArr = this.projectionMatrix.data;
        if (f2 != f3 && f4 != f5 && f6 > StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD && f7 > f6) {
            float f8 = 1.0f / (f3 - f2);
            float f9 = 1.0f / (f5 - f4);
            float f10 = 1.0f / (f7 - f6);
            float f11 = 2.0f * f6;
            fArr[0] = f11 * f8;
            fArr[1] = 0.0f;
            fArr[2] = 0.0f;
            fArr[3] = 0.0f;
            fArr[4] = 0.0f;
            fArr[5] = f11 * f9;
            fArr[6] = 0.0f;
            fArr[7] = 0.0f;
            fArr[8] = (f3 + f2) * f8;
            fArr[9] = (f5 + f4) * f9;
            fArr[10] = (-(f7 + f6)) * f10;
            fArr[11] = -1.0f;
            fArr[12] = 0.0f;
            fArr[13] = 0.0f;
            fArr[14] = (-2.0f) * f7 * f6 * f10;
            fArr[15] = 0.0f;
            this.nearPlane = f6;
            this.farPlane = f7;
            this.areMatricesInitialized = true;
            return;
        }
        throw new IllegalArgumentException("Invalid parameters to setPerspective, valid values:  width != height, bottom != top, near > 0.0f, far > near");
    }

    public Camera(Scene scene) {
        Preconditions.checkNotNull(scene, "Parameter \"scene\" was null.");
        super.setParent(scene);
        boolean z = scene.getView() instanceof ArSceneView;
        this.isArCamera = z;
        if (z) {
            return;
        }
        scene.getView().addOnLayoutChangeListener(new View.OnLayoutChangeListener() { // from class: c.d.b.a.f
            @Override // android.view.View.OnLayoutChangeListener
            public final void onLayoutChange(View view, int i, int i2, int i3, int i4, int i5, int i6, int i7, int i8) {
                Camera.this.a(view, i, i2, i3, i4, i5, i6, i7, i8);
            }
        });
    }
}