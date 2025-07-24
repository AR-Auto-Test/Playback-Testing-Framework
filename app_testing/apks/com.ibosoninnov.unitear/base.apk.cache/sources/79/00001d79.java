package com.google.ar.sceneform.ux;

import com.google.android.material.internal.StaticLayoutBuilderCompat;
import com.google.ar.sceneform.FrameTime;
import com.google.ar.sceneform.Node;
import com.google.ar.sceneform.math.MathHelper;
import com.google.ar.sceneform.math.Vector3;

/* loaded from: classes.dex */
public class ScaleController extends BaseTransformationController<PinchGesture> {
    public static final float DEFAULT_ELASTICITY = 0.15f;
    public static final float DEFAULT_MAX_SCALE = 1.75f;
    public static final float DEFAULT_MIN_SCALE = 0.75f;
    public static final float DEFAULT_SENSITIVITY = 0.75f;
    private static final float ELASTIC_RATIO_LIMIT = 0.8f;
    private static final float LERP_SPEED = 8.0f;
    private float currentScaleRatio;
    private float elasticity;
    private float maxScale;
    private float minScale;
    private float sensitivity;

    public ScaleController(BaseTransformableNode baseTransformableNode, PinchGestureRecognizer pinchGestureRecognizer) {
        super(baseTransformableNode, pinchGestureRecognizer);
        this.minScale = 0.75f;
        this.maxScale = 1.75f;
        this.sensitivity = 0.75f;
        this.elasticity = 0.15f;
    }

    private float getClampedScaleRatio() {
        return Math.min(1.0f, Math.max((float) StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, this.currentScaleRatio));
    }

    private float getElasticDelta() {
        float f2 = this.currentScaleRatio;
        if (f2 > 1.0f) {
            f2 -= 1.0f;
        } else if (f2 >= StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
            return StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        }
        return Math.signum(f2) * (1.0f - (1.0f / ((Math.abs(f2) * this.elasticity) + 1.0f)));
    }

    private float getFinalScale() {
        float clampedScaleRatio = getClampedScaleRatio() + getElasticDelta();
        return (clampedScaleRatio * getScaleDelta()) + this.minScale;
    }

    private float getScaleDelta() {
        float f2 = this.maxScale - this.minScale;
        if (f2 > StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
            return f2;
        }
        throw new IllegalStateException("maxScale must be greater than minScale.");
    }

    public float getElasticity() {
        return this.elasticity;
    }

    public float getMaxScale() {
        return this.maxScale;
    }

    public float getMinScale() {
        return this.minScale;
    }

    public float getSensitivity() {
        return this.sensitivity;
    }

    @Override // com.google.ar.sceneform.ux.BaseTransformationController, com.google.ar.sceneform.Node.LifecycleListener
    public void onActivated(Node node) {
        super.onActivated(node);
        this.currentScaleRatio = (getTransformableNode().getLocalScale().x - this.minScale) / getScaleDelta();
    }

    /* JADX DEBUG: Method merged with bridge method */
    @Override // com.google.ar.sceneform.ux.BaseTransformationController
    public void onEndTransformation(PinchGesture pinchGesture) {
    }

    @Override // com.google.ar.sceneform.ux.BaseTransformationController, com.google.ar.sceneform.Node.LifecycleListener
    public void onUpdated(Node node, FrameTime frameTime) {
        if (isTransforming() || !isEnabled()) {
            return;
        }
        this.currentScaleRatio = MathHelper.lerp(this.currentScaleRatio, getClampedScaleRatio(), MathHelper.clamp(frameTime.getDeltaSeconds() * LERP_SPEED, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 1.0f));
        float finalScale = getFinalScale();
        getTransformableNode().setLocalScale(new Vector3(finalScale, finalScale, finalScale));
    }

    public void setElasticity(float f2) {
        this.elasticity = f2;
    }

    public void setMaxScale(float f2) {
        this.maxScale = f2;
    }

    public void setMinScale(float f2) {
        this.minScale = f2;
    }

    public void setSensitivity(float f2) {
        this.sensitivity = f2;
    }

    /* JADX DEBUG: Method merged with bridge method */
    @Override // com.google.ar.sceneform.ux.BaseTransformationController
    public boolean canStartTransformation(PinchGesture pinchGesture) {
        Node targetNode = pinchGesture.getTargetNode();
        if (targetNode == null) {
            return false;
        }
        BaseTransformableNode transformableNode = getTransformableNode();
        if (targetNode == transformableNode || targetNode.isDescendantOf(transformableNode)) {
            return transformableNode.isSelected() || transformableNode.select();
        }
        return false;
    }

    /* JADX DEBUG: Method merged with bridge method */
    @Override // com.google.ar.sceneform.ux.BaseTransformationController
    public void onContinueTransformation(PinchGesture pinchGesture) {
        if (getTransformableNode().getScene() == null) {
            return;
        }
        this.currentScaleRatio = (pinchGesture.gapDeltaInches() * this.sensitivity) + this.currentScaleRatio;
        float finalScale = getFinalScale();
        getTransformableNode().setLocalScale(new Vector3(finalScale, finalScale, finalScale));
        float f2 = this.currentScaleRatio;
        if (f2 < -0.8f || f2 > 1.8f) {
            pinchGesture.cancel();
        }
    }
}