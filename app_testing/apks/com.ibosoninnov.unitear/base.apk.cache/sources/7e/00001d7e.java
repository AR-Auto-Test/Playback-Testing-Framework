package com.google.ar.sceneform.ux;

import com.google.android.material.internal.StaticLayoutBuilderCompat;
import com.google.ar.sceneform.AnchorNode;
import com.google.ar.sceneform.Camera;
import com.google.ar.sceneform.FrameTime;
import com.google.ar.sceneform.Node;
import com.google.ar.sceneform.Scene;
import com.google.ar.sceneform.math.MathHelper;
import com.google.ar.sceneform.math.Quaternion;
import com.google.ar.sceneform.math.Vector3;
import com.google.ar.sceneform.utilities.Preconditions;

/* loaded from: classes.dex */
public class SimpleTranslationController extends BaseTransformationController<DragGesture> {
    private static final float LERP_SPEED = 12.0f;
    private static final float POSITION_LENGTH_THRESHOLD = 0.01f;
    private static final float ROTATION_DOT_THRESHOLD = 0.99f;
    private float DRAW_DISTANCE;
    private Vector3 desiredLocalPosition;
    private Quaternion desiredLocalRotation;
    private final Vector3 initialForwardInLocal;
    private Vector3 lastArHitResult;

    public SimpleTranslationController(BaseTransformableNode baseTransformableNode, DragGestureRecognizer dragGestureRecognizer) {
        super(baseTransformableNode, dragGestureRecognizer);
        this.DRAW_DISTANCE = 0.3f;
        this.initialForwardInLocal = new Vector3();
    }

    private Quaternion calculateFinalDesiredLocalRotation(Quaternion quaternion) {
        return Quaternion.multiply(Quaternion.rotationBetweenVectors(Vector3.up(), Quaternion.rotateVector(quaternion, Vector3.up())), Quaternion.rotationBetweenVectors(Vector3.forward(), this.initialForwardInLocal)).normalized();
    }

    private static float dotQuaternion(Quaternion quaternion, Quaternion quaternion2) {
        float f2 = (quaternion.y * quaternion2.y) + (quaternion.x * quaternion2.x);
        return (quaternion.w * quaternion2.w) + (quaternion.z * quaternion2.z) + f2;
    }

    private AnchorNode getAnchorNodeOrDie() {
        Node parent = getTransformableNode().getParent();
        if (parent instanceof AnchorNode) {
            return (AnchorNode) parent;
        }
        throw new IllegalStateException("TransformableNode must have an AnchorNode as a parent.");
    }

    private void updatePosition(FrameTime frameTime) {
        Vector3 vector3 = this.desiredLocalPosition;
        if (vector3 == null) {
            return;
        }
        Vector3 lerp = Vector3.lerp(getTransformableNode().getLocalPosition(), vector3, MathHelper.clamp(frameTime.getDeltaSeconds() * LERP_SPEED, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 1.0f));
        if (Math.abs(Vector3.subtract(vector3, lerp).length()) <= POSITION_LENGTH_THRESHOLD) {
            this.desiredLocalPosition = null;
        } else {
            vector3 = lerp;
        }
        getTransformableNode().setLocalPosition(vector3);
    }

    private void updateRotation(FrameTime frameTime) {
        Quaternion quaternion = this.desiredLocalRotation;
        if (quaternion == null) {
            return;
        }
        Quaternion slerp = Quaternion.slerp(getTransformableNode().getLocalRotation(), quaternion, MathHelper.clamp(frameTime.getDeltaSeconds() * LERP_SPEED, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 1.0f));
        if (Math.abs(dotQuaternion(slerp, quaternion)) >= ROTATION_DOT_THRESHOLD) {
            this.desiredLocalRotation = null;
        } else {
            quaternion = slerp;
        }
        getTransformableNode().setLocalRotation(quaternion);
    }

    @Override // com.google.ar.sceneform.ux.BaseTransformationController
    public boolean isTransforming() {
        return (!super.isTransforming() && this.desiredLocalRotation == null && this.desiredLocalPosition == null) ? false : true;
    }

    @Override // com.google.ar.sceneform.ux.BaseTransformationController, com.google.ar.sceneform.Node.LifecycleListener
    public void onUpdated(Node node, FrameTime frameTime) {
        updatePosition(frameTime);
        updateRotation(frameTime);
    }

    /* JADX DEBUG: Method merged with bridge method */
    @Override // com.google.ar.sceneform.ux.BaseTransformationController
    public boolean canStartTransformation(DragGesture dragGesture) {
        Node targetNode = dragGesture.getTargetNode();
        if (targetNode == null) {
            return false;
        }
        BaseTransformableNode transformableNode = getTransformableNode();
        if (targetNode == transformableNode || targetNode.isDescendantOf(transformableNode)) {
            if (transformableNode.isSelected() || transformableNode.select()) {
                Vector3 forward = transformableNode.getForward();
                Node parent = transformableNode.getParent();
                if (parent != null) {
                    this.initialForwardInLocal.set(parent.worldToLocalDirection(forward));
                } else {
                    this.initialForwardInLocal.set(forward);
                }
                Scene scene = getTransformableNode().getScene();
                if (scene == null) {
                    return false;
                }
                this.DRAW_DISTANCE = Vector3.subtract(scene.getCamera().getWorldPosition(), getTransformableNode().getWorldPosition()).length();
                return true;
            }
            return false;
        }
        return false;
    }

    /* JADX DEBUG: Method merged with bridge method */
    @Override // com.google.ar.sceneform.ux.BaseTransformationController
    public void onContinueTransformation(DragGesture dragGesture) {
        Vector3 vector3;
        Scene scene = getTransformableNode().getScene();
        if (scene == null) {
            return;
        }
        Camera camera = scene.getCamera();
        Vector3 position = dragGesture.getPosition();
        Vector3 point = camera.screenPointToRay(position.x, position.y).getPoint(this.DRAW_DISTANCE);
        this.desiredLocalPosition = new Vector3(point.x, point.y, point.z);
        Node parent = getTransformableNode().getParent();
        Quaternion worldRotation = getTransformableNode().getWorldRotation();
        this.desiredLocalRotation = worldRotation;
        if (parent != null && (vector3 = this.desiredLocalPosition) != null && worldRotation != null) {
            this.desiredLocalPosition = parent.worldToLocalPoint(vector3);
            this.desiredLocalRotation = Quaternion.multiply(parent.getWorldRotation().inverted(), (Quaternion) Preconditions.checkNotNull(this.desiredLocalRotation));
        }
        this.lastArHitResult = point;
    }

    /* JADX DEBUG: Method merged with bridge method */
    @Override // com.google.ar.sceneform.ux.BaseTransformationController
    public void onEndTransformation(DragGesture dragGesture) {
        Quaternion quaternion;
        Vector3 worldPosition = getTransformableNode().getWorldPosition();
        Quaternion worldRotation = getTransformableNode().getWorldRotation();
        Quaternion quaternion2 = this.desiredLocalRotation;
        if (quaternion2 != null) {
            getTransformableNode().setLocalRotation(quaternion2);
            quaternion = getTransformableNode().getWorldRotation();
        } else {
            quaternion = worldRotation;
        }
        getTransformableNode().setWorldRotation(quaternion);
        this.initialForwardInLocal.set(getTransformableNode().getParent().worldToLocalDirection(getTransformableNode().getForward()));
        getTransformableNode().setWorldRotation(worldRotation);
        getTransformableNode().setWorldPosition(worldPosition);
        calculateFinalDesiredLocalRotation(Quaternion.identity());
    }
}