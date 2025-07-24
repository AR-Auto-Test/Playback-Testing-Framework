package com.google.ar.sceneform.ux;

import com.google.android.material.internal.StaticLayoutBuilderCompat;
import com.google.ar.core.Anchor;
import com.google.ar.core.Frame;
import com.google.ar.core.HitResult;
import com.google.ar.core.Plane;
import com.google.ar.core.Pose;
import com.google.ar.core.Trackable;
import com.google.ar.core.TrackingState;
import com.google.ar.sceneform.AnchorNode;
import com.google.ar.sceneform.ArSceneView;
import com.google.ar.sceneform.FrameTime;
import com.google.ar.sceneform.Node;
import com.google.ar.sceneform.Scene;
import com.google.ar.sceneform.math.MathHelper;
import com.google.ar.sceneform.math.Quaternion;
import com.google.ar.sceneform.math.Vector3;
import com.google.ar.sceneform.utilities.Preconditions;
import java.util.EnumSet;
import java.util.List;

/* loaded from: classes.dex */
public class TranslationController extends BaseTransformationController<DragGesture> {
    private static final float LERP_SPEED = 12.0f;
    private static final float POSITION_LENGTH_THRESHOLD = 0.01f;
    private static final float ROTATION_DOT_THRESHOLD = 0.99f;
    private EnumSet<Plane.Type> allowedPlaneTypes;
    private Vector3 desiredLocalPosition;
    private Quaternion desiredLocalRotation;
    private final Vector3 initialForwardInLocal;
    private HitResult lastArHitResult;

    public TranslationController(BaseTransformableNode baseTransformableNode, DragGestureRecognizer dragGestureRecognizer) {
        super(baseTransformableNode, dragGestureRecognizer);
        this.initialForwardInLocal = new Vector3();
        this.allowedPlaneTypes = EnumSet.allOf(Plane.Type.class);
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

    public EnumSet<Plane.Type> getAllowedPlaneTypes() {
        return this.allowedPlaneTypes;
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

    public void setAllowedPlaneTypes(EnumSet<Plane.Type> enumSet) {
        this.allowedPlaneTypes = enumSet;
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
                    return true;
                }
                this.initialForwardInLocal.set(forward);
                return true;
            }
            return false;
        }
        return false;
    }

    /* JADX DEBUG: Method merged with bridge method */
    @Override // com.google.ar.sceneform.ux.BaseTransformationController
    public void onContinueTransformation(DragGesture dragGesture) {
        Frame arFrame;
        Vector3 vector3;
        Scene scene = getTransformableNode().getScene();
        if (scene == null || (arFrame = ((ArSceneView) scene.getView()).getArFrame()) == null || arFrame.getCamera().getTrackingState() != TrackingState.TRACKING) {
            return;
        }
        Vector3 position = dragGesture.getPosition();
        List<HitResult> hitTest = arFrame.hitTest(position.x, position.y);
        for (int i = 0; i < hitTest.size(); i++) {
            HitResult hitResult = hitTest.get(i);
            Trackable trackable = hitResult.getTrackable();
            Pose hitPose = hitResult.getHitPose();
            if (trackable instanceof Plane) {
                Plane plane = (Plane) trackable;
                if (plane.isPoseInPolygon(hitPose) && this.allowedPlaneTypes.contains(plane.getType())) {
                    this.desiredLocalPosition = new Vector3(hitPose.tx(), hitPose.ty(), hitPose.tz());
                    this.desiredLocalRotation = new Quaternion(hitPose.qx(), hitPose.qy(), hitPose.qz(), hitPose.qw());
                    Node parent = getTransformableNode().getParent();
                    if (parent != null && (vector3 = this.desiredLocalPosition) != null && this.desiredLocalRotation != null) {
                        this.desiredLocalPosition = parent.worldToLocalPoint(vector3);
                        this.desiredLocalRotation = Quaternion.multiply(parent.getWorldRotation().inverted(), (Quaternion) Preconditions.checkNotNull(this.desiredLocalRotation));
                    }
                    this.desiredLocalRotation = calculateFinalDesiredLocalRotation((Quaternion) Preconditions.checkNotNull(this.desiredLocalRotation));
                    this.lastArHitResult = hitResult;
                    return;
                }
            }
        }
    }

    /* JADX DEBUG: Method merged with bridge method */
    @Override // com.google.ar.sceneform.ux.BaseTransformationController
    public void onEndTransformation(DragGesture dragGesture) {
        Quaternion quaternion;
        HitResult hitResult = this.lastArHitResult;
        if (hitResult == null) {
            return;
        }
        if (hitResult.getTrackable().getTrackingState() == TrackingState.TRACKING) {
            AnchorNode anchorNodeOrDie = getAnchorNodeOrDie();
            Anchor anchor = anchorNodeOrDie.getAnchor();
            if (anchor != null) {
                anchor.detach();
            }
            Anchor createAnchor = hitResult.createAnchor();
            Vector3 worldPosition = getTransformableNode().getWorldPosition();
            Quaternion worldRotation = getTransformableNode().getWorldRotation();
            Quaternion quaternion2 = this.desiredLocalRotation;
            if (quaternion2 != null) {
                getTransformableNode().setLocalRotation(quaternion2);
                quaternion = getTransformableNode().getWorldRotation();
            } else {
                quaternion = worldRotation;
            }
            anchorNodeOrDie.setAnchor(createAnchor);
            getTransformableNode().setWorldRotation(quaternion);
            this.initialForwardInLocal.set(anchorNodeOrDie.worldToLocalDirection(getTransformableNode().getForward()));
            getTransformableNode().setWorldRotation(worldRotation);
            getTransformableNode().setWorldPosition(worldPosition);
        }
        this.desiredLocalPosition = Vector3.zero();
        this.desiredLocalRotation = calculateFinalDesiredLocalRotation(Quaternion.identity());
    }
}