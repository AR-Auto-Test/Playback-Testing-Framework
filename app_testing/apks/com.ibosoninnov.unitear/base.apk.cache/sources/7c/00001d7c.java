package com.google.ar.sceneform.ux;

import com.google.ar.sceneform.Node;
import com.google.ar.sceneform.math.Quaternion;
import com.google.ar.sceneform.math.Vector3;

/* loaded from: classes.dex */
public class SimpleRotationController extends BaseTransformationController<TwistGesture> {
    private float rotationRateDegrees;

    public SimpleRotationController(BaseTransformableNode baseTransformableNode, TwistGestureRecognizer twistGestureRecognizer) {
        super(baseTransformableNode, twistGestureRecognizer);
        this.rotationRateDegrees = 2.5f;
    }

    public float getRotationRateDegrees() {
        return this.rotationRateDegrees;
    }

    /* JADX DEBUG: Method merged with bridge method */
    @Override // com.google.ar.sceneform.ux.BaseTransformationController
    public void onEndTransformation(TwistGesture twistGesture) {
    }

    public void setRotationRateDegrees(float f2) {
        this.rotationRateDegrees = f2;
    }

    /* JADX DEBUG: Method merged with bridge method */
    @Override // com.google.ar.sceneform.ux.BaseTransformationController
    public boolean canStartTransformation(TwistGesture twistGesture) {
        Node targetNode = twistGesture.getTargetNode();
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
    public void onContinueTransformation(TwistGesture twistGesture) {
        if (getTransformableNode().getScene() == null) {
            return;
        }
        getTransformableNode().setLocalRotation(Quaternion.multiply(getTransformableNode().getLocalRotation(), new Quaternion(Vector3.back(), (-twistGesture.getDeltaRotationDegrees()) * this.rotationRateDegrees)));
    }
}