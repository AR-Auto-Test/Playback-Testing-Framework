package com.google.ar.sceneform.ux;

import com.google.ar.sceneform.math.Quaternion;
import com.google.ar.sceneform.math.Vector3;

/* loaded from: classes.dex */
public class RotationController extends BaseTransformationController<TwistGesture> {
    private float rotationRateDegrees;

    public RotationController(BaseTransformableNode baseTransformableNode, TwistGestureRecognizer twistGestureRecognizer) {
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
        return getTransformableNode().isSelected();
    }

    /* JADX DEBUG: Method merged with bridge method */
    @Override // com.google.ar.sceneform.ux.BaseTransformationController
    public void onContinueTransformation(TwistGesture twistGesture) {
        getTransformableNode().setLocalRotation(Quaternion.multiply(getTransformableNode().getLocalRotation(), new Quaternion(Vector3.up(), (-twistGesture.getDeltaRotationDegrees()) * this.rotationRateDegrees)));
    }
}