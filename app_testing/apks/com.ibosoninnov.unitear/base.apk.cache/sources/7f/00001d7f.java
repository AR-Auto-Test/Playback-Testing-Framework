package com.google.ar.sceneform.ux;

/* loaded from: classes.dex */
public class TransformableNode extends BaseTransformableNode {
    private final RotationController rotationController;
    private final ScaleController scaleController;
    private final TranslationController translationController;

    public TransformableNode(TransformationSystem transformationSystem) {
        super(transformationSystem);
        TranslationController translationController = new TranslationController(this, transformationSystem.getDragRecognizer());
        this.translationController = translationController;
        addTransformationController(translationController);
        ScaleController scaleController = new ScaleController(this, transformationSystem.getPinchRecognizer());
        this.scaleController = scaleController;
        addTransformationController(scaleController);
        RotationController rotationController = new RotationController(this, transformationSystem.getTwistRecognizer());
        this.rotationController = rotationController;
        addTransformationController(rotationController);
    }

    public RotationController getRotationController() {
        return this.rotationController;
    }

    public ScaleController getScaleController() {
        return this.scaleController;
    }

    public TranslationController getTranslationController() {
        return this.translationController;
    }
}