package com.google.ar.sceneform.ux;

/* loaded from: classes.dex */
public class SimpleTransformableNode extends BaseTransformableNode {
    private final SimpleRotationController rotationController;
    private final ScaleController scaleController;
    private final SimpleTranslationController translationController;

    public SimpleTransformableNode(TransformationSystem transformationSystem) {
        super(transformationSystem);
        SimpleTranslationController simpleTranslationController = new SimpleTranslationController(this, transformationSystem.getDragRecognizer());
        this.translationController = simpleTranslationController;
        addTransformationController(simpleTranslationController);
        ScaleController scaleController = new ScaleController(this, transformationSystem.getPinchRecognizer());
        this.scaleController = scaleController;
        addTransformationController(scaleController);
        SimpleRotationController simpleRotationController = new SimpleRotationController(this, transformationSystem.getTwistRecognizer());
        this.rotationController = simpleRotationController;
        addTransformationController(simpleRotationController);
    }

    public SimpleRotationController getRotationController() {
        return this.rotationController;
    }

    public ScaleController getScaleController() {
        return this.scaleController;
    }

    public SimpleTranslationController getTranslationController() {
        return this.translationController;
    }
}