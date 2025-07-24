package com.google.ar.sceneform.ux;

import com.google.ar.sceneform.Node;
import com.google.ar.sceneform.rendering.ModelRenderable;

/* loaded from: classes.dex */
public class FootprintSelectionVisualizer implements SelectionVisualizer {
    private final Node footprintNode = new Node();
    private ModelRenderable footprintRenderable;

    @Override // com.google.ar.sceneform.ux.SelectionVisualizer
    public void applySelectionVisual(BaseTransformableNode baseTransformableNode) {
        this.footprintNode.setParent(baseTransformableNode);
    }

    public ModelRenderable getFootprintRenderable() {
        return this.footprintRenderable;
    }

    @Override // com.google.ar.sceneform.ux.SelectionVisualizer
    public void removeSelectionVisual(BaseTransformableNode baseTransformableNode) {
        this.footprintNode.setParent(null);
    }

    public void setFootprintRenderable(ModelRenderable modelRenderable) {
        ModelRenderable makeCopy = modelRenderable.makeCopy();
        this.footprintNode.setRenderable(makeCopy);
        makeCopy.setCollisionShape(null);
        this.footprintRenderable = makeCopy;
    }
}