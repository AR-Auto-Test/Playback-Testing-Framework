package com.google.ar.sceneform.ux;

import android.view.MotionEvent;
import com.google.ar.sceneform.HitTestResult;
import com.google.ar.sceneform.Node;
import java.util.ArrayList;

/* loaded from: classes.dex */
public abstract class BaseTransformableNode extends Node implements Node.OnTapListener, Node.OnAccurateTapListner {
    private final ArrayList<BaseTransformationController<?>> controllers = new ArrayList<>();
    private final TransformationSystem transformationSystem;

    public BaseTransformableNode(TransformationSystem transformationSystem) {
        this.transformationSystem = transformationSystem;
        setOnTapListener(this);
        setOnAccurateTapListner(this);
    }

    public void addTransformationController(BaseTransformationController<?> baseTransformationController) {
        this.controllers.add(baseTransformationController);
    }

    public TransformationSystem getTransformationSystem() {
        return this.transformationSystem;
    }

    public boolean isSelected() {
        return this.transformationSystem.getSelectedNode() == this;
    }

    public boolean isTransforming() {
        for (int i = 0; i < this.controllers.size(); i++) {
            if (this.controllers.get(i).isTransforming()) {
                return true;
            }
        }
        return false;
    }

    @Override // com.google.ar.sceneform.Node.OnAccurateTapListner
    public void onAccurateTap(HitTestResult hitTestResult, MotionEvent motionEvent) {
        select();
    }

    @Override // com.google.ar.sceneform.Node.OnTapListener
    public void onTap(HitTestResult hitTestResult, MotionEvent motionEvent) {
        select();
    }

    public void removeTransformationController(BaseTransformationController<?> baseTransformationController) {
        this.controllers.remove(baseTransformationController);
    }

    public boolean select() {
        return this.transformationSystem.selectNode(this);
    }
}