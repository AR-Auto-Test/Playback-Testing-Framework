package com.google.ar.sceneform.ux;

import android.util.DisplayMetrics;
import android.view.MotionEvent;
import com.google.ar.sceneform.HitTestResult;
import java.util.ArrayList;

/* loaded from: classes.dex */
public class TransformationSystem {
    private final DragGestureRecognizer dragGestureRecognizer;
    private final GesturePointersUtility gesturePointersUtility;
    private final PinchGestureRecognizer pinchGestureRecognizer;
    private final ArrayList<BaseGestureRecognizer<?>> recognizers = new ArrayList<>();
    private BaseTransformableNode selectedNode;
    private SelectionVisualizer selectionVisualizer;
    private final TwistGestureRecognizer twistGestureRecognizer;

    public TransformationSystem(DisplayMetrics displayMetrics, SelectionVisualizer selectionVisualizer) {
        this.selectionVisualizer = selectionVisualizer;
        GesturePointersUtility gesturePointersUtility = new GesturePointersUtility(displayMetrics);
        this.gesturePointersUtility = gesturePointersUtility;
        DragGestureRecognizer dragGestureRecognizer = new DragGestureRecognizer(gesturePointersUtility);
        this.dragGestureRecognizer = dragGestureRecognizer;
        addGestureRecognizer(dragGestureRecognizer);
        PinchGestureRecognizer pinchGestureRecognizer = new PinchGestureRecognizer(gesturePointersUtility);
        this.pinchGestureRecognizer = pinchGestureRecognizer;
        addGestureRecognizer(pinchGestureRecognizer);
        TwistGestureRecognizer twistGestureRecognizer = new TwistGestureRecognizer(gesturePointersUtility);
        this.twistGestureRecognizer = twistGestureRecognizer;
        addGestureRecognizer(twistGestureRecognizer);
    }

    private boolean deselectNode() {
        BaseTransformableNode baseTransformableNode = this.selectedNode;
        if (baseTransformableNode == null) {
            return true;
        }
        if (baseTransformableNode.isTransforming()) {
            return false;
        }
        this.selectionVisualizer.removeSelectionVisual(this.selectedNode);
        this.selectedNode = null;
        return true;
    }

    public void addGestureRecognizer(BaseGestureRecognizer<?> baseGestureRecognizer) {
        this.recognizers.add(baseGestureRecognizer);
    }

    public DragGestureRecognizer getDragRecognizer() {
        return this.dragGestureRecognizer;
    }

    public GesturePointersUtility getGesturePointersUtility() {
        return this.gesturePointersUtility;
    }

    public PinchGestureRecognizer getPinchRecognizer() {
        return this.pinchGestureRecognizer;
    }

    public BaseTransformableNode getSelectedNode() {
        return this.selectedNode;
    }

    public SelectionVisualizer getSelectionVisualizer() {
        return this.selectionVisualizer;
    }

    public TwistGestureRecognizer getTwistRecognizer() {
        return this.twistGestureRecognizer;
    }

    public void onTouch(HitTestResult hitTestResult, MotionEvent motionEvent) {
        for (int i = 0; i < this.recognizers.size(); i++) {
            this.recognizers.get(i).onTouch(hitTestResult, motionEvent);
        }
    }

    public boolean selectNode(BaseTransformableNode baseTransformableNode) {
        if (deselectNode()) {
            if (baseTransformableNode != null) {
                this.selectedNode = baseTransformableNode;
                this.selectionVisualizer.applySelectionVisual(baseTransformableNode);
                return true;
            }
            return true;
        }
        return false;
    }

    public void setSelectionVisualizer(SelectionVisualizer selectionVisualizer) {
        BaseTransformableNode baseTransformableNode = this.selectedNode;
        if (baseTransformableNode != null) {
            this.selectionVisualizer.removeSelectionVisual(baseTransformableNode);
        }
        this.selectionVisualizer = selectionVisualizer;
        BaseTransformableNode baseTransformableNode2 = this.selectedNode;
        if (baseTransformableNode2 != null) {
            selectionVisualizer.applySelectionVisual(baseTransformableNode2);
        }
    }
}