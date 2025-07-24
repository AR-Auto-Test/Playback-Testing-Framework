package com.google.ar.sceneform.ux;

import com.google.ar.sceneform.FrameTime;
import com.google.ar.sceneform.Node;
import com.google.ar.sceneform.ux.BaseGesture;
import com.google.ar.sceneform.ux.BaseGestureRecognizer;

/* loaded from: classes.dex */
public abstract class BaseTransformationController<T extends BaseGesture<T>> implements BaseGestureRecognizer.OnGestureStartedListener<T>, BaseGesture.OnGestureEventListener<T>, Node.LifecycleListener {
    private boolean activeAndEnabled;
    private T activeGesture;
    private boolean enabled;
    private final BaseGestureRecognizer<T> gestureRecognizer;
    private final BaseTransformableNode transformableNode;

    public BaseTransformationController(BaseTransformableNode baseTransformableNode, BaseGestureRecognizer<T> baseGestureRecognizer) {
        this.transformableNode = baseTransformableNode;
        baseTransformableNode.addLifecycleListener(this);
        this.gestureRecognizer = baseGestureRecognizer;
        setEnabled(true);
    }

    private void connectToRecognizer() {
        this.gestureRecognizer.addOnGestureStartedListener(this);
    }

    private void disconnectFromRecognizer() {
        this.gestureRecognizer.removeOnGestureStartedListener(this);
    }

    private void setActiveGesture(T t) {
        T t2 = this.activeGesture;
        if (t2 != null) {
            t2.setGestureEventListener(null);
        }
        this.activeGesture = t;
        if (t != null) {
            t.setGestureEventListener(this);
        }
    }

    private void updateActiveAndEnabled() {
        boolean z = getTransformableNode().isActive() && this.enabled;
        if (z == this.activeAndEnabled) {
            return;
        }
        this.activeAndEnabled = z;
        if (z) {
            connectToRecognizer();
            return;
        }
        disconnectFromRecognizer();
        T t = this.activeGesture;
        if (t != null) {
            t.cancel();
        }
    }

    public abstract boolean canStartTransformation(T t);

    public T getActiveGesture() {
        return this.activeGesture;
    }

    public BaseTransformableNode getTransformableNode() {
        return this.transformableNode;
    }

    public boolean isEnabled() {
        return this.enabled;
    }

    public boolean isTransforming() {
        return this.activeGesture != null;
    }

    @Override // com.google.ar.sceneform.Node.LifecycleListener
    public void onActivated(Node node) {
        updateActiveAndEnabled();
    }

    public abstract void onContinueTransformation(T t);

    @Override // com.google.ar.sceneform.Node.LifecycleListener
    public void onDeactivated(Node node) {
        updateActiveAndEnabled();
    }

    public abstract void onEndTransformation(T t);

    @Override // com.google.ar.sceneform.ux.BaseGesture.OnGestureEventListener
    public void onFinished(T t) {
        onEndTransformation(t);
        setActiveGesture(null);
    }

    @Override // com.google.ar.sceneform.ux.BaseGestureRecognizer.OnGestureStartedListener
    public void onGestureStarted(T t) {
        if (!isTransforming() && canStartTransformation(t)) {
            setActiveGesture(t);
        }
    }

    @Override // com.google.ar.sceneform.Node.LifecycleListener
    public void onUpdated(Node node, FrameTime frameTime) {
    }

    @Override // com.google.ar.sceneform.ux.BaseGesture.OnGestureEventListener
    public void onUpdated(T t) {
        onContinueTransformation(t);
    }

    public void setEnabled(boolean z) {
        this.enabled = z;
        updateActiveAndEnabled();
    }
}