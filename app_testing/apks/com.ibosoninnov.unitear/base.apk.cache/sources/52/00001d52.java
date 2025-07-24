package com.google.ar.sceneform.ux;

import android.view.MotionEvent;
import com.google.ar.sceneform.HitTestResult;
import com.google.ar.sceneform.Node;
import com.google.ar.sceneform.ux.BaseGesture;

/* loaded from: classes.dex */
public abstract class BaseGesture<T extends BaseGesture<T>> {
    private OnGestureEventListener<T> eventListener;
    public final GesturePointersUtility gesturePointersUtility;
    private boolean hasFinished;
    private boolean hasStarted;
    private boolean justStarted;
    public Node targetNode;
    private boolean wasCancelled;

    /* loaded from: classes.dex */
    public interface OnGestureEventListener<T extends BaseGesture<T>> {
        void onFinished(T t);

        void onUpdated(T t);
    }

    public BaseGesture(GesturePointersUtility gesturePointersUtility) {
        this.gesturePointersUtility = gesturePointersUtility;
    }

    private void dispatchFinishedEvent() {
        OnGestureEventListener<T> onGestureEventListener = this.eventListener;
        if (onGestureEventListener != null) {
            onGestureEventListener.onFinished(getSelf());
        }
    }

    private void dispatchUpdateEvent() {
        OnGestureEventListener<T> onGestureEventListener = this.eventListener;
        if (onGestureEventListener != null) {
            onGestureEventListener.onUpdated(getSelf());
        }
    }

    private void start(HitTestResult hitTestResult, MotionEvent motionEvent) {
        this.hasStarted = true;
        this.justStarted = true;
        onStart(hitTestResult, motionEvent);
    }

    public abstract boolean canStart(HitTestResult hitTestResult, MotionEvent motionEvent);

    public void cancel() {
        this.wasCancelled = true;
        onCancel();
        complete();
    }

    public void complete() {
        this.hasFinished = true;
        if (this.hasStarted) {
            onFinish();
            dispatchFinishedEvent();
        }
    }

    public abstract T getSelf();

    public Node getTargetNode() {
        return this.targetNode;
    }

    public boolean hasFinished() {
        return this.hasFinished;
    }

    public boolean hasStarted() {
        return this.hasStarted;
    }

    public float inchesToPixels(float f2) {
        return this.gesturePointersUtility.inchesToPixels(f2);
    }

    public boolean justStarted() {
        return this.justStarted;
    }

    public abstract void onCancel();

    public abstract void onFinish();

    public abstract void onStart(HitTestResult hitTestResult, MotionEvent motionEvent);

    public void onTouch(HitTestResult hitTestResult, MotionEvent motionEvent) {
        if (!this.hasStarted && canStart(hitTestResult, motionEvent)) {
            start(hitTestResult, motionEvent);
            return;
        }
        this.justStarted = false;
        if (this.hasStarted && updateGesture(hitTestResult, motionEvent)) {
            dispatchUpdateEvent();
        }
    }

    public float pixelsToInches(float f2) {
        return this.gesturePointersUtility.pixelsToInches(f2);
    }

    public void setGestureEventListener(OnGestureEventListener<T> onGestureEventListener) {
        this.eventListener = onGestureEventListener;
    }

    public abstract boolean updateGesture(HitTestResult hitTestResult, MotionEvent motionEvent);

    public boolean wasCancelled() {
        return this.wasCancelled;
    }
}