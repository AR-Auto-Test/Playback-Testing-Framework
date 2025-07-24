package com.google.ar.sceneform.ux;

import android.view.MotionEvent;
import com.google.ar.sceneform.HitTestResult;
import com.google.ar.sceneform.ux.BaseGesture;
import java.util.ArrayList;

/* loaded from: classes.dex */
public abstract class BaseGestureRecognizer<T extends BaseGesture<T>> {
    public final GesturePointersUtility gesturePointersUtility;
    public final ArrayList<T> gestures = new ArrayList<>();
    private final ArrayList<OnGestureStartedListener<T>> gestureStartedListeners = new ArrayList<>();

    /* loaded from: classes.dex */
    public interface OnGestureStartedListener<T extends BaseGesture<T>> {
        void onGestureStarted(T t);
    }

    public BaseGestureRecognizer(GesturePointersUtility gesturePointersUtility) {
        this.gesturePointersUtility = gesturePointersUtility;
    }

    private void dispatchGestureStarted(T t) {
        for (int i = 0; i < this.gestureStartedListeners.size(); i++) {
            this.gestureStartedListeners.get(i).onGestureStarted(t);
        }
    }

    private void removeFinishedGestures() {
        for (int size = this.gestures.size() - 1; size >= 0; size--) {
            if (this.gestures.get(size).hasFinished()) {
                this.gestures.remove(size);
            }
        }
    }

    public void addOnGestureStartedListener(OnGestureStartedListener<T> onGestureStartedListener) {
        if (this.gestureStartedListeners.contains(onGestureStartedListener)) {
            return;
        }
        this.gestureStartedListeners.add(onGestureStartedListener);
    }

    public void onTouch(HitTestResult hitTestResult, MotionEvent motionEvent) {
        tryCreateGestures(hitTestResult, motionEvent);
        for (int i = 0; i < this.gestures.size(); i++) {
            T t = this.gestures.get(i);
            t.onTouch(hitTestResult, motionEvent);
            if (t.justStarted()) {
                dispatchGestureStarted(t);
            }
        }
        removeFinishedGestures();
    }

    public void removeOnGestureStartedListener(OnGestureStartedListener<T> onGestureStartedListener) {
        this.gestureStartedListeners.remove(onGestureStartedListener);
    }

    public abstract void tryCreateGestures(HitTestResult hitTestResult, MotionEvent motionEvent);
}