package com.google.ar.sceneform.ux;

import android.view.MotionEvent;
import com.google.ar.sceneform.HitTestResult;
import com.google.ar.sceneform.ux.BaseGestureRecognizer;

/* loaded from: classes.dex */
public class DragGestureRecognizer extends BaseGestureRecognizer<DragGesture> {

    /* loaded from: classes.dex */
    public interface OnGestureStartedListener extends BaseGestureRecognizer.OnGestureStartedListener<DragGesture> {
    }

    public DragGestureRecognizer(GesturePointersUtility gesturePointersUtility) {
        super(gesturePointersUtility);
    }

    @Override // com.google.ar.sceneform.ux.BaseGestureRecognizer
    public void tryCreateGestures(HitTestResult hitTestResult, MotionEvent motionEvent) {
        int actionMasked = motionEvent.getActionMasked();
        int pointerId = motionEvent.getPointerId(motionEvent.getActionIndex());
        if (!(actionMasked == 0 || actionMasked == 5) || this.gesturePointersUtility.isPointerIdRetained(pointerId)) {
            return;
        }
        this.gestures.add(new DragGesture(this.gesturePointersUtility, hitTestResult, motionEvent));
    }
}