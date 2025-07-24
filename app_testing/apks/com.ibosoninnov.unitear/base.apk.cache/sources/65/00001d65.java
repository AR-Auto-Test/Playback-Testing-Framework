package com.google.ar.sceneform.ux;

import android.view.MotionEvent;
import com.google.ar.sceneform.HitTestResult;
import com.google.ar.sceneform.ux.BaseGestureRecognizer;

/* loaded from: classes.dex */
public class PinchGestureRecognizer extends BaseGestureRecognizer<PinchGesture> {

    /* loaded from: classes.dex */
    public interface OnGestureStartedListener extends BaseGestureRecognizer.OnGestureStartedListener<PinchGesture> {
    }

    public PinchGestureRecognizer(GesturePointersUtility gesturePointersUtility) {
        super(gesturePointersUtility);
    }

    @Override // com.google.ar.sceneform.ux.BaseGestureRecognizer
    public void tryCreateGestures(HitTestResult hitTestResult, MotionEvent motionEvent) {
        if (motionEvent.getPointerCount() < 2) {
            return;
        }
        int pointerId = motionEvent.getPointerId(motionEvent.getActionIndex());
        int actionMasked = motionEvent.getActionMasked();
        if (!(actionMasked == 0 || actionMasked == 5) || this.gesturePointersUtility.isPointerIdRetained(pointerId)) {
            return;
        }
        for (int i = 0; i < motionEvent.getPointerCount(); i++) {
            int pointerId2 = motionEvent.getPointerId(i);
            if (pointerId2 != pointerId && !this.gesturePointersUtility.isPointerIdRetained(pointerId2)) {
                this.gestures.add(new PinchGesture(this.gesturePointersUtility, motionEvent, pointerId2, hitTestResult));
            }
        }
    }
}