package com.google.ar.sceneform.ux;

import android.view.MotionEvent;
import c.b.a.a.a;
import com.google.ar.sceneform.HitTestResult;
import com.google.ar.sceneform.math.Vector3;
import com.google.ar.sceneform.ux.BaseGesture;

/* loaded from: classes.dex */
public class DragGesture extends BaseGesture<DragGesture> {
    private static final boolean DRAG_GESTURE_DEBUG = false;
    private static final float SLOP_INCHES = 0.1f;
    private static final String TAG = "DragGesture";
    private final Vector3 delta;
    private final int pointerId;
    private final Vector3 position;
    private final Vector3 startPosition;

    /* loaded from: classes.dex */
    public interface OnGestureEventListener extends BaseGesture.OnGestureEventListener<DragGesture> {
    }

    public DragGesture(GesturePointersUtility gesturePointersUtility, HitTestResult hitTestResult, MotionEvent motionEvent) {
        super(gesturePointersUtility);
        int pointerId = motionEvent.getPointerId(motionEvent.getActionIndex());
        this.pointerId = pointerId;
        Vector3 motionEventToPosition = GesturePointersUtility.motionEventToPosition(motionEvent, pointerId);
        this.startPosition = motionEventToPosition;
        this.position = new Vector3(motionEventToPosition);
        this.delta = Vector3.zero();
        this.targetNode = hitTestResult.getNode();
        debugLog(a.j("Created: ", pointerId));
    }

    private static void debugLog(String str) {
    }

    @Override // com.google.ar.sceneform.ux.BaseGesture
    public boolean canStart(HitTestResult hitTestResult, MotionEvent motionEvent) {
        int pointerId = motionEvent.getPointerId(motionEvent.getActionIndex());
        int actionMasked = motionEvent.getActionMasked();
        if (this.gesturePointersUtility.isPointerIdRetained(this.pointerId)) {
            cancel();
            return false;
        } else if (pointerId == this.pointerId && (actionMasked == 1 || actionMasked == 6)) {
            cancel();
            return false;
        } else if (actionMasked == 3) {
            cancel();
            return false;
        } else if (actionMasked != 2) {
            return false;
        } else {
            if (motionEvent.getPointerCount() > 1) {
                for (int i = 0; i < motionEvent.getPointerCount(); i++) {
                    int pointerId2 = motionEvent.getPointerId(i);
                    if (pointerId2 != this.pointerId && !this.gesturePointersUtility.isPointerIdRetained(pointerId2)) {
                        return false;
                    }
                }
            }
            return Vector3.subtract(GesturePointersUtility.motionEventToPosition(motionEvent, this.pointerId), this.startPosition).length() >= this.gesturePointersUtility.inchesToPixels(SLOP_INCHES);
        }
    }

    public Vector3 getDelta() {
        return new Vector3(this.delta);
    }

    public Vector3 getPosition() {
        return new Vector3(this.position);
    }

    /* JADX DEBUG: Method merged with bridge method */
    @Override // com.google.ar.sceneform.ux.BaseGesture
    public DragGesture getSelf() {
        return this;
    }

    @Override // com.google.ar.sceneform.ux.BaseGesture
    public void onCancel() {
        StringBuilder x = a.x("Cancelled: ");
        x.append(this.pointerId);
        debugLog(x.toString());
    }

    @Override // com.google.ar.sceneform.ux.BaseGesture
    public void onFinish() {
        StringBuilder x = a.x("Finished: ");
        x.append(this.pointerId);
        debugLog(x.toString());
        this.gesturePointersUtility.releasePointerId(this.pointerId);
    }

    @Override // com.google.ar.sceneform.ux.BaseGesture
    public void onStart(HitTestResult hitTestResult, MotionEvent motionEvent) {
        StringBuilder x = a.x("Started: ");
        x.append(this.pointerId);
        debugLog(x.toString());
        this.position.set(GesturePointersUtility.motionEventToPosition(motionEvent, this.pointerId));
        this.gesturePointersUtility.retainPointerId(this.pointerId);
    }

    @Override // com.google.ar.sceneform.ux.BaseGesture
    public boolean updateGesture(HitTestResult hitTestResult, MotionEvent motionEvent) {
        int pointerId = motionEvent.getPointerId(motionEvent.getActionIndex());
        int actionMasked = motionEvent.getActionMasked();
        if (actionMasked == 2) {
            Vector3 motionEventToPosition = GesturePointersUtility.motionEventToPosition(motionEvent, this.pointerId);
            if (Vector3.equals(motionEventToPosition, this.position)) {
                return false;
            }
            this.delta.set(Vector3.subtract(motionEventToPosition, this.position));
            this.position.set(motionEventToPosition);
            debugLog("Updated: " + this.pointerId + " : " + this.position);
            return true;
        } else if (pointerId == this.pointerId && (actionMasked == 1 || actionMasked == 6)) {
            complete();
            return false;
        } else if (actionMasked == 3) {
            cancel();
            return false;
        } else {
            return false;
        }
    }
}