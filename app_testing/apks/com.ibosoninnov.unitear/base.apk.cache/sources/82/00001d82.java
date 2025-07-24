package com.google.ar.sceneform.ux;

import android.view.MotionEvent;
import com.google.ar.sceneform.HitTestResult;
import com.google.ar.sceneform.math.Vector3;
import com.google.ar.sceneform.ux.BaseGesture;

/* loaded from: classes.dex */
public class TwistGesture extends BaseGesture<TwistGesture> {
    private static final float SLOP_ROTATION_DEGREES = 10.0f;
    private static final String TAG = "TwistGesture";
    private static final boolean TWIST_GESTURE_DEBUG = false;
    private float deltaRotationDegrees;
    private final int pointerId1;
    private final int pointerId2;
    private final Vector3 previousPosition1;
    private final Vector3 previousPosition2;
    private final Vector3 startPosition1;
    private final Vector3 startPosition2;

    /* loaded from: classes.dex */
    public interface OnGestureEventListener extends BaseGesture.OnGestureEventListener<TwistGesture> {
    }

    public TwistGesture(GesturePointersUtility gesturePointersUtility, MotionEvent motionEvent, int i, HitTestResult hitTestResult) {
        super(gesturePointersUtility);
        int pointerId = motionEvent.getPointerId(motionEvent.getActionIndex());
        this.pointerId1 = pointerId;
        this.pointerId2 = i;
        Vector3 motionEventToPosition = GesturePointersUtility.motionEventToPosition(motionEvent, pointerId);
        this.startPosition1 = motionEventToPosition;
        Vector3 motionEventToPosition2 = GesturePointersUtility.motionEventToPosition(motionEvent, i);
        this.startPosition2 = motionEventToPosition2;
        this.previousPosition1 = new Vector3(motionEventToPosition);
        this.previousPosition2 = new Vector3(motionEventToPosition2);
        this.targetNode = hitTestResult.getNode();
        debugLog("Created");
    }

    private static float calculateDeltaRotation(Vector3 vector3, Vector3 vector32, Vector3 vector33, Vector3 vector34) {
        Vector3 normalized = Vector3.subtract(vector3, vector32).normalized();
        Vector3 normalized2 = Vector3.subtract(vector33, vector34).normalized();
        return Vector3.angleBetweenVectors(normalized, normalized2) * Math.signum((normalized2.x * normalized.y) - (normalized2.y * normalized.x));
    }

    private static void debugLog(String str) {
    }

    @Override // com.google.ar.sceneform.ux.BaseGesture
    public boolean canStart(HitTestResult hitTestResult, MotionEvent motionEvent) {
        if (!this.gesturePointersUtility.isPointerIdRetained(this.pointerId1) && !this.gesturePointersUtility.isPointerIdRetained(this.pointerId2)) {
            int pointerId = motionEvent.getPointerId(motionEvent.getActionIndex());
            int actionMasked = motionEvent.getActionMasked();
            if (actionMasked == 3) {
                cancel();
                return false;
            }
            if ((actionMasked == 1 || actionMasked == 6) && (pointerId == this.pointerId1 || pointerId == this.pointerId2)) {
                cancel();
                return false;
            } else if (actionMasked != 2) {
                return false;
            } else {
                Vector3 motionEventToPosition = GesturePointersUtility.motionEventToPosition(motionEvent, this.pointerId1);
                Vector3 motionEventToPosition2 = GesturePointersUtility.motionEventToPosition(motionEvent, this.pointerId2);
                Vector3 subtract = Vector3.subtract(motionEventToPosition, this.previousPosition1);
                Vector3 subtract2 = Vector3.subtract(motionEventToPosition2, this.previousPosition2);
                this.previousPosition1.set(motionEventToPosition);
                this.previousPosition2.set(motionEventToPosition2);
                return (Vector3.equals(subtract, Vector3.zero()) || Vector3.equals(subtract2, Vector3.zero()) || Math.abs(calculateDeltaRotation(motionEventToPosition, motionEventToPosition2, this.startPosition1, this.startPosition2)) < SLOP_ROTATION_DEGREES) ? false : true;
            }
        }
        cancel();
        return false;
    }

    public float getDeltaRotationDegrees() {
        return this.deltaRotationDegrees;
    }

    /* JADX DEBUG: Method merged with bridge method */
    @Override // com.google.ar.sceneform.ux.BaseGesture
    public TwistGesture getSelf() {
        return this;
    }

    @Override // com.google.ar.sceneform.ux.BaseGesture
    public void onCancel() {
        debugLog("Cancelled");
    }

    @Override // com.google.ar.sceneform.ux.BaseGesture
    public void onFinish() {
        debugLog("Finished");
        this.gesturePointersUtility.releasePointerId(this.pointerId1);
        this.gesturePointersUtility.releasePointerId(this.pointerId2);
    }

    @Override // com.google.ar.sceneform.ux.BaseGesture
    public void onStart(HitTestResult hitTestResult, MotionEvent motionEvent) {
        debugLog("Started");
        this.gesturePointersUtility.retainPointerId(this.pointerId1);
        this.gesturePointersUtility.retainPointerId(this.pointerId2);
    }

    @Override // com.google.ar.sceneform.ux.BaseGesture
    public boolean updateGesture(HitTestResult hitTestResult, MotionEvent motionEvent) {
        int pointerId = motionEvent.getPointerId(motionEvent.getActionIndex());
        int actionMasked = motionEvent.getActionMasked();
        if (actionMasked == 3) {
            cancel();
            return false;
        }
        if ((actionMasked == 1 || actionMasked == 6) && (pointerId == this.pointerId1 || pointerId == this.pointerId2)) {
            complete();
            return false;
        } else if (actionMasked != 2) {
            return false;
        } else {
            Vector3 motionEventToPosition = GesturePointersUtility.motionEventToPosition(motionEvent, this.pointerId1);
            Vector3 motionEventToPosition2 = GesturePointersUtility.motionEventToPosition(motionEvent, this.pointerId2);
            this.deltaRotationDegrees = calculateDeltaRotation(motionEventToPosition, motionEventToPosition2, this.previousPosition1, this.previousPosition2);
            this.previousPosition1.set(motionEventToPosition);
            this.previousPosition2.set(motionEventToPosition2);
            debugLog("Update: " + this.deltaRotationDegrees);
            return true;
        }
    }
}