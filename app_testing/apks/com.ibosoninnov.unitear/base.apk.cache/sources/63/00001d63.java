package com.google.ar.sceneform.ux;

import android.view.MotionEvent;
import c.b.a.a.a;
import com.google.ar.sceneform.HitTestResult;
import com.google.ar.sceneform.math.Vector3;
import com.google.ar.sceneform.ux.BaseGesture;

/* loaded from: classes.dex */
public class PinchGesture extends BaseGesture<PinchGesture> {
    private static final boolean PINCH_GESTURE_DEBUG = false;
    private static final float SLOP_INCHES = 0.05f;
    private static final float SLOP_MOTION_DIRECTION_DEGREES = 30.0f;
    private static final String TAG = "PinchGesture";
    private float gap;
    private float gapDelta;
    private final int pointerId1;
    private final int pointerId2;
    private final Vector3 previousPosition1;
    private final Vector3 previousPosition2;
    private final Vector3 startPosition1;
    private final Vector3 startPosition2;

    /* loaded from: classes.dex */
    public interface OnGestureEventListener extends BaseGesture.OnGestureEventListener<PinchGesture> {
    }

    public PinchGesture(GesturePointersUtility gesturePointersUtility, MotionEvent motionEvent, int i, HitTestResult hitTestResult) {
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
                Vector3 subtract = Vector3.subtract(this.startPosition1, this.startPosition2);
                Vector3 normalized = subtract.normalized();
                Vector3 motionEventToPosition = GesturePointersUtility.motionEventToPosition(motionEvent, this.pointerId1);
                Vector3 motionEventToPosition2 = GesturePointersUtility.motionEventToPosition(motionEvent, this.pointerId2);
                Vector3 subtract2 = Vector3.subtract(motionEventToPosition, this.previousPosition1);
                Vector3 subtract3 = Vector3.subtract(motionEventToPosition2, this.previousPosition2);
                this.previousPosition1.set(motionEventToPosition);
                this.previousPosition2.set(motionEventToPosition2);
                float dot = Vector3.dot(subtract2.normalized(), normalized.negated());
                float dot2 = Vector3.dot(subtract3.normalized(), normalized);
                float cos = (float) Math.cos(Math.toRadians(30.0d));
                if (Vector3.equals(subtract2, Vector3.zero()) || Math.abs(dot) >= cos) {
                    if (Vector3.equals(subtract3, Vector3.zero()) || Math.abs(dot2) >= cos) {
                        float length = subtract.length();
                        float length2 = Vector3.subtract(motionEventToPosition, motionEventToPosition2).length();
                        this.gap = length2;
                        return Math.abs(length2 - length) >= this.gesturePointersUtility.inchesToPixels(SLOP_INCHES);
                    }
                    return false;
                }
                return false;
            }
        }
        cancel();
        return false;
    }

    @Override // com.google.ar.sceneform.ux.BaseGesture
    public void cancel() {
        super.cancel();
    }

    public float gapDeltaInches() {
        return this.gesturePointersUtility.pixelsToInches(getGapDelta());
    }

    public float gapInches() {
        return this.gesturePointersUtility.pixelsToInches(getGap());
    }

    public float getGap() {
        return this.gap;
    }

    public float getGapDelta() {
        return this.gapDelta;
    }

    /* JADX DEBUG: Method merged with bridge method */
    @Override // com.google.ar.sceneform.ux.BaseGesture
    public PinchGesture getSelf() {
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
            float length = Vector3.subtract(GesturePointersUtility.motionEventToPosition(motionEvent, this.pointerId1), GesturePointersUtility.motionEventToPosition(motionEvent, this.pointerId2)).length();
            float f2 = this.gap;
            if (length == f2) {
                return false;
            }
            this.gapDelta = length - f2;
            this.gap = length;
            StringBuilder x = a.x("Update: ");
            x.append(this.gapDelta);
            debugLog(x.toString());
            return true;
        }
    }
}