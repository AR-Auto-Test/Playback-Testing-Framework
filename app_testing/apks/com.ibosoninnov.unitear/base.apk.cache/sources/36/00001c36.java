package com.google.ar.sceneform;

import android.util.Log;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import com.google.ar.core.Anchor;
import com.google.ar.core.Pose;
import com.google.ar.core.TrackingState;
import com.google.ar.sceneform.math.MathHelper;
import com.google.ar.sceneform.math.Quaternion;
import com.google.ar.sceneform.math.Vector3;
import java.util.List;

/* loaded from: classes.dex */
public class AnchorNode extends Node {
    private static final float SMOOTH_FACTOR = 12.0f;
    private static final String TAG = "AnchorNode";
    private Anchor anchor;
    private boolean isSmoothed = true;
    private boolean wasTracking;

    public AnchorNode() {
    }

    private void setChildrenEnabled(boolean z) {
        List<Node> children = getChildren();
        for (int i = 0; i < children.size(); i++) {
            children.get(i).setEnabled(z);
        }
    }

    private void updateTrackedPose(float f2, boolean z) {
        boolean isTracking = isTracking();
        if (isTracking != this.wasTracking) {
            setChildrenEnabled(isTracking || this.anchor == null);
        }
        Anchor anchor = this.anchor;
        if (anchor != null && isTracking) {
            Pose pose = anchor.getPose();
            Vector3 extractPositionFromPose = ArHelpers.extractPositionFromPose(pose);
            Quaternion extractRotationFromPose = ArHelpers.extractRotationFromPose(pose);
            if (this.isSmoothed && !z) {
                Vector3 worldPosition = getWorldPosition();
                float clamp = MathHelper.clamp(f2 * SMOOTH_FACTOR, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 1.0f);
                worldPosition.set(Vector3.lerp(worldPosition, extractPositionFromPose, clamp));
                super.setWorldPosition(worldPosition);
                super.setWorldRotation(Quaternion.slerp(getWorldRotation(), extractRotationFromPose, clamp));
            } else {
                super.setWorldPosition(extractPositionFromPose);
                super.setWorldRotation(extractRotationFromPose);
            }
            this.wasTracking = isTracking;
            return;
        }
        this.wasTracking = isTracking;
    }

    public Anchor getAnchor() {
        return this.anchor;
    }

    public boolean isSmoothed() {
        return this.isSmoothed;
    }

    public boolean isTracking() {
        Anchor anchor = this.anchor;
        return anchor != null && anchor.getTrackingState() == TrackingState.TRACKING;
    }

    @Override // com.google.ar.sceneform.Node
    public void onUpdate(FrameTime frameTime) {
        updateTrackedPose(frameTime.getDeltaSeconds(), false);
    }

    public void setAnchor(Anchor anchor) {
        this.anchor = anchor;
        boolean z = true;
        if (anchor != null) {
            updateTrackedPose(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, true);
        }
        boolean isTracking = isTracking();
        this.wasTracking = isTracking;
        if (!isTracking && anchor != null) {
            z = false;
        }
        setChildrenEnabled(z);
    }

    @Override // com.google.ar.sceneform.Node
    public void setLocalPosition(Vector3 vector3) {
        if (this.anchor != null) {
            Log.w(TAG, "Cannot call setLocalPosition on AnchorNode while it is anchored.");
        } else {
            super.setLocalPosition(vector3);
        }
    }

    @Override // com.google.ar.sceneform.Node
    public void setLocalRotation(Quaternion quaternion) {
        if (this.anchor != null) {
            Log.w(TAG, "Cannot call setLocalRotation on AnchorNode while it is anchored.");
        } else {
            super.setLocalRotation(quaternion);
        }
    }

    public void setSmoothed(boolean z) {
        this.isSmoothed = z;
    }

    @Override // com.google.ar.sceneform.Node
    public void setWorldPosition(Vector3 vector3) {
        if (this.anchor != null) {
            Log.w(TAG, "Cannot call setWorldPosition on AnchorNode while it is anchored.");
        } else {
            super.setWorldPosition(vector3);
        }
    }

    @Override // com.google.ar.sceneform.Node
    public void setWorldRotation(Quaternion quaternion) {
        if (this.anchor != null) {
            Log.w(TAG, "Cannot call setWorldRotation on AnchorNode while it is anchored.");
        } else {
            super.setWorldRotation(quaternion);
        }
    }

    public AnchorNode(Anchor anchor) {
        setAnchor(anchor);
    }
}