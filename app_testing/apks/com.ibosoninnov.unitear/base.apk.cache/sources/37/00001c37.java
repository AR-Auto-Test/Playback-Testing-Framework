package com.google.ar.sceneform;

import com.google.ar.core.Pose;
import com.google.ar.sceneform.math.Quaternion;
import com.google.ar.sceneform.math.Vector3;

/* loaded from: classes.dex */
public class ArHelpers {
    public static Vector3 extractPositionFromPose(Pose pose) {
        return new Vector3(pose.tx(), pose.ty(), pose.tz());
    }

    public static Quaternion extractRotationFromPose(Pose pose) {
        return new Quaternion(pose.qx(), pose.qy(), pose.qz(), pose.qw());
    }
}