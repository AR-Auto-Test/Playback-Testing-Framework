package com.google.ar.sceneform.math;

import android.animation.TypeEvaluator;

/* loaded from: classes.dex */
public class QuaternionEvaluator implements TypeEvaluator<Quaternion> {
    /* JADX DEBUG: Method merged with bridge method */
    @Override // android.animation.TypeEvaluator
    public Quaternion evaluate(float f2, Quaternion quaternion, Quaternion quaternion2) {
        return Quaternion.slerp(quaternion, quaternion2, f2);
    }
}