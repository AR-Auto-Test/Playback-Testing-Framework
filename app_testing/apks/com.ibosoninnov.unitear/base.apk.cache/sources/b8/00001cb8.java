package com.google.ar.sceneform.math;

import android.animation.TypeEvaluator;

/* loaded from: classes.dex */
public class Vector3Evaluator implements TypeEvaluator<Vector3> {
    /* JADX DEBUG: Method merged with bridge method */
    @Override // android.animation.TypeEvaluator
    public Vector3 evaluate(float f2, Vector3 vector3, Vector3 vector32) {
        return Vector3.lerp(vector3, vector32, f2);
    }
}