package com.google.ar.sceneform.ux;

import android.view.View;
import android.view.animation.Animation;
import android.view.animation.Transformation;

/* loaded from: classes.dex */
public class HandMotionAnimation extends Animation {
    private static final float HALF_PI = 1.5707964f;
    private static final float TWO_PI = 6.2831855f;
    private final View containerView;
    private final View handImageView;

    public HandMotionAnimation(View view, View view2) {
        this.handImageView = view2;
        this.containerView = view;
    }

    @Override // android.view.animation.Animation
    public void applyTransformation(float f2, Transformation transformation) {
        float f3 = (f2 * TWO_PI) + HALF_PI;
        float f4 = this.handImageView.getResources().getDisplayMetrics().density * 25.0f;
        double d2 = f3;
        float cos = f4 * 2.0f * ((float) Math.cos(d2));
        float width = ((this.containerView.getWidth() / 2.0f) + cos) - (this.handImageView.getWidth() / 2.0f);
        this.handImageView.setX(width);
        this.handImageView.setY(((this.containerView.getHeight() / 2.0f) + (f4 * ((float) Math.sin(d2)))) - (this.handImageView.getHeight() / 2.0f));
        this.handImageView.invalidate();
    }
}