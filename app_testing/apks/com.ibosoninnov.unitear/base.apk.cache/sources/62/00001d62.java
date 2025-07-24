package com.google.ar.sceneform.ux;

import android.app.Activity;
import android.content.Context;
import android.util.AttributeSet;
import android.widget.FrameLayout;
import b.b.h.n;

/* loaded from: classes.dex */
public class HandMotionView extends n {
    private static final long ANIMATION_SPEED_MS = 2500;
    private HandMotionAnimation animation;

    public HandMotionView(Context context) {
        super(context);
    }

    @Override // android.widget.ImageView, android.view.View
    public void onAttachedToWindow() {
        super.onAttachedToWindow();
        clearAnimation();
        HandMotionAnimation handMotionAnimation = new HandMotionAnimation((FrameLayout) ((Activity) getContext()).findViewById(R.id.sceneform_hand_layout), this);
        this.animation = handMotionAnimation;
        handMotionAnimation.setRepeatCount(-1);
        this.animation.setDuration(ANIMATION_SPEED_MS);
        this.animation.setStartOffset(1000L);
        startAnimation(this.animation);
    }

    public HandMotionView(Context context, AttributeSet attributeSet) {
        super(context, attributeSet);
    }
}