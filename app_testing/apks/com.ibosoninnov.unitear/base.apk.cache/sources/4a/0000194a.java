package com.google.android.material.transition;

import android.animation.Animator;
import android.animation.ValueAnimator;
import android.view.View;
import android.view.ViewGroup;
import b.z.p;
import b.z.z;
import com.google.android.material.internal.StaticLayoutBuilderCompat;

/* loaded from: classes.dex */
public final class Hold extends z {
    @Override // b.z.z
    public Animator onAppear(ViewGroup viewGroup, View view, p pVar, p pVar2) {
        return ValueAnimator.ofFloat(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
    }

    @Override // b.z.z
    public Animator onDisappear(ViewGroup viewGroup, View view, p pVar, p pVar2) {
        return ValueAnimator.ofFloat(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
    }
}