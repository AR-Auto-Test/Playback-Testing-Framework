package com.google.ar.core;

import android.animation.ValueAnimator;

/* compiled from: InstallActivity.java */
/* loaded from: classes.dex */
public final class q implements ValueAnimator.AnimatorUpdateListener {

    /* renamed from: a  reason: collision with root package name */
    public final /* synthetic */ int f5596a;

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ int f5597b;

    /* renamed from: c  reason: collision with root package name */
    public final /* synthetic */ int f5598c;

    /* renamed from: d  reason: collision with root package name */
    public final /* synthetic */ InstallActivity f5599d;

    public q(InstallActivity installActivity, int i, int i2, int i3) {
        this.f5599d = installActivity;
        this.f5596a = i;
        this.f5597b = i2;
        this.f5598c = i3;
    }

    @Override // android.animation.ValueAnimator.AnimatorUpdateListener
    public final void onAnimationUpdate(ValueAnimator valueAnimator) {
        float animatedFraction = 1.0f - valueAnimator.getAnimatedFraction();
        float animatedFraction2 = valueAnimator.getAnimatedFraction();
        int i = this.f5596a;
        float f2 = this.f5597b * animatedFraction2;
        this.f5599d.getWindow().setLayout((int) ((i * animatedFraction) + f2), (int) ((this.f5598c * animatedFraction) + f2));
        this.f5599d.getWindow().getDecorView().refreshDrawableState();
    }
}