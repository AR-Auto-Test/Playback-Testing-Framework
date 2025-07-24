package com.google.ar.core;

import android.animation.Animator;
import android.animation.AnimatorListenerAdapter;

/* compiled from: InstallActivity.java */
/* renamed from: com.google.ar.core.r  reason: case insensitive filesystem */
/* loaded from: classes.dex */
public final class C0130r extends AnimatorListenerAdapter {

    /* renamed from: a  reason: collision with root package name */
    public final /* synthetic */ InstallActivity f5600a;

    public C0130r(InstallActivity installActivity) {
        this.f5600a = installActivity;
    }

    @Override // android.animation.AnimatorListenerAdapter, android.animation.Animator.AnimatorListener
    public final void onAnimationEnd(Animator animator) {
        this.f5600a.showSpinner();
    }
}