package b.q.b;

import android.animation.Animator;
import android.animation.AnimatorListenerAdapter;
import android.view.View;
import android.view.ViewGroup;
import androidx.fragment.app.Fragment;

/* compiled from: FragmentManager.java */
/* loaded from: classes.dex */
public class r extends AnimatorListenerAdapter {

    /* renamed from: a  reason: collision with root package name */
    public final /* synthetic */ ViewGroup f2515a;

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ View f2516b;

    /* renamed from: c  reason: collision with root package name */
    public final /* synthetic */ Fragment f2517c;

    public r(q qVar, ViewGroup viewGroup, View view, Fragment fragment) {
        this.f2515a = viewGroup;
        this.f2516b = view;
        this.f2517c = fragment;
    }

    @Override // android.animation.AnimatorListenerAdapter, android.animation.Animator.AnimatorListener
    public void onAnimationEnd(Animator animator) {
        this.f2515a.endViewTransition(this.f2516b);
        animator.removeListener(this);
        Fragment fragment = this.f2517c;
        View view = fragment.mView;
        if (view == null || !fragment.mHidden) {
            return;
        }
        view.setVisibility(8);
    }
}