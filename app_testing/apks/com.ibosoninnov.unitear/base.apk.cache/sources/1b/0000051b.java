package b.q.b;

import android.animation.Animator;
import android.animation.AnimatorListenerAdapter;
import android.view.View;
import android.view.ViewGroup;
import androidx.fragment.app.Fragment;
import b.q.b.f0;
import b.q.b.q;

/* compiled from: FragmentAnim.java */
/* loaded from: classes.dex */
public final class g extends AnimatorListenerAdapter {

    /* renamed from: a  reason: collision with root package name */
    public final /* synthetic */ ViewGroup f2450a;

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ View f2451b;

    /* renamed from: c  reason: collision with root package name */
    public final /* synthetic */ Fragment f2452c;

    /* renamed from: d  reason: collision with root package name */
    public final /* synthetic */ f0.a f2453d;

    /* renamed from: e  reason: collision with root package name */
    public final /* synthetic */ b.j.f.b f2454e;

    public g(ViewGroup viewGroup, View view, Fragment fragment, f0.a aVar, b.j.f.b bVar) {
        this.f2450a = viewGroup;
        this.f2451b = view;
        this.f2452c = fragment;
        this.f2453d = aVar;
        this.f2454e = bVar;
    }

    @Override // android.animation.AnimatorListenerAdapter, android.animation.Animator.AnimatorListener
    public void onAnimationEnd(Animator animator) {
        this.f2450a.endViewTransition(this.f2451b);
        Animator animator2 = this.f2452c.getAnimator();
        this.f2452c.setAnimator(null);
        if (animator2 == null || this.f2450a.indexOfChild(this.f2451b) >= 0) {
            return;
        }
        ((q.b) this.f2453d).a(this.f2452c, this.f2454e);
    }
}