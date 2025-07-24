package b.w.b;

import android.animation.Animator;
import android.animation.AnimatorListenerAdapter;
import android.view.View;
import android.view.ViewPropertyAnimator;
import androidx.recyclerview.widget.RecyclerView;
import java.util.Objects;

/* compiled from: DefaultItemAnimator.java */
/* loaded from: classes.dex */
public class g extends AnimatorListenerAdapter {

    /* renamed from: a  reason: collision with root package name */
    public final /* synthetic */ RecyclerView.d0 f2727a;

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ View f2728b;

    /* renamed from: c  reason: collision with root package name */
    public final /* synthetic */ ViewPropertyAnimator f2729c;

    /* renamed from: d  reason: collision with root package name */
    public final /* synthetic */ k f2730d;

    public g(k kVar, RecyclerView.d0 d0Var, View view, ViewPropertyAnimator viewPropertyAnimator) {
        this.f2730d = kVar;
        this.f2727a = d0Var;
        this.f2728b = view;
        this.f2729c = viewPropertyAnimator;
    }

    @Override // android.animation.AnimatorListenerAdapter, android.animation.Animator.AnimatorListener
    public void onAnimationCancel(Animator animator) {
        this.f2728b.setAlpha(1.0f);
    }

    @Override // android.animation.AnimatorListenerAdapter, android.animation.Animator.AnimatorListener
    public void onAnimationEnd(Animator animator) {
        this.f2729c.setListener(null);
        this.f2730d.c(this.f2727a);
        this.f2730d.p.remove(this.f2727a);
        this.f2730d.k();
    }

    @Override // android.animation.AnimatorListenerAdapter, android.animation.Animator.AnimatorListener
    public void onAnimationStart(Animator animator) {
        Objects.requireNonNull(this.f2730d);
    }
}