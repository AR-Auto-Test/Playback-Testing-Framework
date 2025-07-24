package b.w.b;

import android.animation.Animator;
import android.animation.AnimatorListenerAdapter;
import android.view.View;
import android.view.ViewPropertyAnimator;
import androidx.recyclerview.widget.RecyclerView;
import java.util.Objects;

/* compiled from: DefaultItemAnimator.java */
/* loaded from: classes.dex */
public class f extends AnimatorListenerAdapter {

    /* renamed from: a  reason: collision with root package name */
    public final /* synthetic */ RecyclerView.d0 f2723a;

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ ViewPropertyAnimator f2724b;

    /* renamed from: c  reason: collision with root package name */
    public final /* synthetic */ View f2725c;

    /* renamed from: d  reason: collision with root package name */
    public final /* synthetic */ k f2726d;

    public f(k kVar, RecyclerView.d0 d0Var, ViewPropertyAnimator viewPropertyAnimator, View view) {
        this.f2726d = kVar;
        this.f2723a = d0Var;
        this.f2724b = viewPropertyAnimator;
        this.f2725c = view;
    }

    @Override // android.animation.AnimatorListenerAdapter, android.animation.Animator.AnimatorListener
    public void onAnimationEnd(Animator animator) {
        this.f2724b.setListener(null);
        this.f2725c.setAlpha(1.0f);
        this.f2726d.c(this.f2723a);
        this.f2726d.r.remove(this.f2723a);
        this.f2726d.k();
    }

    @Override // android.animation.AnimatorListenerAdapter, android.animation.Animator.AnimatorListener
    public void onAnimationStart(Animator animator) {
        Objects.requireNonNull(this.f2726d);
    }
}