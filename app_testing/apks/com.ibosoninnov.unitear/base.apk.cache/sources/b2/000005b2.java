package b.w.b;

import android.animation.Animator;
import android.animation.AnimatorListenerAdapter;
import android.view.View;
import android.view.ViewPropertyAnimator;
import androidx.recyclerview.widget.RecyclerView;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import java.util.Objects;

/* compiled from: DefaultItemAnimator.java */
/* loaded from: classes.dex */
public class h extends AnimatorListenerAdapter {

    /* renamed from: a  reason: collision with root package name */
    public final /* synthetic */ RecyclerView.d0 f2731a;

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ int f2732b;

    /* renamed from: c  reason: collision with root package name */
    public final /* synthetic */ View f2733c;

    /* renamed from: d  reason: collision with root package name */
    public final /* synthetic */ int f2734d;

    /* renamed from: e  reason: collision with root package name */
    public final /* synthetic */ ViewPropertyAnimator f2735e;

    /* renamed from: f  reason: collision with root package name */
    public final /* synthetic */ k f2736f;

    public h(k kVar, RecyclerView.d0 d0Var, int i, View view, int i2, ViewPropertyAnimator viewPropertyAnimator) {
        this.f2736f = kVar;
        this.f2731a = d0Var;
        this.f2732b = i;
        this.f2733c = view;
        this.f2734d = i2;
        this.f2735e = viewPropertyAnimator;
    }

    @Override // android.animation.AnimatorListenerAdapter, android.animation.Animator.AnimatorListener
    public void onAnimationCancel(Animator animator) {
        if (this.f2732b != 0) {
            this.f2733c.setTranslationX(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
        }
        if (this.f2734d != 0) {
            this.f2733c.setTranslationY(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
        }
    }

    @Override // android.animation.AnimatorListenerAdapter, android.animation.Animator.AnimatorListener
    public void onAnimationEnd(Animator animator) {
        this.f2735e.setListener(null);
        this.f2736f.c(this.f2731a);
        this.f2736f.q.remove(this.f2731a);
        this.f2736f.k();
    }

    @Override // android.animation.AnimatorListenerAdapter, android.animation.Animator.AnimatorListener
    public void onAnimationStart(Animator animator) {
        Objects.requireNonNull(this.f2736f);
    }
}