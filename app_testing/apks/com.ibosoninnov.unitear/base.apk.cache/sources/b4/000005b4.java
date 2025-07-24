package b.w.b;

import android.animation.Animator;
import android.animation.AnimatorListenerAdapter;
import android.view.View;
import android.view.ViewPropertyAnimator;
import androidx.recyclerview.widget.RecyclerView;
import b.w.b.k;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import java.util.Objects;

/* compiled from: DefaultItemAnimator.java */
/* loaded from: classes.dex */
public class j extends AnimatorListenerAdapter {

    /* renamed from: a  reason: collision with root package name */
    public final /* synthetic */ k.a f2741a;

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ ViewPropertyAnimator f2742b;

    /* renamed from: c  reason: collision with root package name */
    public final /* synthetic */ View f2743c;

    /* renamed from: d  reason: collision with root package name */
    public final /* synthetic */ k f2744d;

    public j(k kVar, k.a aVar, ViewPropertyAnimator viewPropertyAnimator, View view) {
        this.f2744d = kVar;
        this.f2741a = aVar;
        this.f2742b = viewPropertyAnimator;
        this.f2743c = view;
    }

    @Override // android.animation.AnimatorListenerAdapter, android.animation.Animator.AnimatorListener
    public void onAnimationEnd(Animator animator) {
        this.f2742b.setListener(null);
        this.f2743c.setAlpha(1.0f);
        this.f2743c.setTranslationX(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
        this.f2743c.setTranslationY(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
        this.f2744d.c(this.f2741a.f2747b);
        this.f2744d.s.remove(this.f2741a.f2747b);
        this.f2744d.k();
    }

    @Override // android.animation.AnimatorListenerAdapter, android.animation.Animator.AnimatorListener
    public void onAnimationStart(Animator animator) {
        k kVar = this.f2744d;
        RecyclerView.d0 d0Var = this.f2741a.f2747b;
        Objects.requireNonNull(kVar);
    }
}