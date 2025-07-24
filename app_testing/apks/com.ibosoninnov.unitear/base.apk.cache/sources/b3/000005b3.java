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
public class i extends AnimatorListenerAdapter {

    /* renamed from: a  reason: collision with root package name */
    public final /* synthetic */ k.a f2737a;

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ ViewPropertyAnimator f2738b;

    /* renamed from: c  reason: collision with root package name */
    public final /* synthetic */ View f2739c;

    /* renamed from: d  reason: collision with root package name */
    public final /* synthetic */ k f2740d;

    public i(k kVar, k.a aVar, ViewPropertyAnimator viewPropertyAnimator, View view) {
        this.f2740d = kVar;
        this.f2737a = aVar;
        this.f2738b = viewPropertyAnimator;
        this.f2739c = view;
    }

    @Override // android.animation.AnimatorListenerAdapter, android.animation.Animator.AnimatorListener
    public void onAnimationEnd(Animator animator) {
        this.f2738b.setListener(null);
        this.f2739c.setAlpha(1.0f);
        this.f2739c.setTranslationX(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
        this.f2739c.setTranslationY(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
        this.f2740d.c(this.f2737a.f2746a);
        this.f2740d.s.remove(this.f2737a.f2746a);
        this.f2740d.k();
    }

    @Override // android.animation.AnimatorListenerAdapter, android.animation.Animator.AnimatorListener
    public void onAnimationStart(Animator animator) {
        k kVar = this.f2740d;
        RecyclerView.d0 d0Var = this.f2737a.f2746a;
        Objects.requireNonNull(kVar);
    }
}