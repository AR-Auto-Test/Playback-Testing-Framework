package b.j.j;

import android.animation.Animator;
import android.animation.AnimatorListenerAdapter;
import android.animation.ValueAnimator;
import android.view.View;
import java.lang.ref.WeakReference;

/* compiled from: ViewPropertyAnimatorCompat.java */
/* loaded from: classes.dex */
public final class s {

    /* renamed from: a  reason: collision with root package name */
    public WeakReference<View> f2231a;

    /* renamed from: b  reason: collision with root package name */
    public int f2232b = -1;

    /* compiled from: ViewPropertyAnimatorCompat.java */
    /* loaded from: classes.dex */
    public class a extends AnimatorListenerAdapter {

        /* renamed from: a  reason: collision with root package name */
        public final /* synthetic */ t f2233a;

        /* renamed from: b  reason: collision with root package name */
        public final /* synthetic */ View f2234b;

        public a(s sVar, t tVar, View view) {
            this.f2233a = tVar;
            this.f2234b = view;
        }

        @Override // android.animation.AnimatorListenerAdapter, android.animation.Animator.AnimatorListener
        public void onAnimationCancel(Animator animator) {
            this.f2233a.a(this.f2234b);
        }

        @Override // android.animation.AnimatorListenerAdapter, android.animation.Animator.AnimatorListener
        public void onAnimationEnd(Animator animator) {
            this.f2233a.b(this.f2234b);
        }

        @Override // android.animation.AnimatorListenerAdapter, android.animation.Animator.AnimatorListener
        public void onAnimationStart(Animator animator) {
            this.f2233a.c(this.f2234b);
        }
    }

    /* compiled from: ViewPropertyAnimatorCompat.java */
    /* loaded from: classes.dex */
    public class b implements ValueAnimator.AnimatorUpdateListener {

        /* renamed from: a  reason: collision with root package name */
        public final /* synthetic */ v f2235a;

        /* renamed from: b  reason: collision with root package name */
        public final /* synthetic */ View f2236b;

        public b(s sVar, v vVar, View view) {
            this.f2235a = vVar;
            this.f2236b = view;
        }

        @Override // android.animation.ValueAnimator.AnimatorUpdateListener
        public void onAnimationUpdate(ValueAnimator valueAnimator) {
            ((View) b.b.c.u.this.f619f.getParent()).invalidate();
        }
    }

    public s(View view) {
        this.f2231a = new WeakReference<>(view);
    }

    public s a(float f2) {
        View view = this.f2231a.get();
        if (view != null) {
            view.animate().alpha(f2);
        }
        return this;
    }

    public void b() {
        View view = this.f2231a.get();
        if (view != null) {
            view.animate().cancel();
        }
    }

    public s c(long j) {
        View view = this.f2231a.get();
        if (view != null) {
            view.animate().setDuration(j);
        }
        return this;
    }

    public s d(t tVar) {
        View view = this.f2231a.get();
        if (view != null) {
            e(view, tVar);
        }
        return this;
    }

    public final void e(View view, t tVar) {
        if (tVar != null) {
            view.animate().setListener(new a(this, tVar, view));
        } else {
            view.animate().setListener(null);
        }
    }

    public s f(v vVar) {
        View view = this.f2231a.get();
        if (view != null) {
            view.animate().setUpdateListener(vVar != null ? new b(this, vVar, view) : null);
        }
        return this;
    }

    public s g(float f2) {
        View view = this.f2231a.get();
        if (view != null) {
            view.animate().translationY(f2);
        }
        return this;
    }
}