package b.z;

import android.animation.Animator;
import android.animation.AnimatorListenerAdapter;
import android.animation.ObjectAnimator;
import android.view.View;
import android.view.ViewGroup;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import java.util.concurrent.atomic.AtomicInteger;

/* compiled from: Fade.java */
/* loaded from: classes.dex */
public class c extends z {

    /* compiled from: Fade.java */
    /* loaded from: classes.dex */
    public class a extends k {

        /* renamed from: a  reason: collision with root package name */
        public final /* synthetic */ View f2868a;

        public a(c cVar, View view) {
            this.f2868a = view;
        }

        @Override // b.z.j.f
        public void onTransitionEnd(j jVar) {
            View view = this.f2868a;
            y yVar = s.f2921a;
            yVar.e(view, 1.0f);
            yVar.a(this.f2868a);
            jVar.removeListener(this);
        }
    }

    /* compiled from: Fade.java */
    /* loaded from: classes.dex */
    public static class b extends AnimatorListenerAdapter {

        /* renamed from: a  reason: collision with root package name */
        public final View f2869a;

        /* renamed from: b  reason: collision with root package name */
        public boolean f2870b = false;

        public b(View view) {
            this.f2869a = view;
        }

        @Override // android.animation.AnimatorListenerAdapter, android.animation.Animator.AnimatorListener
        public void onAnimationEnd(Animator animator) {
            s.f2921a.e(this.f2869a, 1.0f);
            if (this.f2870b) {
                this.f2869a.setLayerType(0, null);
            }
        }

        @Override // android.animation.AnimatorListenerAdapter, android.animation.Animator.AnimatorListener
        public void onAnimationStart(Animator animator) {
            View view = this.f2869a;
            AtomicInteger atomicInteger = b.j.j.q.f2214a;
            if (view.hasOverlappingRendering() && this.f2869a.getLayerType() == 0) {
                this.f2870b = true;
                this.f2869a.setLayerType(2, null);
            }
        }
    }

    public c(int i) {
        setMode(i);
    }

    public final Animator a(View view, float f2, float f3) {
        if (f2 == f3) {
            return null;
        }
        s.f2921a.e(view, f2);
        ObjectAnimator ofFloat = ObjectAnimator.ofFloat(view, s.f2922b, f3);
        ofFloat.addListener(new b(view));
        addListener(new a(this, view));
        return ofFloat;
    }

    @Override // b.z.z, b.z.j
    public void captureStartValues(p pVar) {
        super.captureStartValues(pVar);
        pVar.f2913a.put("android:fade:transitionAlpha", Float.valueOf(s.a(pVar.f2914b)));
    }

    @Override // b.z.z
    public Animator onAppear(ViewGroup viewGroup, View view, p pVar, p pVar2) {
        Float f2;
        float f3 = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        float floatValue = (pVar == null || (f2 = (Float) pVar.f2913a.get("android:fade:transitionAlpha")) == null) ? 0.0f : f2.floatValue();
        if (floatValue != 1.0f) {
            f3 = floatValue;
        }
        return a(view, f3, 1.0f);
    }

    @Override // b.z.z
    public Animator onDisappear(ViewGroup viewGroup, View view, p pVar, p pVar2) {
        s.f2921a.c(view);
        Float f2 = (Float) pVar.f2913a.get("android:fade:transitionAlpha");
        return a(view, f2 != null ? f2.floatValue() : 1.0f, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
    }
}