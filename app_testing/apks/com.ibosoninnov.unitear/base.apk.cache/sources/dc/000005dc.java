package b.y.a;

import android.animation.Animator;
import b.y.a.d;
import com.google.android.material.internal.StaticLayoutBuilderCompat;

/* compiled from: CircularProgressDrawable.java */
/* loaded from: classes.dex */
public class c implements Animator.AnimatorListener {

    /* renamed from: a  reason: collision with root package name */
    public final /* synthetic */ d.a f2831a;

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ d f2832b;

    public c(d dVar, d.a aVar) {
        this.f2832b = dVar;
        this.f2831a = aVar;
    }

    @Override // android.animation.Animator.AnimatorListener
    public void onAnimationCancel(Animator animator) {
    }

    @Override // android.animation.Animator.AnimatorListener
    public void onAnimationEnd(Animator animator) {
    }

    @Override // android.animation.Animator.AnimatorListener
    public void onAnimationRepeat(Animator animator) {
        this.f2832b.a(1.0f, this.f2831a, true);
        d.a aVar = this.f2831a;
        aVar.k = aVar.f2844e;
        aVar.l = aVar.f2845f;
        aVar.m = aVar.f2846g;
        aVar.a((aVar.j + 1) % aVar.i.length);
        d dVar = this.f2832b;
        if (dVar.j) {
            dVar.j = false;
            animator.cancel();
            animator.setDuration(1332L);
            animator.start();
            this.f2831a.b(false);
            return;
        }
        dVar.i += 1.0f;
    }

    @Override // android.animation.Animator.AnimatorListener
    public void onAnimationStart(Animator animator) {
        this.f2832b.i = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
    }
}