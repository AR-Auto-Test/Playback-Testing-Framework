package b.q.b;

import android.view.View;
import android.view.ViewGroup;
import android.view.animation.Animation;
import android.view.animation.AnimationSet;
import android.view.animation.Transformation;

/* compiled from: FragmentAnim.java */
/* loaded from: classes.dex */
public class i extends AnimationSet implements Runnable {

    /* renamed from: b  reason: collision with root package name */
    public final ViewGroup f2474b;

    /* renamed from: c  reason: collision with root package name */
    public final View f2475c;

    /* renamed from: d  reason: collision with root package name */
    public boolean f2476d;

    /* renamed from: e  reason: collision with root package name */
    public boolean f2477e;

    /* renamed from: f  reason: collision with root package name */
    public boolean f2478f;

    public i(Animation animation, ViewGroup viewGroup, View view) {
        super(false);
        this.f2478f = true;
        this.f2474b = viewGroup;
        this.f2475c = view;
        addAnimation(animation);
        viewGroup.post(this);
    }

    @Override // android.view.animation.AnimationSet, android.view.animation.Animation
    public boolean getTransformation(long j, Transformation transformation) {
        this.f2478f = true;
        if (this.f2476d) {
            return !this.f2477e;
        }
        if (!super.getTransformation(j, transformation)) {
            this.f2476d = true;
            b.j.j.k.a(this.f2474b, this);
        }
        return true;
    }

    @Override // java.lang.Runnable
    public void run() {
        if (!this.f2476d && this.f2478f) {
            this.f2478f = false;
            this.f2474b.post(this);
            return;
        }
        this.f2474b.endViewTransition(this.f2475c);
        this.f2477e = true;
    }

    @Override // android.view.animation.Animation
    public boolean getTransformation(long j, Transformation transformation, float f2) {
        this.f2478f = true;
        if (this.f2476d) {
            return !this.f2477e;
        }
        if (!super.getTransformation(j, transformation, f2)) {
            this.f2476d = true;
            b.j.j.k.a(this.f2474b, this);
        }
        return true;
    }
}