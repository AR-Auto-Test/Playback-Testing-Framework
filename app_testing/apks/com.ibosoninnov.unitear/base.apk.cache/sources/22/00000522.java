package b.q.b;

import android.animation.Animator;
import android.view.animation.Animation;

/* compiled from: FragmentAnim.java */
/* loaded from: classes.dex */
public class h {

    /* renamed from: a  reason: collision with root package name */
    public final Animation f2467a;

    /* renamed from: b  reason: collision with root package name */
    public final Animator f2468b;

    public h(Animation animation) {
        this.f2467a = animation;
        this.f2468b = null;
        if (animation == null) {
            throw new IllegalStateException("Animation cannot be null");
        }
    }

    public h(Animator animator) {
        this.f2467a = null;
        this.f2468b = animator;
    }
}