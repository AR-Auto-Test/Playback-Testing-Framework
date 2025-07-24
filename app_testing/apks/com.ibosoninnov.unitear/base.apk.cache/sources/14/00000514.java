package b.q.b;

import android.view.View;
import androidx.fragment.app.Fragment;
import b.j.f.b;

/* compiled from: FragmentAnim.java */
/* loaded from: classes.dex */
public final class e implements b.a {

    /* renamed from: a  reason: collision with root package name */
    public final /* synthetic */ Fragment f2428a;

    public e(Fragment fragment) {
        this.f2428a = fragment;
    }

    @Override // b.j.f.b.a
    public void a() {
        if (this.f2428a.getAnimatingAway() != null) {
            View animatingAway = this.f2428a.getAnimatingAway();
            this.f2428a.setAnimatingAway(null);
            animatingAway.clearAnimation();
        }
        this.f2428a.setAnimator(null);
    }
}