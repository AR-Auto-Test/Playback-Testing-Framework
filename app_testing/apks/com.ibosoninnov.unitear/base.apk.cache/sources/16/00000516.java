package b.q.b;

import android.view.ViewGroup;
import android.view.animation.Animation;
import androidx.fragment.app.Fragment;
import b.q.b.f0;
import b.q.b.q;

/* compiled from: FragmentAnim.java */
/* loaded from: classes.dex */
public final class f implements Animation.AnimationListener {

    /* renamed from: a  reason: collision with root package name */
    public final /* synthetic */ ViewGroup f2436a;

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ Fragment f2437b;

    /* renamed from: c  reason: collision with root package name */
    public final /* synthetic */ f0.a f2438c;

    /* renamed from: d  reason: collision with root package name */
    public final /* synthetic */ b.j.f.b f2439d;

    /* compiled from: FragmentAnim.java */
    /* loaded from: classes.dex */
    public class a implements Runnable {
        public a() {
        }

        @Override // java.lang.Runnable
        public void run() {
            if (f.this.f2437b.getAnimatingAway() != null) {
                f.this.f2437b.setAnimatingAway(null);
                f fVar = f.this;
                ((q.b) fVar.f2438c).a(fVar.f2437b, fVar.f2439d);
            }
        }
    }

    public f(ViewGroup viewGroup, Fragment fragment, f0.a aVar, b.j.f.b bVar) {
        this.f2436a = viewGroup;
        this.f2437b = fragment;
        this.f2438c = aVar;
        this.f2439d = bVar;
    }

    @Override // android.view.animation.Animation.AnimationListener
    public void onAnimationEnd(Animation animation) {
        this.f2436a.post(new a());
    }

    @Override // android.view.animation.Animation.AnimationListener
    public void onAnimationRepeat(Animation animation) {
    }

    @Override // android.view.animation.Animation.AnimationListener
    public void onAnimationStart(Animation animation) {
    }
}