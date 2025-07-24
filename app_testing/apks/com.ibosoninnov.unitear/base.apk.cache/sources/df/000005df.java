package b.y.a;

import android.view.animation.Animation;
import android.view.animation.Transformation;
import androidx.swiperefreshlayout.widget.SwipeRefreshLayout;

/* compiled from: SwipeRefreshLayout.java */
/* loaded from: classes.dex */
public class e extends Animation {

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ SwipeRefreshLayout f2848b;

    public e(SwipeRefreshLayout swipeRefreshLayout) {
        this.f2848b = swipeRefreshLayout;
    }

    @Override // android.view.animation.Animation
    public void applyTransformation(float f2, Transformation transformation) {
        this.f2848b.setAnimationProgress(f2);
    }
}