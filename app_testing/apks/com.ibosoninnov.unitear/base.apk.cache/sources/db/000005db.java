package b.y.a;

import android.animation.ValueAnimator;
import b.y.a.d;

/* compiled from: CircularProgressDrawable.java */
/* loaded from: classes.dex */
public class b implements ValueAnimator.AnimatorUpdateListener {

    /* renamed from: a  reason: collision with root package name */
    public final /* synthetic */ d.a f2829a;

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ d f2830b;

    public b(d dVar, d.a aVar) {
        this.f2830b = dVar;
        this.f2829a = aVar;
    }

    @Override // android.animation.ValueAnimator.AnimatorUpdateListener
    public void onAnimationUpdate(ValueAnimator valueAnimator) {
        float floatValue = ((Float) valueAnimator.getAnimatedValue()).floatValue();
        this.f2830b.d(floatValue, this.f2829a);
        this.f2830b.a(floatValue, this.f2829a, false);
        this.f2830b.invalidateSelf();
    }
}