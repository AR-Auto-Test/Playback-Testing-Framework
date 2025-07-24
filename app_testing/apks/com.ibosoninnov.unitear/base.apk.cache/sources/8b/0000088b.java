package c.c.a.q.j;

import android.graphics.drawable.Animatable;
import android.graphics.drawable.Drawable;
import android.widget.ImageView;

/* compiled from: ImageViewTarget.java */
/* loaded from: classes.dex */
public abstract class e<Z> extends i<ImageView, Z> {

    /* renamed from: d  reason: collision with root package name */
    public Animatable f4157d;

    public e(ImageView imageView) {
        super(imageView);
    }

    @Override // c.c.a.q.j.h
    public void b(Z z, c.c.a.q.k.b<? super Z> bVar) {
        j(z);
    }

    @Override // c.c.a.q.j.h
    public void d(Drawable drawable) {
        j(null);
        ((ImageView) this.f4158b).setImageDrawable(drawable);
    }

    @Override // c.c.a.q.j.h
    public void e(Drawable drawable) {
        j(null);
        ((ImageView) this.f4158b).setImageDrawable(drawable);
    }

    @Override // c.c.a.q.j.h
    public void g(Drawable drawable) {
        this.f4159c.a();
        Animatable animatable = this.f4157d;
        if (animatable != null) {
            animatable.stop();
        }
        j(null);
        ((ImageView) this.f4158b).setImageDrawable(drawable);
    }

    public abstract void i(Z z);

    public final void j(Z z) {
        i(z);
        if (z instanceof Animatable) {
            Animatable animatable = (Animatable) z;
            this.f4157d = animatable;
            animatable.start();
            return;
        }
        this.f4157d = null;
    }

    @Override // c.c.a.n.m
    public void onStart() {
        Animatable animatable = this.f4157d;
        if (animatable != null) {
            animatable.start();
        }
    }

    @Override // c.c.a.n.m
    public void onStop() {
        Animatable animatable = this.f4157d;
        if (animatable != null) {
            animatable.stop();
        }
    }
}