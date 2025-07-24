package b.j.j;

import android.view.View;
import android.view.ViewTreeObserver;
import java.util.Objects;

/* compiled from: OneShotPreDrawListener.java */
/* loaded from: classes.dex */
public final class k implements ViewTreeObserver.OnPreDrawListener, View.OnAttachStateChangeListener {

    /* renamed from: b  reason: collision with root package name */
    public final View f2211b;

    /* renamed from: c  reason: collision with root package name */
    public ViewTreeObserver f2212c;

    /* renamed from: d  reason: collision with root package name */
    public final Runnable f2213d;

    public k(View view, Runnable runnable) {
        this.f2211b = view;
        this.f2212c = view.getViewTreeObserver();
        this.f2213d = runnable;
    }

    public static k a(View view, Runnable runnable) {
        Objects.requireNonNull(view, "view == null");
        k kVar = new k(view, runnable);
        view.getViewTreeObserver().addOnPreDrawListener(kVar);
        view.addOnAttachStateChangeListener(kVar);
        return kVar;
    }

    public void b() {
        if (this.f2212c.isAlive()) {
            this.f2212c.removeOnPreDrawListener(this);
        } else {
            this.f2211b.getViewTreeObserver().removeOnPreDrawListener(this);
        }
        this.f2211b.removeOnAttachStateChangeListener(this);
    }

    @Override // android.view.ViewTreeObserver.OnPreDrawListener
    public boolean onPreDraw() {
        b();
        this.f2213d.run();
        return true;
    }

    @Override // android.view.View.OnAttachStateChangeListener
    public void onViewAttachedToWindow(View view) {
        this.f2212c = view.getViewTreeObserver();
    }

    @Override // android.view.View.OnAttachStateChangeListener
    public void onViewDetachedFromWindow(View view) {
        b();
    }
}